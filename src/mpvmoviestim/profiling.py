"""Render pipeline profiling for MpvMoviestim.

Design
------
- Two preallocated flat ``array('d')`` recorders (one per thread: render
  worker, main/draw thread), row-major layout of ``capacity`` iterations x
  per-thread slot count. The render hot path performs a single
  ``perf_counter()`` call and a single array store per measurement point;
  no method calls, no allocations, no locks (each thread exclusively owns
  its recorder).
- All events are stored as absolute ``perf_counter()`` timestamps;
  durations are computed at retrieval time. Since ``perf_counter()`` is
  the same clock on every thread, events from both threads merge correctly
  by simple sorting.
- GPU-side durations of the mpv render and the blit are measured with
  ``GL_TIME_ELAPSED`` timer queries (pooled per GL context, results
  harvested non-blockingly at the next iteration start and written back
  into the row they belong to).
- Cross-thread fence waits are characterised with a zero-timeout
  ``glClientWaitSync`` poll (already-signalled vs not-ready) plus CPU-side
  timestamps around ``glWaitSync``.
- Each thread's iteration ends with a blocking ``glClientWaitSync`` on a
  private "done" fence issued right after the work fence. The wait
  duration (``gpu_wait``) shows whether the GPU was still busy when the
  CPU finished issuing commands, and the return time (``iter_done``) is a
  good estimate of when the GPU finished the iteration's work, to be
  compared against the screen refresh budget.
- The mpv update callback appends ``(timestamp, is_trigger)`` to a deque
  (thread-safe); the worker drains it fully at iteration start into a
  preallocated wake log. ``is_trigger`` marks the callback that
  transitioned ``render_trigger`` from unset to set, i.e. the one that
  actually woke the worker (subject to the inherent check/set race).

Retrieval: ``Profiler.get_events()`` returns ``(t_rel_s, thread, event,
duration_s)`` tuples sorted by ``t_rel_s`` (relative to the first recorded
timestamp). Instants have duration 0.0. Events on the pseudo-threads
``worker-gpu`` / ``main-gpu`` are GPU-side durations; their timeline
placement is the CPU-side issue time of the corresponding block
(approximate - the GPU may start later).
"""

from __future__ import annotations

import csv
from array import array
from collections import deque
from typing import TYPE_CHECKING

from pyglet import gl

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "Worker_Timestamp_Indices",
    "Render_Timestamp_Indices",
    "GpuTimerPool",
    "Profiler",
    "ThreadRecorder",
]


class Worker_Timestamp_Indices:
    """Worker-thread recorder slot indices (one row per worker iteration)."""

    UPDATE_T0: int
    UPDATE_T1: int
    LOCK_T0: int
    LOCK_T1: int
    WAITSYNC_BLIT_T0: int
    WAITSYNC_BLIT_T1: int
    WAITSYNC_BLIT_STATE: int
    RENDER_T0: int
    RENDER_T1: int
    FENCE_POST_T: int
    FLIP_REQUEST_T: int
    SET_DONE_T: int
    WAIT_DONE_DUR: int
    ITER_DONE_T: int
    GPU_RENDER: int

    __slots__ = (  # noqa: RUF023
        "UPDATE_T0",  # ctx.update() span
        "UPDATE_T1",
        "LOCK_T0",  # buffer_fbo_lock acquire span
        "LOCK_T1",
        "WAITSYNC_BLIT_T0",  # CPU-side span around glWaitSync(blit fence)
        "WAITSYNC_BLIT_T1",
        "WAITSYNC_BLIT_STATE",  # zero-timeout poll: OpenGL values
        # TODO: translate to 1=already done, 2=not ready, 3=failed
        "RENDER_T0",  # ctx.render() span (CPU-side command issue time)
        "RENDER_T1",
        "FENCE_POST_T",  # render fence posted (instant)
        "FLIP_REQUEST_T",  # request to flip buffer indices (instant)
        "SET_DONE_T",  # worker_render_done.set() (instant)
        "WAIT_DONE_DUR",  # blocking clientWaitSync duration on private done-fence (-1: timeout)
        "ITER_DONE_T",  # ~when the GPU finished this iteration's render (instant)
        "GPU_RENDER",  # GL_TIME_ELAPSED duration of ctx.render() (seconds, filled late)
    )

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


class Render_Timestamp_Indices:
    """Render thread recorder slot indices (one row per draw() call)."""

    DRAW_ENTRY_T: int
    LOCK_T0: int
    LOCK_T1: int
    CPU_WAIT_T0: int
    CPU_WAIT_T1: int
    WAITSYNC_RENDER_T0: int
    WAITSYNC_RENDER_T1: int
    WAITSYNC_RENDER_STATE: int
    BLIT_T0: int
    BLIT_T1: int
    FENCE_POST_T: int
    GPU_WAIT: int
    ITER_DONE_T: int
    DRAW_EXIT_T: int
    REPORT_SWAP_T: int
    GPU_BLIT: int

    __slots__ = (  # noqa: RUF023
        "DRAW_ENTRY_T",  # draw() entered (instant)
        "LOCK_T0",  # buffer_fbo_lock acquire span
        "LOCK_T1",
        "CPU_WAIT_T0",  # worker_render_done.wait() span (only when worker mid-render)
        "CPU_WAIT_T1",
        "WAITSYNC_RENDER_T0",  # CPU-side span around glWaitSync(render fence)
        "WAITSYNC_RENDER_T1",
        "WAITSYNC_RENDER_STATE",  # zero-timeout poll: 1=already done, 2=not ready, 3=failed
        "BLIT_T0",  # blit span (CPU-side command issue time)
        "BLIT_T1",
        "FENCE_POST_T",  # blit fence posted (instant)
        "GPU_WAIT",  # blocking clientWaitSync duration on private done-fence (-1: timeout)
        "ITER_DONE_T",  # ~when the GPU finished this iteration's blit (instant)
        "DRAW_EXIT_T",  # draw() finished (instant)
        "REPORT_SWAP_T",  # report_swap() called (instant, stamped into current row)
        "GPU_BLIT",  # GL_TIME_ELAPSED duration of the blit (seconds, filled late)
    )

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


# Slots holding genuine timestamps (used to find the timeline anchor t0 and
# to skip zero/unset slots at retrieval). Duration/state/index slots are
# deliberately excluded.
_WORKER_TS_SLOTS = (
    "UPDATE_T0",
    "UPDATE_T1",
    "LOCK_T0",
    "LOCK_T1",
    "WAITSYNC_BLIT_T0",
    "WAITSYNC_BLIT_T1",
    "RENDER_T0",
    "RENDER_T1",
    "FENCE_POST_T",
    "FLIP_REQUEST_T",
    "SET_DONE_T",
    "ITER_DONE_T",
)
_MAIN_TS_SLOTS = (
    "DRAW_ENTRY_T",
    "LOCK_T0",
    "LOCK_T1",
    "CPU_WAIT_T0",
    "CPU_WAIT_T1",
    "WAITSYNC_RENDER_T0",
    "WAITSYNC_RENDER_T1",
    "BLIT_T0",
    "BLIT_T1",
    "FENCE_POST_T",
    "ITER_DONE_T",
    "DRAW_EXIT_T",
    "REPORT_SWAP_T",
)

_WORKER_SPANS = (
    ("update", Worker_Timestamp_Indices.UPDATE_T0, Worker_Timestamp_Indices.UPDATE_T1),
    ("lock", Worker_Timestamp_Indices.LOCK_T0, Worker_Timestamp_Indices.LOCK_T1),
    (
        "waitsync_blit",
        Worker_Timestamp_Indices.WAITSYNC_BLIT_T0,
        Worker_Timestamp_Indices.WAITSYNC_BLIT_T1,
    ),
    (
        "render_cpu",
        Worker_Timestamp_Indices.RENDER_T0,
        Worker_Timestamp_Indices.RENDER_T1,
    ),
)
_WORKER_INSTANTS = (
    ("fence_post", Worker_Timestamp_Indices.FENCE_POST_T),
    ("buffer_flip", Worker_Timestamp_Indices.FLIP_T),
    ("render_done_set", Worker_Timestamp_Indices.SET_DONE_T),
)
_MAIN_SPANS = (
    ("lock", Render_Timestamp_Indices.LOCK_T0, Render_Timestamp_Indices.LOCK_T1),
    (
        "cpu_wait_worker",
        Render_Timestamp_Indices.CPU_WAIT_T0,
        Render_Timestamp_Indices.CPU_WAIT_T1,
    ),
    (
        "waitsync_render",
        Render_Timestamp_Indices.WAITSYNC_RENDER_T0,
        Render_Timestamp_Indices.WAITSYNC_RENDER_T1,
    ),
    ("blit_cpu", Render_Timestamp_Indices.BLIT_T0, Render_Timestamp_Indices.BLIT_T1),
)
_MAIN_INSTANTS = (
    ("draw_entry", Render_Timestamp_Indices.DRAW_ENTRY_T),
    ("fence_post", Render_Timestamp_Indices.FENCE_POST_T),
    ("draw_exit", Render_Timestamp_Indices.DRAW_EXIT_T),
    ("report_swap", Render_Timestamp_Indices.REPORT_SWAP_T),
)

_POLL_STATE_NAMES = {1.0: "ready", 2.0: "not_ready", 3.0: "failed"}


class ThreadRecorder:
    """Fixed-capacity per-thread timestamp recorder.

    Rows are iterations; each row has ``stride`` slots. Writes in the hot
    path are done inline by the caller as ``buf[base + SLOT] = perf_counter()``
    with ``buf``/``base`` hoisted to locals; this class only manages row
    advancement. When the buffer is full, ``next_iter()`` returns -1 and the
    recorder becomes permanently inactive (stop-when-full policy).
    """

    __slots__ = ("buf", "stride", "base", "rows", "active")

    def __init__(self, capacity: int, n_slots: int) -> None:
        self.stride = n_slots
        self.buf = array("d", [0.0]) * (capacity * n_slots)
        self.base = -n_slots  # first next_iter() yields row 0
        self.rows = 0
        self.active = True

    def next_iter(self) -> int:
        """Advance to the next row. Returns the row base index, or -1 when full."""
        if not self.active:
            return -1
        base = self.base + self.stride
        if base >= len(self.buf):
            self.active = False
            return -1
        self.base = base
        self.rows += 1
        return base


class GpuTimerPool:
    """Pool of GL_TIME_ELAPSED query objects for one GL context/thread.

    ``begin()``/``end()`` bracket a GL block (e.g. ctx.render()). Results are
    harvested non-blockingly by ``collect()`` - call once per iteration on the
    owning thread - and written into the recorder row they belong to; results
    typically arrive one or more iterations after the block was issued.
    ``drain_blocking()`` is for teardown only, while the owning GL context is
    still current. If the context does not support timer queries (GL < 3.3)
    the pool disables itself and all calls become no-ops.
    """

    POOL_SIZE = 8

    def __init__(self, recorder: ThreadRecorder, slot: int) -> None:
        self._rec = recorder
        self._slot = slot
        self._free: list[int] = []
        self._pending: deque[tuple[int, int]] = deque()  # (row_base, query id)
        self._total = 0
        self._cur: int | None = None
        self._ok: bool | None = None  # None = not probed yet

    def _probe(self) -> bool:
        """Check GL version once; GL_TIME_ELAPSED is core in 3.3+."""
        try:
            major = (gl.GLint * 1)()
            minor = (gl.GLint * 1)()
            gl.glGetIntegerv(gl.GL_MAJOR_VERSION, major)
            gl.glGetIntegerv(gl.GL_MINOR_VERSION, minor)
            self._ok = (major[0], minor[0]) >= (3, 3)
        except Exception:  # pylint: disable=broad-exception-caught  # any GL/pyglet failure -> disable
            self._ok = False
        return self._ok

    def begin(self) -> None:
        """Start a GPU time measurement. No-op if unsupported."""
        if self._ok is None and not self._probe():
            return
        if not self._ok:
            return
        if self._free:
            q = self._free.pop()
        elif self._total < self.POOL_SIZE:
            ids = (gl.GLuint * 1)()
            gl.glGenQueries(1, ids)
            q = ids[0]
            self._total += 1
        else:
            # Pool exhausted: block on the oldest pending query (should be rare).
            base, q = self._pending.popleft()
            self._store_blocking(base, q)
        gl.glBeginQuery(gl.GL_TIME_ELAPSED, q)
        self._cur = q

    def end(self, base: int) -> None:
        """Stop the measurement; the result is queued for ``base``'s row."""
        q = self._cur
        if q is None:
            return
        self._cur = None
        gl.glEndQuery(gl.GL_TIME_ELAPSED)
        self._pending.append((base, q))

    def collect(self) -> None:
        """Harvest available results non-blockingly (owning thread, per iteration)."""
        pend = self._pending
        if not pend:
            return
        avail = (gl.GLuint * 1)()
        val = (gl.GLuint64 * 1)()
        buf = self._rec.buf
        slot = self._slot
        while pend:
            base, q = pend[0]
            gl.glGetQueryObjectuiv(q, gl.GL_QUERY_RESULT_AVAILABLE, avail)
            if not avail[0]:
                break  # queries complete in order
            gl.glGetQueryObjectui64v(q, gl.GL_QUERY_RESULT, val)
            buf[base + slot] = val[0] * 1e-9
            pend.popleft()
            self._free.append(q)

    def _store_blocking(self, base: int, q: int) -> None:
        val = (gl.GLuint64 * 1)()
        gl.glGetQueryObjectui64v(q, gl.GL_QUERY_RESULT, val)  # blocks until ready
        self._rec.buf[base + self._slot] = val[0] * 1e-9
        self._free.append(q)

    def drain_blocking(self) -> None:
        """Blocking drain + delete all queries. Teardown only; the owning GL
        context must be current on the calling thread."""
        if self._ok is not True:
            return
        if self._cur is not None:
            gl.glEndQuery(gl.GL_TIME_ELAPSED)
            q, self._cur = self._cur, None
            self._pending.append((self._rec.base, q))
        while self._pending:
            base, q = self._pending.popleft()
            self._store_blocking(base, q)
        if self._free:
            ids = (gl.GLuint * len(self._free))(*self._free)
            gl.glDeleteQueries(len(self._free), ids)
        self._free.clear()
        self._total = 0


class Profiler:
    """Owns both thread recorders, GPU timer pools and the wake log; merges
    the recordings into a single sorted timeline at retrieval time."""

    def __init__(self, capacity: int = 10_000, wake_factor: int = 8) -> None:
        self.worker = ThreadRecorder(capacity, Worker_Timestamp_Indices.COUNT)
        self.main = ThreadRecorder(capacity, Render_Timestamp_Indices.COUNT)
        self.worker_gpu = GpuTimerPool(self.worker, Worker_Timestamp_Indices.GPU_RENDER)
        self.main_gpu = GpuTimerPool(self.main, Render_Timestamp_Indices.GPU_BLIT)
        # Appended by mpv's update callback thread, drained by the worker.
        self.wake_times: deque[tuple[float, bool]] = deque()
        # Preallocated wake log: parallel arrays of stamps and trigger flags.
        self.wake_log_t = array("d", [0.0]) * (capacity * wake_factor)
        self.wake_log_trigger = array("b", [0]) * (capacity * wake_factor)
        self.wake_log_len = 0

    def drain_wakes(self, buf: array, base: int) -> None:
        """Move all queued callback wake stamps into the wake log (worker thread)."""
        dq = self.wake_times
        log_t = self.wake_log_t
        log_f = self.wake_log_trigger
        i = self.wake_log_len
        cap = len(log_t)
        start = i
        count = 0
        while dq:
            t, trig = dq.popleft()
            count += 1
            if i < cap:
                log_t[i] = t
                log_f[i] = 1 if trig else 0
                i += 1
        self.wake_log_len = i
        buf[base + Worker_Timestamp_Indices.WAKE_IDX] = float(start) if count else -1.0
        buf[base + Worker_Timestamp_Indices.WAKE_COUNT] = float(count)

    @staticmethod
    def _first_ts(rec: ThreadRecorder, slots: tuple[int, ...]) -> float:
        """First nonzero timestamp in the recorder (rows are chronological)."""
        buf = rec.buf
        lim = rec.rows * rec.stride
        for base in range(0, lim, rec.stride):
            for s in slots:
                v = buf[base + s]
                if v > 0.0:
                    return v
        return 0.0

    def _find_t0(self) -> float:
        cands = []
        if self.wake_log_len:
            cands.append(self.wake_log_t[0])
        w = self._first_ts(self.worker, _WORKER_TS_SLOTS)
        if w > 0.0:
            cands.append(w)
        m = self._first_ts(self.main, _MAIN_TS_SLOTS)
        if m > 0.0:
            cands.append(m)
        return min(cands) if cands else 0.0

    @staticmethod
    def _emit_rows(
        events: list[tuple[float, str, str, float]],
        t0: float,
        rec: ThreadRecorder,
        thread: str,
        gpu_thread: str,
        gpu_name: str,
        spans: tuple[tuple[str, int, int], ...],
        instants: tuple[tuple[str, int], ...],
        state_slot: int,
        state_t0_slot: int,
        poll_prefix: str,
        gpu_slot: int,
        gpu_t0_slot: int,
        gpu_wait_slot: int,
        iter_done_slot: int,
    ) -> None:
        buf = rec.buf
        stride = rec.stride
        for r in range(rec.rows):
            b = r * stride
            for name, s0, s1 in spans:
                a = buf[b + s0]
                c = buf[b + s1]
                if a > 0.0 and c > 0.0:
                    events.append((a - t0, thread, name, c - a))
            for name, s in instants:
                a = buf[b + s]
                if a > 0.0:
                    events.append((a - t0, thread, name, 0.0))
            st = buf[b + state_slot]
            a = buf[b + state_t0_slot]
            if st > 0.0 and a > 0.0:
                lbl = _POLL_STATE_NAMES.get(st, "unknown")
                events.append((a - t0, thread, f"{poll_prefix}:{lbl}", 0.0))
            g = buf[b + gpu_slot]
            a = buf[b + gpu_t0_slot]
            if g > 0.0 and a > 0.0:
                events.append((a - t0, gpu_thread, gpu_name, g))
            done = buf[b + iter_done_slot]
            if done > 0.0:
                gw = buf[b + gpu_wait_slot]
                if gw > 0.0:
                    events.append((done - gw - t0, thread, "gpu_wait", gw))
                elif gw == -1.0:
                    events.append((done - t0, thread, "gpu_wait_timeout", 0.0))
                events.append((done - t0, thread, "iter_done", 0.0))

    def get_events(self) -> list[tuple[float, str, str, float]]:
        """Merge both threads' recordings into a timestamp-sorted event list.

        Returns
        -------
        list of (t_rel_s, thread, event, duration_s)
            ``t_rel_s`` is relative to the first recorded timestamp. Instants
            have duration 0.0. Events on threads ``worker-gpu`` / ``main-gpu``
            are GPU-side durations placed at the CPU-side issue time.
        """
        t0 = self._find_t0()
        if t0 == 0.0:
            return []
        events: list[tuple[float, str, str, float]] = []
        self._emit_rows(
            events,
            t0,
            self.worker,
            "worker",
            "worker-gpu",
            "render",
            _WORKER_SPANS,
            _WORKER_INSTANTS,
            Worker_Timestamp_Indices.WAITSYNC_BLIT_STATE,
            Worker_Timestamp_Indices.WAITSYNC_BLIT_T0,
            "waitsync_blit_poll",
            Worker_Timestamp_Indices.GPU_RENDER,
            Worker_Timestamp_Indices.RENDER_T0,
            Worker_Timestamp_Indices.WAIT_DONE_DUR,
            Worker_Timestamp_Indices.ITER_DONE_T,
        )
        # per-iteration wake events from the callback thread
        buf = self.worker.buf
        stride = self.worker.stride
        wake_t = self.wake_log_t
        wake_f = self.wake_log_trigger
        wake_n = self.wake_log_len
        for r in range(self.worker.rows):
            b = r * stride
            count = int(buf[b + Worker_Timestamp_Indices.WAKE_COUNT])
            if count <= 0:
                continue
            idx = int(buf[b + Worker_Timestamp_Indices.WAKE_IDX])
            for k in range(count):
                j = idx + k
                if j >= wake_n:
                    break
                t = wake_t[j]
                if t > 0.0:
                    events.append(
                        (t - t0, "mpv-cb", "wake_trigger" if wake_f[j] else "wake", 0.0)
                    )
        self._emit_rows(
            events,
            t0,
            self.main,
            "main",
            "main-gpu",
            "blit",
            _MAIN_SPANS,
            _MAIN_INSTANTS,
            Render_Timestamp_Indices.WAITSYNC_RENDER_STATE,
            Render_Timestamp_Indices.WAITSYNC_RENDER_T0,
            "waitsync_render_poll",
            Render_Timestamp_Indices.GPU_BLIT,
            Render_Timestamp_Indices.BLIT_T0,
            Render_Timestamp_Indices.GPU_WAIT,
            Render_Timestamp_Indices.ITER_DONE_T,
        )
        events.sort(key=lambda e: e[0])
        return events

    def export_csv(self, path: str | Path) -> None:
        """Write the merged timeline to CSV (t_ms, thread, event, duration_ms)."""
        events = self.get_events()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "thread", "event", "duration_ms"])
            for t, thread, name, dur in events:
                w.writerow([f"{t * 1e3:.4f}", thread, name, f"{dur * 1e3:.4f}"])
