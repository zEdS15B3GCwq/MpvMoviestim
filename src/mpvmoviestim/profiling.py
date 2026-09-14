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
from typing import TYPE_CHECKING

from pyglet import gl

if TYPE_CHECKING:
    from pathlib import Path


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
    WAIT_DONE_T0: int
    WAIT_DONE_T1: int
    WAIT_DONE_STATE: int
    ITER_DONE_T: int
    GPU_RENDER_DUR: int

    __slots__ = (  # noqa: RUF023
        "UPDATE_T0",  # ctx.update() span
        "UPDATE_T1",
        "LOCK_T0",  # buffer_fbo_lock acquire span
        "LOCK_T1",
        "WAITSYNC_BLIT_T0",  # CPU-side span around glWaitSync(blit fence)
        "WAITSYNC_BLIT_T1",
        "WAITSYNC_BLIT_STATE",  # zero-timeout poll: OpenGL values
        # TODO: defer translating to 1=already done, 2=not ready, 3=failed
        "RENDER_T0",  # ctx.render() span (CPU-side command issue time)
        "RENDER_T1",
        "FENCE_POST_T",  # render fence posted (instant)
        "FLIP_REQUEST_T",  # request to flip buffer indices (instant)
        "SET_DONE_T",  # worker_render_done.set() (instant)
        "WAIT_DONE_T0",  # blocking clientWaitSync start on private done-fence
        "WAIT_DONE_T1",
        "WAIT_DONE_STATE",  # 1=already done, 2=not ready, 3=failed
        "ITER_DONE_T",  # ~when the GPU finished this iteration's render (instant)
        "GPU_RENDER_DUR",  # GL_TIME_ELAPSED duration of ctx.render() (seconds, filled late)
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
    WAIT_DONE_T0: int
    WAIT_DONE_T1: int
    WAIT_DONE_STATE: int
    DRAW_EXIT_T: int
    GPU_BLIT_DUR: int

    __slots__ = (  # noqa: RUF023
        "DRAW_ENTRY_T",  # draw() entered (instant)
        "LOCK_T0",  # buffer_fbo_lock acquire span
        "LOCK_T1",
        "CPU_WAIT_T0",  # worker_render_done.wait() span (only when worker mid-render)
        "CPU_WAIT_T1",
        "WAITSYNC_RENDER_T0",  # CPU-side span around glWaitSync(render fence)
        "WAITSYNC_RENDER_T1",
        "WAITSYNC_RENDER_STATE",  # zero-timeout poll: 1=already done, 2=not ready, 3=failed
        # TODO: defer translating to 1=already done, 2=not ready, 3=failed
        "BLIT_T0",  # blit span (CPU-side command issue time)
        "BLIT_T1",
        "FENCE_POST_T",  # blit fence posted (instant)
        "WAIT_DONE_T0",  # blocking clientWaitSync start on private done-fence
        "WAIT_DONE_T1",
        "WAIT_DONE_STATE",  # 1=already done, 2=not ready, 3=failed
        "DRAW_EXIT_T",  # ~when the GPU finished this iteration's blit (instant)
        "GPU_BLIT_DUR",  # GL_TIME_ELAPSED duration of the blit (seconds, filled late)
    )

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


_POLL_STATE_NAMES = {1.0: "ready", 2.0: "not_ready", 3.0: "failed"}


class Update_Callback_Indices:
    """Update callback timestamp indices (one row per callback)."""

    UPDATE_T: int
    IS_TRIGGER: int

    __slots__ = (  # noqa: RUF023
        "UPDATE_T",  # update() called (instant)
        "IS_TRIGGER",  # did the callback wake the worker thread?
    )

    def __init__(self):
        for index, name in enumerate(self.__slots__):
            setattr(self, name, index)


class Buffer:
    """Fixed-capacity 2D array allocator.

    Pre-allocates a (nrows x nslots) size 2D array `buf` for storing profiling
    data. The class keeps track of the current row (`base`). Writes should be
    performed as `buf[base + SLOT] = value`. `next_row()` advances to the next
    row, and returns -1 when the buffer is full.
    """

    __slots__ = ("buf", "stride", "base", "rows_written", "active")  # noqa: RUF023

    def __init__(self, nrows: int, nslots: int) -> None:
        self.stride = nslots
        self.buf = array("d", [0.0]) * (nrows * nslots)
        self.base = -nslots  # first next_iter() yields row 0
        self.rows_written = 0
        self.active = True

    def next_row(self) -> int:
        """Advance to the next row. Returns the row base index, or -1 when full."""
        if not self.active:
            return -1
        base = self.base + self.stride
        if base >= len(self.buf):
            self.active = False
            return -1
        self.base = base
        self.rows_written += 1
        return base


class GpuTimer:
    """Helper class to measure elapsed time on GPU in an OpenGL block.

    Uses GL_TIME_ELAPSED timer queries for a block of GL commands. Use
    `begin()`/`end()` to bracket a GL block (e.g. ctx.render()), then
    `collect()` to retrieve the result in a blocking way.

    The intended safe way is to use it together with a sync fence added
    after the block that is CPU-waited on. Waiting on the fence ensures
    that the query result is immediately available so `collect()` doesn't
    incur additional CPU wait.

    One query ID is created on class instantiation and reused for in
    subsequent queries. Always call `end()` and `collect()` after each
    `begin()`, otherwise reusing the same query ID will result in
    undefined behavior.
    """

    def __init__(self) -> None:
        self._id = (gl.GLuint * 1)()
        gl.glGenQueries(1, self._id)

    def begin(self) -> None:
        """Start a GPU time measurement."""
        gl.glBeginQuery(gl.GL_TIME_ELAPSED, self._id)

    def end(self) -> None:
        """Indicate the end of the measured GPU block."""
        gl.glEndQuery(gl.GL_TIME_ELAPSED)

    def collect(self) -> int:
        """Retrieve the measured elapsed time in nanoseconds."""
        val = (gl.GLuint64 * 1)()
        gl.glGetQueryObjectui64v(self._id, gl.GL_QUERY_RESULT, val)
        return val[0]

    def __del__(self) -> None:
        if hasattr(self, "_id"):
            gl.glDeleteQueries(1, self._id)


class Profiler:
    """Main profiling class that manages the worker and main thread recorders,
    GPU timers, and the wake log.

    Reserves profiling buffers for `capacity` iterations, instantiates GPU timers,
    and pre-allocates an update_callback log of `capacity * wake_factor` entries.
    """

    worker_indices: Worker_Timestamp_Indices
    main_indices: Render_Timestamp_Indices
    update_indices: Update_Callback_Indices
    worker: Buffer
    main: Buffer
    update_cb: Buffer
    worker_gpu: GpuTimer
    main_gpu: GpuTimer

    def __init__(self, capacity: int = 10_000, wake_factor: int = 8) -> None:
        # allocate CPU profiling buffers for worker and main threads
        self.worker_indices = Worker_Timestamp_Indices()
        self.main_indices = Render_Timestamp_Indices()
        self.worker = Buffer(capacity, len(self.worker_indices.__slots__))
        self.main = Buffer(capacity, len(self.main_indices.__slots__))

        # allocate GPU profiling timers for worker and main threads
        self.worker_gpu = GpuTimer()
        self.main_gpu = GpuTimer()

        # allocate one more buffer for update callback times
        self.update_indices = Update_Callback_Indices()
        self.update_cb = Buffer(
            capacity * wake_factor, len(self.update_indices.__slots__)
        )

    @staticmethod
    def _first_ts(rec: Buffer, slots: tuple[int, ...]) -> float:
        """First nonzero timestamp in the recorder (rows are chronological)."""
        buf = rec.buf
        lim = rec.rows_written * rec.stride
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
        rec: Buffer,
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
        for r in range(rec.rows_written):
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
            Worker_Timestamp_Indices.GPU_RENDER_DUR,
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
        for r in range(self.worker.rows_written):
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
            Render_Timestamp_Indices.GPU_BLIT_DUR,
            Render_Timestamp_Indices.BLIT_T0,
            Render_Timestamp_Indices.WAIT_DONE_DUR,
            Render_Timestamp_Indices.DRAW_EXIT_T,
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
