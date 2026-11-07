"""Triple-buffered frame exchange between one producer and one consumer thread.

The producer (P) renders video frames into intermediate buffers as soon as they
arrive, the consumer (C) draws the freshest finished frame to the screen. Both
threads use OpenGL contexts of the same share group.

`BufferManager` owns three `FrameBuffer`s and does all synchronisation:

* CPU side: a single lock guards which buffer each thread holds, and a
  condition on that lock lets C wait for the frame P is rendering.
* GPU side: each buffer has one "last use" fence. Whoever takes a buffer makes
  its GPU queue wait for that fence if the other thread placed it
  (`glWaitSync`, the CPU doesn't block), and leaves a new fence when releasing
  the buffer (`glFenceSync` + `glFlush`).

Typical use::

    # P, with its context current, whenever a new frame arrives
    with manager.acquire_write_buffer() as buffer:
        buffer.set_frame_size(width, height, pixel_format)
        render_frame_into(buffer.fbo_id, width, height)

    # C, with its context current, once per screen refresh
    with manager.acquire_read_buffer(timeout=wait_budget) as frame:
        if frame is not None:
            draw_texture(frame.buffer.texture_id, frame.buffer.frame_width, ...)

See `buffer_design.md` for the design goals and the reasons behind them.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, NamedTuple

from psychopy import logging
from pyglet import gl

from .utils import (
    gl_fbo_create,
    gl_fbo_destroy,
    gl_texture_create,
    gl_texture_destroy,
)

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any

BUFFER_COUNT = 3


def _insert_fence_and_flush() -> Any:
    """Insert a fence after the commands queued so far, and submit them.

    The flush is required: another context waiting on a fence that was never
    submitted to the GPU may wait forever.
    """
    fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.glFlush()
    return fence


def _wait_for_fence_and_delete(fence: Any, gpu_wait_needed: bool) -> None:
    """Make the current context's GPU queue wait for `fence` if needed, then delete it.

    `glWaitSync` returns immediately, only the GPU waits. A thread's own fence
    needs no wait: commands within one context are executed in order, and the
    wait could still cost GPU time by draining the pipeline. Deleting the fence
    right after the wait is safe: GL defers the deletion while a wait on it is
    pending.
    """
    if not fence:  # NULL GLsync pointers are falsy
        return
    if gpu_wait_needed:
        gl.glWaitSync(fence, 0, gl.GL_TIMEOUT_IGNORED)
    gl.glDeleteSync(fence)


class FrameBuffer:
    """An intermediate buffer: a texture, the FBO it's attached to, and metadata.

    Attributes
    ----------
    texture_id : int
        GL texture name, 0 if not allocated.
    fbo_id : int
        GL framebuffer name with `texture_id` attached, 0 if not allocated.
    allocated_width, allocated_height : int
        Size of the texture in pixels.
    pixel_format : int
        Internal pixel format of the texture, e.g. GL_RGBA8.
    frame_width, frame_height : int
        Size of the frame stored in the buffer, at most the allocated size.
        The frame always starts at (0, 0) of the texture. Set by P with
        `set_frame_size()`; 0 if the buffer holds no frame.
    frame_serial : int
        Number of the published frame stored in the buffer, 0 if the buffer
        holds no frame that may be shown (empty, or rendering failed).
        Owned by `BufferManager`, only changed under its lock.
    last_use_fence : GLsync or None
        Fence after the last GPU commands that used the buffer.
        Owned by `BufferManager`.
    last_use_fence_placed_by_writer : bool
        True if P placed `last_use_fence`, False if C did.
        Owned by `BufferManager`.
    """

    __slots__ = (
        "allocated_height",
        "allocated_width",
        "fbo_id",
        "frame_height",
        "frame_serial",
        "frame_width",
        "last_use_fence",
        "last_use_fence_placed_by_writer",
        "pixel_format",
        "texture_id",
    )

    def __init__(self) -> None:
        self.texture_id = 0
        self.fbo_id = 0
        self.allocated_width = 0
        self.allocated_height = 0
        self.pixel_format = 0
        self.frame_width = 0
        self.frame_height = 0
        self.frame_serial = 0
        self.last_use_fence: Any = None
        self.last_use_fence_placed_by_writer = False

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(texture_id={self.texture_id}, fbo_id={self.fbo_id}, "
            f"allocated={self.allocated_width}x{self.allocated_height}, "
            f"frame={self.frame_width}x{self.frame_height}, "
            f"frame_serial={self.frame_serial})"
        )

    @property
    def is_allocated(self) -> bool:
        return self.texture_id != 0

    def set_frame_size(
        self, frame_width: int, frame_height: int, pixel_format: int
    ) -> bool:
        """Set the size of the frame about to be rendered, reallocating if needed.

        P calls this before rendering each frame. The buffer is reallocated if
        it's too small or has another pixel format; smaller frames reuse the
        allocation and occupy its (0, 0, frame_width, frame_height) area.

        Parameters
        ----------
        frame_width, frame_height : int
            Frame size in pixels.
        pixel_format : int
            Internal pixel format, e.g. GL_RGBA8.

        Returns
        -------
        bool
            True if the buffer was reallocated. Its contents are undefined then.

        Raises
        ------
        ValueError
            If the size isn't positive or the pixel format is unknown.
        RuntimeError
            If the FBO is incomplete.
        In both cases the buffer is left unchanged.

        Notes
        -----
        * The calling thread's GL context must be current.
        * Only call this inside `BufferManager.acquire_write_buffer()`: the
          buffer then belongs to P, and P's GPU queue already waits for C's
          last use of it.
        """
        reallocated = self._ensure_capacity(frame_width, frame_height, pixel_format)
        self.frame_width = frame_width
        self.frame_height = frame_height
        return reallocated

    def destroy(self) -> None:
        """Delete the texture and FBO. A GL context of the share group must be current."""
        gl_fbo_destroy(self.fbo_id)
        gl_texture_destroy(self.texture_id)
        self.texture_id = 0
        self.fbo_id = 0
        self.allocated_width = 0
        self.allocated_height = 0
        self.frame_width = 0
        self.frame_height = 0
        self.frame_serial = 0

    def _ensure_capacity(self, width: int, height: int, pixel_format: int) -> bool:
        """Reallocate if the buffer can't hold `width` x `height` of `pixel_format`.

        Returns True if the buffer was (re)allocated. On failure, the buffer
        is left unchanged. Allocations only grow: each dimension becomes the
        larger of the requested and the allocated size.

        A new texture and FBO are created instead of re-specifying the existing
        texture, which would be a hazard for a texture shared between contexts.
        GL defers deleting the old ones while pending commands still use them.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid buffer size {width}x{height}")

        fits_already = (
            self.is_allocated
            and width <= self.allocated_width
            and height <= self.allocated_height
            and pixel_format == self.pixel_format
        )
        if fits_already:
            return False

        # only grow: a dimension that already fits keeps its allocated size
        new_width = max(width, self.allocated_width)
        new_height = max(height, self.allocated_height)

        # create the new objects first, so a failure leaves the buffer intact
        new_texture_id = gl_texture_create(new_width, new_height, pixel_format)
        try:
            new_fbo_id = gl_fbo_create(new_texture_id)
        except Exception:
            gl_texture_destroy(new_texture_id)
            raise

        gl_fbo_destroy(self.fbo_id)
        gl_texture_destroy(self.texture_id)

        self.texture_id = new_texture_id
        self.fbo_id = new_fbo_id
        self.allocated_width = new_width
        self.allocated_height = new_height
        self.pixel_format = pixel_format
        # the new texture holds no frame
        self.frame_width = 0
        self.frame_height = 0
        return True


class ReadFrame(NamedTuple):
    """What `BufferManager.acquire_read_buffer()` hands to C.

    Attributes
    ----------
    buffer : FrameBuffer
        The buffer holding the frame. Valid until the `with` block is left.
    frame_serial : int
        Number of the frame, increasing by one for each published frame.
    is_new : bool
        False if C got the same frame in its previous read.
    frames_skipped : int
        Frames published since C's previous read that C never got.
    """

    buffer: FrameBuffer
    frame_serial: int
    is_new: bool
    frames_skipped: int


class _WriteBufferHandle:
    """Reusable context manager returned by `BufferManager.acquire_write_buffer()`."""

    __slots__ = ("_buffer", "_manager")

    def __init__(self, manager: BufferManager) -> None:
        self._manager = manager
        self._buffer: FrameBuffer | None = None

    def __enter__(self) -> FrameBuffer:
        # on failure, the manager has already undone the claim
        self._buffer = self._manager._claim_write_buffer()
        return self._buffer

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        buffer = self._buffer
        self._buffer = None
        if buffer is None:
            return
        # always fence and release, even if rendering failed: commands may be queued
        completion_fence = None
        try:
            completion_fence = _insert_fence_and_flush()
        finally:
            self._manager._release_write_buffer(
                buffer,
                completion_fence,
                publish=exc_type is None and completion_fence is not None,
            )


class _ReadBufferHandle:
    """Reusable context manager returned by `BufferManager.acquire_read_buffer()`."""

    __slots__ = ("_frame", "_is_active", "_manager", "timeout")

    def __init__(self, manager: BufferManager) -> None:
        self._manager = manager
        self._frame: ReadFrame | None = None
        self._is_active = False
        self.timeout = 0.0

    def __enter__(self) -> ReadFrame | None:
        # also catches nesting when the outer block got no frame
        if self._is_active:
            raise RuntimeError("Nested acquire_read_buffer() blocks aren't supported")
        self._is_active = True
        try:
            # on failure, the manager has already undone the claim
            self._frame = self._manager._claim_read_buffer(self.timeout)
        except BaseException:
            self._is_active = False
            raise
        return self._frame

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        frame = self._frame
        self._frame = None
        self._is_active = False
        if frame is None:
            return
        # always fence and release, even if drawing failed: commands may be queued
        completion_fence = None
        try:
            completion_fence = _insert_fence_and_flush()
        finally:
            self._manager._release_read_buffer(frame.buffer, completion_fence)


class BufferManager:
    """Thread-safe triple buffering for one producer (P) and one consumer (C).

    P takes a buffer with `acquire_write_buffer()` whenever a frame arrives, and
    never waits on the CPU. C takes the freshest finished frame with
    `acquire_read_buffer()`, and may wait a limited time for the frame P is
    rendering.

    Parameters
    ----------
    width, height : int
        Initial size of the buffers in pixels.
    pixel_format : int
        Internal pixel format of the buffers, e.g. GL_RGBA8.

    Raises
    ------
    ValueError, RuntimeError
        If allocating the buffers fails (see `FrameBuffer.set_frame_size()`).
        Buffers allocated up to that point are deleted.

    Notes
    -----
    * The constructor and `close()` need a GL context of the share group to
      be current. `acquire_write_buffer()` and `acquire_read_buffer()` need the
      calling thread's context to be current.
    * Buffer selection:
        * P takes neither the newest published buffer nor the one C holds,
          and of the rest (one or two buffers) the one with the oldest frame.
        * C takes the newest published buffer that P doesn't hold.
    * Exceptions are never handled here, only propagated. The manager only
      keeps its bookkeeping consistent: claims are undone or released, and a
      buffer whose rendering failed is never shown. How to respond to the
      exception is up to the caller.
    """

    def __init__(self, width: int, height: int, pixel_format: int) -> None:
        self._lock = threading.Lock()
        # notified by P when a frame is published, and by close()
        self._frame_published = threading.Condition(self._lock)

        self._buffers = tuple(FrameBuffer() for _ in range(BUFFER_COUNT))
        try:
            for buffer in self._buffers:
                buffer._ensure_capacity(width, height, pixel_format)
        except Exception:
            for buffer in self._buffers:
                buffer.destroy()
            raise

        # guarded by _lock
        self._buffer_held_by_writer: FrameBuffer | None = None
        self._buffer_held_by_reader: FrameBuffer | None = None
        self._published_frame_count = 0
        self._last_read_frame_serial = 0
        self._write_started_at: float | None = None
        self._is_closed = False

        # reused for every block, so a frame allocates no context managers
        self._write_buffer_handle = _WriteBufferHandle(self)
        self._read_buffer_handle = _ReadBufferHandle(self)

    # ---- public API ----

    @property
    def buffers(self) -> tuple[FrameBuffer, ...]:
        """All buffers, e.g. for inspection. Only use them inside acquire blocks."""
        return self._buffers

    @property
    def write_buffer_is_held(self) -> bool:
        """True while P holds a buffer, i.e. is rendering a frame.

        A hint without locking, for C to decide whether estimating a wait
        budget is worth it. It may change right after reading it.
        """
        return self._buffer_held_by_writer is not None

    @property
    def write_started_at(self) -> float | None:
        """`time.perf_counter()` when P took its current buffer, None if P holds none.

        A hint without locking, like `write_buffer_is_held`.
        """
        return self._write_started_at

    @property
    def published_frame_count(self) -> int:
        """Number of frames P has published so far."""
        return self._published_frame_count

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def acquire_write_buffer(self) -> _WriteBufferHandle:
        """Return a context manager that hands P a buffer to render into.

        Entering the `with` block takes the buffer and makes P's GPU queue wait
        until C's last use of it is finished. Leaving it fences P's commands,
        releases the buffer and publishes the frame. If the block raises, the
        buffer is released without publishing, and it's never shown, as its
        contents are unknown. The exception propagates.

        Raises
        ------
        RuntimeError
            On entering, if P already holds a buffer or the manager is closed.
        """
        return self._write_buffer_handle

    def acquire_read_buffer(self, timeout: float = 0.0) -> _ReadBufferHandle:
        """Return a context manager that hands C the freshest finished frame.

        Entering the `with` block binds a `ReadFrame`, or None if no frame was
        published yet (or the manager was closed while waiting). It makes C's
        GPU queue wait until P's rendering of the frame is finished. Leaving
        it fences C's commands and releases the buffer, also if the block
        raises. The exception propagates.

        Parameters
        ----------
        timeout : float
            Longest time in seconds that C waits for the frame P is rendering
            at the time of entering. 0 means don't wait. If P isn't rendering,
            there's no wait at all. On timeout, C gets the newest frame P
            finished before.

        Raises
        ------
        ValueError
            If `timeout` is None or negative.
        RuntimeError
            On entering, if C already holds a buffer or the manager is closed.
        """
        if timeout is None or timeout < 0:
            raise ValueError(f"timeout must be a number >= 0, got {timeout!r}")
        self._read_buffer_handle.timeout = timeout
        return self._read_buffer_handle

    def close(self) -> None:
        """Wake a waiting C, delete all fences and buffers. Safe to call repeatedly.

        P and C are expected to have stopped using the buffers. If either still
        holds one, a warning is logged and the buffers are deleted anyway.
        """
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._frame_published.notify_all()

            held_by_writer = self._buffer_held_by_writer
            held_by_reader = self._buffer_held_by_reader
            fences_to_delete = [buffer.last_use_fence for buffer in self._buffers]
            for buffer in self._buffers:
                buffer.last_use_fence = None

        if held_by_writer is not None or held_by_reader is not None:
            logging.warning(
                "BufferManager closed while buffers were in use "
                f"(writer: {held_by_writer}, reader: {held_by_reader})"
            )

        # GL calls outside the lock
        for fence in fences_to_delete:
            if fence:
                gl.glDeleteSync(fence)
        for buffer in self._buffers:
            buffer.destroy()

    # ---- P side, called by _WriteBufferHandle ----

    def _claim_write_buffer(self) -> FrameBuffer:
        with self._lock:
            if self._is_closed:
                raise RuntimeError("BufferManager is closed")
            if self._buffer_held_by_writer is not None:
                raise RuntimeError(
                    "Nested acquire_write_buffer() blocks aren't supported"
                )

            buffer = self._choose_buffer_to_write()
            self._buffer_held_by_writer = buffer
            self._write_started_at = time.perf_counter()
            # take the fence: from now on, only P touches this buffer
            fence_of_last_use = buffer.last_use_fence
            fence_placed_by_writer = buffer.last_use_fence_placed_by_writer
            buffer.last_use_fence = None

        try:
            # GPU waits until C's last draw from this buffer is done
            _wait_for_fence_and_delete(
                fence_of_last_use, gpu_wait_needed=not fence_placed_by_writer
            )
        except BaseException:
            # undo the claim, so the buffer stays usable. If only the deletion
            # failed, the next taker waits on the fence again, which is harmless.
            with self._lock:
                buffer.last_use_fence = fence_of_last_use
                buffer.last_use_fence_placed_by_writer = fence_placed_by_writer
                self._buffer_held_by_writer = None
                self._write_started_at = None
            raise
        return buffer

    def _choose_buffer_to_write(self) -> FrameBuffer:
        """Pick P's buffer. Caller holds the lock."""
        newest_buffer = self._find_newest_published_buffer(excluded=None)
        candidates = [
            buffer
            for buffer in self._buffers
            if buffer is not newest_buffer and buffer is not self._buffer_held_by_reader
        ]
        # with 3 buffers, one or two candidates are left. Empty and failed
        # buffers have serial 0, so they are reused first.
        return min(candidates, key=lambda buffer: buffer.frame_serial)

    def _release_write_buffer(
        self, buffer: FrameBuffer, completion_fence: Any, publish: bool
    ) -> None:
        with self._lock:
            replaced_fence = buffer.last_use_fence
            buffer.last_use_fence = completion_fence
            buffer.last_use_fence_placed_by_writer = True
            self._buffer_held_by_writer = None
            self._write_started_at = None

            if publish:
                self._published_frame_count += 1
                buffer.frame_serial = self._published_frame_count
                self._frame_published.notify()
            else:
                # rendering failed, contents unknown: serial 0 hides the buffer
                # from C, like an empty one
                buffer.frame_serial = 0
                buffer.frame_width = 0
                buffer.frame_height = 0

        # normally None: P took the fence when claiming the buffer
        if replaced_fence:
            gl.glDeleteSync(replaced_fence)

    # ---- C side, called by _ReadBufferHandle ----

    def _claim_read_buffer(self, timeout: float) -> ReadFrame | None:
        with self._lock:
            if self._is_closed:
                raise RuntimeError("BufferManager is closed")
            if self._buffer_held_by_reader is not None:
                raise RuntimeError(
                    "Nested acquire_read_buffer() blocks aren't supported"
                )

            # wait for the frame P is rendering now, if any, not for later ones
            if timeout > 0 and self._buffer_held_by_writer is not None:
                frame_count_at_entry = self._published_frame_count
                self._frame_published.wait_for(
                    lambda: (
                        self._published_frame_count > frame_count_at_entry
                        or self._is_closed
                    ),
                    timeout,
                )
                if self._is_closed:
                    return None

            # newest frame P doesn't hold; P may have started another one meanwhile
            buffer = self._find_newest_published_buffer(
                excluded=self._buffer_held_by_writer
            )
            if buffer is None:
                return None

            self._buffer_held_by_reader = buffer
            previous_read_frame_serial = self._last_read_frame_serial
            frame_serial = buffer.frame_serial
            is_new = frame_serial != previous_read_frame_serial
            frames_skipped = (
                max(frame_serial - previous_read_frame_serial - 1, 0) if is_new else 0
            )
            self._last_read_frame_serial = frame_serial
            # take the fence: P can't take this buffer until C releases it
            fence_of_last_use = buffer.last_use_fence
            fence_placed_by_writer = buffer.last_use_fence_placed_by_writer
            buffer.last_use_fence = None

        try:
            # GPU waits until P's rendering into this buffer is done
            _wait_for_fence_and_delete(
                fence_of_last_use, gpu_wait_needed=fence_placed_by_writer
            )
        except BaseException:
            # undo the claim, so the buffer stays usable (see _claim_write_buffer)
            with self._lock:
                buffer.last_use_fence = fence_of_last_use
                buffer.last_use_fence_placed_by_writer = fence_placed_by_writer
                self._buffer_held_by_reader = None
                self._last_read_frame_serial = previous_read_frame_serial
            raise
        return ReadFrame(buffer, frame_serial, is_new, frames_skipped)

    def _release_read_buffer(self, buffer: FrameBuffer, completion_fence: Any) -> None:
        with self._lock:
            replaced_fence = buffer.last_use_fence
            buffer.last_use_fence = completion_fence
            buffer.last_use_fence_placed_by_writer = False
            self._buffer_held_by_reader = None
            # no notify: P never waits on the CPU

        # normally None: C took the fence when claiming the buffer
        if replaced_fence:
            gl.glDeleteSync(replaced_fence)

    # ---- helpers ----

    def _find_newest_published_buffer(
        self, excluded: FrameBuffer | None
    ) -> FrameBuffer | None:
        """Return the buffer with the newest frame, None if none. Caller holds the lock."""
        newest_buffer = None
        for buffer in self._buffers:
            if buffer is excluded or buffer.frame_serial == 0:
                continue
            if (
                newest_buffer is None
                or buffer.frame_serial > newest_buffer.frame_serial
            ):
                newest_buffer = buffer
        return newest_buffer
