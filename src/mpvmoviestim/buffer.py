"""Triple-buffered frame exchange between one producer and one consumer thread.

The producer (P, render worker) renders each video frame mpv pushes into an
intermediate buffer (texture + FBO) as soon as it arrives. The consumer (C, draw
thread) draws the freshest finished frame once per screen refresh. Both threads
have their own GL context in the same share group (textures, FBOs and sync
objects are shared by pyglet).

Design
------
* Three buffers: P never waits on the CPU, and C always has a finished frame.
  P takes neither the freshest buffer nor C's, but the oldest of the rest.
  C takes the freshest buffer.
* C can wait a limited time (`timeout`) for the frame P is rendering at the
  time of the call, to get the freshest frame possible.
* CPU: one lock guards which buffer each thread holds; a condition on the same
  lock lets C wait for P. No GL calls are made while holding the lock, except
  when freeing buffers after shutdown.
* GPU: each buffer keeps the fence of its last user. The next user's GPU queue
  waits for it (`glWaitSync`, the CPU doesn't block) only if the other thread
  placed it. Releasing a buffer places a new fence and flushes.
* Buffers only grow, one at a time: P reallocates the buffer it is about to
  write if the frame doesn't fit, so the other buffers keep their frames.

Usage
-----
Each thread must have its own GL context current::

    # P (render worker thread)
    try:
        while running:
            buffer = manager.acquire_write_buffer(width, height, pixel_format)
            render_into(buffer.frame_fbo_info())
            manager.release_write_buffer()
    finally:
        manager.shutdown()

    # C (draw thread), once per screen refresh
    buffer = manager.acquire_read_buffer(timeout=wait_budget)
    if buffer is not None:
        try:
            draw(buffer.texture_id, buffer.frame_width, buffer.frame_height, ...)
        finally:
            manager.release_read_buffer()

Error handling
--------------
Errors aren't handled or repaired here: exceptions propagate, and they should
end the program. What happens on an error:

* P: if rendering or allocation raises, P doesn't release its buffer, so the
  unfinished frame is never shown. P's `finally` calls `shutdown()`, which
  frees all buffers, including P's.
* C: after shutdown, `acquire_read_buffer()` raises `RuntimeError` (also when
  woken from a wait), so C learns that P has stopped. P's exception itself
  isn't forwarded: the caller must store it and re-raise it on C's side.
* If C holds a buffer at shutdown, C frees it in `release_read_buffer()`.

Limitations
-----------
* Exactly one producer and one consumer thread. Nested acquires raise.
* Only P calls `shutdown()` (`release_write_buffer()` relies on this).
* Attributes of `FrameBuffer` are read-only for P and C, and buffers may only
  be used between acquire and release.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pyglet import gl

from .utils import (
    gl_fbo_create,
    gl_fbo_destroy,
    gl_texture_create,
    gl_texture_destroy,
)

if TYPE_CHECKING:
    from typing import Any


class FrameBuffer:
    """Intermediate buffer: a texture, the FBO it's attached to, and metadata.

    Created and managed by `TripleBufferManager`; P and C only read the
    attributes, between acquiring and releasing the buffer.

    Attributes
    ----------
    texture_id, fbo_id : int
        GL names of the texture and its FBO, 0 if not allocated.
    pixel_format : int
        Internal pixel format of the texture, e.g. GL_RGBA8.
    allocated_width, allocated_height : int
        Size of the texture in pixels.
    frame_width, frame_height : int
        Size of the frame stored in the buffer. The frame occupies the
        (0, 0, frame_width, frame_height) area of the texture.
    """

    __slots__ = (
        "_has_frame",
        "_last_use_by_writer",
        "_last_use_fence",
        "allocated_height",
        "allocated_width",
        "fbo_id",
        "frame_height",
        "frame_width",
        "pixel_format",
        "texture_id",
    )
    texture_id: int
    fbo_id: int
    pixel_format: int
    allocated_width: int
    allocated_height: int
    frame_width: int
    frame_height: int
    _has_frame: bool  # holds a finished frame that P released
    _last_use_fence: Any  # GLsync after the last user's commands, or None
    _last_use_by_writer: bool  # True if P placed _last_use_fence

    def __init__(self, pixel_format: int, width: int, height: int):
        self.pixel_format = pixel_format
        self.frame_width = 0
        self.frame_height = 0
        self._has_frame = False
        self._last_use_fence = None
        self._last_use_by_writer = False
        self.texture_id = 0
        self.fbo_id = 0
        self.allocated_width = 0
        self.allocated_height = 0
        self._allocate(width, height, pixel_format)

    def _allocate(self, w: int, h: int, pixel_format: int) -> None:
        if self.texture_id != 0 or self.fbo_id != 0:
            raise ValueError("Buffer already allocated")
        tex = gl_texture_create(w, h, pixel_format)
        try:
            fbo = gl_fbo_create(tex)
        except Exception:
            gl_texture_destroy(tex)
            raise
        self.texture_id = tex
        self.fbo_id = fbo
        self.pixel_format = pixel_format
        self.allocated_width = w
        self.allocated_height = h

    def frame_fbo_info(self) -> dict[str, int]:
        """Return mpv's `opengl_fbo` render parameter for rendering into this buffer."""
        return {
            "fbo": self.fbo_id,
            "w": self.frame_width,
            "h": self.frame_height,
            "internal_format": self.pixel_format,
        }

    def guard_before(
        self, for_writing: bool, width: int = 0, height: int = 0, pixel_format: int = 0
    ) -> None:
        """Manager only: prepare the buffer for its new holder.

        For writing, (re)allocate it for the frame size first. Then make the
        holder's GPU queue wait for the other thread's last use.
        """
        if for_writing:
            # a reallocated buffer has no fence: its new texture has no earlier use
            self._ensure_frame_size(width, height, pixel_format)
            self._has_frame = False
        else:
            assert self._has_frame, "FrameBuffer has no frame available for reading"

        if self._last_use_fence is not None:
            # own fence: no wait needed, commands within a context run in order
            if for_writing != self._last_use_by_writer:
                gl.glWaitSync(self._last_use_fence, 0, gl.GL_TIMEOUT_IGNORED)
            # GL defers the deletion while a wait on the fence is pending
            gl.glDeleteSync(self._last_use_fence)
            self._last_use_fence = None

    def _ensure_frame_size(self, w: int, h: int, pixel_format: int) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"Width and height must be positive integers, got {w}x{h}")

        frame_fits = (
            pixel_format == self.pixel_format
            and w <= self.allocated_width
            and h <= self.allocated_height
        )
        if not frame_fits:
            new_w = max(w, self.allocated_width)
            new_h = max(h, self.allocated_height)
            # free first (lower peak VRAM). GL keeps the old objects alive while
            # queued commands still use them, so no fence wait is needed.
            self.deallocate()
            # only grow: a dimension that already fits keeps its allocated size
            self._allocate(
                w=new_w,
                h=new_h,
                pixel_format=pixel_format,
            )
        self.frame_width = w
        self.frame_height = h

    def deallocate(self) -> None:
        """Manager only: delete the fence, texture and FBO. Safe to call repeatedly."""
        self.allocated_width = 0
        self.allocated_height = 0
        self._has_frame = False
        self.frame_height = 0
        self.frame_width = 0
        if self._last_use_fence is not None:
            gl.glDeleteSync(self._last_use_fence)
            self._last_use_fence = None
        if self.texture_id != 0:
            gl_texture_destroy(self.texture_id)
            self.texture_id = 0
        if self.fbo_id != 0:
            gl_fbo_destroy(self.fbo_id)
            self.fbo_id = 0

    def guard_after(self, for_writing: bool) -> None:
        """Manager only: place a fence after the holder's commands, for the next user."""
        if for_writing:
            self._has_frame = True
        self._last_use_fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        # required: the other context may wait forever on an unsubmitted fence
        gl.glFlush()
        self._last_use_by_writer = for_writing

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(texture_id={self.texture_id}, fbo_id={self.fbo_id}, "
            f"pixel_format={self.pixel_format}, allocated_width={self.allocated_width}, "
            f"allocated_height={self.allocated_height}, frame_width={self.frame_width}, "
            f"frame_height={self.frame_height}, has_frame={self._has_frame})"
        )


class TripleBufferManager:
    """Thread-safe triple buffering for one producer (P) and one consumer (C).

    See the module docstring for usage, error handling and limitations.

    Parameters
    ----------
    width, height : int
        Initial size of the three buffers in pixels.
    pixel_format : int
        Internal pixel format of the buffers, e.g. GL_RGBA8.

    Raises
    ------
    ValueError, RuntimeError
        If allocating a buffer fails. Buffers allocated up to then are freed.

    Notes
    -----
    A GL context of the share group must be current.
    """

    __slots__ = (
        "_active",
        "_buffer_order",
        "_buffers",
        "_frame_count",
        "_frame_ready",
        "_lock",
        "_slot_held_by_reader",
        "_slot_held_by_writer",
        # "_timeout_counter",
    )

    _lock: threading.Lock
    _frame_ready: threading.Condition  # on _lock, notified on publish and shutdown
    _frame_count: int  # frames published so far
    _buffers: tuple[FrameBuffer, FrameBuffer, FrameBuffer]
    _slot_held_by_writer: int | None
    _slot_held_by_reader: int | None
    _buffer_order: tuple[int, int, int]  # buffer indices, freshest to oldest
    _active: bool
    # _timeout_counter: int

    def __init__(
        self,
        width: int,
        height: int,
        pixel_format: int,
    ):
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._frame_count = 0
        # self._timeout_counter = 0
        self._slot_held_by_writer = None
        self._slot_held_by_reader = None
        buffers: list[FrameBuffer] = []
        try:
            for _ in range(3):
                buffers.append(FrameBuffer(pixel_format, width, height))
        except BaseException:
            for buf in buffers:
                buf.deallocate()
            raise
        self._buffers = (buffers[0], buffers[1], buffers[2])
        self._buffer_order = (0, 1, 2)  # freshest to oldest
        self._active = True

    def acquire_read_buffer(self, timeout: float = 0.0) -> FrameBuffer | None:
        """C: take the freshest finished frame. Release it with `release_read_buffer()`.

        If P is rendering a frame at the time of the call and `timeout` > 0,
        wait at most `timeout` seconds for that frame (not for later ones). On
        timeout, or without waiting, C gets the freshest frame P finished
        before. C's GPU queue waits for P's rendering to finish; the CPU
        doesn't block.

        Parameters
        ----------
        timeout : float
            Longest wait in seconds. 0 or less doesn't wait.

        Returns
        -------
        FrameBuffer or None
            None if no frame was published yet; don't release it in that case.

        Raises
        ------
        RuntimeError
            If the manager was shut down (also while waiting), or C already
            holds a buffer.
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("TripleBufferManager is not active")
            if self._slot_held_by_reader is not None:
                raise RuntimeError("Reader already holds a buffer")

            # wait only for the frame P is rendering now, not for later ones
            if timeout > 0 and self._slot_held_by_writer is not None:
                frame_count_at_wait_start = self._frame_count
                # writer_finished_in_time =
                self._frame_ready.wait_for(
                    lambda: (
                        self._frame_count > frame_count_at_wait_start
                        or not self._active
                    ),
                    timeout=timeout,
                )
                # if not writer_finished_in_time:
                #     self._timeout_counter += 1
                if not self._active:
                    raise RuntimeError("TripleBufferManager is not active")

            if self._frame_count == 0:
                return None

            freshest_buffer = self._buffer_order[0]
            assert self._slot_held_by_writer != freshest_buffer, (
                "Writer holding the freshest buffer"
            )

            self._slot_held_by_reader = freshest_buffer
            read_buffer = self._buffers[self._slot_held_by_reader]

        # GL calls outside the lock: the buffer is C's until released
        read_buffer.guard_before(for_writing=False)
        return read_buffer

    def release_read_buffer(self) -> None:
        """C: give back the buffer from `acquire_read_buffer()`.

        Places a fence after C's draw commands, so P doesn't overwrite the
        buffer before they're done. After shutdown, frees the buffer instead.

        Raises
        ------
        RuntimeError
            If C holds no buffer.
        """
        if self._slot_held_by_reader is None:
            raise RuntimeError("Reader does not hold a buffer")
        buffer = self._buffers[self._slot_held_by_reader]
        buffer.guard_after(for_writing=False)
        with self._lock:
            # shutdown() leaves the buffer C holds to C
            if not self._active:
                buffer.deallocate()
            self._slot_held_by_reader = None

    def acquire_write_buffer(
        self, width: int, height: int, pixel_format: int
    ) -> FrameBuffer:
        """P: take a buffer for a `width` x `height` frame. Never waits on the CPU.

        The buffer is reallocated (grown) if the frame doesn't fit or the pixel
        format differs. P's GPU queue waits for C's last use of the buffer.
        Publish the frame with `release_write_buffer()`.

        Parameters
        ----------
        width, height : int
            Size of the frame about to be rendered.
        pixel_format : int
            Internal pixel format, e.g. GL_RGBA8.

        Raises
        ------
        RuntimeError
            If the manager was shut down, P already holds a buffer, or the
            reallocated FBO is incomplete.
        ValueError
            If the size isn't positive or the pixel format is unknown.

        Notes
        -----
        If reallocation fails, the buffer is left unallocated and can't be
        used any more: call `shutdown()`.
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("TripleBufferManager is not active")
            if self._slot_held_by_writer is not None:
                raise RuntimeError("Writer already holds a buffer")

            # the oldest buffer, unless C holds it; never the freshest
            if self._slot_held_by_reader is None:
                self._slot_held_by_writer = self._buffer_order[2]
            else:
                oldest_buf = self._buffer_order[2]
                second_oldest_buf = self._buffer_order[1]
                self._slot_held_by_writer = (
                    second_oldest_buf
                    if self._slot_held_by_reader == oldest_buf
                    else oldest_buf
                )
            write_buffer = self._buffers[self._slot_held_by_writer]

        # GL calls outside the lock: the buffer is P's until released
        write_buffer.guard_before(
            for_writing=True, width=width, height=height, pixel_format=pixel_format
        )
        return write_buffer

    def release_write_buffer(self) -> None:
        """P: publish the rendered frame as the freshest one, and wake a waiting C.

        Only call it after successful rendering: a released buffer is shown.
        Does nothing after shutdown.

        Raises
        ------
        RuntimeError
            If P holds no buffer.
        """
        # no lock needed: only P calls shutdown()
        if not self._active:
            return
        writer_slot = self._slot_held_by_writer
        if writer_slot is None:
            raise RuntimeError("Writer does not hold a buffer")
        buffer = self._buffers[writer_slot]
        buffer.guard_after(for_writing=True)

        with self._lock:
            self._update_buffer_order(writer_slot)
            self._frame_count += 1
            self._slot_held_by_writer = None
            self._frame_ready.notify_all()

    def _update_buffer_order(self, newest_slot: int) -> None:
        # Caller holds the lock. P never writes the freshest buffer, so
        # newest_slot is b1 or b2 in practice.
        b0, b1, b2 = self._buffer_order
        if newest_slot == b0:  # No change needed
            return
        elif newest_slot == b1:
            self._buffer_order = (newest_slot, b0, b2)
        else:  # newest_slot == oldest
            self._buffer_order = (newest_slot, b0, b1)

    def shutdown(self) -> None:
        """P: stop the manager and free the buffers. Safe to call repeatedly.

        Wakes a waiting C, whose `acquire_read_buffer()` then raises. Frees all
        buffers, including the one P holds, except the one C holds: C frees
        that in `release_read_buffer()`.

        Notes
        -----
        * Only P may call this, with its GL context current.
        * Call it in P's `finally`, so it also runs when rendering fails.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._frame_ready.notify_all()
            # taken under the lock: C can't claim or release a buffer in between
            reader_slot = self._slot_held_by_reader
        for slot, buffer in enumerate(self._buffers):
            if slot != reader_slot:
                buffer.deallocate()

    @property
    def write_buffer_is_held(self) -> bool:
        """True while P holds a buffer, i.e. is rendering a frame.

        A hint without locking, for C to decide whether estimating a wait
        budget is worth it. It may change right after reading it.
        """
        return self._slot_held_by_writer is not None
