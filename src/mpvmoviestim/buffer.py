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
    _has_frame: bool
    _last_use_fence: Any
    _last_use_by_writer: bool

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
        return {
            "fbo": self.fbo_id,
            "w": self.frame_width,
            "h": self.frame_height,
            "internal_format": self.pixel_format,
        }

    def guard_before(
        self, for_writing: bool, width: int = 0, height: int = 0, pixel_format: int = 0
    ) -> None:
        # Ensure buffer capacity before writing to it. If reallocated, fence is deleted.
        if for_writing:
            self._ensure_frame_size(width, height, pixel_format)
            self._has_frame = False
        else:
            assert self._has_frame, "FrameBuffer has no frame available for reading"

        # Wait for the guard fence if it was set by the other thread, delete the fence
        if self._last_use_fence is not None:
            if for_writing != self._last_use_by_writer:
                gl.glWaitSync(self._last_use_fence, 0, gl.GL_TIMEOUT_IGNORED)
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
        # Set a guard fence for the other thread to wait on
        if for_writing:
            self._has_frame = True
        self._last_use_fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
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
    _frame_ready: threading.Condition
    _frame_count: int
    _buffers: tuple[FrameBuffer, FrameBuffer, FrameBuffer]
    _slot_held_by_writer: int | None
    _slot_held_by_reader: int | None
    _buffer_order: tuple[int, int, int]
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
        with self._lock:
            if not self._active:
                raise RuntimeError("TripleBufferManager is not active")
            if self._slot_held_by_reader is not None:
                raise RuntimeError("Reader already holds a buffer")

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

        read_buffer.guard_before(for_writing=False)
        return read_buffer

    def release_read_buffer(self) -> None:
        if self._slot_held_by_reader is None:
            raise RuntimeError("Reader does not hold a buffer")
        buffer = self._buffers[self._slot_held_by_reader]
        buffer.guard_after(for_writing=False)
        with self._lock:
            if not self._active:
                buffer.deallocate()
            self._slot_held_by_reader = None

    def acquire_write_buffer(
        self, width: int, height: int, pixel_format: int
    ) -> FrameBuffer:
        with self._lock:
            if not self._active:
                raise RuntimeError("TripleBufferManager is not active")
            if self._slot_held_by_writer is not None:
                raise RuntimeError("Writer already holds a buffer")

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

        write_buffer.guard_before(
            for_writing=True, width=width, height=height, pixel_format=pixel_format
        )
        return write_buffer

    def release_write_buffer(self) -> None:
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
        # Update the buffer order to reflect the new freshest buffer
        b0, b1, b2 = self._buffer_order
        if newest_slot == b0:  # No change needed
            return
        elif newest_slot == b1:
            self._buffer_order = (newest_slot, b0, b2)
        else:  # newest_slot == oldest
            self._buffer_order = (newest_slot, b0, b1)

    def shutdown(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._frame_ready.notify_all()
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
