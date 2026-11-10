from __future__ import annotations

import threading
from dataclasses import dataclass
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


@dataclass(slots=True, frozen=True)
class BufferInfo:
    texture_id: int
    fbo_id: int


class FrameBuffer:
    __slots__ = (
        "allocated_height",
        "allocated_width",
        "fbo_id",
        "frame_height",
        "frame_width",
        "has_frame",
        "last_use_by_writer",
        "last_use_fence",
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
    has_frame: bool
    last_use_fence: Any
    last_use_by_writer: bool

    def __init__(self, pixel_format: int, width: int, height: int):
        self.pixel_format = pixel_format
        self._allocate(width, height, pixel_format)
        self.frame_width = 0
        self.frame_height = 0
        self.has_frame = False
        self.last_use_fence = None
        self.last_use_by_writer = False

    def is_valid(self) -> bool:
        return self.texture_id != 0
        # and self.fbo_id != 0

    def _allocate(self, w: int, h: int, pixel_format: int) -> None:
        if self.texture_id != 0 or self.fbo_id != 0:
            return
        tex = gl_texture_create(w, h, pixel_format)
        try:
            fbo = gl_fbo_create(tex)
        except Exception:
            gl_texture_destroy(tex)
            raise
        self.texture_id = tex
        self.fbo_id = fbo
        self.allocated_width = w
        self.allocated_height = h

    def frame_fbo_info(self) -> dict[str, int]:
        return {
            "fbo": self.fbo_id,
            "w": self.frame_width,
            "h": self.frame_height,
            "internal_format": self.pixel_format,
        }

    def _frame_fits(self, w: int, h: int, pixel_format: int) -> bool:
        return (
            pixel_format == self.pixel_format
            and w <= self.allocated_width
            and h <= self.allocated_height
        )

    def release(self) -> None:
        if self.texture_id != 0:
            gl_texture_destroy(self.texture_id)
            self.texture_id = 0
        if self.fbo_id != 0:
            gl_fbo_destroy(self.fbo_id)
            self.fbo_id = 0
        self.allocated_width = 0
        self.allocated_height = 0

    def _ensure_frame_size(self, w: int, h: int, pixel_format: int) -> None:
        if not self._frame_fits(w, h, pixel_format):
            self._allocate(w, h, pixel_format)
        self.frame_width = w
        self.frame_height = h

    def __del__(self) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(texture_id={self.texture_id}, fbo_id={self.fbo_id}, "
            f"pixel_format={self.pixel_format}, allocated_width={self.allocated_width}, "
            f"allocated_height={self.allocated_height}, frame_width={self.frame_width}, "
            f"frame_height={self.frame_height}, has_frame={self.has_frame})"
        )

    def before_use(self, width: int, height: int, pixel_format: int, for_writing: bool) -> None:




class TripleBufferManager:
    __slots__ = (
        "_active",
        "_buffers",
        "_frame_ready",
        "_freshest_slot",
        "_lock",
        "_slot_held_by_reader",
        "_slot_held_by_writer",
    )

    _lock: threading.Lock
    _frame_ready: threading.Condition
    _buffers: tuple[FrameBuffer, ...]
    _slot_held_by_writer: int | None
    _slot_held_by_reader: int | None
    _freshest_slot: int
    _active: bool

    def __init__(
        self,
        num_buffers: int = 3,
        width: int = 1920,
        height: int = 1080,
        pixel_format: int = gl.GL_RGBA8,
    ):
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._slot_held_by_writer = None
        self._slot_held_by_reader = None
        self._buffers = tuple(
            FrameBuffer(pixel_format=pixel_format, width=width, height=height)
            for _ in range(num_buffers)
        )
        self._freshest_slot = 0
        self._active = True
