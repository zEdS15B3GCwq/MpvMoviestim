# mypy: disable_error_code = import-untyped
from __future__ import annotations

import ctypes
import importlib
from typing import TYPE_CHECKING

import pyglet
from pyglet import gl

from psychopy import logging

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


def _resolve_gl_proc_with_pyglet(name: bytes) -> int:
    """Resolve GL function name to pointer using pyglet backends.

    Uses pyglet's internal OS-specific `pyglet.gl.lib_*` modules.

    Parameters
    ----------
    name : bytes
        GL function name

    Returns
    -------
    int
        proc address or 0 if not available or on error
    """
    name_bytes = name
    name_str = name_bytes.decode("utf-8", errors="ignore") or None

    platform = pyglet.compat_platform

    def _try_lib_export(lib: ModuleType, nm: str) -> int | None:
        func = getattr(lib, nm, None)
        if func is not None:
            try:
                val = ctypes.cast(func, ctypes.c_void_p).value
            except (TypeError, ValueError, AttributeError):
                # try attribute fallback
                val = getattr(func, "value", None)
            if val is not None:
                return int(val)
        return None

    # Windows
    if platform in ("win32", "cygwin"):
        pyglet_lib_name = "pyglet.gl.lib_wgl"
        getprocaddress_func_names = ["wglGetProcAddress"]

    # Linux
    elif platform.startswith("linux"):
        pyglet_lib_name = "pyglet.gl.lib_glx"
        getprocaddress_func_names = ["glXGetProcAddressARB", "glXGetProcAddress"]

    # macOS
    elif platform == "darwin":
        pyglet_lib_name = "pyglet.gl.lib_agl"
        getprocaddress_func_names = []

    else:
        logging.error(
            "Unsupported platform. Expected Windows (win32), Linux (linux), "
            f"macOS (darwin), got: {platform}"
        )
        return 0

    # import platfofm-specific pyglet lib, fail with 0 if not successful
    try:
        pyglet_lib = importlib.import_module(pyglet_lib_name)
    except ImportError:
        logging.error(
            f"Pyglet's platform-specific OpenGL library {pyglet_lib_name} not found",
        )
        return 0

    # plan A: check if func is exposed as attribute of gl_lib
    if name_str is not None:
        gl_lib = getattr(pyglet_lib, "gl_lib", None)
        if gl_lib is not None:
            addr = _try_lib_export(gl_lib, name_str)
            if addr is not None:
                return addr

    # plan B: call get_proc_address func(s) in library
    for func_name in getprocaddress_func_names:
        func = getattr(pyglet_lib, func_name, None)
        if func is not None:
            try:
                func.argtypes = [ctypes.c_char_p]
                func.restype = ctypes.c_void_p
                addr = func(name_bytes)
                if addr:
                    return int(addr)
            except (TypeError, OSError):
                continue

    logging.error(f"Proc address for OpenGL function {name_str} not found")
    return 0


def get_proc_address(_ctx, name: bytes) -> int:
    """Return GL function address or 0.

    Parameters
    ----------
    _ctx : OpenGL context

    name : bytes
        OpenGL function name

    Returns
    -------
    int
        proc address or 0 if not available or on error
    """
    try:
        return _resolve_gl_proc_with_pyglet(name)
    except Exception as e:  # pylint: disable=W0718
        logging.warning(f"get_proc_address unexpected error: {e}")
        return 0


def create_texture(w, h, preferred_format="rgba8") -> tuple[int, bool]:
    """Create a texture.

    Parameters
    ----------
    w, h : int
        width and height of texture in pixels

    preferred_format : str
        texture pixel format, one of: rgba8, rgba16f, rgba32f
        unknown or unsupported formats fall back to rgba8

    Returns
    -------
    int
        texture ID
    bool
        true if the incoming pixels and internal storage are floating point

    Notes
    -----
    * Incoming pixel format is always expected to be GL_RGBA
    * Incoming pixel type is determined from the preferred format:
      GL_FLOAT for float types and GL_UNSIGNED BYTE for unsigned byte types.
    * Internal floating point formats may not be supported on some mobile
      or legacy platforms. If not available, the internal format is set to
      GL_RGBA8, and the incoming pixel type to GL_UNSIGNED_BYTE. It would
      be possible to allow floating point pixel data uploads and let the
      driver take care of the conversion, but this module was designed to
      try to avoid such situations. Therefore, the incoming type is fixed
      to the storage type. Callers are expected to check the 2nd return
      value that indicates whether the incoming pixels and the storage
      types are expected to be floating point or not.
    """
    # Generate texture
    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    tex_id = tex.value

    # Bind and configure texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    # Pick texture storage format parameters
    incoming_format = getattr(gl, "GL_RGBA", gl.GL_RGBA)

    # Pick internal format according to requested format with fallback to GL_RGBA
    # This code is somewhat defensive in not assuming that the requested formats exist
    internal_format = gl.GL_RGBA8
    incoming_pixel_type = gl.GL_UNSIGNED_BYTE
    label = "GL_RGBA8"
    incoming_is_float = False
    label = ""
    if (
        preferred_format.lower() == "rgba32f"
        and (t := getattr(gl, "GL_RGBA32F", None)) is not None
    ):
        internal_format = t
        incoming_pixel_type = gl.GL_FLOAT
        label = "GL_RGBA32F"
        incoming_is_float = True
    elif (
        preferred_format.lower() == "rgba16f"
        and (t := getattr(gl, "GL_RGBA16F", None)) is not None
    ):
        internal_format = t
        incoming_pixel_type = gl.GL_FLOAT
        label = "GL_RGBA16F"
        incoming_is_float = True

    logging.debug(
        f"Generating GL texture with requested format {preferred_format}: "
        f"internal format={label}, "
        f"incoming type={'GL_FLOAT' if incoming_pixel_type == gl.GL_FLOAT else 'GL_UNSIGNED_BYTE'}, "
        "incoming format=GL_RGBA"
    )

    # Allocate storage
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        internal_format,
        w,
        h,
        0,
        incoming_format,
        incoming_pixel_type,
        None,
    )

    # Unbind
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    return tex_id, incoming_is_float


def destroy_texture(tex_id: int) -> None:
    """Destroy a texture.

    Parameters
    ----------
    tex_id : int
        GL texture ID
    """
    if tex_id != 0:
        t = ctypes.c_uint(tex_id)
        gl.glDeleteTextures(1, ctypes.byref(t))


def create_fbo(tex_id: int) -> int:
    """Create a framebuffer object (FBO) for the supplied texture.

    Parameters
    ----------
    tex_id : int
        GL texture ID

    Returns
    -------
    int
        FBO ID
    """
    # Generate FBO
    fbo = ctypes.c_uint(0)
    gl.glGenFramebuffers(1, ctypes.byref(fbo))
    fbo_id = fbo.value

    # Bind FBO and attach texture
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)
    gl.glFramebufferTexture2D(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex_id, 0
    )

    # Check status
    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        logging.error(f"Intermediate FBO incomplete (status={status}")

    # Unbind
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return fbo_id


def destroy_fbo(fbo_id: int) -> None:
    """Destroy a framebuffer object (FBO).

    Parameters
    ----------
    fbo_id : int
        FBO ID
    """
    if fbo_id != 0:
        f = ctypes.c_uint(fbo_id)
        gl.glDeleteFramebuffers(1, ctypes.byref(f))


def _intersect_rect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Calculate intersection of two rectangles.

    Parameters
    ----------
    a, b : tuple[int, int, int, int]
        Rectangles as (x, y, w, h).

    Returns
    -------
    tuple[int, int, int, int] or None
        Intersection as (x, y, w, h), or None if empty.

    Notes
    -----
    * if-else clauses are much faster than min/max().
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = ax if ax > bx else bx  # max
    y0 = ay if ay > by else by
    x1 = ax + aw if ax + aw < bx + bw else bx + bw  # min
    y1 = ay + ah if ay + ah < by + bh else by + bh
    if x1 <= x0 or y1 <= y0:
        # return (0, 0, -1, -1)
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _intersect_screen_rect(
    scr: tuple[int, int], rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Calculate intersection of a rectangle and the screen.

    Parameters
    ----------
    scr : tuple[int, int]
        Screen size as (width, height).
    rect : tuple[int, int, int, int]
        Rectangle as (x, y, w, h)


    Returns
    -------
    tuple[int, int, int, int] or None
        Intersection as (x, y, w, h), or None if empty.

    Notes
    -----
    * if-else clauses are much faster than min/max().
    """
    aw, ah = scr
    bx, by, bw, bh = rect
    x0 = bx if bx > 0 else 0  # max
    y0 = by if by > 0 else 0
    x1 = aw if aw < bx + bw else bx + bw  # min
    y1 = ah if ah < by + bh else by + bh
    if x1 <= x0 or y1 <= y0:
        # return (0, 0, -1, -1)
        return None
    return (x0, y0, x1 - x0, y1 - y0)


# TODO: test if the if shortcuts make it faster or slower
def get_blit_fn(
    src_size: tuple[int, int],
    dst_rect: tuple[int, int, int, int],
    win_size: tuple[int, int],
    # ) -> Callable[[int, int], None] | None:
) -> Callable[[], None] | None:
    """
    Return a blit function with parameters fixed except for FBO IDs.

    Source and target coordinates for the OpenGL glBlitFramebuffer
    function are calculated once and then reused each time the returned
    blit function is called. When the target coordinates change, a new
    blit function has to be created.

    Coordinates change on:
    * window resize
    * source resize
    * target rectangle resize / move

    Parameters
    ----------
    src_size : tuple[int, int]
        Source size as (width, height).
    dst_rect : tuple[int, int, int, int]
        Destination rectangle as (x, y, w, h).
    win_size : tuple[int, int]
        Window size as (width, height).

    Returns
    -------
    Callable[[int, int], None] | None
        Blit function: blit(source_FBO, dest_FBO)
        or None if destination rectangle falls outside of the window.
    """
    dst_clipped = _intersect_screen_rect(win_size, dst_rect)
    if dst_clipped is None:
        return None

    sw, sh = src_size
    dx, dy, dw, dh = dst_rect
    w_ratio = sw / float(dw)
    h_ratio = sh / float(dh)
    cx, cy, cw, ch = dst_clipped

    # Proportional source rect
    # bottom-left point: if no clip on target's bottom-left, it means it's inside the screen,
    #                    so we can start from source's (0,0)
    sx0 = 0 if cx == dx else int((cx - dx) * w_ratio)
    sy0 = 0 if cy == dy else int((cy - dy) * h_ratio)
    # top-right point: if no clip on target's top-right, then (sw,sh)
    # sx1 = sw if cx1 == dx1 else int(sw - (dx1 - cx1) * w_ratio)
    # sy1 = sh if cy1 == dy1 else int(sh - (dy1 - cy1) * h_ratio)
    sx1 = sw if cx + cw == dx + dw else sx0 + int(cw * w_ratio)
    sy1 = sh if cy + ch == dy + dh else sy0 + int(ch * h_ratio)

    # If source and dest are same size and aligned, use nearest for speed/sharpness
    filter_type = (
        gl.GL_NEAREST if ((sx1 - sx0) == cw and (sy1 - sy0) == ch) else gl.GL_LINEAR
    )

    # def blit_fn(src_fbo, draw_fbo):
    def blit_fn():
        # Bind FBOs
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

        # Draw
        gl.glBlitFramebuffer(
            sx0,
            sy0,
            sx1,
            sy1,
            cx,
            cy,
            cx + cw,
            cy + ch,
            gl.GL_COLOR_BUFFER_BIT,
            filter_type,
        )

        # Unbind FBOs
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

    return blit_fn


def test_blit(
    src_size: tuple[int, int],
    dst_rect: tuple[int, int, int, int],
    win_size: tuple[int, int],
    # src_fbo: int,
    # draw_fbo: int,
) -> None:
    """
    do blit (for testing pre-blit calculation performance)
    """
    dst_clipped = _intersect_screen_rect(win_size, dst_rect)
    if dst_clipped is None:
        return None

    sw, sh = src_size
    dx, dy, dw, dh = dst_rect
    w_ratio = sw / float(dw)
    h_ratio = sh / float(dh)
    cx, cy, cw, ch = dst_clipped

    # Proportional source rect
    # bottom-left point: if no clip on target's bottom-left, it means it's inside the screen,
    #                    so we can start from source's (0,0)
    sx0 = 0 if cx == dx else int((cx - dx) * w_ratio)
    sy0 = 0 if cy == dy else int((cy - dy) * h_ratio)
    # top-right point: if no clip on target's top-right, then (sw,sh)
    # sx1 = sw if cx1 == dx1 else int(sw - (dx1 - cx1) * w_ratio)
    # sy1 = sh if cy1 == dy1 else int(sh - (dy1 - cy1) * h_ratio)
    sx1 = sw if cx + cw == dx + dw else sx0 + int(cw * w_ratio)
    sy1 = sh if cy + ch == dy + dh else sy0 + int(ch * h_ratio)

    # If source and dest are same size and aligned, use nearest for speed/sharpness
    filter_type = (
        gl.GL_NEAREST if ((sx1 - sx0) == cw and (sy1 - sy0) == ch) else gl.GL_LINEAR
    )

    # Bind FBOs
    # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

    # Draw
    gl.glBlitFramebuffer(
        sx0,
        sy0,
        sx1,
        sy1,
        cx,
        cy,
        cx + cw,
        cy + ch,
        gl.GL_COLOR_BUFFER_BIT,
        filter_type,
    )

    # Unbind FBOs
    # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
