# mypy: disable_error_code = import-untyped
from __future__ import annotations

import ctypes
import importlib
from typing import TYPE_CHECKING

import pyglet
import pyglet.window
from psychopy import logging
from pyglet import gl

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from mpv import MpvRenderContext


_PIXEL_FORMAT_MAP: dict[int, tuple[int, int, str]] = {
    int(getattr(gl, "GL_RGBA32F", 0x8814)): (gl.GL_RGBA, gl.GL_FLOAT, "RGBA32F"),
    int(getattr(gl, "GL_RGB32F", 0x8815)): (gl.GL_RGB, gl.GL_FLOAT, "RGB32F"),
    int(getattr(gl, "GL_RGBA16F", 0x881A)): (gl.GL_RGBA, gl.GL_HALF_FLOAT, "RGBA16F"),
    int(getattr(gl, "GL_RGB16F", 0x881B)): (gl.GL_RGB, gl.GL_HALF_FLOAT, "RGB16F"),
    int(getattr(gl, "GL_RGBA8", 0x8058)): (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, "RGBA8"),
    int(getattr(gl, "GL_RGB8", 0x8051)): (gl.GL_RGB, gl.GL_UNSIGNED_BYTE, "RGB8"),
    int(getattr(gl, "GL_RGB10_A2", 0x8059)): (
        gl.GL_RGBA,
        gl.GL_UNSIGNED_INT_2_10_10_10_REV,
        "RGB10_A2",
    ),
    int(getattr(gl, "GL_RGB10", 0x8052)): (
        gl.GL_RGB,
        gl.GL_UNSIGNED_INT_10_10_10_2,
        "RGB10",
    ),
    int(getattr(gl, "GL_RGBA16", 0x805B)): (gl.GL_RGBA, gl.GL_UNSIGNED_SHORT, "RGBA16"),
    int(getattr(gl, "GL_RGB16", 0x8054)): (gl.GL_RGB, gl.GL_UNSIGNED_SHORT, "RGB16"),
}


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


def get_proc_address(_ctx: MpvRenderContext, name: bytes) -> int:
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


def create_texture(w: int, h: int, internal_format: int = gl.GL_RGBA8) -> int:
    """Create a texture.

    Parameters
    ----------
    w, h : int
        width and height of texture in pixels

    internal_format : int
        texture pixel format, e.g. GL_RGBA8, GL_RGBA16, etc.

    Returns
    -------
    int
        texture ID

    Notes
    -----
    * Incoming pixel format is always expected to be GL_RGBA
    * Incoming pixel type is determined from the preferred format:
      GL_FLOAT for float types and GL_UNSIGNED BYTE for unsigned byte types.
    * Internal floating point formats may not be supported on some mobile
      or legacy platforms.
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
    # incoming_format = getattr(gl, "GL_RGBA", gl.GL_RGBA)

    # # Pick internal format according to requested format with fallback to GL_RGBA
    # # This code is somewhat defensive in not assuming that the requested formats exist
    # internal_format = gl.GL_RGBA8
    # incoming_pixel_type = gl.GL_UNSIGNED_BYTE
    # label = "GL_RGBA8"
    # incoming_is_float = False
    # label = ""
    # if (
    #     format.lower() == "rgba32f"
    #     and (t := getattr(gl, "GL_RGBA32F", None)) is not None
    # ):
    #     internal_format = t
    #     incoming_pixel_type = gl.GL_FLOAT
    #     label = "GL_RGBA32F"
    #     incoming_is_float = True
    # elif (
    #     format.lower() == "rgba16f"
    #     and (t := getattr(gl, "GL_RGBA16F", None)) is not None
    # ):
    #     internal_format = t
    #     incoming_pixel_type = gl.GL_FLOAT
    #     label = "GL_RGBA16F"
    #     incoming_is_float = True

    # lookup incoming format values in mapping; fail with error if unknown
    incoming_format, incoming_pixel_type, label = _PIXEL_FORMAT_MAP.get(
        internal_format, (0, 0, "")
    )
    if label == "":
        raise ValueError(f"Unknown incoming format {hex(internal_format)}")

    logging.debug(f"Allocating GL texture with format {label}")

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

    return tex_id


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


def blit_with_draw_rect(
    draw_rect: tuple[int, int, int, int],
    src_fbo: int,
    draw_fbo: int,
    flipHoriz: bool = False,
    flipVert: bool = False,
) -> None:
    """Blit image to FBO/tex

    This routine is intended to cached, ready-to-draw FBO/tex.The image must
    already be scaled to the target size. Image rect may extend over screen
    borders - those parts are automatically trimmed by the blit.

    Parameters
    ----------
    draw_rect : tuple[int, int, int, int]
        Draw rectangle of image data with fields: x, y, w, h.
    src_fbo: int
        Source FBO identifier.
    draw_fbo: int
        Destination FBO identifier.
    flipHoriz: bool
        Flip image horizontally.
    flipVert: bool
        Flip image vertically.

    Notes
    -----
    * This function draws the rect [0, 0, w, h] from the source to the rect
      [x, y, x+w, y+h] on the draw target, not considering flips.
    * PsychoPy's strategy is to set the draw target FBO once and then draw all
      screen elements to it without setting the FBO again. Here, we initially
      set both read and draw FBOs to be safe that the correct ones are active,
      since libMPV may have changed them, but only save and restore the state
      of the read FBO. Our target FBO is the same as the one Psychopy uses,
      so it can be left as is.
    """

    # Save previously bound FBOs
    previous_read_FBO = ctypes.c_int()
    gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING, ctypes.byref(previous_read_FBO))
    # previous_read_FBO = gl.glGetInteger(gl.GL_READ_FRAMEBUFFER_BINDING)
    # previous_draw_FBO = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)

    # Bind FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

    sx0, sy0 = 0, 0  # source bottom-left
    sx1, sy1 = draw_rect[2:4]  # source top-right
    if flipHoriz:
        sx0, sx1 = sx1, sx0
    if flipVert:
        sy0, sy1 = sy1, sy0
    dx0, dx1 = draw_rect[0], draw_rect[0] + draw_rect[2]  # destination bottom-left
    dy0, dy1 = draw_rect[1], draw_rect[1] + draw_rect[3]  # destination top-right

    # Draw
    gl.glBlitFramebuffer(
        sx0,
        sy0,
        sx1,
        sy1,
        dx0,
        dy0,
        dx1,
        dy1,
        gl.GL_COLOR_BUFFER_BIT,  # copy colour data
        gl.GL_NEAREST,  # no resize necessary
    )

    # Restore FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous_read_FBO.value)
    # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous_read_FBO)
    # Restoring the draw FBO is unnecessary as we use the same target as PsychoPy
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, previous_draw_FBO)


# TODO: simple blit without resizing, assuming same-size render, let gl take care of clips
def test_blit(
    src_size: tuple[int, int],
    dst_rect: tuple[int, int, int, int],
    win_size: tuple[int, int],
    src_fbo: int,
    draw_fbo: int,
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

    # FBO handling
    # PsychoPy's strategy is to set the FBOs once, and use textured quad draws
    # to it without setting them again.

    # Save previously bound FBOs
    previous_read_FBO = gl.glGetInteger(gl.GL_READ_FRAMEBUFFER_BINDING)
    # Saving the draw FBO is unnecessary as we don't restore it
    previous_draw_FBO = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)

    # Bind FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

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

    # Restore FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous_read_FBO)
    # Restoring the draw FBO is unnecessary as we use the same target as PsychoPy
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, previous_draw_FBO)


def test_blit_clip_no_resize(
    src_size: tuple[int, int],
    dst_rect: tuple[int, int, int, int],
    win_size: tuple[int, int],
    src_fbo: int,
    draw_fbo: int,
) -> None:
    """
    do blit (without any resizing or clip calculation)

    OpenGL may be able to take care of clipping
    Resizing should be unnecessary if FBO allocation is changed in a way
    so that the largest possible size of texture is allocated and then
    only a portion of it is used for drawing if the element is resized smaller.
    """
    dst_clipped = _intersect_screen_rect(win_size, dst_rect)
    if dst_clipped is None:
        return None

    sw, sh = src_size
    dx, dy, dw, dh = dst_rect

    # source and dest are supposed to be the same size, use nearest for speed/sharpness
    filter_type = gl.GL_NEAREST

    # FBO handling
    # PsychoPy's strategy is to set the FBOs once, and use textured quad draws
    # to it without setting them again.

    # Save previously bound FBOs
    previous_read_FBO = gl.glGetInteger(gl.GL_READ_FRAMEBUFFER_BINDING)
    # Saving the draw FBO is unnecessary as we don't restore it
    # previous_draw_FBO = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)

    # Bind FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

    # Draw
    gl.glBlitFramebuffer(
        0,
        0,
        sw,
        sh,
        dx,
        dy,
        dx + dw,
        dy + dh,
        gl.GL_COLOR_BUFFER_BIT,
        filter_type,
    )

    # Restore FBOs
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous_read_FBO)
    # Restoring the draw FBO is unnecessary as we use the same target as PsychoPy
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, previous_draw_FBO)


def create_shadow_window(main_window: Any) -> Any:
    """Create an invisible pyglet window that shares *main_window*'s OpenGL context.

    Parameters
    ----------
    main_window : pyglet.window.BaseWindow
        The primary pyglet window whose context the shadow window will share.
        When called from PsychoPy code, pass ``psychopy_win.winHandle``.

    Returns
    -------
    pyglet.window.BaseWindow
        The hidden shadow window. Keep a reference to prevent it being garbage collected.

    Notes
    -----
    ``create_context()`` with shared context doesn't work with pyglet 1.4/1.5,
    so we create a hidden window instead, and rely on pyglet's internal context
    sharing behavior.
    """

    shadow_window = pyglet.window.Window(width=100, height=100, visible=False)

    # is this necessary to hand-off context?
    # shadow_window.switch_to()  # redundant, already in Window.__init__()
    # gl.current_context = None  # redundant, has no effect?

    # Restore the main window's context as current on this thread (shadow
    # window's __init__ made its own context current).
    main_window.switch_to()
    main_window.activate()
    # gl.current_context = main_window.context  # already set by switch_to()

    return shadow_window

    # This below doesn't work with pyglet 1.4/1.5 - create_context() fails
    # platform = pyglet.window.get_platform()
    # display = platform.get_default_display()
    # screen = display.get_default_screen()
    # template = pyglet.gl.Config()
    # config = screen.get_best_config(template)
    # shared_context = config.create_context(share=main_window.context)
    # shadow = pyglet.window.Window(
    #     width=1, height=1, visible=False, context=shared_context
    # )


def make_context_current(window: Any) -> None:
    """Make *window*'s OpenGL context current on the calling thread.

    Alias of ``window.switch_to()``.

    Parameters
    ----------
    window : pyglet.window.BaseWindow
        Window whose context should become current.
    """
    window.switch_to()


def release_context() -> None:
    """Release the current OpenGL context on the calling thread.

    Calls the appropriate platform-specific unbind function.  Safe to call
    even if no context is current (errors are swallowed).
    """
    platform = pyglet.compat_platform
    if platform in ("win32", "cygwin"):
        try:
            from pyglet.gl import wgl  # pylint: disable=import-outside-toplevel

            wgl.wglMakeCurrent(None, None)
        except Exception:  # pylint: disable=broad-except
            pass
    elif platform.startswith("linux"):
        try:
            from pyglet.gl import glx  # pylint: disable=import-outside-toplevel

            ctx = pyglet.gl.current_context
            if ctx is not None:
                glx.glXMakeCurrent(ctx._display, 0, None)
        except Exception:  # pylint: disable=broad-except
            pass
    # macOS: NSOpenGLContext.clearCurrentContext() — not required for our use-case


def windows_set_scaling_aware() -> None:
    """Tell Windows we're DPI-aware to get full screen resolution."""
    if pyglet.compat_platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_int64(-4))
            print("Windows dpi-ware process context set successfully")
        except Exception as e:  # pylint: disable=W0718
            print(f"WARNING: failed to set process DPI awareness context: {e}")


def windows_get_screen_dpi() -> tuple[int, int] | None:
    """Get active screen's DPI on Windows."""
    if pyglet.compat_platform == "win32":
        # get active screen's dpi
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
            return dpi_x, dpi_y
        except Exception as e:  # pylint: disable=W0718
            print(f"WARNING: failed to get active screen DPI: {e}")
    return None
