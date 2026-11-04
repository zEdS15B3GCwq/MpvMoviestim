# mypy: disable_error_code = import-untyped
from __future__ import annotations

import ctypes
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

import pyglet
import pyglet.window
from psychopy import logging
from pyglet import gl

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from mpv import MpvRenderContext
    from psychopy import visual
    from pyglet.window import BaseWindow

T = TypeVar("T")

# OpenGL sized internal formats, with fallback values for pyglet versions lacking them
_GL_RGB8 = int(getattr(gl, "GL_RGB8", 0x8051))
_GL_RGBA8 = int(getattr(gl, "GL_RGBA8", 0x8058))
_GL_RGB10 = int(getattr(gl, "GL_RGB10", 0x8052))
_GL_RGB10_A2 = int(getattr(gl, "GL_RGB10_A2", 0x8059))
_GL_RGB16 = int(getattr(gl, "GL_RGB16", 0x8054))
_GL_RGBA16 = int(getattr(gl, "GL_RGBA16", 0x805B))
_GL_RGB16F = int(getattr(gl, "GL_RGB16F", 0x881B))
_GL_RGBA16F = int(getattr(gl, "GL_RGBA16F", 0x881A))
_GL_RGB32F = int(getattr(gl, "GL_RGB32F", 0x8815))
_GL_RGBA32F = int(getattr(gl, "GL_RGBA32F", 0x8814))

# ARB_color_buffer_float, available in compatibility and legacy contexts
_GL_RGBA_FLOAT_MODE_ARB = 0x8820

# Pixel format -> (incoming format, incoming pixel type, name) for texture allocation
_PIXEL_FORMAT_MAP: dict[int, tuple[int, int, str]] = {
    _GL_RGBA32F: (gl.GL_RGBA, gl.GL_FLOAT, "RGBA32F"),
    _GL_RGB32F: (gl.GL_RGB, gl.GL_FLOAT, "RGB32F"),
    _GL_RGBA16F: (gl.GL_RGBA, gl.GL_HALF_FLOAT, "RGBA16F"),
    _GL_RGB16F: (gl.GL_RGB, gl.GL_HALF_FLOAT, "RGB16F"),
    _GL_RGBA8: (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, "RGBA8"),
    _GL_RGB8: (gl.GL_RGB, gl.GL_UNSIGNED_BYTE, "RGB8"),
    _GL_RGB10_A2: (gl.GL_RGBA, gl.GL_UNSIGNED_INT_2_10_10_10_REV, "RGB10_A2"),
    # GL_UNSIGNED_INT_10_10_10_2 is a 4-component packed type, invalid with GL_RGB
    _GL_RGB10: (gl.GL_RGBA, gl.GL_UNSIGNED_INT_2_10_10_10_REV, "RGB10"),
    _GL_RGBA16: (gl.GL_RGBA, gl.GL_UNSIGNED_SHORT, "RGBA16"),
    _GL_RGB16: (gl.GL_RGB, gl.GL_UNSIGNED_SHORT, "RGB16"),
}

# Screen backbuffer bit counts (red, green, blue, alpha, is_float) -> pixel format
_BITS_TO_PIXEL_FORMAT: dict[tuple[int, int, int, int, bool], int] = {
    (8, 8, 8, 8, False): _GL_RGBA8,
    (8, 8, 8, 0, False): _GL_RGB8,
    (10, 10, 10, 2, False): _GL_RGB10_A2,
    (10, 10, 10, 0, False): _GL_RGB10,
    (16, 16, 16, 16, False): _GL_RGBA16,
    (16, 16, 16, 0, False): _GL_RGB16,
    (16, 16, 16, 16, True): _GL_RGBA16F,
    (16, 16, 16, 0, True): _GL_RGB16F,
    (32, 32, 32, 32, True): _GL_RGBA32F,
    (32, 32, 32, 0, True): _GL_RGB32F,
}

# Blit target pixel format -> intermediate buffer pixel format
# * RGB formats become RGBA: RGB16F/RGB32F aren't required to be color-renderable,
#   so FBOs using them may be incomplete on some drivers.
# * 32-bit floats become RGBA16F: ~11 bits of precision near 1.0 is enough for
#   video, at half the memory and bandwidth.
_INTERMEDIATE_FORMAT_MAP: dict[int, int] = {
    _GL_RGB8: _GL_RGBA8,
    _GL_RGBA8: _GL_RGBA8,
    _GL_RGB10: _GL_RGB10_A2,
    _GL_RGB10_A2: _GL_RGB10_A2,
    _GL_RGB16: _GL_RGBA16,
    _GL_RGBA16: _GL_RGBA16,
    _GL_RGB16F: _GL_RGBA16F,
    _GL_RGBA16F: _GL_RGBA16F,
    _GL_RGB32F: _GL_RGBA16F,
    _GL_RGBA32F: _GL_RGBA16F,
}

DitherDepth: TypeAlias = int | Literal["no"]
"""Value of mpv's `dither-depth` option: bits per component, or "no"."""

# Screen backbuffer pixel format -> mpv dither depth. Dithering is needed when the
# output is quantized to 8 or 10 bits, 16-bit and float outputs need none.
_DITHER_DEPTH_MAP: dict[int, DitherDepth] = {
    _GL_RGB8: 8,
    _GL_RGBA8: 8,
    _GL_RGB10: 10,
    _GL_RGB10_A2: 10,
    _GL_RGB16: "no",
    _GL_RGBA16: "no",
    _GL_RGB16F: "no",
    _GL_RGBA16F: "no",
    _GL_RGB32F: "no",
    _GL_RGBA32F: "no",
}

# Used when a pixel format can't be determined
_DEFAULT_INTERMEDIATE_FORMAT = _GL_RGBA8
_DEFAULT_DITHER_DEPTH: DitherDepth = 8


@dataclass(frozen=True, slots=True)
class PsychoPyRenderingEnvironment:
    """PsychoPy's rendering environment: its draw target and the screen backbuffer.

    PsychoPy draws into the screen backbuffer directly, or into an FBO if
    `useFBO=True`, which is then drawn to the screen backbuffer on flip.

    Attributes
    ----------
    use_fbo : bool
        True if PsychoPy draws into an FBO (`useFBO=True`).
    target_fbo : int
        PsychoPy's draw target FBO, or 0 when it's the screen backbuffer.
    w, h : int
        Size of PsychoPy's draw target in pixels.
    target_format : int
        Pixel format of PsychoPy's FBO. 0 when the target is the screen
        backbuffer (see `screen_format`), or if it can't be determined.
    samples : int
        Number of samples of PsychoPy's draw target, 0 if not multisampled.
        Blitting into a multisampled target is not possible (GL_INVALID_OPERATION).
    screen_bits : tuple[int, int, int, int]
        Red, green, blue and alpha bit counts of the screen backbuffer.
    screen_is_float : bool
        True if the screen backbuffer has floating point components.
    screen_format : int
        Pixel format of the screen backbuffer, 0 if it can't be determined.
    """

    use_fbo: bool
    target_fbo: int
    w: int
    h: int
    target_format: int
    samples: int
    screen_bits: tuple[int, int, int, int]
    screen_is_float: bool
    screen_format: int


def _gl_get_int(pname: int) -> int:
    """Return the value of a named GL integer."""
    out = ctypes.c_int(0)
    gl.glGetIntegerv(pname, out)
    return out.value


def _gl_try_get_bool(pname: int) -> bool | None:
    """Return the value of a named GL boolean, or None if it isn't supported."""
    # clear stale errors, so the check below only sees ours
    for _ in range(16):
        if gl.glGetError() == gl.GL_NO_ERROR:
            break

    value = gl.GLboolean(0)
    try:
        gl.glGetBooleanv(pname, ctypes.byref(value))
    except gl.GLException:  # raised instead of setting the error when debug_gl is on
        return None
    if gl.glGetError() != gl.GL_NO_ERROR:
        return None
    return bool(value.value)


def _gl_name(obj: Any) -> int:
    """Return a GL object name stored either as int or as ctypes value (e.g. c_uint)."""
    return int(getattr(obj, "value", obj))


def _get_backbuffer_bits() -> tuple[int, int, int, int, bool]:
    """Return the backbuffer's RGBA bit counts and whether it's floating point.

    Notes
    -----
    * The window's OpenGL context must be current.
    * GL_*_BITS and GL_RGBA_FLOAT_MODE_ARB are legacy queries. They work in the
      compatibility and legacy contexts that pyglet 1.4/1.5 create, but not in
      core profiles.
    """
    # the queries describe the bound draw framebuffer, make sure it's the backbuffer
    saved_draw_fbo = _gl_get_int(int(gl.GL_DRAW_FRAMEBUFFER_BINDING))
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

    red = _gl_get_int(int(gl.GL_RED_BITS))
    green = _gl_get_int(int(gl.GL_GREEN_BITS))
    blue = _gl_get_int(int(gl.GL_BLUE_BITS))
    alpha = _gl_get_int(int(gl.GL_ALPHA_BITS))
    # bit counts can't distinguish e.g. RGBA16 from RGBA16F
    is_float = bool(_gl_try_get_bool(_GL_RGBA_FLOAT_MODE_ARB))

    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved_draw_fbo)
    return red, green, blue, alpha, is_float


def _get_texture_format(tex_id: int) -> int:
    """Return the internal pixel format of a 2D texture."""
    saved_texture = _gl_get_int(int(gl.GL_TEXTURE_BINDING_2D))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)

    value = ctypes.c_int(0)
    gl.glGetTexLevelParameteriv(
        gl.GL_TEXTURE_2D, 0, int(gl.GL_TEXTURE_INTERNAL_FORMAT), value
    )

    gl.glBindTexture(gl.GL_TEXTURE_2D, saved_texture)
    return value.value


def _get_framebuffer_samples(fbo: int) -> int:
    """Return the number of samples of a framebuffer, or 0 if it isn't multisampled."""
    saved_draw_fbo = _gl_get_int(int(gl.GL_DRAW_FRAMEBUFFER_BINDING))
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo)

    # GL_SAMPLE_BUFFERS and GL_SAMPLES describe the bound draw framebuffer
    samples = 0
    if _gl_get_int(int(gl.GL_SAMPLE_BUFFERS)) > 0:
        samples = max(_gl_get_int(int(gl.GL_SAMPLES)), 1)

    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved_draw_fbo)
    return samples


def pixel_format_id_to_name(format_id: int) -> str:
    f = _PIXEL_FORMAT_MAP.get(format_id, None)
    return f[2] if f is not None else ""


def get_psychopy_rendering_environment(
    win: visual.Window,
) -> PsychoPyRenderingEnvironment:
    """Return PsychoPy's rendering environment: its draw target and the screen.

    Parameters
    ----------
    win : psychopy.visual.Window
        Psychopy's window (pyglet-based).

    Returns
    -------
    PsychoPyRenderingEnvironment

    Notes
    -----
    * The window's OpenGL context is made current before querying it.
    * Typically, PsychoPy's FBO has the RGBA32F pixel format, while the screen
      backbuffer has something like RGB8.
    * Use `recommend_intermediate_pixel_format()` and `recommend_dithering_depth()`
      to derive settings from the result.
    """
    assert win.winHandle is not None
    make_gl_context_current(win.winHandle)

    # screen backbuffer: get pixel bit counts, convert to pixel format
    red, green, blue, alpha, screen_is_float = _get_backbuffer_bits()
    screen_format = _BITS_TO_PIXEL_FORMAT.get(
        (red, green, blue, alpha, screen_is_float), 0
    )

    # bpc = getattr(win, "bpc", None)
    # if bpc is not None:
    #     print(f"Psychopy window reports {bpc} bits per channel (win.bpc).")

    # PsychoPy's draw target: the screen backbuffer, or an FBO if useFBO=True
    w, h = win.frameBufferSize
    if win.useFBO:
        target_fbo = _gl_name(win.frameBuffer)
        target_format = _get_texture_format(_gl_name(win.frameTexture))
    else:
        target_fbo = 0
        target_format = 0

    # multisampling of the draw target
    samples = _get_framebuffer_samples(target_fbo)

    env = PsychoPyRenderingEnvironment(
        use_fbo=bool(win.useFBO),
        target_fbo=target_fbo,
        w=w,
        h=h,
        target_format=target_format,
        samples=samples,
        screen_bits=(red, green, blue, alpha),
        screen_is_float=screen_is_float,
        screen_format=screen_format,
    )

    # log
    screen_name = pixel_format_id_to_name(screen_format) or "<undetermined>"
    target_name = (
        (pixel_format_id_to_name(target_format) or "<undetermined>")
        if env.use_fbo
        else "<screen>"
    )
    logging.info(
        f"Psychopy's rendering target: fbo={target_fbo}, size={w}x{h}, "
        f"format={target_name}, samples={samples}; "
        f"screen backbuffer: format={screen_name}, "
        f"bits={red}/{green}/{blue}/{alpha}, float={screen_is_float}"
    )
    if samples > 0:
        logging.warning(
            f"Psychopy's rendering target is multisampled ({samples} samples), "
            "blitting into it is not possible"
        )

    return env


def recommend_intermediate_pixel_format(env: PsychoPyRenderingEnvironment) -> int:
    """Return the recommended pixel format of the intermediate buffers.

    The intermediate buffers are blitted to PsychoPy's draw target, so their
    format matches the precision of that target (see `_INTERMEDIATE_FORMAT_MAP`).
    Pass the same format to mpv as the FBO's `internal_format`.

    Parameters
    ----------
    env : PsychoPyRenderingEnvironment
        Result of `get_psychopy_rendering_environment()`.

    Returns
    -------
    int
        OpenGL pixel format.
    """
    # without an FBO, or if its format is unknown, the screen is the reference
    blit_format = env.target_format or env.screen_format
    return _INTERMEDIATE_FORMAT_MAP.get(blit_format, _DEFAULT_INTERMEDIATE_FORMAT)


def recommend_dithering_depth(env: PsychoPyRenderingEnvironment) -> DitherDepth:
    """Return the recommended value of mpv's `dither-depth` option.

    The image is finally quantized in the screen backbuffer, so dithering is
    matched to its depth (see `_DITHER_DEPTH_MAP`). With a float intermediate
    buffer, mpv's `dither-depth=auto` wouldn't dither at all.

    Parameters
    ----------
    env : PsychoPyRenderingEnvironment
        Result of `get_psychopy_rendering_environment()`.

    Returns
    -------
    DitherDepth
        Bits per component, or "no".
    """
    return _DITHER_DEPTH_MAP.get(env.screen_format, _DEFAULT_DITHER_DEPTH)


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

    def _try_lib_export(lib: ModuleType, name: str) -> int | None:
        func = getattr(lib, name, None)
        if func is not None:
            try:
                val = ctypes.cast(func, ctypes.c_void_p).value
            except (TypeError, ValueError, AttributeError):
                # try attribute fallback
                val = getattr(func, "value", None)
            if val is not None:
                return int(val)
        return None

    logging.info(
        f"Resolving OpenGL function {name_str} using pyglet backend for platform {platform}"
    )
    getprocaddress_func_names: list[str] = []

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
            f"Pyglet's platform-specific OpenGL library {pyglet_lib_name} not found"
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


def get_gl_proc_address(_ctx: MpvRenderContext, name: bytes) -> int:
    """Return GL function address, or 0 if not available or on error.

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
    except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
        logging.error(f"get_proc_address encountered an unexpected error: {e}")
        return 0


def gl_texture_create(w: int, h: int, internal_format: int = gl.GL_RGBA8) -> int:
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


def gl_texture_destroy(tex_id: int) -> None:
    """Destroy a texture.

    Parameters
    ----------
    tex_id : int
        GL texture ID
    """
    if tex_id != 0:
        t = ctypes.c_uint(tex_id)
        gl.glDeleteTextures(1, ctypes.byref(t))


def gl_fbo_create(tex_id: int) -> int:
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


def gl_fbo_destroy(fbo_id: int) -> None:
    """Destroy a framebuffer object (FBO).

    Parameters
    ----------
    fbo_id : int
        FBO ID
    """
    if fbo_id != 0:
        f = ctypes.c_uint(fbo_id)
        gl.glDeleteFramebuffers(1, ctypes.byref(f))


def gl_blit_with_draw_rect(
    draw_rect: tuple[int, int, int, int],
    src_fbo: int,
    draw_fbo: int,
    flipHoriz: bool = False,
    flipVert: bool = False,
) -> None:
    """Blit image to FBO/tex

    This routine copies image data between FBOs without scaling. The rect
    [0, 0, w, h] from the source is drawn to the rect [x, y, x+w, y+h]on the
    draw target, not considering flips. The image rect may extend over screen
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
    PsychoPy's strategy is to set the draw target FBO once and then draw all
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
    dx0, dy0 = draw_rect[0], draw_rect[1]  # destination bottom-left
    dx1, dy1 = (
        dx0 + draw_rect[2],
        dy0 + draw_rect[3],
    )  # destination top-right

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
    # Restoring the draw FBO is unnecessary as we use the same target as PsychoPy
    # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, previous_draw_FBO.value)


def create_shadow_window(main_window: BaseWindow) -> BaseWindow:
    """Create an invisible pyglet window that shares `main_window`'s OpenGL context.

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
    `create_context()` with shared context doesn't work with pyglet 1.4/1.5,
    so we create a hidden window instead, and rely on pyglet's internal context
    sharing behavior.
    """

    shadow_window = pyglet.window.Window(width=100, height=100, visible=False)

    # keep shadow window out of pyglet's global event/idle loop
    # to prevent crashing on window move
    pyglet.app.windows.remove(shadow_window)

    # shadow_window.switch_to()  # redundant, already in Window.__init__()

    # Restore the main window's context as current on this thread (shadow
    # window's __init__ made its own context current).
    main_window.switch_to()
    # main_window.activate()

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


def make_gl_context_current(window: Any) -> None:
    """Make *window*'s OpenGL context current on the calling thread.

    Alias of ``window.switch_to()``.

    Parameters
    ----------
    window : pyglet.window.BaseWindow
        Window whose context should become current.
    """
    window.switch_to()


def release_gl_context() -> None:
    """Release the current OpenGL context on the calling thread.

    Calls the appropriate platform-specific unbind function.  Safe to call
    even if no context is current (errors are swallowed).
    """
    platform = pyglet.compat_platform
    if platform in ("win32", "cygwin"):
        try:
            wgl_lib = importlib.import_module("pyglet.gl.lib_wgl")

            wgl_lib.wglMakeCurrent(None, None)  # ty: ignore[unresolved-attribute]
        except Exception as e:  # pylint: disable=broad-except  # noqa: BLE001
            logging.error(f"Error when releasing OpenGL context on {platform}: {e}")
    elif platform.startswith("linux"):
        try:
            glx_lib = importlib.import_module("pyglet.gl.lib_glx")

            ctx = pyglet.gl.current_context
            if ctx is not None:
                glx_lib.glXMakeCurrent(ctx._display, 0, None)  # pylint: disable=protected-access
        except Exception as e:  # pylint: disable=broad-except  # noqa: BLE001
            logging.error(f"Error when releasing OpenGL context on {platform}: {e}")
    # macOS: NSOpenGLContext.clearCurrentContext() — not required for our use-case


def windows_get_screen_dpi() -> tuple[int, int] | None:
    """Get active screen's DPI on Windows."""
    if pyglet.compat_platform == "win32":
        # get active screen's dpi
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
            return dpi_x, dpi_y
        except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
            logging.error(f"Error when getting active screen DPI: {e}")
    return None


def windows_set_process_dpi_awareness(level: bool | int = True):
    """Set Windows process DPI awareness level.

    Parameters
    ----------
    level : bool or int
        If True, set to "system" awareness.
        If False, set to "unaware".
        If int, use the corresponding DPI_AWARENESS_CONTEXT value:
        -1: DPI_AWARENESS_CONTEXT_UNAWARE
        -2: DPI_AWARENESS_CONTEXT_SYSTEM_AWARE
        -3: DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE
        -4: DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    """
    if pyglet.compat_platform != "win32":
        logging.error("Error: this function is only supported on Windows.")
    if isinstance(level, bool):
        level = -2 if level else -1
    elif not isinstance(level, int):
        raise TypeError("level must be a bool or int")
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_int64(level))
        logging.info(f"Windows process DPI awareness context set to {level}")
    except Exception as e:  # pylint: disable=W0718  #  noqa: BLE001
        logging.error(f"Error when setting process DPI awareness context: {e}")


def find_parent_properties(
    cls: type[T], prop_names: list[str]
) -> dict[str, Callable[[T, Any], None]]:
    """Find property setters in parent classes of a given class.

    Parameters
    ----------
    cls : type
        The class to inspect.
    prop_names : list[str]
        List of property names to look for in parent classes.

    Returns
    -------
    dict[str, Callable[[T, Any], None]]
        A dictionary mapping property names to their corresponding property
        object setters (e.g. size.setter(self, value)) found in parent classes.
        Properties without setters are ignored.
    """
    parent_prop_setters: dict[str, Callable[[T, Any], None]] = {}
    mro = cls.__mro__
    start_index = mro.index(cls)
    for propname in prop_names:
        for parent_cls in mro[start_index + 1 :]:
            prop = vars(parent_cls).get(propname)
            if prop is not None and getattr(prop, "fset", None) is not None:
                parent_prop_setters[propname] = prop.fset
                break  # Stop searching after finding the first setter
        else:
            logging.warning(
                f"Property '{propname}' setter not found in parent classes"
                f" of '{cls.__name__}'"
            )
    return parent_prop_setters
