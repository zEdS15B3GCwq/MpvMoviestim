# mypy: disable_error_code = import-untyped
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

import pyglet
import pyglet.window
from psychopy import logging
from pyglet import gl

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

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
# * Half floats are allocated with GL_FLOAT: no data is uploaded, and GL_HALF_FLOAT
#   as incoming type needs ARB_half_float_pixel, which legacy contexts may lack.
_PIXEL_FORMAT_MAP: dict[int, tuple[int, int, str]] = {
    _GL_RGBA32F: (gl.GL_RGBA, gl.GL_FLOAT, "RGBA32F"),
    _GL_RGB32F: (gl.GL_RGB, gl.GL_FLOAT, "RGB32F"),
    _GL_RGBA16F: (gl.GL_RGBA, gl.GL_FLOAT, "RGBA16F"),
    _GL_RGB16F: (gl.GL_RGB, gl.GL_FLOAT, "RGB16F"),
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

# Failure values that some Windows drivers return from wglGetProcAddress instead of NULL
_INVALID_PROC_ADDRESSES = frozenset((1, 2, 3, ctypes.c_void_p(-1).value))

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


@functools.cache
def _gl_proc_lookup() -> tuple[Any, tuple[Callable[[bytes], int | None], ...]]:
    """Return the platform's GL library and its get_proc_address functions.

    Uses pyglet's internal OS-specific `pyglet.gl.lib_*` modules. The
    get_proc_address functions are private ctypes prototypes, so pyglet's own
    function objects (used for its lazy extension loading) are left untouched.
    The result is cached, the lookup runs once per process.

    Returns
    -------
    tuple
        GL library (ctypes library, or None if unavailable), and the
        get_proc_address functions (possibly empty).
    """
    platform = pyglet.compat_platform

    # Windows
    if platform in ("win32", "cygwin"):
        pyglet_lib_name = "pyglet.gl.lib_wgl"
        getprocaddress_func_names: tuple[str, ...] = ("wglGetProcAddress",)
        functype = ctypes.WINFUNCTYPE  # ty: ignore[unresolved-attribute]

    # Linux
    elif platform.startswith("linux"):
        pyglet_lib_name = "pyglet.gl.lib_glx"
        getprocaddress_func_names = ("glXGetProcAddressARB", "glXGetProcAddress")
        functype = ctypes.CFUNCTYPE

    # macOS: all GL functions are exported by the OpenGL framework
    elif platform == "darwin":
        pyglet_lib_name = "pyglet.gl.lib_agl"
        getprocaddress_func_names = ()
        functype = ctypes.CFUNCTYPE

    else:
        logging.error(
            "Unsupported platform. Expected Windows (win32), Linux (linux), "
            f"macOS (darwin), got: {platform}"
        )
        return None, ()

    # import platform-specific pyglet lib
    try:
        pyglet_lib = importlib.import_module(pyglet_lib_name)
    except ImportError:
        logging.error(
            f"Pyglet's platform-specific OpenGL library {pyglet_lib_name} not found"
        )
        return None, ()

    gl_lib = getattr(pyglet_lib, "gl_lib", None)
    if gl_lib is None:
        logging.error(f"{pyglet_lib_name} has no OpenGL library")
        return None, ()

    getters: list[Callable[[bytes], int | None]] = []
    prototype = functype(ctypes.c_void_p, ctypes.c_char_p)
    for func_name in getprocaddress_func_names:
        try:
            getters.append(prototype((func_name, gl_lib)))
        except (AttributeError, TypeError):
            continue
    return gl_lib, tuple(getters)


def _resolve_gl_proc_with_pyglet(name: bytes) -> int:
    """Resolve GL function name to pointer using pyglet backends.

    Parameters
    ----------
    name : bytes
        GL function name

    Returns
    -------
    int
        proc address or 0 if not available or on error

    Notes
    -----
    mpv resolves hundreds of functions and probes optional extensions, so
    lookups and misses are only logged at debug level.
    """
    gl_lib, getters = _gl_proc_lookup()
    if gl_lib is None:
        return 0

    # plan A: function exported by the GL library (on Windows, only GL 1.1 is).
    # Indexing creates a new function object, unlike getattr, which caches it
    # on the shared library object.
    name_str = name.decode("ascii", errors="ignore")
    try:
        addr = ctypes.cast(gl_lib[name_str], ctypes.c_void_p).value
    except (AttributeError, TypeError, ValueError):
        addr = None
    if addr:
        return int(addr)

    # plan B: call get_proc_address func(s)
    for getter in getters:
        try:
            addr = getter(name)
        except OSError:
            continue
        if addr and addr not in _INVALID_PROC_ADDRESSES:
            return int(addr)

    logging.debug(f"Proc address for OpenGL function {name_str} not found")
    return 0


def get_gl_proc_address(_ctx: Any, name: bytes) -> int:
    """Return GL function address, or 0 if not available or on error.

    Parameters
    ----------
    _ctx : Any
        Opaque `get_proc_address_ctx` pointer passed through by mpv (unused).
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


def gl_texture_create(w: int, h: int, internal_format: int = _GL_RGBA8) -> int:
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

    Raises
    ------
    ValueError
        If `internal_format` is not in `_PIXEL_FORMAT_MAP`.

    Notes
    -----
    * Incoming pixel format and type are looked up in `_PIXEL_FORMAT_MAP`.
      No data is uploaded, they only need to be valid for `internal_format`.
    * The previous texture binding is restored.
    * Internal floating point formats may not be supported on some mobile
      or legacy platforms.
    """
    # lookup incoming format values in mapping; fail with error if unknown
    incoming_format, incoming_pixel_type, label = _PIXEL_FORMAT_MAP.get(
        internal_format, (0, 0, "")
    )
    if label == "":
        raise ValueError(f"Unknown incoming format {hex(internal_format)}")

    logging.debug(f"Allocating GL texture with format {label}")

    saved_texture = _gl_get_int(int(gl.GL_TEXTURE_BINDING_2D))

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

    # Restore previous binding
    gl.glBindTexture(gl.GL_TEXTURE_2D, saved_texture)

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

    Raises
    ------
    RuntimeError
        If the FBO is incomplete. The FBO is deleted in that case.

    Notes
    -----
    The previous read and draw framebuffer bindings are restored, so this is
    safe to call in PsychoPy's context, also with `useFBO=True`.
    """
    saved_read_fbo = _gl_get_int(int(gl.GL_READ_FRAMEBUFFER_BINDING))
    saved_draw_fbo = _gl_get_int(int(gl.GL_DRAW_FRAMEBUFFER_BINDING))

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

    # Restore previous bindings
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, saved_read_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved_draw_fbo)

    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        gl_fbo_destroy(fbo_id)
        raise RuntimeError(f"Intermediate FBO incomplete (status={hex(status)})")

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
    screen elements to it without setting the FBO again. Here, both read and
    draw FBOs are bound explicitly, so the blit doesn't depend on whatever is
    currently bound, but only the read FBO is saved and restored. Our draw
    target is the same as the one PsychoPy uses, so it can be left bound.
    """

    # Save previously bound read FBO
    previous_read_fbo = _gl_get_int(int(gl.GL_READ_FRAMEBUFFER_BINDING))
    previous_draw_fbo = _gl_get_int(int(gl.GL_DRAW_FRAMEBUFFER_BINDING))

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

    # Restore read FBO
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous_read_fbo)
    # Restoring the draw FBO is unnecessary as we use the same target as PsychoPy
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, previous_draw_fbo)


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
    * `create_context()` with shared context doesn't work with pyglet 1.4/1.5,
      so we create a hidden window instead, and rely on pyglet's internal context
      sharing behavior: the new context shares with `gl.current_context`.
      Hence `main_window`'s context is made current first.
    * `main_window`'s config is reused, as some platforms (macOS in particular)
      need compatible pixel formats for sharing. If that fails, pyglet's default
      config is used.
    * On Linux, call `linux_init_xlib_threads()` before any window is created
      if the shadow context is used from another thread.
    """
    # pyglet shares the new context with whichever context is current
    main_window.switch_to()

    try:
        shadow_window = pyglet.window.Window(
            width=100, height=100, visible=False, config=main_window.config
        )
    except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
        logging.warning(
            f"Creating shadow window with the main window's config failed ({e}), "
            "retrying with the default config"
        )
        main_window.switch_to()
        shadow_window = pyglet.window.Window(width=100, height=100, visible=False)

    # keep shadow window out of pyglet's global event/idle loop
    # to prevent crashing on window move
    pyglet.app.windows.remove(shadow_window)

    # Restore the main window's context as current on this thread (shadow
    # window's __init__ made its own context current). switch_to() also sets
    # gl.current_context.
    main_window.switch_to()

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


def make_gl_context_current(window: BaseWindow) -> None:
    """Make *window*'s OpenGL context current on the calling thread.

    Alias of ``window.switch_to()``.

    Parameters
    ----------
    window : pyglet.window.BaseWindow
        Window whose context should become current.

    Notes
    -----
    pyglet's `gl.current_context` is a single value for the whole process, not
    per thread ("XXX not per-thread" in pyglet). Calling this from a worker
    thread therefore also changes the main thread's pyglet bookkeeping. This is
    harmless as long as nothing on the worker thread relies on pyglet-level
    state tied to the current context (e.g. pyglet's GL error checks with
    `debug_gl=True`, which PsychoPy turns off).
    """
    window.switch_to()


def release_gl_context() -> None:
    """Release the current OpenGL context on the calling thread.

    Calls the appropriate platform-specific unbind function. Safe to call
    even if no context is current (errors are logged, not raised).

    Notes
    -----
    * On Windows, a context still current in another thread can't be deleted,
      so a worker thread must release the shadow context before the shadow
      window is closed on the main thread.
    * pyglet's `Context.detach()` isn't used: on Windows it deletes the context.
    * `gl.current_context` is left untouched, as it's process-wide (see
      `make_gl_context_current()`).
    """
    platform = pyglet.compat_platform
    try:
        if platform in ("win32", "cygwin"):
            wgl = importlib.import_module("pyglet.gl.wgl")
            if not wgl.wglMakeCurrent(None, None):
                logging.warning("wglMakeCurrent failed to release OpenGL context")
        elif platform.startswith("linux"):
            glx = importlib.import_module("pyglet.gl.glx")
            # display of this thread's current context, NULL if none is current
            x_display = glx.glXGetCurrentDisplay()
            if x_display:
                glx.glXMakeCurrent(x_display, 0, None)
        elif platform == "darwin":
            cocoapy = importlib.import_module("pyglet.libs.darwin.cocoapy")
            cocoapy.ObjCClass("NSOpenGLContext").clearCurrentContext()
    except Exception as e:  # pylint: disable=broad-except  # noqa: BLE001
        logging.error(f"Error when releasing OpenGL context on {platform}: {e}")


def linux_init_xlib_threads() -> None:
    """Initialize Xlib for multithreaded use on Linux (`XInitThreads()`).

    pyglet never calls `XInitThreads()`, but making the shadow context current
    in a worker thread uses the same X display connection as the main thread.
    Without this call, Xlib isn't thread-safe and may crash at random.

    Notes
    -----
    Must be called before PsychoPy or pyglet open the X display, i.e. before any
    window is created. Does nothing on other platforms.
    """
    if not pyglet.compat_platform.startswith("linux"):
        return
    try:
        libx11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        if not libx11.XInitThreads():
            logging.error("XInitThreads failed")
    except OSError as e:
        logging.error(f"Error when initializing Xlib threads: {e}")


def windows_get_screen_dpi(window: BaseWindow | None = None) -> tuple[int, int] | None:
    """Get the screen DPI on Windows.

    Parameters
    ----------
    window : pyglet.window.BaseWindow, optional
        If given, return the DPI of the monitor this window is on (e.g. pass
        PsychoPy's ``win.winHandle``). Otherwise, return the system (primary
        monitor) DPI.

    Returns
    -------
    tuple[int, int] or None
        Horizontal and vertical DPI, or None on other platforms or on error.

    Notes
    -----
    If the process isn't DPI-aware, Windows virtualizes the DPI and this always
    returns 96 (see `windows_set_process_dpi_awareness()`).
    """
    if pyglet.compat_platform != "win32":
        return None
    try:
        # private library instances, so setting argtypes doesn't affect ctypes.windll
        user32 = ctypes.WinDLL("user32")
        hwnd = getattr(window, "_hwnd", None) if window is not None else None
        if hwnd:
            # Windows 10 1607+
            user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]
            user32.GetDpiForWindow.restype = ctypes.c_uint
            dpi = user32.GetDpiForWindow(hwnd)
            if dpi:
                return dpi, dpi
            logging.warning("GetDpiForWindow failed, returning system DPI")

        gdi32 = ctypes.WinDLL("gdi32")
        user32.GetDC.argtypes = [ctypes.c_void_p]
        user32.GetDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi32.GetDeviceCaps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        hdc = user32.GetDC(None)
        try:
            dpi_x = gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            dpi_y = gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
        finally:
            user32.ReleaseDC(None, hdc)
        return dpi_x, dpi_y
    except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
        logging.error(f"Error when getting screen DPI: {e}")
    return None


def windows_set_process_dpi_awareness(level: bool | int = True) -> None:
    """Set Windows process DPI awareness level.

    Must be called before any window is created. Fails if the awareness was
    already set, e.g. by the application manifest or an earlier call.

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
        return
    if isinstance(level, bool):
        level = -2 if level else -1
    elif not isinstance(level, int):
        raise TypeError("level must be a bool or int")
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # DPI_AWARENESS_CONTEXT is a pointer-sized handle
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_ssize_t]
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_int
        if user32.SetProcessDpiAwarenessContext(level):
            logging.info(f"Windows process DPI awareness context set to {level}")
        else:
            err = ctypes.get_last_error()
            logging.error(
                f"Setting process DPI awareness context to {level} failed: "
                f"{ctypes.FormatError(err)} ({err})"
            )
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

    Notes
    -----
    Like normal attribute lookup, the search stops at the nearest parent that
    defines the name. If that definition has no setter, the name is left out
    (with a warning), rather than returning a deeper parent's setter, which
    would bypass the override.
    """
    parent_prop_setters: dict[str, Callable[[T, Any], None]] = {}
    for propname in prop_names:
        for parent_cls in cls.__mro__[1:]:
            prop = vars(parent_cls).get(propname)
            if prop is None:
                continue
            fset = getattr(prop, "fset", None)
            if fset is not None:
                parent_prop_setters[propname] = fset
            else:
                logging.warning(
                    f"'{propname}' in '{parent_cls.__name__}' (parent class of "
                    f"'{cls.__name__}') has no property setter"
                )
            break  # Stop at the nearest definition
        else:
            logging.warning(
                f"Property '{propname}' setter not found in parent classes"
                f" of '{cls.__name__}'"
            )
    return parent_prop_setters
