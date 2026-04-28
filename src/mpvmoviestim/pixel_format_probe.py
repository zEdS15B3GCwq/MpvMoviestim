from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from pyglet import gl

if TYPE_CHECKING:
    from psychopy import visual


_INTERNAL_FORMAT_MAP: dict[int, str] = {
    int(getattr(gl, "GL_RGBA32F", 0x8814)): "rgba32f",
    int(getattr(gl, "GL_RGB32F", 0x8815)): "rgb32f",
    int(getattr(gl, "GL_RGBA16F", 0x881A)): "rgba16f",
    int(getattr(gl, "GL_RGB16F", 0x881B)): "rgb16f",
    int(getattr(gl, "GL_RGBA8", 0x8058)): "rgba8",
    int(getattr(gl, "GL_RGB8", 0x8051)): "rgb8",
    int(getattr(gl, "GL_RGB10_A2", 0x8059)): "rgb10_a2",
    int(getattr(gl, "GL_RGB10", 0x8052)): "rgb10",
    int(getattr(gl, "GL_RGBA16", 0x805B)): "rgba16",
    int(getattr(gl, "GL_RGB16", 0x8054)): "rgb16",
}


def _infer_backbuffer_format(red: int, green: int, blue: int, alpha: int) -> int:
    key = (red, green, blue, alpha)
    if key == (8, 8, 8, 8):
        return int(getattr(gl, "GL_RGBA8", 0x8058))
    if key == (8, 8, 8, 0):
        return int(getattr(gl, "GL_RGB8", 0x8051))
    if key == (10, 10, 10, 2):
        return int(getattr(gl, "GL_RGB10_A2", 0x8059))
    if key == (10, 10, 10, 0):
        return int(getattr(gl, "GL_RGB10", 0x8052))
    if key == (16, 16, 16, 16):
        return int(getattr(gl, "GL_RGBA16", 0x805B))
    if key == (16, 16, 16, 0):
        return int(getattr(gl, "GL_RGB16", 0x8054))
    return 0


def get_psychopy_target_pixel_format(
    win: visual.Window,
) -> tuple[dict[str, int], str]:
    """Return (pixel_format, target_fbo_id) for a PsychoPy/Pyglet window.

    Parameters
    ----------
    win: psychopy.visual.Window
        Psychopy's window (pyglet-based).

    Returns
    -------
    dict[str, Any] (with elements corresponding to mpv.MpvOpenGLFBO)
        fbo: int
            Psychopy's render FBO, or 0 when the window backbuffer is the target.
        w: int
            Width of the target buffer
        h: int
            Height of the target buffer
        internal_format: int
            OpenGL number of the best matching format.
            This field is only present if a best format can be determined.
    str
        Best-match format string (for example ``rgba32f``), or empty string
        if undetermined.

    Notes
    -----
    The OpenGL context of the provided window is assumed to be current.
    """

    def _get_gl_int(pname: int) -> int:
        """Return the value of a named GL integer."""
        out = ctypes.c_int(0)
        gl.glGetIntegerv(pname, out)
        return int(out.value)

    target_fbo: int = 0
    format_name: str = ""

    if win.useFBO:
        tex_id: int = win.frameTexture

        # bind the window's texture
        saved_texture = _get_gl_int(int(gl.GL_TEXTURE_BINDING_2D))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)

        internal_format_value = ctypes.c_int(0)
        gl.glGetTexLevelParameteriv(
            gl.GL_TEXTURE_2D,
            0,
            int(gl.GL_TEXTURE_INTERNAL_FORMAT),
            internal_format_value,
        )

        # restore the previous texture binding
        gl.glBindTexture(gl.GL_TEXTURE_2D, saved_texture)

        internal_fmt = int(internal_format_value.value)
        format_name = _INTERNAL_FORMAT_MAP.get(internal_fmt, "")

        # if format_name != "rgba32f":
        #     logging.info(
        #         "PsychoPy intermediate FBO texture format is "
        #         f"{format_name if format_name else str(hex(internal_fmt))} "
        #         "(expected rgba32f)."
        #     )

        target_fbo = win.frameBuffer.value

    else:
        red_bits = _get_gl_int(int(gl.GL_RED_BITS))
        green_bits = _get_gl_int(int(gl.GL_GREEN_BITS))
        blue_bits = _get_gl_int(int(gl.GL_BLUE_BITS))
        alpha_bits = _get_gl_int(int(gl.GL_ALPHA_BITS))

        bpc = getattr(win, "bpc", None)
        if bpc is not None:
            print(f"Psychopy window reports {bpc} bits per channel (win.bpc).")

        internal_fmt = _infer_backbuffer_format(
            red_bits, green_bits, blue_bits, alpha_bits
        )
        format_name = _INTERNAL_FORMAT_MAP.get(internal_fmt, "")
        # logging.info(
        #     "Psychopy window's backbuffer channel bits are "
        #     f"({red_bits}, {green_bits}, {blue_bits}, {alpha_bits}); "
        #     f"chosen format: {format_name if format_name else '<default>'}."
        # )

    w, h = win.frameBufferSize
    fbo_info: dict[str, int] = {
        "w": w,
        "h": h,
        "fbo": target_fbo,
    }
    if internal_fmt != 0:
        fbo_info["internal_format"] = internal_fmt

    return fbo_info, format_name
