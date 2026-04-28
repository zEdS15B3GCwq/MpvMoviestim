#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MPV video player with zero-copy rendering to PsychoPy window.

This script demonstrates using libmpv's render context API to achieve
hardware-accelerated, zero-copy video playback directly into PsychoPy's
OpenGL framebuffer. This variant resolves GL proc addresses using pyglet
internal backends and keeps an explicit wrapper for MPV.
"""

from __future__ import annotations

import ctypes
import importlib
import os
from pprint import pprint
from time import perf_counter
from typing import TYPE_CHECKING, Any

import mpv  # type: ignore
import pyglet  # type: ignore
from pyglet import gl  # type: ignore

# Import PsychoPy components
from psychopy import event, logging, visual

if TYPE_CHECKING:
    from types import ModuleType

abort_loop = False


# ------ Configuration (simple globals for experimentation) ------
# Set to a path (string) to force a particular file, or leave None to autodetect
VIDEO_PATH = "4k144.mkv"
# Window size (WxH)
WIN_SIZE = (3840, 2160)
# Target rectangle inside the window where the video should be rendered:
# (x, y, w, h) in window pixel coordinates, or None to render full window.
# Example: place at (100, 50) with size 1270x800
TARGET_RECT = (0, 0, 3840, 2160)
# Mode: 'auto' -> blit for partial, direct if fullscreen; 'blit' -> glBlitFramebuffer
MODE = "auto"

# MPV scaling options (defaults)
# --scale=<filter>
# The filter function to use when upscaling video.
#
# bilinear
# Bilinear hardware texture filtering (fastest, very low quality). This is the
# default when using the fast profile.
#
# lanczos
# Lanczos scaling. Provides good balance between quality and performance. This
# is the default for scale. The number of taps can be controlled with
# scale-radius, but is best left unchanged.
# (This filter is an alias for sinc-windowed sinc)
#
# ewa_lanczos
# Elliptic weighted average Lanczos scaling. Also known as Jinc. Relatively slow,
# but very good quality. The radius can be controlled with scale-radius.
# Increasing the radius makes the filter sharper but adds more ringing.
# (This filter is an alias for jinc-windowed jinc)
#
# ewa_lanczossharp
# A slightly sharpened version of ewa_lanczos. This is the default when using the
# high-quality profile. Blur value determined by method originally developed by
# Nicolas Robidoux for Image Magick, see:
# https://www.imagemagick.org/discourse-server/viewtopic.php?p=89068#p89068
#
# ewa_lanczos4sharpest
# Very sharp scaler, but also slightly slower than ewa_lanczossharp. Prone to
# ringing, so it's recommended to combine this with an anti-ringing shader.
# On --vo=gpu-next, setting this filter enables built-in anti-ringing, so no extra
# action needs to be taken. For more details, see:
# https://www.imagemagick.org/discourse-server/viewtopic.php?p=128587#p128587
#
# mitchell
# Mitchell-Netravali. Piecewise cubic filter with a support of radius 2.0. Provides
# a balanced compromise of all scaling artifacts. This filter has both B and C set
# to 1/3. The B and C parameters can be controlled with --scale-param1 and
# --scale-param2.
#
# hermite
# Hermite spline. Similar to bicubic but with B set to 0.0. This filter has the
# special property of having a support of radius 1.0, making it very fast in
# comparison, but prone to blocking. This is the default for --dscale.
#
# catmull_rom
# Catmull-Rom spline. Similar to mitchell, but with B and C set to 0.0 and 0.5
# respectively. This filter is sharper than mitchell, but prone to ringing.
#
# oversample
# A version of nearest neighbour that (naively) oversamples pixels, so that pixels
# overlapping edges get linearly interpolated instead of rounded. This essentially
# removes the small imperfections and judder artifacts caused by nearest-neighbour
# interpolation, in exchange for adding some blur. This can also be used for frame
# mixing, where it is commonly known as "smoothmotion" (see --tscale).
#
# linear
# A --tscale filter.
MPV_SCALE = "lanczos"
MPV_DSCALE = "hermite"

# MPV_FBO_FORMAT = "rgba32f"
MPV_FBO_FORMAT = "rgba8"

DESYNC_VIDEO = True


# Audio configuration

# --audio-device (RW)
# Set the audio device. This directly reads/writes the --audio-device option, but on
# write accesses, the audio output will be scheduled for reloading.
# Writing this property while no audio output is active will not automatically enable
# audio. (This is also true in the case when audio was disabled due to reinitialization
# failure after a previous write access to audio-device.)
# This property also doesn't tell you which audio device is actually in use.
AUDIO_DEVICE = (
    None  # Set to device name string to select specific device, or None for default
)

# --ao=null
# Produces no audio output but maintains video playback speed.
NO_AUDIO = False  # Set to True to disable audio output (sets ao to null)

# --audio-exclusive=<yes|no> (Python: True / False?)
# Enable exclusive output mode. In this mode, the system is usually locked out,
# and only mpv will be able to output audio.
# This only works for some audio outputs, such as wasapi, coreaudio, pipewire and
# audiounit. Other audio outputs silently ignore this option. They either have no
# concept of exclusive mode, or the mpv side of the implementation is missing.
AUDIO_EXCLUSIVE = False

# --volume=<value>
# Set the startup volume. 0 means silence, 100 means no volume reduction or
# amplification. Negative values can be passed for compatibility, but are treated as 0.
# --volume-max=<100.0-1000.0>
# Set the maximum amplification level in percent (default: 130). A value of 130 will
# allow you to adjust the volume up to about double the normal level.
# --volume-gain=<db>
# Set the volume gain in dB. This is applied on top of other volume and gain settings.
# --volume-gain-max=<0.0-150.0>, --volume-gain-min=<-150.0-0.0>
# Set the volume gain range in dB (default: -96 dB min, 12 dB max).
VOLUME = 100
VOLUME_GAIN = 0.0

DO_TIMING = True
t_render = []
t_flip = []
t_draw = []

frames_dropped_total = frames_dropped_early = None
frame_count = 0


def resolve_gl_proc_with_pyglet(name: bytes) -> int:
    """Resolve GL function pointer using pyglet backends (returns int or 0).

    Uses pyglet's internal `pyglet.gl.lib_*` modules
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
        raise ValueError(
            "Unsupported platform. Expected Windows (win32), Linux (linux), "
            f"macOS (darwin), got: {platform}"
        )

    try:
        pyglet_lib = importlib.import_module(pyglet_lib_name)
    except ImportError:
        logging.error(
            "Pyglet's platform-specific OpenGL library %s not found", pyglet_lib_name
        )
        return 0

    # check if exposed as attribute
    if name_str is not None:
        gl_lib = getattr(pyglet_lib, "gl_lib", None)
        if gl_lib is not None:
            addr = _try_lib_export(gl_lib, name_str)
            if addr is not None:
                return addr

    # call get_proc_address func in library
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


# MPV callback wrapper and persistent reference
def get_proc_address(_ctx, name: bytes) -> int:
    try:
        return resolve_gl_proc_with_pyglet(name)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
        logging.warning("get_proc_address unexpected error: %s", e)
        return 0


# Keep an explicit MPV wrapper alive for the lifetime of the player
# `mpv.MpvGlGetProcAddressFn` will create the proper C-callable object.
c_getproc = mpv.MpvGlGetProcAddressFn(get_proc_address)


def _intersect_rect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Intersect two rectangles.

    Parameters
    ----------
    a, b : tuple
        Rectangles as (x, y, w, h).

    Returns
    -------
    tuple or None
        Intersection as (x, y, w, h), or None if empty.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = ax if bx < ax else bx
    y0 = ay if by < ay else by
    x1 = ax + aw if bx + bw < ax + aw else bx + bw
    y1 = ay + ah if by + bh < ay + ah else by + bh
    if x1 <= x0 or y1 <= y0:
        # return (0, 0, -1, -1)
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def create_intermediate_fbo(w, h, fbo_format="rgba8"):
    """Create an intermediate texture + FBO. Returns (tex_id, fbo_id, internal_format).

    Uses float format when available for high-precision scaling.
    """
    print(">>>>>>>>>>>>>>Creating intermediate FBO...")
    # Determine preferred sized internal formats (if available)
    rgba32f = getattr(gl, "GL_RGBA32F", None)
    rgba16f = getattr(gl, "GL_RGBA16F", None)
    rgba8 = getattr(gl, "GL_RGBA8", None)

    # Pick internalformat according to requested fbo_format with sensible fallbacks
    if "32f" in fbo_format and rgba32f is not None:
        internal = rgba32f
    elif "16f" in fbo_format and rgba16f is not None:
        internal = rgba16f
    elif rgba8 is not None:
        internal = rgba8
    else:
        # last resort: generic GL_RGBA
        internal = getattr(gl, "GL_RGBA", gl.GL_RGBA)

    # Use GL_RGBA as the pixel format; choose type based on whether internal is float
    pixel_format = getattr(gl, "GL_RGBA", gl.GL_RGBA)
    float_types = (rgba32f, rgba16f)
    if internal in float_types:
        tex_type_preferred = getattr(gl, "GL_FLOAT", gl.GL_FLOAT)
    else:
        tex_type_preferred = getattr(gl, "GL_UNSIGNED_BYTE", gl.GL_UNSIGNED_BYTE)

    # Create texture as before
    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    tex_id = tex.value
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    # Allocate storage. Try preferred type first (float for float internal formats),
    # but fall back to unsigned byte if allocation/validation fails on some drivers.
    if internal is None:
        internal = gl.GL_RGBA
        tex_type_preferred = gl.GL_UNSIGNED_BYTE

    try:
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal,
            w,
            h,
            0,
            pixel_format,
            tex_type_preferred,
            None,
        )
    except Exception as e:
        # Fallback: try unsigned byte if float allocation fails
        logging.warning(
            "glTexImage2D with type %s failed (%s), retrying with GL_UNSIGNED_BYTE",
            tex_type_preferred,
            e,
        )
        try:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                internal,
                w,
                h,
                0,
                pixel_format,
                gl.GL_UNSIGNED_BYTE,
                None,
            )
        except Exception as e2:
            logging.error("glTexImage2D fallback also failed: %s", e2)
            raise

    fbo = ctypes.c_uint(0)
    gl.glGenFramebuffers(1, ctypes.byref(fbo))
    fbo_id = fbo.value
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)
    gl.glFramebufferTexture2D(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex_id, 0
    )

    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        logging.error("Intermediate FBO incomplete (status=%s)", status)

    # Unbind
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex_id, fbo_id, internal


def destroy_intermediate_fbo(tex_id, fbo_id):
    t = ctypes.c_uint(tex_id)
    f = ctypes.c_uint(fbo_id)
    try:
        gl.glDeleteTextures(1, ctypes.byref(t))
    except (AttributeError, TypeError, OSError):
        # Best-effort cleanup; ignore if GL context already gone or args invalid
        logging.debug("glDeleteTextures cleanup failed for tex=%s", tex_id)
    try:
        gl.glDeleteFramebuffers(1, ctypes.byref(f))
    except (AttributeError, TypeError, OSError):
        logging.debug("glDeleteFramebuffers cleanup failed for fbo=%s", fbo_id)


def composite_intermediate_blit(src_fbo, src_size, dst_rect, win_rect, draw_fbo):
    """Composite by blitting from `src_fbo` to PsychoPy window FBO.

    `src_size` is (w,h) of source; `dst_rect` is (x,y,w,h) desired in window.
    `win_rect` is (0,0,win_w,win_h).
    This function computes clipped dst and proportional src and calls glBlitFramebuffer.
    """
    if dst_rect is None:
        return
    dst_clipped = _intersect_rect(dst_rect, win_rect)
    if not dst_clipped:
        return
    dx, dy, _, _ = dst_rect
    cx, cy, cw, ch = dst_clipped

    # compute proportional source rect in source coordinates
    sx0 = int((cx - dx) * (src_size[0] / float(dst_rect[2])))
    sy0 = int((cy - dy) * (src_size[1] / float(dst_rect[3])))
    sx1 = int(sx0 + cw * (src_size[0] / float(dst_rect[2])))
    sy1 = int(sy0 + ch * (src_size[1] / float(dst_rect[3])))

    # bind read/draw
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)

    # Note: OpenGL origin is bottom-left; coordinates assumed consistent with window pixels
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
        gl.GL_LINEAR,
    )

    # unbind
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)


def log_info(level: int, prefix: str, text: str) -> None:
    print(f"@@@@@@@@: {level=}, {prefix=}, {text=}")
    logging.exp(f"!!!!!!!!!!!!! {text}")
    # logging.flush()


def main() -> None:
    """Main execution function."""
    global frame_count
    logging.console.setLevel(logging.EXP)

    # Select video file
    video_files = [
        "5.mkv",
        "test1_b2w_transition_jpeg.mkv",
        "test3_rgbmains_h265_main444.mkv",
    ]
    video_path = None

    for video_file in video_files:
        if os.path.exists(video_file):
            video_path = video_file
            break

    if not video_path:
        print(f"ERROR: No video file found. Looking for: {video_files}")
        return

    print(f"Using video: {video_path}")

    # Create PsychoPy window with FBO enabled
    print("Creating PsychoPy window...")
    ctypes.windll.user32.SetProcessDPIAware()
    win = visual.Window(
        size=list(WIN_SIZE),
        fullscr=True,
        useFBO=False,  # Enable FBO for offscreen rendering
        waitBlanking=False,
        units="pix",
        color=(-1, -1, -1),  # black background
    )
    # print(WIN_SIZE, win.size, win.frameBufferSize, win.getContentScaleFactor())
    # return

    # Create MPV player with hardware decoding and libmpv video output
    print("Initializing MPV player...")
    # Allow overriding video path selection by global VIDEO_PATH
    # Create MPV with explicit kwargs (avoid expanding a dict to satisfy some type checkers)

    # Prepare audio options based on configuration
    mpv_options: dict[str, Any] = {
        # "log_handler": print,
        "log_handler": log_info,
        # "loglevel": "info",
        "vo": "libmpv",
        "hwdec": "auto",
        "gpu_api": "opengl",
        "scale": MPV_SCALE,
        "dscale": MPV_DSCALE,
        "fbo_format": MPV_FBO_FORMAT,
        "pause": True,
        "keep-open": True,
    }

    if DESYNC_VIDEO:
        mpv_options["video_sync"] = "display-desync"
        global NO_AUDIO
        NO_AUDIO = True

    # Configure audio settings
    if NO_AUDIO:
        # Disable audio output by setting ao to null
        mpv_options["ao"] = "null"
        print("Audio disabled (ao=null)")
    else:
        # Configure audio device and exclusive mode if not disabled
        if AUDIO_DEVICE is not None:
            mpv_options["audio_device"] = AUDIO_DEVICE
            print(f"Audio device set to: {AUDIO_DEVICE}")
        if AUDIO_EXCLUSIVE:
            mpv_options["audio_exclusive"] = "yes"
            print("Audio exclusive mode enabled")
        if VOLUME is not None:
            mpv_options["volume"] = VOLUME
            print(f"Volume set to: {VOLUME}")
        if VOLUME_GAIN is not None:
            mpv_options["volume_gain"] = VOLUME_GAIN
            print(f"Volume gain set to: {VOLUME_GAIN}")

    player = mpv.MPV(**mpv_options)

    # Set looping
    # player.loop_playlist = "inf"  # Loop indefinitely

    # Create render context using high-level MpvRenderContext wrapper
    print("Creating MPV render context...")
    render_ctx = mpv.MpvRenderContext(
        player,
        "opengl",
        opengl_init_params={
            # Pass the explicit C-callable wrapper we created above
            "get_proc_address": c_getproc
        },
    )

    # Allow explicit VIDEO_PATH override before falling back to discovered files
    if VIDEO_PATH:
        if os.path.exists(VIDEO_PATH):
            video_path = VIDEO_PATH
        else:
            print(f"WARNING: VIDEO_PATH set but file not found: {VIDEO_PATH}")

    player.observe_property("frame-drop-count", on_drop)
    player.observe_property("eof-reached", on_eof)
    # later: player.unobserve_property('frame-drop-count', on_drop)

    # Load and play video
    print(f"Loading video: {video_path}")
    player.play(video_path)
    player.wait_until_paused()
    # logging.flush()
    # sleep(0.1)
    # player.wait_until_playing()

    # List available audio devices (both property access methods)
    print("\n=== Audio Device Information ===")
    try:
        # Method 1: Try Python property access
        audio_device_list: list[str | dict[str, str]] | None = player.audio_device_list
        if audio_device_list:
            print(
                f"Available audio devices (via property): {len(audio_device_list)} found"
            )
            for idx, device in enumerate(audio_device_list):
                if isinstance(device, dict):
                    name = device.get("name", "unknown")
                    desc = device.get("description", "no description")
                    print(f"  [{idx}] {name}: {desc}")
                else:
                    print(f"  [{idx}] {device}")
        else:
            print("No audio devices found via property access")
    except Exception as e:
        print(f"Could not access audio_device_list via property: {e}")

    # try:
    #     # Method 2: Try raw property access
    #     audio_device_list_raw = player._get_property("audio-device-list")
    #     if audio_device_list_raw:
    #         print(
    #             f"\nAvailable audio devices (via _get_property): {len(audio_device_list_raw)} found"
    #         )
    #         for idx, device in enumerate(audio_device_list_raw):
    #             if isinstance(device, dict):
    #                 name = device.get("name", "unknown")
    #                 desc = device.get("description", "no description")
    #                 print(f"  [{idx}] {name}: {desc}")
    #             else:
    #                 print(f"  [{idx}] {device}")
    #     else:
    #         print("No audio devices found via _get_property")
    # except Exception as e:
    #     print(f"Could not access audio-device-list via _get_property: {e}")

    # Log current audio configuration
    print("\n=== Current audio configuration:")
    try:
        current_ao = player.ao
        print(f"  Audio output driver (ao): {current_ao}")
    except Exception as e:
        print(f"  Could not read ao property: {e}")

    try:
        current_audio_device = player.audio_device
        print(f"  Selected audio device: {current_audio_device}")
    except Exception as e:
        print(f"  Could not read audio_device property: {e}")

    try:
        current_audio_exclusive = player.audio_exclusive
        print(f"  Audio exclusive mode: {current_audio_exclusive}")
    except Exception as e:
        print(f"  Could not read audio_exclusive property: {e}")

    print("=" * 35)

    # Log hardware decoding status
    print("\n=== Current video configuration:")
    hwdec_current = player.hwdec_current
    print(
        f"Hardware decoding: {hwdec_current if hwdec_current else 'software (no hwdec)'}"
    )

    print(
        f"gpu api: {player.gpu_api}, gpu hwdec interop: {player.gpu_hwdec_interop}, gpu sw: {player.gpu_sw}"
    )
    print(
        f"hwdec: {player.hwdec}, hwdec codecs: {player.hwdec_codecs}, hwdec current: {player.hwdec_current}, hwdec_interop: {player.hwdec_interop}, hwdec_software_fallback: {player.hwdec_software_fallback}"
    )
    print(
        f"video: {player.video}, video_bitrate: {player.video_bitrate}, video_codec: {player.video_codec}, video_format: {player.video_format}, video_scale_x/y: {player.video_scale_x}/{player.video_scale_y}"
    )

    # pprint(player.properties)

    # Get window dimensions
    win_width, win_height = win.size

    # Get FBO ID (or 0 for default framebuffer if useFBO=False)
    # Extract the integer value from GLuint ctypes object
    fbo_id = win.frameBuffer.value if win.useFBO else 0
    print(f"Rendering to FBO ID: {fbo_id}, size: {win_width}x{win_height}")

    # Create instruction text
    instructions = visual.TextStim(
        win,
        text="Press SPACE to toggle pause\nPress ESC or Q to quit",
        pos=(0, win_height / 2 - 40),
        height=20,
        color="white",
        units="pix",
    )

    # Intermediate FBO/texture state for paths B/C
    intermediate_tex = None
    intermediate_fbo = None

    # Try to read video's native size for aspect warnings (best-effort)
    vparams = getattr(player, "video_params", None)
    if vparams and isinstance(vparams, dict):
        video_w = int(vparams.get("w", 0))
        video_h = int(vparams.get("h", 0))
        video_ar = float(vparams.get("aspect", 0)) or (
            video_w / float(video_h) if video_h else 0
        )
    else:
        video_w = video_h = video_ar = 0
    print(f"Video size: {video_w}x{video_h} (ar: {round(video_ar, 3)})")
    print(
        f"Window size: {win_width}x{win_height} (ar: {round(win_width / win_height, 3)})"
    )

    # pprint(dir(player))
    print("==== video_params")
    pprint(player.video_params)

    print("==== audio_params")
    pprint(player.audio_params)

    direct_render = TARGET_RECT is None or TARGET_RECT == (
        0,
        0,
        win_width,
        win_height,
    )
    # direct_render = False
    if direct_render:
        print("<<<<< Direct Render")
    else:
        print(">>>>> Blit Render")

    # Paths B/C: render into intermediate FBO sized to target and composite
    _, _, tw, th = TARGET_RECT

    # Aspect ratio warning (best-effort)
    if video_w and video_h:
        if abs((tw / float(th)) - video_ar) > 1e-3:
            print(
                f"Target aspect ratio {tw}x{th} differs from "
                f"video ({video_ar}); video will be padded with black bars."
            )

    # Ensure intermediate FBO exists and matches size
    if not direct_render:
        intermediate_tex, intermediate_fbo, _ = create_intermediate_fbo(
            tw, th, MPV_FBO_FORMAT
        )
    else:
        intermediate_tex = intermediate_fbo = None

    player.pause = False
    # player.wait_until_playing()

    # Main render loop
    # clock = core.Clock()

    global t_render, t_flip, t_draw
    big_long_render = []
    big_long_flip = []

    try:
        while not abort_loop:
            # Check for keyboard input
            # keys = event.getKeys(["escape", "q", "space"])
            # if "escape" in keys or "q" in keys:
            #     print("\nQuitting...")
            #     break

            # if "space" in keys:
            #     # Toggle pause
            #     if player.pause:
            #         player.pause = False
            #         print("Resumed")
            #     else:
            #         player.pause = True
            #         print("Paused")
            if DO_TIMING:
                t0 = perf_counter()
                t_draw.append(t0)
            if direct_render:
                # Path A: direct full-window render
                render_ctx.render(
                    opengl_fbo={"fbo": fbo_id, "w": win_width, "h": win_height},
                    flip_y=True,
                )
                if DO_TIMING:
                    t_render.append(perf_counter() - t0)
                    if t_render[-1] > 0.02:
                        big_long_render.append(frame_count)

            else:
                # Render MPV into intermediate FBO (MPV does scaling)
                render_ctx.render(
                    opengl_fbo={"fbo": intermediate_fbo, "w": tw, "h": th},
                    flip_y=True,
                )
                if DO_TIMING:
                    t_render.append(perf_counter() - t0)
                    if t_render[-1] > 0.02:
                        big_long_render.append(frame_count)

                # Bind PsychoPy's framebuffer as the draw target
                if win.useFBO:
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

                # Composite: use blit
                win_rect = (0, 0, win_width, win_height)
                composite_intermediate_blit(
                    intermediate_fbo,
                    (tw, th),
                    TARGET_RECT,
                    win_rect,
                    win.frameBuffer if win.useFBO else 0,
                )

            # Bind PsychoPy's framebuffer as the draw target
            if win.useFBO:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

            # Draw instructions on top
            gl.glViewport(0, 0, win_width, win_height)
            instructions.draw()

            # Flip to display
            if DO_TIMING:
                t0 = perf_counter()
            win.flip()
            if DO_TIMING:
                t_flip.append(perf_counter() - t0)
                if t_flip[-1] > 0.02:
                    big_long_flip.append(frame_count)

            frame_count += 1
            # Print status every n frames
            # n = 20
            # if frame_count % n == 0:
            #     elapsed = clock.getTime()
            #     fps = frame_count / elapsed if elapsed > 0 else 0
            #     time_pos = player.time_pos if player.time_pos else 0
            #     duration = player.duration if player.duration else 0
            #     print(
            #         f"Frame: {frame_count}, FPS: {fps:.1f}, Position: {time_pos:.1f}/{duration:.1f}s"
            #     )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if DO_TIMING and len(t_render) > 0 and len(t_flip) > 0:
            t_draw.append(perf_counter())
            import numpy as np

            t_render = np.array(t_render)
            t_flip = np.array(t_flip)
            t_frames = 1000 * np.diff(t_draw)
            frame_histcount = np.histogram(
                t_frames, bins=[0, 0.1, 1, 2, 4, 6, 8, 10, 20, 100, 100000]
            )

            print(
                f"render: avg={1000 * np.sum(t_render) / len(t_render):.3f} ms, "
                f"({1000 * np.min(t_render):.3f} - {1000 * np.max(t_render):.3f} "
                f"+/-{1000 * np.std(t_render):.3f}) ms "
                f"over {len(t_render)} calls"
            )
            print(
                f"flip: avg={1000 * np.sum(t_flip) / len(t_flip):.3f} ms, "
                f"({1000 * np.min(t_flip):.3f} - {1000 * np.max(t_flip):.3f} "
                f"+/-{1000 * np.std(t_flip):.3f}) ms "
                f"over {len(t_flip)} calls"
            )
            print(
                f"big long frames at count: render {big_long_render}, flip {big_long_flip}"
            )
            print(f"frame duration distribution: {frame_histcount[0]}")
            # print(
            #     f"loop rate:  {n_flip / (t_render_sum + t_flip_sum + 1e-9):.1f} flips/s (lower bound)"
            # )
        # Cleanup
        print("\nCleaning up...")
        player.stop()

        player.unobserve_property("frame-drop-count", on_drop)
        print(
            f"total frames dropped: {frames_dropped_total} (early: {frames_dropped_early})"
        )

        # Free render context
        # Destroy intermediate FBO/texture if created
        try:
            if "intermediate_tex" in locals() and intermediate_tex is not None:
                destroy_intermediate_fbo(intermediate_tex, intermediate_fbo)
                print("Intermediate FBO/texture destroyed")
        except (RuntimeError, OSError, TypeError) as e:
            logging.debug("Error destroying intermediate FBO: %s", e)

        if render_ctx:
            render_ctx.free()
            print("MPV render context freed")

        # Stop and cleanup player
        player.terminate()
        print("MPV player terminated")

        # Close window
        win.close()
        print("Window closed")

        print("Done!")


# def cleanup(player: mpv.MPV, render_ctx: mpv.MpvRenderContext):
#     print("\nCleaning up...")
#     player.stop()

#     player.unobserve_property("frame-drop-count", on_drop)
#     print(
#         f"total frames dropped: {frames_dropped_total} (early: {frames_dropped_early})"
#     )

#     # Free render context
#     # Destroy intermediate FBO/texture if created
#     try:
#         if "intermediate_tex" in locals() and intermediate_tex is not None:
#             destroy_intermediate_fbo(intermediate_tex, intermediate_fbo)
#             print("Intermediate FBO/texture destroyed")
#     except (RuntimeError, OSError, TypeError) as e:
#         logging.debug("Error destroying intermediate FBO: %s", e)

#     if render_ctx:
#         render_ctx.free()
#         print("MPV render context freed")

#     # Stop and cleanup player
#     player.terminate()
#     print("MPV player terminated")

#     # Close window
#     win.close()
#     print("Window closed")


def on_drop(prop_name, value):
    print(f"Property {prop_name} changed -> {value}")
    global frames_dropped_total, frames_dropped_early
    if value is None:
        return
    if value == 0:
        if frames_dropped_total is not None:
            print(
                "Frame drop counts reset to 0. Previous values: "
                f"total={frames_dropped_total}, early={frames_dropped_early}"
            )
        frames_dropped_early = frames_dropped_total = 0
        return
    if frames_dropped_total is None:
        frames_dropped_total = frames_dropped_early = 0
    frames_dropped_total = value
    if frame_count < 10:
        frames_dropped_early = value


def on_eof(prop_name, value) -> None:
    print(f"Property {prop_name} changed -> {value}")
    global abort_loop
    if value:
        abort_loop = True


if __name__ == "__main__":
    main()
