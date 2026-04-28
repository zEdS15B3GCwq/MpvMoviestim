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
import sys
from pprint import pprint
from typing import TYPE_CHECKING

import pyglet
from pyglet import gl

# Import PsychoPy components
from psychopy import core, event, logging, visual

if TYPE_CHECKING:
    from types import ModuleType

# Import MPV
try:
    import mpv
except ImportError:
    print("ERROR: python-mpv not installed. Install with: pip install python-mpv")
    sys.exit(1)

# ------ Configuration (simple globals for experimentation) ------
# Set to a path (string) to force a particular file, or leave None to autodetect
VIDEO_PATH = "5.mkv"
# Window size (WxH)
WIN_SIZE = (1600, 1200)
# Target rectangle inside the window where the video should be rendered:
# (x, y, w, h) in window pixel coordinates, or None to render full window.
# Example: place at (100, 50) with size 1270x800
TARGET_RECT = (0, 0, 1600, 1200)
# Mode: 'auto' -> blit for partial, direct if fullscreen; 'blit' -> glBlitFramebuffer
MODE = "auto"

# MPV scaling options (defaults)
MPV_SCALE = "lanczos"
MPV_DSCALE = "hermite"
MPV_FBO_FORMAT = "rgba32f"

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


def _intersect_rect(a, b):
    """Intersect rects a and b. Rects are (x,y,w,h). Returns (x,y,w,h) or None if empty."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def create_intermediate_fbo(w, h, fbo_format="rgba32f"):
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


def main() -> None:
    """Main execution function."""
    global frame_count

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
    win = visual.Window(
        size=WIN_SIZE,
        fullscr=False,
        useFBO=False,  # Enable FBO for offscreen rendering
        # waitBlanking=True,
        # units="pix",
        color=(-1, -1, -1),  # black background
    )

    # Create MPV player with hardware decoding and libmpv video output
    print("Initializing MPV player...")
    # Allow overriding video path selection by global VIDEO_PATH
    # Create MPV with explicit kwargs (avoid expanding a dict to satisfy some type checkers)
    player = mpv.MPV(
        log_handler=print,
        loglevel="info",
        vo="libmpv",
        hwdec="auto",
        gpu_api="opengl",
        scale=MPV_SCALE,
        dscale=MPV_DSCALE,
        fbo_format=MPV_FBO_FORMAT,
        # d3d11va_zero_copy="yes",
        pause=True,
    )

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
    # player.wait_until_playing()

    # Log hardware decoding status
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

    # pprint(dir(player))
    pprint(player.video_params)

    direct_render = TARGET_RECT is None or TARGET_RECT == (
        0,
        0,
        win_width,
        win_height,
    )
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

    print("\nPlayback started. Controls:")
    print("  SPACE: Toggle pause")
    print("  ESC/Q: Quit")
    print("\nRendering...")

    player.pause = False
    # player.wait_until_playing()

    # Main render loop
    # clock = core.Clock()

    try:
        while True:
            # Check for keyboard input
            keys = event.getKeys(["escape", "q", "space"])
            if "escape" in keys or "q" in keys:
                print("\nQuitting...")
                break

            if "space" in keys:
                # Toggle pause
                if player.pause:
                    player.pause = False
                    print("Resumed")
                else:
                    player.pause = True
                    print("Paused")

            # Choose rendering path based on TARGET_RECT and MODE
            if direct_render:
                # Path A: direct full-window render (existing behavior)
                render_ctx.render(
                    opengl_fbo={"fbo": fbo_id, "w": win_width, "h": win_height},
                    flip_y=True,
                )
            else:
                # Render MPV into intermediate FBO (MPV does scaling)
                render_ctx.render(
                    opengl_fbo={"fbo": intermediate_fbo, "w": tw, "h": th},
                    flip_y=True,
                )

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
            win.flip()

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


if __name__ == "__main__":
    main()
