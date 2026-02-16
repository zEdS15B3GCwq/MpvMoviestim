#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MPV video player with zero-copy rendering to PsychoPy window.

This script demonstrates using libmpv's render context API to achieve
hardware-accelerated, zero-copy video playback directly into PsychoPy's
OpenGL framebuffer. This variant resolves GL proc addresses using pyglet
internal backends and keeps an explicit wrapper for MPV.
"""

import ctypes
import importlib
import os
import sys
from ctypes import c_char_p, c_void_p, cast

import pyglet
from pyglet import gl

# Import PsychoPy components
from psychopy import core, event, logging, visual

# Import MPV
try:
    import mpv
except ImportError:
    print("ERROR: python-mpv not installed. Install with: pip install python-mpv")
    sys.exit(1)


def resolve_gl_proc_with_pyglet(name):
    """Resolve GL function pointer using pyglet backends (returns int or 0).

    Uses pyglet's internal `pyglet.gl.lib_*` modules when available and
    falls back to loading system libraries.
    """
    name_bytes = name if isinstance(name, bytes) else name.encode("utf-8")
    name_str = name_bytes.decode("ascii", errors="ignore") or None

    platform = pyglet.compat_platform

    def _try_lib_export(lib, nm):
        if not lib or not nm:
            return None
        func = getattr(lib, nm, None)
        if func is None:
            return None
        try:
            val = cast(func, c_void_p).value
            if val is not None:
                return int(val)
        except (TypeError, ValueError, AttributeError):
            # try attribute fallback
            val = getattr(func, "value", None)
            if val is not None:
                return int(val)
        return None

    # Windows
    if platform in ("win32", "cygwin"):
        _lib_wgl = None
        try:
            _lib_wgl = importlib.import_module("pyglet.gl.lib_wgl")
        except ImportError:
            pass

        # library export first
        addr = _try_lib_export(getattr(_lib_wgl, "gl_lib", None), name_str)
        if addr:
            return addr

        # wglGetProcAddress fallback
        wgl = getattr(_lib_wgl, "wglGetProcAddress", None)
        if wgl is not None:
            try:
                wgl.argtypes = [c_char_p]
                wgl.restype = c_void_p
                addr = wgl(name_bytes)
                if addr:
                    return int(addr)
            except (TypeError, OSError):
                pass

        # final fallback: opengl32
        try:
            opengl32 = ctypes.windll.opengl32
            func = getattr(opengl32, name_str, None) if name_str else None
            if func is not None:
                val = cast(func, c_void_p).value
                if val is not None:
                    return int(val)
        except (AttributeError, OSError):
            pass

        return 0

    # Linux
    if platform.startswith("linux"):
        _lib_glx = None
        try:
            _lib_glx = importlib.import_module("pyglet.gl.lib_glx")
        except ImportError:
            pass

        addr = _try_lib_export(getattr(_lib_glx, "gl_lib", None), name_str)
        if addr:
            return addr

        for proc_name in ("glXGetProcAddressARB", "glXGetProcAddress"):
            glX = getattr(_lib_glx, proc_name, None)
            if glX is None:
                continue
            try:
                glX.argtypes = [c_char_p]
                glX.restype = c_void_p
                addr = glX(name_bytes)
                if addr:
                    return int(addr)
            except (TypeError, OSError):
                continue

        # final fallback: try libGL
        for lib_name in ("libGL.so.1", "libGL.so"):
            try:
                libGL = ctypes.CDLL(lib_name)
            except OSError:
                libGL = None
            if libGL is None:
                continue
            func = getattr(libGL, name_str, None) if name_str else None
            if func is not None:
                try:
                    val = cast(func, c_void_p).value
                    if val is not None:
                        return int(val)
                except (TypeError, ValueError, AttributeError):
                    continue

        return 0

    # macOS
    if platform == "darwin":
        _lib_agl = None
        try:
            _lib_agl = importlib.import_module("pyglet.gl.lib_agl")
        except ImportError:
            pass

        addr = _try_lib_export(getattr(_lib_agl, "gl_lib", None), name_str)
        if addr:
            return addr

        try:
            opengl_framework = ctypes.CDLL(
                "/System/Library/Frameworks/OpenGL.framework/OpenGL"
            )
            func = getattr(opengl_framework, name_str, None) if name_str else None
            if func is not None:
                val = cast(func, c_void_p).value
                if val is not None:
                    return int(val)
        except OSError:
            pass

        return 0

    return 0


# MPV callback wrapper and persistent reference
def get_proc_address(_ctx, name):
    try:
        return resolve_gl_proc_with_pyglet(name)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
        logging.warning("get_proc_address unexpected error: %s", e)
        return 0


# Keep an explicit MPV wrapper alive for the lifetime of the player
# `mpv.MpvGlGetProcAddressFn` will create the proper C-callable object.
c_getproc = mpv.MpvGlGetProcAddressFn(get_proc_address)


def main():
    """Main execution function."""

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
        size=(1280, 720),
        fullscr=False,
        useFBO=True,  # Enable FBO for offscreen rendering
        # waitBlanking=True,
        units="pix",
        color=(-1, -1, -1),  # black background
    )

    # Create MPV player with hardware decoding and libmpv video output
    print("Initializing MPV player...")
    player = mpv.MPV(
        vo="libmpv",  # Required for render context API
        hwdec="auto",  # Enable hardware decoding (will use NVDEC/VAAPI/etc.)
        log_handler=print,
        loglevel="trace",
        gpu_api="opengl",
        fbo_format="rgba32f",
        **{"msg-level": "vo=trace"},
    )

    # Set looping
    player.loop_playlist = "inf"  # Loop indefinitely

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

    # Load and play video
    print(f"Loading video: {video_path}")
    player.play(video_path)
    player.wait_until_playing()

    # Log hardware decoding status
    hwdec_current = player.hwdec_current
    print(
        f"Hardware decoding: {hwdec_current if hwdec_current else 'software (no hwdec)'}"
    )

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

    # Viewport positioning (optional - for demonstration)
    viewport_x, viewport_y = 0, 0
    viewport_width, viewport_height = win_width, win_height

    print("\nPlayback started. Controls:")
    print("  SPACE: Toggle pause")
    print("  ESC/Q: Quit")
    print("\nRendering...")

    # Main render loop
    frame_count = 0
    clock = core.Clock()

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

            # Render MPV frame directly to window's FBO using high-level API
            # Set viewport for the MPV render target, then render.
            gl.glViewport(viewport_x, viewport_y, viewport_width, viewport_height)
            render_ctx.render(
                opengl_fbo={"fbo": fbo_id, "w": win_width, "h": win_height},
                flip_y=True,
            )

            # Bind PsychoPy's framebuffer as the draw target
            if win.useFBO:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

            # Draw instructions on top
            instructions.draw()

            # Flip to display
            win.flip()

            frame_count += 1

            # Print status every 60 frames
            if frame_count % 60 == 0:
                elapsed = clock.getTime()
                fps = frame_count / elapsed if elapsed > 0 else 0
                time_pos = player.time_pos if player.time_pos else 0
                duration = player.duration if player.duration else 0
                print(
                    f"Frame: {frame_count}, FPS: {fps:.1f}, Position: {time_pos:.1f}/{duration:.1f}s"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("\nCleaning up...")

        # Free render context
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


if __name__ == "__main__":
    main()
