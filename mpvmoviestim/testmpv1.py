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


def main() -> None:
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
        size=(1600, 1200),
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
        loglevel="info",
        gpu_api="opengl",
        fbo_format="rgba32f",
        # **{"msg-level": "vo=trace"},
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
    viewport_width, viewport_height = 1270, 800

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
            gl.glViewport(viewport_x, viewport_y, 10, 10)
            render_ctx.render(
                opengl_fbo={"fbo": fbo_id, "w": win_width, "h": win_height},
                flip_y=True,
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
