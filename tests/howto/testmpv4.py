#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MPV video player with zero-copy rendering to PsychoPy window.

This script demonstrates using libmpv's render context API to achieve
hardware-accelerated, zero-copy video playback directly into PsychoPy's
OpenGL framebuffer.
"""

import ctypes
import os
import sys

from pyglet import gl

# Import PsychoPy components
from psychopy import core, event, logging, visual

# Import MPV
try:
    import mpv
except ImportError:
    print("ERROR: python-mpv not installed. Install with: pip install python-mpv")
    sys.exit(1)

# Platform-specific OpenGL function pointer loading
if sys.platform == "win32":
    # Windows: Use wglGetProcAddress from opengl32.dll
    opengl32 = ctypes.windll.opengl32
    wglGetProcAddress = opengl32.wglGetProcAddress
    wglGetProcAddress.argtypes = [ctypes.c_char_p]
    wglGetProcAddress.restype = ctypes.c_void_p

    def get_proc_address(ctx, name):
        """OpenGL function address resolver for MPV on Windows."""
        try:
            func_name = name if isinstance(name, bytes) else name.encode("utf-8")
            addr = wglGetProcAddress(func_name)

            # wglGetProcAddress returns 0, 1, 2, 3, or -1 on failure
            if addr and addr > 3:
                return addr

            # Fallback: get from opengl32.dll for core GL 1.1 functions
            try:
                func = getattr(opengl32, func_name.decode("utf-8"))
                return ctypes.cast(func, ctypes.c_void_p).value
            except (AttributeError, UnicodeDecodeError):
                return 0
        except Exception as e:
            logging.warning(f"Failed to get proc address for {name}: {e}")
            return 0

elif sys.platform.startswith("linux"):
    # Linux: Use glXGetProcAddress from libGL.so
    try:
        # Try to load libGL.so with common versioned names
        for lib_name in ["libGL.so.1", "libGL.so"]:
            try:
                libGL = ctypes.CDLL(lib_name)
                break
            except OSError:
                continue
        else:
            raise OSError("Could not load libGL.so")

        # glXGetProcAddress or glXGetProcAddressARB
        try:
            glXGetProcAddress = libGL.glXGetProcAddressARB
        except AttributeError:
            glXGetProcAddress = libGL.glXGetProcAddress

        glXGetProcAddress.argtypes = [ctypes.c_char_p]
        glXGetProcAddress.restype = ctypes.c_void_p

        def get_proc_address(ctx, name):
            """OpenGL function address resolver for MPV on Linux."""
            try:
                func_name = name if isinstance(name, bytes) else name.encode("utf-8")
                addr = glXGetProcAddress(func_name)
                return addr if addr else 0
            except Exception as e:
                logging.warning(f"Failed to get proc address for {name}: {e}")
                return 0

    except (OSError, AttributeError) as e:
        logging.error(f"Failed to load glXGetProcAddress: {e}")
        logging.error("Falling back to PyOpenGL implementation")

        # Fallback: Try PyOpenGL if available
        try:
            from OpenGL import GLX

            def get_proc_address(ctx, name):
                """OpenGL function address resolver for MPV on Linux (PyOpenGL fallback)."""
                try:
                    func_name = (
                        name.decode("utf-8") if isinstance(name, bytes) else name
                    )
                    address = GLX.glXGetProcAddress(func_name)
                    return ctypes.cast(address, ctypes.c_void_p).value if address else 0
                except Exception as e:
                    logging.warning(f"Failed to get proc address for {name}: {e}")
                    return 0
        except ImportError:
            logging.error("PyOpenGL not available. Install with: pip install PyOpenGL")
            sys.exit(1)

elif sys.platform == "darwin":
    # macOS: OpenGL functions are in the OpenGL framework
    try:
        opengl_framework = ctypes.CDLL(
            "/System/Library/Frameworks/OpenGL.framework/OpenGL"
        )

        def get_proc_address(ctx, name):
            """OpenGL function address resolver for MPV on macOS."""
            try:
                func_name = name.decode("utf-8") if isinstance(name, bytes) else name

                # On macOS, OpenGL functions are directly available in the framework
                try:
                    func = getattr(opengl_framework, func_name)
                    return ctypes.cast(func, ctypes.c_void_p).value
                except AttributeError:
                    # Function not found
                    return 0
            except Exception as e:
                logging.warning(f"Failed to get proc address for {name}: {e}")
                return 0

    except OSError as e:
        logging.error(f"Failed to load OpenGL framework: {e}")
        sys.exit(1)

else:
    logging.error(f"Unsupported platform: {sys.platform}")
    logging.error("Supported platforms: Windows (win32), Linux (linux), macOS (darwin)")
    sys.exit(1)


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
            "get_proc_address": mpv.MpvGlGetProcAddressFn(get_proc_address)
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
    # You can use glViewport to render video to a specific region
    # For now, we render to the entire window
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

            # MPV's render may leave GL state (framebuffer, blending, program,
            # active texture, viewport) in an unexpected configuration. Reassert
            # the minimal state PsychoPy expects for drawing text into its FBO.
            # Bind PsychoPy's framebuffer as the draw target
            if win.useFBO:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)
            # Ensure we draw to the colour attachment and correct viewport
            # try:
            #     gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
            #     gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            # except Exception:
            #     # Some contexts may not expose glDrawBuffer/glReadBuffer; ignore
            #     pass
            # gl.glViewport(0, 0, win_width, win_height)

            # Restore common state used by PsychoPy text rendering
            # gl.glActiveTexture(gl.GL_TEXTURE0)
            # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            # gl.glEnable(gl.GL_BLEND)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # gl.glUseProgram(0)

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
