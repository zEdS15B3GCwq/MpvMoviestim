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
import os
import threading
from pprint import pprint
from time import perf_counter, sleep
from typing import Any

import matplotlib.pyplot as plt
import mpv  # type: ignore
import numpy as np
from pyglet import gl  # type: ignore

# Import PsychoPy components
from psychopy import event, logging, visual
from testutils import get_proc_address, intersect_rect

abort_loop = False
need_render = False
update_count = frame_count = frames_dropped_total = frames_dropped_early = 0

N_EARLY_FRAMES = 10
PREROLL_FIRST_FRAMES = 5
RENDER_ON_UPDATE = False
FORCE_BLIT = True
DESYNC_VIDEO = False
TIMING_OFFSET_ZERO = False
WAIT_BLANK = True

SCREEN_REFRESH = 144
VIDEO_PATH = "4k144.mkv"
WIN_SIZE = (3840, 2160)
# TARGET_RECT = (200, 200, 1920, 1080)
TARGET_RECT = (0, 0, 3840, 2160)
MODE = "auto"
MPV_SCALE = "lanczos"
MPV_DSCALE = "hermite"
MPV_FBO_FORMAT = "rgba16f"

AUDIO_DEVICE = None
NO_AUDIO = False
AUDIO_EXCLUSIVE = False
VOLUME = 100
VOLUME_GAIN = 0.0

DO_TIMING = True

# Keep an explicit MPV wrapper alive for the lifetime of the player
# `mpv.MpvGlGetProcAddressFn` will create the proper C-callable object.
c_getproc = mpv.MpvGlGetProcAddressFn(get_proc_address)




def main() -> None:
    """Main execution function."""
    logging.console.setLevel(logging.EXP)


    # Create PsychoPy window with FBO enabled
    print("Creating PsychoPy window...")
    # ctypes.windll.shcore.SetProcessDpiAwareness(2)
    # ctypes.windll.user32.SetProcessDPIAware()
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_int64(-4))
    win = visual.Window(
        size=list(WIN_SIZE),
        fullscr=True,
        useFBO=False,  # Enable FBO for offscreen rendering
        waitBlanking=WAIT_BLANK,
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

    if TIMING_OFFSET_ZERO:
        mpv_options["video_timing_offset"] = 0

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
    if RENDER_ON_UPDATE:
        render_ctx.update_cb = on_update

    # Allow explicit VIDEO_PATH override before falling back to discovered files
    if VIDEO_PATH:
        if os.path.exists(VIDEO_PATH):
            video_path = VIDEO_PATH
        else:
            print(f"WARNING: VIDEO_PATH set but file not found: {VIDEO_PATH}")

    player.observe_property("frame-drop-count", on_drop)
    player.observe_property("eof-reached", on_eof)

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

    direct_render = (
        not FORCE_BLIT
        and not RENDER_ON_UPDATE
        and (
            TARGET_RECT is None
            or TARGET_RECT
            == (
                0,
                0,
                win_width,
                win_height,
            )
        )
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
    intermediate_tex, intermediate_fbo, _ = create_intermediate_fbo(
        tw, th, MPV_FBO_FORMAT
    )

    if PREROLL_FIRST_FRAMES > 0:
        player.pause = False
        times: list[tuple(float, float)] = []
        for _ in range(PREROLL_FIRST_FRAMES):
            t0 = perf_counter()
            render_ctx.render(
                opengl_fbo={"fbo": intermediate_fbo, "w": tw, "h": th},
                block_for_target_time=False,
                flip_y=True,
            )
            t1 = perf_counter()
            win.flip()
            times.append((t1 - t0, perf_counter() - t1))

        player.pause = True
        print(
            "Initial frame render: "
            f"{[(round(t * 1000, 3), round(u * 1000, 3)) for t, u in times]}"
            " ms"
        )
        sleep(1)

    print(
        f"FPS: {player.container_fps=}, {player.estimated_vf_fps=}, {player.display_fps=}, Duration: {player.duration=  }"
    )
    if player.duration is not None and player.duration > 0:
        nframes = int(player.duration * SCREEN_REFRESH * 1.1)
    else:
        nframes = 10000

    # player.wait_until_playing()

    # Main render loop
    global need_render, render_count, frame_count
    timings = np.zeros(
        (nframes, 8), dtype=np.float64
    )  # loop start, before_render, after_render, before blit, after blit, before flip, after flip, loop end
    t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0.0
    frame_count = render_count = 0
    player.pause = False

    try:
        while not abort_loop:
            if DO_TIMING:
                t0 = perf_counter()

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

            if direct_render:
                # Path A: direct full-window render
                if DO_TIMING:
                    t1 = perf_counter()
                render_ctx.render(
                    opengl_fbo={"fbo": fbo_id, "w": win_width, "h": win_height},
                    block_for_target_time=False,
                    flip_y=True,
                )
                if DO_TIMING:
                    t2 = perf_counter()
                render_count += 1

            else:
                if not RENDER_ON_UPDATE or (need_render and render_ctx.update()):
                    # Render MPV into intermediate FBO (MPV does scaling)
                    if DO_TIMING:
                        t1 = perf_counter()
                    render_ctx.render(
                        opengl_fbo={"fbo": intermediate_fbo, "w": tw, "h": th},
                        block_for_target_time=False,
                        flip_y=True,
                    )
                    if DO_TIMING:
                        t2 = perf_counter()
                    need_render = False
                    render_count += 1
                else:
                    t1 = t2 = 0

                # Bind PsychoPy's framebuffer as the draw target
                if win.useFBO:
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

                # Composite: use blit
                win_rect = (0, 0, win_width, win_height)
                if DO_TIMING:
                    t3 = perf_counter()
                composite_intermediate_blit(
                    intermediate_fbo,
                    (tw, th),
                    TARGET_RECT,
                    win_rect,
                    win.frameBuffer if win.useFBO else 0,
                )
                if DO_TIMING:
                    t4 = perf_counter()

            # Bind PsychoPy's framebuffer as the draw target
            if win.useFBO:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, win.frameBuffer)

            # Draw instructions on top
            gl.glViewport(0, 0, win_width, win_height)
            instructions.draw()

            # Flip to display
            if DO_TIMING:
                t5 = perf_counter()
            win.flip()
            if DO_TIMING:
                t6 = perf_counter()

            # if frame_count == 100:
            #     print("--------seeking")
            #     player.pause = True
            #     player.seek(0, reference="absolute")
            if DO_TIMING:
                t7 = perf_counter()
                timings[frame_count, :] = (t0, t1, t2, t3, t4, t5, t6, t7)
            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if DO_TIMING:
            timings[frame_count, 0] = perf_counter()

        # Cleanup
        print("\nCleaning up...")
        player.stop()

        player.unobserve_property("frame-drop-count", on_drop)

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

        display_stats(frame_count, render_count, timings)

        print("Done!")


def display_stats(frame_count, render_count, timings):
    print("==========================STATISTICS=================================")

    print(f"Counts: {update_count=}  {render_count=}  /  {frame_count=}")

    print(
        f"Total frames dropped: {frames_dropped_total} (early: {frames_dropped_early})"
    )

    timings = timings * 1000
    y = timings[0:frame_count, 0]
    t_frame = np.diff(timings[0:frame_count, 0])
    timings = timings[0 : frame_count - 1, :]
    t_render = timings[:, 2] - timings[:, 1]
    t_blit = timings[:, 4] - timings[:, 3]
    t_flip = timings[:, 6] - timings[:, 5]
    t_process = timings[:, 7] - timings[:, 0]
    x = range(len(t_render))

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

    ctypes.windll.user32.SetProcessDPIAware(2)
    fig = plt.figure(figsize=(22, 8))

    plt.subplot(2, 3, (1, 3))
    y = np.diff(y)

    # plt.plot(range(len(y)), y)
    plt.plot(x, t_frame, "b-", label="frame")
    plt.plot(x, t_render, "r-", label="render")
    plt.plot(x, t_flip, "g-", label="flip")
    plt.plot(x, t_blit, "c-", label="blit")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Time (ms)")
    # plt.plot(
    #     t,
    #     t_frame,
    #     "r-",
    #     t,
    #     t_render,
    #     "b-",
    #     t,
    #     t_blit,
    #     "g-",
    #     t,
    #     t_flip,
    #     "y-",
    #     t,
    #     t_process,
    #     "k-",
    # )

    # exp_ms = 1000 / SCREEN_REFRESH
    # bins = [
    #     0,
    #     exp_ms - 3,
    #     exp_ms - 2,
    #     exp_ms - 1,
    #     exp_ms,
    #     exp_ms + 1,
    #     exp_ms + 2,
    #     exp_ms + 3,
    #     exp_ms + 10,
    #     np.inf,
    # ]
    # bins = [
    #     0,
    #     exp_ms * 0.5,
    #     exp_ms * 0.6,
    #     exp_ms * 0.9,
    #     exp_ms,
    #     exp_ms * 1.1,
    #     exp_ms * 1.4,
    #     exp_ms * 2.0,
    #     np.inf,
    # ]
    bins = list(range(20))
    bins.append(np.inf)

    plt.subplot(2, 3, 5)
    # hc = np.histogram(t_frame, bins=bins)[0]
    plt.hist(t_frame, bins=bins, rwidth=0.9)

    plt.xlabel("Frame time (ms)")
    plt.ylabel("Count")
    plt.xticks(range(22))

    plt.show()


def on_drop(prop_name, value):
    print(f"Property {prop_name} changed -> {value}")
    global frames_dropped_total, frames_dropped_early

    if value is None or value == 0:
        if frames_dropped_total > 0:
            print(
                "Frame drop counts reset to 0. Previous values: "
                f"total={frames_dropped_total}, early={frames_dropped_early}"
            )
        frames_dropped_early = frames_dropped_total = 0
        return

    if frame_count < N_EARLY_FRAMES:
        frames_dropped_total = frames_dropped_early = value
    else:
        frames_dropped_total = value


def on_eof(prop_name, value) -> None:
    print(f"Property {prop_name} changed -> {value}")
    global abort_loop
    if value:
        abort_loop = True


def on_update() -> None:
    global need_render, update_count
    need_render = True
    update_count += 1


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # triple buffering
        # decode buffer - decoding  thread draws into FBO/tex
        # drawing buffer - main thread draws this to window
        # ready buffer - next for drawing or decoding
        self.decode_buffer = (None, None)  # (tex_id, fence_id)
        self.draw_buffer = (None, None)
        self.ready_buffer = (None, None)

        self.target_size = (0, 0)

        # Flags
        self.running = True
        self.paused = True

    def set_eof(self):
        self.running = False

    def draw_next(self):
        with self.lock:
            if self.ready_buffer is not None:



if __name__ == "__main__":
    main()
