from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import mpv
import numpy as np
import pyglet
from matplotlib import pyplot as plt
from psychopy import logging, visual
from psychopy.visual import vlcmoviestim

from mpvmoviestim import mpvmoviestim, utils


@dataclass(frozen=True)
class TestOptions:
    # TestOptions = SimpleNamespace(
    psychopy_window_use_fbo = False
    psychopy_window_wait_blanking = True
    psychopy_window_size = [3840, 2160]
    psychopy_window_fullscreen = True
    mpv_scale = "lanczos"
    mpv_dscale = "hermite"
    # mpv_fbo_format = "rgba8"
    # mpv_set_timing_offset = True,
    mpv_enable_audio = True
    mpv_report_swap = True
    VIDEO_FILE = Path(r".\tests\media\4k144.mkv")


@dataclass(frozen=True)
class TestOptions2:
    # TestOptions = SimpleNamespace(
    psychopy_window_use_fbo = False
    psychopy_window_wait_blanking = True
    psychopy_window_size = [1280, 720]
    psychopy_window_fullscreen = False
    mpv_scale = "lanczos"
    mpv_dscale = "hermite"
    # mpv_fbo_format = "rgba8"
    # mpv_set_timing_offset = True,
    mpv_enable_audio = True
    mpv_report_swap = True
    VIDEO_FILE = Path(r".\tests\media\out72.mkv")  # 1920 x 960


test_options = TestOptions()

if not test_options.VIDEO_FILE.exists():
    raise FileNotFoundError(
        f"ERROR: No video file found. Looking for: {test_options.VIDEO_FILE}"
    )


def main() -> None:
    # set logging level to EXPERIMENT
    logging.console.setLevel(logging.INFO)

    # ignore Windows screen scaling
    utils.windows_set_scaling_aware()

    if test_options.psychopy_window_fullscreen:
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()

        window_size = [
            min(screen.width, test_options.psychopy_window_size[0]),
            min(screen.height, test_options.psychopy_window_size[1]),
        ]
    else:
        window_size = test_options.psychopy_window_size

    # create PsychoPy window
    win = visual.Window(
        size=window_size,
        fullscr=test_options.psychopy_window_fullscreen,
        useFBO=test_options.psychopy_window_use_fbo,
        waitBlanking=test_options.psychopy_window_wait_blanking,
        units="pix",
        color=(-1, -1, -1),  # black background
    )
    win.recordFrameIntervals = True

    player = vlcmoviestim.VlcMovieStim(
        win,
        filename=str(test_options.VIDEO_FILE),
        noAudio=not test_options.mpv_enable_audio,
        autoStart=False,
        size=(window_size[0], window_size[1]),
        pos=(0, 0),
    )

    # player.preroll()

    # # Create instruction text
    # instructions = visual.TextStim(
    #     win,
    #     text="Press SPACE to toggle pause\nPress ESC or Q to quit",
    #     pos=(0, 2160 // 2 - 40),
    #     height=20,
    #     color="white",
    #     units="pix",
    # )

    times = np.zeros((500,), dtype=np.float64)

    player.play()
    for i in range(500):
        t0 = perf_counter()
        player.draw()
        times[i] = perf_counter() - t0
        win.flip()

    player.stop()
    win.close()

    frame_intervals = np.array(win.frameIntervals)
    frame_intervals *= 1000

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(times.size)
    ax.plot(x, times, color="black", linewidth=2, label="draw")
    x = np.arange(frame_intervals.size)
    ax.plot(x, frame_intervals, color="red", linewidth=2, label="frame interval")

    # --- Styling ---
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stacked Bar + Line Plot")
    ax.set_ylim(0, np.max(frame_intervals[2:]) * 1.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    logging.flush()
