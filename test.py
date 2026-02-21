# trying to render into intermediate FBO, then blit immmediately to screen
# may need fence/sync

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pyglet
from matplotlib import pyplot as plt
from psychopy import logging, visual

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
    raise FileNotFoundError(f"ERROR: No video file found. Looking for: {VIDEO_FILE}")


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

    player = mpvmoviestim.MpvMoviestim(
        window=win,
        file=test_options.VIDEO_FILE,
        noAudio=not test_options.mpv_enable_audio,
        autoStart=False,
        block=True,
    )

    # player.preroll()

    # Create instruction text
    instructions = visual.TextStim(
        win,
        text="Press SPACE to toggle pause\nPress ESC or Q to quit",
        pos=(0, 2160 // 2 - 40),
        height=20,
        color="white",
        units="pix",
    )

    player.play(block=True)
    row = np.zeros((8,), dtype=np.float64)
    times = np.zeros((1000, 8), dtype=np.float64)
    n = 0
    while n < times.shape[0]:
        t0 = perf_counter()
        player.draw(row)

        instructions.draw()

        t1 = perf_counter()
        row[6] = t1 - t0
        times[n] = row
        win.flip()
        n += 1

    frame_intervals = np.array(win.frameIntervals)
    n = min(n, frame_intervals.shape[0], times.shape[0])
    times = times[:n]
    times[:, 7] = frame_intervals[:n]
    times *= 1000

    player.stop()
    win.close()

    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Stacked bar plot ---
    labels = [
        "checks",
        "report_swap",
        "update",
        "backup viewport",
        "render",
        "restore viewport",
        "external",
        "frame interval",
    ]
    bottom = np.zeros(n)
    for i in range(6):  # columns 0..5
        ax.bar(x, times[:, i], bottom=bottom, label=labels[i])
        bottom += times[:, i]

    # --- Line plots ---
    ax.plot(x, times[:, 6], color="black", linewidth=2, label="external")
    ax.plot(x, times[:, 7], color="red", linewidth=2, label="frame interval")

    # --- Styling ---
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stacked Bar + Line Plot")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # for each column in times, calculate mean value for rows >=2
    # print out the result using labels, e.g. column 0, label0
    for i in range(times.shape[1]):
        mean_value = np.mean(times[2:, i])
        print(f"{labels[i]}: {mean_value:.3f} ms")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    logging.flush()
