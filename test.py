from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pyglet
from psychopy import logging, visual

from mpvmoviestim import mpvmoviestim, utils

# VIDEO_FILE = Path(r"test_videos\4k144.mkv")
VIDEO_FILE = Path(r".\tests\media\out72.mkv")  # 1920 x 960

if not VIDEO_FILE.exists():
    raise FileNotFoundError(f"ERROR: No video file found. Looking for: {VIDEO_FILE}")


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


test_options = TestOptions2()


def main() -> None:
    # set logging level to EXPERIMENT
    logging.console.setLevel(logging.EXP)

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

    player = mpvmoviestim.MpvMoviestim(
        window=win,
        file=VIDEO_FILE,
        noAudio=not test_options.mpv_enable_audio,
        autoStart=False,
        block=True,
    )

    player.play(block=True)
    t0 = perf_counter()
    while perf_counter() - t0 < 10:
        player.draw()
        win.flip()


if __name__ == "__main__":
    main()
    logging.flush()
