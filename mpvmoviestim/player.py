from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyglet
from psychopy import logging, visual

from . import mpvmoviestim, utils

VIDEO_FILE = Path(r"test_videos\4k144.mkv")

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
    mpv_fbo_format = "rgba8"
    # mpv_set_timing_offset = True,
    mpv_enable_audio = True
    mpv_report_swap = True


test_options = TestOptions()


def main() -> None:
    # set logging level to EXPERIMENT
    logging.console.setLevel(logging.EXP)

    # ignore Windows screen scaling
    utils.windows_set_scaling_aware()

    display = pyglet.canvas.get_display()
    screen = display.get_default_screen()

    window_size = (
        min(screen.width, test_options.psychopy_window_size[0]),
        min(screen.height, test_options.psychopy_window_size[1]),
    )

    # create PsychoPy window
    win = visual.Window(
        size=window_size,
        fullscr=test_options.psychopy_window_fullscreen,
        useFBO=test_options.psychopy_window_use_fbo,
        waitBlanking=test_options.psychopy_window_wait_blanking,
        units="pix",
        color=(-1, -1, -1),  # black background
    )

    player = mpvmoviestim.MpvmMoviestim(win, noAudio=not test_options.mpv_enable_audio)
