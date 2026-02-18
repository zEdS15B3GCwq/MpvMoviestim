from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mpv
import pyglet
from psychopy import event, logging, visual
from pyglet import gl

from . import utils

if TYPE_CHECKING:
    from typing import Any, Literal


VIDEO_FILE = Path(r"test_videos\4k144.mkv")

if not VIDEO_FILE.exists():
    raise FileNotFoundError(f"ERROR: No video file found. Looking for: {VIDEO_FILE}")


@dataclass(frozen=True, slots=True)
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


class MpvPlayer:
    c_getproc: ctypes._CFunctionType
    player: mpv.MPV
    mpv_options: dict[str, bool | int | float | str] = {
        "vo": "libmpv",
        "hwdec": "auto-safe",
        "gpu_api": "opengl",
        "scale": "bilinear",
        # "scale": "lanczos",
        # "dscale": "hermite",
        "fbo_format": "rgba16f",
        "pause": True,
        "keep-open": True,
        "wid": 0,
    }
    mpv_render_ctx = None
    render_mode: Literal["auto"] | Literal["direct"] | Literal["threaded"] = "auto"

    def __init__(self, audio=True, extra_options=None):
        self.c_getproc = mpv.MpvGlGetProcAddressFn(utils.get_proc_address)

        if audio:
            self.mpv_options["volume"] = 100
            self.mpv_options["volume_gain"] = 0
            self.mpv_options["audio_device"] = "auto"
            # "audio_exclusive": "yes",
        else:
            self.mpv_options["ao"] = "null"

        if extra_options is not None:
            self.mpv_options.update(extra_options)

        self.player = mpv.MPV(**self.mpv_options)  # type: ignore

        self.mpv_render_ctx = mpv.MpvRenderContext(
            self.player,
            "opengl",
            opengl_init_params={
                # Pass the explicit C-callable wrapper we created above
                "get_proc_address": self.c_getproc
            },
        )


def mpv_log_fn(level: int, prefix: str, text: str) -> None:
    print(f"MPV: {level=}, {prefix=}, {text=}")
    logging.exp(f"MPV: {text}")


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

    mpv_options = {
        "log_handler": mpv_log_fn,
        "loglevel": "info",
        "wid": 0,
    }

    player = MpvPlayer(audio=test_options.mpv_enable_audio, extra_options=mpv_options)
