from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING

import mpv
from psychopy import logging, visual

from . import utils

if TYPE_CHECKING:
    from typing import Any

# these are possible performance tweaks
# "scale": "bilinear",
# "scale": "lanczos",
# "dscale": "hermite",
# "fbo_format": "rgba16f",
# "dither-depth": 8,
# "dither": "fruit",
# "audio_exclusive": "yes",


class MpvmMoviestim:
    c_getproc: ctypes._CFunctionType
    window: visual.Window
    player: mpv.MPV
    mpv_render_ctx: mpv.MpvRenderContext
    mpv_options: dict[str, Any]
    mpv_default_options: dict[str, Any] = {
        "vo": "libmpv",
        "hwdec": "auto-safe",
        "gpu_api": "opengl",
        "pause": True,
        "keep-open": True,
        "wid": 0,
    }
    mpv_default_audio_options: dict[str, bool | int | float | str] = {
        "volume": 100,
        "volume_gain": 0,
        "audio_device": "auto",
        # "audio-stream-silence": True,  # feeds ao silent audio even when paused
    }
    loaded_movie: Path | None = None

    def __init__(
        self,
        window: visual.Window,
        fileName: Path | str | None = None,
        autoStart: bool = False,
        noAudio: bool = False,
        mpv_options: dict[str, Any] | None = None,
    ):
        # combine default options with user options, and add log handler
        self.mpv_options = self.mpv_default_options.copy()
        self.mpv_options.update({"log_handler": self._mpv_log_fn, "loglevel": "info"})

        if noAudio:
            self.mpv_options["ao"] = "null"
        else:
            self.mpv_options.update(self.mpv_default_audio_options)

        if mpv_options is not None:
            self.mpv_options.update(mpv_options)

        # create MPV player instance
        self.player = mpv.MPV(**self.mpv_options)  # type: ignore
        # self.player.observe_property("frame-drop-count", self._on_drop)
        self.player.observe_property("eof-reached", self._on_eof)

        # setup OpenGL context
        self.c_getproc = mpv.MpvGlGetProcAddressFn(utils.get_proc_address)
        self.mpv_render_ctx = mpv.MpvRenderContext(
            self.player,
            "opengl",
            opengl_init_params={"get_proc_address": self.c_getproc},
        )
        self.window = window

        if fileName is not None:
            self.loadMovie(fileName)

        if autoStart:
            self.play()

    def _mpv_log_fn(self, level: int, prefix: str, text: str) -> None:
        print(f"MPV: {level=}, {prefix=}, {text=}")
        logging.exp(f"MPV: {text}")

    def _on_eof(self, prop_name, value) -> None:
        print(f"EOF reached. Property {prop_name} changed to {value}")
        # TODO

    def _on_drop(self, prop_name, value) -> None:
        print(f"Frame dropped. Property {prop_name} changed to {value}")
        # TODO

    def loadMovie(self, fileName: Path | str) -> None:
        if isinstance(fileName, str):
            fileName = Path(fileName)
        if not fileName.exists():
            logging.error(f"File '{fileName}' does not exist.")
            raise FileNotFoundError(f"File '{fileName}' does not exist.")
        self.player.play(fileName)

    def load(self, fileName: Path | str) -> None:
        self.loadMovie(fileName)

    def play(self) -> None:
        # TODO
        pass

    def pause(self) -> None:
        # TODO
        pass

    def stop(self) -> None:
        # TODO
        pass
