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
    _c_getproc: ctypes._CFunctionType
    _window: visual.Window
    _player: mpv.MPV
    _mpv_render_ctx: mpv.MpvRenderContext
    _mpv_options: dict[str, Any]
    _mpv_default_options: dict[str, Any] = {
        "vo": "libmpv",
        "hwdec": "auto-safe",
        "gpu_api": "opengl",
        "pause": True,
        "keep-open": True,
        "wid": 0,
    }
    _mpv_default_audio_options: dict[str, bool | int | float | str] = {
        "volume": 100,
        "volume_gain": 0,
        "audio_device": "auto",
        # "audio-stream-silence": True,  # feeds ao silent audio even when paused
    }
    _loaded_movie: Path | None = None

    def __init__(
        self,
        window: visual.Window,
        file: Path | str,
        autoStart: bool = False,
        noAudio: bool = False,
        volume: int | float = 100,
        mpv_options: dict[str, Any] | None = None,
    ):
        # combine default options with user options, and add log handler
        self._mpv_options = self._mpv_default_options.copy()
        self._mpv_options.update({"log_handler": self._mpv_log_fn, "loglevel": "info"})

        if noAudio:
            self._mpv_options["ao"] = "null"
        else:
            self._mpv_options.update(self._mpv_default_audio_options)
            self._mpv_options["volume"] = volume

        if mpv_options is not None:
            self._mpv_options.update(mpv_options)

        # create MPV player instance
        self._player = mpv.MPV(**self._mpv_options)  # type: ignore
        # self.player.observe_property("frame-drop-count", self._on_drop)
        self._player.observe_property("eof-reached", self._on_eof)

        # setup OpenGL context
        self._c_getproc = mpv.MpvGlGetProcAddressFn(utils.get_proc_address)
        self._mpv_render_ctx = mpv.MpvRenderContext(
            self._player,
            "opengl",
            opengl_init_params={"get_proc_address": self._c_getproc},
        )
        self._window = window

        self.loadMovie(file)

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

    def loadMovie(self, file: Path | str) -> None:
        if isinstance(file, str):
            file = Path(file)
        if not file.exists():
            logging.error(f"File '{file}' does not exist.")
            raise FileNotFoundError(f"File '{file}' does not exist.")
        self.player.play(file)

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

    def preroll(self) -> None:
        # TODO
        pass
