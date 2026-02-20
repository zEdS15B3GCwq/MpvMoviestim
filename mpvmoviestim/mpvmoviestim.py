from __future__ import annotations

import ctypes
import functools
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import mpv
from psychopy import logging, visual
from pyglet import gl

from . import utils
from .pixel_format_probe import get_psychopy_target_pixel_format

if TYPE_CHECKING:
    from typing import Any, Callable

# these are possible performance tweaks
# "scale": "bilinear",
# "scale": "lanczos",
# "dscale": "hermite",
# "fbo_format": "rgba16f",
# "dither-depth": 8,
# "dither": "fruit",
# "audio_exclusive": "yes",


def log_pre_post(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        instance: MpvmMoviestim = args[0]
        print(
            f"PRE {func.__name__}: "
            f"state={instance._player_state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        result = func(*args, **kwargs)
        print(
            f"POST {func.__name__}: "
            f"state={instance._player_state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        return result

    return wrapper


def state_guard(
    allowed_state: MpvState | list[MpvState] | None = None,
    forbidden_state: MpvState | list[MpvState] | None = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:

    allowed_states = (
        None
        if allowed_state is None
        else [allowed_state]
        if isinstance(allowed_state, MpvState)
        else allowed_state
    )

    forbidden_states = (
        None
        if forbidden_state is None
        else [forbidden_state]
        if isinstance(forbidden_state, MpvState)
        else forbidden_state
    )

    def decorator(func: Callable[..., None]) -> Callable[..., None]:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            instance: MpvmMoviestim = args[0]
            actual_state = instance._player_state  # pylint: disable=protected-access

            if allowed_states is not None and actual_state not in allowed_states:
                logging.warning(
                    f"'{func.__name__}' failed: state must be one of "
                    f"[{','.join([s.name for s in allowed_states])}], "
                    f"got {actual_state.name}"
                )
                return

            if forbidden_states is not None and actual_state in forbidden_states:
                logging.warning(
                    f"'{func.__name__}' failed: state must not be one of "
                    f"[{','.join([s.name for s in forbidden_states])}], "
                    f"got {actual_state.name}"
                )
                return

            func(*args, **kwargs)

        return wrapper

    return decorator


class MpvState(Enum):
    UNKNOWN = auto()
    SHUTDOWN = auto()  # MPV player core has shut down after quit()
    IDLE = auto()  # MPV core is active, no file loaded
    PAUSED = auto()  # file loaded but not playing
    PLAYING = auto()  # playing


class MpvMoviestim:
    _c_getproc: ctypes._CFunctionType
    _window: visual.Window
    _player: mpv.MPV
    _mpv_render_ctx: mpv.MpvRenderContext
    _mpv_options: dict[str, Any]
    _mpv_default_options: dict[str, Any] = {
        "vo": "libmpv",  # render using the render_context API
        "hwdec": "auto-safe",  # automatically choose H/W decoding pipeline
        "gpu_api": "opengl",  # use OpenGL API
        "keep-open": True,  # pause when reaching the end of the current file
        "idle": True,  # do not quit when there is no file to play
        "pause": True,  # start paused
        "wid": 0,  # do not create a new window (implied by other settings)
    }
    _mpv_default_audio_options: dict[str, Any] = {
        "volume": 100,  # set volume to 100%
        "volume_gain": 0,  # another way to set loudness
        "audio_device": "auto",  # automatically choose audio output device
        "audio-stream-silence": True,  # feeds ao silent audio even when paused
    }
    _loaded_movie: Path | None = None
    _autostart: bool
    _player_state: MpvState
    _threaded_mode: bool
    _target_fbo: mpv.MpvOpenGLFBO

    def __init__(
        self,
        window: visual.Window,
        file: Path | str,
        autoStart: bool = False,
        noAudio: bool = False,
        volume: int | float = 100,
        mpv_options: dict[str, Any] | None = None,
        block: bool = False,
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

        self._window = window
        self._autostart = autoStart
        # self._mpv_options["pause"] = not autoStart
        self._player_state = MpvState.UNKNOWN
        self._threaded_mode = False
        self._report_swap_on_next = False

        if self._threaded_mode:
            raise NotImplementedError("Threaded mode not implemented yet.")
        else:
            self._init_mpv_player()
            self.loadMovie(file, block)

    @state_guard(allowed_state=MpvState.UNKNOWN)
    def _init_mpv_player(self) -> None:
        # create MPV player instance
        try:
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

            # determine the best pixel format to render to
            # this is straightforward if useFBO is set,
            # needs some guessing based on bit-per-colour values if not
            inferred_fbo_format, format_name = get_psychopy_target_pixel_format(
                self._window
            )
            logging.info(
                f"Psychopy's rendering FBO: fbo={inferred_fbo_format['fbo']}, "
                f"size={inferred_fbo_format['w']}x{inferred_fbo_format['h']}, "
                f"format={format_name if format_name else '<undetermined>'}"
            )
            self._target_fbo = mpv.MpvOpenGLFBO(**inferred_fbo_format)

            # set IDLE state, meaning core is active, file not loaded
            self._player_state = MpvState.IDLE
        except Exception as e:
            logging.error(f"Failed to initialize MPV player: {e}")
            raise

    def _mpv_log_fn(self, level: int, prefix: str, text: str) -> None:
        print(f"MPV: {level=}, {prefix=}, {text=}")
        logging.exp(f"MPV: {text}")

    def _on_eof(self, prop_name, value) -> None:
        print(
            f"EOF reached. Property {prop_name} changed to {value}; "
            f"mpv state: {self._mpv_state.name}"
        )
        # TODO

    def _on_drop(self, prop_name, value) -> None:
        print(f"Frame dropped. Property {prop_name} changed to {value}")
        # TODO

    @log_pre_post
    @state_guard(forbidden_state=[MpvState.UNKNOWN, MpvState.SHUTDOWN])
    def loadMovie(self, file: Path | str, block: bool = True) -> None:
        # TODO figure out: how to implement autoStart, can pause=True be set here?
        if isinstance(file, str):
            file = Path(file)
        file = file.resolve()
        if not file.exists():
            logging.error(f"File '{file}' does not exist.")
            raise FileNotFoundError(f"File '{file}' does not exist.")
        # self._player.loadfile(
        #     filename=str(file), mode="replace", pause=not self._autostart
        # )
        # if self._autostart:
        #     if block:
        #         self._player.wait_until_playing()
        #     self._player_state = MpvState.PLAYING
        # else:
        #     if block:
        #         self._player.wait_until_paused()
        #     self._player_state = MpvState.PAUSED
        self._player.loadfile(filename=str(file), mode="replace")
        self._loaded_movie = file
        if block:
            self._player.wait_until_paused()
        logging.exp(f"Loaded movie '{file}' with autostart set to {self._autostart}.")
        if self._autostart:
            self.play(block)

    def load(self, fileName: Path | str, block: bool = False) -> None:
        self.loadMovie(fileName, block)

    @log_pre_post
    @state_guard(allowed_state=MpvState.PAUSED)
    def play(self, block: bool = False) -> None:
        self._report_swap_on_next = False
        self._player.pause = False
        if block:
            self._player.wait_until_playing()
        self._player_state = MpvState.PLAYING
        logging.exp("State change: PAUSED -> PLAYING")

    @log_pre_post
    @state_guard(allowed_state=MpvState.PLAYING)
    def pause(self, block: bool = False) -> None:
        self._player.pause = True
        if block:
            self._player.wait_until_paused()
        self._player_state = MpvState.PAUSED
        logging.exp("State change: PLAYING -> PAUSED")

    @log_pre_post
    @state_guard(allowed_state=[MpvState.PAUSED, MpvState.PLAYING])
    def stop(self, block: bool = False) -> None:
        # TODO: check if it's really core-idle afterwards or mpv shuts down
        self._player.stop()
        if block:
            self._player.wait_for_property("idle-active", timeout=5, catch_errors=True)
        self._player_state = MpvState.IDLE
        logging.exp(f"State change: {self._player_state.name} -> IDLE")

    def preroll(self) -> None:
        # TODO
        pass

    def draw(self) -> None:
        if self._player_state != MpvState.PLAYING:
            logging.warning(
                f"Cannot draw(), expected PLAYING state, got {self._player_state.name}."
            )
            return

        if self._threaded_mode:
            raise NotImplementedError("Threaded mode not implemented yet.")

        ctx = self._mpv_render_ctx
        if ctx is None:
            raise RuntimeError("render context is None")

        if self._report_swap_on_next:
            ctx.report_swap()
            self._report_swap_on_next = False

        if ctx.update():
            self._report_swap_on_next = True

        # save current viewport
        viewport = (ctypes.c_int * 4)()
        gl.glGetIntegerv(gl.GL_VIEWPORT, viewport)

        # render frame directly either to screen backbuffer or PsychoPy's FBO
        ctx.render(
            opengl_fbo=self._target_fbo, flip_y=True, block_for_target_time=False
        )

        # restore viewport
        gl.glViewport(*viewport)

    @property
    def state(self) -> MpvState:
        return self._player_state

    @property
    def _mpv_state(self) -> MpvState:
        # TODO: test
        if self._player is None:
            return MpvState.UNKNOWN
        if self._player.pause:
            return MpvState.PAUSED
        if self._player.core_shutdown:
            return MpvState.SHUTDOWN
        if self._player.idle_active:
            return MpvState.IDLE
        if not self._player.core_idle:
            return MpvState.PLAYING
        return MpvState.UNKNOWN
