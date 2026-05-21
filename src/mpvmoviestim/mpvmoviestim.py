# trying to render into intermediate FBO, then blit immmediately to screen
# may need fence/sync

# TODO: verify if locks/fences are guaranteed to resolve at some time - need timeout?

from __future__ import annotations

import ctypes
import dataclasses
import functools
import threading
from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import mpv
from psychopy import logging, visual
from psychopy.tools.monitorunittools import convertToPix
from pyglet import gl

from . import utils
from .pixel_format_probe import get_psychopy_target_pixel_format

if TYPE_CHECKING:
    from typing import Any, Callable

    from pyglet.window import BaseWindow

# these are possible performance tweaks
# "scale": "bilinear",
# "scale": "lanczos",
# "dscale": "hermite",
# "fbo_format": "rgba16f",  # no reason to specify this, target fbo format does not affect pipeline
# however, for performance gains, one could specify a lesser quality format
# internal fbo format priority order: [rgba16f, rgba16hf, rgba16, rgb10_a2, rgba8]
# "dither-depth": 8,  # mpv already selects this for rgba8 target
# "dither": "fruit",  # already default
# "audio_exclusive": "yes",


__all__ = ["MpvMoviestim"]


def log_pre_post(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log the player and MPV states before and after method execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        instance: MpvMoviestim = args[0]
        print(
            f"PRE {func.__name__}: "  # ty: ignore
            f"state={instance._player_state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        result = func(*args, **kwargs)
        print(
            f"POST {func.__name__}: "  # ty: ignore
            f"state={instance._player_state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        return result

    return wrapper


def state_guard(
    allowed_state: PlayerState | list[PlayerState] | None = None,
    forbidden_state: PlayerState | list[PlayerState] | None = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator to enforce allowed and forbidden states when running methods."""

    allowed_states = (
        None
        if allowed_state is None
        else [allowed_state]
        if isinstance(allowed_state, PlayerState)
        else allowed_state
    )

    forbidden_states = (
        None
        if forbidden_state is None
        else [forbidden_state]
        if isinstance(forbidden_state, PlayerState)
        else forbidden_state
    )

    def decorator(func: Callable[..., None]) -> Callable[..., None]:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            instance: MpvMoviestim = args[0]
            actual_state = instance._player_state  # pylint: disable=protected-access

            if allowed_states is not None and actual_state not in allowed_states:
                logging.warning(
                    f"'{func.__name__}' failed: state must be one of "  # ty: ignore
                    f"[{','.join([s.name for s in allowed_states])}], "
                    f"got {actual_state.name}"
                )
                return

            if forbidden_states is not None and actual_state in forbidden_states:
                logging.warning(
                    f"'{func.__name__}' failed: state must not be one of "  # ty: ignore
                    f"[{','.join([s.name for s in forbidden_states])}], "
                    f"got {actual_state.name}"
                )
                return

            func(*args, **kwargs)

        return wrapper

    return decorator


_mpv_default_options: dict[str, Any] = {
    "vo": "libmpv",  # render using the render_context API
    "hwdec": "auto-safe",  # automatically choose H/W decoding pipeline
    "gpu_api": "opengl",  # use OpenGL API
    "keep-open": True,  # pause when reaching the end of the current file
    # "idle": True,  # do not quit when there is no file to play
    "pause": True,  # start paused
    # "wid": 0,  # do not create a new window (implied by other settings)
    # "keepaspect": False,
}
_mpv_default_audio_options: dict[str, Any] = {
    "volume": 100,  # set volume to 100%
    "volume_gain": 0,  # another way to set loudness
    "audio_device": "auto",  # automatically choose audio output device
    "audio-stream-silence": True,  # feeds ao silent audio even when paused
}


class PlayerState(Enum):
    UNSPECIFIED = auto()
    SHUTDOWN = auto()  # MPV player core has shut down after quit()
    IDLE = auto()  # MPV core is active, no file loaded
    PAUSED = auto()  # file loaded but not playing
    PLAYING = auto()  # playing


@dataclasses.dataclass
class ThreadingState:
    """All threading and FBO-handoff state owned by the render worker."""

    # Core objects
    worker_thread: threading.Thread | None = None
    shadow_window: BaseWindow | None = None

    # Double buffering
    present_fbo_idx: int = -1
    worker_fbo_idx: int = 0
    intermediate_fbos: tuple[dict[str, int], dict[str, int]] | None = None
    intermediate_fbo_textures: tuple[int, int] | None = None

    # Synchronisation
    render_trigger: threading.Event = dataclasses.field(default_factory=threading.Event)
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    worker_init_done: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    worker_is_rendering: bool = False
    main_is_blitting: bool = False
    worker_render_done: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    fbo_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    render_fences: list[Any] = dataclasses.field(default_factory=lambda: [None, None])
    blit_fences: list[Any] = dataclasses.field(default_factory=lambda: [None, None])


class MpvMoviestim:
    # PsychoPy
    _window: visual.Window
    _position: tuple[int | float, int | float]  # position in Psychopy units
    _size: tuple[int | float, int | float] | None  # size in Psychopy units
    # Media
    _loaded_movie: Path
    _autostart: bool
    _media_size: tuple[int, int] | None
    # Player core
    _player_state: PlayerState
    _threading_state: ThreadingState
    _profiling: bool
    _timings: list[list[float]]
    # MPV and OpenGL
    _player: mpv.MPV
    _mpv_options: dict[str, Any]
    _bounding_rect_px: tuple[int, int, int, int]  # presentation bounding rect in pix
    _c_getproc: ctypes._CFunctionType
    _mpv_render_ctx: mpv.MpvRenderContext
    _target_fbo_info: dict[str, int]  # mpv.MpvOpenGLFBO
    _report_swap_on_next: bool

    def __init__(
        self,
        window: visual.Window,
        file: Path | str,
        autoStart: bool = False,
        noAudio: bool = False,
        volume: float = 1.0,
        mpv_options: dict[str, Any] | None = None,
        pos: tuple[int | float, int | float] = (0, 0),
        size: tuple[int | float, int | float] | None = None,
        profiling: bool = False,
    ):
        # combine default options with user options, and add log handler
        self._mpv_options = _mpv_default_options.copy()
        self._mpv_options.update({"log_handler": self._mpv_log_fn, "loglevel": "info"})

        if noAudio:
            self._mpv_options["ao"] = "null"
        else:
            self._mpv_options.update(_mpv_default_audio_options)
            self._mpv_options["volume"] = (
                0 if volume < 0 else 100 if volume > 1 else int(volume * 100)
            )

        if mpv_options is not None:
            self._mpv_options.update(mpv_options)

        self._window = window
        self._size = size
        self._position = pos
        self._media_size = None
        self._autostart = autoStart
        self._player_state = PlayerState.UNSPECIFIED
        self._report_swap_on_next = False
        self._profiling = profiling

        self._update_bounding_rect()

        # Synchronisation primitives — must exist before worker thread starts.
        self._threading_state = ThreadingState()

        self._init_mpv_player()
        self.loadMovie(file)

    @log_pre_post
    @state_guard(allowed_state=PlayerState.UNSPECIFIED)
    def _init_mpv_player(self) -> None:
        # create MPV player instance
        try:
            self._player = mpv.MPV(**self._mpv_options)
            self._player.observe_property("eof-reached", self._on_eof)

            # Build the GL proc-address resolver (a plain function pointer —
            # no GL context required yet).
            self._c_getproc = mpv.MpvGlGetProcAddressFn(utils.get_proc_address)

            # Determine the best pixel format to render to.  Must happen while
            # the main window's context is current (psychopy keeps it current
            # on the main thread).
            inferred_fbo_format, format_name = get_psychopy_target_pixel_format(
                self._window
            )
            logging.info(
                f"Psychopy's rendering FBO: fbo={inferred_fbo_format['fbo']}, "
                f"size={inferred_fbo_format['w']}x{inferred_fbo_format['h']}, "
                f"format={format_name if format_name else '<undetermined>'}"
            )
            self._target_fbo_info = inferred_fbo_format

            # Create the shadow window (shared context) on the main thread so
            # that its DC/surface is set up before the worker uses it.
            # Pass winHandle (the underlying pyglet window) so utils.py does not
            # depend on PsychoPy.
            self._threading_state.shadow_window = utils.create_shadow_window(
                self._window.winHandle
            )

            # Start worker — it takes the shadow context, creates the render
            # context and the double-buffered intermediate FBOs, then signals
            # worker_init_done.
            self._threading_state.worker_thread = threading.Thread(
                target=self._render_worker, name="mpv-render-worker", daemon=True
            )
            self._threading_state.worker_thread.start()
            self._threading_state.worker_init_done.wait()  # block until worker is ready

            # set IDLE state, meaning core is active, file not loaded
            self._player_state = PlayerState.IDLE
        except Exception as e:
            logging.error(f"Failed to initialize MPV player: {e}")
            raise

    def _update_bounding_rect(self) -> None:
        """Update pixel-based bounding rectangle from Psychopy-based size and position."""
        # calculate the media element's position and size in pixels
        # position is always in window units, and is relative to the centre of the window
        # position means the centre of the media element
        # size, if given, is also in window units
        # if size is None, use media size in pixels (and ignore window units)
        # For OpenGL rendering, we need the bottom-left corner and the element size (w/h)
        # in pixels. First, calculate the bottom-left and top-right corners in window units,
        # then convert them to pixels using PsychoPy's `convertToPix` function.
        # In `convertToPix`, set `pos` to [0,0] and provice the vertices as list[list[float, float]].
        # The function calculates the vertex vectors relative to the position vector.
        # Example: pix = convertToPix(pos=[0, 0], vertices=[[-1, -1], [1, 1]], units="norm", win=win)
        # this calculates the bottom-left and top-right corners of the window in pixels.

        if self._size is None and self._media_size is None:
            logging.info(
                "Size not specified and media size not available yet, not updating bounding rect."
            )
            return

        # if units are not pix, convert what's necessary to pixels
        if not self._window.units == "pix":
            if self._size is not None:
                vertices = [
                    (
                        self._position[0] - self._size[0] / 2,
                        self._position[1] - self._size[1] / 2,
                    ),
                    (
                        self._position[0] + self._size[0] / 2,
                        self._position[1] + self._size[1] / 2,
                    ),
                ]
                bottom_left_px, top_right_px = convertToPix(
                    pos=[0, 0],
                    vertices=vertices,
                    units=self._window.units,
                    win=self._window,
                )
                self._bounding_rect_px = (
                    bottom_left_px[0],
                    bottom_left_px[1],
                    top_right_px[0],
                    top_right_px[1],
                )
                return
            else:
                pos_px = convertToPix(
                    pos=[0, 0],
                    vertices=[self._position],
                    units=self._window.units,
                    win=self._window,
                )[0]

            pos_px = self._position

            if self._size is not None:
                half_w = self._size[0] / 2
                half_h = self._size[1] / 2
                self._bounding_rect_px = (
                    int(self._position[0] - half_w),
                    int(self._position[1] - half_h),
                    int(self._position[0] + half_w),
                    int(self._position[1] + half_h),
                )
            else:
                # use media size in pixels
                self._bounding_rect_px = (
                    int(self._position[0] - self._media_size[0] / 2),
                    int(self._position[1] - self._media_size[1] / 2),
                    int(self._position[0] + self._media_size[0] / 2),
                    int(self._position[1] + self._media_size[1] / 2),
                )

        # if self._size is None:
        #     if self._media_size is None:
        #         logging.info(
        #             "Size not specified and media size not available yet, not updating bounding rect."
        #         )
        #         return
        #     else:
        #         centre_x, centre_y = self._window.size[0] / 2, self._window.size[1] / 2
        #         if self._window.units == "pix":
        #             pos_x, pos_y = (
        #                 int(centre_x + self._position[0]),
        #                 int(centre_y + self._position[1]),
        #             )
        #         else:
        #             centre_to_pos_vector = convertToPix(
        #                 pos=[0, 0],
        #                 vertices=[[self._position[0], self._position[1]]],
        #                 units=self._window.units,
        #                 win=self._window,
        #             )[0]
        #             pos_x, pos_y = (
        #                 int(centre_x + centre_to_pos_vector[0]),
        #                 int(centre_y + centre_to_pos_vector[1]),
        #             )
        #         w, h = self._media_size
        #         x, y = pos_x - w // 2, pos_y - h // 2
        #         self._bounding_rect_px = (x, y, x + w, y + h)
        # else:

        #     half_w = size[0] / 2
        #     half_h = size[1] / 2
        #     vertices = [
        #         [position[0] - half_w, position[1] - half_h],
        #         [position[0] + half_w, position[1] + half_h],
        #     ]
        #     corners_pix = convertToPix(
        #         pos=[0, 0], vertices=vertices, units=win_units, win=self._window
        #     )
        #     bl = (int(corners_pix[0][0]), int(corners_pix[0][1]))
        #     tr = (int(corners_pix[1][0]), int(corners_pix[1][1]))
        #     size_pix = (tr[0] - bl[0], tr[1] - bl[1])
        # else:
        #     # Use native media size in pixels; convert only the position
        #     media_w = self._player.width or 0
        #     media_h = self._player.height or 0
        #     pos_pix = convertToPix(
        #         pos=[0, 0],
        #         vertices=[[position[0], position[1]]],
        #         units=win_units,
        #         win=self._window,
        #     )
        #     cx = int(pos_pix[0][0])
        #     cy = int(pos_pix[0][1])
        #     size_pix = (media_w, media_h)
        #     bl = (cx - media_w // 2, cy - media_h // 2)

    def _make_intermediate_fbo(self) -> tuple[dict[str, int], int]:
        """Allocate one intermediate FBO + backing texture on the current GL context.

        Must be called while the shadow (worker) context is current.

        In order to avoid allocating FBOs/textures while playing the media, the
        intermediate FBOs/textures allocated here will have a size that is the larger
        of the target FBO (PsychoPy's rendering FBO or screen backbuffer) and the
        specified size of the media element. Smaller display sizes can use these
        surfaces without needing to allocate new FBOs/textures. Larger display sizes
        are not supported.

        Returns
        -------
        dict[str, int]
            fbo info
        int
            texture id
        """
        target_w = self._target_fbo_info["w"]
        target_h = self._target_fbo_info["h"]
        w = max(target_w, self._size[0] if self._size is not None else 0)
        h = max(target_h, self._size[1] if self._size is not None else 0)
        internal_format = self._target_fbo_info["internal_format"]
        tex_id = utils.create_texture(w, h, internal_format)
        fbo_id = utils.create_fbo(tex_id)
        info: dict[str, int] = {
            "fbo": fbo_id,
            "w": w,
            "h": h,
            "internal_format": internal_format,
        }
        logging.info(f"Intermediate FBO created: {info}. texture: {tex_id}")
        return info, tex_id

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _mpv_update_callback(self) -> None:
        """Called by MPV on its internal C thread when a new frame may be ready.

        Body MUST be trivially fast — only set the event, nothing else.
        """
        self._threading_state.render_trigger.set()

    def _render_worker(self) -> None:
        """Worker thread: owns the shadow GL context and MPV render context.

        Lifecycle
        ---------
        1. Make shadow context current (once, for the life of this thread).
        2. Create ``MpvRenderContext`` with ``advanced_control=True``.
        3. Create double-buffered intermediate FBOs.
        4. Signal ``_worker_init_done`` to unblock the main thread.
        5. Loop: wait for ``_render_trigger``, call ``ctx.update()`` / ``ctx.render()``.
        6. On exit: free render context, destroy FBOs/textures, release context.
        """
        ts = self._threading_state

        # --- one-time init on this thread ---
        assert ts.shadow_window is not None, (
            "shadow_window must be set before the worker starts"
        )
        utils.make_context_current(ts.shadow_window)
        # ts.shadow_window.switch_to()

        # Log which renderer this context sees (sanity check that sharing works).
        renderer = gl.glGetString(gl.GL_RENDERER)
        logging.info(
            f"MPV render worker: GL_RENDERER = {renderer if renderer else '<unknown>'}"
        )

        self._mpv_render_ctx = mpv.MpvRenderContext(
            self._player,
            "opengl",
            opengl_init_params={"get_proc_address": self._c_getproc},
            advanced_control=True,
        )
        # The update callback body MUST only set the event — nothing else.
        self._mpv_render_ctx.update_cb = self._mpv_update_callback

        # Double-buffered intermediate FBOs (created on the shadow context).
        fbo1, tex1 = self._make_intermediate_fbo()
        fbo2, tex2 = self._make_intermediate_fbo()
        ts.intermediate_fbos = (
            fbo1,
            fbo2,
        )
        ts.intermediate_fbo_textures = (
            tex1,
            tex2,
        )
        ts.worker_fbo_idx = 0
        ts.present_fbo_idx = -1  # no frame ready yet

        ts.worker_init_done.set()  # unblock main thread

        # --- render loop ---
        while True:
            ts.render_trigger.wait()
            ts.render_trigger.clear()
            if ts.stop_event.is_set():
                break

            if not self._mpv_render_ctx.update():
                # Spurious callback (non-frame event); nothing to render.
                continue

            ts.worker_render_done.clear()
            with ts.fbo_lock:
                ts.worker_is_rendering = True
                target_idx = ts.worker_fbo_idx

            assert ts.intermediate_fbos is not None, (
                "intermediate_fbos must be set during worker init"
            )
            fbo_info = ts.intermediate_fbos[target_idx]

            # GPU-side: wait until the main thread has finished blitting from
            # this FBO before we overwrite it.
            blit_fence = ts.blit_fences[target_idx]
            if blit_fence is not None:
                gl.glWaitSync(blit_fence, 0, gl.GL_TIMEOUT_IGNORED)
                gl.glDeleteSync(blit_fence)
                ts.blit_fences[target_idx] = None

            self._mpv_render_ctx.render(
                opengl_fbo=fbo_info,
                flip_y=True,
                block_for_target_time=False,
            )

            # Post a fence so the main thread can wait for this render to finish
            # before it blits.
            ts.render_fences[target_idx] = gl.glFenceSync(
                gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0
            )
            # it appears that flushing after fencesync is expected (need to check spec)
            gl.glFlush()

            # Hand off: publish which FBO is ready, clear rendering flag, flip index.
            with ts.fbo_lock:
                ts.worker_is_rendering = False
                ts.present_fbo_idx = target_idx
                ts.worker_fbo_idx = 1 - target_idx

            # After lock release so present_fbo_idx is committed before draw() reads it.
            ts.worker_render_done.set()

        # --- cleanup (shadow context still current on this thread) ---
        self._mpv_render_ctx.free()
        if ts.intermediate_fbo_textures is not None:
            for tex in ts.intermediate_fbo_textures:
                utils.destroy_texture(tex)
        if ts.intermediate_fbos is not None:
            for fbo in ts.intermediate_fbos:
                utils.destroy_fbo(fbo["fbo"])
        utils.release_context()

    def _mpv_log_fn(self, level: int, prefix: str, text: str) -> None:
        print(f"MPV: {level=}, {prefix=}, {text=}")
        logging.exp(f"MPV: {text}")

    @log_pre_post
    def _on_eof(self, prop_name, value) -> None:
        # eof-reached -> None at init
        # eof-reached -> False when playback starts
        # eof-reached -> True at EOF
        # with keep-open set, player pauses instead of closing file
        print(f"on eof: property {prop_name} changed to {value}")
        if value:  # and prop_name == "eof-reached"  # no need
            logging.exp("EOF reached")
            self._player_state = PlayerState.IDLE

    def _on_drop(self, prop_name, value) -> None:
        print(f"Frame dropped. Property {prop_name} changed to {value}")
        # TODO

    # def _on_event(self, event) -> None:
    #     if event == mpv.MpvEventID.SHUTDOWN:
    #         self._on_shutdown()

    # @log_pre_post
    # def _on_shutdown(self) -> None:
    #     print("MPV shutdown")
    #     logging.info("on shutdown called")
    #     self._player_state = MpvState.SHUTDOWN
    #     if hasattr(self, "_intermediate_tex_id"):
    #         utils.destroy_texture(self._intermediate_tex_id)
    #     if hasattr(self, "_intermediate_fbo_info"):
    #         utils.destroy_fbo(self._intermediate_fbo_info["fbo"])

    @log_pre_post
    @state_guard(forbidden_state=[PlayerState.UNSPECIFIED, PlayerState.SHUTDOWN])
    def loadMovie(self, file: Path | str) -> None:
        """Load a movie file into the player, replacing any currently loaded file.

        Parameters
        ----------
        file : Path | str
            Path to the movie file to load.

        """
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
        self._player.wait_until_paused()
        self._player_state = PlayerState.PAUSED

        self._player.wait_for_property("video-params")
        video_params = self._player.video_params
        self._media_size: tuple[int, int] = video_params["w"], video_params["h"]  # type: ignore

        # self._blit_fn = utils.get_blit_fn

        logging.exp(f"Loaded movie '{file}' with autostart set to {self._autostart}.")
        if self._autostart:
            self.play()

    def load(self, fileName: Path | str) -> None:
        """Alias for loadMovie()"""
        self.loadMovie(fileName)

    @log_pre_post
    @state_guard(allowed_state=PlayerState.PAUSED)
    def play(self, block: bool = False) -> None:
        self._report_swap_on_next = False
        self._player.pause = False
        if block:
            self._player.wait_until_playing()
        self._player_state = PlayerState.PLAYING
        logging.exp("State change: PAUSED -> PLAYING")

    @log_pre_post
    @state_guard(allowed_state=PlayerState.PLAYING)
    def pause(self, block: bool = False) -> None:
        self._player.pause = True
        if block:
            self._player.wait_until_paused()
        self._player_state = PlayerState.PAUSED
        logging.exp("State change: PLAYING -> PAUSED")

    @log_pre_post
    @state_guard(allowed_state=[PlayerState.PAUSED, PlayerState.PLAYING])
    def stop(self) -> None:
        self._player_state = PlayerState.SHUTDOWN
        print(f"frame-drop-count: {self._player.frame_drop_count}")

        # Ask MPV to stop the current file and wait until it is idle.
        self._player.stop()
        # self._player.terminate()
        # self._player.wait_for_shutdown()
        # self._player.stop()
        print("idle-active wait")
        self._player.wait_for_property("idle-active")

        # Drain the render worker BEFORE quitting the MPV core.  With
        # advanced_control=True a permanent hang results if we call
        # wait_for_shutdown() while the worker is still alive.
        ts = self._threading_state
        ts.stop_event.set()
        ts.render_trigger.set()  # wake worker if it is blocked on .wait()
        assert ts.worker_thread is not None, (
            "worker_thread must be set before stop() is called"
        )
        ts.worker_thread.join()

        # Now safe to terminate the MPV core.
        self._player.quit()
        print("shutdown wait")
        self._player.wait_for_shutdown()
        # Intermediate FBO/texture cleanup is handled inside the worker thread.

    # @log_pre_post
    # @state_guard(allowed_state=MpvState.PAUSED)
    # def preroll(self) -> None:
    #     # vol = self._player.volume
    #     # self._player.volume = 0
    #     # self.play(block=True)
    #     # self.pause(block=True)
    #     # self._player.volume = vol
    #     self._player.command("frame-step", "1")

    def draw(self) -> None:
        """Blit the latest worker-rendered frame to PsychoPy's FBO.

        This method should be called in PsychoPy's main render loop
        to draw the current video frame onto the screen.

        Notes
        -----
        * Video frames are rendered asynchronously by a separate
        worker thread into double-buffered intermediate FBOs used
        in this method for drawing.
        * If ``self.profiling`` is True, this method also records
        various timing metrics for performance analysis into
        ``self.timings``.
        """
        # Don't use the guard wrapper here for performance reasons.
        if self._player_state != PlayerState.PLAYING:
            logging.warning(
                f"Cannot draw(), expected PLAYING state, got {self._player_state.name}."
            )
            return

        ts = self._threading_state

        # Read the index of the most recently completed worker frame.
        t0 = perf_counter()
        with ts.fbo_lock:
            present_idx = ts.present_fbo_idx
            worker_is_rendering = ts.worker_is_rendering
        timings[0] = perf_counter() - t0

        if worker_is_rendering:
            # Worker is mid-render: CPU-wait so we get the newest frame this cycle.
            t0 = perf_counter()
            ts.worker_render_done.wait()  # no timeout; GPU serializes after this
            with ts.fbo_lock:
                present_idx = ts.present_fbo_idx
            timings[1] += perf_counter() - t0

        if present_idx == -1:
            logging.error(
                "draw() called but no frame has been rendered yet — nothing to blit."
            )
            timings[2:] = 0
            return

        if ts.intermediate_fbos is None:
            logging.error(
                "draw() called but intermediate FBOs are not initialized — nothing to blit."
            )
            timings[2:] = 0
            return

        fbo_info = ts.intermediate_fbos[present_idx]

        # GPU-side wait: stall the GPU command queue (not the CPU) until the
        # worker's render into this FBO is complete.
        t0 = perf_counter()
        render_fence = ts.render_fences[present_idx]
        if render_fence is not None:
            gl.glWaitSync(render_fence, 0, gl.GL_TIMEOUT_IGNORED)
            gl.glDeleteSync(render_fence)
            ts.render_fences[present_idx] = None
        timings[2] = perf_counter() - t0

        # Snapshot NEXT_FRAME_INFO while we have a frame available.
        # move this to worker
        # t0 = perf_counter()
        # mpv._mpv_render_context_get_info(  # pylint: disable=E1101,W0212  # type: ignore
        #     self._mpv_render_ctx._handle,
        #     frameinfo,  # pylint: disable=W0212
        # )
        # timings[3] = perf_counter() - t0

        # Blit intermediate FBO → PsychoPy's target FBO.
        t0 = perf_counter()
        tw, th = self._target_fbo_info["w"], self._target_fbo_info["h"]
        utils.test_blit(
            (fbo_info["w"], fbo_info["h"]),
            (0, 0, tw, th),
            (tw, th),
            fbo_info["fbo"],
            self._target_fbo_info["fbo"],
        )
        timings[3] = perf_counter() - t0

        # Post a blit fence so the worker knows it's safe to write to this FBO
        # again once the blit is GPU-complete.
        t0 = perf_counter()
        ts.blit_fences[present_idx] = gl.glFenceSync(
            gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0
        )
        self._report_swap_on_next = True
        timings[4] = perf_counter() - t0

        gl.glFlush()

    def report_swap(self) -> None:
        if self._report_swap_on_next:
            self._mpv_render_ctx.report_swap()
            self._report_swap_on_next = False

    @property
    def state(self) -> PlayerState:
        return self._player_state

    @property
    def _mpv_state(self) -> PlayerState:
        # TODO: test
        if not hasattr(self, "_player") or self._player is None:
            return PlayerState.UNSPECIFIED
        if self._player.core_shutdown:
            return PlayerState.SHUTDOWN
        if self._player.pause:
            return PlayerState.PAUSED
        if self._player.idle_active:
            return PlayerState.IDLE
        if not self._player.core_idle:
            return PlayerState.PLAYING
        return PlayerState.UNSPECIFIED
