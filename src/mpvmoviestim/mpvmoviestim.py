# threaded, uses intermediate FBO
# target: display-vdrop when report_swap used, otherwise audio sync

# Based on libmpv code analysis by sonnnet 4.6 & 5, libmpv has
# two display modes that don't modify the audio channel:
# a vsyncs_count display strategy where the display duration of
# vframes is optimised to minimise presentation delay jitter, and
# an audio-sync mode that only considers elapsed time and PTS to
# decide when to show the next vframe. The first strategy needs to
# be aware of screen refreshes, but vo=libmpv that we use here has
# no way to report vsyncs, so report_swap() has to be used. There's
# no point in trying to implement this algorithm ourselves, as we
# need the same information, and if we have that then we can just
# let mpv do the calculation. In this case we need to ask users
# to invoke report_swap() within each vsync cycle, preferably right
# after PsychoPy returns from flip. LibMPV caches ready vframes
# between vsync cycles, so it's efficient to ask it to re-render
# the same vframe on each cycle. This mode also requires the option
# display-fps-override (or something similar) to be set. We can use
# the value provided by PsychoPy.
# The audio sync mode doesn't need report_swap(), in fact,
# report_swap() should not be invoked at all. If MPV thinks that
# swaps are reported, it will wait on it, which will delay the
# render loop if report_swap is not used consistently. In this
# mode, vframes are only cached in a decoded but not presentation-
# ready state (no scaling, colour conversion, etc. done), so
# re-rendering via render_ctx_render incurs an additional cost.
# Because of this, it's better to just render into an intermediate
# FBO/tex and re-use that. If we allow rescales during playback,
# then we need to invalidate this intermediate FBO and let MPV
# do the scaling etc. again.

# Another mode to consider is advanced control mode. In some
# situations, it may be advantageous to support it. It seems that
# supporting it only requires update() being called immediately
# after mpv calls our update_callback(). While the non-advanced
# operating mode could be single-threaded (draw loop calls update,
# renders into FBO if new vframe available, draws to screen),
# because of the hope to support advanced control mode, let's stick
# to a threaded model.

# A threaded implementation uses a worker thread that handles
# update callbacks, calls mpv's update and renders into the
# intermediate FBO when new frames are available, while the main
# thread draw the newest available frame onto the screen (backbuffer
# or Psychopy's FBO if used). We need double buffering to avoid
# the threads waiting on each other. Buffer swaps should be done by
# the main thread, otherwise we need to ensure that the worker
# cannot ever snatch away a buffer that the main thread was just
# about to render.

# TODO: verify if locks/fences are guaranteed to resolve at some time - need timeout?
# TODO: import namedtuple, convert dataclass to namedtuple?

from __future__ import annotations

import ctypes
import dataclasses
import functools
import importlib
import threading
from array import array
from concurrent.futures import TimeoutError as FutureTimeoutError
from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from psychopy import logging, visual
from psychopy.tools.monitorunittools import convertToPix
from pyglet import gl

from . import pixel_format, utils
from .profiling import MS, WS, Profiler

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import mpv
    from pyglet.window import BaseWindow

TIMEOUT_DEFAULT_S = 5.0

# Dummy write target for profiling stamp sites when profiling is disabled;
# never indexed (guarded by base >= 0, which is always False in that case).
_EMPTY_TIMES = array("d")

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


__all__ = ["MpvMoviestim", "MpvMoviestimState"]

P = ParamSpec("P")
R = TypeVar("R")


def _log_pre_post(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log the player and MPV states before and after method execution."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        instance: MpvMoviestim = cast(MpvMoviestim, args[0])
        print(
            f"PRE {func.__name__}: "  # ty: ignore
            f"state={instance._state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        result = func(*args, **kwargs)
        print(
            f"POST {func.__name__}: "  # ty: ignore
            f"state={instance._state.name}; "  # pylint: disable=protected-access
            f"mpv={instance._mpv_state.name}"  # pylint: disable=protected-access
        )
        return result

    return wrapper


def _state_guard(
    allowed_state: MpvMoviestimState | list[MpvMoviestimState] | None = None,
    forbidden_state: MpvMoviestimState | list[MpvMoviestimState] | None = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator to enforce allowed and forbidden states when running methods."""

    allowed_states = (
        None
        if allowed_state is None
        else [allowed_state]
        if isinstance(allowed_state, MpvMoviestimState)
        else allowed_state
    )

    forbidden_states = (
        None
        if forbidden_state is None
        else [forbidden_state]
        if isinstance(forbidden_state, MpvMoviestimState)
        else forbidden_state
    )

    def decorator(func: Callable[..., None]) -> Callable[..., None]:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            instance: MpvMoviestim = args[0]
            actual_state = instance._state  # pylint: disable=protected-access

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
    "keep-open": "always",  # pause when reaching the end of the current file
    "pause": True,  # start paused
    "idle": True,  # do not quit when there is no file to play (needed for rewind?)
    # "wid": 0,  # do not create a new window (implied by other settings)
    # "keepaspect": False,  # (what does this do?)
    "video-sync": "display-vdrop",
}
# TODO: test if display-vdrop is OK to set here; not setting fps and not calling report_swap
#       should automatically revert back to audio sync mode. Check vframe PTSes to see if this
#       happens - in display sync the PTS is always 0, in audio sync it's the actual PTS.

_mpv_default_audio_options: dict[str, Any] = {
    "volume": 100,  # set volume to 100%
    "volume_gain": 0,  # another way to set loudness
    "audio_device": "auto",  # automatically choose audio output device
    # "audio-stream-silence": True,  # feeds ao silent audio even when paused
    # audio-stream-silence usage discouraged by manual
}


class MpvMoviestimState(Enum):
    """Internal state of MPVMoviesStim. Also used for mpv's states."""

    UNSPECIFIED = auto()  # not initialised or unknown state
    NO_MEDIA = auto()  # MPV core is active, no file loaded
    PAUSED = auto()  # file loaded but not playing
    PLAYING = auto()  # playing
    EOF_REACHED = auto()  # EOF reached
    SHUTDOWN = auto()  # MPV player core has shut down after quit()


@dataclasses.dataclass
class ThreadingState:
    """Synchronisation primitives and core objects related to the rendering worker thread.

    - The worker thread is responsible for rendering MPV vframes into intermediate FBOs.
    - The main thread is responsible for blitting from the intermediate FBOs to PsychoPy's window.
    - MPV supplies vframes some time before their PTS, therefore, future vframes need to be
      kept until their target time. We need 3 intermediate FBOs in total (triple bufferint):
        - 1 for main to blit from
        - 1 for the worker to render into
        - 1 may be needed to hold a future frame not presented yet
    - The worker thread owns the shadow GL context, MPV render context, and intermediate FBOs.

    Attributes
    ----------
    worker_thread : threading.Thread | None
        The thread object for the rendering worker.
    shadow_window : BaseWindow | None
        The shadow window that provides the GL context for the worker thread.
    intermediate_fbos : tuple[dict[str, int], dict[str, int]] | None
        The double-buffered intermediate FBOs used for rendering frames in the worker thread.
    intermediate_fbo_textures : tuple[int, int] | None
        Texture IDs for the intermediate FBOs, needed for cleanup.
    present_fbo_idx : int
        The index of the intermediate FBO that is currently ready for presentation (blitting).
    worker_fbo_idx : int
        The index of the intermediate FBO that the worker thread is currently rendering into.
    render_trigger : threading.Event
        Event set by the worker thread's MPV update callback to wake up the worker thread.
    stop_event : threading.Event
        Event set by the main thread to signal the worker thread to stop and exit.
    worker_init_done : threading.Event
        Event set by the worker thread once it has completed initialization.
    worker_is_rendering : bool
        Flag indicating whether the worker thread is currently rendering a frame. The main
        thread checks this flag to decide whether to wait for the current render to finish before
        blitting.
    worker_render_done : threading.Event
        Event set by the worker thread once it has finished rendering a frame to wake up the main
        thread if it is waiting for the render to finish before blitting.
    flip_required : bool
        Flag set by the worker thread to signal that the intermediate FBOs are ready to be flipped.
    fbo_lock : threading.Lock
        Lock to protect access to the intermediate FBO indices and rendering flag when flipping
        between buffers.
    render_fences : list[Any]
        List of OpenGL sync objects (fences) set by the worker thread after rendering a frame into
        an intermediate FBO, which the main thread waits on before blitting from that FBO. This is
        to ensure GPU-side synchronization.
    blit_fences : list[Any]
        List of OpenGL sync objects (fences) set by the main thread after blitting from an
        intermediate FBO, which the worker thread waits on before rendering a new frame into
        that FBO.
    profiler : Profiler | None
        The profiler instance when profiling is enabled, otherwise None. Created on the
        main thread before the worker starts; treated as read-only afterwards.
    """

    # Core objects
    worker_thread: threading.Thread | None = None
    shadow_window: BaseWindow | None = None

    # Double buffering
    intermediate_fbo_infos: tuple[dict[str, int], dict[str, int]] | None = None
    intermediate_fbo_textures: tuple[int, int] | None = None
    present_fbo_idx: int = -1
    worker_fbo_idx: int = 0

    # Synchronisation
    render_trigger: threading.Event = dataclasses.field(default_factory=threading.Event)
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    worker_init_done: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    worker_is_rendering: bool = False
    worker_render_done: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    buffer_flip_required: bool = False
    buffer_fbo_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    render_fences: list[Any] = dataclasses.field(default_factory=lambda: [None, None])
    blit_fences: list[Any] = dataclasses.field(default_factory=lambda: [None, None])

    # Profiling (None when disabled)
    profiler: Profiler | None = None


class MpvMoviestim:
    # PsychoPy
    _window: visual.Window
    _position: tuple[int | float, int | float]  # position in Psychopy units
    _size: tuple[int | float, int | float] | None  # size in Psychopy units
    _monitor_framerate: float | None  # display-vdrop sync mode active if provided
    flip_horizontal: bool
    flip_vertical: bool
    # Media
    _loaded_movie: Path
    _autostart: bool
    _media_size: tuple[int, int] | None
    _draw_rect: tuple[int, int, int, int] | None  # display rect (x, y, w, h)
    # Player core
    _state: MpvMoviestimState
    _threading_state: ThreadingState
    _prof: Profiler | None
    # MPV and OpenGL
    _mpv_lib: ModuleType
    _player: mpv.MPV
    _mpv_options: dict[str, Any]
    # TODO: _advanced_control: bool
    _c_getproc: ctypes._CFunctionType
    _mpv_render_ctx: mpv.MpvRenderContext
    _target_fbo_info: dict[str, int]  # mpv.MpvOpenGLFBO
    # _report_swap: bool

    def __init__(
        self,
        window: visual.Window,
        file: Path | str,
        autoStart: bool = False,
        noAudio: bool = False,
        volume: float = 1.0,
        monitor_framerate: float | None = None,
        mpv_options: dict[str, Any] | None = None,
        pos: tuple[int | float, int | float] = (0, 0),
        size: tuple[int | float, int | float] | None = None,
        flipHoriz: bool = False,
        flipVert: bool = False,
        profiling: bool = False,
        profiling_capacity: int = 10_000,
        # TODO: advanced_control: bool = True,
        # TODO: verify audio/display-vdrop modes
    ):
        self._mpv_options = _mpv_default_options.copy()
        # add log handler
        self._mpv_options.update({"log_handler": self._mpv_log_fn, "loglevel": "info"})
        # add audio options
        if noAudio:
            self._mpv_options["ao"] = "null"
        else:
            self._mpv_options.update(_mpv_default_audio_options)
            self._mpv_options["volume"] = (
                0 if volume < 0 else 100 if volume > 1 else int(volume * 100)
            )
        # if monitor fps is provided, try display-vdrop video sync mode
        # user can get fps with: window.getActualFrameRate()
        if monitor_framerate is not None:
            self._monitor_framerate = monitor_framerate
            self._mpv_options["video-sync"] = "display-vdrop"
        else:
            self._monitor_framerate = None
            self._mpv_options["video-sync"] = "audio"
        # apply user-specified MPV options
        if mpv_options is not None:
            self._mpv_options.update(mpv_options)

        self._window = window
        self._size = size
        self._position = pos
        self._autostart = autoStart
        self.flip_horizontal = flipHoriz
        self.flip_vertical = flipVert
        # self._advanced_control = advanced_control

        self._state = MpvMoviestimState.UNSPECIFIED
        self._media_size = None
        self._draw_rect = self._bounding_rect(size, None, pos, window)
        logging.info(f"setting draw rect to: {self._draw_rect}")

        # self._report_swap = False

        # Synchronisation primitives — must exist before worker thread starts.
        self._threading_state = ThreadingState()

        # Profiling — must be created before the worker thread starts.
        if profiling:
            self._prof = Profiler(capacity=profiling_capacity)
            self._threading_state.profiler = self._prof
        else:
            self._prof = None

        # lazy load MPV
        self._mpv_lib = importlib.import_module("mpv")

        self._init_mpv_player()
        self.loadMovie(file)

    @_log_pre_post
    @_state_guard(allowed_state=MpvMoviestimState.UNSPECIFIED)
    def _init_mpv_player(self) -> None:

        # create MPV player instance
        try:
            self._player = self._mpv_lib.MPV(**self._mpv_options)
            self._player.observe_property("eof-reached", self._on_eof)

            # Build the GL proc-address resolver (a plain function pointer —
            # no GL context required yet).
            self._c_getproc = self._mpv_lib.MpvGlGetProcAddressFn(
                utils.get_proc_address
            )

            # Determine the best pixel format to render to.
            # Must happen while the main window's context is current.
            inferred_fbo_info = pixel_format.get_psychopy_fbo_info(self._window)
            format_name = (
                "<undetermined>"
                if "internal_format" not in inferred_fbo_info
                else pixel_format.get_internal_format_name(
                    inferred_fbo_info["internal_format"]
                )
            )
            logging.info(
                f"Psychopy's rendering FBO: fbo={inferred_fbo_info['fbo']}, "
                f"size={inferred_fbo_info['w']}x{inferred_fbo_info['h']}, "
                f"format={format_name}"
            )
            self._target_fbo_info = inferred_fbo_info

            # TODO: direct rendering / no buffer when media > screen fps, worker thread only update_cb

            # Create the shadow window (shared context) on the main thread so
            # that its DC/surface is set up before the worker uses it.
            # Pass winHandle (the underlying pyglet window) so utils.py does not
            # depend on PsychoPy.
            logging.info("Creating shadow window for context sharing.")
            self._threading_state.shadow_window = utils.create_shadow_window(
                self._window.winHandle
            )

            # Start worker — it takes the shadow context, creates the render
            # context and the double-buffered intermediate FBOs, then signals
            # worker_init_done.
            logging.info("Instantiating and starting renderer worker thread.")
            self._threading_state.worker_thread = threading.Thread(
                target=self._render_worker, name="mpv-render-worker", daemon=True
            )
            self._threading_state.worker_thread.start()
            logging.info("Waiting for worker thread to initialise.")
            if not self._threading_state.worker_init_done.wait(
                timeout=TIMEOUT_DEFAULT_S
            ):
                raise TimeoutError(
                    "Timed out waiting for render worker thread to initialise."
                )
            logging.info("Finished waiting for worker thread.")
            # set IDLE state, meaning core is active, file not loaded
            self._state = MpvMoviestimState.NO_MEDIA
        except Exception as e:
            logging.error(f"Failed to initialize MPV player: {e}")
            raise

    @staticmethod
    def _bounding_rect(
        size: tuple[float, float] | None,
        media_size: tuple[float, float] | None,
        position: tuple[float, float],
        window: visual.Window,
    ) -> tuple[int, int, int, int] | None:
        """Calculate pixel-based bounding rectangle from Psychopy-based size and position.

        Psychopy's non-pixel units and are centre-based position reference are converted
        to screen pixels.

        Notes
        -----
        - Display bounding rect (x, y, w, h in window pixels) is needed for OpenGL blit.
        - Position = centre of media element relative to window centre; size = width and height.
        - Psychopy's window size is in pixels; position and size can be in any Psychopy units.
        - Units other than pixels are converted using Psychopy's `convertToPix` function.
        - If display size is not given, media size is used if available.
        - If neither size nor media size are available, None is returned.
        """
        logging.info(
            f"_bounding rect params: {size=}, {media_size=}, {position=}, {window=}"
        )

        if size is None and media_size is None:
            logging.info(
                "Size not specified and media size not available yet, not updating bounding rect."
            )
            return None
        # The above guard is supposed to prevent both media size and size being None at the same time,
        # but type checkers don't seem to understand this logic, so additional asserts were needed
        # below to silence errors.

        screen_centre_px: tuple[int, int] = (
            window.size[0] / 2,
            window.size[1] / 2,
        )

        # if units are not pix, convert to pixels what's necessary
        if window.units != "pix":
            if size is not None:
                # if display size is provided, calculate bounding rect directly
                # get vectors from screen centre to bottom-left/top-right of media in pixels
                # we directly calculate corner positions to allow for non-rectangular units
                corners = [
                    (
                        position[0] - size[0] / 2,
                        position[1] - size[1] / 2,
                    ),
                    (
                        position[0] + size[0] / 2,
                        position[1] + size[1] / 2,
                    ),
                ]
                bottom_left_px, top_right_px = cast(
                    tuple[tuple[float, float], tuple[float, float]],
                    convertToPix(
                        pos=[0, 0],
                        vertices=corners,
                        units=window.units,
                        win=window,
                    ),
                )
            else:
                assert media_size is not None  # guaranteed, silences errors
                # display size not provided, only convert position to px
                # pos_px: screen centre -> media element centre vector in pixels
                pos_px: tuple[float, float] = convertToPix(
                    pos=[0, 0],
                    vertices=[position],
                    units=window.units,
                    win=window,
                )[0]
                bottom_left_px = (
                    pos_px[0] - media_size[0] / 2,
                    pos_px[1] - media_size[1] / 2,
                )
                top_right_px = (
                    pos_px[0] + media_size[0] / 2,
                    pos_px[1] + media_size[1] / 2,
                )
            # bounding rect absolute coordinates = screen centre position +  corner vectors
            bounding_rect = (
                int(bottom_left_px[0] + screen_centre_px[0]),
                int(bottom_left_px[1] + screen_centre_px[1]),
                int(top_right_px[0] - bottom_left_px[0]),
                int(top_right_px[1] - bottom_left_px[1]),
            )
        else:
            # Everything is in pixels, we only need to decide what display size to use
            size_px = size if size is not None else media_size
            assert size_px is not None
            bounding_rect = (
                int(screen_centre_px[0] + position[0] - size_px[0] / 2),
                int(screen_centre_px[1] + position[1] - size_px[1] / 2),
                int(size_px[0]),
                int(size_px[1]),
            )
        return bounding_rect

    def _make_intermediate_fbo(self) -> tuple[dict[str, int], int]:
        """Allocate one intermediate FBO + backing texture on the current GL context.

        Must be called while the shadow (worker) context is current.

        We only allocate intermediate FBOs/textures once at the start of the worker thread
        to avoid doing any GL resource allocation during playback. The size of the FBO
        is the larger of the window size and the media display size. During playback,
        only resize operations to a size equal to or smaller than this are supported.
        It is, therefore, recommended to initially set the display to the largest
        expected size.

        Returns
        -------
        dict[str, int]
            fbo info
        int
            texture id
        """
        w = max(
            self._target_fbo_info["w"],
            self._draw_rect[2] if self._draw_rect is not None else 0,
        )
        h = max(
            self._target_fbo_info["h"],
            self._draw_rect[3] if self._draw_rect is not None else 0,
        )
        internal_format = self._target_fbo_info.get(
            "internal_format", pixel_format.default_pixel_format
        )
        tex_id = utils.create_texture(w, h, internal_format)
        fbo_id = utils.create_fbo(tex_id)
        info: dict[str, int] = {
            "fbo": fbo_id,
            "w": w,
            "h": h,
            "internal_format": internal_format,
        }
        logging.info(
            f"Intermediate FBO created: {info}; texture: {tex_id}; "
            f"internal format: {pixel_format.get_internal_format_name(internal_format)}"
        )
        return info, tex_id

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _mpv_update_callback(self) -> None:
        """Called by MPV when a new frame may be ready.

        This function needs to be minimal so that it doesn't cause any
        lag. If advanced control mode is enabled, MPV's update()
        must be called back without delay otherwise it stalls MPV.
        As per MPV API, render_ctx_update cannot be called from the
        update_cb, and should be called from the render thread.
        """
        ts = self._threading_state
        if ts.profiler is not None:
            # (timestamp, is_trigger): is_trigger marks the callback that
            # transitioned render_trigger from unset to set, i.e. the one that
            # actually woke the worker (best effort, subject to check/set race).
            ts.profiler.wake_times.append(
                (perf_counter(), not ts.render_trigger.is_set())
            )
        ts.render_trigger.set()

    def _render_worker(self) -> None:
        """Worker thread: render new frames into intermediate buffers

        Owns the shadow GL context and MPV render context.
        Responsible for drawing new frames into an intermediate FBO

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
        if ts.profiler is not None:
            profiler_cpu = ts.profiler.worker
            profiler_gpu = ts.profiler.worker_gpu
            buf = profiler_cpu.buf

        # --- one-time init on this thread ---
        if ts.shadow_window is None:
            raise ValueError("shadow_window must be set before the worker starts")
        # TODO: error type to raise
        utils.make_context_current(ts.shadow_window)
        # ts.shadow_window.switch_to()

        # Log which renderer this context sees (sanity check that sharing works).
        renderer = gl.glGetString(gl.GL_RENDERER)
        logging.info(
            f"MPV render worker: GL_RENDERER = {renderer if renderer else '<unknown>'}"
        )

        self._mpv_render_ctx = self._mpv_lib.MpvRenderContext(
            self._player,
            "opengl",
            opengl_init_params={"get_proc_address": self._c_getproc},
            # TODO: advanced_control=self._advanced_control,
        )
        self._mpv_render_ctx.update_cb = self._mpv_update_callback

        # Double-buffered intermediate FBOs (created on the shadow context).
        fbo1, tex1 = self._make_intermediate_fbo()
        fbo2, tex2 = self._make_intermediate_fbo()
        ts.intermediate_fbo_infos = (
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
            ts.render_trigger.wait()  # wait until the update callback is called
            ts.render_trigger.clear()

            base = -1 if profiler_cpu is None else profiler_cpu.next_iter()
            profiling_enabled = base >= 0
            if profiling_enabled:
                assert profiler is not None and profiler_gpu is not None
                profiler.drain_wakes(buf, base)
                profiler_gpu.collect()  # harvest GPU results from previous iterations
                buf[base + WS.UPDATE_T0] = perf_counter()

            # drain pending MPV updates
            update_result = self._mpv_render_ctx.update()
            if profiling_enabled:
                buf[base + WS.UPDATE_T1] = perf_counter()

            # stop only after draining updates to avoid hanging MPV core (necessary?)
            if ts.stop_event.is_set():
                break

            # Ignore callbacks when there is no new frame to render
            # (can happen when advanced control is enabled)
            if not update_result:
                continue

            # Set busy flag to tell main not to flip buffers while we're rendering.
            # Main will also delay its blit until this is done so that it can show
            # the newest frame. We essentially never need to wait for main as it only
            # keeps the lock while flipping buffers, and we don't need to know whether
            # the buffer we render to is the same as before or the other.
            ts.worker_render_done.clear()
            if profiling_enabled:
                buf[base + WS.LOCK_T0] = perf_counter()
            with ts.buffer_fbo_lock:
                ts.worker_is_rendering = True
                target_idx = ts.worker_fbo_idx
            if profiling_enabled:
                buf[base + WS.LOCK_T1] = perf_counter()

            if ts.intermediate_fbo_infos is None:
                raise ValueError("intermediate_fbos must be set during worker init")
            # TODO: better error type?
            fbo_info = ts.intermediate_fbo_infos[target_idx]

            # GPU-side: wait until the main thread has finished blitting from
            # this FBO before we overwrite it.
            blit_fence = ts.blit_fences[target_idx]
            if blit_fence is not None:
                if profiling_enabled:
                    buf[base + WS.WAITSYNC_BLIT_T0] = perf_counter()
                    # zero-timeout poll: was the main thread's blit already done?
                    st = gl.glClientWaitSync(blit_fence, 0, 0)
                    buf[base + WS.WAITSYNC_BLIT_STATE] = (
                        1.0
                        if st in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED)
                        else 2.0
                        if st == gl.GL_TIMEOUT_EXPIRED
                        else 3.0
                    )
                gl.glWaitSync(blit_fence, 0, gl.GL_TIMEOUT_IGNORED)
                gl.glDeleteSync(blit_fence)
                ts.blit_fences[target_idx] = None
                if profiling_enabled:
                    buf[base + WS.WAITSYNC_BLIT_T1] = perf_counter()

            if profiling_enabled:
                assert profiler_gpu is not None
                buf[base + WS.RENDER_T0] = perf_counter()
                profiler_gpu.begin()
            self._mpv_render_ctx.render(
                opengl_fbo=fbo_info,
                flip_y=True,
                block_for_target_time=False,
            )
            if profiling_enabled:
                profiler_gpu.end(base)
                buf[base + WS.RENDER_T1] = perf_counter()

            # Post a fence so the main thread can wait for this render to finish
            # before it blits.
            ts.render_fences[target_idx] = gl.glFenceSync(
                gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0
            )
            # it appears that flushing after fencesync is expected (need to check spec)
            gl.glFlush()

            done_fence = None
            if profiling_enabled:
                buf[base + WS.FENCE_POST_T] = perf_counter()
                # Private fence for the end-of-iteration GPU-done wait: issued at
                # the same command-stream point as the shared render fence, but
                # owned exclusively by the worker so there is no race with the
                # main thread's glDeleteSync on the shared fence.
                done_fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

            # Hand off: publish which FBO is ready, clear rendering flag, flip index.
            with ts.buffer_fbo_lock:
                ts.worker_is_rendering = False
                ts.present_fbo_idx = target_idx
                ts.worker_fbo_idx = 1 - target_idx
            if profiling_enabled:
                buf[base + WS.FLIP_T] = perf_counter()

            # After lock release so present_fbo_idx is committed before draw() reads it.
            ts.worker_render_done.set()
            if profiling_enabled:
                buf[base + WS.SET_DONE_T] = perf_counter()
                # End-of-iteration marker: block until the GPU has finished this
                # iteration's render. Placed AFTER worker_render_done.set() so the
                # main thread is never delayed by this profiling-only wait.
                assert done_fence is not None
                tw0 = perf_counter()
                st = gl.glClientWaitSync(
                    done_fence, gl.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000
                )
                gl.glDeleteSync(done_fence)
                tw1 = perf_counter()
                buf[base + WS.GPU_WAIT] = (
                    tw1 - tw0
                    if st in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED)
                    else -1.0
                )
                buf[base + WS.ITER_DONE_T] = tw1

        # --- cleanup (shadow context still current on this thread) ---
        if profiler_gpu is not None:
            # blocking drain so the final iterations' GPU durations are recorded
            profiler_gpu.drain_blocking()
        self._mpv_render_ctx.free()
        if ts.intermediate_fbo_textures is not None:
            for tex in ts.intermediate_fbo_textures:
                utils.destroy_texture(tex)
        if ts.intermediate_fbo_infos is not None:
            for fbo in ts.intermediate_fbo_infos:
                utils.destroy_fbo(fbo["fbo"])
        utils.release_context()

    def _mpv_log_fn(self, level: int, prefix: str, text: str) -> None:
        # print(f"MPV: {level=}, {prefix=}, {text=}")
        logging.exp(f"MPV({prefix}): {text}")

    @_log_pre_post
    def _on_eof(self, prop_name, value) -> None:
        # value -> None at init
        # value -> False when playback starts
        # value -> True at EOF
        # with keep-open set, player pauses instead of closing file
        print(f"on eof: property {prop_name} changed to {value}")
        assert prop_name == "eof-reached", (
            "_on_eof called with prop other than eof-reached"
        )
        if value:
            logging.exp("EOF reached")
            self._state = MpvMoviestimState.EOF_REACHED

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

    @_log_pre_post
    @_state_guard(
        forbidden_state=[MpvMoviestimState.UNSPECIFIED, MpvMoviestimState.SHUTDOWN]
    )
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
        logging.exp("Loading movie file")
        self._player.loadfile(filename=str(file), mode="replace")
        self._loaded_movie = file
        logging.exp("waiting until paused")
        self._mpv_wait(
            lambda: self._player.wait_until_paused(timeout=TIMEOUT_DEFAULT_S),
            f"Timed out waiting for MPV to reach a paused state after loading '{file}'.",
        )
        self._state = MpvMoviestimState.PAUSED

        logging.exp("waiting for video params property")
        self._mpv_wait(
            lambda: self._player.wait_for_property(
                "video-params", timeout=TIMEOUT_DEFAULT_S
            ),
            "Timed out waiting for 'video-params' property after loading movie.",
        )
        video_params = self._player.video_params
        self._media_size = video_params["w"], video_params["h"]
        if self._draw_rect is None:
            self._draw_rect = self._bounding_rect(
                self._size, self._media_size, self._position, self._window
            )
        logging.info(f"loadmovie: setting draw rect to: {self._draw_rect}")

        logging.exp(f"Loaded movie '{file}' with autostart set to {self._autostart}.")
        if self._autostart:
            self.play()

    def load(self, fileName: Path | str) -> None:
        """Alias for loadMovie()"""
        self.loadMovie(fileName)

    @_log_pre_post
    @_state_guard(allowed_state=MpvMoviestimState.PAUSED)
    def play(self, block: bool = False) -> None:
        # TODO: threading state needs to be reset on play() following stop() or loadMovie()
        # self._report_swap = False
        self._player.pause = False
        if block:
            self._mpv_wait(
                lambda: self._player.wait_until_playing(timeout=TIMEOUT_DEFAULT_S),
                "Timed out waiting for MPV to start playing.",
            )
        self._state = MpvMoviestimState.PLAYING
        logging.exp(
            "State change: PAUSED -> PLAYING"
            f"(self state={self._state.name}, MPV state={self._mpv_state.name})"
        )

    @_log_pre_post
    @_state_guard(allowed_state=MpvMoviestimState.PLAYING)
    def pause(self, block: bool = False) -> None:
        self._player.pause = True
        if block:
            self._mpv_wait(
                lambda: self._player.wait_until_paused(timeout=TIMEOUT_DEFAULT_S),
                "Timed out waiting for MPV to pause.",
            )
        self._state = MpvMoviestimState.PAUSED
        logging.exp(
            "State change: PLAYING -> PAUSED"
            f"(self state={self._state.name}, MPV state={self._mpv_state.name})"
        )

    @_log_pre_post
    @_state_guard(
        allowed_state=[
            MpvMoviestimState.PAUSED,
            MpvMoviestimState.PLAYING,
            MpvMoviestimState.EOF_REACHED,
        ]
    )
    def stop(self) -> None:
        self._state = MpvMoviestimState.SHUTDOWN
        logging.exp(f"frame-drop-count: {self._player.frame_drop_count}")

        # Ask MPV to stop the current file and wait until it is idle.
        self._player.stop()
        # self._player.terminate()
        # self._player.wait_for_shutdown()
        # self._player.stop()
        logging.info("idle-active wait")
        self._mpv_wait(
            lambda: self._player.wait_for_property(
                "idle-active", timeout=TIMEOUT_DEFAULT_S
            ),
            "Timed out waiting for MPV to report idle-active during stop().",
        )

        # Drain the render worker BEFORE quitting the MPV core.  With
        # advanced_control=True a permanent hang results if we call
        # wait_for_shutdown() while the worker is still alive.
        logging.info("telling worker thread to stop")
        ts = self._threading_state
        ts.stop_event.set()
        ts.render_trigger.set()  # wake worker if it is blocked on .wait()
        assert ts.worker_thread is not None, (
            "worker_thread must be set before stop() is called"
        )
        logging.info("waiting for worker thread to finish (join)")
        ts.worker_thread.join(timeout=TIMEOUT_DEFAULT_S)
        if ts.worker_thread.is_alive():
            logging.error(
                "Render worker did not stop within timeout; it is likely stuck in a "
                "blocking GL/mpv call. Aborting further shutdown to avoid racing "
                "player.quit()/context.free() against the still-running worker."
            )
            raise RuntimeError("mpv render worker failed to stop")

        # Drain pending GPU timer queries on the main context so the final
        # iterations' GPU durations are recorded (context still current here).
        if self._prof is not None:
            try:
                self._prof.main_gpu.drain_blocking()
            except Exception as e:
                logging.info(f"profiling: main GPU timer drain failed: {e}")

        # Only safe to reach here once the worker has actually exited and freed
        # _mpv_render_ctx / FBOs itself.
        logging.info("quitting mpv")
        self._player.quit()
        logging.info("waiting for mpv shutdown")
        self._mpv_wait(
            lambda: self._player.wait_for_shutdown(timeout=TIMEOUT_DEFAULT_S),
            "Timed out waiting for MPV core shutdown.",
        )
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
        * If profiling was enabled at construction, this method also records
        per-iteration timestamps; retrieve them with ``get_profiling_data()``.
        """
        # Don't use the guard wrapper here for performance reasons.
        if self._state != MpvMoviestimState.PLAYING:
            logging.warning(
                f"Cannot draw(), expected PLAYING state, got {self._state.name}."
            )
            return

        ts = self._threading_state

        prof = self._prof
        rec = prof.main if prof is not None else None
        base = -1 if rec is None else rec.next_iter()
        buf = rec.buf if rec is not None else _EMPTY_TIMES
        profiling_enabled = base >= 0
        if profiling_enabled:
            assert prof is not None
            prof.main_gpu.collect()  # harvest GPU results from previous iterations
            buf[base + MS.DRAW_ENTRY_T] = perf_counter()

        # Read the index of the most recently completed worker frame.
        if profiling_enabled:
            buf[base + MS.LOCK_T0] = perf_counter()
        with ts.buffer_fbo_lock:
            worker_is_rendering = ts.worker_is_rendering
            # flip buffers is necessary and worker isn't using them
            if not worker_is_rendering and ts.buffer_flip_required:
                ts.present_fbo_idx, ts.worker_fbo_idx = (
                    ts.worker_fbo_idx,
                    1 - ts.worker_fbo_idx,
                )
                ts.buffer_flip_required = False
        if profiling_enabled:
            buf[base + MS.LOCK_T1] = perf_counter()

        # if worker is mid-render: CPU-wait so we get the newest frame this cycle
        if worker_is_rendering:
            if profiling_enabled:
                buf[base + MS.CPU_WAIT_T0] = perf_counter()
            ts.worker_render_done.wait()
            if profiling_enabled:
                buf[base + MS.CPU_WAIT_T1] = perf_counter()
            with ts.buffer_fbo_lock:
                # flip buffers, worker is not using them for sure now
                if ts.buffer_flip_required:
                    ts.present_fbo_idx, ts.worker_fbo_idx = (
                        ts.worker_fbo_idx,
                        1 - ts.worker_fbo_idx,
                    )
                    ts.buffer_flip_required = False

        present_idx = ts.present_fbo_idx  # only main writes this; safe outside lock

        if present_idx == -1:
            logging.error("draw() called but no frame has been rendered yet.")
            return

        if ts.intermediate_fbo_infos is None:
            logging.error("draw() called but intermediate FBOs are not initialized.")
            return

        if self._draw_rect is None:
            logging.error("draw() called but draw rectangle is undefined.")
            return

        fbo_info = ts.intermediate_fbo_infos[present_idx]

        # GPU-side wait: stall the GPU command queue (not the CPU) until the
        # worker's render into this FBO is complete.
        if (render_fence := ts.render_fences[present_idx]) is not None:
            if profiling_enabled:
                buf[base + MS.WAITSYNC_RENDER_T0] = perf_counter()
                # zero-timeout poll: was the worker's render already done?
                st = gl.glClientWaitSync(render_fence, 0, 0)
                buf[base + MS.WAITSYNC_RENDER_STATE] = (
                    1.0
                    if st in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED)
                    else 2.0
                    if st == gl.GL_TIMEOUT_EXPIRED
                    else 3.0
                )
            gl.glWaitSync(render_fence, 0, gl.GL_TIMEOUT_IGNORED)
            gl.glDeleteSync(render_fence)
            ts.render_fences[present_idx] = None
            if profiling_enabled:
                buf[base + MS.WAITSYNC_RENDER_T1] = perf_counter()

        # Snapshot NEXT_FRAME_INFO while we have a frame available.
        # move this to worker
        # mpv._mpv_render_context_get_info(  # pylint: disable=E1101,W0212  # type: ignore
        #     self._mpv_render_ctx._handle,
        #     frameinfo,  # pylint: disable=W0212
        # )

        # Blit intermediate FBO → PsychoPy's target FBO.
        if profiling_enabled:
            assert prof is not None
            buf[base + MS.BLIT_T0] = perf_counter()
            prof.main_gpu.begin()
        utils.blit_with_draw_rect(
            self._draw_rect,
            fbo_info["fbo"],
            self._target_fbo_info["fbo"],
            self.flip_horizontal,
            self.flip_vertical,
        )
        if profiling_enabled:
            prof.main_gpu.end(base)
            buf[base + MS.BLIT_T1] = perf_counter()
        # x, y, w, h = self._draw_rect
        # utils.test_blit(
        #     (w, h),
        #     (x, y, w, h),
        #     (w, h),
        #     fbo_info["fbo"],
        #     self._target_fbo_info["fbo"],
        # )

        # Post a blit fence so the worker knows it's safe to write to this FBO
        # again once the blit is GPU-complete.
        ts.blit_fences[present_idx] = gl.glFenceSync(
            gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0
        )
        gl.glFlush()

        if profiling_enabled:
            buf[base + MS.FENCE_POST_T] = perf_counter()
            # End-of-iteration marker: block until the GPU has finished the blit.
            # Private fence (owned exclusively by the main thread) to avoid
            # racing the worker's waits on the shared blit fence.
            done_fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            tw0 = perf_counter()
            st = gl.glClientWaitSync(
                done_fence, gl.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000
            )
            gl.glDeleteSync(done_fence)
            tw1 = perf_counter()
            buf[base + MS.GPU_WAIT] = (
                tw1 - tw0
                if st in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED)
                else -1.0
            )
            buf[base + MS.ITER_DONE_T] = tw1
            buf[base + MS.DRAW_EXIT_T] = perf_counter()

        # self._report_swap = True

    @_state_guard(allowed_state=MpvMoviestimState.PLAYING)
    def report_swap(self) -> None:
        """Report a screen buffer swap to MPV.

        Notes
        -----
        Based on Claude's analysis of libMPV code and the manual, report_swap()
        must be called right after each flip if we want to use libMPV's
        display-based video sync modes (display-vdrop in particular), as
        the vo=libMPV driver has no control over or information of the actual
        swap events. LibMPV uses these events to track the screen's frame rate.
        Calls to report_swap() must be consistent - if even called once, MPV
        expects it to be called on each screen buffer swap, otherwise its
        calculation will be off. If the swap is not reported at all, or if
        MPV detects a long timeout (>200ms, I believe), it reverts to audio-
        based sync mode. Therefore, the best practice is to either always
        call report_swap() after each flip if one wishes to use display-vdrop
        mode, or to not call it ever if audio-sync is to be used.
        Ensuring this is entirely left to the user.
        """
        # if self._report_swap:
        rec = self._prof.main if self._prof is not None else None
        if rec is not None and rec.active and rec.base >= 0:
            # stamp into the current draw() row (base advances on the next draw)
            rec.buf[rec.base + MS.REPORT_SWAP_T] = perf_counter()
        self._mpv_render_ctx.report_swap()
        # self._report_swap = False

    def get_profiling_data(self) -> list[tuple[float, str, str, float]]:
        """Return the merged, timestamp-sorted profiling event list.

        Each event is ``(t_rel_s, thread, event_name, duration_s)``; instants
        have duration 0.0. Returns an empty list if profiling is disabled.
        See the ``profiling`` module docstring for event semantics.
        """
        if self._prof is None:
            return []
        return self._prof.get_events()

    def export_profiling_csv(self, path: Path | str) -> None:
        """Export the merged profiling timeline to CSV
        (columns: t_ms, thread, event, duration_ms). No-op if profiling is off.
        """
        if self._prof is not None:
            self._prof.export_csv(path)

    @property
    def state(self) -> MpvMoviestimState:
        return self._state

    @property
    def _mpv_state(self) -> MpvMoviestimState:
        # TODO: test
        if not hasattr(self, "_player") or self._player is None:
            return MpvMoviestimState.UNSPECIFIED
        if self._player.core_shutdown:
            return MpvMoviestimState.SHUTDOWN
        if self._player.pause:
            return MpvMoviestimState.PAUSED
        if self._player.idle_active:
            return MpvMoviestimState.NO_MEDIA
        if not self._player.core_idle:
            return MpvMoviestimState.PLAYING
        if self._player.eof_reached:
            return MpvMoviestimState.EOF_REACHED
        return MpvMoviestimState.UNSPECIFIED

    def _mpv_wait(
        self,
        call: Callable[[], Any],
        error_message: str,
    ) -> Any:
        """Run an MPV blocking wait call, converting timeouts to `TimeoutError`,
        and re-raising mpv.ShutdownError on a player shutdown while waiting.

        Parameters
        ----------
        call : Callable[[], Any]
            A zero-argument callable wrapping the actual mpv wait call, e.g.
            ``lambda: player.wait_until_paused(timeout=5.0)``.
        error_message : str
            Message to use when raising `TimeoutError` on timeout.

        Returns
        -------
        Any
            The return value of `call`, if any.

        Raises
        ------
        TimeoutError
            If the wait times out (covers both the builtin `TimeoutError` and
            `concurrent.futures.TimeoutError`, which are distinct classes prior
            to Python 3.11).
        mpv.ShutdownError
            Python-mpv raises this exception if the player is shut down during
            the wait. This error is masked then waiting for shutdown, though.
        """
        try:
            return call()
        except (TimeoutError, FutureTimeoutError) as e:
            raise TimeoutError(error_message) from e
        except self._mpv_lib.ShutdownError as e:
            raise RuntimeError("MPV core shut down unexpectedly while waiting.") from e
