# MpvMovieStim Design

## Purpose

`MpvMoviestim` is intended to present libmpv-decoded video as a PsychoPy visual
stimulus. It delegates media decoding, audio playback, and video timing to mpv,
then transfers rendered video into PsychoPy's OpenGL target. The design prioritizes
predictable frame presentation and low overhead in PsychoPy's draw loop. It also enables a GPU-only decoding and presentation pipeline, when supported by HW.

## Components

- `MpvMoviestim` is the public PsychoPy stimulus class. It owns an mpv player,
  tracks playback state, maps stimulus geometry to a pixel rectangle, and draws
  video into the PsychoPy window.
- `MpvMoviestimState` exposes the stimulus's internal lifecycle states.
- `utils.py` provides OpenGL setup and resource helpers: resolving GL procedures,
  inspecting PsychoPy's framebuffer, creating textures and framebuffers, blitting,
  and creating a shared hidden pyglet window for a worker context.
- libmpv owns media decoding, audio, playback clocks, and video synchronization.
  The stimulus does not implement a decoder or its own playback clock.

The package exports `MpvMoviestim` and `MpvMoviestimState` from
`mpvmoviestim`.

## Dependencies

- Python >=3.10, <3.13, mirroring PsychoPy
- PsychoPy >=2026.2.5
- Pyglet >=1.4, <2, mirroring PsychoPy
- mpv >=1.0.8
- libmpv DLL (`libmpv-2.dll`) on the path

The implementation requires a pyglet-backed PsychoPy window and a current, compatible OpenGL context. It relies on pyglet's platform-specific GL procedure resolution and shared-context behavior.

GLFW backend support may be added in the future, but a quick check of PsychoPy code shows that even with a GLFW backend, PsychoPy relies on Pyglet's GL interface (confirm!).

## Rendering Design

### Push mode (double-buffered, two-threaded)

The constructor defaults to `draw_mode="double-buffered"` (to be renamed), which does the following:

1. The main thread creates a hidden pyglet window sharing the PsychoPy window's
OpenGL resources.
1. A render **worker** makes that context current, owns an mpv OpenGL render
context, and renders decoded frames into intermediate framebuffers.
1. The worker and main thread hand off two intermediate buffers. OpenGL sync
fences protect the rendered frame before the main thread reads it and protect
a buffer from being overwritten while a blit is still in flight.
1. `draw()` (which invokes `_draw_threaded()`) on the main thread blits the latest available buffer into PsychoPy's
current target framebuffer, and is responsible for flipping the intermediate
buffers. PsychoPy remains responsible for the screen flip. In case *worker* is in the process of rendering a new frame, `draw()` waits on the worker to finish and uses the newest frame available.

In this operation mode, libmpv **pushes** frames by calling `_mpv_update_callback()`. The callback wakes the worker, which then calls `render_ctx.update()` to flush mpv's message queue and renders any newly available frame.

LibMPV also has an *advanced control* mode that offers some advantages (faster CPU-GPU copies, for example). In advanced control mode, MPV can call the update callback for many reasons, not just to signal a new frame's availablility. Even if those messages aren't interesting for our use case, it is important to call `render_ctx.update()` soon after the callback to avoid stalling MPV's pipeline. Support for this mode is not implemented, but the double-buffered mode is design such that support should be trivial to add.

### Pull mode

`draw_mode="immediate"` (to be renamed) operates in **pull** mode, in which there is no separate worker loop that is activated on demand by MPV, instead,`draw()` checks `render_ctx.update()` as it runs on each screen refresh. This implementation is simple: it runs on a single thread, and every render and draw-related step is performed inside `draw()`.

This mode isn't capable of handling MPV's advanced control mode, which needs an update callback that can read MPV's message queue quickly.

Immediate mode has 2 planned modes of operation:

1. An intermediate buffer is used to store video frames when MPV has new ones ready. Video frames are blitted from the intermediate buffer to the screen.
2. Optionally, when *media framerate* >= *screen refresh rate*, then no intermediate buffer is used and libmpv is used to render directly into Psychopy's screen buffer. The intent is to keep the overhead minimal.

When the media framerate equals or exceeds that of the screen, MPV is expected to have a new video frame ready on each refresh cycle, so there's no reason to buffer images for later use. Testing is required to show that this actually more performant in practice. Which mode is used is ultimately controlled by the user, MpvMovieStim can recommend one.

### CPU and GPU-level synchronisation

*Push* mode is implemented with two threads that share certain CPU and GPU resources. Synchronisation is required both on CPU and GPU layers. For example, one thread writes to an intermediate buffer, the other reads from it. On the CPU level, simultaneous use can be prevented using locks and flags, but that's not sufficient as the threads are also separate and access shared resources on the GPU. In order to prevent one thread writing to an FBO/texture while the other is reading from it, GPU pipelines use synchronisation fences.

*Pull* mode doesn't need synchronisation as it runs on a single thread both on CPU and GPU.

Below is a table of synchronisation objects used:

| Name | Type | Purpose |
| --- | --- | --- |
| buffer_fbo_lock | threading.Lock | |

### Framebuffer and geometry

By design choice, MpvMovieStim uses `glBlitFrameBuffer` to draw images to the screen, instead of textured quads and shaders. This choice was meant for simplicity, and could change in the future. This operation is fast at copying axis-aligned rectangular pixels between buffers, but doesn't support custom image shapes, rotation, alpha blending or high-quality sampling. As a consequence, MpvMovieStim only supports axis-aligned rectangular images, and doesn't support rotation, opacity / alpha blending and some other operations.

MPV's video transformation methods are assumed to be high quality and performant, and produce better results than `glBlitFrameBuffer`. Mirroring (horizontal or vertical flipping) and clipping operations don't require resampling and don't degrade image quality even if performed by blit. Therefore, MpvMovieStim aims to rely on MPV to perform all video transformations, with the exception of mirroring and clipping.

Depending on the `useFBO` setting of the PsychoPy window, stimulus video data needs to be drawn either to the screen's backbuffer or to an FBO. MpvMovieStim identifies the render target at initialisation and determines its dimensions and (when detectable) pixel format. Intermediate buffers are allocated in the same pixel format and size.

Media display resizing is supported - smaller sizes can reuse the same allocated buffers, just occupy a smaller area in them. Larger display sizes require buffer re-allocation, which degrades performance. When users know the largest display size in advance, they can avoid buffer reallocation by specifying that size at instantiation, which ensures that buffers of the same size are allocated initially.

Position, size, anchor, horizontal/vertical flip, and unit conversion of the stimulus are handled by PsychoPy mixin classes (`BaseVisualStim` etc.). MpvMovieStim reads display pixel coordinates from these parent classes. Some of the properties defined in parent classes are overridden to ensure that non-supported operations have no effect (rotation, alpha threshold, arbitrary shapes, opacity), and that the stimulus display rectangle is immediately updated following changes.

## Playback and state

The state enum contains `UNSPECIFIED`, `NO_MEDIA`, `PAUSED`, `PLAYING`,
`EOF_REACHED`, and `SHUTDOWN`, with the following meaning:

| State | Meaning |
| --- | --- |
| UNSPECIFIED | Class initialisation, no MPV player available, or unknown state |
| NO_MEDIA | MPV core is active, no media file loaded |
| PAUSED | Media loaded, playback paused |
| PLAYING | Media loaded, playing |
| EOF_REACHED | Media loaded, EOF reached (MPV is paused due to pause-on-eof) |
| SHUTDOWN | MPV core has shut down |

General state flow:

1. MpvMovieStim starts in UNSPECIFIED state, starts an MPV player instance and sets NO_MEDIA.
2. `loadMovie()` is called either by user or by constructor if file was specified. It tells MPV to load the file, waits for MPV to reach paused state and supply media parameters. It sets the PAUSED state.
3. `play()` starts playback and sets PLAYING state. This is called automatically if `autoStart==True` is provided to the constructor.
4. Playback can be paused again with `pause()`, and started again with `play()`.
5. `stop()` is currently a terminal shutdown operation: it stops playback, drains and joins the worker in threaded mode, frees render resources, and quits mpv.
6. If the media is played to the end, MPV triggers the `_on_eof()` callback, which sets the EOF_REACHED state. MPV is configured with `keep-open==always`, and by default `keep-open-pause==True`, which causes it to enter paused state on EOF instead of terminating. The loaded media file remains seekable.

Repeated loading, restarting after stop, and playlist management are not currently established features.

## Frame timing

MPV has several frame timing strategies, configured by `video-sync`. Of interest to us are the ones that preserve audio-video sync without modifying the audio stream: `audio` and `display-vdrop`. The following is based on AI analysis of libmpv code (Sonnet 4.5 mostly):

- **audio** mode calculates video frame presentation times from time elapsed on the audio stream. It drops video frames when necessary.
- **Display-vdrop** also drops video frames, but its algorithm is more sophisticated in that it minimises presentation time jitter. For example, if the audio mode would display frames for 1-1-3 cycles, display-vdrop might smooth this to 1-2-2 (not real data). To do this, however, it needs accurate screen refresh rate information. This is usually supplied by the MPV video driver, but the driver we use to retrieve frames (`vo==libmpv`) has no control over or knowledge of refreshes. Our assumption is that we can supply the screen framerate in `display-fps-override`, and always call `render_ctx.report_swap()` immediately after PsychoPy `win.flip()`. If MPV notices that the information provided is not correct, it falls back to *audio* mode. In this operation mode, PTS of video frames will always show 0. We'll need to verify whether this mode works as assumed.

If `monitor_framerate` is defined at instantiation, MPV is configured to use display-vdrop mode. It is the user's responsibility to call `report_swap()` each time, immediately after `win.flip()`.

On MpvMovieStim's end, these vsync modes do not need special support. The *push* display mode is woken up by mpv whenever a new frame is available in either mode, and then the new frame is copied to an intermediate buffer. MPV takes care of frame timing, and only announces new frames with a small gap

## MPV settings

The following tables list MPV settings that are configured in the constructor:

### Video

| Setting | MpvMovieStim default | Required | Notes |
| --- | --- | --- | --- |
| `vo` | libmpv | yes | to use render context API |
| `gpu_api` | opengl | yes | openGL API for retrieving video frames |
| `keep-open` | always (yes may be acceptable with playlist support in the future) | yes | pause on EOF, don't terminate MPV when playlist is finished |
| `idle` | True | yes | don't terminate MPV when playlist is empty (may not be necessary?)
| `pause` | True | yes | start paused to allow pre-roll and retrieval of media information |
| `hwdec` | auto-safe | | enable hardware decoding when available |
| `video-sync` | display-vdrop or audio | | these modes don't alter the audio stream; some use cases may require other modes |
| `display-fps-override` | | | set to `monitor_framerate` in constructor to enable display-vdrop mode |
| `keepaspect` | False | | set to `keep_aspect_ratio` in constructor to use black bars and maintain video aspect ratio (True) or to stretch video (False) |

### Audio

| Setting | MpvMovieStim default | Required | Notes |
| --- | --- | --- | --- |
| audio_device | auto | | let MPV choose audio device |
| volume | 100 | | full volume; set to `volume` * 100 in constructor if specified |
| volume_gain | 0 | | another loudness setting, applied on top of volume |
| ao | null | | disables audio output; only set if `noAudio==True`, otherwise audio is enabled by default and this is not set |

Audio is enabled by default. `audio-exclusive` is currently commented out and not set by the constructor.

### Other

| Setting | MpvMovieStim default | Required | Notes |
| --- | --- | --- | --- |
| `log_handler` | `_mpv_log_fn()` | yes | redirect MPV log output to PsychoPy logging |
| `eof-reached` | `_on_eof()` | yes | this is set by `observe_proprerty()` to handle EOF events |

### User-defined

Users can supply MPV settings in the `mpv_options` dictionary argument of the constructor. These settings are merged into the settings dictionary, overwriting previous settings without any checks. This means that users need to be careful not to supply settings that are incompatible with MpvMovieStim, but is also allows power users to customise operation.

## Current Gaps and Risks

- Comments describe advanced control, resize behavior, profiling, and reusable
  playback, but these are not consistently present in the active public API or
  implementation. In particular, profiling tests and experimental scripts are
  not evidence that profiling is integrated into the active class.
- The checked-in tests include isolated helpers and exploratory scripts; they do
  not establish that the complete threaded playback, swap-timing, or shutdown
  paths work. Some test references also appear out of sync with the active module.

These are implementation gaps, not intended guarantees. The default mode and
basic API should be considered design intent until these mismatches are resolved
and end-to-end playback is validated.

## Open Design Questions

- Should `stop()` remain terminal, or should the player support unloading and loading another movie without rebuilding the stimulus?
- Playlist support?
- Change blit to textured quad drawing to support opacity, rotation, etc.?
- Does *advanced control* offer tangible benefits, and should support for it be implemented? The 2-threaded mode was already designed with this in mind, so probably no significant changes are required.
