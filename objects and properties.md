# Owned objects and properties

## Shared

- _state
  init, mpv init, load movie, start, pause, stop, eof

### Media

- _loaded_movie
- _media_size

### Presentation

- _monitor_framerate
- \_position, \_size, flip_*, \_draw_rect
- _autostart

### OpenGl

- _c_getproc

### MPV

- _player
- _mpv_render_ctx
_ _mpv_lib
- _mpv_options

### Psychopy

- _window
  init, mpv init, bounding rect (> make fbo, draw)
- _target_fbo_info

## Threaded-only

Basically everything in `ThreadingState`

- worker_thread
- shadow_window
- intermediate_fbo_*
- *_fbo_idx
- events: stop_worker, wakeup_worker, worker_init_done, worker_render_done
- sync flags: worker_is_rendering, buffer_flip_required
- buffer_fbo_lock
- sync fences: render_fences, blit_fences

- MpvMoviestim._threading_state
