# Buffer management design (`buffer.py`)

Design intent for the intermediate frame buffers shared by the render worker and
the draw thread. It records the goals, the decisions and the reasons behind them,
so later changes can tell what is essential and what is incidental.

## Context

Two threads exchange video frames through intermediate buffers (texture + FBO):

- **P (producer)**: the render worker. libmpv pushes frames at its own pace, and
  P renders each one into a buffer as soon as it can, so it never stalls mpv.
- **C (consumer)**: the draw thread (PsychoPy's `draw()`). Once per screen refresh
  it draws the freshest finished frame into PsychoPy's target.

Both threads have their own OpenGL context in the same share group (via pyglet's
shared hidden window). Textures, FBOs and sync objects are shared between them.

## Goals

In order of priority:

1. **Thread safety on CPU and GPU.** A buffer is never written while it is read,
   neither by the CPU-side bookkeeping nor by queued GPU commands.
2. **P never waits on the CPU.** Waiting would stall mpv's pipeline.
3. **C gets the freshest frame possible**, and may wait a limited time for the
   frame P is rendering, as long as it can still draw it in the same refresh.
4. **C can skip frames.** Dropping frames that a newer one replaced is fine.
5. **Minimal locking and waiting**: lock sections are a few assignments, and no
   GL call is made while holding the lock.
6. **Readable P and C loops.** Synchronisation lives in one place, the render and
   draw code stays in the loops.

## Decisions

### Three buffers

With two buffers, while C holds one, P has exactly one buffer left and may have
to overwrite the newest finished frame before C has shown it. If C then can't
wait for P, it can only show an older frame, typically the one it showed before.

With three buffers, there's always a finished frame newer than the one C last
got, so not waiting (or a timeout) costs little: C shows a frame that is nearly
the freshest while P renders the next one.

The worst case is the same for two and three buffers: when C waits, one refresh
must fit the rest of P's render plus C's draw. A third buffer doesn't shorten
that, but GPU-side fence waits tend to be shorter, as P usually reuses a buffer
that C released long ago. The cost is one more texture of VRAM (about 33 MB for
4K RGBA8). Bandwidth is unchanged: each frame is still written once and read once.

### Buffer selection

- **P** takes neither the buffer it finished last (the newest frame) nor the one
  C holds. Of the rest, one or two buffers, it takes the one with the oldest
  frame. This keeps the newest frame available to C at all times.
- **C** takes the newest published buffer that P doesn't hold.

Frames are numbered (`frame_serial`) when published; 0 means empty. "Oldest" and
"newest" compare serials.

### C's optional wait

`read(timeout)` waits only if P holds a buffer at the time of the call, and only
for that frame to be published (`published_frame_count` increased), not for later
frames. Afterwards it takes the newest finished buffer, whichever that is.

- The frame C waited for isn't reserved for it. If P publishes again before C
  wakes, C gets the newer frame. P only received that frame because mpv's timing
  made it due now, so showing it is right.
- If P publishes and immediately starts another frame before C wakes, the frame
  that woke C is still intact: P doesn't take the newest buffer.
- `timeout=0` never waits. `None` and negative values are rejected: waiting
  without a limit isn't part of the design (a paused source would block C).
- The timeout is the most C is willing to wait, so there's still time to draw in
  the same refresh. How to compute it is up to C (see "Wait budget" below).

`write_buffer_is_held` and `write_started_at` are lock-free hints. They let C
skip a possibly costly budget estimate when P is idle. The check and the later
`read()` can race (P may finish in between), which is harmless: `read()` decides
under the lock whether to wait.

### Synchronisation objects

| Object | Count | Purpose |
| --- | --- | --- |
| `threading.Lock` | 1 | Guards which buffer each thread holds, frame serials, fence slots, the frame counter. |
| `threading.Condition` on that lock | 1 | Lets C wait for P to publish a frame. It's the same lock, not a second sync object. |
| `GLsync` fence slot per buffer | 3 | GPU-side ordering of each buffer's uses. |

A plain `Lock` is used instead of the condition's default `RLock`: nothing locks
re-entrantly, and accidental nesting fails immediately instead of hiding a bug.

### One fence per buffer

Each buffer keeps the fence of the GPU commands that last used it, whether that
was P's render or C's draw. The CPU-side bookkeeping already guarantees that only
one thread uses a buffer at a time, so separate read and write fences aren't
needed:

1. Taking a buffer (under the lock) also takes its fence out of the slot.
2. Outside the lock, the taker calls `glWaitSync` on it, so its own GPU queue
   waits; the CPU doesn't. The fence is then deleted (GL defers the deletion
   while a wait on it is pending).
3. Releasing the buffer inserts a new fence, calls `glFlush`, and stores the
   fence in the slot (under the lock).

Waiting on one's own earlier fence is harmless: commands within one context are
executed in order.

`glFlush` after `glFenceSync` is mandatory. A context waiting on a fence that was
never submitted may wait forever. The buffer is released before C swaps buffers,
so the swap's implicit flush can't be relied on.

C must re-bind the texture after taking the buffer: GL only guarantees that
another context's changes are visible once they're complete and the object is
bound again in the observing context.

### All synchronisation in the manager

`BufferManager` does both the CPU-side locking and the GPU-side fencing. The two
are tied together: the fence must be taken while the buffer is claimed under the
lock, and the new fence stored before the buffer is released. Splitting the GPU
part into a separate context manager on the buffer would rely on every call site
nesting both correctly, and a wrong order would only show as occasional corrupted
frames. The manager is therefore specific to `FrameBuffer`, not generic.

### Context managers only

`write()` and `read(timeout)` return context managers; there are no public
get/release methods, so a release can't be forgotten:

```python
with manager.write() as buffer:      # P
    buffer.ensure_capacity(w, h, fmt)
    render_into(buffer.fbo_id)
    buffer.frame_width, buffer.frame_height = w, h

with manager.read(timeout=budget) as frame:    # C
    if frame is not None:
        draw(frame.buffer)
```

- Entering takes the buffer and issues the GPU wait. Leaving fences, flushes and
  releases, also if the block raised (commands may already be queued).
- If P's block raises, the frame isn't published and the buffer is marked empty,
  as its contents are unknown.
- `read()` binds `None` if no frame was published yet, otherwise a `ReadFrame`
  with the buffer, its `frame_serial`, `is_new` and `frames_skipped`.
- The context managers live on the manager, as entering and leaving changes the
  manager's state. They are two reusable handle objects, so no generator or
  context manager is allocated per frame.
- Nested blocks raise `RuntimeError` (one producer, one consumer).

The render/draw code itself (mpv's render call, the blit or textured quad) stays
in the P and C loops, so important code isn't fragmented into small methods.

### Allocation and resizing

- The manager's constructor allocates all three buffers, before media plays, so
  playback doesn't start with allocation delays.
- During playback, buffers are resized one at a time:
  `FrameBuffer.ensure_capacity()`, called by P inside `write()` before rendering.
  At that point P owns the buffer and its GPU queue already waits for C's last
  use of it. After a size change, all buffers are resized within three frames.
- Because only the buffer P is about to write is reallocated, the other two keep
  their frames, and C always has something to draw. Media isn't unloaded, only
  paused or at EOF, so there's no use for reallocating all buffers at once later.
- Buffers only grow: smaller frames reuse the allocation and occupy part of it.
  A different pixel format also reallocates.
- Reallocation creates a new texture and FBO and deletes the old ones, instead
  of re-specifying the shared texture in place, which would be a hazard across
  contexts.
- Each buffer records its allocated size and the size of the frame it holds
  (`frame_width`, `frame_height`, set by P) separately: while a resize is in
  progress, buffers can have different sizes, and C must use the frame's size.

### Shutdown

`close()` wakes a waiting C (its `read()` binds `None`), deletes the fences and
buffers, and is safe to call repeatedly. P and C are expected to have stopped
using the buffers; if not, a warning is logged and the buffers are deleted
anyway.

## Requirements on callers

- `write()` and `read()` need the calling thread's GL context to be current; the
  constructor and `close()` need a context of the share group.
- P sets `frame_width`/`frame_height` of the buffer it rendered.
- C re-binds the texture each frame (see above) and uses the frame's size.
- Buffers are only used inside `write()`/`read()` blocks.

## Performance

The abstraction costs roughly 1-2 µs per frame and thread (two method calls, a
`with` block, an uncontended lock), against GL calls of about 0.5-1 µs each via
ctypes, which are needed anyway, and a 10 ms frame at 100 Hz. Lock sections
contain only assignments; ctypes releases the GIL during GL calls, so both
threads' GL work runs in parallel.

## Wait budget (not part of `buffer.py`)

CPU timings only measure how long queueing commands takes, not GPU execution.
A budget decision should be based on GPU timings:

- Measure P's render and C's draw with `GL_TIME_ELAPSED` timer queries (per
  context, read back a frame or two later to avoid stalls).
- Track a percentile (p95 for adaptive operation, p99 for strict timing) and an
  EMA for drift.
- Wait if `remaining P time + C draw time + margin` fits before the next vsync;
  `write_started_at` provides the start of P's current render.

Alternatively, let the user choose: always wait (eager), never wait, or a fixed
budget. If GPU work misses the vsync, the frame isn't dropped but shown a
refresh late, and the previous frame stays on screen one refresh longer.

## Known limitations

- Exactly one producer and one consumer thread.
- The frame C waits for isn't guaranteed to be the one it gets: a newer one may
  replace it (intended, see above).
- FBOs are used from both contexts, relying on pyglet 1.4/1.5's context sharing.
