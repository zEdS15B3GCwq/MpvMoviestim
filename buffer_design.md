# Buffer management design (`buffer.py`)

This document records the design intent for the intermediate frame buffers that
the render worker and the draw thread share: the goals, the decisions, and the
reasons for them. It's meant to help later changes tell what is essential and
what is incidental.

## Context

Two threads pass video frames to each other through intermediate buffers. Each
buffer is a texture with an FBO attached.

- **P (producer)** is the render worker. libmpv pushes frames at its own pace.
  P renders each one into a buffer as soon as it can, so it never stalls mpv.
- **C (consumer)** is the draw thread (PsychoPy's `draw()`). Once per screen
  refresh, it draws the freshest finished frame into PsychoPy's target.

Each thread has its own OpenGL context. Both contexts are in the same share
group, set up through pyglet's shared hidden window, so textures, FBOs and sync
objects are shared between them.

## Goals

In order of priority:

1. **Thread safety on CPU and GPU.** A buffer is never written while it's being
   read. That holds for the CPU-side bookkeeping and for queued GPU commands.
2. **P never waits on the CPU.** Waiting would stall mpv's pipeline.
3. **C gets the freshest frame possible.** C may wait a limited time for the
   frame P is rendering, as long as it can still draw it in the same refresh.
4. **C can skip frames.** Dropping a frame that a newer one has replaced is fine.
5. **Locking and waiting stay minimal.** Each lock section is a few assignments,
   and no GL call is made while the lock is held.
6. **The P and C loops stay readable.** Synchronisation lives in one place. The
   render and draw code stays in the loops.

## Decisions

### Three buffers

With two buffers, P has only one buffer left while C holds the other. P may then
have to overwrite the newest finished frame before C has shown it. If C can't
wait for P at that point, it can only show an older frame, usually the one it
showed before.

With three buffers, there's always a finished frame newer than the one C last
got. So when C doesn't wait, or its wait times out, it loses little: it shows a
frame that is nearly the freshest while P renders the next one.

The worst case is the same with two or three buffers. When C waits, one refresh
has to fit the rest of P's render plus C's draw, and a third buffer doesn't
shorten that.

What a third buffer does improve is the GPU-side fence waits. They tend to be
shorter, because P usually reuses a buffer that C released long ago.

The costs:

- **VRAM:** one more texture, about 33 MB for 4K RGBA8.
- **Bandwidth:** unchanged. Each frame is still written once and read once.

### Buffer selection

- **P** never takes the buffer it finished last (the newest frame), or the buffer
  C holds. That leaves one or two buffers, and P takes the one with the oldest
  frame. This way the newest frame is always available to C.
- **C** takes the newest published buffer that P doesn't hold.

Frames are numbered when P publishes them (`frame_serial`). "Oldest" and "newest"
compare these numbers. A serial of 0 means the buffer holds no frame that may be
shown: it's either empty, or rendering into it failed (see
[Errors](#errors-and-exceptions)).

### C's optional wait

`acquire_read_buffer(timeout)` waits only if P holds a buffer at the time of the
call. It then waits only for that frame to be published (`published_frame_count`
goes up), not for later frames. After the wait, it takes the newest finished
buffer, whichever that is.

- **The frame C waited for isn't reserved for it.** If P publishes again before
  C wakes up, C gets the newer frame. P only received that frame because mpv's
  timing made it due now, so showing it is right.
- **The frame that woke C stays intact.** If P publishes and immediately starts
  another frame before C wakes up, the new frame goes into a different buffer,
  because P never takes the newest one.
- **Allowed timeouts.**
  - `timeout=0` never waits.
  - `None` and negative values are rejected. Waiting without a limit isn't part
    of the design, because a paused source would block C indefinitely.
- **What the timeout means.** It's the longest C is willing to wait, chosen so
  there's still time to draw in the same refresh. How to compute it is up to C
  (see [Wait budget](#wait-budget-not-part-of-bufferpy)).

`write_buffer_is_held` and `write_started_at` are hints that are read without
locking. They let C skip a possibly costly budget estimate when P is idle. The
check can race with the later `acquire_read_buffer()`, because P may finish in
between. That's harmless: `acquire_read_buffer()` decides under the lock whether
to wait.

### Synchronisation objects

| Object | Count | Purpose |
| --- | --- | --- |
| `threading.Lock` | 1 | Guards which buffer each thread holds, the frame serials, the fence slots, and the frame counter. |
| `threading.Condition` on that lock | 1 | Lets C wait for P to publish a frame. It uses the same lock, so it isn't a second lock. |
| `GLsync` fence slot per buffer | 3 | Orders each buffer's uses on the GPU. |

The condition uses a plain `Lock` instead of its default `RLock`. Nothing locks
re-entrantly, and if something ever nests by mistake, it fails immediately
instead of hiding the bug.

### One fence per buffer, skipped for one's own fences

Each buffer keeps the fence of the GPU commands that last used it, whether that
was P's render or C's draw. It also records which thread placed that fence
(`last_use_fence_placed_by_writer`).

The CPU-side bookkeeping already guarantees that only one thread uses a buffer at
a time. So separate read and write fences aren't needed, and a buffer changes
hands like this:

1. **Claiming (under the lock).** Taking a buffer also takes its fence out of the
   buffer's slot.
2. **Waiting (outside the lock).** If the *other* thread placed the fence, the
   new holder calls `glWaitSync`. That makes the holder's GPU queue wait; the
   CPU doesn't block. Either way, the fence is then deleted. GL defers the
   deletion while a wait on it is still pending.
3. **Releasing.** The holder inserts a new fence, calls `glFlush`, and stores the
   fence and its owner in the slot (under the lock).

**Skipping the wait on one's own fence** is safe: commands within one context are
executed in order. It also saves time, because `glWaitSync` isn't free even
though it doesn't block the CPU. It can force the GPU to finish everything before
the fence before starting anything after it, which removes the usual overlap of
consecutive commands.

One's own fence comes up often:

- **C re-reads its own buffer** whenever no new frame has arrived. With 24 fps
  video on a 60 Hz display, that's most reads.
- **P reuses a buffer C never read**, i.e. a dropped frame or a failed render.

The fence status isn't queried (`glGetSynciv`). That query may itself cost a
round-trip to the driver.

**`glFlush` after `glFenceSync` is mandatory.** A context that waits on a fence
that was never submitted may wait forever. The buffer is released before C swaps
the screen buffers, so the implicit flush in the swap can't be relied on.

**C must re-bind the texture after taking the buffer.** GL only guarantees that
another context's changes are visible once they're complete *and* the object has
been bound again in the context that reads it.

### All synchronisation in the manager

`BufferManager` does both the CPU-side locking and the GPU-side fencing, because
the two depend on each other:

- The fence must be taken from the slot while the buffer is claimed under the
  lock.
- The new fence must be stored before the buffer is released.

The GPU part could be split into a separate context manager on the buffer. But
then every call site would have to nest the two blocks in the right order, and a
wrong order would only show up as occasional corrupted frames.

The manager is therefore specific to `FrameBuffer` and isn't generic. It never
touches a buffer's contents.

### Context managers only

`acquire_write_buffer()` and `acquire_read_buffer(timeout)` return context
managers. There are no public get/release methods, so a release can't be
forgotten:

```python
with manager.acquire_write_buffer() as buffer:              # P
    buffer.set_frame_size(width, height, pixel_format)
    render_into(buffer.fbo_id)

with manager.acquire_read_buffer(timeout=budget) as frame:  # C
    if frame is not None:
        draw(frame.buffer)   # area (0, 0, frame_width, frame_height)
```

- **Entering a block** takes the buffer and issues the GPU wait.
- **Leaving a block** fences, flushes and releases the buffer. This also happens
  when the block raises, because commands may already be queued.
- **What C gets:** `acquire_read_buffer()` binds `None` if no frame has been
  published yet. Otherwise it binds a `ReadFrame` with the buffer, its
  `frame_serial`, `is_new` and `frames_skipped`.
- **Where the context managers live:** on the manager, because entering and
  leaving them changes the manager's state. They're two reusable handle objects,
  so no generator or context manager is allocated per frame.
- **No nesting:** nested blocks raise `RuntimeError`, because there's one
  producer and one consumer.

The render and draw code itself stays in the P and C loops: mpv's render call,
the blit or the textured quad. That keeps important code from being split into
small methods all over the codebase.

### Allocation and resizing

- **Initial allocation:** the manager's constructor allocates all three buffers
  before media plays, so playback doesn't start with allocation delays.
- **Resizing during playback** happens one buffer at a time.
  - P calls `FrameBuffer.set_frame_size(width, height, pixel_format)` inside
    `acquire_write_buffer()` before rendering each frame. It records the size of
    the frame about to be rendered, and reallocates the buffer if it's too small.
  - At that point P owns the buffer, and its GPU queue already waits for C's last
    use of it.
  - After a size change, all buffers are resized within three frames.
- **C always has something to draw.** Only the buffer P is about to write is
  reallocated, so the other two keep their frames. Media is never unloaded, only
  paused or at EOF, so there's no use for reallocating all buffers at once later.
- **What triggers reallocation:** buffers only grow. A smaller frame reuses the
  allocation and occupies its `(0, 0, frame_width, frame_height)` area. When a
  frame is larger in one dimension, each dimension becomes the larger of the
  allocated and the requested size, so a 16×16 buffer asked for 32×8 becomes
  32×16. A different pixel format also reallocates.
- **How it reallocates:** it creates a new texture and FBO and deletes the old
  ones. Re-specifying the shared texture in place would be a hazard across
  contexts. If creating the new objects fails, the buffer is left unchanged.
- **Two sizes per buffer:** each buffer records its allocated size and its frame
  size separately. While a resize is in progress, buffers can have different
  sizes, so C must use the frame size of the buffer it got.

### Errors and exceptions

`buffer.py` never repairs anything or handles exceptions locally. Exceptions
always propagate. The manager only guarantees that its own bookkeeping stays
consistent:

- **A claim is never left behind.** If anything raises while a buffer is being
  claimed, including the GPU wait, the claim is undone and the fence is put back.
  Leaving a block always releases the buffer and places a fence.
- **A failed render is never shown.** If P's block raises, the frame isn't
  published, and the buffer's `frame_serial` and frame size are set to 0. C
  treats the buffer like an empty one and never picks it. P picks it first next
  time, because 0 is the oldest serial, and the next successful render
  overwrites its contents. The GL objects themselves are intact, so using the
  buffer again doesn't fail.
- **Allocation failures leave the buffer unchanged.** `set_frame_size()` creates
  the new objects before deleting the old ones.

What can raise, and the state it leaves:

| Source | Example | State afterwards | Repeats? |
| --- | --- | --- | --- |
| P's render call | mpv error, shutdown in progress | Buffer hidden, objects intact | Usually not |
| `set_frame_size()` | Bad size or format (`ValueError`), FBO incomplete (format unsupported, possibly out of memory) | Buffer unchanged | Yes, every frame |
| C's draw | Blit or quad error | Nothing to fix; C only reads | Probably every frame |
| GL calls in general | Only raise if pyglet's `debug_gl` is on (PsychoPy turns it off) | Depends on the call | — |
| Misuse | No context current, nested blocks, use after `close()` | Bookkeeping consistent | Yes |

> [!IMPORTANT]
> **The caller must handle these exceptions; `buffer.py` deliberately doesn't.**
> In particular, if P's worker thread dies quietly, mpv stalls and C shows the
> last frame forever. The worker should catch the exception, store it and stop,
> and the main thread (e.g. `draw()`) should re-raise it, so the program fails
> visibly. The manager doesn't store or forward exceptions: that's a separate
> responsibility.

### Shutdown

`close()` wakes a waiting C, whose `acquire_read_buffer()` then binds `None`. It
then deletes the fences and buffers. Calling it more than once is safe.

P and C are expected to have stopped using the buffers by then. If either still
holds one, a warning is logged and the buffers are deleted anyway.

## Requirements on callers

- **GL context:** `acquire_write_buffer()` and `acquire_read_buffer()` need the
  calling thread's GL context to be current. The constructor and `close()` need
  a context of the share group.
- **P** calls `set_frame_size()` on its buffer before rendering each frame.
- **C** re-binds the texture each frame (see above), and draws only the frame's
  area of the texture.
- **Buffers** are only used inside acquire blocks.
- **Exceptions** are handled by the caller (see the box above).

## Performance

The abstraction costs roughly 1–2 µs per frame and thread: two method calls, a
`with` block and an uncontended lock. For comparison:

- Each GL call through ctypes costs about 0.5–1 µs, and those calls are needed
  anyway.
- A frame at 100 Hz lasts 10 ms.

Lock sections contain only assignments. ctypes releases the GIL during GL calls,
so the two threads' GL work runs in parallel.

## Wait budget (not part of `buffer.py`)

CPU timings only measure how long it takes to queue commands, not how long the
GPU takes to run them. A budget decision should be based on GPU timings:

- **Measuring:** time P's render and C's draw with `GL_TIME_ELAPSED` timer
  queries. Query objects are per context, and results should be read a frame or
  two later to avoid stalls.
- **Tracking:** keep a percentile of these timings (p95 for adaptive operation,
  p99 for strict timing) and an EMA to follow drift.
- **Deciding:** wait if `remaining P time + C draw time + margin` fits before
  the next vsync. `write_started_at` gives the start of P's current render.

Alternatively, let the user choose one of three modes:

- always wait (eager)
- never wait
- wait up to a fixed budget

If GPU work misses the vsync, the frame isn't dropped. It's shown one refresh
late, and the previous frame stays on screen one refresh longer.

## Known limitations

- Exactly one producer thread and one consumer thread.
- C may not get the exact frame it waited for: a newer one can replace it. This
  is intended (see above).
- FBOs are used from both contexts, relying on pyglet 1.4/1.5's context sharing.
