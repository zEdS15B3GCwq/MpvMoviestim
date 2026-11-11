# Buffer management design (`buffer.py`)

This document records the design intent for the intermediate frame buffers that
the render worker and the draw thread share: the goals, the decisions, and the
reasons for them. It's meant to help later changes tell what is essential and
what is incidental. Usage details are in `buffer.py`'s docstrings.

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
   and GL calls are made outside the lock.
6. **Lean code.** One producer, one consumer, not a general-purpose library: no
   context managers, no statistics, few methods. The render and draw code stays
   in the P and C loops.

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

The manager keeps the buffer indices ordered from freshest to oldest frame
(`_buffer_order`). Publishing a frame moves its buffer to the front.

- **P** takes the oldest buffer, or the second oldest if C holds the oldest. It
  never takes the freshest one, so the newest frame is always available to C.
- **C** takes the freshest buffer.

No pinning or reservation is needed: the buffer C holds is simply skipped by P.
Before the first frame is published, `acquire_read_buffer()` returns `None`.

### C's optional wait

`acquire_read_buffer(timeout)` waits only if P holds a buffer at the time of the
call. It then waits only for that frame to be published (the frame count goes
up), not for later frames. After the wait, it takes the freshest buffer,
whichever that is.

- **The frame C waited for isn't reserved for it.** If P publishes again before
  C wakes up, C gets the newer frame. P only received that frame because mpv's
  timing made it due now, so showing it is right.
- **The frame that woke C stays intact.** If P publishes and immediately starts
  another frame before C wakes up, the new frame goes into a different buffer,
  because P never takes the freshest one.
- **Timeout values.** `timeout <= 0` never waits. Waiting without a limit isn't
  part of the design, because a paused source would block C indefinitely.
- **What the timeout means.** It's the longest C is willing to wait, chosen so
  there's still time to draw in the same refresh. How to compute it is up to C
  (see [Wait budget](#wait-budget-not-part-of-bufferpy)).

`write_buffer_is_held` is a hint that is read without locking. It lets C skip a
possibly costly budget estimate when P is idle. The check can race with the
later `acquire_read_buffer()`, because P may finish in between. That's harmless:
`acquire_read_buffer()` decides under the lock whether to wait.

### Synchronisation objects

| Object | Count | Purpose |
| --- | --- | --- |
| `threading.Lock` | 1 | Guards which buffer each thread holds, the buffer order, the frame count and the active flag. |
| `threading.Condition` on that lock | 1 | Lets C wait for P to publish a frame, or for shutdown. It uses the same lock, so it isn't a second lock. |
| `GLsync` fence per buffer | 3 | Orders each buffer's uses on the GPU. |

The condition uses a plain `Lock` instead of its default `RLock`. Nothing locks
re-entrantly, and if something ever nests by mistake, it fails immediately
instead of hiding the bug.

### One fence per buffer, skipped for one's own fences

Each buffer keeps the fence of the GPU commands that last used it, whether that
was P's render or C's draw, and records which thread placed it. The CPU-side
bookkeeping already guarantees that only one thread holds a buffer at a time, so
separate read and write fences aren't needed. A buffer changes hands like this:

1. **Claiming (under the lock).** The thread marks the buffer as its own.
2. **Waiting (outside the lock).** If the *other* thread placed the fence, the
   new holder calls `glWaitSync`. That makes the holder's GPU queue wait; the
   CPU doesn't block. Either way, the fence is then deleted. GL defers the
   deletion while a wait on it is still pending.
3. **Releasing.** The holder inserts a new fence and calls `glFlush` (outside
   the lock), then gives the buffer back (under the lock).

**Skipping the wait on one's own fence** is safe: commands within one context are
executed in order. It also saves time, because `glWaitSync` isn't free even
though it doesn't block the CPU. It can force the GPU to finish everything before
the fence before starting anything after it, which removes the usual overlap of
consecutive commands.

One's own fence comes up often:

- **C re-reads its own buffer** whenever no new frame has arrived. With 24 fps
  video on a 60 Hz display, that's most reads.
- **P reuses a buffer C never read**, i.e. a dropped frame.

The fence status isn't queried (`glGetSynciv`). That query may itself cost a
round-trip to the driver.

**`glFlush` after `glFenceSync` is mandatory.** A context that waits on a fence
that was never submitted may wait forever. The buffer is released before C swaps
the screen buffers, so the implicit flush in the swap can't be relied on.

**C must re-bind the texture after taking the buffer.** GL only guarantees that
another context's changes are visible once they're complete *and* the object has
been bound again in the context that reads it.

### All synchronisation in the manager

`TripleBufferManager` does both the CPU-side locking and the GPU-side fencing,
because the two depend on each other: the GPU wait must follow the claim, and the
new fence must be placed before the buffer is given back. Splitting them would
make every call site responsible for the right order, and a wrong order would
only show up as occasional corrupted frames.

The manager is therefore specific to `FrameBuffer` and isn't generic. It never
touches a buffer's contents. `FrameBuffer`'s GL helpers (`guard_before`,
`guard_after`, `deallocate`) are for the manager only; P and C only read the
buffer's attributes.

### Explicit acquire and release

The API is four calls, with `try`/`finally` in the caller:

```python
# P (render worker thread)
try:
    while running:
        buffer = manager.acquire_write_buffer(width, height, pixel_format)
        render_into(buffer.frame_fbo_info())
        manager.release_write_buffer()     # publishes the frame
finally:
    manager.shutdown()

# C (draw thread), once per screen refresh
buffer = manager.acquire_read_buffer(timeout=wait_budget)
if buffer is not None:
    try:
        draw(buffer.texture_id, buffer.frame_width, buffer.frame_height)
    finally:
        manager.release_read_buffer()
```

There are no context managers. With one producer and one consumer, the two
call sites are easy to get right, and explicit calls make the error path visible:

- **P releases only after a successful render.** Releasing publishes the frame,
  so P must not release in a `finally`. A failed render stays unpublished.
- **C always releases**, in a `finally`, because its draw commands may already
  be queued and P must not overwrite the buffer before they're done.
- **No nesting:** acquiring twice from the same side raises `RuntimeError`.

### Allocation and resizing

- **Initial allocation:** the constructor allocates all three buffers before
  media plays, so playback doesn't start with allocation delays. If one fails,
  the ones already allocated are freed.
- **Resizing during playback** happens one buffer at a time.
  - P passes the frame size and pixel format to `acquire_write_buffer()`. The
    buffer records the frame size, and is reallocated if the frame doesn't fit.
  - After a size change, all buffers are resized within three frames.
- **C always has something to draw.** Only the buffer P is about to write is
  reallocated, so the other two keep their frames. Media is never unloaded, only
  paused or at EOF, so there's no use for reallocating all buffers at once.
- **What triggers reallocation:** buffers only grow. A smaller frame reuses the
  allocation and occupies its `(0, 0, frame_width, frame_height)` area. When a
  frame is larger in one dimension, each dimension becomes the larger of the
  allocated and the requested size, so a 16×16 buffer asked for 32×8 becomes
  32×16. A different pixel format also reallocates.
- **How it reallocates:** it deletes the old texture and FBO first, then creates
  new ones, which keeps peak VRAM lower.
  - **No fence wait is needed.** GL keeps a deleted object alive while queued
    commands (C's last draw) still use it. The new texture has no earlier use, so
    its fence is simply dropped.
  - **Name reuse is safe.** If the new texture gets the old one's name, C's
    queued commands still refer to the old object: names are only looked up
    when a command is issued.
  - **If creating the new objects fails**, the buffer is left unallocated. P
    holds it and can't continue; its `finally` calls `shutdown()`.
- **Two sizes per buffer:** each buffer records its allocated size and its frame
  size separately. While a resize is in progress, buffers can have different
  sizes, so C must use the frame size of the buffer it got.

### Errors and exceptions

`buffer.py` never repairs anything or handles exceptions locally. Exceptions
propagate and are expected to end the program. The manager only guarantees that
an unfinished frame is never shown, and that `shutdown()` frees the resources.

- **P fails** (render, allocation, or a GL call): P doesn't release its buffer,
  so the frame is never published. P's `finally` calls `shutdown()`, which frees
  all buffers, including P's.
- **C after shutdown:** `acquire_read_buffer()` raises `RuntimeError`, also when
  woken from a wait. That's how C learns that P has stopped.
- **C holds a buffer at shutdown:** `shutdown()` leaves it alone, because C's
  queued commands may still use it. C frees it in `release_read_buffer()`.
- **C fails while drawing:** its `finally` releases the buffer as usual.
- **`release_write_buffer()` after shutdown** does nothing, so it can't mask the
  original exception.

What can raise:

| Source | Example | Afterwards |
| --- | --- | --- |
| P's render call | mpv error, shutdown in progress | Frame unpublished; `shutdown()` frees it |
| `acquire_write_buffer()` reallocation | Bad size (`ValueError`), FBO incomplete (format unsupported, possibly out of memory) | Buffer unallocated; `shutdown()` frees the rest |
| C's draw | Blit or quad error | C releases in its `finally` |
| GL calls in general | Only raise if pyglet's `debug_gl` is on (PsychoPy turns it off) | Depends on the call |
| Misuse | Nested acquire, release without acquire, use after shutdown | `RuntimeError`, bookkeeping unchanged |

> [!IMPORTANT]
> **The caller must forward P's exception; `buffer.py` deliberately doesn't.**
> C only sees `RuntimeError` from `acquire_read_buffer()` after shutdown. The
> worker should store its exception before stopping, and the main thread (e.g.
> `draw()`) should re-raise it, so the program fails visibly with the real cause.

### Shutdown

Only P calls `shutdown()`, with its GL context current, in the `finally` around
its loop. It marks the manager inactive, wakes a waiting C, and frees every
buffer except the one C holds. Calling it more than once is safe.

Restricting shutdown to P keeps the code lean:

- `release_write_buffer()` can check the active flag without the lock, because
  only P's own thread can change it.
- P's buffer is never freed while P is still using it.
- C's held slot is read under the lock at shutdown, so C can't claim or release a
  buffer in between. C frees its own buffer later, in its own context.

## Requirements on callers

- **GL context:** each thread's own context must be current when it calls the
  manager. The constructor needs a context of the share group.
- **P** passes each frame's size to `acquire_write_buffer()`, releases only after
  a successful render, and calls `shutdown()` in a `finally`.
- **C** releases in a `finally`, re-binds the texture each frame, and draws only
  the frame's area of the texture.
- **Buffers** are only used between acquire and release, and their attributes are
  read-only.
- **Exceptions** from P are forwarded by the caller (see the box above).

## Performance

The abstraction costs roughly 1–2 µs per frame and thread: two method calls and
an uncontended lock. For comparison:

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
  the next vsync. P would have to record when it started its current render;
  the manager doesn't.

Alternatively, let the user choose one of three modes:

- always wait (eager)
- never wait
- wait up to a fixed budget

If GPU work misses the vsync, the frame isn't dropped. It's shown one refresh
late, and the previous frame stays on screen one refresh longer.

## Known limitations

- Exactly one producer thread and one consumer thread.
- Only P may call `shutdown()`.
- C may not get the exact frame it waited for: a newer one can replace it. This
  is intended (see above).
- After an error the manager can't be reused: shut it down and create a new one.
- If a GL call inside `acquire_read_buffer()` raises (only with `debug_gl` on),
  the claim stays in place and `shutdown()` doesn't free that buffer.
- FBOs are used from both contexts, relying on pyglet 1.4/1.5's context sharing.
