"""Tests for `mpvmoviestim.buffer`: FrameBuffer and BufferManager.

Three groups:

1. Logic tests with fake GL and allocation functions: buffer selection, fence
   handoff, error bookkeeping, misuse, resizing. No GPU needed.
2. Threaded timing tests, also with fake GL: C's optional wait for P.
3. A GL integration test with real shared contexts, skipped if no GL context
   can be created.
"""

from __future__ import annotations

import ctypes
import importlib
import random
import sys
import threading
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


def _import_package_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Import `buffer` and `utils`, also when the package's __init__ can't be imported.

    The package's __init__ imports mpvmoviestim.py, which may be mid-refactor.
    In that case, the modules are loaded under a private package name, so the
    real `mpvmoviestim` entry in sys.modules is left alone.
    """
    try:
        return (
            importlib.import_module("mpvmoviestim.buffer"),
            importlib.import_module("mpvmoviestim.utils"),
        )
    except Exception:  # noqa: BLE001  # any error in the package's __init__
        package_name = "_mpvmoviestim_buffer_under_test"
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src" / "mpvmoviestim")
        ]
        sys.modules[package_name] = package
        return (
            importlib.import_module(f"{package_name}.buffer"),
            importlib.import_module(f"{package_name}.utils"),
        )


buffer_module, utils_module = _import_package_modules()
BufferManager = buffer_module.BufferManager
FrameBuffer = buffer_module.FrameBuffer

GL_RGBA8 = 0x8058
GL_RGBA16F = 0x881A


# ---- fakes ----


class FakeGL:
    """Records fence calls and checks that fences are used correctly."""

    GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
    GL_TIMEOUT_IGNORED = 0xFFFFFFFFFFFFFFFF

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_fence = 1
        self.live_fences: set[int] = set()
        self.created_fences: list[int] = []
        self.waited_fences: list[int] = []
        self.deleted_fences: list[int] = []
        self.flush_count = 0
        self.fail_next_wait = False

    def glFenceSync(self, condition: int, flags: int) -> int:
        assert condition == self.GL_SYNC_GPU_COMMANDS_COMPLETE
        assert flags == 0
        with self._lock:
            fence = self._next_fence
            self._next_fence += 1
            self.live_fences.add(fence)
            self.created_fences.append(fence)
        return fence

    def glFlush(self) -> None:
        with self._lock:
            self.flush_count += 1

    def glWaitSync(self, fence: int, flags: int, timeout: int) -> None:
        assert flags == 0
        assert timeout == self.GL_TIMEOUT_IGNORED
        with self._lock:
            assert fence in self.live_fences, f"wait on deleted fence {fence}"
            if self.fail_next_wait:
                self.fail_next_wait = False
                raise RuntimeError("simulated glWaitSync failure")
            self.waited_fences.append(fence)

    def glDeleteSync(self, fence: int) -> None:
        with self._lock:
            assert fence in self.live_fences, f"fence {fence} deleted twice"
            self.live_fences.remove(fence)
            self.deleted_fences.append(fence)


class FakeAllocator:
    """Stands in for the texture and FBO helpers of utils.py."""

    def __init__(self) -> None:
        self._next_name = 1
        self.live_textures: set[int] = set()
        self.live_fbos: set[int] = set()
        self.textures_until_failure: int | None = None
        self.fail_next_fbo = False

    def gl_texture_create(self, width: int, height: int, pixel_format: int) -> int:
        if self.textures_until_failure is not None:
            if self.textures_until_failure == 0:
                raise ValueError("simulated texture allocation failure")
            self.textures_until_failure -= 1
        texture_id = self._next_name
        self._next_name += 1
        self.live_textures.add(texture_id)
        return texture_id

    def gl_texture_destroy(self, texture_id: int) -> None:
        if texture_id != 0:
            self.live_textures.remove(texture_id)

    def gl_fbo_create(self, texture_id: int) -> int:
        assert texture_id in self.live_textures
        if self.fail_next_fbo:
            self.fail_next_fbo = False
            raise RuntimeError("simulated incomplete FBO")
        fbo_id = self._next_name
        self._next_name += 1
        self.live_fbos.add(fbo_id)
        return fbo_id

    def gl_fbo_destroy(self, fbo_id: int) -> None:
        if fbo_id != 0:
            self.live_fbos.remove(fbo_id)


@pytest.fixture(name="fake_gl")
def fixture_fake_gl(monkeypatch: pytest.MonkeyPatch) -> FakeGL:
    fake_gl = FakeGL()
    monkeypatch.setattr(buffer_module, "gl", fake_gl)
    return fake_gl


@pytest.fixture(name="fake_allocator")
def fixture_fake_allocator(monkeypatch: pytest.MonkeyPatch) -> FakeAllocator:
    fake_allocator = FakeAllocator()
    for function_name in (
        "gl_texture_create",
        "gl_texture_destroy",
        "gl_fbo_create",
        "gl_fbo_destroy",
    ):
        monkeypatch.setattr(
            buffer_module, function_name, getattr(fake_allocator, function_name)
        )
    return fake_allocator


@pytest.fixture(name="manager")
def fixture_manager(fake_gl: FakeGL, fake_allocator: FakeAllocator) -> Iterator[Any]:
    manager = BufferManager(16, 16, GL_RGBA8)
    yield manager
    manager.close()
    # every fence and GL object is deleted exactly once (FakeGL checks "once")
    assert fake_gl.live_fences == set()
    assert fake_allocator.live_textures == set()
    assert fake_allocator.live_fbos == set()


def publish_frame(manager: Any, width: int = 16, height: int = 16) -> Any:
    """Render and publish one (fake) frame, return the buffer it went into."""
    with manager.acquire_write_buffer() as buffer:
        buffer.set_frame_size(width, height, GL_RGBA8)
    return buffer


# ---- construction ----


def test_constructor_allocates_all_buffers(manager: Any, fake_allocator: FakeAllocator):
    assert len(manager.buffers) == 3
    assert len(fake_allocator.live_textures) == 3
    assert len(fake_allocator.live_fbos) == 3
    for buffer in manager.buffers:
        assert buffer.is_allocated
        assert (buffer.allocated_width, buffer.allocated_height) == (16, 16)
        assert buffer.pixel_format == GL_RGBA8
        assert buffer.frame_serial == 0
        assert (buffer.frame_width, buffer.frame_height) == (0, 0)


def test_constructor_failure_deletes_allocated_buffers(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    fake_allocator.textures_until_failure = 2  # the third texture fails
    with pytest.raises(ValueError, match="simulated"):
        BufferManager(16, 16, GL_RGBA8)
    assert fake_allocator.live_textures == set()
    assert fake_allocator.live_fbos == set()


# ---- reading and writing ----


def test_read_before_first_frame_gets_none(manager: Any):
    with manager.acquire_read_buffer() as frame:
        assert frame is None


def test_published_frame_is_read(manager: Any):
    written_buffer = publish_frame(manager, 10, 12)
    assert manager.published_frame_count == 1

    with manager.acquire_read_buffer() as frame:
        assert frame.buffer is written_buffer
        assert frame.frame_serial == 1
        assert frame.is_new
        assert frame.frames_skipped == 0
        assert (frame.buffer.frame_width, frame.buffer.frame_height) == (10, 12)


def test_repeated_read_is_not_new(manager: Any):
    publish_frame(manager)
    with manager.acquire_read_buffer() as first_frame:
        pass
    with manager.acquire_read_buffer() as second_frame:
        assert second_frame.buffer is first_frame.buffer
        assert second_frame.frame_serial == 1
        assert not second_frame.is_new
        assert second_frame.frames_skipped == 0


def test_frames_skipped_counts_frames_never_read(manager: Any):
    publish_frame(manager)
    with manager.acquire_read_buffer():
        pass
    for _ in range(3):
        publish_frame(manager)
    with manager.acquire_read_buffer() as frame:
        assert frame.frame_serial == 4
        assert frame.frames_skipped == 2


def test_reader_gets_newest_frame_writer_does_not_hold(manager: Any):
    publish_frame(manager)
    publish_frame(manager)
    with manager.acquire_write_buffer() as buffer_being_written:
        assert buffer_being_written.frame_serial == 0  # the unused third buffer
        with manager.acquire_read_buffer() as frame:
            assert frame.frame_serial == 2
            assert frame.buffer is not buffer_being_written


def test_writer_avoids_newest_and_readers_buffer(manager: Any):
    first_buffer = publish_frame(manager)
    second_buffer = publish_frame(manager)
    third_buffer = publish_frame(manager)
    assert len({first_buffer, second_buffer, third_buffer}) == 3

    # C holds the newest frame: P takes the oldest of the other two
    with manager.acquire_read_buffer() as frame:
        assert frame.buffer is third_buffer
        with manager.acquire_write_buffer() as buffer:
            assert buffer is first_buffer

    # now frame 4 is newest (first buffer); C holds an older one (third buffer)
    with manager.acquire_read_buffer() as frame:
        assert frame.buffer is first_buffer
    # P: not the newest (first), so the oldest of second and third
    with manager.acquire_write_buffer() as buffer:
        assert buffer is second_buffer


@pytest.mark.parametrize("seed", range(5))
def test_random_interleavings_keep_invariants(
    seed: int, fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    """Random single-threaded interleavings of P and C blocks, checked step by step."""
    rng = random.Random(seed)
    manager = BufferManager(16, 16, GL_RGBA8)
    write_handle = manager.acquire_write_buffer()
    held_write_buffer = None
    held_read_frame = None
    last_read_serial = 0

    for _ in range(2000):
        # snapshot of each buffer's serial and fence slot before the next step
        serials = {buffer: buffer.frame_serial for buffer in manager.buffers}
        fences = {
            buffer: (buffer.last_use_fence, buffer.last_use_fence_placed_by_writer)
            for buffer in manager.buffers
        }
        published = [buffer for buffer in manager.buffers if serials[buffer] > 0]
        newest = max(published, key=serials.__getitem__) if published else None
        waits_before = len(fake_gl.waited_fences)

        action = rng.choice(("write", "read"))
        if action == "write" and held_write_buffer is None:
            held_write_buffer = write_handle.__enter__()
            reader_buffer = held_read_frame.buffer if held_read_frame else None
            allowed = [
                b for b in manager.buffers if b is not newest and b is not reader_buffer
            ]
            assert held_write_buffer in allowed
            assert serials[held_write_buffer] == min(serials[b] for b in allowed)
            # GPU wait only on C's fence
            fence, placed_by_writer = fences[held_write_buffer]
            expected_waits = 1 if fence and not placed_by_writer else 0
            assert len(fake_gl.waited_fences) - waits_before == expected_waits
            held_write_buffer.set_frame_size(rng.choice((8, 16, 24)), 16, GL_RGBA8)

        elif action == "write":
            render_fails = rng.random() < 0.1
            exception = RuntimeError("render failed") if render_fails else None
            write_handle.__exit__(
                type(exception) if exception else None, exception, None
            )
            if render_fails:
                assert held_write_buffer.frame_serial == 0
            else:
                assert held_write_buffer.frame_serial == manager.published_frame_count
            held_write_buffer = None

        elif held_read_frame is None:
            read_handle = manager.acquire_read_buffer()
            held_read_frame = read_handle.__enter__()
            candidates = [b for b in published if b is not held_write_buffer]
            expected = max(candidates, key=serials.__getitem__) if candidates else None
            if expected is None:
                assert held_read_frame is None
                read_handle.__exit__(None, None, None)
                continue
            assert held_read_frame.buffer is expected
            assert held_read_frame.frame_serial >= last_read_serial
            last_read_serial = held_read_frame.frame_serial
            # GPU wait only on P's fence
            fence, placed_by_writer = fences[expected]
            expected_waits = 1 if fence and placed_by_writer else 0
            assert len(fake_gl.waited_fences) - waits_before == expected_waits

        else:
            manager.acquire_read_buffer().__exit__(None, None, None)
            held_read_frame = None

    if held_write_buffer is not None:
        write_handle.__exit__(None, None, None)
    if held_read_frame is not None:
        manager.acquire_read_buffer().__exit__(None, None, None)
    manager.close()
    assert fake_gl.live_fences == set()
    assert fake_allocator.live_textures == set()


# ---- fences ----


def test_fence_handoff_skips_own_fences(manager: Any, fake_gl: FakeGL):
    publish_frame(manager)
    writer_fence = fake_gl.created_fences[-1]

    # C waits on P's fence
    with manager.acquire_read_buffer():
        assert fake_gl.waited_fences == [writer_fence]
    reader_fence = fake_gl.created_fences[-1]

    # repeated read: C's own fence is deleted, but not waited on
    with manager.acquire_read_buffer():
        assert fake_gl.waited_fences == [writer_fence]
        assert reader_fence in fake_gl.deleted_fences


def test_writer_waits_on_readers_fence(manager: Any, fake_gl: FakeGL):
    first_buffer = publish_frame(manager)
    with manager.acquire_read_buffer():
        pass
    reader_fence = fake_gl.created_fences[-1]
    publish_frame(manager)
    publish_frame(manager)

    # P now takes the first buffer again, which C used last
    with manager.acquire_write_buffer() as buffer:
        assert buffer is first_buffer
        assert fake_gl.waited_fences[-1] == reader_fence


def test_every_release_flushes(manager: Any, fake_gl: FakeGL):
    publish_frame(manager)
    with manager.acquire_read_buffer():
        pass
    assert fake_gl.flush_count == 2


# ---- errors ----


def test_failed_render_is_hidden_and_reused_first(manager: Any, fake_gl: FakeGL):
    shown_buffer = publish_frame(manager)

    with (
        pytest.raises(RuntimeError, match="render failed"),
        manager.acquire_write_buffer() as failed_buffer,
    ):
        failed_buffer.set_frame_size(16, 16, GL_RGBA8)
        raise RuntimeError("render failed")

    assert not manager.write_buffer_is_held
    assert manager.published_frame_count == 1
    assert failed_buffer.frame_serial == 0
    assert (failed_buffer.frame_width, failed_buffer.frame_height) == (0, 0)
    assert failed_buffer.last_use_fence  # still fenced: commands may be queued

    with manager.acquire_read_buffer() as frame:
        assert frame.buffer is shown_buffer

    with manager.acquire_write_buffer() as buffer:
        assert buffer.frame_serial == 0  # an empty or failed buffer comes first


def test_exception_in_read_block_releases_buffer(manager: Any):
    publish_frame(manager)
    with pytest.raises(KeyError), manager.acquire_read_buffer() as frame:
        raise KeyError("draw failed")
    assert frame.buffer.last_use_fence  # fenced
    with manager.acquire_read_buffer() as frame_again:
        assert frame_again.buffer is frame.buffer


def test_failed_gpu_wait_undoes_write_claim(manager: Any, fake_gl: FakeGL):
    # make P's next buffer one that C used last, so P has to wait on C's fence
    first_buffer = publish_frame(manager)
    with manager.acquire_read_buffer():
        pass
    reader_fence = fake_gl.created_fences[-1]
    publish_frame(manager)
    publish_frame(manager)

    fake_gl.fail_next_wait = True
    with (
        pytest.raises(RuntimeError, match="simulated glWaitSync"),
        manager.acquire_write_buffer(),
    ):
        pytest.fail("block must not run")

    assert not manager.write_buffer_is_held
    assert manager.write_started_at is None
    assert first_buffer.last_use_fence == reader_fence  # put back
    assert reader_fence in fake_gl.live_fences

    with manager.acquire_write_buffer() as buffer:
        assert buffer is first_buffer
        assert fake_gl.waited_fences[-1] == reader_fence


def test_failed_gpu_wait_undoes_read_claim(manager: Any, fake_gl: FakeGL):
    written_buffer = publish_frame(manager)
    writer_fence = fake_gl.created_fences[-1]

    fake_gl.fail_next_wait = True
    with (
        pytest.raises(RuntimeError, match="simulated glWaitSync"),
        manager.acquire_read_buffer(),
    ):
        pytest.fail("block must not run")

    assert written_buffer.last_use_fence == writer_fence  # put back
    with manager.acquire_read_buffer() as frame:
        assert frame.buffer is written_buffer
        assert frame.is_new  # the failed read didn't count
        assert fake_gl.waited_fences == [writer_fence]


# ---- misuse ----


@pytest.mark.parametrize("bad_timeout", [None, -0.001, -1])
def test_invalid_timeout_is_rejected(manager: Any, bad_timeout: Any):
    with pytest.raises(ValueError, match="timeout"):
        manager.acquire_read_buffer(timeout=bad_timeout)


def test_nested_write_blocks_raise(manager: Any):
    with (
        manager.acquire_write_buffer(),
        pytest.raises(RuntimeError, match="Nested"),
        manager.acquire_write_buffer(),
    ):
        pass
    assert not manager.write_buffer_is_held


def test_nested_read_blocks_raise(manager: Any):
    # also when the outer block got no frame
    with manager.acquire_read_buffer() as frame:
        assert frame is None
        with (
            pytest.raises(RuntimeError, match="Nested"),
            manager.acquire_read_buffer(),
        ):
            pass

    publish_frame(manager)
    with manager.acquire_read_buffer() as frame:
        assert frame is not None
        with (
            pytest.raises(RuntimeError, match="Nested"),
            manager.acquire_read_buffer(),
        ):
            pass


def test_use_after_close_raises(fake_gl: FakeGL, fake_allocator: FakeAllocator):
    manager = BufferManager(16, 16, GL_RGBA8)
    publish_frame(manager)
    manager.close()
    manager.close()  # repeated close is fine
    assert manager.is_closed
    assert fake_gl.live_fences == set()
    assert fake_allocator.live_textures == set()
    with (
        pytest.raises(RuntimeError, match="closed"),
        manager.acquire_write_buffer(),
    ):
        pass
    with pytest.raises(RuntimeError, match="closed"), manager.acquire_read_buffer():
        pass


# ---- frame size and reallocation ----


def test_smaller_frame_reuses_allocation(manager: Any, fake_allocator: FakeAllocator):
    with manager.acquire_write_buffer() as buffer:
        texture_before = buffer.texture_id
        assert buffer.set_frame_size(8, 4, GL_RGBA8) is False
        assert buffer.texture_id == texture_before
        assert (buffer.allocated_width, buffer.allocated_height) == (16, 16)
        assert (buffer.frame_width, buffer.frame_height) == (8, 4)


def test_larger_frame_reallocates(manager: Any, fake_allocator: FakeAllocator):
    with manager.acquire_write_buffer() as buffer:
        texture_before, fbo_before = buffer.texture_id, buffer.fbo_id
        assert buffer.set_frame_size(32, 8, GL_RGBA8) is True
        assert buffer.texture_id != texture_before
        assert texture_before not in fake_allocator.live_textures
        assert fbo_before not in fake_allocator.live_fbos
        assert (buffer.allocated_width, buffer.allocated_height) == (32, 16)
        assert (buffer.frame_width, buffer.frame_height) == (32, 8)


def test_pixel_format_change_reallocates(manager: Any):
    with manager.acquire_write_buffer() as buffer:
        assert buffer.set_frame_size(8, 8, GL_RGBA16F) is True
        assert buffer.pixel_format == GL_RGBA16F


@pytest.mark.parametrize("size", [(0, 16), (16, 0), (-1, 16)])
def test_invalid_frame_size_leaves_buffer_unchanged(
    manager: Any, size: tuple[int, int]
):
    with (
        pytest.raises(ValueError, match="Invalid buffer size"),
        manager.acquire_write_buffer() as buffer,
    ):
        texture_before = buffer.texture_id
        buffer.set_frame_size(*size, GL_RGBA8)
    assert buffer.texture_id == texture_before
    assert buffer.is_allocated


def test_incomplete_fbo_leaves_buffer_unchanged(
    manager: Any, fake_allocator: FakeAllocator
):
    fake_allocator.fail_next_fbo = True
    with (
        pytest.raises(RuntimeError, match="incomplete FBO"),
        manager.acquire_write_buffer() as buffer,
    ):
        texture_before, fbo_before = buffer.texture_id, buffer.fbo_id
        buffer.set_frame_size(64, 64, GL_RGBA8)
    assert (buffer.texture_id, buffer.fbo_id) == (texture_before, fbo_before)
    assert (buffer.allocated_width, buffer.allocated_height) == (16, 16)
    # the new texture was deleted again: 3 buffers, 3 textures
    assert len(fake_allocator.live_textures) == 3


# ---- hints ----


def test_write_hints(manager: Any):
    assert not manager.write_buffer_is_held
    assert manager.write_started_at is None
    before = time.perf_counter()
    with manager.acquire_write_buffer():
        assert manager.write_buffer_is_held
        assert manager.write_started_at >= before
    assert not manager.write_buffer_is_held
    assert manager.write_started_at is None


# ---- threaded: C's optional wait ----


class HeldWrite:
    """Holds a write block on another thread until `finish()` is called."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._started = threading.Event()
        self._finish = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with self._manager.acquire_write_buffer() as buffer:
            buffer.set_frame_size(16, 16, GL_RGBA8)
            self._started.set()
            self._finish.wait(timeout=10)

    def __enter__(self) -> HeldWrite:  # noqa: PYI034
        self._thread.start()
        assert self._started.wait(timeout=5)
        return self

    def finish(self, delay: float = 0.0) -> None:
        if delay:
            threading.Timer(delay, self._finish.set).start()
        else:
            self._finish.set()

    def __exit__(self, *exc_info: object) -> None:
        self._finish.set()
        self._thread.join(timeout=5)


def test_zero_timeout_does_not_wait(manager: Any):
    publish_frame(manager)
    with HeldWrite(manager):
        started = time.perf_counter()
        with manager.acquire_read_buffer(timeout=0) as frame:
            elapsed = time.perf_counter() - started
        assert frame.frame_serial == 1
        assert elapsed < 0.05


def test_no_wait_when_writer_is_idle(manager: Any):
    publish_frame(manager)
    started = time.perf_counter()
    with manager.acquire_read_buffer(timeout=5) as frame:
        elapsed = time.perf_counter() - started
    assert frame.frame_serial == 1
    assert elapsed < 0.5


def test_reader_waits_for_frame_in_flight(manager: Any):
    publish_frame(manager)
    with HeldWrite(manager) as held_write:
        held_write.finish(delay=0.05)
        started = time.perf_counter()
        with manager.acquire_read_buffer(timeout=5) as frame:
            elapsed = time.perf_counter() - started
    assert frame.frame_serial == 2
    assert 0.03 < elapsed < 2


def test_reader_wait_times_out(manager: Any):
    publish_frame(manager)
    with HeldWrite(manager):
        started = time.perf_counter()
        with manager.acquire_read_buffer(timeout=0.05) as frame:
            elapsed = time.perf_counter() - started
        assert frame.frame_serial == 1  # the frame P finished before
        assert 0.03 < elapsed < 2


def test_reader_gets_published_frame_when_writer_restarts(manager: Any):
    """P publishes and starts the next frame before C wakes: C gets the published one."""
    publish_frame(manager)
    second_write_started = threading.Event()
    release_second_write = threading.Event()
    release_first_write = threading.Event()
    first_write_started = threading.Event()

    def writer() -> None:
        with manager.acquire_write_buffer():
            first_write_started.set()
            release_first_write.wait(timeout=5)
        with manager.acquire_write_buffer():
            second_write_started.set()
            release_second_write.wait(timeout=5)

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()
    assert first_write_started.wait(timeout=5)
    threading.Timer(0.05, release_first_write.set).start()

    with manager.acquire_read_buffer(timeout=5) as frame:
        assert frame.frame_serial == 2

    assert second_write_started.wait(timeout=5)
    release_second_write.set()
    writer_thread.join(timeout=5)


def test_close_wakes_waiting_reader(fake_gl: FakeGL, fake_allocator: FakeAllocator):
    manager = BufferManager(16, 16, GL_RGBA8)
    publish_frame(manager)
    read_results: list[Any] = []

    def reader() -> None:
        with manager.acquire_read_buffer(timeout=5) as frame:
            read_results.append(frame)

    with HeldWrite(manager):
        reader_thread = threading.Thread(target=reader, daemon=True)
        started = time.perf_counter()
        reader_thread.start()
        time.sleep(0.05)
        manager.close()
        reader_thread.join(timeout=5)
        elapsed = time.perf_counter() - started
    assert read_results == [None]
    assert elapsed < 2


# ---- GL integration: real shared contexts ----


@pytest.fixture(scope="module", name="gl_windows")
def fixture_gl_windows() -> Iterator[tuple[Any, Any]]:
    """A hidden main window and a shadow window sharing its context."""
    try:
        import pyglet

        main_window = pyglet.window.Window(64, 64, visible=False)
        shadow_window = utils_module.create_shadow_window(main_window)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL context available: {error}")
    yield main_window, shadow_window
    shadow_window.close()
    main_window.close()


def test_gl_two_threads_never_show_wrong_frame(gl_windows: tuple[Any, Any]):
    """P clears each frame to a colour encoding its serial, C checks every read.

    Random GPU workloads and resizes stress the fence handoff between contexts.
    """
    from pyglet import gl

    main_window, shadow_window = gl_windows
    utils_module.make_gl_context_current(main_window)
    manager = BufferManager(16, 16, GL_RGBA8)
    frames_to_produce = 300
    producer_errors: list[BaseException] = []
    rng = random.Random(1)

    def producer() -> None:
        try:
            utils_module.make_gl_context_current(shadow_window)
            frame_size = 16
            for _ in range(frames_to_produce):
                with manager.acquire_write_buffer() as buffer:
                    if rng.random() < 0.05:
                        frame_size = rng.choice((8, 16, 24, 32))
                    buffer.set_frame_size(frame_size, frame_size, GL_RGBA8)
                    serial_after_publish = manager.published_frame_count + 1
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, buffer.fbo_id)
                    gl.glViewport(0, 0, frame_size, frame_size)
                    gl.glClearColor((serial_after_publish % 256) / 255.0, 0, 0, 1)
                    for _ in range(rng.randint(1, 50)):  # vary the GPU work
                        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                time.sleep(rng.uniform(0, 0.003))
        except BaseException as error:  # noqa: BLE001
            producer_errors.append(error)
        finally:
            utils_module.release_gl_context()

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    mismatches: list[str] = []
    frames_read = 0
    utils_module.make_gl_context_current(main_window)
    while producer_thread.is_alive() or frames_read == 0:
        with manager.acquire_read_buffer(
            timeout=rng.choice((0, 0.001, 0.005))
        ) as frame:
            if frame is None:
                continue
            frames_read += 1
            buffer = frame.buffer
            pixels = (
                ctypes.c_ubyte * (buffer.allocated_width * buffer.allocated_height * 4)
            )()
            gl.glBindTexture(gl.GL_TEXTURE_2D, buffer.texture_id)
            gl.glGetTexImage(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            if pixels[0] != frame.frame_serial % 256:
                mismatches.append(f"frame {frame.frame_serial}: red={pixels[0]}")
        time.sleep(rng.uniform(0, 0.002))

    producer_thread.join(timeout=10)
    manager.close()
    assert producer_errors == []
    assert manager.published_frame_count == frames_to_produce
    assert frames_read > 0
    assert mismatches == []
