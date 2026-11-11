"""Tests for `mpvmoviestim.buffer`: FrameBuffer and TripleBufferManager.

Three groups:

1. Logic tests with fake GL and allocation functions: buffer selection, fence
   handoff, shutdown and error paths, resizing. No GPU needed.
2. Threaded timing tests, also with fake GL: C's optional wait for P.
3. A GL integration test with real shared contexts, skipped if no GL context
   can be created.

The tests only use the public API, as P and C would.
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
TripleBufferManager = buffer_module.TripleBufferManager

GL_RGBA8 = 0x8058
GL_RGBA16F = 0x881A
SIZE = 16


# ---- fakes ----


class FakeGL:
    """Records fence calls, and which thread role placed and waited on each fence.

    Tests set `role` ("P" or "C") before acting as that thread; the threaded
    tests leave it alone and don't check roles.
    """

    GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
    GL_TIMEOUT_IGNORED = 0xFFFFFFFFFFFFFFFF

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_fence = 1
        self.role = "?"
        self.live_fences: set[int] = set()
        self.fence_creator: dict[int, str] = {}
        self.waits: list[tuple[str, int]] = []  # (waiting role, fence)
        self.flush_count = 0
        self.fail_next_wait = False

    def glFenceSync(self, condition: int, flags: int) -> int:
        assert condition == self.GL_SYNC_GPU_COMMANDS_COMPLETE
        assert flags == 0
        with self._lock:
            fence = self._next_fence
            self._next_fence += 1
            self.live_fences.add(fence)
            self.fence_creator[fence] = self.role
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
            self.waits.append((self.role, fence))

    def glDeleteSync(self, fence: int) -> None:
        with self._lock:
            assert fence in self.live_fences, f"fence {fence} deleted twice"
            self.live_fences.remove(fence)


class FakeAllocator:
    """Stands in for the texture and FBO helpers of utils.py."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_name = 1
        self.live_textures: dict[int, tuple[int, int, int]] = {}  # id -> w, h, format
        self.live_fbos: set[int] = set()
        self.textures_until_failure: int | None = None
        self.fail_next_fbo = False

    def gl_texture_create(self, width: int, height: int, pixel_format: int) -> int:
        with self._lock:
            if self.textures_until_failure is not None:
                if self.textures_until_failure == 0:
                    raise ValueError("simulated texture allocation failure")
                self.textures_until_failure -= 1
            texture_id = self._next_name
            self._next_name += 1
            self.live_textures[texture_id] = (width, height, pixel_format)
        return texture_id

    def gl_texture_destroy(self, texture_id: int) -> None:
        with self._lock:
            del self.live_textures[texture_id]

    def gl_fbo_create(self, texture_id: int) -> int:
        with self._lock:
            assert texture_id in self.live_textures
            if self.fail_next_fbo:
                self.fail_next_fbo = False
                raise RuntimeError("simulated incomplete FBO")
            fbo_id = self._next_name
            self._next_name += 1
            self.live_fbos.add(fbo_id)
        return fbo_id

    def gl_fbo_destroy(self, fbo_id: int) -> None:
        with self._lock:
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


def assert_everything_freed(fake_gl: FakeGL, fake_allocator: FakeAllocator) -> None:
    assert fake_gl.live_fences == set()
    assert fake_allocator.live_textures == {}
    assert fake_allocator.live_fbos == set()


@pytest.fixture(name="manager")
def fixture_manager(fake_gl: FakeGL, fake_allocator: FakeAllocator) -> Iterator[Any]:
    """A manager that is shut down after the test; checks that nothing leaked."""
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    yield manager
    manager.shutdown()
    assert_everything_freed(fake_gl, fake_allocator)


def publish_frame(manager: Any, width: int = SIZE, height: int = SIZE) -> Any:
    """Render and publish one (fake) frame as P, return its buffer."""
    buffer = manager.acquire_write_buffer(width, height, GL_RGBA8)
    manager.release_write_buffer()
    return buffer


def read_frame(manager: Any, timeout: float = 0.0) -> Any:
    """Acquire and release a frame as C, return its buffer (or None)."""
    buffer = manager.acquire_read_buffer(timeout)
    if buffer is not None:
        manager.release_read_buffer()
    return buffer


# ---- construction ----


def test_constructor_allocates_three_buffers(
    manager: Any, fake_allocator: FakeAllocator
):
    assert sorted(fake_allocator.live_textures.values()) == [(SIZE, SIZE, GL_RGBA8)] * 3
    assert len(fake_allocator.live_fbos) == 3


def test_constructor_failure_frees_allocated_buffers(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    fake_allocator.textures_until_failure = 2  # the third texture fails
    with pytest.raises(ValueError, match="simulated"):
        TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    assert_everything_freed(fake_gl, fake_allocator)


def test_constructor_frees_texture_of_incomplete_fbo(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    fake_allocator.fail_next_fbo = True
    with pytest.raises(RuntimeError, match="incomplete FBO"):
        TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    assert_everything_freed(fake_gl, fake_allocator)


# ---- reading and writing ----


def test_read_before_first_frame_gets_none(manager: Any):
    assert manager.acquire_read_buffer() is None
    assert manager.acquire_read_buffer(timeout=1.0) is None  # P idle: no wait
    # nothing was claimed, so releasing raises
    with pytest.raises(RuntimeError, match="does not hold"):
        manager.release_read_buffer()


def test_published_frame_is_read(manager: Any):
    written = manager.acquire_write_buffer(10, 12, GL_RGBA8)
    assert written.frame_fbo_info() == {
        "fbo": written.fbo_id,
        "w": 10,
        "h": 12,
        "internal_format": GL_RGBA8,
    }
    manager.release_write_buffer()

    read = manager.acquire_read_buffer()
    assert read is written
    assert (read.frame_width, read.frame_height) == (10, 12)
    manager.release_read_buffer()


def test_reader_gets_freshest_frame(manager: Any):
    publish_frame(manager)
    publish_frame(manager)
    newest = publish_frame(manager)
    assert read_frame(manager) is newest
    assert read_frame(manager) is newest  # repeated read: same frame


def test_writer_takes_oldest_buffer_not_held_by_reader(manager: Any):
    first = publish_frame(manager)
    second = publish_frame(manager)
    third = publish_frame(manager)
    assert len({id(first), id(second), id(third)}) == 3

    # C holds nothing: P takes the oldest
    assert publish_frame(manager) is first  # order now: first, third, second

    # C holds the freshest (first): P takes the oldest (second)
    assert manager.acquire_read_buffer() is first
    assert publish_frame(manager) is second  # order: second, first, third
    manager.release_read_buffer()

    # C holds the oldest (third): P takes the second oldest (first)
    publish_frame(manager)  # third: order: third, second, first
    assert manager.acquire_read_buffer() is third
    publish_frame(manager)  # first: order: first, third, second
    publish_frame(manager)  # second: order: second, first, third (C holds third)
    assert manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8) is first
    manager.release_write_buffer()
    manager.release_read_buffer()


def test_reader_gets_freshest_while_writer_renders(manager: Any):
    newest = publish_frame(manager)
    being_written = manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
    assert being_written is not newest
    assert manager.acquire_read_buffer() is newest
    manager.release_read_buffer()
    manager.release_write_buffer()


@pytest.mark.parametrize("seed", range(5))
def test_random_interleavings_keep_invariants(
    seed: int, fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    """Random single-threaded interleavings of P and C, checked step by step.

    Checks buffer selection and that each side's GPU queue waits exactly on
    the other side's fences, using only what P and C can observe.
    """
    rng = random.Random(seed)
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    publish_index: dict[int, int] = {}  # id(buffer) -> when it was last published
    last_fence_role: dict[int, str] = {}  # id(buffer) -> role of its last fence
    texture_seen: dict[int, int] = {}  # id(buffer) -> texture_id when P last took it
    writer_buffer = None
    reader_buffer = None
    published_count = 0

    def allowed_for_writer(buffer: Any) -> bool:
        freshest = max(publish_index, key=publish_index.__getitem__, default=None)
        return id(buffer) != freshest and buffer is not reader_buffer

    for _ in range(2000):
        action = rng.choice(("write", "read"))
        waits_before = len(fake_gl.waits)

        if action == "write" and writer_buffer is None:
            fake_gl.role = "P"
            size = rng.choice((8, 16, 24))
            writer_buffer = manager.acquire_write_buffer(size, SIZE, GL_RGBA8)
            assert allowed_for_writer(writer_buffer)
            # a reallocated buffer has a new texture, and its fence is dropped
            previous_texture = texture_seen.get(id(writer_buffer))
            reallocated = previous_texture not in (None, writer_buffer.texture_id)
            texture_seen[id(writer_buffer)] = writer_buffer.texture_id
            if reallocated:
                last_fence_role.pop(id(writer_buffer), None)
            expected_wait = last_fence_role.get(id(writer_buffer)) == "C"
            assert len(fake_gl.waits) - waits_before == int(expected_wait)
            assert writer_buffer.frame_width == size

        elif action == "write":
            fake_gl.role = "P"
            manager.release_write_buffer()
            published_count += 1
            publish_index[id(writer_buffer)] = published_count
            last_fence_role[id(writer_buffer)] = "P"
            writer_buffer = None

        elif reader_buffer is None:
            fake_gl.role = "C"
            reader_buffer = manager.acquire_read_buffer()
            if published_count == 0:
                assert reader_buffer is None
                continue
            freshest = max(publish_index, key=publish_index.__getitem__)
            assert id(reader_buffer) == freshest
            expected_wait = last_fence_role.get(id(reader_buffer)) == "P"
            assert len(fake_gl.waits) - waits_before == int(expected_wait)

        else:
            fake_gl.role = "C"
            manager.release_read_buffer()
            last_fence_role[id(reader_buffer)] = "C"
            reader_buffer = None

        # every wait is on the other side's fence
        for waiting_role, fence in fake_gl.waits[waits_before:]:
            assert fake_gl.fence_creator[fence] != waiting_role

    # P shuts down (also freeing its own buffer), C releases afterwards
    manager.shutdown()
    if reader_buffer is not None:
        manager.release_read_buffer()
    assert_everything_freed(fake_gl, fake_allocator)


# ---- fences ----


def test_reader_waits_on_writers_fence_only(manager: Any, fake_gl: FakeGL):
    fake_gl.role = "P"
    publish_frame(manager)
    fake_gl.role = "C"
    read_frame(manager)
    read_frame(manager)  # C's own fence: deleted, but not waited on
    assert [fake_gl.fence_creator[fence] for _, fence in fake_gl.waits] == ["P"]


def test_writer_waits_on_readers_fence_only(manager: Any, fake_gl: FakeGL):
    fake_gl.role = "P"
    first = publish_frame(manager)
    fake_gl.role = "C"
    read_frame(manager)  # first now carries C's fence
    fake_gl.role = "P"
    publish_frame(manager)
    publish_frame(manager)
    waits_before = len(fake_gl.waits)
    assert manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8) is first
    new_waits = fake_gl.waits[waits_before:]
    assert [fake_gl.fence_creator[fence] for _, fence in new_waits] == ["C"]
    manager.release_write_buffer()


def test_writer_skips_own_fence(manager: Any, fake_gl: FakeGL):
    # C never reads: every buffer P reuses carries P's own fence
    fake_gl.role = "P"
    for _ in range(6):
        publish_frame(manager)
    assert fake_gl.waits == []


def test_reallocation_drops_fence_without_waiting(manager: Any, fake_gl: FakeGL):
    fake_gl.role = "P"
    first = publish_frame(manager)
    fake_gl.role = "C"
    read_frame(manager)  # first carries C's fence
    fake_gl.role = "P"
    publish_frame(manager)
    publish_frame(manager)
    assert manager.acquire_write_buffer(2 * SIZE, SIZE, GL_RGBA8) is first
    assert fake_gl.waits == [("C", 1)]  # only C's earlier wait on P's first fence
    manager.release_write_buffer()


def test_every_release_flushes(manager: Any, fake_gl: FakeGL):
    publish_frame(manager)
    read_frame(manager)
    assert fake_gl.flush_count == 2


# ---- shutdown and errors ----


def test_shutdown_is_idempotent_and_blocks_further_use(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    publish_frame(manager)
    manager.shutdown()
    manager.shutdown()
    assert_everything_freed(fake_gl, fake_allocator)
    with pytest.raises(RuntimeError, match="not active"):
        manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
    with pytest.raises(RuntimeError, match="not active"):
        manager.acquire_read_buffer()
    manager.release_write_buffer()  # does nothing after shutdown


def test_failed_render_is_never_shown_and_freed_by_shutdown(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    shown = publish_frame(manager)

    # P's render fails: P doesn't release, its loop-level finally shuts down
    with pytest.raises(KeyError):
        try:
            failed = manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
            assert manager.acquire_read_buffer() is shown  # C never gets `failed`
            manager.release_read_buffer()
            raise KeyError("render failed")
        finally:
            manager.shutdown()

    assert failed.texture_id == 0  # P's own buffer was freed too
    assert_everything_freed(fake_gl, fake_allocator)


def test_shutdown_leaves_readers_buffer_to_reader(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    publish_frame(manager)
    held_by_reader = manager.acquire_read_buffer()

    manager.shutdown()
    assert held_by_reader.texture_id in fake_allocator.live_textures  # still drawable
    assert len(fake_allocator.live_textures) == 1

    manager.release_read_buffer()  # C frees it
    assert held_by_reader.texture_id == 0
    assert_everything_freed(fake_gl, fake_allocator)


def test_failed_reallocation_then_shutdown_frees_everything(
    fake_gl: FakeGL, fake_allocator: FakeAllocator
):
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    fake_allocator.fail_next_fbo = True
    try:
        with pytest.raises(RuntimeError, match="incomplete FBO"):
            manager.acquire_write_buffer(2 * SIZE, 2 * SIZE, GL_RGBA8)
    finally:
        manager.shutdown()
    assert_everything_freed(fake_gl, fake_allocator)


@pytest.mark.parametrize("size", [(0, SIZE), (SIZE, 0), (-1, SIZE)])
def test_invalid_frame_size_raises(manager: Any, size: tuple[int, int]):
    with pytest.raises(ValueError, match="positive"):
        manager.acquire_write_buffer(*size, GL_RGBA8)


def test_misuse_raises(manager: Any):
    with pytest.raises(RuntimeError, match="does not hold"):
        manager.release_write_buffer()
    with pytest.raises(RuntimeError, match="does not hold"):
        manager.release_read_buffer()

    manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
    with pytest.raises(RuntimeError, match="already holds"):
        manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
    manager.release_write_buffer()

    manager.acquire_read_buffer()
    with pytest.raises(RuntimeError, match="already holds"):
        manager.acquire_read_buffer()
    manager.release_read_buffer()


# ---- frame size and reallocation ----


def test_smaller_frame_reuses_allocation(manager: Any, fake_allocator: FakeAllocator):
    buffer = manager.acquire_write_buffer(8, 4, GL_RGBA8)
    assert fake_allocator.live_textures[buffer.texture_id] == (SIZE, SIZE, GL_RGBA8)
    assert (buffer.allocated_width, buffer.allocated_height) == (SIZE, SIZE)
    assert (buffer.frame_width, buffer.frame_height) == (8, 4)
    manager.release_write_buffer()


def test_larger_frame_grows_each_dimension(manager: Any, fake_allocator: FakeAllocator):
    textures_before = set(fake_allocator.live_textures)
    buffer = manager.acquire_write_buffer(2 * SIZE, SIZE // 2, GL_RGBA8)
    assert buffer.texture_id not in textures_before
    assert len(fake_allocator.live_textures) == 3  # the old texture was freed
    # width grows, height keeps the larger allocated size
    assert (buffer.allocated_width, buffer.allocated_height) == (2 * SIZE, SIZE)
    assert (buffer.frame_width, buffer.frame_height) == (2 * SIZE, SIZE // 2)
    manager.release_write_buffer()


def test_pixel_format_change_reallocates_once(manager: Any):
    publish_frame(manager)
    publish_frame(manager)
    buffer = manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA16F)
    texture_after_change = buffer.texture_id
    assert buffer.pixel_format == GL_RGBA16F
    manager.release_write_buffer()
    # cycle back to the same buffer: same format again doesn't reallocate
    for _ in range(2):
        manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA16F)
        manager.release_write_buffer()
    assert manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA16F) is buffer
    assert buffer.texture_id == texture_after_change
    manager.release_write_buffer()


# ---- hint ----


def test_write_buffer_is_held(manager: Any):
    assert not manager.write_buffer_is_held
    manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
    assert manager.write_buffer_is_held
    manager.release_write_buffer()
    assert not manager.write_buffer_is_held


# ---- threaded: C's optional wait ----


class HeldWrite:
    """P on another thread: holds a buffer until told to release or shut down."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._acquired = threading.Event()
        self._go = threading.Event()
        self._shutdown_instead = False
        self.buffer: Any = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.buffer = self._manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8)
        self._acquired.set()
        self._go.wait(timeout=10)
        if self._shutdown_instead:
            self._manager.shutdown()
        else:
            self._manager.release_write_buffer()

    def __enter__(self) -> HeldWrite:  # noqa: PYI034
        self._thread.start()
        assert self._acquired.wait(timeout=5)
        return self

    def release(self, delay: float = 0.0) -> None:
        threading.Timer(delay, self._go.set).start()

    def shut_down(self, delay: float = 0.0) -> None:
        self._shutdown_instead = True
        threading.Timer(delay, self._go.set).start()

    def __exit__(self, *exc_info: object) -> None:
        self._go.set()
        self._thread.join(timeout=5)


def timed_read(manager: Any, timeout: float) -> tuple[Any, float]:
    """Acquire and release as C; return the buffer and how long acquiring took."""
    started = time.perf_counter()
    buffer = manager.acquire_read_buffer(timeout)
    elapsed = time.perf_counter() - started
    if buffer is not None:
        manager.release_read_buffer()
    return buffer, elapsed


def test_zero_timeout_does_not_wait(manager: Any):
    previous = publish_frame(manager)
    with HeldWrite(manager):
        buffer, elapsed = timed_read(manager, timeout=0)
    assert buffer is previous
    assert elapsed < 0.05


def test_no_wait_when_writer_is_idle(manager: Any):
    previous = publish_frame(manager)
    buffer, elapsed = timed_read(manager, timeout=5)
    assert buffer is previous
    assert elapsed < 0.5


def test_reader_waits_for_frame_in_flight(manager: Any):
    publish_frame(manager)
    with HeldWrite(manager) as held_write:
        held_write.release(delay=0.05)
        buffer, elapsed = timed_read(manager, timeout=5)
        assert buffer is held_write.buffer
    assert 0.03 < elapsed < 2


def test_reader_wait_times_out(manager: Any):
    previous = publish_frame(manager)
    with HeldWrite(manager):
        buffer, elapsed = timed_read(manager, timeout=0.05)
    assert buffer is previous  # the frame P finished before
    assert 0.03 < elapsed < 2


def test_reader_waits_for_first_frame(manager: Any):
    with HeldWrite(manager) as held_write:
        held_write.release(delay=0.05)
        buffer, _ = timed_read(manager, timeout=5)
        assert buffer is held_write.buffer


def test_first_frame_wait_times_out_with_none(manager: Any):
    with HeldWrite(manager):
        buffer, elapsed = timed_read(manager, timeout=0.05)
    assert buffer is None
    assert 0.03 < elapsed < 2


def test_reader_gets_awaited_frame_when_writer_restarts(manager: Any):
    """P publishes and starts the next frame before C wakes: C gets the published one."""
    publish_frame(manager)
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()
    release_second = threading.Event()
    written: list[Any] = []

    def writer() -> None:
        written.append(manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8))
        first_acquired.set()
        release_first.wait(timeout=5)
        manager.release_write_buffer()
        written.append(manager.acquire_write_buffer(SIZE, SIZE, GL_RGBA8))
        second_acquired.set()
        release_second.wait(timeout=5)
        manager.release_write_buffer()

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()
    assert first_acquired.wait(timeout=5)
    threading.Timer(0.05, release_first.set).start()

    buffer, _ = timed_read(manager, timeout=5)
    assert buffer is written[0]

    assert second_acquired.wait(timeout=5)
    release_second.set()
    writer_thread.join(timeout=5)


def test_shutdown_wakes_waiting_reader(manager: Any):
    publish_frame(manager)
    with HeldWrite(manager) as held_write:
        held_write.shut_down(delay=0.05)
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="not active"):
            manager.acquire_read_buffer(timeout=5)
        assert time.perf_counter() - started < 2


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
    """P clears each frame to a colour encoding its number, C checks every read.

    Random GPU workloads and resizes stress the fence handoff between contexts.
    """
    from pyglet import gl

    main_window, shadow_window = gl_windows
    utils_module.make_gl_context_current(main_window)
    manager = TripleBufferManager(SIZE, SIZE, GL_RGBA8)
    frames_to_produce = 300
    # frame number per buffer, set by P before publishing. C reads the entry of
    # the buffer it holds, which P can't touch until C releases it.
    frame_number_in_buffer: dict[int, int] = {}
    producer_done = threading.Event()
    consumer_done = threading.Event()
    producer_errors: list[BaseException] = []

    def producer() -> None:
        rng = random.Random(1)
        utils_module.make_gl_context_current(shadow_window)
        try:
            frame_size = SIZE
            for frame_number in range(1, frames_to_produce + 1):
                if rng.random() < 0.05:
                    frame_size = rng.choice((8, 16, 24, 32))
                buffer = manager.acquire_write_buffer(frame_size, frame_size, GL_RGBA8)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, buffer.fbo_id)
                gl.glViewport(0, 0, frame_size, frame_size)
                gl.glClearColor((frame_number % 256) / 255.0, 0, 0, 1)
                for _ in range(rng.randint(1, 50)):  # vary the GPU work
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                frame_number_in_buffer[id(buffer)] = frame_number
                manager.release_write_buffer()
                time.sleep(rng.uniform(0, 0.003))
            producer_done.set()
            consumer_done.wait(timeout=10)
        except BaseException as error:  # noqa: BLE001
            producer_errors.append(error)
        finally:
            manager.shutdown()
            utils_module.release_gl_context()

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    rng = random.Random(2)
    mismatches: list[str] = []
    frames_read = 0
    last_frame_number = 0
    utils_module.make_gl_context_current(main_window)
    while not producer_done.is_set() and producer_thread.is_alive():
        buffer = manager.acquire_read_buffer(timeout=rng.choice((0, 0.001, 0.005)))
        if buffer is None:
            continue
        try:
            frames_read += 1
            expected = frame_number_in_buffer[id(buffer)]
            if expected < last_frame_number:
                mismatches.append(f"frame {expected} after {last_frame_number}")
            last_frame_number = expected
            pixels = (
                ctypes.c_ubyte * (buffer.allocated_width * buffer.allocated_height * 4)
            )()
            gl.glBindTexture(gl.GL_TEXTURE_2D, buffer.texture_id)
            gl.glGetTexImage(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            if pixels[0] != expected % 256:
                mismatches.append(f"frame {expected}: red={pixels[0]}")
        finally:
            manager.release_read_buffer()
        time.sleep(rng.uniform(0, 0.002))

    consumer_done.set()
    producer_thread.join(timeout=10)
    assert producer_errors == []
    assert frames_read > 0
    assert mismatches == []
