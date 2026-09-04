"""Unit tests for the profiling module (no GL context or window required).

GL-dependent parts (GpuTimerPool query lifecycle) are not covered here as
they require a current OpenGL context.
"""

from __future__ import annotations

import csv
from time import perf_counter

import pytest

from mpvmoviestim.profiling import (
    Profiler,
    Render_Timestamp_Indices,
    ThreadRecorder,
    Worker_Timestamp_Indices,
)


class TestThreadRecorder:
    def test_row_advancement(self):
        rec = ThreadRecorder(capacity=3, n_slots=2)
        assert rec.next_iter() == 0
        assert rec.next_iter() == 2
        assert rec.next_iter() == 4
        assert rec.rows == 3

    def test_stop_when_full(self):
        rec = ThreadRecorder(capacity=2, n_slots=2)
        assert rec.next_iter() == 0
        assert rec.next_iter() == 2
        assert rec.next_iter() == -1
        assert rec.active is False
        # stays inactive
        assert rec.next_iter() == -1
        assert rec.rows == 2

    def test_zero_capacity_is_inactive_immediately(self):
        rec = ThreadRecorder(capacity=0, n_slots=4)
        assert rec.next_iter() == -1
        assert rec.active is False

    def test_buffer_preallocated_zeroed(self):
        rec = ThreadRecorder(capacity=5, n_slots=3)
        assert len(rec.buf) == 15
        assert all(v == 0.0 for v in rec.buf)


class TestWakeLog:
    def test_drain_wakes_records_all_with_trigger_flag(self):
        prof = Profiler(capacity=4, wake_factor=2)
        rec = prof.worker
        base = rec.next_iter()
        prof.wake_times.append((100.0, True))  # the trigger
        prof.wake_times.append((100.001, False))
        prof.wake_times.append((100.002, False))
        prof.drain_wakes(rec.buf, base)

        assert rec.buf[base + Worker_Timestamp_Indices.WAKE_IDX] == 0.0
        assert rec.buf[base + Worker_Timestamp_Indices.WAKE_COUNT] == 3.0
        assert prof.wake_log_len == 3
        assert list(prof.wake_log_t[:3]) == [100.0, 100.001, 100.002]
        assert list(prof.wake_log_trigger[:3]) == [1, 0, 0]
        assert len(prof.wake_times) == 0

    def test_drain_wakes_empty(self):
        prof = Profiler(capacity=4)
        rec = prof.worker
        base = rec.next_iter()
        prof.drain_wakes(rec.buf, base)
        assert rec.buf[base + Worker_Timestamp_Indices.WAKE_IDX] == -1.0
        assert rec.buf[base + Worker_Timestamp_Indices.WAKE_COUNT] == 0.0

    def test_wake_log_stops_when_full_but_count_continues(self):
        prof = Profiler(capacity=1, wake_factor=2)  # wake log capacity = 2
        rec = prof.worker
        base = rec.next_iter()
        for i in range(4):
            prof.wake_times.append((100.0 + i * 0.001, i == 0))
        prof.drain_wakes(rec.buf, base)
        # all 4 counted, only 2 stored
        assert rec.buf[base + Worker_Timestamp_Indices.WAKE_COUNT] == 4.0
        assert prof.wake_log_len == 2

    def test_wakes_accumulate_across_iterations(self):
        prof = Profiler(capacity=4, wake_factor=4)
        rec = prof.worker
        b0 = rec.next_iter()
        prof.wake_times.append((100.0, True))
        prof.drain_wakes(rec.buf, b0)
        b1 = rec.next_iter()
        prof.wake_times.append((101.0, True))
        prof.wake_times.append((101.5, False))
        prof.drain_wakes(rec.buf, b1)
        assert (
            rec.buf[b1 + Worker_Timestamp_Indices.WAKE_IDX] == 1.0
        )  # second iteration's wakes start at 1
        assert rec.buf[b1 + Worker_Timestamp_Indices.WAKE_COUNT] == 2.0


class TestGetEvents:
    def _make_profiler(self) -> Profiler:
        return Profiler(capacity=8, wake_factor=4)

    def test_empty_when_nothing_recorded(self):
        assert self._make_profiler().get_events() == []

    def test_merged_sorted_by_time_across_threads(self):
        prof = self._make_profiler()
        wb = prof.worker.buf
        mb = prof.main.buf
        assert prof.worker.next_iter() == 0
        assert prof.main.next_iter() == 0
        # worker row
        wb[Worker_Timestamp_Indices.UPDATE_T0] = 100.000
        wb[Worker_Timestamp_Indices.UPDATE_T1] = 100.001
        wb[Worker_Timestamp_Indices.RENDER_T0] = 100.002
        wb[Worker_Timestamp_Indices.RENDER_T1] = 100.004
        wb[Worker_Timestamp_Indices.FENCE_POST_T] = 100.005
        # main row: starts before worker (e.g. draw of previous cycle)
        mb[Render_Timestamp_Indices.DRAW_ENTRY_T] = 99.900
        mb[Render_Timestamp_Indices.BLIT_T0] = 99.910
        mb[Render_Timestamp_Indices.BLIT_T1] = 99.912
        mb[Render_Timestamp_Indices.DRAW_EXIT_T] = 99.913

        events = prof.get_events()
        times = [e[0] for e in events]
        assert times == sorted(times)
        # t0 anchored at first event (main draw_entry at 99.900)
        assert events[0] == (0.0, "main", "draw_entry", 0.0)
        by_name = {(e[1], e[2]): e for e in events}
        assert by_name[("worker", "update")][3] == pytest.approx(0.001)
        assert by_name[("worker", "render_cpu")][3] == pytest.approx(0.002)
        assert by_name[("main", "blit_cpu")][3] == pytest.approx(0.002)
        # relative placement: worker update starts 100 ms after draw entry
        assert by_name[("worker", "update")][0] == pytest.approx(0.100)

    def test_zero_slots_are_skipped(self):
        prof = self._make_profiler()
        wb = prof.worker.buf
        prof.worker.next_iter()
        wb[Worker_Timestamp_Indices.UPDATE_T0] = 100.0
        wb[Worker_Timestamp_Indices.UPDATE_T1] = 100.001
        # no other stamps: lock/render/etc. must not appear
        events = prof.get_events()
        names = [e[2] for e in events]
        assert names == ["update"]

    def test_duration_slots_not_mistaken_for_timestamps(self):
        prof = self._make_profiler()
        wb = prof.worker.buf
        prof.worker.next_iter()
        wb[Worker_Timestamp_Indices.RENDER_T0] = 5000.0
        wb[Worker_Timestamp_Indices.RENDER_T1] = 5000.001
        wb[Worker_Timestamp_Indices.GPU_RENDER] = (
            0.002  # duration, small - must not become t0
        )
        wb[Worker_Timestamp_Indices.WAITSYNC_BLIT_STATE] = (
            1.0  # state value - must not become t0
        )
        events = prof.get_events()
        # t0 must be 5000.0 (first real timestamp), so first event is at 0
        assert min(e[0] for e in events) == pytest.approx(0.0)
        gpu = [e for e in events if e[1] == "worker-gpu"]
        assert len(gpu) == 1
        assert gpu[0][2] == "render"
        assert gpu[0][3] == pytest.approx(0.002)

    def test_poll_state_events(self):
        prof = self._make_profiler()
        mb = prof.main.buf
        prof.main.next_iter()
        mb[Render_Timestamp_Indices.WAITSYNC_RENDER_T0] = 100.0
        mb[Render_Timestamp_Indices.WAITSYNC_RENDER_T1] = 100.0001
        mb[Render_Timestamp_Indices.WAITSYNC_RENDER_STATE] = 2.0  # not ready
        events = prof.get_events()
        names = [e[2] for e in events]
        assert "waitsync_render" in names
        assert "waitsync_render_poll:not_ready" in names

    def test_gpu_wait_and_iter_done(self):
        prof = self._make_profiler()
        wb = prof.worker.buf
        prof.worker.next_iter()
        wb[Worker_Timestamp_Indices.SET_DONE_T] = 100.0
        wb[Worker_Timestamp_Indices.WAIT_DONE_DUR] = (
            0.003  # CPU waited 3 ms for the GPU
        )
        wb[Worker_Timestamp_Indices.ITER_DONE_T] = 100.005
        events = prof.get_events()
        by_name = {e[2]: e for e in events}
        assert by_name["iter_done"][0] == pytest.approx(0.005)
        # gpu_wait starts 3 ms before iter_done
        assert by_name["gpu_wait"][0] == pytest.approx(0.002)
        assert by_name["gpu_wait"][3] == pytest.approx(0.003)

    def test_gpu_wait_timeout_marker(self):
        prof = self._make_profiler()
        mb = prof.main.buf
        prof.main.next_iter()
        mb[Render_Timestamp_Indices.ITER_DONE_T] = 100.0
        mb[Render_Timestamp_Indices.GPU_WAIT] = -1.0  # timeout/failure encoding
        events = prof.get_events()
        names = [e[2] for e in events]
        assert "gpu_wait_timeout" in names
        assert "gpu_wait" not in names

    def test_wake_events_in_output(self):
        prof = self._make_profiler()
        rec = prof.worker
        base = rec.next_iter()
        prof.wake_times.append((100.0, True))
        prof.wake_times.append((100.001, False))
        prof.drain_wakes(rec.buf, base)
        rec.buf[base + Worker_Timestamp_Indices.UPDATE_T0] = 100.002
        rec.buf[base + Worker_Timestamp_Indices.UPDATE_T1] = 100.003
        events = prof.get_events()
        wakes = [e for e in events if e[1] == "mpv-cb"]
        assert [e[2] for e in wakes] == ["wake_trigger", "wake"]
        # wake_trigger is the first event overall (earliest timestamp)
        assert events[0][2] == "wake_trigger"

    def test_report_swap_stamped_into_current_row(self):
        prof = self._make_profiler()
        rec = prof.main
        base = rec.next_iter()
        rec.buf[base + Render_Timestamp_Indices.DRAW_ENTRY_T] = 100.0
        # report_swap() writes into the *current* row without advancing it
        rec.buf[rec.base + Render_Timestamp_Indices.REPORT_SWAP_T] = 100.007
        events = prof.get_events()
        rs = [e for e in events if e[2] == "report_swap"]
        assert len(rs) == 1
        assert rs[0][0] == pytest.approx(0.007)


class TestExportCsv:
    def test_csv_roundtrip(self, tmp_path):
        prof = Profiler(capacity=8)
        mb = prof.main.buf
        prof.main.next_iter()
        mb[Render_Timestamp_Indices.DRAW_ENTRY_T] = 100.0
        mb[Render_Timestamp_Indices.BLIT_T0] = 100.001
        mb[Render_Timestamp_Indices.BLIT_T1] = 100.0025
        out = tmp_path / "prof.csv"
        prof.export_csv(out)

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["t_ms", "thread", "event", "duration_ms"]
        data = {r[2]: r for r in rows[1:]}
        assert data["draw_entry"][1] == "main"
        assert float(data["draw_entry"][0]) == pytest.approx(0.0)
        assert float(data["blit_cpu"][0]) == pytest.approx(1.0, abs=1e-3)  # ms
        assert float(data["blit_cpu"][3]) == pytest.approx(1.5, abs=1e-3)  # ms

    def test_csv_empty(self, tmp_path):
        prof = Profiler(capacity=8)
        out = tmp_path / "empty.csv"
        prof.export_csv(out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [["t_ms", "thread", "event", "duration_ms"]]


class TestRecorderPerfSmoke:
    """Sanity check that the hot-path write pattern is allocation-free and fast."""

    def test_stamp_write_speed(self):
        rec = ThreadRecorder(capacity=10_000, n_slots=Worker_Timestamp_Indices.COUNT)
        buf = rec.buf
        t0 = perf_counter()
        for _ in range(10_000):
            base = rec.next_iter()
            if base >= 0:
                buf[base + Worker_Timestamp_Indices.UPDATE_T0] = perf_counter()
                buf[base + Worker_Timestamp_Indices.UPDATE_T1] = perf_counter()
                buf[base + Worker_Timestamp_Indices.RENDER_T0] = perf_counter()
                buf[base + Worker_Timestamp_Indices.RENDER_T1] = perf_counter()
        elapsed = perf_counter() - t0
        # 10k iterations x 4 stamps: must be far below a single refresh cycle total
        assert elapsed < 0.5  # very generous; typical is ~5-15 ms
        assert rec.rows == 10_000
