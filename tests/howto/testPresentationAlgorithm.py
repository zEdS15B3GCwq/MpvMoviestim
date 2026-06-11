"""
Simulation of media player frame presentation strategies.

This script simulates two different algorithms for deciding when to present video frames:
1. Naive algorithm: Always pick the frame with largest PTS that is ready (PTS <= current_time)
2. Optimized algorithm (libmpv-style): Uses display sync error accumulation to minimize jitter

The simulation tracks:
- Which frame is presented at each vsync
- The presentation time error (actual_presentation_time - target_PTS)
- Number of frames currently buffered
- Whether any frames are dropped without being presented

The optimized algorithm mimics libmpv's handle_display_sync_frame():
  ratio = (frame_duration + display_sync_error) / vsync_interval
  num_vsyncs = round(ratio)  // round-to-nearest for better jitter distribution
  display_sync_error += frame_duration - num_vsyncs * vsync_interval
"""

from dataclasses import dataclass


@dataclass
class FrameInfo:
    """Information about a decoded video frame."""

    index: int
    pts: float  # Presentation time in ms


def simulate_media_player(
    num_vsyncs: int = 200,
    screen_fps: float = 60.0,
    media_fps: float = 24.0,
    target_time_offset: float = 0.025,  # 25ms before PTS = 1.5 / screen_fps
) -> None:
    """
    Simulate media player frame presentation with two different algorithms.

    Args:
        num_vsyncs: Number of vsync cycles to simulate
        media_fps: Frames per second of the media
        screen_fps: Refresh rate of the screen in Hz
        supply_offset: Time before PTS that frames are supplied (in seconds)
    """

    # Calculate timing parameters
    num_frames: int = int(num_vsyncs / screen_fps * media_fps) + 1
    frame_duration: float = 1000.0 / media_fps  # Duration of each frame in ms
    vsync_interval: float = 1000.0 / screen_fps  # Time between vsyncs in ms

    # Generate simulated frames with their delivery times
    frames: list[FrameInfo] = []
    for i in range(num_frames):
        pts: float = (i * 1000) / media_fps  # When the frame should be presented
        frames.append(FrameInfo(index=i, pts=pts))

    # State for naive algorithm
    naive_buffer: list[int] = []  # Frames available to display
    naive_displayed: list[int] = [0] * num_frames  # Counts of how many times a frame was presented

    # State for optimized algorithm
    opt_buffer: list[int] = []  # Frames available to display
    opt_displayed: list[int] = [0] * num_frames  # Counts of how many times a frame was presented
    opt_display_sync_error = 0.0  # Accumulated jitter error
    opt_current_frame_index = None  # Index of frame currently being held
    opt_vsyncs_remaining = 0  # How many more vsyncs to display current frame

    # Track which frames have been delivered
    next_frame_to_deliver = 0

    # Print header
    print("=" * 130)
    print("MEDIA PLAYER FRAME PRESENTATION SIMULATION")
    print("=" * 130)
    print(
        f"Media FPS: {media_fps:5.1f} | Screen FPS: {screen_fps:5.1f} | "
        f"Frame Duration: {frame_duration * 1000:6.2f}ms | Vsync Interval: {vsync_interval * 1000:6.2f}ms"
    )
    print(
        f"Supply Offset: {target_time_offset * 1000:5.2f}ms | Total Frames: {num_frames:3d} | Total Vsyncs: {num_vsyncs:3d}"
    )
    print("=" * 130)
    print()

    # Print column headers
    print(
        f"{'T (ms)':>9} | "
        f"{'NAIVE ALGORITHM':^45} | "
        f"{'OPTIMIZED ALGORITHM (libmpv-style)':^45}"
    )
    print(f"{'-' * 9} | {'-' * 45} | {'-' * 45}")
    print(
        f"{'':>9} | "
        f"{'Frame':>8} {'Err(ms)':>8} {'Buf':>4} {'Drop':>7} {'Skip':>7} | "
        f"{'Frame':>8} {'Err(ms)':>8} {'Buf':>4} {'Drop':>7} {'Skip':>7}"
    )
    print("-" * 130)

    # Simulation loop
    for vsync_num in range(num_vsyncs):
        current_time = (1000 * vsync_num) / screen_fps

        # ========== FRAME DELIVERY ==========
        # Deliver new frames that are ready (their delivery time has come)
        current_time_with_offset = current_time - target_time_offset
        while (
            next_frame_to_deliver < len(frames)
            and frames[next_frame_to_deliver].pts <= current_time_with_offset
        ):
            naive_buffer.append(next_frame_to_deliver)
            opt_buffer.append(next_frame_to_deliver)
            next_frame_to_deliver += 1

        # assert that buffers are sorted in increasing pts order
        for i in range(len(naive_buffer) - 1):
            assert frames[naive_buffer[i]].pts < frames[naive_buffer[i + 1]].pts
        for i in range(len(opt_buffer) - 1):
            assert frames[opt_buffer[i]].pts < frames[opt_buffer[i + 1]].pts

        # ========== NAIVE ALGORITHM ==========
        naive_frame_index = None
        naive_presentation_error = 0.0
        naive_frames_dropped_this_vsync: list[int] = []

        # Frames ready to display or older (PTS <= current_time)
        ready_for_naive = [i for i in naive_buffer if frames[i].pts <= current_time]

        assert len(ready_for_naive) >= 1, "naive has no frames to display"

        # Pick the one with smallest PTS (earliest)
        naive_frame_index = ready_for_naive[-1]
        naive_displayed[naive_frame_index] += 1
        naive_presentation_error = current_time - frames[naive_frame_index].pts

        # Check for skipped frames
        # (anything older than current frame index that has not been displayed)
        naive_frames_dropped_this_vsync = [i for i in ready_for_naive[:-1] if naive_displayed[i] == 0]

        # Remove old displayed and skipped frames from buffer
        naive_buffer = naive_buffer[naive_frame_index:]

        # ========== OPTIMIZED ALGORITHM ==========
        opt_frame_index = None
        opt_presentation_error = 0.0
        opt_frame_dropped_this_vsync = False

        # If current frame has finished its display duration, select a new one
        if opt_vsyncs_remaining <= 0:
            # Find the frame with smallest PTS that is ready to display (PTS <= current_time)
            ready_for_opt = [i for i in opt_buffer if frames[i].pts <= current_time]

            if ready_for_opt:
                # Pick the one with smallest PTS (earliest)
                opt_frame_index = min(ready_for_opt, key=lambda i: frames[i].pts)
                opt_current_frame_index = opt_frame_index

                # Calculate how many vsyncs to display this frame using libmpv's algorithm
                frame_dur = frames[opt_frame_index].duration
                ratio = (frame_dur + opt_display_sync_error) / vsync_interval
                opt_vsyncs_remaining = max(round(ratio), 1)  # At least 1 vsync

                # Update error accumulation for next iteration
                opt_display_sync_error += (
                    frame_dur - opt_vsyncs_remaining * vsync_interval
                )

                frames[opt_frame_index].presented = True
                opt_displayed.add(opt_frame_index)
                opt_presentation_error = current_time - frames[opt_frame_index].pts
        else:
            # Continue displaying the current frame
            opt_frame_index = opt_current_frame_index
            opt_vsyncs_remaining -= 1

        # Check for frames that have become too old and will never be displayed
        stale_threshold = current_time - frame_duration * 2
        for frame_idx in opt_buffer:
            if (
                frames[frame_idx].pts < stale_threshold
                and frame_idx not in opt_displayed
            ):
                opt_skipped.add(frame_idx)
                opt_frame_dropped_this_vsync = True

        # Remove displayed frames and skipped frames from buffer
        opt_buffer = [
            i for i in opt_buffer if i not in opt_displayed and i not in opt_skipped
        ]

        # ========== OUTPUT ==========
        naive_frame_str = (
            f"#{naive_frame_index}" if naive_frame_index is not None else "---"
        )
        naive_error_str = (
            f"{naive_presentation_error * 1000:6.2f}"
            if naive_frame_index is not None
            else "---"
        )
        naive_dropped_str = "X" if naive_frame_dropped_this_vsync else ""

        opt_frame_str = f"#{opt_frame_index}" if opt_frame_index is not None else "---"
        opt_error_str = (
            f"{opt_presentation_error * 1000:6.2f}"
            if opt_frame_index is not None
            else "---"
        )
        opt_dropped_str = "X" if opt_frame_dropped_this_vsync else ""

        print(
            f"{current_time_ms:>9.2f} | "
            f"{naive_frame_str:>8} {naive_error_str:>8} {len(naive_buffer):>4d} "
            f"{naive_dropped_str:>7} {len(naive_skipped):>7} | "
            f"{opt_frame_str:>8} {opt_error_str:>8} {len(opt_buffer):>4d} "
            f"{opt_dropped_str:>7} {len(opt_skipped):>7}"
        )

    # ========== FINAL SUMMARY ==========
    print("-" * 130)
    print()
    print("SUMMARY")
    print("-" * 130)

    total_frames_presented_naive = len(naive_displayed)
    total_frames_presented_opt = len(opt_displayed)
    total_frames_skipped_naive = len(naive_skipped)
    total_frames_skipped_opt = len(opt_skipped)

    print(
        f"Naive Algorithm:    {total_frames_presented_naive:3d} frames displayed | "
        f"{total_frames_skipped_naive:3d} frames dropped"
    )
    print(
        f"Optimized Algorithm: {total_frames_presented_opt:3d} frames displayed | "
        f"{total_frames_skipped_opt:3d} frames dropped"
    )
    print()

    if naive_skipped:
        print(f"Naive dropped frames: {sorted(naive_skipped)}")
    if opt_skipped:
        print(f"Optimized dropped frames: {sorted(opt_skipped)}")
    print()
    print("=" * 130)


if __name__ == "__main__":
    # Run simulation with default parameters
    # These represent typical scenarios:
    # - 24 fps video on 60 fps display (ratio = 2.5)
    # - Frames delivered 1ms before their PTS
    # - 100 frames × ~200 vsyncs gives reasonable simulation duration

    # print("\nScenario 1: 24 fps video on 60 Hz display (1:2.5 ratio)")
    # print("=" * 130)
    # simulate_media_player(
    #     num_frames=100,
    #     num_vsyncs=200,
    #     media_fps=24.0,
    #     screen_fps=60.0,
    #     supply_offset=0.001,  # 1ms before PTS
    # )

    # print("\n\n")
    # print("Scenario 2: 30 fps video on 60 Hz display (1:2 ratio)")
    # print("=" * 130)
    # simulate_media_player(
    #     num_frames=100,
    #     num_vsyncs=200,
    #     media_fps=30.0,
    #     screen_fps=60.0,
    #     supply_offset=0.001,  # 1ms before PTS
    # )

    # print("\n\n")
    # print("Scenario 3: 23.976 fps video on 60 Hz display (NTSC timing)")
    # print("=" * 130)
    # simulate_media_player(
    #     num_frames=100,
    #     num_vsyncs=300,
    #     media_fps=23.976,
    #     screen_fps=60.0,
    #     supply_offset=0.002,  # 2ms before PTS
    # )
