"""Minimal standalone script to verify that a timed-out mpv wait call
raises a timeout exception (builtin TimeoutError or
concurrent.futures.TimeoutError).

Run with:
    python tests/test_mpv_wait_timeout.py
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError

import mpv

TIMEOUT_S = 1.0


def main() -> None:
    player = mpv.MPV()
    try:
        print("Waiting for a property that will never become true...")
        try:
            # "pause" starts as False and we never change it, and we use a
            # condition that will never be satisfied, so this call must
            # time out.
            player.wait_for_property(
                "pause", cond=lambda v: v is True, timeout=TIMEOUT_S
            )
        except (TimeoutError, FutureTimeoutError) as e:
            print(f"OK: got expected timeout exception: {type(e).__name__}: {e}")
        else:
            print("FAIL: no exception raised, expected a timeout.")
    finally:
        player.terminate()


if __name__ == "__main__":
    main()
