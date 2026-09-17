import ctypes
from pathlib import Path
from time import perf_counter, sleep

from psychopy import logging, visual

import mpvmoviestim.mpvmoviestim as movie

WIN_SIZE = (2560, 1600)
WAIT_BLANK = True  # wait for blank after flip
# TODO: try setting wait for blank to False for report_swap()


def init_pp() -> visual.Window:
    print("setting pp.loggging level to INFO")
    logging.console.setLevel(logging.INFO)

    print("setting windows dpi awareness")
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_int64(-4))

    print("creating pp window")
    win = visual.Window(
        size=list(WIN_SIZE),
        fullscr=False,
        useFBO=False,  # test both
        waitBlanking=WAIT_BLANK,
        units="norm",
        color=(-1, -1, -1),  # black background
    )
    sleep(5.0)
    for _ in range(10):
        win.flip()

    return win


def main() -> None:
    # file = Path(r"tests\media\4k144.mkv")
    # file = Path(r"tests\media\5.mkv")
    # file = Path(r"tests\media\long.mp4")
    file = Path(r"tests\media\out144.mkv")
    assert file.exists()
    t0 = perf_counter()

    win = init_pp()
    print(
        f"{perf_counter() - t0:.3} | {win.monitorFramePeriod=}, frame rate = {1 / win.monitorFramePeriod} Hz"
    )

    player = movie.MpvMoviestim(
        win,
        file,
        pos=(0, 0),
        size=(2, 2),
        monitor_framerate=1 / win.monitorFramePeriod,
        nvtx=True,
    )

    # print(f"{perf_counter() - t0:.3} | video-sync={player._player.video_sync}")
    sleep(0.5)

    print(f"{perf_counter() - t0:.3} | {player.state}")

    print(f"{perf_counter() - t0:.3} | >>>> PLAY 0")
    player.play(True)
    print(f"{perf_counter() - t0:.3} | >>>> PLAY 1")
    while player.state == movie.MpvMoviestimState.PLAYING:
        player.draw()
        win.flip()
        player.report_swap()

    player.stop()
    win.close()
    print(f"{perf_counter() - t0:.3} | >>>> FINISH")


if __name__ == "__main__":
    main()
