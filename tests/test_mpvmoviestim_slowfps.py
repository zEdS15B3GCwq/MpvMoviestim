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
        fullscr=True,
        useFBO=False,  # test both
        waitBlanking=WAIT_BLANK,
        units="norm",
        color=(-1, -1, -1),  # black background
    )
    sleep(5.0)
    for i in range(10):
        win.flip()

    return win


def main() -> None:
    # file = Path(r"tests\media\4k144.mkv")
    # file = Path(r"tests\media\5.mkv")
    file = Path(r"tests\media\long.mp4")
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
        flipHoriz=False,
        flipVert=True,
    )

    # print(f"{perf_counter() - t0:.3} | video-sync={player._player.video_sync}")
    sleep(0.5)

    print(f"{perf_counter() - t0:.3} | {player.state}")

    print(f"{perf_counter() - t0:.3} | >>>> PLAY 0")
    player.play(True)
    print(f"{perf_counter() - t0:.3} | >>>> PLAY 1")
    i = 0
    while player.state == movie.MpvMoviestimState.PLAYING:
        player.draw()
        win.flip()
        player.report_swap()
        i += 1
        # if i == 10:
        #     print(f"{perf_counter() - t0:.3} | video-sync={player._player.video_sync}")
        if i == 200:
            print(f"{perf_counter() - t0:.3} | >>>> PAUSE")
            player.pause()
            sleep(5.0)
            player.play()
            print(f"{perf_counter() - t0:.3} | >>>> PLAY")
        if i == 400:
            print(f"{perf_counter() - t0:.3} | >>>> BREAK")
            break

    player.stop()
    win.close()
    print(f"{perf_counter() - t0:.3} | >>>> FINISH")


if __name__ == "__main__":
    main()
