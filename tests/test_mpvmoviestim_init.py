import ctypes
from pathlib import Path
from time import perf_counter, sleep

from psychopy import event, logging, visual

import mpvmoviestim.mpvmoviestim as movie

WIN_SIZE = (2560, 1600)
WAIT_BLANK = True  # wait for blank after flip
# TODO: try setting wait for blank to False for report_swap()


def init_pp() -> visual.Window:
    print("setting pp.loggging to level EXP")
    logging.console.setLevel(logging.EXP)

    print("setting windows dpi awareness")
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_int64(-4))

    print("creating pp window")
    win = visual.Window(
        size=list(WIN_SIZE),
        fullscr=True,
        useFBO=False,  # test both
        waitBlanking=WAIT_BLANK,
        units="pix",
        color=(-1, -1, -1),  # black background
    )

    return win


def main() -> None:
    # file = Path(r"tests\media\4k144.mkv")
    file = Path(r"tests\media\5.mkv")
    assert file.exists()

    win = init_pp()

    player = movie.MpvMoviestim(win, file)

    sleep(2.0)

    print(player.state)
    win.close()


if __name__ == "__main__":
    main()
