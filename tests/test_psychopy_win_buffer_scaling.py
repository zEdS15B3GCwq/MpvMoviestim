from pathlib import Path
from time import sleep

from psychopy import logging, visual

from mpvmoviestim.pixel_format import (
    get_psychopy_fbo_info,
    resolve_pixel_format_id_to_name,
)
from mpvmoviestim.utils import windows_get_screen_dpi, windows_set_process_dpi_awareness

WIN_SIZE = (2560, 1600)
WAIT_BLANK = True  # wait for blank after flip
# TODO: try setting wait for blank to False for report_swap()


def init_pp(set_dpi_aware: bool = True) -> visual.Window:
    print("setting pp.loggging level to INFO")
    logging.console.setLevel(logging.INFO)

    if set_dpi_aware:
        windows_set_process_dpi_awareness(True)

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
    win = init_pp(False)

    print("DPI unaware?")
    print(f"{win.monitorFramePeriod=}")
    print(f"{win.getActualFrameRate()=}")
    print(f"{win.getContentScaleFactor()=}")
    print(f"{win.useFBO=}, {win.frameBufferSize=}")
    print(f"{win.size}")
    print(f"screen dpi={windows_get_screen_dpi()}")
    fbo_info = get_psychopy_fbo_info(win)
    print(f"FBO info: {fbo_info}")
    print(
        f"Pixel format: {resolve_pixel_format_id_to_name(fbo_info.get('internal_format', 0))}"
    )

    win.close()

    win = init_pp(True)

    print("DPI aware")
    print(f"{win.monitorFramePeriod=}")
    print(f"{win.getActualFrameRate()=}")
    print(f"{win.getContentScaleFactor()=}")
    print(f"{win.useFBO=}, {win.frameBufferSize=}")
    print(f"{win.size}")
    print(f"screen dpi={windows_get_screen_dpi()}")
    fbo_info = get_psychopy_fbo_info(win)
    print(f"FBO info: {fbo_info}")
    print(
        f"Pixel format: {resolve_pixel_format_id_to_name(fbo_info.get('internal_format', 0))}"
    )
    win.close()


if __name__ == "__main__":
    main()
