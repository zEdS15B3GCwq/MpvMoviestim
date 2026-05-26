import pytest
from psychopy.monitors import Monitor
from psychopy.visual import Window

from mpvmoviestim.mpvmoviestim import MpvMoviestim


@pytest.fixture(scope="module")
def _monitor() -> Monitor:
    mon = Monitor(name="test-monitor", distance=60, width=40)
    mon.setSizePix([1920, 1080])
    return mon


@pytest.fixture(scope="module")
def win_pix(_monitor: Monitor):
    win = Window(
        size=(1920, 1080),
        units="pix",
        monitor=_monitor,
        fullscr=False,
        allowGUI=False,
    )
    try:
        yield win
    finally:
        win.close()


@pytest.fixture(scope="module")
def win_norm(_monitor: Monitor):
    win = Window(
        size=(1920, 1080),
        units="norm",
        monitor=_monitor,
        fullscr=False,
        allowGUI=False,
    )
    try:
        yield win
    finally:
        win.close()


def test_bounding_rect_returns_none_without_size_or_media(win_pix) -> None:
    rect = MpvMoviestim.bounding_rect(
        size=None,
        media_size=None,
        position=(0.0, 0.0),
        window=win_pix,
    )
    assert rect is None


@pytest.mark.parametrize(
    ("size", "media_size", "position", "expected"),
    [
        ((1920.0, 1080.0), None, (0.0, 0.0), (0, 0, 1920, 1080)),
        ((100.0, 100.0), None, (0.0, 0.0), (910, 490, 100, 100)),
        ((200.0, 150.0), None, (100.0, -50.0), (960, 415, 200, 150)),
        (None, (640.0, 480.0), (0.0, 0.0), (640, 300, 640, 480)),
        ((200.0, 100.0), (640.0, 480.0), (0.0, 0.0), (860, 490, 200, 100)),
        (None, (400.0, 300.0), (200.0, 100.0), (960, 490, 400, 300)),
        ((400.0, 300.0), None, (-800.0, 0.0), (-40, 390, 400, 300)),
        ((400.0, 300.0), None, (800.0, 450.0), (1560, 840, 400, 300)),
    ],
)
def test_bounding_rect_pix(size, media_size, position, expected, win_pix) -> None:
    rect = MpvMoviestim.bounding_rect(
        size=size,
        media_size=media_size,
        position=position,
        window=win_pix,
    )
    assert rect == expected


@pytest.mark.parametrize(
    ("size", "media_size", "position", "expected"),
    [
        ((2.0, 2.0), None, (0.0, 0.0), (0, 0, 1920, 1080)),
        ((1.0, 1.0), None, (0.0, 0.0), (480, 270, 960, 540)),
        ((0.5, 0.5), None, (0.0, 0.0), (720, 405, 480, 270)),
        ((0.5, 0.5), None, (0.5, 0.5), (1200, 675, 480, 270)),
        ((1.0, 1.0), None, (-0.5, -0.5), (0, 0, 960, 540)),
        (None, (480.0, 270.0), (0.0, 0.0), (720, 405, 480, 270)),
        (None, (640.0, 480.0), (0.5, 0.5), (1120, 570, 640, 480)),
        ((1.5, 1.5), None, (0.5, 0.5), (720, 405, 1440, 810)),
        ((0.5, 0.5), None, (-1.25, -1.25), (-480, -270, 480, 270)),
    ],
)
def test_bounding_rect_norm(size, media_size, position, expected, win_norm) -> None:
    rect = MpvMoviestim.bounding_rect(
        size=size,
        media_size=media_size,
        position=position,
        window=win_norm,
    )
    assert rect == expected
