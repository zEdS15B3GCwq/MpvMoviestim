from psychopy.monitors import Monitor
from psychopy.tools.monitorunittools import convertToPix
from psychopy.visual import Window


def test_unit_conversion():
    mon = Monitor(name="mymonitor", distance=60, width=40)
    mon.setSizePix([3840, 2160])
    win = Window(size=(3840, 2160), units="deg", monitor=mon)
    pix = convertToPix(pos=[0, 0], vertices=[[-1, -1], [1, 1]], units="norm", win=win)
    print(pix)
    pix = convertToPix(pos=[1, 0], vertices=[[-2, -2], [2, 2]], units="norm", win=win)
    print(pix)
    pix = convertToPix(pos=[0, 0], vertices=[[-2, -2], [2, 2]], units="norm", win=win)
    print(pix)


if __name__ == "__main__":
    test_unit_conversion()
