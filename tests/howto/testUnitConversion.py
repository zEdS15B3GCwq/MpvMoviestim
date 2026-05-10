from psychopy.monitors import Monitor
from psychopy.tools.monitorunittools import convertToPix
from psychopy.visual import Window


def test_unit_conversion():
    mon = Monitor(name="mymonitor", distance=60, width=40)
    mon.setSizePix([3840, 2160])
    win = Window(size=(800, 600), units="deg", monitor=mon)
    pix = convertToPix(pos=[0, 0], vertices=[[0, -10], [10, 10]], units="deg", win=win)
    print(pix)


if __name__ == "__main__":
    test_unit_conversion()
