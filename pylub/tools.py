
import numpy as np


def adaptiveLimits(ax):

    def offset(x, y): return 0.05 * (x - y) if (x - y) != 0 else 1.

    for a in ax.flat:

        y_min = np.amin(a.lines[0].get_ydata())
        y_max = np.amax(a.lines[0].get_ydata())

        a.set_ylim(y_min - offset(y_max, y_min), y_max + offset(y_max, y_min))

    return ax


def time_to_HHMMSS(t):

    MM = t // 60
    HH = int(MM // 60)
    MM = int(MM - HH * 60)
    SS = int(t - HH * 60 * 60 - MM * 60)

    return HH, MM, SS
