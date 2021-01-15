
import numpy as np


def adaptiveLimits(ax):

    a0y_min = np.amin(ax[0, 0].lines[0].get_ydata())
    a0y_max = np.amax(ax[0, 0].lines[0].get_ydata())
    a1y_min = np.amin(ax[0, 1].lines[0].get_ydata())
    a1y_max = np.amax(ax[0, 1].lines[0].get_ydata())
    a2y_min = np.amin(ax[1, 0].lines[0].get_ydata())
    a2y_max = np.amax(ax[1, 0].lines[0].get_ydata())
    a3y_min = np.amin(ax[1, 1].lines[0].get_ydata())
    a3y_max = np.amax(ax[1, 1].lines[0].get_ydata())

    def offset(x, y): return 0.05 * (x - y) if (x - y) != 0 else 1.

    ax[0, 0].set_ylim(a0y_min - offset(a0y_max, a0y_min), a0y_max + offset(a0y_max, a0y_min))
    ax[0, 1].set_ylim(a1y_min - offset(a1y_max, a1y_min), a1y_max + offset(a1y_max, a1y_min))
    ax[1, 0].set_ylim(a2y_min - offset(a2y_max, a2y_min), a2y_max + offset(a2y_max, a2y_min))
    ax[1, 1].set_ylim(a3y_min - offset(a3y_max, a3y_min), a3y_max + offset(a3y_max, a3y_min))

    return ax


def time_to_HHMMSS(t):

    MM = t // 60
    HH = int(MM // 60)
    MM = int(MM - HH * 60)
    SS = int(t - HH * 60 * 60 - MM * 60)

    return HH, MM, SS
