#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pylub.helper.data_parser import getData
from pylub.helper.plot import plot_timeseries

if __name__ == "__main__":

    # TODO: ArgumentParser
    files = getData(".")
    fig, ax = plot_timeseries(files, "dt")
    plt.show()
