#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pylub.helper.plot import Plot

if __name__ == "__main__":

    # TODO: ArgumentParser
    files = Plot(".")
    fig, ax = files.plot_timeseries(files, "dt")
    plt.show()
