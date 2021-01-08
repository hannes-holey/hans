#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pylub.helper.plot import Plot


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="attr", default="mass", choices=["mass", "vmax", "vSound", "dt", "eps"], help="variable (default: mass)")
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    files = Plot(args.path)
    fig, ax = files.plot_timeseries(args.attr)
    plt.show()
