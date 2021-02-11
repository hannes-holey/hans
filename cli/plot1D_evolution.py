#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt

from pylub.plottools import Plot


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="choice", default="all", choices=["all", "rho", "p", "jx", "jy"], help="variable (default: all)")
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")
    parser.add_argument('-f', dest="freq", type=int, default=1, help="plot frequency (default: 1)")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    files = Plot(args.path, mode="single")
    fig, ax, cb = files.plot_cut_evolution(choice=args.choice, dir=args.dir, freq=args.freq)
    plt.show()
