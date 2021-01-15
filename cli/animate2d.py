#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt

from pylub.plottools import Plot


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="choice", default="rho", choices=["rho", "p", "jx", "jy"], help="variable (default: rho)")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    files = Plot(args.path, mode="single")
    fig, ax, ani = files.animate2D(args.choice)
    plt.show()
    # ani.save(f"animation.mp4", fps=30)
