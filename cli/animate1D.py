#!/usr/bin/env python3

"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from hans.plottools import DatasetSelector, adaptiveLimits


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default="p", choices=["rho", "p", "jx", "jy"], help="variable (default: p)")
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")

    return parser


def update_line(i, A, t):
    """
    Updates the plot in animation

    Parameters
    ----------
    i : int
        iterator
    A : list
        contains 1D arrays of field variables (centerline)
    t : list
        contains time
    """

    line[0].set_ydata(A[i][1])

    adaptiveLimits(ax)

    fig.suptitle("Time: {:.1g} s".format(t[i]))


if __name__ == "__main__":

    ylabels = {"rho": r"Density $\rho$",
               "p": r"Pressure $p$",
               "jx": r"Momentum density $j_x$",
               "jy": r"Momentum denisty $j_y$"}

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path, mode="single")

    data = files.get_centerlines(key=args.key, dir=args.dir)

    for fn, fdata in data.items():
        print("Animating ", fn)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        t = list(fdata[args.key].keys())
        A = list(fdata[args.key].values())

        line = ax.plot(A[0][0], A[0][1])

        # Adjust ticks
        if args.dir == "x":
            ax.set_xlabel(r'$x/L_x$')
        elif args.dir == "y":
            ax.set_xlabel(r'$y/L_y$')

        ax.set_ylabel(ylabels[args.key])

    ani = animation.FuncAnimation(fig, update_line, frames=len(A), fargs=(A, t), interval=100, repeat=True)

    plt.show()
