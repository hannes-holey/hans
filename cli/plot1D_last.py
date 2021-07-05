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

from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "rho", "p", "jx", "jy"], help="variable (default: None)")
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")

    return parser


if __name__ == "__main__":

    ylabels = {"rho": r"Density $\rho$",
               "p": r"Pressure $p$",
               "jx": r"Momentum density $j_x$",
               "jy": r"Momentum denisty $j_y$"}

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path)

    if args.key is None:
        data = files.get_centerline(dir=args.dir)
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8), tight_layout=True)
        ax[1, 0].set_xlabel(rf"Distance ${args.dir}$")
        ax[1, 1].set_xlabel(rf"Distance ${args.dir}$")

        for fn, fdata in data.items():
            print("Plotting ", fn)
            for (key, (xdata, ydata)), axis in zip(fdata.items(), ax.flat):
                axis.plot(xdata, ydata)
                axis.set_ylabel(ylabels[key])

    else:
        data = files.get_centerline(key=args.key, dir=args.dir)
        fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)
        for fn, fdata in data.items():
            print("Plotting ", fn)
            xdata, ydata = fdata[args.key]
            ax.plot(xdata, ydata)

        ax.set_ylabel(ylabels[args.key])
        ax.set_xlabel(rf"Distance ${args.dir}$")

    plt.show()
