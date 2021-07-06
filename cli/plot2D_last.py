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
import matplotlib.ticker as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "rho", "p", "jx", "jy"], help="variable (default: None)")

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
        data = files.get_field()

        for fn, fdata in data.items():
            fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8), tight_layout=True)
            print("Plotting ", fn)
            for (key, zdata), axis in zip(fdata.items(), ax.flat):
                im = axis.imshow(zdata.T, extent=(0, 1, 0, 1))

                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.1)

                fmt = tk.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))

                cbar = plt.colorbar(im, cax=cax, format=fmt, orientation="vertical")
                cbar.set_label(ylabels[key])

                axis.set_xlabel(r"$x/L_x$")
                axis.set_ylabel(r"$y/L_y$")

    else:
        data = files.get_field(key=args.key)
        for fn, fdata in data.items():
            fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)
            print("Plotting ", fn)
            zdata = fdata[args.key]
            im = ax.imshow(zdata.T, extent=(0, 1, 0, 1))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            fmt = tk.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))

            cbar = plt.colorbar(im, cax=cax, format=fmt, orientation="vertical")
            cbar.set_label(ylabels[args.key])

            ax.set_xlabel(r"$x/L_x$")
            ax.set_ylabel(r"$y/L_y$")

    plt.show()
