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
    parser.add_argument('-f', dest="freq", type=int, default=1, help="plot frequency (default: 1)")

    return parser


if __name__ == "__main__":

    ylabels = {"rho": r"Density $\rho$",
               "p": r"Pressure $p$",
               "jx": r"Momentum density $j_x$",
               "jy": r"Momentum denisty $j_y$"}

    cmap = plt.cm.coolwarm

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path, mode="single")

    if args.key is None:
        data = files.get_centerlines(freq=args.freq, dir=args.dir)
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8))

        for fn, fdata in data.items():
            ax[1, 0].set_xlabel(rf"Distance ${args.dir}$")
            ax[1, 1].set_xlabel(rf"Distance ${args.dir}$")
            maxT = max(fdata["rho"])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

            for (key, tdict), axis in zip(fdata.items(), ax.flat):
                for time, (xdata, ydata) in tdict.items():
                    axis.plot(xdata, ydata, color=cmap(time/maxT))
                axis.set_ylabel(ylabels[key])

            fig.colorbar(sm, ax=ax.ravel().tolist(), label='time $t$', extend='max')

    else:
        data = files.get_centerlines(key=args.key, freq=args.freq, dir=args.dir)
        fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)

        for fn, fdata in data.items():
            maxT = max(fdata[args.key])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

            for time, (xdata, ydata) in fdata[args.key].items():
                ax.plot(xdata, ydata, color=cmap(time/maxT))
                ax.set_ylabel(ylabels[args.key])
                ax.set_xlabel(rf"Distance ${args.dir}$")

            fig.colorbar(sm, ax=ax, label='time $t$', extend='max')

    plt.show()
