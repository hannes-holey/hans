#
# Copyright 2020, 2023 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "rho", "p", "jx", "jy"], help="variable (default: None)")
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")
    parser.add_argument('-n', dest="step", default=-1)

    return parser


def main():

    ylabels = {"rho": r"Density $\rho$",
               "jx": r"Momentum density $j_x$",
               "jy": r"Momentum denisty $j_y$",
               "p": r"Pressure $p$",
               "tau_bot": r"Shear stress (bottom) $\tau_{xz}^\text{bot}$",
               "tau_top": r"Shear stress (top) $\tau_{xz}^\text{top}$",
               }

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path, mode="single")
    fn, = files.get_filenames()
    _, xdata, ydata = files.get_centerlines_gp(gp_index=args.step, dir=args.dir)[0]
    print("Plotting ", fn)

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6.4, 7.2))

    ax[2, 0].set_xlabel(rf"Distance ${args.dir}$")
    ax[2, 1].set_xlabel(rf"Distance ${args.dir}$")

    keys = ["rho", "jx", "jy", "p", "tau_bot", "tau_top"]

    for key, axis in zip(keys, ax.T.flat):

        for i in range(len(ydata[key])):

            if key in ["p", "tau_bot", "tau_top"]:
                y, var_y = ydata[key][i]
                ci = 1.96 * np.sqrt(var_y[:, 0])

                p, = axis.plot(xdata, y)
                axis.fill_between(xdata, y + ci, y - ci, color=p.get_color(), alpha=0.3, lw=0.)

            else:
                y = ydata[key][i]
                axis.plot(xdata, y)

        axis.set_ylabel(ylabels[key])

    plt.show()


if __name__ == "__main__":
    main()
