#
# Copyright 2021-2022 Hannes Holey
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "rho", "p", "jx", "jy"], help="variable (default: p)")

    return parser


def update_grids(i, A, t):
    """
    Updates the plot in animation

    Parameters
    ----------
    i : int
        iterator
    A : list
        contains 2D arrays of field variables
    t : list
        contains time
    """

    if type(A) is dict:
        for ax, k in zip(fig.axes, A.keys()):
            im, = ax.get_images()
            im.set_array(A[k][i].T)
            if i > 0:
                im.set_clim(vmin=np.amin(A[k][:i]), vmax=np.amax(A[k][:i]))
    else:
        ax = fig.axes[0]
        im, = ax.get_images()
        im.set_array(A[i].T)

        if i > 0:
            im.set_clim(vmin=np.amin(A[:i]), vmax=np.amax(A[:i]))

    fig.suptitle("Time: {:.1g} s".format(t[i]))


def main():

    ylabels = {"rho": r"Density $\rho$",
               "p": r"Pressure $p$",
               "jx": r"Momentum density $j_x$",
               "jy": r"Momentum denisty $j_y$"}

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path, mode="single")
    fn, = files.get_filenames()
    time, zdata = files.get_fields(key=args.key)[0]
    print("Animating ", fn)

    global fig, ax

    if args.key is None:
        fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6))
        for key, axis in zip(zdata.keys(), ax.flat):
            im = axis.imshow(zdata[key][0].T, extent=(0, 1, 0, 1))

            # colorbar
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.3)
            fmt = tk.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            cbar = plt.colorbar(im, cax=cax, format=fmt, orientation="vertical")
            cbar.set_label(ylabels[key])

            # set x-/y-labels
            axis.set_xlabel(r'$x/L_x$')
            axis.set_ylabel(r'$y/L_y$')
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        # initial plot
        im = ax.imshow(zdata[0].T, extent=(0, 1, 0, 1))

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fmt = tk.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = plt.colorbar(im, cax=cax, format=fmt, orientation="vertical")
        cbar.set_label(ylabels[args.key])

        # set x-/y-labels
        ax.set_xlabel(r'$x/L_x$')
        ax.set_ylabel(r'$y/L_y$')

    ani = animation.FuncAnimation(fig, update_grids, frames=len(time), fargs=(zdata, time), interval=100, repeat=True)

    plt.show()


if __name__ == "__main__":
    main()
