#!/usr/bin/env python3

"""
MIT License

Copyright 2022 Hannes Holey

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
import numpy as np

from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-s', '--save', dest="save", action="store_true", default=False,
                        help="save height distribution as binary file in NumPy format (default: False)")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path, mode="single")
    fn, = files.get_filenames()
    zdata, = files.get_height()

    fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)
    print("Plotting ", fn)

    im = ax.imshow(zdata.T, extent=(0, 1, 0, 1))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    fmt = tk.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))

    cbar = plt.colorbar(im, cax=cax, format=fmt, orientation="vertical")
    cbar.set_label(r"Gap height  $h$")

    ax.set_xlabel(r"$x/L_x$")
    ax.set_ylabel(r"$y/L_y$")

    if args.save:
        ofn = fn.rstrip('.nc') + "_height.npy"
        np.save(ofn, zdata.T)

    plt.show()
