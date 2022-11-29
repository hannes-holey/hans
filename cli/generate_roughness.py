#
# Copyright 2022 Hannes Holey
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from argparse import ArgumentParser

from hans.geometry import fourier_synthesis


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('filename', metavar="FILENAME")
    parser.add_argument('-Nx',
                        dest="Nx",
                        default=128,
                        type=int,
                        help="Nx (default: 128)")
    parser.add_argument('-Ny',
                        dest="Ny",
                        default=128,
                        type=int,
                        help="Ny (default: 128)")
    parser.add_argument('-Lx',
                        dest="Lx",
                        default=1.,
                        type=float,
                        help="Lx (default: 1.)")
    parser.add_argument('-Ly',
                        dest="Ly",
                        default=1.,
                        type=float,
                        help="Ly (default: 1.)")
    parser.add_argument('-H', '--Hurst',
                        dest="hurst",
                        default=0.8,
                        type=float,
                        help="Hurst exponent (default: 0.8)")
    parser.add_argument('--rms-height',
                        dest="rms_height",
                        default=None,
                        type=float,
                        help="Root-mean squared height (default: None)")
    parser.add_argument('--rms-slope',
                        dest="rms_slope",
                        default=None,
                        type=float,
                        help="Root-mean squared slope (default: None)")
    parser.add_argument('--short-cutoff',
                        dest="short_cutoff",
                        default=None,
                        type=float,
                        help="Short wavelength cutoff (default: None)")
    parser.add_argument('--long-cutoff',
                        dest="long_cutoff",
                        default=None,
                        type=float,
                        help="Long wavelength cutoff (default: None)")
    parser.add_argument('--rolloff',
                        dest="rolloff",
                        default=1.0,
                        type=float,
                        help="Rolloff (default: 1.0)")
    parser.add_argument('--seed',
                        dest="seed",
                        default=None,
                        type=int,
                        help="Random seed (default: None)")
    parser.add_argument('--plot',
                        dest="plot",
                        default=False,
                        action="store_true",
                        help="Show surface plot (default: False)")

    return parser


def plot_surface(arr):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, tight_layout=True)

    x = np.arange(arr.shape[0]) / arr.shape[0]
    y = np.arange(arr.shape[1]) / arr.shape[1]

    xx, yy = np.meshgrid(x, y)

    ax.plot_surface(xx, yy, arr, cmap=cm.viridis, linewidth=0, antialiased=False)

    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    topo = fourier_synthesis((args.Nx, args.Ny),
                             (args.Lx, args.Ly),
                             args.hurst,
                             rms_height=args.rms_height,
                             rms_slope=args.rms_slope,
                             short_cutoff=args.short_cutoff,
                             long_cutoff=args.long_cutoff,
                             rolloff=args.rolloff)

    print(f"Created height profile ranging from {np.amin(topo):.5e} to {np.amax(topo):.5e}.")

    np.save(args.filename, topo)

    if args.plot:
        plot_surface(topo)


if __name__ == "__main__":
    main()
