#
# Copyright 2020, 2022 Hannes Holey
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
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")
    parser.add_argument('-s', '--save', dest="save", action="store_true", default=False,
                        help="save height distribution as binary file in NumPy format (default: False)")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path)
    data = files.get_centerline_height(dir=args.dir)
    fns = files.get_filenames()

    fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)

    for fn, (xdata, ydata) in zip(fns, data):
        print("Plotting ", fn)
        ax.plot(xdata, ydata)

    ax.set_ylim(0.,)
    ax.set_ylabel(r"Gap height  $h$")
    ax.set_xlabel(rf"Distance ${args.dir}$")

    if args.save:
        ofn = fn.rstrip('.nc') + f"_height_{args.dir}.npy"
        np.save(ofn, ydata[:, None])

    plt.show()
