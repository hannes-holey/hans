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
import matplotlib.animation as animation


from hans.plottools import DatasetSelector, adaptiveLimits


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "rho", "p", "jx", "jy"], help="variable (default: None)")
    parser.add_argument('-d', dest="dir", default="x", choices=["x", "y"], help="cutting direction (default: x)")

    return parser


def update_lines(i, A, t):
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

    if type(A) is dict:
        for ax, k in zip(fig.axes, A.keys()):
            line, = ax.get_lines()
            line.set_ydata(A[k][i])
            adaptiveLimits(ax)
    else:
        ax, = fig.axes
        line, = ax.get_lines()
        line.set_ydata(A[i])
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
    fn, = files.get_filenames()
    time, xdata, ydata = files.get_centerlines(key=args.key, dir=args.dir)[0]
    print("Animating ", fn)

    if args.key is None:
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.4, 4.8))
        for key, axis in zip(ydata.keys(), ax.flat):
            axis.plot(xdata, ydata[key][0])
            axis.set_ylabel(ylabels[key])

            # Adjust ticks
            if args.dir == "x":
                axis.set_xlabel(r'$x$')
            elif args.dir == "y":
                axis.set_xlabel(r'$y$')
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(xdata, ydata[0])
        ax.set_ylabel(ylabels[args.key])

        # Adjust ticks
        if args.dir == "x":
            ax.set_xlabel(r'$x$')
        elif args.dir == "y":
            ax.set_xlabel(r'$y$')

    ani = animation.FuncAnimation(fig, update_lines, frames=len(time), fargs=(ydata, time), interval=100, repeat=True)

    plt.show()
