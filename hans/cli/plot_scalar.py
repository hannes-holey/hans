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

from hans.plottools import DatasetSelector


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', dest="path", default="data", help="path (default: data)")
    parser.add_argument('-v', dest="key", default=None, choices=[None, "mass",
                                                                 "vmax", "vSound", "dt", "eps", "ekin"], help="variable (default: None)")
    parser.add_argument('-f', dest="freq", type=int, default=1, help="plot frequency (default: 1)")
    return parser


def main():

    ylabels = {"mass": r"Mass $m$",
               "vmax": r"Max. velocity $v_\mathrm{max}$",
               "vSound": r"Velocity of sound $c$",
               "dt": r"Time step $\Delta t$",
               "eps": r"$\Vert\rho_{n+1} -\rho_n \Vert /(\Vert\rho_n\Vert\,CFL)$",
               "ekin": r"Kinetic energy $E_\mathrm{kin}$"}

    parser = get_parser()
    args = parser.parse_args()

    files = DatasetSelector(args.path)
    fns = files.get_filenames()
    data = files.get_scalar(key=args.key, freq=args.freq)

    if args.key is None:
        fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6.4, 7.2), tight_layout=True)
        ax[2, 0].set_xlabel(r"Time $t$")
        ax[2, 1].set_xlabel(r"Time $t$")

        for fn, (time, ydata) in zip(fns, data):
            print("Plotting ", fn)
            for key, axis in zip(ydata.keys(), ax.flat):
                axis.plot(time, ydata[key])
                axis.set_ylabel(ylabels[key])

                if key == "eps":
                    axis.set_yscale("log")

    else:
        fig, ax = plt.subplots(1, figsize=(6.4, 4.8), tight_layout=True)
        ax.set_ylabel(ylabels[args.key])
        ax.set_xlabel(r"Time $t$")

        for fn, (time, ydata) in zip(fns, data):
            print("Plotting ", fn)
            ax.plot(time, ydata)

        if args.key == "eps":
            ax.set_yscale("log")

    plt.show()


if __name__ == "__main__":
    main()
