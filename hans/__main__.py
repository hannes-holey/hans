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


import os
from mpi4py import MPI
from argparse import ArgumentParser

from hans.input import Input


def get_parser():
    """Parses input arguments from command line.

    Returns
    -------
    ArgumentParser
        Parser object.

    """

    parser = ArgumentParser()

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input',
                          dest="filename",
                          help="path to input file",
                          required=True)

    # optional arguments
    parser.add_argument('-p', '--plot',
                        dest='plot',
                        default=False,
                        help="on-the-fly plot option",
                        action='store_true')
    parser.add_argument('-r', '--restart',
                        dest="restart_file",
                        default=None,
                        help="restart simulation from last step of specified file")
    parser.add_argument('-o', '--output',
                        dest="out_dir",
                        default="data",
                        help="output directory (default: ./data)")
    parser.add_argument('-n', '--name',
                        dest="out_name",
                        default=None,
                        help="output filename (without extension *.nc)")

    return parser


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        parser = get_parser()
        args = parser.parse_args()

        if args.plot and comm.Get_size() > 1:
            print("Live-plotting not implemented for parallel execution", flush=True)
            comm.Abort()

        inputFile = os.path.join(os.getcwd(), args.filename)
        if args.restart_file is not None:
            restartFile = os.path.join(os.getcwd(), args.restart_file)
        else:
            restartFile = None

        myProblem = Input(inputFile, restartFile).getProblem()

    else:
        myProblem = None
        args = None

    args = comm.bcast(args, root=0)
    myProblem = comm.bcast(myProblem, root=0)
    myProblem.run(out_dir=args.out_dir, out_name=args.out_name, plot=args.plot)
