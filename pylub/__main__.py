import os
from mpi4py import MPI
from argparse import ArgumentParser

from pylub.input import Input


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-p', '--plot', dest='plot', default=False, help="on-the-fly plot option", action='store_true')
    parser.add_argument('-r', '--restart', dest="restart_file", default=None, help="restart simulation from last step of specified file")
    parser.add_argument('-o', '--output', dest="out_dir", default="data", help="output directory (default: ./data)")
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', dest="filename", help="path to input file", required=True)

    return parser


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        parser = get_parser()
        args = parser.parse_args()

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
    myProblem.run(args.plot, args.out_dir)
