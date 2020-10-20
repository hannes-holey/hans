import os
from argparse import ArgumentParser
from fhl2d.problem import Input


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--plot', dest='plot', default=False, help="on-the-fly plot option", action='store_true')
    parser.add_argument('--reduced-output', dest='reducedOut', default=False, help="don't write pressure field", action='store_true')
    parser.add_argument('--restart', dest="restart_file", default=None, help="restart simulation from last step of specified file")
    parser.add_argument('-o', dest="out_dir", default="data", help="output directory (default: ./data)")
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", dest="filename", help="path to input file", required=True)

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    inputFile = os.path.join(os.getcwd(), args.filename)
    if args.restart_file is not None:
        restartFile = os.path.join(os.getcwd(), args.restart_file)
    else:
        restartFile = None
    myProblem = Input(inputFile, restartFile).getProblem()
    myProblem.run(args.plot, args.reducedOut, args.out_dir)
