import os
from argparse import ArgumentParser
from fhl2d import problem


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--plot', dest='plot', default=False, help="on-the-fly plot option", action='store_true')
    parser.add_argument('--reduced-output', dest='reducedOut', default=False, help="don't write pressure field", action='store_true')
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", dest="filename", help="path to input file", required=True)

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    inputFile = os.path.join(os.getcwd(), args.filename)
    myProblem = problem.yamlInput(inputFile).getProblem()
    myProblem.run(args.plot, args.reducedOut)
