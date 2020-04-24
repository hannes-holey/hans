#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import argparse


class yamlInput:

    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.read()

    def read(self):
        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            self.options = inp['options']
            self.disc = inp['disc']
            self.geometry = inp['geometry']
            self.numerics = inp['numerics']
            self.material = inp['material']

    def getProblem(self):

        thisProblem = Problem(self.options, self.disc, self.geometry, self.numerics, self.material)

        return thisProblem


class Problem:

    def __init__(self, options, disc, geometry, numerics, material):

        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

    def solve(self):
        from run.run import Run
        Run(self.options, self.disc, self.geometry, self.numerics, self.material)


def main():

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", dest="filename", help="path to input file", required=True)
    args = parser.parse_args()

    inputFile = os.path.join(os.getcwd(), args.filename)
    myInput = yamlInput(inputFile)
    myProblem = myInput.getProblem()
    myProblem.solve()


if __name__ == "__main__":
    main()
