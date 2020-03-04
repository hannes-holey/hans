#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import sys
import time

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
    try :
      name = sys.argv[1]
    except:
      print("Usage : ./main.py <config-name>")
      quit()

    inputFile = './config/' + str(name) + '.yaml'

    tStart = time.time()

    myInput = yamlInput(inputFile)
    myProblem = myInput.getProblem()
    myProblem.solve()

    tDiff = time.time() - tStart

    MM = tDiff//60
    HH = MM//60
    MM = HH - HH*60
    SS = tDiff - HH *60*60 - MM*60

    print("Total wall clock time: %02d:%02d:%02d" % (HH, MM, SS))

if __name__=="__main__":
    main()
