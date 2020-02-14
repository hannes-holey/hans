#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import sys

class yamlInput:

    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.read()

    def read(self):
        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            self.options = inp['options']
            self.geometry = inp['geometry']
            self.numerics = inp['numerics']
            self.material = inp['material']
            self.BC = inp['BC']
            self.EOS = inp['EOS']

    def getProblem(self):

        thisProblem = Problem(self.options, self.geometry, self.numerics,\
                              self.material, self.BC, self.EOS)

        return thisProblem


class Problem:

    def __init__(self, options, geometry, numerics, material, BC, EOS):

        self.options = options
        self.geometry = geometry
        self.numerics = numerics
        self.material = material
        self.BC = BC
        self.EOS = EOS

    def solve(self):

        solverClass = self.options['solver']

        if solverClass == 'LaxFriedrichs':
            from solver.solver import LaxFriedrichs
            LaxFriedrichs(self.options, self.geometry, self.numerics, self.material, self.BC, self.EOS)
        elif solverClass == 'EulerCentral':
            from solver.solver import EulerCentral
            EulerCentral(self.options, self.geometry, self.numerics, self.material, self.BC, self.EOS)

def main():
    try :
      name = sys.argv[1]
    except:
      print("Usage : ./main.py <config-name>")
      quit()

    inputFile = './config/' + str(name) + '.yaml'
    myInput = yamlInput(inputFile)
    myProblem = myInput.getProblem()
    myProblem.solve()

if __name__=="__main__":
    main()
