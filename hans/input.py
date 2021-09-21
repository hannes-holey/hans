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


import yaml

from hans.problem import Problem


class Input:

    def __init__(self, inputFile, restartFile=None):
        """
        Reads the yaml input file that defines a problem.

        Parameters
        ----------
        inputFile : str
            filename of the .yaml input file
        restartFile : str
            filename of the .nc data file (default: None)

        """
        self.inputFile = inputFile
        self.restartFile = restartFile

    def getProblem(self):
        """
        Parses the yaml input file, performs sanity checks,
        and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Problem Object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            # Mandatory inputs
            options = inp['options']
            disc = inp['disc']
            geometry = inp['geometry']
            numerics = inp['numerics']
            material = inp['material']
            bc = inp['BC']

            # Optional inputs
            if "surface" in inp.keys():
                surface = inp["surface"]
            else:
                surface = None

            if "IC" in inp.keys():
                ic = inp["IC"]
            elif self.restartFile is not None:
                ic = {}
                ic["type"] = "restart"
                ic["file"] = self.restartFile
            else:
                ic = None

        thisProblem = Problem(options,
                              disc,
                              bc,
                              geometry,
                              numerics,
                              material,
                              surface,
                              ic)

        return thisProblem
