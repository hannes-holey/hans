#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
import yaml
# import argparse


class yamlInput:
    """Reads the yaml input file to define a problem.

    Attributes
    ----------
    inputFile : str
        filename of the .yaml input file
    """

    def __init__(self, inputFile):
        """ Constructor

        Parameters
        ----------
        inputFile : str
            filename of the .yaml input file
        """
        self.inputFile = inputFile

    def getProblem(self):
        """Parses through yaml file and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            self.options = inp['options']
            self.disc = inp['disc']
            self.geometry = inp['geometry']
            self.numerics = inp['numerics']
            self.material = inp['material']

        thisProblem = Problem(self.options, self.disc, self.geometry, self.numerics, self.material)

        return thisProblem


class Problem:
    """Problem class, contains all information about a specific simulation

    Attributes
    ----------
    options : dict
        Contains IO options.
    disc : dict
        Contains discretization parameters.
    geometry : dict
        Contains geometry parameters.
    numerics : dict
        Contains parameters for the numeric scheme.
    material : dict
        Contains material parameters.

    """

    def __init__(self, options, disc, geometry, numerics, material):
        """Constructor

        Parameters
        ----------
        options : dict
            Contains IO options.
        disc : dict
            Contains discretization parameters.
        geometry : dict
            Contains geometry parameters.
        numerics : dict
            Contains parameters for the numeric scheme.
        material : dict
            Contains material parameters.
        """

        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

    def run(self, plot=False, reducedOut=False):
        """Short summary.

        Parameters
        ----------
        plot : type
            Description of parameter `plot` (the default is False).
        reducedOut : type
            Description of parameter `reducedOut` (the default is False).

        Returns
        -------
        type
            Description of returned object.

        """
        """Starts the simulation.

        Parameters
        ----------
        plot : bool
            Flag for live plotting (the default is False).
        reducedOut : bool
            if True, no pressure outpur is written (the default is False).
        """
        from .run import Run
        Run(self.options, self.disc, self.geometry, self.numerics, self.material, plot, reducedOut)
