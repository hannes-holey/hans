#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
import yaml
import netCDF4
import warnings
import numpy as np


class Input:
    """Reads the yaml input file to define a problem.

    Attributes
    ----------
    inputFile : str
        filename of the .yaml input file
    """

    def __init__(self, inputFile, restartFile):
        """ Constructor

        Parameters
        ----------
        inputFile : str
            filename of the .yaml input file
        """
        self.inputFile = inputFile
        self.restartFile = restartFile

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

        q_init = None
        if self.restartFile is not None:
            self.checkDisc()
            q_init = self.getInitialField()

        thisProblem = Problem(self.options, self.disc, self.geometry, self.numerics, self.material, q_init)

        return thisProblem

    def checkDisc(self):
        file = netCDF4.Dataset(self.restartFile)

        disc = {'dx': file.dx, 'dy': file.dy, 'Nx': file.Nx, 'Ny': file.Ny}

        geometry = {'type': file.type, 'U': file.U, 'V': file.V}
        if file.type == "journal":
            geometry.update([('CR', file.CR), ('eps', file.eps)])
        elif file.type == "inclined":
            geometry.update([('h1', file.h1), ('h2', file.h2)])

        material = {'EOS': file.EOS, 'shear': file.shear, 'bulk': file.bulk, 'P0': file.P0, 'rho0': file.rho0, 'T0': file.T0}
        if file.EOS == "DH":
            material.update([('C1', file.C1), ('C2', file.C2)])
        elif file.EOS == "Tait":
            material.update([('K', file.K), ('n', file.n)])
        elif file.EOS == "PL":
            material.update([('alpha', file.alpha)])

        assert self.disc == disc

        if self.geometry != geometry:
            warnings.warn("Restart simulation with different geometry", UserWarning)

        if self.material != material:
            warnings.warn("Restart simulation with different material", UserWarning)

    def getInitialField(self):
        file = netCDF4.Dataset(self.restartFile)

        q0 = np.empty([3, self.disc['Nx'], self.disc['Ny']])
        q0[0] = np.array(file.variables['jx'])[-1]
        q0[1] = np.array(file.variables['jy'])[-1]
        q0[2] = np.array(file.variables['rho'])[-1]

        return q0


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

    def __init__(self, options, disc, geometry, numerics, material, q_init):
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
        self.q_init = q_init

    def run(self, plot=False, reducedOut=False):
        """Starts the simulation.

        Parameters
        ----------
        plot : bool
            Flag for live plotting (the default is False).
        reducedOut : bool
            if True, no pressure outpur is written (the default is False).
        """
        from .run import Run
        Run(self.options, self.disc, self.geometry, self.numerics, self.material, plot, reducedOut, self.q_init)
