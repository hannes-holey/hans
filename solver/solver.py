#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py

from eos.eos import DowsonHigginson
from geo.geometry import Analytic

#from field.field import ScalarField
from field.field import VectorField

from flux.flux import Flux

class Solver:

    def __init__(self, options, disc, geometry, numerics, material):

        self.material = material
        self.disc = disc
        self.numerics = numerics

        self.name = str(options['name'])

        self.U = float(geometry['U'])
        self.V = float(geometry['V'])

        self.numFlux = str(numerics['numFlux'])
        self.dt = float(numerics['dt'])

        self.mu = float(material['mu'])
        self.P0 = float(material['P0'])
        self.rho0 = float(material['rho0'])

        # Stokes assumption
        self.material['lambda'] = -2./3. * self.mu

        # Gap height
        self.height = VectorField(disc)

        if self.name == 'journal':
            self.height.fromFunctionXY(Analytic(disc, geometry).journalBearing, 0)
        elif self.name == 'inclined' or self.name == 'poiseuille':
            self.height.fromFunctionXY(Analytic(disc, geometry).linearSlider, 0)

        self.height.getGradients()

        self.q = VectorField(disc)
        self.q.fill(self.rho0, 2)

        if self.name == 'inclined':
            self.q.field[2][0,:] = DowsonHigginson(material).isoT_density(self.P0)
            self.q.field[2][-1,:] = DowsonHigginson(material).isoT_density(self.P0)
        elif self.name == 'poiseuille':
            self.q.field[2][-1,:] = DowsonHigginson(material).isoT_density(self.P0)
            self.q.field[2][0,:] = DowsonHigginson(material).isoT_density(2. * self.P0)

        self.rhs = VectorField(disc)

    def solve(self, i):

        if self.numFlux == 'LF':
            fXE = Flux(self.disc, self.numerics, self.material).getFlux_LF(self.q, self.height, -1, 0)
            fXW = Flux(self.disc, self.numerics, self.material).getFlux_LF(self.q, self.height,  1, 0)
            fYN = Flux(self.disc, self.numerics, self.material).getFlux_LF(self.q, self.height, -1, 1)
            fYS = Flux(self.disc, self.numerics, self.material).getFlux_LF(self.q, self.height,  1, 1)

        elif self.numFlux == 'LW':
            fXE = Flux(self.disc, self.numerics, self.material).getFlux_LW(self.q, self.height, -1, 0)
            fXW = Flux(self.disc, self.numerics, self.material).getFlux_LW(self.q, self.height,  1, 0)
            fYN = Flux(self.disc, self.numerics, self.material).getFlux_LW(self.q, self.height, -1, 1)
            fYS = Flux(self.disc, self.numerics, self.material).getFlux_LW(self.q, self.height,  1, 1)

        self.rhs.computeRHS(fXE, fXW, fYN, fYS)

        self.rhs.addStress_wall(self.q, self.height, self.mu, self.U, self.V)

        # explicit time step
        self.q.updateExplicit(self.rhs, self.dt)

        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)

        # if self.writeOutput == True:
        #     mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
        #     vmax = np.amax(np.abs(self.vel.field[0]))
        #     maxFlux_X = np.amax(self.q.field[0])
        #     netFluxX = np.sum(self.q.field[0])
        #     with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
        #         f.write("%.8e \t %.8e \t %.8e \t %.8e \n" % (i*self.dt, mass, vmax, netFluxX))

        # return self.q
