#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py

from eos.eos import DowsonHigginson
from geo.geometry import Analytic
from field.field import VectorField
from flux.flux import Flux

class Solver:

    def __init__(self, options, disc, geometry, numerics, material):

        self.material = material
        self.disc = disc
        self.numerics = numerics

        self.name = str(options['name'])
        self.writeOutput = int(options['writeOutput'])
        self.writeInterval = int(options['writeInterval'])

        self.U = float(geometry['U'])
        self.V = float(geometry['V'])

        self.numFlux = str(numerics['numFlux'])
        self.dt = float(numerics['dt'])
        self.maxIt = int(numerics['maxIt'])

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

        self.file_tag = 1
        self.ani_tag = 1

        while str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5' in os.listdir('./output'):
            self.file_tag += 1

        while str(self.name) + '_' + str(self.ani_tag).zfill(4) + '.mp4' in os.listdir('./output/animations'):
            self.ani_tag += 1

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

        if self.writeOutput == True:
            self.write(i)

    def write(self, i):

        # HDF5 output file
        if i % self.writeInterval == 0:

            file = h5py.File('./output/' + str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5', 'a')

            if str(i).zfill(len(str(self.maxIt))) not in file:

                g1 =file.create_group(str(i).zfill(len(str(self.maxIt))))

                g1.create_dataset('j_x',   data = self.q.field[0])
                g1.create_dataset('j_y',   data = self.q.field[1])
                g1.create_dataset('rho',   data = self.q.field[2])
                g1.create_dataset('press', data = DowsonHigginson(self.material).isoT_pressure(self.q.field[2]))

                g1.attrs.create('time', self.dt*i)
                g1.attrs.create('mass', self.mass)

            file.close()
