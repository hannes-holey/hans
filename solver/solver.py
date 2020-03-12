#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py

from eos.eos import DowsonHigginson, PowerLaw
from geo.geometry import Analytic
from field.field import VectorField
from flux.flux import Flux

class Solver:

    def __init__(self, options, disc, geometry, numerics, material):

        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

        self.name = str(options['name'])
        self.writeOutput = int(options['writeOutput'])
        self.writeInterval = int(options['writeInterval'])

        self.U = float(geometry['U'])
        self.V = float(geometry['V'])

        self.numFlux = str(numerics['numFlux'])
        self.dt = float(numerics['dt'])

        self.maxIt= int(self.numerics['maxT'] * 1e9 /self.numerics['dt'])

        self.mu = float(material['mu'])
        self.P0 = float(material['P0'])
        self.rho0 = float(material['rho0'])

        if self.material['EOS'] == 'DH':
            self.eqOfState = DowsonHigginson(self.material)
        elif self.material['EOS'] == 'PL':
            self.eqOfState = PowerLaw(self.material)

        self.time = 0

        # Stokes assumption
        self.material['lambda'] = -2./3. * self.mu

        # Gap height
        self.height = VectorField(disc)

        if self.name == 'journal':
            self.height.fromFunctionXY(Analytic(disc, geometry).journalBearing, 0)
        else:
            self.height.fromFunctionXY(Analytic(disc, geometry).linearSlider, 0)

        self.height.getGradients()



        self.q = VectorField(disc)
        self.q.fill(self.rho0, 2)

        if self.name == 'inclined':
            self.q.field[2][0,:] = self.eqOfState.isoT_density(self.P0)
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(self.P0)
        elif self.name == 'poiseuille':
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(self.P0)
            self.q.field[2][0,:] = self.eqOfState.isoT_density(2. * self.P0)
        elif self.name == 'droplet':
            self.q.fill_circle(1.e-4, self.eqOfState.isoT_density(1.5*self.P0), 2)
        elif self.name == 'wavefront':
            self.q.fill_line(0.25, 5e-5, self.eqOfState.isoT_density(2.*self.P0), 0, 2)
            # self.q.field[2][int(self.q.Nx/2)-2 : int(self.q.Nx/2)+2,int(self.q.Ny/2)-2 : int(self.q.Ny/2)+2] = self.eqOfState.isoT_density(2* self.P0)

        self.rhs = VectorField(disc)

        self.file_tag = 1
        self.ani_tag = 1

        while str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5' in os.listdir('./output'):
            self.file_tag += 1

        while str(self.name) + '_' + str(self.ani_tag).zfill(4) + '.mp4' in os.listdir('./output/animations'):
            self.ani_tag += 1

    def solve(self, i):

        vXmax = np.amax(self.q.field[0]/self.q.field[2])
        vYmax = np.amax(self.q.field[1]/self.q.field[2])

        self.vSound = np.amax(DowsonHigginson(self.material).soundSpeed(self.q.field[2]))


        if bool(self.numerics['adaptive']) == True:
            if i == 0:
                self.dt = self.dt
            else:
                self.dt = 0.02 * min(self.q.dx, self.q.dy)/self.vSound

        if self.numFlux == 'LF':
            fXE = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LF(self.q, self.height, -1, 0)
            fXW = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LF(self.q, self.height,  1, 0)
            fYN = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LF(self.q, self.height, -1, 1)
            fYS = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LF(self.q, self.height,  1, 1)

        elif self.numFlux == 'LW':
            fXE = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LW(self.q, self.height, -1, 0)
            fXW = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LW(self.q, self.height,  1, 0)
            fYN = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LW(self.q, self.height, -1, 1)
            fYS = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_LW(self.q, self.height,  1, 1)

        elif self.numFlux == 'MC':
            fXE = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_MC(self.q, self.height, -1, 0)
            fXW = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_MC(self.q, self.height,  1, 0)
            fYN = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_MC(self.q, self.height, -1, 1)
            fYS = Flux(self.disc, self.geometry, self.numerics, self.material).getFlux_MC(self.q, self.height,  1, 1)

        self.rhs.computeRHS(fXE, fXW, fYN, fYS)

        self.rhs.addStress_wall(self.q, self.height, self.mu, self.U, self.V)

        # explicit time step
        self.q.updateExplicit(self.rhs, self.dt)

        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
        self.vmax = np.amax(1./self.q.field[2]*np.sqrt(self.q.field[0]**2 + self.q.field[1]**2))

        self.cfl = self.vSound * self.dt/min(self.q.dx, self.q.dy)
        # self.C = vXmax*self.dt/self.q.dx + vYmax*self.dt/self.q.dy

        # self.C = 0.5 * self.q.dx*self.q.dy/(vXmax*self.q.dy + vYmax*self.q.dx)

        self.time += self.dt

        if self.writeOutput == True:
            self.write(i)

    def write(self, i):

        # HDF5 output file
        if i % self.writeInterval == 0:

            file = h5py.File('./output/' + str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5', 'a')

            if 'config' not in file:
                g0 = file.create_group('config')

                categories = {'options': self.options,
                              'disc': self.disc,
                              'geometry': self.geometry,
                              'numerics': self.numerics,
                              'material': self.material}

                for cat_key, cat_val in categories.items():
                    g1 = file.create_group('config/' + cat_key)

                    for key, value in cat_val.items():
                        g1.attrs.create(str(key), value)

            if str(i).zfill(len(str(self.maxIt))) not in file:

                g1 =file.create_group(str(i).zfill(len(str(self.maxIt))))

                g1.create_dataset('j_x',   data = self.q.field[0])
                g1.create_dataset('j_y',   data = self.q.field[1])
                g1.create_dataset('rho',   data = self.q.field[2])
                g1.create_dataset('press', data = DowsonHigginson(self.material).isoT_pressure(self.q.field[2]))

                g1.attrs.create('time', self.time)
                g1.attrs.create('mass', self.mass)
                g1.attrs.create('vmax', self.vmax)
                g1.attrs.create('CFL', self.cfl)
                g1.attrs.create('vSound', self.vSound)
                g1.attrs.create('dt', self.dt)

            file.close()
