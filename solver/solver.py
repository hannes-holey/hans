#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from eos.eos import DowsonHigginson, PowerLaw
from geo.geometry import Analytic
from field.field import VectorField
from flux.flux import Flux
from stress.stress import Newtonian


class Solver:

    def __init__(self, disc, geometry, numerics, material):

        self.type = str(geometry['type'])
        self.numFlux = str(numerics['numFlux'])
        self.adaptive = bool(numerics['adaptive'])
        self.dt = float(numerics['dt'])
        self.C = numerics['C']
        self.fluct = bool(material['Fluctuating'])

        if material['EOS'] == 'DH':
            self.eqOfState = DowsonHigginson(material)
        elif material['EOS'] == 'PL':
            self.eqOfState = PowerLaw(material)

        self.time = 0

        # Gap height
        self.height = VectorField(disc)

        if self.type == 'journal':
            self.height.fromFunctionXY(Analytic(disc, geometry).journalBearing, 0, axis=0)
        else:
            self.height.fromFunctionXY(Analytic(disc, geometry).linearSlider, 0, axis=0)

        self.height.getGradients()

        P0 = float(material['P0'])
        rho0 = float(material['rho0'])

        self.q = VectorField(disc)
        self.q.field[2] = rho0

        if self.type == 'inclined':
            self.q.field[2][0,:] = self.eqOfState.isoT_density(P0)
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(P0)
        elif self.type == 'poiseuille':
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(P0)
            self.q.field[2][0,:] = self.eqOfState.isoT_density(2. * P0)
        elif self.type == 'droplet':
            self.q.fill_circle(self.eqOfState.isoT_density(2. * P0), 2)
        elif self.type == 'wavefront':
            self.q.fill_line(self.eqOfState.isoT_density(2. * P0), 0, 2)

        self.Flux = Flux(disc, geometry, numerics, material)
        self.Newtonian = Newtonian(disc, geometry, material)

        # self.HFlux = HyperbolicFlux(disc, geometry, numerics, material)
        # self.vSound = self.eqOfState.soundSpeed(rho0)

    def solve(self, i):

        self.vSound = np.amax(self.eqOfState.soundSpeed(self.q.field[2]))
        self.vmax = max(np.amax(1. / self.q.field[2] * np.sqrt(self.q.field[0]**2 + self.q.field[1]**2)), 1e-3)

        # self.vmax = self.vSound

        if self.adaptive is True:
            if i == 0:
                self.dt = self.dt
            else:
                self.dt = self.C * min(self.q.dx, self.q.dy) / (self.vSound + self.vmax)

        # viscousStress, stress, cov3, p = self.Newtonian.stress_avg(self.q, self.height, self.dt)

        # if self.fluct is True:
        #     stress.addNoise_FH(cov3)

        # if self.numFlux == 'LF':
        #     fXE = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, -1, 0)
        #     fXW = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, 1, 0)
        #     fYN = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, -1, 1)
        #     fYS = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, 1, 1)
        #
        #     rhs = -1. / self.q.dx * (fXE.field - fXW.field) - 1. / self.q.dy * (fYN.field - fYS.field)
        #     source = self.Flux.getSource(viscousStress, self.q, self.height, self.dt)
        #     rhs += source.field
        #     self.q.field += self.dt * rhs

        if self.numFlux == 'LW':
            self.q = self.Flux.Richtmyer(self.q, self.height, self.dt)

        elif self.numFlux == 'MC_old':
            self.q = self.Flux.MacCormack_total(self.q, self.height, self.dt)

        elif self.numFlux == 'MC':
            self.q = self.Flux.MacCormack(self.q, self.height, self.dt, i)

        elif self.numFlux == 'RK3':
            self.q = self.Flux.RungeKutta3(self.q, self.height, self.dt, i)

        # some scalar output
        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
        self.time += self.dt

        vmax_new = np.amax(np.sqrt(self.q.field[0] * self.q.field[0] + self.q.field[1] * self.q.field[1]) / self.q.field[2])
        self.eps = abs(vmax_new - self.vmax) / self.vmax / self.C
        # self.eps = 1.
