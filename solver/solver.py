#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from eos.eos import DowsonHigginson, PowerLaw
from geo.geometry import Analytic
from field.field import VectorField
from flux.flux import Flux

class Solver:

    def __init__(self, disc, geometry, numerics, material):

        self.type = str(geometry['type'])

        self.numFlux = str(numerics['numFlux'])
        self.adaptive = bool(numerics['adaptive'])
        self.dt = float(numerics['dt'])
        self.maxIt = int(numerics['maxT'] * 1e9 /self.dt)
        self.C = numerics['C']

        if material['EOS'] == 'DH':
            self.eqOfState = DowsonHigginson(material)
        elif material['EOS'] == 'PL':
            self.eqOfState = PowerLaw(material)

        self.time = 0

        # Stokes assumption
        if material['Stokes'] == True:
            material['lambda'] = -2./3. * material['mu']

        # Gap height
        self.height = VectorField(disc)

        if self.type == 'journal':
            self.height.fromFunctionXY(Analytic(disc, geometry).journalBearing, 0)
        else:
            self.height.fromFunctionXY(Analytic(disc, geometry).linearSlider, 0)

        self.height.getGradients()

        P0 = float(material['P0'])
        rho0 = float(material['rho0'])

        self.q = VectorField(disc)
        self.q.fill(rho0, 2)

        if self.type == 'inclined':
            self.q.field[2][0,:] = self.eqOfState.isoT_density(P0)
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(P0)
        elif self.type == 'poiseuille':
            self.q.field[2][-1,:] = self.eqOfState.isoT_density(P0)
            self.q.field[2][0,:] = self.eqOfState.isoT_density(2. * P0)
        elif self.type == 'droplet':
            self.q.fill_circle(1.e-4, self.eqOfState.isoT_density(2. * P0), 2)
        elif self.type == 'wavefront':
            self.q.fill_line(0.25, 5e-5, self.eqOfState.isoT_density(2. * P0), 0, 2)

        self.Flux = Flux(disc, geometry, numerics, material)

        self.rhs = VectorField(disc)

    def solve(self, i):

        self.vSound = np.amax(self.eqOfState.soundSpeed(self.q.field[2]))
        self.vmax = max(np.amax(1./self.q.field[2]*np.sqrt(self.q.field[0]**2 + self.q.field[1]**2)), 1e-3)

        if self.adaptive == True:
            if i == 0:
                self.dt = self.dt
            else:
                self.dt = self.C * min(self.q.dx, self.q.dy)/self.vSound

        if self.numFlux == 'LF':
            fXE = self.Flux.getFlux_LF(self.q, self.height, -1, 0)
            fXW = self.Flux.getFlux_LF(self.q, self.height,  1, 0)
            fYN = self.Flux.getFlux_LF(self.q, self.height, -1, 1)
            fYS = self.Flux.getFlux_LF(self.q, self.height,  1, 1)

        elif self.numFlux == 'LW':
            fXE = self.Flux.getFlux_LW(self.q, self.height, -1, 0)
            fXW = self.Flux.getFlux_LW(self.q, self.height,  1, 0)
            fYN = self.Flux.getFlux_LW(self.q, self.height, -1, 1)
            fYS = self.Flux.getFlux_LW(self.q, self.height,  1, 1)

        elif self.numFlux == 'MC':
            fXE = self.Flux.getFlux_MC(self.q, self.height, -1, 0)
            fXW = self.Flux.getFlux_MC(self.q, self.height,  1, 0)
            fYN = self.Flux.getFlux_MC(self.q, self.height, -1, 1)
            fYS = self.Flux.getFlux_MC(self.q, self.height,  1, 1)

        self.rhs.field[0] = 1./self.rhs.dx * (fXE.field[0] - fXW.field[0]) + 1./self.rhs.dy * (fYN.field[0] - fYS.field[0])
        self.rhs.field[1] = 1./self.rhs.dx * (fXE.field[1] - fXW.field[1]) + 1./self.rhs.dy * (fYN.field[1] - fYS.field[1])
        self.rhs.field[2] = 1./self.rhs.dx * (fXE.field[2] - fXW.field[2]) + 1./self.rhs.dy * (fYN.field[2] - fYS.field[2])

        corr = self.Flux.fluxAnalytic(self.q, self.height)

        self.rhs.field[0] -= corr.field[0]
        self.rhs.field[1] -= corr.field[1]
        self.rhs.field[2] -= corr.field[2]

        # explicit time step
        self.q.updateExplicit(self.rhs, self.dt)

        # some scalar output
        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
        self.time += self.dt

        self.eps = abs(np.amax(1./self.q.field[2]*np.sqrt(self.q.field[0]**2 + self.q.field[1]**2)) - self.vmax)/(self.vmax/self.dt)
