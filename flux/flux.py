#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian


class Flux:

    def __init__(self, disc, numerics, material):

        self.disc = disc

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        self.dt = float(numerics['dt'])

        self.periodicX = bool(numerics['periodicX'])
        self.periodicY = bool(numerics['periodicY'])

        self.material = material


    def getFlux_LF(self, q, h, d, ax):

        if ax == 0:
            f1 = -Newtonian(self.disc).Reynolds(q, self.material, 0).field[0]
            f2 = -Newtonian(self.disc).Reynolds(q, self.material, 5).field[0]
            f3 = q.field[0]
            dx = self.dx
        elif ax == 1:
            f1 = -Newtonian(self.disc).Reynolds(q, self.material, 5).field[0]
            f2 = -Newtonian(self.disc).Reynolds(q, self.material, 1).field[0]
            f3 = q.field[1]
            dx = self.dy

        flux = VectorField(self.disc)

        flux.field[0] = 0.5 * (f1 + np.roll(f1, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = ax))
        flux.field[1] = 0.5 * (f2 + np.roll(f2, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = ax))
        flux.field[2] = 0.5 * (f3 + np.roll(f3, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = ax))

        # if self.periodicX == 0:
        #     if d == -1:
        #         self.flux.field[0][-1,:] = -stress.field[0][-1,:]
        #         self.flux.field[1][-1,:] = -stress.field[5][-1,:]
        #         self.field[2][-1,:] = q.field[0][-1,:]
        #     elif d == 1:
        #         self.flux.field[0][0,:] = -stress.field[0][0,:]
        #         self.flux.field[1][0,:] = -stress.field[5][0,:]
        #         self.field[2][0,:] = q.field[0][0,:]

        # if self.periodicY == 0:
        #     if d == -1:
        #         self.field[0][-1,:] = -stress.field[5][-1,:]
        #         self.field[1][-1,:] = -stress.field[1][-1,:]
        #         self.field[2][-1,:] = q.field[1][-1,:]
        #
        #     elif d == 1:
        #         self.field[0][0,:] = -stress.field[5][0,:]
        #         self.field[1][0,:] = -stress.field[1][0,:]
        #         self.field[2][0,:] = q.field[1][0,:]

        return flux

    def getQ_LW(self, q, h, d, ax):

        if ax == 0:
            f1 = -Newtonian(self.disc).Reynolds(q, self.material, 0).field[0]
            f2 = -Newtonian(self.disc).Reynolds(q, self.material, 5).field[0]
            f3 = q.field[0]
            dx = self.dx
        elif ax == 1:
            f1 = -Newtonian(self.disc).Reynolds(q, self.material, 5).field[0]
            f2 = -Newtonian(self.disc).Reynolds(q, self.material, 1).field[0]
            f3 = q.field[1]
            dx = self.dy

        Q = VectorField(self.disc)

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = ax)) - self.dt/(2. * dx) * d * (f1 - np.roll(f1, d, axis = ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = ax)) - self.dt/(2. * dx) * d * (f2 - np.roll(f2, d, axis = ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = ax)) - self.dt/(2. * dx) * d * (f3 - np.roll(f3, d, axis = ax))

        return Q

    def getFlux_LW(self, q, h, d, ax):

        flux = VectorField(self.disc)

        Q = self.getQ_LW(q, h, d, ax)

        if ax ==0:
            flux.field[0] = -Newtonian(self.disc).Reynolds(Q, self.material, 0).field[0]
            flux.field[1] = -Newtonian(self.disc).Reynolds(Q, self.material, 5).field[0]
            flux.field[2] = Q.field[0]
        elif ax == 1:
            flux.field[0] = -Newtonian(self.disc).Reynolds(Q, self.material, 5).field[0]
            flux.field[1] = -Newtonian(self.disc).Reynolds(Q, self.material, 1).field[0]
            flux.field[2] = Q.field[1]

        return flux
