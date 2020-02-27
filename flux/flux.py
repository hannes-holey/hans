#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian

class LaxFriedrichs:

    def __init__(self, disc, numerics, material):

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
        self.disc = disc

        self.flux = VectorField(disc)

    def fluxX(self, q, h, d):

        s0 = Newtonian(self.disc).Reynolds(q, self.material, 0)
        s1 = Newtonian(self.disc).Reynolds(q, self.material, 1)
        s5 = Newtonian(self.disc).Reynolds(q, self.material, 5)

        self.flux.field[0] = -0.5 * (s0.field[0] + np.roll(s0.field[0], d, axis = 0)) - self.dx/(2. * self.dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 0))
        self.flux.field[1] = -0.5 * (s5.field[0] + np.roll(s5.field[0], d, axis = 0)) - self.dx/(2. * self.dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 0))
        self.flux.field[2] =  0.5 * (q.field[0]  + np.roll(q.field[0],  d, axis = 0)) - self.dx/(2. * self.dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 0))

        if self.periodicX == 0:
            if d == -1:
                self.flux.field[0][-1,:] = -stress.field[0][-1,:]
                self.flux.field[1][-1,:] = -stress.field[5][-1,:]
                # self.field[2][-1,:] = q.field[0][-1,:]
            elif d == 1:
                self.flux.field[0][0,:] = -stress.field[0][0,:]
                self.flux.field[1][0,:] = -stress.field[5][0,:]
                # self.field[2][0,:] = q.field[0][0,:]

        return self.flux

    def fluxY(self, q, h, d):

        s0 = Newtonian(self.disc).Reynolds(q, self.material, 0)
        s1 = Newtonian(self.disc).Reynolds(q, self.material, 1)
        s5 = Newtonian(self.disc).Reynolds(q, self.material, 5)

        self.flux.field[0] = -0.5 * (s5.field[0] + np.roll(s5.field[0], d, axis = 1)) - self.dy/(2. * self.dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 1))
        self.flux.field[1] = -0.5 * (s1.field[0] + np.roll(s1.field[0], d, axis = 1)) - self.dy/(2. * self.dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 1))
        self.flux.field[2] =  0.5 * (q.field[1]  + np.roll(q.field[1],  d, axis = 1)) - self.dy/(2. * self.dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 1))

        if self.periodicY == 0:
            if d == -1:
                self.field[0][-1,:] = -stress.field[5][-1,:]
                self.field[1][-1,:] = -stress.field[1][-1,:]
                self.field[2][-1,:] = q.field[1][-1,:]

            elif d == 1:
                self.field[0][0,:] = -stress.field[5][0,:]
                self.field[1][0,:] = -stress.field[1][0,:]
                self.field[2][0,:] = q.field[1][0,:]

        return self.flux

    def addStress_wall(self, height, q, mu, U, V):
        self.field[0] -= 6.*mu*(U*q.field[2] - 2.*q.field[0])/(q.field[2]*height.field[0]**2)
        self.field[1] -= 6.*mu*(V*q.field[2] - 2.*q.field[1])/(q.field[2]*height.field[0]**2)

    def computeRHS(self, fXE, fXW, fYN, fYS):
        self.field[0] = 1./self.dx * (fXE.field[0] - fXW.field[0]) + 1./self.dy * (fYN.field[0] - fYS.field[0])
        self.field[1] = 1./self.dx * (fXE.field[1] - fXW.field[1]) + 1./self.dy * (fYN.field[1] - fYS.field[1])
        self.field[2] = 1./self.dx * (fXE.field[2] - fXW.field[2]) + 1./self.dy * (fYN.field[2] - fYS.field[2])

# class LaxWendroff(NumericalFlux):
#
#     def __init__(self, Nx, Ny, Lx, Ly, stressFunc):
#         super().__init_(Nx, Ny, Lx, Ly, stressFunc)
#
#     def QX_LF(self, stress, q, dt, d, periodic):
#         self.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = 0)) + dt/(2. * self.dx) * d * (stress.field[0] - np.roll(stress.field[0], d, axis = 0))
#         self.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = 0)) + dt/(2. * self.dx) * d * (stress.field[5] - np.roll(stress.field[5], d, axis = 0))
#         self.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = 0)) - dt/(2. * self.dx) * d * (q.field[0] - np.roll(q.field[0], d, axis = 0))
#
#
#     def QY_LF(self, stress, q, dt, d, periodic):
#         self.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = 1)) + dt/(2. * self.dy) * d * (stress.field[5] - np.roll(stress.field[5], d, axis = 1))
#         self.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = 1)) + dt/(2. * self.dy) * d * (stress.field[1] - np.roll(stress.field[1], d, axis = 1))
#         self.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = 1)) - dt/(2. * self.dy) * d * (q.field[1] - np.roll(q.field[1], d, axis = 1))
#
#     def computeRHS_LW(self, fXE, fXW, fYN, fYS, Q_E, Q_W, Q_N, Q_S):
#         self.field[0] = - 1./self.dx * (fXE.field[0] - fXW.field[0]) - 1./self.dy * (fYN.field[5] - fYS.field[5])
#         self.field[1] = - 1./self.dx * (fXE.field[5] - fXW.field[5]) - 1./self.dy * (fYN.field[5] - fYS.field[5])
#         self.field[2] =   1./self.dx * (Q_E.field[0] - Q_W.field[0]) + 1./self.dy * (Q_N.field[1] - Q_S.field[1])
