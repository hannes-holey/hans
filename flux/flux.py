#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian


class Flux:

    def __init__(self, disc, geometry, numerics, material):

        self.disc = disc
        self.geometry = geometry
        self.material = material

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        self.dt = float(numerics['dt'])
        self.periodicX = bool(numerics['periodicX'])
        self.periodicY = bool(numerics['periodicY'])
        self.rey = bool(numerics['Rey'])

    def getFlux_LF(self, q, h, d, ax):

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress = Newtonian(self.disc).average_w4(q, h, self.geometry, self.material)

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
            dx = self.dx
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]
            dx = self.dy

        flux = VectorField(self.disc)

        flux.field[0] = 0.5 * (f1 + np.roll(f1, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = ax))
        flux.field[1] = 0.5 * (f2 + np.roll(f2, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = ax))
        flux.field[2] = 0.5 * (f3 + np.roll(f3, d, axis = ax)) - dx/(2. * self.dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = ax))

        if self.periodicX == False:
            if d == -1:
                flux.field[0][-1,:] = f1[-1,:] # Neumann
                flux.field[1][-1,:] = f2[-1,:] # Neumann
                flux.field[2][-1,:] = flux.field[2][-2,:] # Dirichlet

            elif d == 1:
                flux.field[0][0,:] = f1[0,:]
                flux.field[1][0,:] = f2[0,:]
                flux.field[2][0,:] = flux.field[2][1,:]

        if self.periodicY == False:
            if d == -1:
                flux.field[0][:,-1] = f1[:,-1]
                flux.field[1][:,-1] = f2[:,-1]
                flux.field[2][:,-1] = flux.field[2][:,-2]

            elif d == 1:
                flux.field[0][:,0] = f1[:,0]
                flux.field[1][:,0] = f2[:,0]
                flux.field[2][:,0] = flux.field[2][:,1]

        return flux

    def getQ_LW(self, q, h, d, ax):

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress = Newtonian(self.disc).average_w4(q, h, self.geometry, self.material)

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
            dx = self.dx
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]
            dx = self.dy

        Q = VectorField(self.disc)

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = ax)) - self.dt/(2. * dx) * d * (f1 - np.roll(f1, d, axis = ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = ax)) - self.dt/(2. * dx) * d * (f2 - np.roll(f2, d, axis = ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = ax)) - self.dt/(2. * dx) * d * (f3 - np.roll(f3, d, axis = ax))

        if self.periodicX == False:
            if d == -1:
                Q.field[0][-1,:] = q.field[0][-1,:]
                Q.field[1][-1,:] = q.field[1][-1,:]
                Q.field[2][-1,:] = q.field[2][-1,:]

            elif d == 1:
                Q.field[0][0,:] = q.field[0][0,:]
                Q.field[1][0,:] = q.field[1][0,:]
                Q.field[2][0,:] = q.field[2][0,:]

        if self.periodicY == False:
            if d == -1:
                pass
        #         Q.field[0][:,-1] = q.field[0][:,-1]
        #         Q.field[1][:,-1] = q.field[1][:,-1]
        #         Q.field[2][:,-1] = Q.field[2][:,-2]

            elif d == 1:
                pass
        #         Q.field[0][:,0] = q.field[0][:,0]
        #         Q.field[1][:,0] = q.field[1][:,0]
        #         Q.field[2][:,0] = Q.field[2][:,1]

        return Q

    def getFlux_LW(self, q, h, d, ax):

        Q = self.getQ_LW(q, h, d, ax)

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(Q, self.material)
            stress_center = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress_center = Newtonian(self.disc).average_w4(q, h, self.geometry, self.material)
            h.stagArray(d, ax)
            stress = Newtonian(self.disc).average_w4(Q, h, self.geometry, self.material)

        flux = VectorField(self.disc)

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = Q.field[0]
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = Q.field[1]

        flux.field[0] = f1
        flux.field[1] = f2
        flux.field[2] = f3

        if self.periodicX == False:
            if d == -1:
                pass
                # flux.field[0][-1,:] = -stress_center.field[0][-1,:] # Neumann
                # flux.field[1][-1,:] = -stress_center.field[2][-1,:] # Neumann
                # flux.field[2][-1,:] = f3[-2,:] # Dirichlet

            elif d == 1:
                pass
                # flux.field[0][0,:] = -stress_center.field[0][-1,:] # Neumann
                # flux.field[1][0,:] = -stress_center.field[2][-1,:] # Neumann
                # flux.field[2][0,:] = f3[1,:]

        if self.periodicY == False:
            if d == -1:
                pass
                # flux.field[0][:,-1] = f1[:,-1]
                # flux.field[1][:,-1] = f2[:,-1]
                # flux.field[2][:,-1] = flux.field[2][:,-2]

            elif d == 1:
                pass
                # flux.field[0][:,0] = f1[:,0]
                # flux.field[1][:,0] = f2[:,0]
                # flux.field[2][:,0] = flux.field[2][:,1]

        return flux

    def getQ_MC(self, q, h, d, ax):

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress = Newtonian(self.disc).average_w4(q, h, self.geometry, self.material)

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
            dx = self.dx
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]
            dx = self.dy

        Q = VectorField(self.disc)

        Q.field[0] = q.field[0] + self.dt/dx * (f1 - np.roll(f1, -1, axis = ax))
        Q.field[1] = q.field[1] + self.dt/dx * (f2 - np.roll(f2, -1, axis = ax))
        Q.field[2] = q.field[2] + self.dt/dx * (f3 - np.roll(f3, -1, axis = ax))

        return Q

    def getFlux_MC(self, q, h, d, ax):

        Q = self.getQ_MC(q, h, d, ax)

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(Q, self.material)
            stress_center = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress_center = Newtonian(self.disc).average_w4(q, h, self.geometry, self.material)
            # h.stagArray(d, ax)
            stress = Newtonian(self.disc).average_w4(Q, h, self.geometry, self.material)

        flux = VectorField(self.disc)

        if ax == 0:
            f1_p = -stress.field[0]
            f2_p = -stress.field[2]
            f3_p = Q.field[0]
            f1 = -stress_center.field[0]
            f2 = -stress_center.field[2]
            f3 = q.field[0]
        elif ax == 1:
            f1_p = -stress.field[2]
            f2_p = -stress.field[1]
            f3_p = Q.field[1]
            f1 = -stress_center.field[2]
            f2 = -stress_center.field[1]
            f3 = q.field[1]

        if d == -1:
            flux.field[0] = 0.5 * (np.roll(f1, d, axis=ax) + f1_p)
            flux.field[1] = 0.5 * (np.roll(f2, d, axis=ax) + f2_p)
            flux.field[2] = 0.5 * (np.roll(f3, d, axis=ax) + f3_p)

        if d == 1:
            flux.field[0] = 0.5 * (np.roll(f1_p, d, axis=ax) + f1)
            flux.field[1] = 0.5 * (np.roll(f2_p, d, axis=ax) + f2)
            flux.field[2] = 0.5 * (np.roll(f3_p, d, axis=ax) + f3)

        return flux
