#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian
from eos.eos import DowsonHigginson, PowerLaw

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
            stress = Newtonian(self.disc).full(q, h, self.geometry, self.material)

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
            stress = Newtonian(self.disc).full(q, h, self.geometry, self.material)

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
                Q.field[0][:,-1] = q.field[0][:,-1]
                Q.field[1][:,-1] = q.field[1][:,-1]
                Q.field[2][:,-1] = Q.field[2][:,-2]

            elif d == 1:
                Q.field[0][:,0] = q.field[0][:,0]
                Q.field[1][:,0] = q.field[1][:,0]
                Q.field[2][:,0] = Q.field[2][:,1]

        return Q

    def getFlux_LW(self, q, h, d, ax):

        Q = self.getQ_LW(q, h, d, ax)

        if self.rey == True:
            stress = Newtonian(self.disc).reynolds(Q, self.material)
            stress_center = Newtonian(self.disc).reynolds(q, self.material)
        elif self.rey == False:
            stress_center = Newtonian(self.disc).full(q, h, self.geometry, self.material)
            h.stagArray(d, ax)
            stress = Newtonian(self.disc).full(Q, h, self.geometry, self.material)

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
            stress = Newtonian(self.disc).full(q, h, self.geometry, self.material)

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
            stress_center = Newtonian(self.disc).full(q, h, self.geometry, self.material)
            h.stagArray(d, ax)
            stress = Newtonian(self.disc).full(Q, h, self.geometry, self.material)

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

    def addAnalytic(self, rhs, q, h):

        mu = self.material['mu']
        lam = self.material['lambda']

        U = self.geometry['U']
        V = self.geometry['V']

        if self.material['EOS'] == 'DH':
            eqOfState = DowsonHigginson(self.material)
        elif self.material['EOS'] == 'PL':
            eqOfState = PowerLaw(self.material)

        P = eqOfState.isoT_pressure(q.field[2])
        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]
        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        if self.rey == True:
            rhs.field[0] -= P * hx/h0 + 6.* mu*(U*rho - 2.*j_x)/(rho * h0**2)
            rhs.field[1] -= P * hy/h0 + 6.* mu*(V*rho - 2.*j_y)/(rho * h0**2)
            rhs.field[2] -= -j_x * hx / h0 - j_y * hy / h0
        elif self.rey == False:
            rhs.field[0] -= (8*(lam/2 + mu)*(U*rho - 1.5*j_x)*hx**2 + (4*(V*rho - 1.5*j_y)*(mu + lam)*hy + P*h0*rho)*hx \
                            + 4*mu*((U*rho - 1.5*j_x)*hy**2 + 1.5*U*rho - 3*j_x))/(h0**2*rho)
            rhs.field[1] -= (8*(V*rho - 1.5*j_y)*(lam/2 + mu)*hy**2 + (4*hx*(U*rho - 1.5*j_x)*(mu + lam) + P*h0*rho)*hy \
                            + 4*mu*((V*rho - 1.5*j_y)*hx**2 + 1.5*V*rho - 3*j_y))/(h0**2*rho)
            rhs.field[2] -= -j_x * hx / h0 - j_y * hy / h0

        return rhs
