#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson, PowerLaw

class Field:

    def __init__(self, disc):
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])
        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])

        self.dx = self.Lx/(self.Nx)
        self.dy = self.Ly/(self.Ny)

        x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.Nx)
        y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.Ny)
        xx, yy = np.meshgrid(x,y)

        self.xx = xx.T
        self.yy = yy.T

        self.field = []

        for k in range(self.ndim):
            self.field.append(np.zeros(shape = (self.Nx, self.Ny), dtype=np.float64))

    def fill(self, value, comp):
        self.field[comp] = np.ones(shape=(self.Nx, self.Ny)) * value

    def fill_circle(self, radius, value, comp):
        for i in range(self.Nx):
            for j in range(self.Ny):
                if ((self.xx[i,j]-self.Lx/2)**2 + (self.yy[i,j]-self.Ly/2)**2) < radius**2:
                    self.field[comp][i,j] = value

    def fill_line(self, loc, width, value, ax, comp):
        for i in range(self.Nx):
            for j in range(self.Ny):
                if ax == 0:
                    if abs(self.xx[i,j] - loc * self.Lx) < width/2.:
                        self.field[comp][i,j] = value
                elif ax == 1:
                    if abs(self.yy[i,j]- loc * self.Ly) < width/2.:
                        self.field[comp][i,j] = value


    def fromFunctionXY(self, func, comp):
        self.field[comp]=func(self.xx, self.yy)

    def fromFunctionField(self, func, arg, comp):
        self.field[comp] = func(arg)

    def fromField(self, field):
        for i in range(self.ndim):
            self.field[i] = field.field[i]

    def edgesField(self):
        self.dx = self.Lx/(self.Nx - 1)
        self.dy = self.Ly/(self.Ny - 1)

        x = np.linspace(0., self.Lx, self.Nx)
        y = np.linspace(0., self.Ly, self.Ny)
        xx, yy = np.meshgrid(x,y)

        self.xx = xx.T
        self.yy = yy.T

    def stagArray(self, dir, ax):

        for i in range(self.ndim):
            self.field[i] = 0.5 * (self.field[i] + np.roll(self.field[i], dir, axis = ax))

class ScalarField(Field):

    def __init__(self, disc):
        self.ndim = 1
        super().__init__(disc)

class VectorField(Field):

    def __init__(self, disc):
        self.ndim = 3
        super().__init__(disc)

    def getGradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"
        self.field[1] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[0]
        self.field[2] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[1]

    def computeRHS(self, fXE, fXW, fYN, fYS):
        self.field[0] = 1./self.dx * (fXE.field[0] - fXW.field[0]) + 1./self.dy * (fYN.field[0] - fYS.field[0])
        self.field[1] = 1./self.dx * (fXE.field[1] - fXW.field[1]) + 1./self.dy * (fYN.field[1] - fYS.field[1])
        self.field[2] = 1./self.dx * (fXE.field[2] - fXW.field[2]) + 1./self.dy * (fYN.field[2] - fYS.field[2])

    # def addStress_wall(self, q, h, mu, U, V):
    #     self.field[0] -= 6.*mu*(U*q.field[2] - 2.*q.field[0])/(q.field[2]*h.field[0]**2)
    #     self.field[1] -= 6.*mu*(V*q.field[2] - 2.*q.field[1])/(q.field[2]*h.field[0]**2)

    def addStress_wall(self, q, h, U, V, mat, rey):

        mu = mat['mu']
        lam = mat['lambda']

        if mat['EOS'] == 'DH':
            eqOfState = DowsonHigginson(mat)
        elif mat['EOS'] == 'PL':
            eqOfState = PowerLaw(mat)

        P = eqOfState.isoT_pressure(q.field[2])
        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]
        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        if rey == True:
            self.field[0] -= P * hx/h0 + 6.* mu*(U*rho - 2.*j_x)/(rho * h0**2)
            self.field[1] -= P * hy/h0 + 6.* mu*(V*rho - 2.*j_y)/(rho * h0**2)
            self.field[2] -= -j_x * hx / h0 - j_y * hy / h0
        elif rey == False:
            self.field[0] -= (8*(lam/2 + mu)*(U*rho - 1.5*j_x)*hx**2 + (4*(V*rho - 1.5*j_y)*(mu + lam)*hy + P*h0*rho)*hx + 4*mu*((U*rho - 1.5*j_x)*hy**2 + 1.5*U*rho - 3*j_x))/(h0**2*rho)

            self.field[1] -= (8*(V*rho - 1.5*j_y)*(lam/2 + mu)*hy**2 + (4*hx*(U*rho - 1.5*j_x)*(mu + lam) + P*h0*rho)*hy + 4*mu*((V*rho - 1.5*j_y)*hx**2 + 1.5*V*rho - 3*j_y))/(h0**2*rho)

            self.field[2] -= -j_x * hx / h0 - j_y * hy / h0

## v1

        # self.field[0] -= P * h.field[1]/h.field[0] + (6. * mu * (U * q.field[2] - 2.*q.field[0]) + 8.*(lam/2.+ mu)*(U*q.field[2] - 1.5 * q.field[0]))/(h.field[0]**2 * q.field[2])
        #
        # self.field[1] -=  1./(h.field[0]**2 * q.field[2]) * (2 * mu * (2. * h.field[1]**2 * V * q.field[0] - 3.*h.field[1]**2 * q.field[1] + 3. * V * q.field[2] - 6.*q.field[1]))
        #
        # self.field[2] -= -q.field[0] * h.field[1] / h.field[0] - q.field[1] * h.field[2] / h.field[0]

    def updateExplicit(self, rhs, dt):
        self.field[0] = self.field[0] - dt * rhs.field[0]
        self.field[1] = self.field[1] - dt * rhs.field[1]
        self.field[2] = self.field[2] - dt * rhs.field[2]

class TensorField(Field):

    def __init__(self, disc):
        self.ndim = 6
        super().__init__(disc)

    # def addNoise(self, frac):
    #     for i in range(self.ndim):
    #         mean = self.field[i]
    #         mu = frac * np.amax(abs(self.field[i]))
    #         self.field[i] += np.random.normal(mean, mu)
