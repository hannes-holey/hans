#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Field:

    def __init__(self, disc, ndim):

        self.ndim = ndim
        self.disc = disc

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

        out = Field(self.disc, self.ndim)

        for i in range(self.ndim):
            out.field[i] = 0.5 * (self.field[i] + np.roll(self.field[i], dir, axis = ax))

        return out

    def cellAverage(self):

        out = Field(self.disc, self.ndim)

        E = self.stagArray(-1, 0)
        W = self.stagArray( 1, 0)

        NE = E.stagArray(-1, 1)
        SE = E.stagArray( 1, 1)
        NW = W.stagArray(-1, 1)
        SW = W.stagArray( 1, 1)

        for i in range(self.ndim):
            out.field[i] = 0.25 * (NE.field[i] + SE.field[i] + NW.field[i] + SW.field[i])

        return out

    def copy(self):

        out = Field(self.disc , self.ndim)

        for i in range (self.ndim):
            out.field[i] = self.field[i]

        return out 

class ScalarField(Field):

    def __init__(self, disc):
        self.ndim = 1
        super().__init__(disc, self.ndim)

class VectorField(Field):

    def __init__(self, disc):
        self.ndim = 3
        super().__init__(disc, self.ndim)

    def getGradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"
        self.field[1] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[0]
        self.field[2] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[1]

    def updateExplicit(self, rhs, dt):
        self.field[0] = self.field[0] - dt * rhs.field[0]
        self.field[1] = self.field[1] - dt * rhs.field[1]
        self.field[2] = self.field[2] - dt * rhs.field[2]

class TensorField(Field):

    def __init__(self, disc):
        self.ndim = 6
        super().__init__(disc, self.ndim)

    # def addNoise(self, frac):
    #     for i in range(self.ndim):
    #         mean = self.field[i]
    #         mu = frac * np.amax(abs(self.field[i]))
    #         self.field[i] += np.random.normal(mean, mu)
