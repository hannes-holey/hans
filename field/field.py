#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Field:

    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Ly
        self.Ly = Ly

        x = np.linspace(0.0, self.Lx, self.Nx)
        y = np.linspace(0.0, self.Ly, self.Ny)
        xx, yy = np.meshgrid(x,y)

        self.xx = xx.T
        self.yy = yy.T

        self.field_buffer = np.zeros((self.ndim, self.Nx, self.Ny))

        self.field = []

        for k in range(self.ndim):
            self.field.append(np.ndarray(shape=(Nx,Ny), dtype=np.float64, buffer=self.field_buffer[k]))

class scalarField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 1
        super().__init__(Nx, Ny, Lx, Ly)

    def normal(self, loc, scale):
        "instantiate scalar field with normal distributed values"
        self.field[0] = np.random.normal(loc, scale, size=(self.Nx, self.Ny))

    def fromFunctionXY(self, func):
        "get scalar field from function of XY position"
        self.field[0]=func(self.xx, self.yy)

    def fromFunctionField(self, func, arg):
        "get scalar field from function of a 2D scalar field"
        self.field[0] = func(arg)

    def computeGrad(self):
        "Compute gradient field of a given scalar field in 2D"
        grad = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        # create E/W/N/S arrays
        pE = np.roll(self.field[0], 1,axis=0)
        pW = np.roll(self.field[0],-1,axis=0)
        pN = np.roll(self.field[0], 1,axis=1)
        pS = np.roll(self.field[0],-1,axis=1)

        # central differences
        grad.field[0] = (pE - pW)/2./dx
        grad.field[1] = (pN - pS)/2./dy
        grad.field[2] = np.zeros((self.Nx, self.Ny))

        # pE = self.field[0][1:self.Nx, :]
        # pW = self.field[0][0:self.Nx-1, :]
        # pN = self.field[0][:, 0:self.Ny-1]
        # pS = self.field[0][:, 0:self.Ny-1]
        #
        # # central differences
        # grad.field[0][1:self.Nx-1,:] = (pE - pW)/2./dx
        # grad.field[1][:,1:self.Ny-1] = (pN - pS)/2./dy
        # grad.field[2] = np.zeros((self.Nx, self.Ny))

        return grad

    def setDirichlet(self, value):
        self.field[0][0,:] = value
        self.field[0][-1,:] = value
        self.field[0][:,0] = value
        self.field[0][:,-1] = value


    def updateDens(self, field, dt):
        self.field[0] = dt * field.field[0] + self.field[0]

class vectorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 3
        super().__init__(Nx, Ny, Lx, Ly)

    def fromField(self, field):
        for i in range(self.ndim):
            self.field[i] = field.field[i]

    def addFluxContribution(self, func, velX, velY, h, fluxX, fluxY, rho):
        self.field[0] = -self.field[0] - func(velX, h, fluxX, rho)
        self.field[1] = -self.field[1] - func(velY, h, fluxY, rho)

    def computeDiv(self):
        "Compute divergence using central differences"

        div = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        # create E/W/N/S arrays
        pE = np.roll(self.field[0], 1,axis=0)
        pW = np.roll(self.field[0],-1,axis=0)
        pN = np.roll(self.field[1], 1,axis=1)
        pS = np.roll(self.field[1],-1,axis=1)

        div.field[0] = (pE - pW)/2./dx + (pN - pS)/2./dy

        return div

    def updateFlux(self, field, dt):
        self.field[0] = dt * field.field[0] + self.field[0]
        self.field[1] = dt * field.field[1] + self.field[1]

class tensorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 6
        super().__init__(Nx, Ny, Lx, Ly)


    def getStressNewtonian(self, p, func, velX, velY,  h):

        for i in range(int(self.ndim/2)):
            self.field[i] =  -p                         # normal

        self.field[3] = func(velY, h)                   # sigma_yz
        self.field[4] = func(velX, h)                   # sigma_xz
        self.field[5] = np.zeros((self.Nx, self.Ny))    # sigma_xy
