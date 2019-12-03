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

    def computeGrad(self):

        grad = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        pE = np.roll(self.field[0], 1,axis=0)
        pW = np.roll(self.field[0],-1,axis=0)
        pN = np.roll(self.field[0], 1,axis=1)
        pS = np.roll(self.field[0],-1,axis=1)

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        print(dx)

        grad.field[0] = (pE -pW)/2./dx
        grad.field[1] = (pN -pS)/2./dy

        return grad


class vectorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 2
        super().__init__(Nx, Ny, Lx, Ly)

class tensorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 6
        super().__init__(Nx, Ny, Lx, Ly)
