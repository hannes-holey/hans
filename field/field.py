#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Field:

    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Ly
        self.Ly = Ly

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

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
        "get scalar field from function"
        self.field[0] = func(arg)

    def neigharray(self):
        "shifts array A to get arrays with East, West neighbors of each point"
        AE = np.roll(self.field[0], -1, axis=0)  # east  array
        AW = np.roll(self.field[0], +1, axis=0)  # west  array
        AN = np.roll(self.field[0], -1, axis=1)  # north array
        AS = np.roll(self.field[0], +1, axis=1)  # south array
        return (AE, AW, AN, AS)

    def stagarray(self):
        "calculates A array on staggered grid (arithmetic mean between central value and neighbors)"
        AE, AW, AN, AS = neigharray(self.field[0])
        Ae = 0.5 * (self.field[0] + AE)
        Aw = 0.5 * (self.field[0] + AW)
        An = 0.5 * (self.field[0] + AN)
        As = 0.5 * (self.field[0] + AS)
        return (Ae, Aw, An, As)

    def computeDiv_CD_np(self, field):
        "Compute divergence using central differences"

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        self.field[0] = np.gradient(field.field[0],dx,dy, edge_order=2)[0] + np.gradient(field.field[2],dx,dy,edge_order=2)[1]

    def computeDiv_CD(self, field):
        "Compute divergence using central differences"

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        S0E = np.roll(field.field[0], -1, axis=0)
        S0W = np.roll(field.field[0],  1, axis=0)
        S0N = np.roll(field.field[0], -1, axis=1)
        S0S = np.roll(field.field[0],  1, axis=1)

        S1E = np.roll(field.field[1], -1, axis=0)
        S1W = np.roll(field.field[1],  1, axis=0)
        S1N = np.roll(field.field[1], -1, axis=1)
        S1S = np.roll(field.field[1],  1, axis=1)

        # ghost cells
        S0E[-1,:] = field.field[0][-1,:]
        S0W[0,:]  = field.field[0][0,:]
        S0N[:,-1] = field.field[0][:,-1]
        S0S[:,0]  = field.field[0][:,0]

        S1E[-1,:] = field.field[1][-1,:]
        S1W[0,:]  = field.field[1][0,:]
        S1N[:,-1] = field.field[1][:,-1]
        S1S[:,0]  = field.field[1][:,0]

        self.field[0]= (S0E - S0W)/(2.*dx) + (S1N - S1S)/(2.*dy)

        self.field[0][0,:] *= 2.
        self.field[0][-1,:] *= 2.
        self.field[0][:,0] *= 2.
        self.field[0][:,-1] *= 2.

    def computeHelper_LW(self, field):

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        S0E = np.roll(field.field[0], -1, axis=0)
        S0W = np.roll(field.field[0],  1, axis=0)
        S0N = np.roll(field.field[0], -1, axis=1)
        S0S = np.roll(field.field[0],  1, axis=1)

        S1E = np.roll(field.field[1], -1, axis=0)
        S1W = np.roll(field.field[1],  1, axis=0)
        S1N = np.roll(field.field[1], -1, axis=1)
        S1S = np.roll(field.field[1],  1, axis=1)

        # ghost cells
        S0E[-1,:] = field.field[0][-1,:]
        S0W[0,:]  = field.field[0][0,:]
        S0N[:,-1] = field.field[0][:,-1]
        S0S[:,0]  = field.field[0][:,0]

        S1E[-1,:] = field.field[1][-1,:]
        S1W[0,:]  = field.field[1][0,:]
        S1N[:,-1] = field.field[1][:,-1]
        S1S[:,0]  = field.field[1][:,0]

        self.field[0] = (S0W - 2.*field.field[0] + S0E)/(2.*dx**2) + (S1S - 2.*field.field[1] + S1N)/(2.*dy**2)

        self.field[0][0,:] *= 2.
        self.field[0][-1,:] *= 2.
        self.field[0][:,0] *= 2.
        self.field[0][:,-1] *= 2.

    def setDirichletXw(self, value):
        self.field[0][0,:] = value
        # self.field[0][1,:] = value

    def setDirichletXe(self, value):
        self.field[0][-1,:] = value
        # self.field[0][-2,:] = value

    def setDirichletYn(self, value):
        self.field[0][:,-1] = value
        # self.field[0][1,:] = value

    def setDirichletYs(self, value):
        self.field[0][:,0] = value
        # self.field[0][-2,:] = value

    def setDirichletY(self, value):
        self.field[0][:,0] = value
        self.field[0][:,-1] = value
        # self.field[0][:,0] = value
        # self.field[0][:,-1] = value

class vectorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 3
        super().__init__(Nx, Ny, Lx, Ly)

    def fromField(self, field):
        for i in range(self.ndim):
            self.field[i] = field.field[i]

    def computeDiv_CD_np(self, field):
        "Compute divergence using central differences"

        # uniform grid
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        self.field[0] = np.gradient(field.field[0],dx,dy, edge_order=2)[0] +\
                        np.gradient(field.field[5],dx,dy,edge_order=2)[1]
        self.field[1] = np.gradient(field.field[5],dx,dy, edge_order=2)[0] +\
                        np.gradient(field.field[1],dx,dy,edge_order=2)[1]

    def computeFluxDiv_np(self, field):
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        self.field[2] = -np.gradient(field.field[0],dx,dy, edge_order=1)[0] \
                        -np.gradient(field.field[1],dx,dy,edge_order=1)[1]

    def compute_CD1(self, field, ax):

        if ax == 0:
            dx = self.Lx/self.Nx
        elif ax == 1:
            dx = self.Ly/self.Ny

        f0E = np.roll(field.field[0], -1, axis=ax)
        f0W = np.roll(field.field[0],  1, axis=ax)

        f1E = np.roll(field.field[1], -1, axis=ax)
        f1W = np.roll(field.field[1],  1, axis=ax)

        f2E = np.roll(field.field[2], -1, axis=ax)
        f2W = np.roll(field.field[2],  1, axis=ax)

        # ghost cells
        if ax == 0:
            f0E[-1,:] = field.field[0][-1,:]
            f0W[0,:]  = field.field[0][0,:]

            f1E[-1,:] = field.field[1][-1,:]
            f1W[0,:]  = field.field[1][0,:]

            f2E[-1,:] = field.field[2][-1,:]
            f2W[0,:]  = field.field[2][0,:]

        elif ax == 1:
            f0E[:,-1] = field.field[0][:,-1]
            f0W[:,0]  = field.field[0][:,0]

            f1E[:,-1] = field.field[1][:,-1]
            f1W[:,0]  = field.field[1][:,0]

            f2E[:,-1] = field.field[2][:,-1]
            f2W[:,0]  = field.field[2][:,0]

        self.field[0] =  (f0E - f0W)/(2.*dx)
        self.field[1] =  (f1E - f1W)/(2.*dx)
        self.field[2] =  (f2E - f2W)/(2.*dx)

        # boundary correction
        if ax == 0:
            self.field[0][0,:] *= 2.
            self.field[0][-1,:] *= 2.

            self.field[1][0,:] *= 2.
            self.field[1][-1,:] *= 2.

            self.field[2][0,:] *= 2.
            self.field[2][-1,:] *= 2.
        elif ax == 1:
            self.field[0][:,0] *= 2.
            self.field[0][:,-1] *= 2.

            self.field[1][:,0] *= 2.
            self.field[1][:,-1] *= 2.

            self.field[2][:,0] *= 2.
            self.field[2][:,-1] *= 2.

    def compute_CD2(self, field, ax):

        if ax == 0:
            dx = self.Lx/self.Nx
        elif ax == 1:
            dx = self.Ly/self.Ny

        f0E = np.roll(field.field[0], -1, axis=ax)
        f0W = np.roll(field.field[0],  1, axis=ax)

        f1E = np.roll(field.field[1], -1, axis=ax)
        f1W = np.roll(field.field[1],  1, axis=ax)

        f2E = np.roll(field.field[2], -1, axis=ax)
        f2W = np.roll(field.field[2],  1, axis=ax)

        # ghost cells
        if ax == 0:
            f0E[-1,:] = field.field[0][-1,:]
            f0W[0,:]  = field.field[0][0,:]

            f1E[-1,:] = field.field[1][-1,:]
            f1W[0,:]  = field.field[1][0,:]

            f2E[-1,:] = field.field[2][-1,:]
            f2W[0,:]  = field.field[2][0,:]
        elif ax == 1:
            f0E[:,-1] = field.field[0][:,-1]
            f0W[:,0]  = field.field[0][:,0]

            f1E[:,-1] = field.field[1][:,-1]
            f1W[:,0]  = field.field[1][:,0]

            f2E[:,-1] = field.field[2][:,-1]
            f2W[:,0]  = field.field[2][:,0]

        self.field[0] =  (f0E - 2 * field.field[0] + f0W)/(2.*dx)
        self.field[1] =  (f1E - 2 * field.field[1] + f1W)/(2.*dx)
        self.field[2] =  (f2E - 2 * field.field[2] + f2W)/(2.*dx)

        # boundary correction
        if ax == 0:
            self.field[0][0,:] *= 2.
            self.field[0][-1,:] *= 2.

            self.field[1][0,:] *= 2.
            self.field[1][-1,:] *= 2.

            self.field[2][0,:] *= 2.
            self.field[2][-1,:] *= 2.
        elif ax == 1:
            self.field[0][:,0] *= 2.
            self.field[0][:,-1] *= 2.

            self.field[1][:,0] *= 2.
            self.field[1][:,-1] *= 2.

            self.field[2][:,0] *= 2.
            self.field[2][:,-1] *= 2.

    def compute_RHS_1(self, stress, q):
        """
        Compute RHS using central difference scheme. Boundaries with fw/bw differences.
        Identical with np.gradient with edge_order=1.
        """

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        S0E = np.roll(stress.field[0], -1, axis=0)
        S0W = np.roll(stress.field[0],  1, axis=0)
        S0N = np.roll(stress.field[0], -1, axis=1)
        S0S = np.roll(stress.field[0],  1, axis=1)

        S1E = np.roll(stress.field[1], -1, axis=0)
        S1W = np.roll(stress.field[1],  1, axis=0)
        S1N = np.roll(stress.field[1], -1, axis=1)
        S1S = np.roll(stress.field[1],  1, axis=1)

        S5E = np.roll(stress.field[5], -1, axis=0)
        S5W = np.roll(stress.field[5],  1, axis=0)
        S5N = np.roll(stress.field[5], -1, axis=1)
        S5S = np.roll(stress.field[5],  1, axis=1)

        q0E = np.roll(q.field[0], -1, axis=0)
        q0W = np.roll(q.field[0],  1, axis=0)
        q0N = np.roll(q.field[0], -1, axis=1)
        q0S = np.roll(q.field[0],  1, axis=1)

        q1E = np.roll(q.field[1], -1, axis=0)
        q1W = np.roll(q.field[1],  1, axis=0)
        q1N = np.roll(q.field[1], -1, axis=1)
        q1S = np.roll(q.field[1],  1, axis=1)

        # ghost cells
        S0E[-1,:] = stress.field[0][-1,:]
        S0W[0,:]  = stress.field[0][0,:]
        S0N[:,-1] = stress.field[0][:,-1]
        S0S[:,0]  = stress.field[0][:,0]

        S1E[-1,:] = stress.field[1][-1,:]
        S1W[0,:]  = stress.field[1][0,:]
        S1N[:,-1] = stress.field[1][:,-1]
        S1S[:,0]  = stress.field[1][:,0]

        S5E[-1,:] = stress.field[5][-1,:]
        S5W[0,:]  = stress.field[5][0,:]
        S5N[:,-1] = stress.field[5][:,-1]
        S5S[:,0]  = stress.field[5][:,0]

        q0E[-1,:] = q.field[0][-1,:]
        q0W[0,:]  = q.field[0][0,:]
        q0N[:,-1] = q.field[0][:,-1]
        q0S[:,0]  = q.field[0][:,0]

        q1E[-1,:] = q.field[1][-1,:]
        q1W[0,:]  = q.field[1][0,:]
        q1N[:,-1] = q.field[1][:,-1]
        q1S[:,0]  = q.field[1][:,0]

        self.field[0] =  (S0E - S0W)/(2.*dx) + (S5N - S5S)/(2.*dy)
        self.field[1] =  (S5E - S5W)/(2.*dx) + (S1N - S1S)/(2.*dy)
        self.field[2] = -(q0E - q0W)/(2.*dx) - (q1N - q1S)/(2.*dy)

        # self.field[0][1:-1,1:-1] =  (S0E[1:-1,1:-1] - S0W[1:-1,1:-1])/(2.*dx) + (S5N[1:-1,1:-1] - S5S[1:-1,1:-1])/(2.*dy)
        # self.field[1][1:-1,1:-1] =  (S5E[1:-1,1:-1] - S5W[1:-1,1:-1])/(2.*dx) + (S1N[1:-1,1:-1] - S1S[1:-1,1:-1])/(2.*dy)
        # self.field[2][1:-1,1:-1] =  -(q0E[1:-1,1:-1] - q0W[1:-1,1:-1])/(2.*dx) - (q1N[1:-1,1:-1] - q1S[1:-1,1:-1])/(2.*dy)

        # self.field[0][1:-1,:] =  (S0E[1:-1,:] - S0W[1:-1,:])/(2.*dx) + (S5N[1:-1,:] - S5S[1:-1,:])/(2.*dy)
        # self.field[1][1:-1,:] =  (S5E[1:-1,:] - S5W[1:-1,:])/(2.*dx) + (S1N[1:-1,:] - S1S[1:-1,:])/(2.*dy)
        # self.field[2][1:-1,:] =  - (q0E[1:-1,:] - q0W[1:-1,:])/(2.*dx) - (q1N[1:-1,:] - q1S[1:-1,:])/(2.*dy)


        # boundary correction
        # self.field[0][0,:] *= 2.
        # self.field[0][-1,:] *= 2.
        # self.field[0][:,0] *= 2.
        # self.field[0][:,-1] *= 2.
        #
        # self.field[1][0,:] *= 2.
        # self.field[1][-1,:] *= 2.
        # self.field[1][:,0] *= 2.
        # self.field[1][:,-1] *= 2.
        #
        # self.field[2][0,:] *= 2.
        # self.field[2][-1,:] *= 2.
        # self.field[2][:,0] *= 2.
        # self.field[2][:,-1] *= 2.

    def compute_RHS_2(self, stress, q):
        """
        Compute RHS using central difference scheme. Boundaries with fw/bw differences.
        Identical with np.gradient with edge_order=1.
        """

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        S0E = np.roll(stress.field[0], -1, axis=0)
        S0W = np.roll(stress.field[0],  1, axis=0)
        S0N = np.roll(stress.field[0], -1, axis=1)
        S0S = np.roll(stress.field[0],  1, axis=1)

        S1E = np.roll(stress.field[1], -1, axis=0)
        S1W = np.roll(stress.field[1],  1, axis=0)
        S1N = np.roll(stress.field[1], -1, axis=1)
        S1S = np.roll(stress.field[1],  1, axis=1)

        S5E = np.roll(stress.field[5], -1, axis=0)
        S5W = np.roll(stress.field[5],  1, axis=0)
        S5N = np.roll(stress.field[5], -1, axis=1)
        S5S = np.roll(stress.field[5],  1, axis=1)

        q0E = np.roll(q.field[0], -1, axis=0)
        q0W = np.roll(q.field[0],  1, axis=0)
        q0N = np.roll(q.field[0], -1, axis=1)
        q0S = np.roll(q.field[0],  1, axis=1)

        q1E = np.roll(q.field[1], -1, axis=0)
        q1W = np.roll(q.field[1],  1, axis=0)
        q1N = np.roll(q.field[1], -1, axis=1)
        q1S = np.roll(q.field[1],  1, axis=1)

        # ghost cells
        S0E[-1,:] = stress.field[0][-1,:]
        S0W[0,:]  = stress.field[0][0,:]
        S0N[:,-1] = stress.field[0][:,-1]
        S0S[:,0]  = stress.field[0][:,0]

        S1E[-1,:] = stress.field[1][-1,:]
        S1W[0,:]  = stress.field[1][0,:]
        S1N[:,-1] = stress.field[1][:,-1]
        S1S[:,0]  = stress.field[1][:,0]

        S5E[-1,:] = stress.field[5][-1,:]
        S5W[0,:]  = stress.field[5][0,:]
        S5N[:,-1] = stress.field[5][:,-1]
        S5S[:,0]  = stress.field[5][:,0]

        q0E[-1,:] = q.field[0][-1,:]
        q0W[0,:]  = q.field[0][0,:]
        q0N[:,-1] = q.field[0][:,-1]
        q0S[:,0]  = q.field[0][:,0]

        q1E[-1,:] = q.field[1][-1,:]
        q1W[0,:]  = q.field[1][0,:]
        q1N[:,-1] = q.field[1][:,-1]
        q1S[:,0]  = q.field[1][:,0]

        self.field[0] =  (S0E - stress.field[0] + S0W)/(2.*dx*dx) + (S5N - stress.field[5] + S5S)/(2.*dy*dy)
        self.field[1] =  (S5E - stress.field[5] + S5W)/(2.*dx*dx) + (S1N - stress.field[1] + S1S)/(2.*dy*dy)
        self.field[2] = -(q0E - q.field[0] + q0W)/(2.*dx*dx) - (q1N - q.field[1] + q1S)/(2.*dy*dy)

        # boundary correction
        self.field[0][0,:] *= 2.
        self.field[0][-1,:] *= 2.
        self.field[0][:,0] *= 2.
        self.field[0][:,-1] *= 2.

        self.field[1][0,:] *= 2.
        self.field[1][-1,:] *= 2.
        self.field[1][:,0] *= 2.
        self.field[1][:,-1] *= 2.

        self.field[2][0,:] *= 2.
        self.field[2][-1,:] *= 2.
        self.field[2][:,0] *= 2.
        self.field[2][:,-1] *= 2.

    def compute_RHS_np(self, stress, q):

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        self.field[0] =  np.gradient(stress.field[0], dx, dy, edge_order=1)[0] + np.gradient(stress.field[5], dx, dy, edge_order=1)[1]
        self.field[1] =  np.gradient(stress.field[5], dx, dy, edge_order=1)[0] + np.gradient(stress.field[1], dx, dy, edge_order=1)[1]
        self.field[2] = -np.gradient(q.field[0], dx, dy, edge_order=1)[0] - np.gradient(q.field[1], dx, dy, edge_order=1)[1]

    def setDirichletX(self, value):
        self.field[0][0,:] = value
        self.field[0][-1,:] = value
        self.field[1][0,:] = value
        self.field[1][-1,:] = value = -self.field[1] - func(h, fluxY, rho)

    def setPeriodicX(self):
        self.field[0][-1,:] = self.field[0][0,:]
        self.field[1][-1,:] = self.field[1][0,:]

    def setPeriodicY(self):
        # self.field[0][:,0] = self.field[0][:,-1]
        # self.field[1][:,0] = self.field[1][:,-1]
        self.field[0][:,-1] = self.field[0][:,0]
        self.field[1][:,-1] = self.field[1][:,0]

    def update_LF(self, field, dt):

        old_0 = np.zeros_like(self.field[0])
        old_1 = np.zeros_like(self.field[1])
        old_2 = np.zeros_like(self.field[2])

        old_0[1:-1,:] = 0.25 *  (np.roll(self.field[0][1:-1,:],1,axis=0) + np.roll(self.field[0][1:-1,:],-1,axis=0)+\
                                    np.roll(self.field[0][1:-1,:],1,axis=1) + np.roll(self.field[0][1:-1,:],-1,axis=1))
        old_1[1:-1,:] = 0.25 *  (np.roll(self.field[1][1:-1,:],1,axis=0) + np.roll(self.field[1][1:-1,:],-1,axis=0)+\
                                    np.roll(self.field[1][1:-1,:],1,axis=1) + np.roll(self.field[1][1:-1,:],-1,axis=1))
        old_2[1:-1,:] = 0.25 *  (np.roll(self.field[2][1:-1,:],1,axis=0) + np.roll(self.field[2][1:-1,:],-1,axis=0)+\
                                    np.roll(self.field[2][1:-1,:],1,axis=1) + np.roll(self.field[2][1:-1,:],-1,axis=1))

        old_0[0,:] = 1./2. *  (self.field[0][1,:] + self.field[0][0,:])
        old_1[0,:] = 1./2. *  (self.field[1][1,:] + self.field[1][0,:])
        old_2[0,:] = 1./2. *  (self.field[2][1,:] + self.field[2][0,:])

        old_0[-1,:] = 1./2. *  (self.field[0][-2,:] + self.field[0][-1,:])
        old_1[-1,:] = 1./2. *  (self.field[1][-2,:] + self.field[1][-1,:])
        old_2[-1,:] = 1./2. *  (self.field[2][-2,:] + self.field[2][-1,:])

        # self.field[0][1:-1,1:-1] = dt * field.field[0][1:-1,1:-1] + old_0[1:-1,1:-1]
        # self.field[1][1:-1,1:-1] = dt * field.field[1][1:-1,1:-1] + old_1[1:-1,1:-1]
        # self.field[2][1:-1,1:-1] = dt * field.field[2][1:-1,1:-1] + old_2[1:-1,1:-1]

        # self.field[0][1:-1,:] = dt * field.field[0][1:-1,:] + old_0[1:-1,:]
        # self.field[1][1:-1,:] = dt * field.field[1][1:-1,:] + old_1[1:-1,:]
        # self.field[2][1:-1,:] = dt * field.field[2][1:-1,:] + old_2[1:-1,:]

        self.field[0] = dt * field.field[0] + old_0
        self.field[1] = dt * field.field[1] + old_1
        self.field[2] = dt * field.field[2] + old_2

    def update_LF_full(self, field, dt):

        old_0 = np.zeros_like(self.field[0])
        old_1 = np.zeros_like(self.field[1])
        old_2 = np.zeros_like(self.field[2])

        old_0 = 0.25 *  (np.roll(self.field[0],1,axis=0) + np.roll(self.field[0],-1,axis=0)+\
                        np.roll(self.field[0],1,axis=1) + np.roll(self.field[0],-1,axis=1))
        old_1 = 0.25 *  (np.roll(self.field[1],1,axis=0) + np.roll(self.field[1],-1,axis=0)+\
                        np.roll(self.field[1],1,axis=1) + np.roll(self.field[1],-1,axis=1))
        old_2 = 0.25 *  (np.roll(self.field[2],1,axis=0) + np.roll(self.field[2],-1,axis=0)+\
                        np.roll(self.field[2],1,axis=1) + np.roll(self.field[2],-1,axis=1))

        # V1: left and right boundaries FTCS, rest LF
        old_0[0,:] = self.field[0][0,:]
        old_1[0,:] = self.field[1][0,:]
        old_2[0,:] = self.field[2][0,:]

        old_0[-1,:] = self.field[0][-1,:]
        old_1[-1,:] = self.field[1][-1,:]
        old_2[-1,:] = self.field[2][-1,:]

        self.field[0] = dt * field.field[0] + old_0
        self.field[1] = dt * field.field[1] + old_1
        self.field[2] = dt * field.field[2] + old_2


        # V2: left FTCS, inside LF, right: not yet considered
        # old_0[0,:] = self.field[0][0,:]
        # old_1[0,:] = self.field[1][0,:]
        # old_2[0,:] = self.field[2][0,:]
        #
        # self.field[0][:-1,:] = dt * field.field[0][:-1,:] + old_0[:-1,:]
        # self.field[1][:-1,:] = dt * field.field[1][:-1,:] + old_1[:-1,:]
        # self.field[2][:-1,:] = dt * field.field[2][:-1,:] + old_2[:-1,:]


        # self.field[0][1:-1,1:-1] = dt * field.field[0][1:-1,1:-1] + old_0[1:-1,1:-1]
        # self.field[1][1:-1,1:-1] = dt * field.field[1][1:-1,1:-1] + old_1[1:-1,1:-1]
        # self.field[2][1:-1,1:-1] = dt * field.field[2][1:-1,1:-1] + old_2[1:-1,1:-1]

        # self.field[0][1:-1,:] = dt * field.field[0][1:-1,:] + old_0[1:-1,:]
        # self.field[1][1:-1,:] = dt * field.field[1][1:-1,:] + old_1[1:-1,:]
        # self.field[2][1:-1,:] = dt * field.field[2][1:-1,:] + old_2[1:-1,:]

        # self.field[0][0,:] = dt * field.field[0][0,:] + self.field[0][0,:]
        # self.field[1][0,:] = dt * field.field[1][0,:] + self.field[1][0,:]
        # self.field[2][0,:] = dt * field.field[2][0,:] + self.field[2][0,:]

        # self.field[0][-1,:] = dt * field.field[0][-1,:] + self.field[0][-1,:]
        # self.field[1][-1,:] = dt * field.field[1][-1,:] + self.field[1][-1,:]
        # self.field[2][-1,:] = dt * field.field[2][-1,:] + self.field[2][-1,:]

    def updateLF_new(self, field, dt):

        old_0 = -self.field[0]
        old_1 = -self.field[1]
        old_2 = -self.field[2]

        old_0 += 0.5 *  (np.roll(self.field[0],1,axis=0) + np.roll(self.field[0],-1,axis=0)+\
                        np.roll(self.field[0],1,axis=1) + np.roll(self.field[0],-1,axis=1))
        old_1 += 0.5 *  (np.roll(self.field[1],1,axis=0) + np.roll(self.field[1],-1,axis=0)+\
                        np.roll(self.field[1],1,axis=1) + np.roll(self.field[1],-1,axis=1))
        old_2 += 0.5 *  (np.roll(self.field[2],1,axis=0) + np.roll(self.field[2],-1,axis=0)+\
                        np.roll(self.field[2],1,axis=1) + np.roll(self.field[2],-1,axis=1))

        old_0[0,:] = 0.5 * ( - self.field[0][0,:] + np.roll(self.field[0][0,:],-1,axis=0)+\
                            np.roll(self.field[0][0,:],1,axis=0) + self.field[0][1,:])
        old_1[0,:] = 0.5 * ( - self.field[1][0,:] + np.roll(self.field[1][0,:],-1,axis=0)+\
                            np.roll(self.field[1][0,:],1,axis=0) + self.field[1][1,:])
        old_2[0,:] = 0.5 * ( - self.field[2][0,:] + np.roll(self.field[2][0,:],-1,axis=0)+\
                            np.roll(self.field[2][0,:],1,axis=0) + self.field[2][1,:])

        old_0[-1,:] = 0.5 * ( - self.field[0][-1,:] + np.roll(self.field[0][-1,:],-1,axis=0)+\
                            np.roll(self.field[0][-1,:],1,axis=0) + self.field[0][-2,:])
        old_1[-1,:] = 0.5 * ( - self.field[1][-1,:] + np.roll(self.field[1][-1,:],-1,axis=0)+\
                            np.roll(self.field[1][-1,:],1,axis=0) + self.field[1][-2,:])
        old_2[-1,:] = 0.5 * ( - self.field[2][-1,:] + np.roll(self.field[-1][0,:],-1,axis=0)+\
                            np.roll(self.field[2][-1,:],1,axis=0) + self.field[2][-2,:])



        # old_0[0,:] = self.field[0][0,:]
        # old_1[0,:] = self.field[1][0,:]
        # old_2[0,:] = self.field[2][0,:]

        # self.field[0][:-1,:] = dt * field.field[0][:-1,:] + old_0[:-1,:]
        # self.field[1][:-1,:] = dt * field.field[1][:-1,:] + old_1[:-1,:]
        # self.field[2][:-1,:] = dt * field.field[2][:-1,:] + old_2[:-1,:]

        #
        # old_0[-1,:] = self.field[0][-1,:]
        # old_1[-1,:] = self.field[1][-1,:]
        # old_2[-1,:] = self.field[2][-1,:]

        self.field[0] = dt * field.field[0] + old_0
        self.field[1] = dt * field.field[1] + old_1
        self.field[2] = dt * field.field[2] + old_2

        # self.field[0][1:-1,:] = dt * field.field[0][1:-1,:] + old_0[1:-1,:]
        # self.field[1][1:-1,:] = dt * field.field[1][1:-1,:] + old_1[1:-1,:]
        # self.field[2][1:-1,:] = dt * field.field[2][1:-1,:] + old_2[1:-1,:]

    def updateFlux(self, field, dt):
        self.field[0] = dt * field.field[0] + self.field[0]
        self.field[1] = dt * field.field[1] + self.field[1]
        self.field[2] = dt * field.field[2] + self.field[2]


    def updateFlux_LW(self, field1, field2, dt):
        self.field[0] = self.field[0] + dt * field1.field[0] + dt**2 * field2.field[0]
        self.field[1] = self.field[1] + dt * field1.field[1] + dt**2 * field2.field[1]
        self.field[2] = self.field[2] + dt * field1.field[2] + dt**2 * field2.field[2]

        # self.field[0] = self.field[0] + dt * field1.field[0]
        # self.field[1] = self.field[1] + dt * field1.field[1]
        # self.field[2] = self.field[2] + dt * field1.field[2]

    def update_final(self, f1, f2, f3, f4, dt):
        self.field[0] = self.field[0] + dt * (f1.field[0] + f2.field[0]) + dt**2 * (f3.field[0] + f4.field[0])
        self.field[1] = self.field[1] + dt * (f1.field[1] + f2.field[1]) + dt**2 * (f3.field[1] + f4.field[1])
        self.field[2] = self.field[2] + dt * (f1.field[2] + f2.field[2]) + dt**2 * (f3.field[2] + f4.field[2])

    def getPgrad_np(self, press):

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny

        self.field[0] = np.gradient(press.field[0],dx,dy, edge_order=1)[0]
        self.field[1] = np.gradient(press.field[0],dx,dy, edge_order=1)[1]

    def getStressDiv_avg(self, height, rho, fluxX, fluxY, mu, lam, sx, sy, U, V):
        self.field[0] = -self.field[0] + (8*(mu + lam/2.)*(U*rho - 1.5*fluxX)*sx**2 + 6*mu*(U*rho - 2.*fluxX))/(rho*height**2)
        self.field[1] = -self.field[1] + (4*(V*rho - 1.5*fluxX)*mu*sx**2 + 6*mu*(V*rho - 2.*fluxY))/(rho*height**2)

    def addStress_wall(self, height, rho, fluxX, fluxY, mu, U, V):
        self.field[0] -= 6.*mu*(U*rho - 2.*fluxX)/(rho*height**2)
        self.field[1] -= 6.*mu*(V*rho - 2.*fluxY)/(rho*height**2)

    def fluxX_LF(self, stress, q, dt, d):
        self.field[0] = 0.5 * (-stress.field[0] - np.roll(stress.field[0], d, axis = 0)) - self.dx/(2*dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 0))
        self.field[1] = 0.5 * (-stress.field[5] - np.roll(stress.field[5], d, axis = 0)) - self.dx/(2*dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 0))
        self.field[2] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = 0)) - self.dx/(2*dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 0))

        if d == -1:
            self.field[0][-1,:] = self.field[0][-2,:]
            self.field[1][-1,:] = self.field[1][-2,:]
            self.field[2][-1,:] = self.field[2][-2,:]
        elif d == 1:
            self.field[0][0,:] = self.field[0][1,:]
            self.field[1][0,:] = self.field[1][1,:]
            self.field[2][0,:] = self.field[2][1,:]


    def fluxY_LF(self, stress, q, dt, d):
        self.field[0] = 0.5 * (-stress.field[5] - np.roll(stress.field[5], d, axis = 1)) - self.dy/(2*dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 1))
        self.field[1] = 0.5 * (-stress.field[1] - np.roll(stress.field[1], d, axis = 1)) - self.dy/(2*dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 1))
        self.field[2] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = 1)) - self.dy/(2*dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 1))


    def computeRHS(self, stress, q, dt):
        self.field[0] =   1./self.dx * (self.fluxX_LF(stress, q, dt, -1)[0] - self.fluxX_LF(stress, q, dt, 1)[0])\
                        + 1./self.dy * (self.fluxY_LF(stress, q, dt, -1)[0] - self.fluxY_LF(stress, q, dt, 1)[0])

        self.field[1] =   1./self.dx * (self.fluxX_LF(stress, q, dt, -1)[1] - self.fluxX_LF(stress, q, dt, 1)[1])\
                        + 1./self.dy * (self.fluxY_LF(stress, q, dt, -1)[1] - self.fluxY_LF(stress, q, dt, 1)[1])

        self.field[2] =   1./self.dx * (self.fluxX_LF(stress, q, dt, -1)[2] - self.fluxX_LF(stress, q, dt, 1)[2])\
                        + 1./self.dy * (self.fluxY_LF(stress, q, dt, -1)[2] - self.fluxY_LF(stress, q, dt, 1)[2])

    def compRHS(self, fXE, fXW, fYN, fYS):
        self.field[0] = 1./self.dx * (fXE.field[0] - fXW.field[0]) + 1./self.dy * (fYN.field[0] - fYS.field[0])
        self.field[1] = 1./self.dx * (fXE.field[1] - fXW.field[1]) + 1./self.dy * (fYN.field[1] - fYS.field[1])
        self.field[2] = 1./self.dx * (fXE.field[2] - fXW.field[2]) + 1./self.dy * (fYN.field[2] - fYS.field[2])

    def updateLF(self, rhs, dt):
        self.field[0] = self.field[0] - dt * rhs.field[0]
        self.field[1] = self.field[1] - dt * rhs.field[1]
        self.field[2] = self.field[2] - dt * rhs.field[2]

class tensorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 6
        super().__init__(Nx, Ny, Lx, Ly)

    def getStressNewtonian(self, press, height, rho, fluxX, fluxY, mu, lam, sx, sy, U, V):
        self.field[0] = -press - (2.*(U*rho-2.*fluxX)*(mu+lam/2.)*sx + (V*rho-2.*fluxY)*lam*sy)/(rho*height)
        self.field[1] = -press - (2.*(V*rho-2.*fluxY)*(mu+lam/2.)*sy + (U*rho-2.*fluxX)*lam*sx)/(rho*height)
        self.field[2] = -press - ((V*rho-2.*fluxY)*lam*sy + (U*rho-2.*fluxX)*lam*sx)/(rho*height)
        self.field[3] = np.zeros_like(self.field[0])
        self.field[4] = np.zeros_like(self.field[0])
        self.field[5] = -mu*(sx * (V*rho-2.*fluxY) + sy * (U*rho-2.*fluxX))/(rho*height)

    def getStressNewtonian_avg(self, press, height, rho, fluxX, fluxY, mu, lam, sx, sy, U, V):
        self.field[0] = -press - (4.*(U*rho-1.5*fluxX)*(mu+lam/2.)*sx + 2.*lam*(V*rho-1.5*fluxY)*sy)/(rho*height)
        self.field[1] = -press - (4.*(V*rho-1.5*fluxY)*(mu+lam/2.)*sy + 2.*lam*(U*rho-1.5*fluxX)*sx)/(rho*height)
        self.field[2] = -press - (2.*(U*rho-1.5*fluxX)*lam*sx + 2.*(V*rho-1.5*fluxY)*lam*sx)/(rho*height)
        self.field[3] = mu*V/height
        self.field[4] = mu*U/height
        self.field[5] = -2.*mu*(sx * (V*rho-1.5*fluxY) + sy * (U*rho-1.5*fluxX))/(rho*height)




    def addNoise(self, frac):
        for i in range(self.ndim):
            mean = self.field[i]
            mu = frac * np.amax(abs(self.field[i]))
            self.field[i] += np.random.normal(mean, mu)
