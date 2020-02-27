#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson

class Field:

    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Ly
        self.Ly = Ly

        self.dx = self.Lx/(self.Nx)
        self.dy = self.Ly/(self.Ny)

        x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.Nx)
        y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.Ny)
        xx, yy = np.meshgrid(x,y)

        self.xx = xx.T
        self.yy = yy.T

        self.field_buffer = np.zeros((self.ndim, self.Nx, self.Ny))

        self.field = []

        for k in range(self.ndim):
            self.field.append(np.ndarray(shape=(Nx,Ny), dtype=np.float64, buffer=self.field_buffer[k]))

    def fromFunctionXY(self, func, comp):
        self.field[comp]=func(self.xx, self.yy)

    def fromFunctionField(self, func, arg, dim):

        for i in range(dim):
            self.field[i] = func(arg)

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

    def eastArray(self):
        field = self.Field(self.Nx, self.Ny, self.LX, self.Ny)
        for i in range(self.ndim):
            field.field[i] = 0.5 (self.field[i] + np.roll(self.field[i], -1, axis=0))

        return field

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

class vectorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 3
        super().__init__(Nx, Ny, Lx, Ly)


    def getGradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"
        self.field[1] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[0]
        self.field[2] = np.gradient(self.field[0], self.dx, self.dy, edge_order = 2)[1]

    # def addShearStress_wall(self, height, q, mu, lam, U, V, sx, sy):
    #     self.field[0] -=    -8.*mu*(((U*q.field[2] - 1.5*q.field[0])*sx**2 + q.field[2]*V*sx*sy/4. + 3.*(sy**2 - 8.)*(U*q.field[2] - 2.*q.field[0])/4.)*mu\
    #                         + ((U*q.field[2] - 1.5*q.field[0])*sx**2 + sx*sy*(V*q.field[2] - 1.5*q.field[1]) - 3.*U*q.field[2] + 6.*q.field[0])*lam)\
    #                         /(q.field[2]*(-mu*sx**2 - mu*sy**2 + 4.*lam + 8.*mu)*height.field[0]**2)
    #     self.field[1] -=    -8.*mu*(((V*q.field[2] - 1.5*q.field[1])*sy**2 + q.field[2]*U*sx*sy/4. + 3.*(sx**2 - 8.)*(V*q.field[2] - 2.*q.field[1])/4.)*mu\
    #                         + ((V*q.field[2] - 1.5*q.field[1])*sy**2 + sx*sy*(U*q.field[2] - 1.5*q.field[0]) - 3.*V*q.field[2] + 6.*q.field[1])*lam)\
    #                         /(q.field[2]*(-mu*sx**2 - mu*sy**2 + 4.*lam + 8.*mu)*height.field[0]**2)

    def QX_LF(self, stress, q, dt, d, periodic):
        self.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = 0)) + dt/(2. * self.dx) * d * (stress.field[0] - np.roll(stress.field[0], d, axis = 0))
        self.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = 0)) + dt/(2. * self.dx) * d * (stress.field[5] - np.roll(stress.field[5], d, axis = 0))
        self.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = 0)) - dt/(2. * self.dx) * d * (q.field[0] - np.roll(q.field[0], d, axis = 0))


    def QY_LF(self, stress, q, dt, d, periodic):
        self.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis = 1)) + dt/(2. * self.dy) * d * (stress.field[5] - np.roll(stress.field[5], d, axis = 1))
        self.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis = 1)) + dt/(2. * self.dy) * d * (stress.field[1] - np.roll(stress.field[1], d, axis = 1))
        self.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis = 1)) - dt/(2. * self.dy) * d * (q.field[1] - np.roll(q.field[1], d, axis = 1))


    def fluxX_LF(self, stress, q, dt, d, periodic):
        self.field[0] = -0.5 * (stress.field[0] + np.roll(stress.field[0], d, axis = 0)) - self.dx/(2. * dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 0))
        self.field[1] = -0.5 * (stress.field[5] + np.roll(stress.field[5], d, axis = 0)) - self.dx/(2. * dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 0))
        self.field[2] =  0.5 * (q.field[0]      + np.roll(q.field[0],      d, axis = 0)) - self.dx/(2. * dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 0))

        if periodic == 0:
            if d == -1:
                self.field[0][-1,:] = -stress.field[0][-1,:]
                self.field[1][-1,:] = -stress.field[5][-1,:]
                # self.field[2][-1,:] = q.field[0][-1,:]
            elif d == 1:
                self.field[0][0,:] = -stress.field[0][0,:]
                self.field[1][0,:] = -stress.field[5][0,:]
                # self.field[2][0,:] = q.field[0][0,:]

    def fluxY_LF(self, stress, q, dt, d, periodic):
        self.field[0] = -0.5 * (stress.field[5] + np.roll(stress.field[5], d, axis = 1)) - self.dy/(2. * dt) * d * (q.field[0] - np.roll(q.field[0], d, axis = 1))
        self.field[1] = -0.5 * (stress.field[1] + np.roll(stress.field[1], d, axis = 1)) - self.dy/(2. * dt) * d * (q.field[1] - np.roll(q.field[1], d, axis = 1))
        self.field[2] =  0.5 * (q.field[1]      + np.roll(q.field[1],      d, axis = 1)) - self.dy/(2. * dt) * d * (q.field[2] - np.roll(q.field[2], d, axis = 1))

        if periodic == 0:
            if d == -1:
                self.field[0][-1,:] = -stress.field[5][-1,:]
                self.field[1][-1,:] = -stress.field[1][-1,:]
                self.field[2][-1,:] = q.field[1][-1,:]

            elif d == 1:
                self.field[0][0,:] = -stress.field[5][0,:]
                self.field[1][0,:] = -stress.field[1][0,:]
                self.field[2][0,:] = q.field[1][0,:]

    def computeRHS(self, fXE, fXW, fYN, fYS):
        self.field[0] = 1./self.dx * (fXE.field[0] - fXW.field[0]) + 1./self.dy * (fYN.field[0] - fYS.field[0])
        self.field[1] = 1./self.dx * (fXE.field[1] - fXW.field[1]) + 1./self.dy * (fYN.field[1] - fYS.field[1])
        self.field[2] = 1./self.dx * (fXE.field[2] - fXW.field[2]) + 1./self.dy * (fYN.field[2] - fYS.field[2])

    def computeRHS_LW(self, fXE, fXW, fYN, fYS, Q_E, Q_W, Q_N, Q_S):
        self.field[0] = - 1./self.dx * (fXE.field[0] - fXW.field[0]) - 1./self.dy * (fYN.field[5] - fYS.field[5])
        self.field[1] = - 1./self.dx * (fXE.field[5] - fXW.field[5]) - 1./self.dy * (fYN.field[5] - fYS.field[5])
        self.field[2] =   1./self.dx * (Q_E.field[0] - Q_W.field[0]) + 1./self.dy * (Q_N.field[1] - Q_S.field[1])

    def addStress_wall(self, height, q, mu, U, V):
        self.field[0] -= 6.*mu*(U*q.field[2] - 2.*q.field[0])/(q.field[2]*height.field[0]**2)
        self.field[1] -= 6.*mu*(V*q.field[2] - 2.*q.field[1])/(q.field[2]*height.field[0]**2)

    def update_explicit(self, rhs, dt):
        self.field[0] = self.field[0] - dt * rhs.field[0]
        self.field[1] = self.field[1] - dt * rhs.field[1]
        self.field[2] = self.field[2] - dt * rhs.field[2]

class tensorField(Field):

    def __init__(self, Nx, Ny, Lx, Ly):
        self.ndim = 6
        super().__init__(Nx, Ny, Lx, Ly)

    def getStressNewtonian(self, h, q, mu, lam, U, V, rho0, P0):

        self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)

        self.field[0] = -self.field[0] - (2.*(U*q.field[2]-2.*q.field[0])*(mu+lam/2.)*h.field[1] + (V*q.field[2]-2.*q.field[1])*lam*h.field[2])/(q.field[2]*h.field[0])
        self.field[1] = -self.field[1] - (2.*(V*q.field[2]-2.*q.field[1])*(mu+lam/2.)*h.field[2] + (U*q.field[2]-2.*q.field[0])*lam*h.field[1])/(q.field[2]*h.field[0])
        self.field[2] = -self.field[2] - ((V*q.field[2]-2.*q.field[1])*lam*h.field[2] + (U*q.field[2]-2.*q.field[0])*lam*h.field[1])/(q.field[2]*h.field[0])
        self.field[3] = np.zeros_like(self.field[0])
        self.field[4] = np.zeros_like(self.field[0])
        self.field[5] = -mu*(h.field[1] * (V*q.field[2]-2.*q.field[1]) + h.field[2] * (U*q.field[2]-2.*q.field[0]))/(q.field[2]*h.field[0])

    def getStressNewtonian_avg(self, h, q, mu, lam, U, V, rho0, P0):

        self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)

        self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
        self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[0])*h.field[1])/(q.field[2]*h.field[0])
        self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
        self.field[3] = mu*V/h.field[0]
        self.field[4] = mu*U/h.field[0]
        self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])

    def getStressNewtonian_avg2(self, h, q, mu, lam, U, V, rho0, P0):

        self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)

        self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
        self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[0])*h.field[1])/(q.field[2]*h.field[0])
        self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
        self.field[3] = -4*(((0.5*U*h.field[1]**2 + 0.25*U*h.field[2]**2 + 0.25*V*h.field[1]*h.field[2] - 2.*U)*mu + lam*(U*h.field[1]**2 + V*h.field[1]*h.field[2] - U))*q.field[2] - \
                        1.5*lam*h.field[1]*(q.field[0]*h.field[1] + q.field[1]*h.field[2]))*mu/(q.field[2]*(-mu*h.field[1]**2 - mu*h.field[2]**2 + 4*lam + 8*mu)*h.field[0])
        self.field[4] = -4*(((0.5*V*h.field[2]**2 + 0.25*V*h.field[1]**2 + 0.25*U*h.field[1]*h.field[2] - 2.*V)*mu + lam*(V*h.field[2]**2 + U*h.field[1]*h.field[2] - V))*q.field[2] - \
                        1.5*lam*h.field[2]*(q.field[0]*h.field[1] + q.field[1]*h.field[2]))*mu/(q.field[2]*(-mu*h.field[1]**2 - mu*h.field[2]**2 + 4*lam + 8*mu)*h.field[0])
        self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])

    def getStressNewtonian_avg4(self, h, q, mu, lam, U, V, rho0, P0):

        self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2], 3)

        self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
        self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[1])*h.field[1])/(q.field[2]*h.field[0])
        self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
        self.field[3] = ((4*U*(lam + mu)*q.field[2] - 6*lam*q.field[0])*h.field[1]**2 + 4*((lam + mu/4)*V*q.field[2] - (3*lam*q.field[1])/2)*h.field[1]*h.field[2] + 3*U*mu*q.field[2]*h.field[2]**2)/(3*q.field[2]*(h.field[1]**2 + h.field[2]**2)*h.field[0])
        self.field[4] = ((4*V*(lam + mu)*q.field[2] - 6*lam*q.field[1])*h.field[2]**2 + 4*((lam + mu/4)*U*q.field[2] - (3*lam*q.field[0])/2)*h.field[1]*h.field[2] + 3*V*mu*q.field[2]*h.field[1]**2)/(3*q.field[2]*(h.field[1]**2 + h.field[2]**2)*h.field[0])
        self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])

    def getStressNewtonian_Rey(self, h, q, mu, lam, U, V, rho0, P0):

        self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2], 2)

        self.field[0] = -self.field[0]
        self.field[1] = -self.field[1]


    def addNoise(self, frac):
        for i in range(self.ndim):
            mean = self.field[i]
            mu = frac * np.amax(abs(self.field[i]))
            self.field[i] += np.random.normal(mean, mu)
