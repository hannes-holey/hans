#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson

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

    def addStress_wall(self, q, h, mu, U, V):
        self.field[0] -= 6.*mu*(U*q.field[2] - 2.*q.field[0])/(q.field[2]*h.field[0]**2)
        self.field[1] -= 6.*mu*(V*q.field[2] - 2.*q.field[1])/(q.field[2]*h.field[0]**2)

    def updateExplicit(self, rhs, dt):
        self.field[0] = self.field[0] - dt * rhs.field[0]
        self.field[1] = self.field[1] - dt * rhs.field[1]
        self.field[2] = self.field[2] - dt * rhs.field[2]

class TensorField(Field):

    def __init__(self, disc):
        self.ndim = 6
        super().__init__(disc)

    # def getStressNewtonian(self, h, q, mu, lam, U, V, rho0, P0):
    #
    #     self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)
    #
    #     self.field[0] = -self.field[0] - (2.*(U*q.field[2]-2.*q.field[0])*(mu+lam/2.)*h.field[1] + (V*q.field[2]-2.*q.field[1])*lam*h.field[2])/(q.field[2]*h.field[0])
    #     self.field[1] = -self.field[1] - (2.*(V*q.field[2]-2.*q.field[1])*(mu+lam/2.)*h.field[2] + (U*q.field[2]-2.*q.field[0])*lam*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[2] = -self.field[2] - ((V*q.field[2]-2.*q.field[1])*lam*h.field[2] + (U*q.field[2]-2.*q.field[0])*lam*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[3] = np.zeros_like(self.field[0])
    #     self.field[4] = np.zeros_like(self.field[0])
    #     self.field[5] = -mu*(h.field[1] * (V*q.field[2]-2.*q.field[1]) + h.field[2] * (U*q.field[2]-2.*q.field[0]))/(q.field[2]*h.field[0])
    #
    # def getStressNewtonian_avg(self, h, q, mu, lam, U, V, rho0, P0):
    #
    #     self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)
    #
    #     self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
    #     self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[0])*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[3] = mu*V/h.field[0]
    #     self.field[4] = mu*U/h.field[0]
    #     self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])
    #
    # def getStressNewtonian_avg2(self, h, q, mu, lam, U, V, rho0, P0):
    #
    #     self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2],3)
    #
    #     self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
    #     self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[0])*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[3] = -4*(((0.5*U*h.field[1]**2 + 0.25*U*h.field[2]**2 + 0.25*V*h.field[1]*h.field[2] - 2.*U)*mu + lam*(U*h.field[1]**2 + V*h.field[1]*h.field[2] - U))*q.field[2] - \
    #                     1.5*lam*h.field[1]*(q.field[0]*h.field[1] + q.field[1]*h.field[2]))*mu/(q.field[2]*(-mu*h.field[1]**2 - mu*h.field[2]**2 + 4*lam + 8*mu)*h.field[0])
    #     self.field[4] = -4*(((0.5*V*h.field[2]**2 + 0.25*V*h.field[1]**2 + 0.25*U*h.field[1]*h.field[2] - 2.*V)*mu + lam*(V*h.field[2]**2 + U*h.field[1]*h.field[2] - V))*q.field[2] - \
    #                     1.5*lam*h.field[2]*(q.field[0]*h.field[1] + q.field[1]*h.field[2]))*mu/(q.field[2]*(-mu*h.field[1]**2 - mu*h.field[2]**2 + 4*lam + 8*mu)*h.field[0])
    #     self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])
    #
    # def getStressNewtonian_avg4(self, h, q, mu, lam, U, V, rho0, P0):
    #
    #     self.fromFunctionField(DowsonHigginson(rho0, P0).isoT_pressure, q.field[2], 3)
    #
    #     self.field[0] = -self.field[0] - (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
    #     self.field[1] = -self.field[1] - (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[1])*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[2] = -self.field[2] - (2.*(U*q.field[2]-1.5*q.field[0])*lam*h.field[1] + 2.*(V*q.field[2]-1.5*q.field[1])*lam*h.field[1])/(q.field[2]*h.field[0])
    #     self.field[3] = ((4*U*(lam + mu)*q.field[2] - 6*lam*q.field[0])*h.field[1]**2 + 4*((lam + mu/4)*V*q.field[2] - (3*lam*q.field[1])/2)*h.field[1]*h.field[2] + 3*U*mu*q.field[2]*h.field[2]**2)/(3*q.field[2]*(h.field[1]**2 + h.field[2]**2)*h.field[0])
    #     self.field[4] = ((4*V*(lam + mu)*q.field[2] - 6*lam*q.field[1])*h.field[2]**2 + 4*((lam + mu/4)*U*q.field[2] - (3*lam*q.field[0])/2)*h.field[1]*h.field[2] + 3*V*mu*q.field[2]*h.field[1]**2)/(3*q.field[2]*(h.field[1]**2 + h.field[2]**2)*h.field[0])
    #     self.field[5] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])
    #
    # def addNoise(self, frac):
    #     for i in range(self.ndim):
    #         mean = self.field[i]
    #         mu = frac * np.amax(abs(self.field[i]))
    #         self.field[i] += np.random.normal(mean, mu)
