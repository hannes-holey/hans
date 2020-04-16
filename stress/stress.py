#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson, PowerLaw
from field.field import ScalarField, VectorField, TensorField

class Newtonian:

    def __init__(self, disc, geometry, material):

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def viscousStress_avg(self, q, h, dt):

        out = VectorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        mu = float(self.mat['mu'])
        T = float(self.mat['T0'])

        if bool(self.mat['Stokes']) == True:
            lam = -2/3 * mu
        else:
            lam = self.mat['lambda']

        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        dz = np.amin(h0)

        if bool(self.mat['Rey']) == False:
            # origin center
            # out.field[0] =  (-3*(lam + 2*mu)*(U*rho - 2*j_x)*hx - 3*lam*(V*rho - 2*j_y)*hy)/(2*h0*rho)
            # out.field[1] =  (-3*(lam + 2*mu)*(V*rho - 2*j_y)*hy - 3*lam*(U*rho - 2*j_x)*hx)/(2*h0*rho)
            # out.field[2] = -3*((V*rho - 2*j_y)*hx + hy*(U*rho - 2*j_x))*mu/(2*h0*rho)

            # origin bottom
            out.field[0] = (-4*(U*rho - (3*j_x)/2)*(mu + lam/2)*hx - 2*lam*(V*rho - (3*j_y)/2)*hy)/(h0*rho)
            out.field[1] = (-4*(V*rho - (3*j_y)/2)*(mu + lam/2)*hy - 2*lam*(U*rho - (3*j_x)/2)*hx)/(h0*rho)
            out.field[2] = -2*mu*((V*rho - (3*j_y)/2)*hx + hy*(U*rho - (3*j_x)/2))/(h0*rho)

        cov = self.getCovariance(out.ndim, mu, lam, T, q.dx, q.dy, dz, dt)

        return out, cov

    def stress_avg(self, q, h, dt):

        if self.mat['EOS'] == 'DH':
            eqOfState = DowsonHigginson(self.mat)
        elif self.mat['EOS'] == 'PL':
            eqOfState = PowerLaw(self.mat)

        out, cov = self.viscousStress_avg(q, h, dt)

        out.field[0] -= eqOfState.isoT_pressure(q.field[2])
        out.field[1] -= eqOfState.isoT_pressure(q.field[2])

        return out, cov

    def viscousStress_wall(self, q, h, dt, bound):

        out = TensorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        mu = float(self.mat['mu'])
        T = float(self.mat['T0'])

        if bool(self.mat['Stokes']) == True:
            lam = -2/3 * mu
        else:
            lam = self.mat['lambda']

        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        dz = np.amin(h0)

        if bound == 1:
            if bool(self.mat['Rey']) == False:
                out.field[0] = (-8*(mu + lam/2)*(U*rho - (3*j_x)/2)*hx - 4*hy*lam*(V*rho - (3*j_y)/2))/(h0*rho)
                out.field[1] = (-8*(mu + lam/2)*(V*rho - (3*j_y)/2)*hy - 4*hx*lam*(U*rho - (3*j_x)/2))/(h0*rho)
                out.field[2] = -4*lam*((U*rho - (3*j_x)/2)*hx + hy*(V*rho - (3*j_y)/2))/(h0*rho)
                out.field[3] = 2*mu*(2*V*rho - 3*j_y)/(h0*rho)
                out.field[4] = 2*mu*(2*U*rho - 3*j_x)/(h0*rho)
                out.field[5] = -4*((V*rho - (3*j_y)/2)*hx + hy*(U*rho - (3*j_x)/2))*mu/(h0*rho)
            else:
                out.field[3] = 2*mu*(2*V*rho - 3*j_y)/(h0*rho)
                out.field[4] = 2*mu*(2*U*rho - 3*j_x)/(h0*rho)
        elif bound == 0:
            out.field[3] = -2*mu*(V*rho - 3*j_y)/(h0*rho)
            out.field[4] = -2*mu*(U*rho - 3*j_x)/(h0*rho)

        if bool(self.mat['Fluctuating']) == True:
            cov = self.getCovariance(out.ndim, mu, lam, T, q.dx, q.dy, dz, dt)
            out.addNoise_FH(cov)

        return out

    def getCovariance(self, ndim, mu, lam, T, dx, dy, dz, dt):
        if ndim == 3:
            cov = np.array([[mu * (2 + lam), lam, 0.],
                            [lam, mu * (2 + lam), 0.],
                            [0., 0., mu]])
        elif ndim == 6:
            cov = np.array([[mu * (2 + lam), lam, lam, 0., 0., 0.],
                            [lam, mu * (2 + lam), lam, 0., 0., 0.],
                            [lam, lam, mu * (2 + lam), 0., 0., 0.],
                            [0., 0., 0., mu, 0., 0.],
                            [0., 0., 0., 0., mu, 0.],
                            [0., 0., 0., 0., 0., mu]])

        cov *= 2. * 1.38064852e-23 * T / (dx * dy * dt * dz)

        return cov
