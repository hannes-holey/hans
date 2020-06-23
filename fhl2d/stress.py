#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .eos import EquationOfState
from .field import VectorField, TensorField


class Newtonian:

    def __init__(self, disc, geometry, material):

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def viscousStress_avg(self, q, h, dt):

        out = VectorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        T = float(self.mat['T0'])
        mu = float(self.mat['shear'])
        ceta = float(self.mat['bulk'])
        lam = ceta - 2 / 3 * mu

        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        dz = np.amin(h0)

        if bool(self.mat['Rey']) is False:
            # origin bottom
            out.field[0] = (-4 * (U * rho - (3 * j_x) / 2) * (mu + lam / 2) * hx - 2 * lam * (V * rho - (3 * j_y) / 2) * hy) / (h0 * rho)
            out.field[1] = (-4 * (V * rho - (3 * j_y) / 2) * (mu + lam / 2) * hy - 2 * lam * (U * rho - (3 * j_x) / 2) * hx) / (h0 * rho)
            out.field[2] = -2 * mu * ((V * rho - (3 * j_y) / 2) * hx + hy * (U * rho - (3 * j_x) / 2)) / (h0 * rho)

        cov = self.getCovariance(out.ndim, mu, lam, T, q.dx, q.dy, dz, dt)

        return out, cov

    def stress_avg(self, q, h, dt):

        # if self.mat['EOS'] == 'DH':
        #     eqOfState = DowsonHigginson(self.mat)
        # elif self.mat['EOS'] == 'PL':
        #     eqOfState = PowerLaw(self.mat)

        viscStress, cov = self.viscousStress_avg(q, h, dt)
        stress = VectorField(self.disc)

        pressure = EquationOfState(self.mat).isoT_pressure(q.field[2])

        stress.field[0] = viscStress.field[0] - pressure
        stress.field[1] = viscStress.field[1] - pressure

        return viscStress, stress, cov, pressure

    def getPressure(self, q):
        # if self.mat['EOS'] == 'DH':
        #     eqOfState = DowsonHigginson(self.mat)
        # elif self.mat['EOS'] == 'PL':
        #     eqOfState = PowerLaw(self.mat)

        pressure = EquationOfState(self.mat).isoT_pressure(q[2])

        return pressure

    def viscousStress_wall(self, q, h, dt, bound):

        out = TensorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        mu = float(self.mat['shear'])
        ceta = float(self.mat['bulk'])
        lam = ceta - 2 / 3 * mu

        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        # dz = np.amin(h0)

        if bound == 1:
            if bool(self.mat['Rey']) is False:
                out.field[0] = (-8 * (mu + lam / 2) * (U * rho - (3 * j_x) / 2) * hx - 4 * hy * lam * (V * rho - (3 * j_y) / 2)) / (h0 * rho)
                out.field[1] = (-8 * (mu + lam / 2) * (V * rho - (3 * j_y) / 2) * hy - 4 * hx * lam * (U * rho - (3 * j_x) / 2)) / (h0 * rho)
                out.field[2] = -4 * lam * ((U * rho - (3 * j_x) / 2) * hx + hy * (V * rho - (3 * j_y) / 2)) / (h0 * rho)
                out.field[3] = 2 * mu * (2 * V * rho - 3 * j_y) / (h0 * rho)
                out.field[4] = 2 * mu * (2 * U * rho - 3 * j_x) / (h0 * rho)
                out.field[5] = -4 * ((V * rho - (3 * j_y) / 2) * hx + hy * (U * rho - (3 * j_x) / 2)) * mu / (h0 * rho)
            else:
                out.field[3] = 2 * mu * (2 * V * rho - 3 * j_y) / (h0 * rho)
                out.field[4] = 2 * mu * (2 * U * rho - 3 * j_x) / (h0 * rho)
        elif bound == 0:
            out.field[3] = -2 * mu * (V * rho - 3 * j_y) / (h0 * rho)
            out.field[4] = -2 * mu * (U * rho - 3 * j_x) / (h0 * rho)

        # if bool(self.mat['Fluctuating']) is True:
        #     cov = self.getCovariance(out.ndim, mu, lam, T, q.dx, q.dy, dz, dt)
        #     out.addNoise_FH(cov)

        return out

    def getCovariance(self, ndim, mu, lam, T, dx, dy, dz, dt):
        if ndim == 3:
            cov = np.array([[2 * mu + lam, lam, 0.],
                            [lam, 2 * mu + lam, 0.],
                            [0., 0., mu]])
        elif ndim == 6:
            cov = np.array([[2 * mu + lam, lam, lam, 0., 0., 0.],
                            [lam, 2 * mu + lam, lam, 0., 0., 0.],
                            [lam, lam, 2 * mu + lam, 0., 0., 0.],
                            [0., 0., 0., mu, 0., 0.],
                            [0., 0., 0., 0., mu, 0.],
                            [0., 0., 0., 0., 0., mu]])

        cov *= 2. * 1.38064852e-23 * T / (dx * dy * dt * dz)

        return cov
