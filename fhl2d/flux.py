#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .field import VectorField
from .stress import Newtonian


class Flux:

    def __init__(self, disc, geometry, numerics, material):

        self.disc = disc
        self.geometry = geometry
        self.material = material

        self.periodicX = bool(numerics['periodicX'])
        self.periodicY = bool(numerics['periodicY'])
        self.fluct = bool(material['Fluctuating'])

    def getFlux_LF(self, q, h, stress, dt, d, ax):

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
            dx = q.dx
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]
            dx = q.dy

        flux = VectorField(self.disc)

        flux.field[0] = 0.5 * (f1 + np.roll(f1, d, axis=ax)) - dx / (2. * dt) * d * (q.field[0] - np.roll(q.field[0], d, axis=ax))
        flux.field[1] = 0.5 * (f2 + np.roll(f2, d, axis=ax)) - dx / (2. * dt) * d * (q.field[1] - np.roll(q.field[1], d, axis=ax))
        flux.field[2] = 0.5 * (f3 + np.roll(f3, d, axis=ax)) - dx / (2. * dt) * d * (q.field[2] - np.roll(q.field[2], d, axis=ax))

        if self.periodicX is False:
            if d == -1:
                flux.field[0][-1,:] = f1[-1,:]  # Neumann
                flux.field[1][-1,:] = f2[-1,:]  # Neumann
                flux.field[2][-1,:] = flux.field[2][-2,:]  # Dirichlet

            elif d == 1:
                flux.field[0][0,:] = f1[0,:]
                flux.field[1][0,:] = f2[0,:]
                flux.field[2][0,:] = flux.field[2][1,:]

        if self.periodicY is False:
            if d == -1:
                flux.field[0][:,-1] = f1[:,-1]
                flux.field[1][:,-1] = f2[:,-1]
                flux.field[2][:,-1] = flux.field[2][:,-2]

            elif d == 1:
                flux.field[0][:,0] = f1[:,0]
                flux.field[1][:,0] = f2[:,0]
                flux.field[2][:,0] = flux.field[2][:,1]

        return flux

    def LaxStep(self, q, h, stress, dt, dir, ax):

        if ax == 0:
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
            dx = q.dx
        elif ax == 1:
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]
            dx = q.dy

        Q = VectorField(self.disc)

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], dir, axis=ax)) - dt / (2. * dx) * dir * (f1 - np.roll(f1, dir, axis=ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], dir, axis=ax)) - dt / (2. * dx) * dir * (f2 - np.roll(f2, dir, axis=ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], dir, axis=ax)) - dt / (2. * dx) * dir * (f3 - np.roll(f3, dir, axis=ax))

        H = h.stagArray(dir, ax)

        vStress, Stress, cov, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, H, dt)

        if self.fluct is True:
            Stress.addNoise_FH(cov)

        flux = np.empty_like(q.field)

        if ax == 0:
            flux[0] = -Stress.field[0]
            flux[1] = -Stress.field[2]
            flux[2] = Q.field[0]
        elif ax == 1:
            flux[0] = -Stress.field[2]
            flux[1] = -Stress.field[1]
            flux[2] = Q.field[1]

        return flux

    def Richtmyer(self, q, h, dt):

        viscousStress, stress, cov3, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(q, h, dt)

        if self.fluct is True:
            stress.addNoise_FH(cov3)

        fXE = self.LaxStep(q, h, stress, dt, -1, 0)
        fXW = self.LaxStep(q, h, stress, dt, 1, 0)
        fYN = self.LaxStep(q, h, stress, dt, -1, 1)
        fYS = self.LaxStep(q, h, stress, dt, 1, 1)

        src = self.getSource(viscousStress, q, h, dt)

        Q = VectorField(self.disc)

        Q.field = q.field - dt / q.dx * (fXE - fXW) - dt / q.dy * (fYN - fYS) + src

        return Q

    def totalFluxFW_BW(self, q, stress, dt, dir, ax):

        if ax == 0:
            F1 = -stress.field[0]
            F2 = -stress.field[2]
            F3 = q.field[0]
            dx = q.dx
        elif ax == 1:
            F1 = -stress.field[2]
            F2 = -stress.field[1]
            F3 = q.field[1]
            dx = q.dy

        flux = np.empty_like(q.field)

        flux[0] = dt / dx * (-dir) * (np.roll(F1, dir, axis=ax) - F1)
        flux[1] = dt / dx * (-dir) * (np.roll(F2, dir, axis=ax) - F2)
        flux[2] = dt / dx * (-dir) * (np.roll(F3, dir, axis=ax) - F3)

        return flux

    def MacCormack_total(self, q, h, dt, corrector=True):

        if corrector:
            Q = self.MacCormack_total(q, h, dt, corrector=False)
            dir = -1                # forwards difference
        else:
            Q = q
            dir = 1                 # backwards difference

        viscousStress, stress, cov3, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)

        if self.fluct is True:
            stress.addNoise_FH(cov3)

        fX = self.totalFluxFW_BW(Q, stress, dt, dir, 0)
        fY = self.totalFluxFW_BW(Q, stress, dt, dir, 1)

        src = self.getSource(viscousStress, Q, h, dt)

        Q.field = Q.field - fX - fY + src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        return Q

    def getSource(self, stress, q, h, dt):

        out = np.zeros_like(q.field)

        Q = self.cellAverageFaces(q)
        # print(np.mean(q.field[2]) - np.mean(Q.field[2]))

        # Q = q

        stress_wall_top = Newtonian(self.disc, self.geometry, self.material).viscousStress_wall(Q, h, dt, 1)
        stress_wall_bot = Newtonian(self.disc, self.geometry, self.material).viscousStress_wall(Q, h, dt, 0)

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        j_x = Q.field[0]
        j_y = Q.field[1]

        # j_x = q.field[0]
        # j_y = q.field[1]

        out[0] = ((stress.field[0] - stress_wall_top.field[0]) * hx + (stress.field[2] - stress_wall_top.field[5]) * hy + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0
        out[1] = ((stress.field[2] - stress_wall_top.field[5]) * hx + (stress.field[1] - stress_wall_top.field[1]) * hy + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0
        out[2] = -j_x * hx / h0 - j_y * hy / h0

        # out.field *= dt

        return out * dt

    def cellAverageFaces(self, q):

        Q_avg = VectorField(self.disc)

        qE = 0.5 * (np.roll(q.field, -1, axis=1) + q.field)
        qW = 0.5 * (np.roll(q.field, 1, axis=1) + q.field)
        qN = 0.5 * (np.roll(q.field, -1, axis=2) + q.field)
        qS = 0.5 * (np.roll(q.field, 1, axis=2) + q.field)

        Q_avg.field = 0.25 * (qE + qW + qN + qS)

        # qX = self.cubicInterpolation(q, 0)
        # qY = self.cubicInterpolation(q, 1)
        # Q_avg.field = 0.25 * (np.roll(qX, 1, axis=0) + np.roll(qY, 1, axis=1) + qX + qY)

        return q

    def hyperbolicFlux(self, q, p, ax):

        F = np.zeros_like(q)

        if ax == 1:
            F[0] = p
            F[2] = q[0]

        elif ax == 2:
            F[1] = p
            F[2] = q[1]

        return F

    def cubicInterpolation(self, q, ax):

        # weights proposed by Bell et al., PRE 76 (2007)
        # a1 = (np.sqrt(7) + 1) / 4
        # a2 = (np.sqrt(7) - 1) / 4

        # weights from piecewise parabolic method (PPM)
        # Collela and Woodward, J. Comp. Phys. 54 (1984)
        a1 = 7 / 12
        a2 = 1 / 12

        Q = a1 * (q.field + np.roll(q.field, -1, axis=ax)) - a2 * (np.roll(q.field, 1, axis=ax) + np.roll(q.field, -2, axis=ax))

        return Q

    def hyperbolicTVD(self, q, dt, ax):

        Q = self.cubicInterpolation(q, ax)
        P = Newtonian(self.disc, self.geometry, self.material).getPressure(Q)
        F = self.hyperbolicFlux(Q, P, ax)

        flux = np.roll(F, 0, axis=ax) - np.roll(F, 1, axis=ax)

        if ax == 1:
            dx = q.dx
        elif ax == 2:
            dx = q.dy

        return dt / dx * flux

    def hyperbolicCD(self, q, p, dt, ax):

        F = self.hyperbolicFlux(q.field, p, ax)

        flux = np.roll(F, -1, axis=ax) - np.roll(F, 1, axis=ax)

        if ax == 1:
            dx = q.dx

        elif ax == 2:
            dx = q.dy

        return dt / (2 * dx) * flux

    def hyperbolicFW_BW(self, q, p, dt, dir, ax):

        F = self.hyperbolicFlux(q.field, p, ax)
        flux = -dir * (np.roll(F, dir, axis=ax) - F)

        if ax == 1:
            dx = q.dx

        elif ax == 2:
            dx = q.dy

        return dt / dx * flux

    def diffusiveFlux(self, visc_stress, ax):

        D = np.zeros_like(visc_stress)
        if ax == 1:
            D[0] = visc_stress[0]
            D[1] = visc_stress[2]

        elif ax == 2:
            D[0] = visc_stress[2]
            D[1] = visc_stress[1]

        return D

    def diffusiveCD(self, q, visc_stress, dt, ax):

        D = self.diffusiveFlux(visc_stress.field, ax)
        flux = np.roll(D, -1, axis=ax) - np.roll(D, 1, axis=ax)

        if ax == 1:
            dx = q.dx

        elif ax == 2:
            dx = q.dy

        return dt / (2 * dx) * flux

    def stochasticFlux(self, cov, h, dt, ax, seed):

        Nx = int(self.disc['Nx'])
        Ny = int(self.disc['Ny'])
        dx = float(self.disc['dx'])
        dy = float(self.disc['dy'])
        dz = np.amin(h.field[0])

        mu = float(self.material['shear'])
        ceta = float(self.material['bulk'])
        lam = ceta - 2 / 3 * mu

        T = float(self.material['T0'])
        kB = 1.38064852e-23

        S = np.zeros([3, Nx, Ny])

        # can be used to modify, how often random fields are generated (within one timestep)
        # np.random.seed(seed)

        # mean = np.zeros(3)
        # noise = np.random.multivariate_normal(mean, cov, size=(Nx, Ny))
        # noise = noise.transpose(2, 0, 1)

        # print(seed, noise[0][10,10])
        # noise[1] = - noise[0]

        R11 = np.sqrt(2 * kB * T * (2 * mu + lam) / dx / dy / dz / dt) * np.random.normal(size=(Nx,Ny))
        R22 = -R11
        R12 = np.sqrt(2 * kB * T * mu / dx / dy / dz / dt) * np.random.normal(size=(Nx,Ny))

        if ax == 1:
            S[0] = R11 - np.roll(R11, 1, axis=ax - 1)
            S[1] = R12 - np.roll(R12, 1, axis=ax - 1)
            dx = dx
        if ax == 2:
            S[0] = R12 - np.roll(R12, 1, axis=ax - 1)
            S[1] = R22 - np.roll(R22, 1, axis=ax - 1)
            dx = dy

        # if ax == 1:
        #     S[0] = noise[0] - np.roll(noise[0], 1, axis=ax - 1)
        #     S[1] = noise[2] - np.roll(noise[2], 1, axis=ax - 1)
        #     dx = dx
        # if ax == 2:
        #     S[0] = noise[2] - np.roll(noise[2], 1, axis=ax - 1)
        #     S[1] = noise[1] - np.roll(noise[1], 1, axis=ax - 1)
        #     dx = dy

        return dt / dx * S

    def MacCormack(self, q, h, dt, i, corrector=True):

        if corrector:
            Q = self.MacCormack(q, h, dt, i, corrector=False)
            dir = -1                # forwards difference
            weight = np.sqrt(2)
        else:
            Q = q
            dir = 1                 # backwards difference
            weight = np.sqrt(2)

        viscousStress, stress, cov3, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)

        fX = self.hyperbolicFW_BW(Q, p, dt, dir, 1)
        fY = self.hyperbolicFW_BW(Q, p, dt, dir, 2)

        if bool(self.material['Fluctuating']) is True:
            sX = weight * self.stochasticFlux(cov3, h, dt, 1, i)
            sY = weight * self.stochasticFlux(cov3, h, dt, 2, i)
        else:
            sX = np.zeros_like(fX)
            sY = np.zeros_like(fX)

        if bool(self.material['Rey']) is False:
            dX = self.diffusiveCD(Q, viscousStress, dt, 1)
            dY = self.diffusiveCD(Q, viscousStress, dt, 2)
        else:
            dX = np.zeros_like(fX)
            dY = np.zeros_like(fX)

        src = self.getSource(viscousStress, Q, h, dt)

        Q.field = Q.field - fX - fY + dX + dY + sX + sY + src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        return Q

    def RungeKutta3(self, q, h, dt, i, step=3):

        assert step in np.arange(1,4)

        if step == 3:
            Q = self.RungeKutta3(q, h, dt, i, step=2)
            weight = np.sqrt(3)
            # weight = 15 / 16
        if step == 2:
            Q = self.RungeKutta3(q, h, dt, i, step=1)
            weight = np.sqrt(3)
            # weight = 3 / 2
        if step == 1:
            Q = q
            weight = np.sqrt(3)
            # weight = 3 / 4

        viscousStress, stress, cov3, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)

        fX = self.hyperbolicTVD(Q, dt, 1)
        fY = self.hyperbolicTVD(Q, dt, 2)

        if bool(self.material['Fluctuating']) is True:
            sX = weight * self.stochasticFlux(cov3, h, dt, 1, i)
            sY = weight * self.stochasticFlux(cov3, h, dt, 2, i)
        else:
            sX = np.zeros_like(fX)
            sY = np.zeros_like(fX)

        if bool(self.material['Rey']) is False:
            dX = self.diffusiveCD(Q, viscousStress, dt, 1)
            dY = self.diffusiveCD(Q, viscousStress, dt, 2)
        else:
            dX = np.zeros_like(fX)
            dY = np.zeros_like(fX)

        src = self.getSource(viscousStress, Q, h, dt)

        tmp = Q.field - fX - fY + dX + dY + sX + sY + src

        if step == 3:
            Q.field = 1 / 3 * q.field + 2 / 3 * tmp

        if step == 2:
            Q.field = 3 / 4 * q.field + 1 / 4 * tmp

        if step == 1:
            Q.field = tmp

        return Q
