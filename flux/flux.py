#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian


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

    # def getQ_LW(self, q, h, stress, dt, d, ax):
    #
    #     if ax == 0:
    #         f1 = -stress.field[0]
    #         f2 = -stress.field[2]
    #         f3 = q.field[0]
    #         dx = q.dx
    #     elif ax == 1:
    #         f1 = -stress.field[2]
    #         f2 = -stress.field[1]
    #         f3 = q.field[1]
    #         dx = q.dy
    #
    #     Q = VectorField(self.disc)
    #
    #     Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis=ax)) - dt / (2. * dx) * d * (f1 - np.roll(f1, d, axis=ax))
    #     Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis=ax)) - dt / (2. * dx) * d * (f2 - np.roll(f2, d, axis=ax))
    #     Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis=ax)) - dt / (2. * dx) * d * (f3 - np.roll(f3, d, axis=ax))
    #
    #     return Q
    #
    # def getFlux_LW(self, q, h, stress, dt, d, ax):
    #
    #     Q = self.getQ_LW(q, h, stress, dt, d, ax)
    #     stress_bound = stress.stagArray(d, ax + 1)
    #
    #     flux = VectorField(self.disc)
    #
    #     if ax == 0:
    #         f1 = -stress_bound.field[0]
    #         f2 = -stress_bound.field[2]
    #         f3 = Q.field[0]
    #     elif ax == 1:
    #         f1 = -stress_bound.field[2]
    #         f2 = -stress_bound.field[1]
    #         f3 = Q.field[1]
    #
    #     flux.field[0] = f1
    #     flux.field[1] = f2
    #     flux.field[2] = f3
    #
    #     return flux

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
        flux = VectorField(self.disc)

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], dir, axis=ax)) - dt / (2. * dx) * dir * (f1 - np.roll(f1, dir, axis=ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], dir, axis=ax)) - dt / (2. * dx) * dir * (f2 - np.roll(f2, dir, axis=ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], dir, axis=ax)) - dt / (2. * dx) * dir * (f3 - np.roll(f3, dir, axis=ax))

        H = h.stagArray(dir, ax)
        Stress = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, H, dt)[1]
        # if self.fluct is True:
        #     Stress.addNoise_FH(cov3)

        if ax == 0:
            flux.field[0] = -Stress.field[0]
            flux.field[1] = -Stress.field[2]
            flux.field[2] = Q.field[0]
        elif ax == 1:
            flux.field[0] = -Stress.field[2]
            flux.field[1] = -Stress.field[1]
            flux.field[2] = Q.field[1]

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

        Q.field = q.field - dt / q.dx * (fXE.field - fXW.field) - dt / q.dy * (fYN.field - fYS.field) + src.field

        return Q

    # def getQ_MC(self, q, h, stress, dt, d, ax):
    #
    #     if ax == 0:
    #         f1 = -stress.field[0]
    #         f2 = -stress.field[2]
    #         f3 = q.field[0]
    #         dx = q.dx
    #     elif ax == 1:
    #         f1 = -stress.field[2]
    #         f2 = -stress.field[1]
    #         f3 = q.field[1]
    #         dx = q.dy
    #
    #     Q = VectorField(self.disc)
    #
    #     Q.field[0] = q.field[0] + dt / dx * (f1 - np.roll(f1, -1, axis=ax))
    #     Q.field[1] = q.field[1] + dt / dx * (f2 - np.roll(f2, -1, axis=ax))
    #     Q.field[2] = q.field[2] + dt / dx * (f3 - np.roll(f3, -1, axis=ax))
    #
    #     return Q
    #
    # def getFlux_MC(self, q, h, stress, dt, d, ax):
    #
    #     Q = self.getQ_MC(q, h, stress, dt, d, ax)
    #
    #     _, stress_tmp, cov, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)
    #     if self.fluct is True:
    #         stress_tmp.addNoise_FH(cov)
    #
    #     flux = VectorField(self.disc)
    #
    #     if ax == 0:
    #         f1_p = -stress_tmp.field[0]
    #         f2_p = -stress_tmp.field[2]
    #         f3_p = Q.field[0]
    #         f1 = -stress.field[0]
    #         f2 = -stress.field[2]
    #         f3 = q.field[0]
    #     elif ax == 1:
    #         f1_p = -stress_tmp.field[2]
    #         f2_p = -stress_tmp.field[1]
    #         f3_p = Q.field[1]
    #         f1 = -stress.field[2]
    #         f2 = -stress.field[1]
    #         f3 = q.field[1]
    #
    #     if d == -1:
    #         flux.field[0] = 0.5 * (np.roll(f1, d, axis=ax) + f1_p)
    #         flux.field[1] = 0.5 * (np.roll(f2, d, axis=ax) + f2_p)
    #         flux.field[2] = 0.5 * (np.roll(f3, d, axis=ax) + f3_p)
    #
    #     if d == 1:
    #         flux.field[0] = 0.5 * (np.roll(f1_p, d, axis=ax) + f1)
    #         flux.field[1] = 0.5 * (np.roll(f2_p, d, axis=ax) + f2)
    #         flux.field[2] = 0.5 * (np.roll(f3_p, d, axis=ax) + f3)
    #
    #     return flux

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

        flux = VectorField(self.disc)

        flux.field[0] = dt / dx * (-dir) * (np.roll(F1, dir, axis=ax) - F1)
        flux.field[1] = dt / dx * (-dir) * (np.roll(F2, dir, axis=ax) - F2)
        flux.field[2] = dt / dx * (-dir) * (np.roll(F3, dir, axis=ax) - F3)

        return flux

    def MacCormack(self, q, h, dt, corrector=True):

        if corrector:
            Q = self.MacCormack_split(q, h, dt, corrector=False)
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

        Q.field = Q.field - fX.field - fY.field + src.field

        return Q

    def getSource(self, stress, q, h, dt):

        out = VectorField(self.disc)

        stress_wall_top = Newtonian(self.disc, self.geometry, self.material).viscousStress_wall(q, h, dt, 1)
        stress_wall_bot = Newtonian(self.disc, self.geometry, self.material).viscousStress_wall(q, h, dt, 0)

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        j_x = q.field[0]
        j_y = q.field[1]

        out.field[0] = ((stress.field[0] - stress_wall_top.field[0]) * hx + (stress.field[2] - stress_wall_top.field[5]) * hy + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0
        out.field[1] = ((stress.field[2] - stress_wall_top.field[5]) * hx + (stress.field[1] - stress_wall_top.field[1]) * hy + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0
        out.field[2] = -j_x * hx / h0 - j_y * hy / h0

        out.field *= dt

        return out

    def hyperbolicFW_BW(self, q, p, dt, d, ax):

        if ax == 0:
            F1 = p
            F2 = np.zeros_like(F1)
            F3 = q.field[0]

            dx = q.dx

        elif ax == 1:
            F1 = np.zeros_like(p)
            F2 = p
            F3 = q.field[1]

            dx = q.dy

        flux = VectorField(self.disc)

        flux.field[0] = dt / dx * (-d) * (np.roll(F1, d, axis=ax) - F1)
        flux.field[1] = dt / dx * (-d) * (np.roll(F2, d, axis=ax) - F2)
        flux.field[2] = dt / dx * (-d) * (np.roll(F3, d, axis=ax) - F3)

        return flux

    def diffusiveCD(self, q, visc_stress, dt, ax):

        if ax == 0:
            D1 = visc_stress.field[0]
            D2 = visc_stress.field[2]
            D3 = np.zeros_like(D1)
            dx = q.dx
        elif ax == 1:
            D1 = visc_stress.field[2]
            D2 = visc_stress.field[1]
            D3 = np.zeros_like(D1)
            dx = q.dy

        flux = VectorField(self.disc)

        flux.field[0] = dt / (2 * dx) * (np.roll(D1, -1, axis=ax) - np.roll(D1, 1, axis=ax))
        flux.field[1] = dt / (2 * dx) * (np.roll(D2, -1, axis=ax) - np.roll(D2, 1, axis=ax))
        flux.field[2] = dt / (2 * dx) * (np.roll(D3, -1, axis=ax) - np.roll(D3, 1, axis=ax))

        return flux

    def stochasticFlux(self, cov, dt, ax):

        flux_left = VectorField(self.disc)
        flux_right = VectorField(self.disc)
        out = VectorField(self.disc)

        flux_left.addNoise_FH(cov)
        flux_right.addNoise_FH(cov)

        if ax == 0:
            f1 = flux_left.field[0] - flux_right.field[0]
            f2 = flux_left.field[2] - flux_right.field[2]
            dx = flux_left.dx
        if ax == 1:
            f1 = flux_left.field[2] - flux_right.field[2]
            f2 = flux_left.field[1] - flux_right.field[1]
            dx = flux_left.dy

        out.field[0] = dt / dx * f1 * np.sqrt(2)
        out.field[1] = dt / dx * f2 * np.sqrt(2)

        return out

    def MacCormack_split(self, q, h, dt, corrector=True):

        if corrector:
            Q = self.MacCormack_split(q, h, dt, corrector=False)
            dir = -1                # forwards difference
        else:
            Q = q
            dir = 1                 # backwards difference

        viscousStress, stress, cov3, p = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)

        fX = self.hyperbolicFW_BW(Q, p, dt, dir, 0)
        fY = self.hyperbolicFW_BW(Q, p, dt, dir, 1)

        if bool(self.material['Fluctuating']) is True:
            sX = self.stochasticFlux(cov3, dt, 0)
            sY = self.stochasticFlux(cov3, dt, 1)
        else:
            sX = VectorField(self.disc)
            sY = VectorField(self.disc)

        if bool(self.material['Rey']) is True:
            dX = VectorField(self.disc)
            dY = VectorField(self.disc)
        else:
            dX = self.diffusiveCD(Q, viscousStress, dt, 0)
            dY = self.diffusiveCD(Q, viscousStress, dt, 1)

        src = self.getSource(viscousStress, Q, h, dt)

        Q.field = Q.field - fX.field - fY.field + dX.field + dY.field + sX.field + sY.field + src.field

        return Q
