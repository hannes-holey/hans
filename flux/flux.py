#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from field.field import VectorField
from stress.stress import Newtonian
# from eos.eos import DowsonHigginson, PowerLaw


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

    def getQ_LW(self, q, h, stress, dt, d, ax):

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

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], d, axis=ax)) - dt / (2. * dx) * d * (f1 - np.roll(f1, d, axis=ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], d, axis=ax)) - dt / (2. * dx) * d * (f2 - np.roll(f2, d, axis=ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], d, axis=ax)) - dt / (2. * dx) * d * (f3 - np.roll(f3, d, axis=ax))

        if self.periodicX is False:
            if d == -1:
                pass
                # Q.field[0][-1,:] = q.field[0][-1,:]
                # Q.field[1][-1,:] = q.field[1][-1,:]
                # Q.field[2][-1,:] = q.field[2][-1,:]

            elif d == 1:
                pass
                # Q.field[0][0,:] = q.field[0][0,:]
                # Q.field[1][0,:] = q.field[1][0,:]
                # Q.field[2][0,:] = q.field[2][0,:]

        if self.periodicY is False:
            if d == -1:
                pass
                # Q.field[0][:,-1] = q.field[0][:,-1]
                # Q.field[1][:,-1] = q.field[1][:,-1]
                # Q.field[2][:,-1] = Q.field[2][:,-2]

            elif d == 1:
                pass
                # Q.field[0][:,0] = q.field[0][:,0]
                # Q.field[1][:,0] = q.field[1][:,0]
                # Q.field[2][:,0] = Q.field[2][:,1]

        return Q

    def getFlux_LW(self, q, h, stress, dt, d, ax):

        Q = self.getQ_LW(q, h, stress, dt, d, ax)
        stress_bound = stress.stagArray(d, ax + 1)

        flux = VectorField(self.disc)

        if ax == 0:
            f1 = -stress_bound.field[0]
            f2 = -stress_bound.field[2]
            f3 = Q.field[0]
        elif ax == 1:
            f1 = -stress_bound.field[2]
            f2 = -stress_bound.field[1]
            f3 = Q.field[1]

        flux.field[0] = f1
        flux.field[1] = f2
        flux.field[2] = f3

        if self.periodicX is False:
            if d == -1:
                pass
                # flux.field[0][-1,:] = -stress_center.field[0][-1,:] # Neumann
                # flux.field[1][-1,:] = -stress_center.field[2][-1,:] # Neumann
                # flux.field[2][-1,:] = f3[-2,:] # Dirichlet

            elif d == 1:
                pass
                # flux.field[0][0,:] = -stress_center.field[0][-1,:] # Neumann
                # flux.field[1][0,:] = -stress_center.field[2][-1,:] # Neumann
                # flux.field[2][0,:] = f3[1,:]

        if self.periodicY is False:
            if d == -1:
                pass
                # flux.field[0][:,-1] = f1[:,-1]
                # flux.field[1][:,-1] = f2[:,-1]
                # flux.field[2][:,-1] = flux.field[2][:,-2]

            elif d == 1:
                pass
                # flux.field[0][:,0] = f1[:,0]
                # flux.field[1][:,0] = f2[:,0]
                # flux.field[2][:,0] = flux.field[2][:,1]

        return flux

    def getQ_MC(self, q, h, stress, dt, d, ax):

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

        Q.field[0] = q.field[0] + dt / dx * (f1 - np.roll(f1, -1, axis=ax))
        Q.field[1] = q.field[1] + dt / dx * (f2 - np.roll(f2, -1, axis=ax))
        Q.field[2] = q.field[2] + dt / dx * (f3 - np.roll(f3, -1, axis=ax))

        return Q

    def getFlux_MC(self, q, h, stress, dt, d, ax):

        Q = self.getQ_MC(q, h, stress, dt, d, ax)

        stress_tmp, cov = Newtonian(self.disc, self.geometry, self.material).stress_avg(Q, h, dt)
        if self.fluct is True:
            stress_tmp.addNoise_FH(cov)

        flux = VectorField(self.disc)

        if ax == 0:
            f1_p = -stress_tmp.field[0]
            f2_p = -stress_tmp.field[2]
            f3_p = Q.field[0]
            f1 = -stress.field[0]
            f2 = -stress.field[2]
            f3 = q.field[0]
        elif ax == 1:
            f1_p = -stress_tmp.field[2]
            f2_p = -stress_tmp.field[1]
            f3_p = Q.field[1]
            f1 = -stress.field[2]
            f2 = -stress.field[1]
            f3 = q.field[1]

        if d == -1:
            flux.field[0] = 0.5 * (np.roll(f1, d, axis=ax) + f1_p)
            flux.field[1] = 0.5 * (np.roll(f2, d, axis=ax) + f2_p)
            flux.field[2] = 0.5 * (np.roll(f3, d, axis=ax) + f3_p)

        if d == 1:
            flux.field[0] = 0.5 * (np.roll(f1_p, d, axis=ax) + f1)
            flux.field[1] = 0.5 * (np.roll(f2_p, d, axis=ax) + f2)
            flux.field[2] = 0.5 * (np.roll(f3_p, d, axis=ax) + f3)

        return flux

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

        return out
