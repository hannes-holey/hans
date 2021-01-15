import numpy as np
from pylub.field import VectorField
from pylub.stress import Deterministic
from pylub.bc import BoundaryCondition


class Flux:

    def __init__(self, disc, BC, geometry, material):

        self.disc = disc
        self.geometry = geometry
        self.material = material
        self.BC = BC

        self.detStress = Deterministic(self.disc, self.geometry, self.material)

    def LaxStep(self, q, h, visc_stress, dt, dir, ax):

        delta = {1: q.dx, 2: q.dy}
        p = self.detStress.pressure(q.field)
        F = self.hyperbolicFlux(q.field, p, ax) + self.diffusiveFlux(visc_stress.field, ax)

        Q = VectorField(self.disc)
        Q.field = 0.5 * (q.field + np.roll(q.field, dir, axis=ax)) - dt / (2. * delta[ax]) * dir * (F - np.roll(F, dir, axis=ax))

        return Q

    def fluxLW(self, q, h, visc_stress, dt, dir, ax):

        Q = self.LaxStep(q, h, visc_stress, dt, dir, ax)

        H = h.stagArray(dir, ax)
        _, viscous_stress = self.detStress.stress_avg(Q, H)
        p = self.detStress.pressure(Q.field)
        flux = self.hyperbolicFlux(Q.field, p, ax) + self.diffusiveFlux(viscous_stress.field, ax)

        return flux

    def getSource(self, stress, q, h):

        out = np.zeros_like(q.field)

        stress_wall_top = self.detStress.viscousStress_wall(q, h, "top")
        stress_wall_bot = self.detStress.viscousStress_wall(q, h, "bottom")

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = -j_x * hx / h0 - j_y * hy / h0

        out[1] = ((j_x * j_x / rho + stress.field[0] - stress_wall_top.field[0]) * hx
                  + (j_x * j_y / rho + stress.field[2] - stress_wall_top.field[5]) * hy
                  + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0

        out[2] = ((j_y * j_x / rho + stress.field[2] - stress_wall_top.field[5]) * hx
                  + (j_y * j_y / rho + stress.field[1] - stress_wall_top.field[1]) * hy
                  + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0

        # origin center
        # out[0] = -j_x * hx / h0 - j_y * hy / h0
        #
        # out[1] = ((j_x * j_x / rho + stress.field[0] - (stress_wall_top.field[0] + stress_wall_bot.field[0]) / 2) * hx
        #           + (j_x * j_y / rho + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * hy
        #           + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0
        #
        # out[2] = ((j_y * j_x / rho + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * hx
        #           + (j_y * j_y / rho + stress.field[1] - (stress_wall_top.field[1] + stress_wall_bot.field[1]) / 2) * hy
        #           + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0

        return out

    def hyperbolicFlux(self, q, p, ax):

        F = np.zeros_like(q)

        if ax == 1:
            F[0] = q[1]
            F[1] = q[1] * q[1] / q[0] + p
            F[2] = q[2] * q[1] / q[0]

        elif ax == 2:
            F[0] = q[2]
            F[1] = q[1] * q[2] / q[0]
            F[2] = q[2] * q[2] / q[0] + p

        return F

    def hyperbolicFW_BW(self, q, p, dt, dir):

        Fx = self.hyperbolicFlux(q.field, p, 1)
        Fy = self.hyperbolicFlux(q.field, p, 2)

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx)
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy)

        return dt / q.dx * flux_x, dt / q.dy * flux_y

    def diffusiveFlux(self, visc_stress, ax):

        D = np.zeros_like(visc_stress)
        if ax == 1:
            D[1] = visc_stress[0]
            D[2] = visc_stress[2]

        elif ax == 2:
            D[1] = visc_stress[2]
            D[2] = visc_stress[1]

        return D

    def diffusiveCD(self, q, visc_stress, dt):

        Dx = self.diffusiveFlux(visc_stress.field, 1)
        Dy = self.diffusiveFlux(visc_stress.field, 2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return dt / (2 * q.dx) * flux_x, dt / (2 * q.dy) * flux_y

    def Richtmyer(self, q, h, dt):

        stress, viscous_stress = self.detStress.stress_avg(q, h)

        fXE = self.fluxLW(q, h, viscous_stress, dt, -1, 1)
        fXW = self.fluxLW(q, h, viscous_stress, dt, 1, 1)
        fYN = self.fluxLW(q, h, viscous_stress, dt, -1, 2)
        fYS = self.fluxLW(q, h, viscous_stress, dt, 1, 2)

        src = self.getSource(viscous_stress, q, h)

        Q = VectorField(self.disc)

        Q.field = q.field - dt * ((fXE - fXW) / q.dy + (fYN - fYS) / q.dy - src)

        Q = BoundaryCondition(self.disc, self.BC, self.material).fill_ghost_cell(Q)

        return Q

    def MacCormack(self, q, h, dt, corrector=True):

        if corrector:
            Q = self.MacCormack(q, h, dt, corrector=False)
            dir = -1                # forwards difference
        else:
            Q = q
            dir = 1                 # backwards difference

        stress, viscousStress = self.detStress.stress_avg(Q, h)
        p = self.detStress.pressure(Q.field)

        fX, fY = self.hyperbolicFW_BW(Q, p, dt, dir)
        dX, dY = self.diffusiveCD(Q, viscousStress, dt)

        src = self.getSource(viscousStress, Q, h)
        Q.field = Q.field - fX - fY + dX + dY + dt * src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        Q = BoundaryCondition(self.disc, self.BC, self.material).fill_ghost_cell(Q)

        return Q
