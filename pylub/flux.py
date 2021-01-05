import numpy as np
from .field import VectorField
from .stress import Deterministic


class Flux:

    def __init__(self, disc, geometry, numerics, material):

        self.disc = disc
        self.geometry = geometry
        self.material = material
        self.numerics = numerics

        self.detStress = Deterministic(self.disc, self.geometry, self.numerics, self.material)

    def LaxStep(self, q, h, stress, dt, dir, ax):

        if ax == 0:
            f1 = q.field[0]
            f2 = -stress.field[0]
            f3 = -stress.field[2]
            dx = q.dx
        elif ax == 1:
            f1 = q.field[1]
            f2 = -stress.field[2]
            f3 = -stress.field[1]
            dx = q.dy

        Q = VectorField(self.disc)

        Q.field[0] = 0.5 * (q.field[0] + np.roll(q.field[0], dir, axis=ax)) - dt / (2. * dx) * dir * (f1 - np.roll(f1, dir, axis=ax))
        Q.field[1] = 0.5 * (q.field[1] + np.roll(q.field[1], dir, axis=ax)) - dt / (2. * dx) * dir * (f2 - np.roll(f2, dir, axis=ax))
        Q.field[2] = 0.5 * (q.field[2] + np.roll(q.field[2], dir, axis=ax)) - dt / (2. * dx) * dir * (f3 - np.roll(f3, dir, axis=ax))

        H = h.stagArray(dir, ax)

        stress, viscous_stress = self.detStress.stress_avg(Q, H, dt)

        flux = np.empty_like(q.field)

        if ax == 0:
            flux[0] = Q.field[1]
            flux[1] = -stress.field[0]
            flux[2] = -stress.field[2]
        elif ax == 1:
            flux[0] = Q.field[2]
            flux[1] = -stress.field[2]
            flux[2] = -stress.field[1]

        return flux

    def Richtmyer(self, q, h, dt):

        stress, viscous_stress = self.detStress.stress_avg(q, h, dt)

        fXE = self.LaxStep(q, h, stress, dt, -1, 0)
        fXW = self.LaxStep(q, h, stress, dt, 1, 0)
        fYN = self.LaxStep(q, h, stress, dt, -1, 1)
        fYS = self.LaxStep(q, h, stress, dt, 1, 1)

        src = self.getSource(viscous_stress, q, h, dt)

        Q = VectorField(self.disc)

        Q.field = q.field - dt / q.dx * (fXE - fXW) - dt / q.dy * (fYN - fYS) + src

        return Q

    def getSource(self, stress, q, h, dt):

        out = np.zeros_like(q.field)

        stress_wall_top = self.detStress.viscousStress_wall(q, h, dt, "top")
        stress_wall_bot = self.detStress.viscousStress_wall(q, h, dt, "bottom")

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

        return out * dt

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

    def MacCormack(self, q, h, dt, corrector=True):

        if corrector:
            Q = self.MacCormack(q, h, dt, corrector=False)
            dir = -1                # forwards difference
        else:
            Q = q
            dir = 1                 # backwards difference

        stress, viscousStress = self.detStress.stress_avg(Q, h, dt)
        p = self.detStress.pressure(Q.field)

        fX, fY = self.hyperbolicFW_BW(Q, p, dt, dir)

        if bool(self.numerics['Rey']) is False:
            dX, dY = self.diffusiveCD(Q, viscousStress, dt)
        else:
            dX = dY = np.zeros_like(fX)

        src = self.getSource(viscousStress, Q, h, dt)
        Q.field = Q.field - fX - fY + dX + dY + src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        return Q
