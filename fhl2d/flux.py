import numpy as np
from .field import VectorField
from .stress import Deterministic, Stochastic


class Flux:

    def __init__(self, disc, geometry, numerics, material):

        self.disc = disc
        self.geometry = geometry
        self.material = material
        self.numerics = numerics

        self.periodicX = bool(numerics['periodicX'])
        self.periodicY = bool(numerics['periodicY'])
        self.fluct = bool(numerics['Fluctuating'])
        if self.fluct:
            self.corr = bool(numerics['correction'])

        self.detStress = Deterministic(self.disc, self.geometry, self.numerics, self.material)
        self.stochStress = Stochastic(self.disc, self.material)

    def getFlux_LF(self, q, h, stress, dt, d, ax):

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

        flux = VectorField(self.disc)

        flux.field[0] = 0.5 * (f1 + np.roll(f1, d, axis=ax)) - dx / (2. * dt) * d * (q.field[0] - np.roll(q.field[0], d, axis=ax))
        flux.field[1] = 0.5 * (f2 + np.roll(f2, d, axis=ax)) - dx / (2. * dt) * d * (q.field[1] - np.roll(q.field[1], d, axis=ax))
        flux.field[2] = 0.5 * (f3 + np.roll(f3, d, axis=ax)) - dx / (2. * dt) * d * (q.field[2] - np.roll(q.field[2], d, axis=ax))

        if self.periodicX is False:
            if d == -1:
                flux.field[0][-1, :] = flux.field[0][-2, :]  # Dirichlet
                flux.field[1][-1, :] = f2[-1, :]  # Neumann
                flux.field[2][-1, :] = f3[-1, :]  # Neumann

            elif d == 1:
                flux.field[0][0, :] = flux.field[0][1, :]
                flux.field[1][0, :] = f2[0, :]
                flux.field[2][0, :] = f3[0, :]

        if self.periodicY is False:
            if d == -1:
                flux.field[0][:, -1] = flux.field[0][:, -2]
                flux.field[1][:, -1] = f2[:, -1]
                flux.field[2][:, -1] = f3[:, -1]

            elif d == 1:
                flux.field[0][:, 0] = flux.field[0][:, 1]
                flux.field[1][:, 0] = f2[:, 0]
                flux.field[2][:, 0] = f3[:, 0]

        return flux

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

    def totalFluxFW_BW(self, q, stress, dt, dir, ax):

        if ax == 0:
            F1 = q.field[1]
            F2 = -stress.field[0]
            F3 = -stress.field[2]
            dx = q.dx
        elif ax == 1:
            F1 = q.field[2]
            F2 = -stress.field[2]
            F3 = -stress.field[1]
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

        stress, viscous_stress = self.detStress.stress_avg(Q, h, dt)

        fX = self.totalFluxFW_BW(Q, stress, dt, dir, 0)
        fY = self.totalFluxFW_BW(Q, stress, dt, dir, 1)

        src = self.getSource(viscous_stress, Q, h, dt)

        Q.field = Q.field - fX - fY + src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        return Q

    def getSource(self, stress, stochastic_stress, q, h, dt):

        out = np.zeros_like(q.field)

        stress_wall_top = self.detStress.viscousStress_wall(q, h, dt, "top")
        stress_wall_bot = self.detStress.viscousStress_wall(q, h, dt, "bottom")

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        # stress_wall_top.field += stochastic_stress
        # stress_wall_bot.field += np.roll(stochastic_stress, 1, axis=1)

        # stress.field[0] += stochastic_stress[0]
        # stress.field[1] += stochastic_stress[1]
        # stress.field[2] += stochastic_stress[5]

        out[0] = -j_x * hx / h0 - j_y * hy / h0

        out[1] = ((j_x * j_x / rho + stress.field[0] - (stress_wall_top.field[0] + stress_wall_bot.field[0]) / 2) * hx
                  + (j_x * j_y / rho + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * hy
                  + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0

        out[2] = ((j_y * j_x / rho + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * hx
                  + (j_y * j_y / rho + stress.field[1] - (stress_wall_top.field[1] + stress_wall_bot.field[1]) / 2) * hy
                  + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0

        # out[1] = ((j_x * j_x / rho + stress.field[0] - stress_wall_top.field[0]) * hx
        #           + (j_x * j_y / rho + stress.field[2] - stress_wall_top.field[5]) * hy
        #           + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h0

        # out[2] = ((j_y * j_x / rho + stress.field[2] - stress_wall_top.field[5]) * hx
        #           + (j_y * j_y / rho + stress.field[1] - stress_wall_top.field[1]) * hy
        #           + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h0

        return out * dt

    def cellAverageFaces(self, q):

        Q_avg = VectorField(self.disc)

        qE = 0.5 * (np.roll(q.field, -1, axis=1) + q.field)
        qW = 0.5 * (np.roll(q.field, 1, axis=1) + q.field)
        qN = 0.5 * (np.roll(q.field, -1, axis=2) + q.field)
        qS = 0.5 * (np.roll(q.field, 1, axis=2) + q.field)

        Q_avg.field = 0.25 * (qE + qW + qN + qS)

        return Q_avg

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

    def hyperbolicTVD(self, q, dt):

        Qx = self.cubicInterpolation(q, 1)
        Px = self.detStress.pressure(Qx)

        Qy = self.cubicInterpolation(q, 2)
        Py = self.detStress.pressure(Qy)

        Fx = self.hyperbolicFlux(Qx, Px, 1)
        Fy = self.hyperbolicFlux(Qy, Py, 2)

        flux_x = Fx - np.roll(Fx, 1, axis=1)
        flux_y = Fy - np.roll(Fy, 1, axis=2)

        return dt / q.dx * flux_x, dt / q.dy * flux_y

    def hyperbolicCD(self, q, p, dt, ax):

        F = self.hyperbolicFlux(q.field, p, ax)

        flux = np.roll(F, -1, axis=ax) - np.roll(F, 1, axis=ax)

        if ax == 1:
            dx = q.dx

        elif ax == 2:
            dx = q.dy

        return dt / (2 * dx) * flux

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

    def stochasticFlux(self, sFlux, h, dt):

        Nx = int(self.disc['Nx'])
        Ny = int(self.disc['Ny'])
        dx = float(self.disc['dx'])
        dy = float(self.disc['dy'])
        dz = h.field[0]

        Sx_E = np.zeros([3, Nx, Ny])
        Sx_W = np.zeros([3, Nx, Ny])
        Sy_N = np.zeros([3, Nx, Ny])
        Sy_S = np.zeros([3, Nx, Ny])

        if self.corr:
            corrX = dx / dz
            corrY = dy / dz
        else:
            corrX = corrY = 1.

        Sx_E[1] = sFlux[0]
        Sx_E[2] = sFlux[5]
        Sx_W[1] = np.roll(sFlux[0], 1, axis=0)
        Sx_W[2] = np.roll(sFlux[5], 1, axis=0)

        Sy_N[1] = sFlux[5]
        Sy_N[2] = sFlux[1]
        Sy_S[1] = np.roll(sFlux[5], 1, axis=1)
        Sy_S[2] = np.roll(sFlux[1], 1, axis=1)

        return dt / dx * (Sx_E - Sx_W) * corrX, dt / dy * (Sy_N - Sy_S) * corrY

    def MacCormack(self, q, h, dt, W_A, W_B, corrector=True):

        if corrector:
            Q = self.MacCormack(q, h, dt, W_A, W_B, corrector=False)
            dir = -1                # forwards difference
        else:
            Q = q
            dir = 1                 # backwards difference

        stress, viscousStress = self.detStress.stress_avg(Q, h, dt)
        p = self.detStress.pressure(Q.field)

        fX, fY = self.hyperbolicFW_BW(Q, p, dt, dir)

        if bool(self.numerics['Fluctuating']) is True:
            stochastic_stress = self.stochStress.full_tensor(W_A, W_B, h, dt, int(corrector) + 1)
            sX, sY = self.stochasticFlux(stochastic_stress, h, dt)
        else:
            stochastic_stress = np.zeros([6, q.Nx, q.Ny])
            sX = sY = np.zeros_like(fX)

        if bool(self.numerics['Rey']) is False:
            dX, dY = self.diffusiveCD(Q, viscousStress, dt)
        else:
            dX = dY = np.zeros_like(fX)

        src = self.getSource(viscousStress, stochastic_stress, Q, h, dt)
        Q.field = Q.field - fX - fY + dX + dY + sX + sY + src

        if corrector:
            Q.field = 0.5 * (Q.field + q.field)

        return Q

    def RungeKutta3(self, q, h, dt, W_A, W_B, stage=3):

        assert stage in np.arange(1, 4)

        if stage == 3:
            Q = self.RungeKutta3(q, h, dt, W_A, W_B, stage=2)
        if stage == 2:
            Q = self.RungeKutta3(q, h, dt, W_A, W_B, stage=1)
        if stage == 1:
            Q = q

        stress, viscousStress = self.detStress.stress_avg(Q, h, dt)

        fX, fY = self.hyperbolicTVD(Q, dt)

        if bool(self.numerics['Fluctuating']) is True:
            stochastic_stress = self.stochStress.full_tensor(W_A, W_B, h, dt, stage)
            sX, sY = self.stochasticFlux(stochastic_stress, h, dt)
        else:
            stochastic_stress = np.zeros([6, q.Nx, q.Ny])
            sX = sY = np.zeros_like(fX)

        if bool(self.numerics['Rey']) is False:
            dX, dY = self.diffusiveCD(Q, viscousStress, dt)
        else:
            dX = dY = np.zeros_like(fX)

        src = self.getSource(viscousStress, stochastic_stress, Q, h, dt)
        tmp = Q.field - fX - fY + dX + dY + sX + sY + src

        if stage == 3:
            Q.field = 1 / 3 * q.field + 2 / 3 * tmp

        if stage == 2:
            Q.field = 3 / 4 * q.field + 1 / 4 * tmp

        if stage == 1:
            Q.field = tmp

        return Q
