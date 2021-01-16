import numpy as np

from pylub.field import VectorField
from pylub.stress import SymmetricStressField, StressField
from pylub.eos import EquationOfState


class ConservedField(VectorField):

    def __init__(self, disc, BC, geometry, material, grid=True):

        super().__init__(disc, grid)

        self.disc = disc
        self.BC = BC
        self.geometry = geometry
        self.material = material

        self.viscous_stress = SymmetricStressField(disc, geometry, material, grid=False)
        self.upper_stress = StressField(disc, geometry, material, grid=False)
        self.lower_stress = StressField(disc, geometry, material, grid=False)

    def update(self, h, dt, corrector=True):

        if corrector:
            q = self.field.copy()
            self.update(h, dt, corrector=False)
            dir = 1                # backwards difference
        else:
            dir = -1                 # forwards difference

        self.viscous_stress.set(self.field, h)
        self.upper_stress.set(self.field, h, "top")
        self.lower_stress.set(self.field, h, "bottom")

        p = EquationOfState(self.material).isoT_pressure(self.field[0])

        fX, fY = self.hyperbolicFW_BW(p, dt, dir)
        dX, dY = self.diffusiveCD(dt)

        src = self.getSource(h)

        self.field = self.field - fX - fY + dX + dY + dt * src

        if corrector:
            self.field = 0.5 * (self.field + q)

        self.fill_ghost_cell()

    def fill_ghost_cell(self):
        self.periodic()
        self.dirichlet()
        self.neumann()

    def periodic(self):
        x0 = np.array(list(self.BC["x0"]))
        y0 = np.array(list(self.BC["y0"]))

        self.field[x0 == "P", 0, :] = self.field[x0 == "P", -2, :]
        self.field[x0 == "P", -1, :] = self.field[x0 == "P", 1, :]
        self.field[y0 == "P", :, 0] = self.field[y0 == "P", :, -2]
        self.field[y0 == "P", :, -1] = self.field[y0 == "P", :, 1]

    def dirichlet(self):

        x0 = np.array(list(self.BC["x0"]))
        x1 = np.array(list(self.BC["x1"]))
        y0 = np.array(list(self.BC["y0"]))
        y1 = np.array(list(self.BC["y1"]))

        rhox0 = rhox1 = rhoy0 = rhoy1 = float(self.material["rho0"])

        if "D" in x0 and "px0" in self.BC.keys():
            px0 = float(self.BC["px0"])
            rhox0 = EquationOfState(self.material).isoT_density(px0)

        if "D" in x1 and "px1" in self.BC.keys():
            px1 = float(self.BC["px1"])
            rhox1 = EquationOfState(self.material).isoT_density(px1)

        if "D" in y0 and "py0" in self.BC.keys():
            py0 = float(self.BC["py0"])
            rhoy0 = EquationOfState(self.material).isoT_density(py0)

        if "D" in y1 and "py1" in self.BC.keys():
            py1 = float(self.BC["py1"])
            rhoy1 = EquationOfState(self.material).isoT_density(py1)

        self.field[x0 == "D", 0, :] = 2. * rhox0 - self.field[x0 == "D", 1, :]
        self.field[x1 == "D", -1, :] = 2. * rhox1 - self.field[x0 == "D", -2, :]
        self.field[y0 == "D", :, 0] = 2. * rhoy0 - self.field[y0 == "D", :, 1]
        self.field[y1 == "D", :, -1] = 2. * rhoy1 - self.field[y0 == "D", :, -2]

    def neumann(self):
        x0 = np.array(list(self.BC["x0"]))
        x1 = np.array(list(self.BC["x1"]))
        y0 = np.array(list(self.BC["y0"]))
        y1 = np.array(list(self.BC["y1"]))

        self.field[x0 == "N", 0, :] = self.field[x0 == "N", 1, :]
        self.field[x1 == "N", -1, :] = self.field[x0 == "N", -2, :]
        self.field[y0 == "N", :, 0] = self.field[y0 == "N", :, 1]
        self.field[y1 == "N", :, -1] = self.field[y0 == "N", :, -2]

    def getSource(self, h):

        out = np.zeros_like(self.field)

        stress = self.viscous_stress.field
        stress_top = self.upper_stress.field
        stress_bot = self.lower_stress.field

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = -self.field[1] * h[1] / h[0] - self.field[2] * h[2] / h[0]

        out[1] = ((self.field[1] * self.field[1] / self.field[0] + stress[0] - stress_top[0]) * h[1]
                  + (self.field[1] * self.field[2] / self.field[0] + stress[2] - stress_top[5]) * h[2]
                  + stress_top[4] - stress_bot[4]) / h[0]

        out[2] = ((self.field[2] * self.field[1] / self.field[0] + stress[2] - stress_top[5]) * h[1]
                  + (self.field[2] * self.field[2] / self.field[0] + stress[1] - stress_top[1]) * h[2]
                  + stress_top[3] - stress_bot[3]) / h[0]

        # origin center
        # out[0] = -self.field[1] * h[1] / h[0] - self.field[2] * h[2] / h[0]
        #
        # out[1] = ((self.field[1] * self.field[1] / self.field[0] + stress.field[0] - (stress_wall_top.field[0] + stress_wall_bot.field[0]) / 2) * h[1]
        #           + (self.field[1] * self.field[2] / self.field[0] + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * h[2]
        #           + stress_wall_top.field[4] - stress_wall_bot.field[4]) / h[0]
        #
        # out[2] = ((self.field[2] * self.field[1] / self.field[0] + stress.field[2] - (stress_wall_top.field[5] + stress_wall_bot.field[5]) / 2) * h[1]
        #           + (self.field[2] * self.field[2] / self.field[0] + stress.field[1] - (stress_wall_top.field[1] + stress_wall_bot.field[1]) / 2) * h[2]
        #           + stress_wall_top.field[3] - stress_wall_bot.field[3]) / h[0]

        return out

    def hyperbolicFlux(self, p, ax):

        F = np.zeros_like(self.field)

        if ax == 1:
            F[0] = self.field[1]
            F[1] = self.field[1] * self.field[1] / self.field[0] + p
            F[2] = self.field[2] * self.field[1] / self.field[0]

        elif ax == 2:
            F[0] = self.field[2]
            F[1] = self.field[1] * self.field[2] / self.field[0]
            F[2] = self.field[2] * self.field[2] / self.field[0] + p

        return F

    def hyperbolicFW_BW(self, p, dt, dir):

        Fx = self.hyperbolicFlux(p, 1)
        Fy = self.hyperbolicFlux(p, 2)

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx)
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy)

        return dt / self.dx * flux_x, dt / self.dy * flux_y

    def diffusiveFlux(self, ax):

        D = np.zeros_like(self.field)
        if ax == 1:
            D[1] = self.viscous_stress.field[0]
            D[2] = self.viscous_stress.field[2]

        elif ax == 2:
            D[1] = self.viscous_stress.field[2]
            D[2] = self.viscous_stress.field[1]

        return D

    def diffusiveCD(self, dt):

        Dx = self.diffusiveFlux(1)
        Dy = self.diffusiveFlux(2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return dt / (2 * self.dx) * flux_x, dt / (2 * self.dy) * flux_y

    # def LaxStep(self, q, h, visc_stress, dt, dir, ax):
    #
    #     delta = {1: q.dx, 2: q.dy}
    #     p = self.detStress.pressure(q.field)
    #     F = self.hyperbolicFlux(q.field, p, ax) + self.diffusiveFlux(visc_stress.field, ax)
    #
    #     Q = VectorField(self.disc, grid=False)
    #     Q.field = 0.5 * (q.field + np.roll(q.field, dir, axis=ax)) - dt / (2. * delta[ax]) * dir * (F - np.roll(F, dir, axis=ax))
    #
    #     return Q
    #
    # def fluxLW(self, q, h, visc_stress, dt, dir, ax):
    #
    #     Q = self.LaxStep(q, h, visc_stress, dt, dir, ax)
    #
    #     H = h.stagArray(dir, ax)
    #     _, viscous_stress = self.detStress.stress_avg(Q, H)
    #     p = self.detStress.pressure(Q.field)
    #     flux = self.hyperbolicFlux(Q.field, p, ax) + self.diffusiveFlux(viscous_stress.field, ax)
    #
    #     return flux

    # def Richtmyer(self, q, h, dt):
    #
    #     stress, viscous_stress = self.detStress.stress_avg(q, h)
    #
    #     fXE = self.fluxLW(q, h, viscous_stress, dt, -1, 1)
    #     fXW = self.fluxLW(q, h, viscous_stress, dt, 1, 1)
    #     fYN = self.fluxLW(q, h, viscous_stress, dt, -1, 2)
    #     fYS = self.fluxLW(q, h, viscous_stress, dt, 1, 2)
    #
    #     src = self.getSource(viscous_stress, q, h)
    #
    #     Q = VectorField(self.disc, grid=False)
    #
    #     Q.field = q.field - dt * ((fXE - fXW) / q.dy + (fYN - fYS) / q.dy - src)
    #
    #     Q = BoundaryCondition(self.disc, self.BC, self.material).fill_ghost_cell(Q)
    #
    #     return Q
