import numpy as np

from pylub.field import VectorField
from pylub.stress import SymStressField2D, SymStressField3D
from pylub.eos import EquationOfState
from pylub.geometry import GapHeight


class ConservedField(VectorField):

    def __init__(self, disc, BC, geometry, material, numerics, q_init=None, t_init=None, grid=True):

        super().__init__(disc, grid)

        self.BC = BC
        self.material = material

        self.stokes = bool(numerics["stokes"])
        self.numFlux = str(numerics["numFlux"])
        self.adaptive = bool(numerics["adaptive"])

        self.x0 = np.array(list(self.BC["x0"]))
        self.x1 = np.array(list(self.BC["x1"]))
        self.y0 = np.array(list(self.BC["y0"]))
        self.y1 = np.array(list(self.BC["y1"]))

        if q_init is not None:
            self._field[:, 1:-1, 1:-1] = q_init
            self.fill_ghost_cell()
        else:
            self._field[0] = float(material['rho0'])

        self.height = GapHeight(disc, geometry)

        if t_init is not None:
            self._time = t_init[0]
            self._dt = t_init[1]
        else:
            self._time = 0.
            self._dt = float(numerics["dt"])

        self._eps = 1.

        if self.adaptive:
            self.C = float(numerics["C"])

        self.viscous_stress = SymStressField2D(disc, geometry, material, grid=False)
        self.upper_stress = SymStressField3D(disc, geometry, material, grid=False)
        self.lower_stress = SymStressField3D(disc, geometry, material, grid=False)

    @property
    def mass(self):
        return np.sum(self._field[0] * self.height.field[0] * self.dx * self.dy)

    @property
    def vSound(self):
        return EquationOfState(self.material).soundSpeed(self._field[0])

    @property
    def vmax(self):
        return self.vSound + np.amax(np.sqrt(self._field[1]**2 + self._field[2]**2) / self._field[0])

    @property
    def time(self):
        return self._time

    @property
    def dt(self):
        return self._dt

    @property
    def eps(self):
        return self._eps

    def update(self, i):

        # MacCormack forward backward
        if self.numFlux == "MC" or self.numFlux == "MC_fb":
            try:
                self.mac_cormack(0)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Mac Cormack backward forward
        elif self.numFlux == "MC_bf":
            try:
                self.mac_cormack(1)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # MacCormack alternating
        elif self.numFlux == "MC_alt":
            try:
                self.mac_cormack(i % 2)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()
            try:
                self.richtmyer()
            except NotImplementedError:
                print("Lax-Wendroff scheme not implemented. Exit!")
                quit()
        if self.numFlux == "RK3":
            try:
                self.runge_kutta3()
            except NotImplementedError:
                print("Runge-Kutta 3 scheme not implemented. Exit!")
                quit()

    def mac_cormack(self, switch):

        if switch == 0:
            directions = [-1, 1]
        elif switch == 1:
            directions = [1, -1]

        for dir in [-1, 1]:
            self.viscous_stress.set(self._field, self.height.field)
            self.upper_stress.set(self._field, self.height.field, "top")
            self.lower_stress.set(self._field, self.height.field, "bottom")

            p = EquationOfState(self.material).isoT_pressure(self._field[0])
        for dir in directions:

            fX, fY = self.hyperbolicFW_BW(p, dir)
            dX, dY = self.diffusiveCD()
            src = self.getSource(self._field, self.height.field)

            self._field = self._field - fX - fY + dX + dY + self._dt * src
            self.fill_ghost_cell()

        self._field = 0.5 * (self._field + q0)

        self.post_integrate(q0)

    def richtmyer(self):
        raise NotImplementedError

    def runge_kutta3(self):
        raise NotImplementedError

    def post_integrate(self, q0):
        self._time += self._dt

        if self.adaptive:
            self._dt = self.C * min(self.dx, self.dy) / self.vmax

        self._eps = np.linalg.norm(self._field[0] - q0[0]) / np.linalg.norm(q0[0]) / self.C

    def fill_ghost_cell(self):
        self.periodic()
        self.dirichlet()
        self.neumann()

    def periodic(self):

        self._field[self.x0 == "P", 0, :] = self._field[self.x0 == "P", -2, :]
        self._field[self.x1 == "P", -1, :] = self._field[self.x1 == "P", 1, :]
        self._field[self.y0 == "P", :, 0] = self._field[self.y0 == "P", :, -2]
        self._field[self.y1 == "P", :, -1] = self._field[self.y1 == "P", :, 1]

    def dirichlet(self):

        rhox0 = rhox1 = rhoy0 = rhoy1 = float(self.material["rho0"])

        if "D" in self.x0 and "px0" in self.BC.keys():
            px0 = float(self.BC["px0"])
            rhox0 = EquationOfState(self.material).isoT_density(px0)

        if "D" in self.x1 and "px1" in self.BC.keys():
            px1 = float(self.BC["px1"])
            rhox1 = EquationOfState(self.material).isoT_density(px1)

        if "D" in self.y0 and "py0" in self.BC.keys():
            py0 = float(self.BC["py0"])
            rhoy0 = EquationOfState(self.material).isoT_density(py0)

        if "D" in self.y1 and "py1" in self.BC.keys():
            py1 = float(self.BC["py1"])
            rhoy1 = EquationOfState(self.material).isoT_density(py1)

        self._field[self.x0 == "D", 0, :] = 2. * rhox0 - self._field[self.x0 == "D", 1, :]
        self._field[self.x1 == "D", -1, :] = 2. * rhox1 - self._field[self.x1 == "D", -2, :]
        self._field[self.y0 == "D", :, 0] = 2. * rhoy0 - self._field[self.y0 == "D", :, 1]
        self._field[self.y1 == "D", :, -1] = 2. * rhoy1 - self._field[self.y0 == "D", :, -2]

    def neumann(self):
        self._field[self.x0 == "N", 0, :] = self._field[self.x0 == "N", 1, :]
        self._field[self.x1 == "N", -1, :] = self._field[self.x1 == "N", -2, :]
        self._field[self.y0 == "N", :, 0] = self._field[self.y0 == "N", :, 1]
        self._field[self.y1 == "N", :, -1] = self._field[self.y0 == "N", :, -2]

    def getSource(self, q, h):

        out = np.zeros_like(q)

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = (-q[1] * h[1] - q[2] * h[2]) / h[0]

        out[1] = ((q[1] * q[1] / q[0] + self.viscous_stress.field[0] - self.upper_stress.field[0]) * h[1] +
                  (q[1] * q[2] / q[0] + self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[2] +
                  self.upper_stress.field[4] - self.lower_stress.field[4]) / h[0]

        out[2] = ((q[2] * q[1] / q[0] + self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[1] +
                  (q[2] * q[2] / q[0] + self.viscous_stress.field[1] - self.upper_stress.field[1]) * h[2] +
                  self.upper_stress.field[3] - self.lower_stress.field[3]) / h[0]

        # origin center
        # out[0] = -q[1] * h[1] / h[0] - q[2] * h[2] / h[0]
        #
        # out[1] = ((q[1] * q[1] / q[0] + self.viscous_stress.field[0] - (self.upper_stress.field[0] + self.lower_stress.field[0]) / 2) * h[1]
        #           + (q[1] * q[2] / q[0] + self.viscous_stress.field[2] - (self.upper_stress.field[5] + self.lower_stress.field[5]) / 2) * h[2]
        #           + self.upper_stress.field[4] - self.lower_stress.field[4]) / h[0]
        #
        # out[2] = ((q[2] * q[1] / q[0] + self.viscous_stress.field[2] - (self.upper_stress.field[5] + self.lower_stress.field[5]) / 2) * h[1]
        #           + (q[2] * q[2] / q[0] + self.viscous_stress.field[1] - (self.upper_stress.field[1] + self.lower_stress.field[1]) / 2) * h[2]
        #           + self.upper_stress.field[3] - self.lower_stress.field[3]) / h[0]

        return out

    def hyperbolicFlux(self, f, p, ax):

        F = np.zeros_like(f)

        if ax == 1:
            if self.stokes:
                F[0] = f[1]
                F[1] = p
            else:
                F[0] = f[1]
                F[1] = f[1] * f[1] / f[0] + p
                F[2] = f[2] * f[1] / f[0]

        elif ax == 2:
            if self.stokes:
                F[0] = f[2]
                F[2] = p
            else:
                F[0] = f[2]
                F[1] = f[1] * f[2] / f[0]
                F[2] = f[2] * f[2] / f[0] + p

        return F

    def hyperbolicFW_BW(self, p, dir):

        Fx = self.hyperbolicFlux(self._field, p, 1)
        Fy = self.hyperbolicFlux(self._field, p, 2)

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx)
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy)

        return self._dt / self.dx * flux_x, self._dt / self.dy * flux_y

    def diffusiveFlux(self, ax):

        D = np.zeros_like(self._field)
        if ax == 1:
            D[1] = self.viscous_stress.field[0]
            D[2] = self.viscous_stress.field[2]

        elif ax == 2:
            D[1] = self.viscous_stress.field[2]
            D[2] = self.viscous_stress.field[1]

        return D

    def diffusiveCD(self):

        Dx = self.diffusiveFlux(1)
        Dy = self.diffusiveFlux(2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return self._dt / (2 * self.dx) * flux_x, self._dt / (2 * self.dy) * flux_y
