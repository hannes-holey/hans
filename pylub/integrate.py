import numpy as np
from mpi4py import MPI

from pylub.field import VectorField
from pylub.stress import SymStressField2D, SymStressField3D, SymStochStressField3D
from pylub.geometry import GapHeight
from pylub.eos import EquationOfState


class ConservedField(VectorField):

    def __init__(self, disc, BC, geometry, material, numerics, q_init=None, t_init=None):

        super().__init__(disc)

        self.BC = BC
        self.material = material

        self.stokes = bool(numerics["stokes"])
        self.integrator = str(numerics["integrator"])
        self.adaptive = bool(numerics["adaptive"])
        self.fluctuating = bool(numerics["fluctuating"])

        self.x0 = np.array(list(self.BC["x0"]))
        self.x1 = np.array(list(self.BC["x1"]))
        self.y0 = np.array(list(self.BC["y0"]))
        self.y1 = np.array(list(self.BC["y1"]))

        if "D" in self.x0:
            self.rhox0 = self.BC["rhox0"]
        if "D" in self.x1:
            self.rhox1 = self.BC["rhox1"]
        if "D" in self.y0:
            self.rhoy0 = self.BC["rhoy0"]
        if "D" in self.y1:
            self.rhoy1 = self.BC["rhoy1"]

        if q_init is not None:
            self._field[:, 1:-1, 1:-1] = q_init[:, self.wo_ghost_x, self.wo_ghost_y]
            self.fill_ghost_buffer()
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

        self.viscous_stress = SymStressField2D(disc, geometry, material)
        self.upper_stress = SymStressField3D(disc, geometry, material)
        self.lower_stress = SymStressField3D(disc, geometry, material)

        if self.fluctuating:
            self.stoch_stress = SymStochStressField3D(disc, geometry, material, grid=False)

    @property
    def mass(self):
        local_mass = np.sum(self.inner[0] * self.height.inner[0] * self.dx * self.dy)
        return self.comm.allreduce(local_mass, op=MPI.SUM)

    @property
    def vSound(self):
        local_vSound = EquationOfState(self.material).soundSpeed(self.inner[0])
        return self.comm.allreduce(local_vSound, op=MPI.MAX)

    @property
    def vmax(self):
        local_vmax = np.amax(np.sqrt(self.inner[1]**2 + self.inner[2]**2) / self.inner[0])
        return self.comm.allreduce(local_vmax, op=MPI.MAX)

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
        if self.integrator == "MC" or self.integrator == "MC_fb":
            try:
                self.mac_cormack(0)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Mac Cormack backward forward
        elif self.integrator == "MC_bf":
            try:
                self.mac_cormack(1)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # MacCormack alternating
        elif self.integrator == "MC_alt":
            try:
                self.mac_cormack(i % 2)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Richtmyer two-step Lax-Wendroff
        elif self.integrator == "LW":
            try:
                self.richtmyer()
            except NotImplementedError:
                print("Lax-Wendroff scheme not implemented. Exit!")
                quit()

        # 3rd order Runge-Kutta
        elif self.integrator == "RK3":
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

        q0 = self._field.copy()

        for dir in directions:

            fX, fY = self.predictor_corrector(self._field, self.height.field, dir)
            src = self.getSource(self._field, self.height.field)

            self._field = self._field - fX - fY + self._dt * src

        self._field = 0.5 * (self._field + q0)
        self.fill_ghost_buffer()

        self.post_integrate(q0)

    def richtmyer(self):

        q0 = self._field.copy()

        # Step 1
        self.lax_step()
        # Step 2
        self.leap_frog(q0)

        self.fill_ghost_buffer()

        self.post_integrate(q0)

    def runge_kutta3(self):

        W_A = np.random.normal(size=(3, 3, self.Nx+2, self.Ny+2))
        W_B = np.random.normal(size=(3, 3, self.Nx+2, self.Ny+2))

        q0 = self._field.copy()

        for stage in np.arange(1, 4):
            self.viscous_stress.set(self._field, self.height.field)
            self.upper_stress.set(self._field, self.height.field, "top")
            self.lower_stress.set(self._field, self.height.field, "bottom")
            self.stoch_stress.set(W_A, W_B, self.height.field, self._dt, stage)

            dX, dY = self.diffusiveCD()
            fX, fY = self.hyperbolicTVD()
            sX, sY = self.stochasticFlux()
            src = self.getSource(self._field, self.height.field)

            tmp = self._field - fX - fY + dX + dY + sX + sY + self._dt * src

            if stage == 1:
                self._field = tmp
            elif stage == 2:
                self._field = 3 / 4 * q0 + 1 / 4 * tmp
            elif stage == 3:
                self._field = 1 / 3 * q0 + 2 / 3 * tmp

            self.fill_ghost_buffer()

        self.post_integrate(q0)

    def post_integrate(self, q0):
        self._time += self._dt

        vSound = EquationOfState(self.material).soundSpeed(self.inner[0])
        vmax = np.amax(np.sqrt(self.inner[1]**2 + self.inner[2]**2) / self.inner[0])

        if self.adaptive:
            self._dt = self.C * min(self.dx, self.dy) / (vmax + vSound)

        self._dt = self.comm.allreduce(self._dt, op=MPI.MIN)

        diff = np.sum((self.inner[0] - q0[0, 1:-1, 1:-1])**2)
        denom = np.sum(q0[0, 1:-1, 1:-1]**2)

        diff = self.comm.allreduce(diff, op=MPI.SUM)
        denom = self.comm.allreduce(denom, op=MPI.SUM)

        self._eps = np.sqrt(diff / denom) / self.C

    def fill_ghost_buffer(self):

        # Send to left, receive from right
        recvbuf = np.ascontiguousarray(self._field[:, -1, :])
        self.comm.Sendrecv(np.ascontiguousarray(self._field[:, 1, :]), self.ld, recvbuf=recvbuf, source=self.ls)

        if self.ls >= 0:
            self._field[:, -1, :] = recvbuf
        else:
            self._field[self.x1 == "D", -1, :] = 2. * self.rhox1 - self._field[self.x1 == "D", -2, :]
            self._field[self.x1 == "N", -1, :] = self._field[self.x1 == "N", -2, :]

        # Send to right, receive from left
        recvbuf = np.ascontiguousarray(self._field[:, 0, :])
        self.comm.Sendrecv(np.ascontiguousarray(self._field[:, -2, :]), self.rd, recvbuf=recvbuf, source=self.rs)

        if self.rs >= 0:
            self._field[:, 0, :] = recvbuf
        else:
            self._field[self.x0 == "D", 0, :] = 2. * self.rhox0 - self._field[self.x0 == "D", 1, :]
            self._field[self.x0 == "N", 0, :] = self._field[self.x0 == "N", 1, :]

        # Send to bottom, receive from top
        recvbuf = np.ascontiguousarray(self._field[:, :, -1])
        self.comm.Sendrecv(np.ascontiguousarray(self._field[:, :, 1]), self.bd, recvbuf=recvbuf, source=self.bs)

        if self.bs >= 0:
            self._field[:, :, -1] = recvbuf
        else:
            self._field[self.y1 == "D", :, -1] = 2. * self.rhoy1 - self._field[self.y1 == "D", :, -2]
            self._field[self.y1 == "N", :, -1] = self._field[self.y1 == "N", :, -2]

        # Send to top, receive from bottom
        recvbuf = np.ascontiguousarray(self._field[:, :, 0])
        self.comm.Sendrecv(np.ascontiguousarray(self._field[:, :, -2]), self.td, recvbuf=recvbuf, source=self.ts)

        if self.ts >= 0:
            self._field[:, :, 0] = recvbuf
        else:
            self._field[self.y0 == "D", :, 0] = 2. * self.rhoy0 - self._field[self.y0 == "D", :, 1]
            self._field[self.y0 == "N", :, 0] = self._field[self.y0 == "N", :, 1]

    def getSource(self, q, h):

        self.viscous_stress.set(q, h)
        self.upper_stress.set(q, h, "top")
        self.lower_stress.set(q, h, "bottom")

        out = np.zeros_like(q)

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = (-q[1] * h[1] - q[2] * h[2]) / h[0]

        if self.stokes:
            out[1] = ((q[1] * q[1] / q[0] + self.viscous_stress.field[0] - self.upper_stress.field[0]) * h[1] +
                      (q[1] * q[2] / q[0] + self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[2] +
                      self.upper_stress.field[4] - self.lower_stress.field[4]) / h[0]

            out[2] = ((q[2] * q[1] / q[0] + self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[1] +
                      (q[2] * q[2] / q[0] + self.viscous_stress.field[1] - self.upper_stress.field[1]) * h[2] +
                      self.upper_stress.field[3] - self.lower_stress.field[3]) / h[0]
        else:
            out[1] = ((self.viscous_stress.field[0] - self.upper_stress.field[0]) * h[1] +
                      (self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[2] +
                      self.upper_stress.field[4] - self.lower_stress.field[4]) / h[0]

            out[2] = ((self.viscous_stress.field[2] - self.upper_stress.field[5]) * h[1] +
                      (self.viscous_stress.field[1] - self.upper_stress.field[1]) * h[2] +
                      self.upper_stress.field[3] - self.lower_stress.field[3]) / h[0]

        return out

    def predictor_corrector(self, f, h, dir):

        Fx = self.hyperbolicFlux(f, 1) - self.diffusiveFlux(f, h, 1)
        Fy = self.hyperbolicFlux(f, 2) - self.diffusiveFlux(f, h, 2)

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx) * self._dt / self.dx
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy) * self._dt / self.dy

        return flux_x, flux_y

    def lax_step(self):

        QS = self.edgeE
        QW = self.edgeN
        QN = np.roll(QS, -1, axis=2)
        QE = np.roll(QW, -1, axis=1)

        hS = self.height.edgeE
        hW = self.height.edgeN
        hN = np.roll(hS, -1, axis=2)
        hE = np.roll(hW, -1, axis=1)

        FS = self.hyperbolicFlux(QS, 1) - self.diffusiveFlux(QS, hS, 1)
        FW = self.hyperbolicFlux(QW, 2) - self.diffusiveFlux(QW, hW, 2)
        FN = self.hyperbolicFlux(QN, 1) - self.diffusiveFlux(QN, hN, 1)
        FE = self.hyperbolicFlux(QE, 2) - self.diffusiveFlux(QE, hE, 2)

        src = self.getSource(self.verticeNE, self.height.verticeNE)

        self._field = self.verticeNE - self._dt / (2. * self.dx) * (FE - FW) - self._dt / (2. * self.dy) * (FN - FS) + src * self._dt / 2.

    def leap_frog(self, q0):

        QE = self.edgeS
        QN = self.edgeW
        QW = np.roll(QE, 1, axis=1)
        QS = np.roll(QN, 1, axis=2)

        FE = self.hyperbolicFlux(QE, 1) - self.diffusiveFlux(QE, self.height.edgeE, 1)
        FN = self.hyperbolicFlux(QN, 2) - self.diffusiveFlux(QN, self.height.edgeN, 2)
        FW = self.hyperbolicFlux(QW, 1) - self.diffusiveFlux(QW, self.height.edgeW, 1)
        FS = self.hyperbolicFlux(QS, 2) - self.diffusiveFlux(QS, self.height.edgeS, 2)

        src = self.getSource(self.verticeSW, self.height.field)

        self._field = q0 - self._dt / self.dx * (FE - FW) - self._dt / self.dy * (FN - FS) + src * self.dt

    def hyperbolicFlux(self, f, ax):

        p = EquationOfState(self.material).isoT_pressure(f[0])

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

    def diffusiveFlux(self, f, h, ax):

        self.viscous_stress.set(f, h)

        D = np.zeros_like(f)
        if ax == 1:
            D[1] = self.viscous_stress.field[0]
            D[2] = self.viscous_stress.field[2]

        elif ax == 2:
            D[1] = self.viscous_stress.field[2]
            D[2] = self.viscous_stress.field[1]

        return D

    def cubicInterpolation(self, q, ax):

        # weights proposed by Bell et al., PRE 76 (2007)
        # a1 = (np.sqrt(7) + 1) / 4
        # a2 = (np.sqrt(7) - 1) / 4

        # weights from piecewise parabolic method (PPM)
        # Collela and Woodward, J. Comp. Phys. 54 (1984)
        a1 = 7 / 12
        a2 = 1 / 12

        Q = a1 * (q + np.roll(q, -1, axis=ax)) - a2 * (np.roll(q, 1, axis=ax) + np.roll(q, -2, axis=ax))

        return Q

    def hyperbolicTVD(self):

        Qx = self.cubicInterpolation(self._field, 1)
        Px = EquationOfState(self.material).isoT_pressure(Qx[0])

        Qy = self.cubicInterpolation(self._field, 2)
        Py = EquationOfState(self.material).isoT_pressure(Qy[0])

        Fx = self.hyperbolicFlux(Qx, Px, 1)
        Fy = self.hyperbolicFlux(Qy, Py, 2)

        flux_x = Fx - np.roll(Fx, 1, axis=1)
        flux_y = Fy - np.roll(Fy, 1, axis=2)

        return self._dt / self.dx * flux_x, self._dt / self.dy * flux_y

    def hyperbolicCD(self, q, p, dt, ax):

        F = self.hyperbolicFlux(q.field, p, ax)

        flux = np.roll(F, -1, axis=ax) - np.roll(F, 1, axis=ax)

        if ax == 1:
            dx = q.dx

        elif ax == 2:
            dx = q.dy

        return dt / (2 * dx) * flux

    def diffusiveCD(self):

        Dx = self.diffusiveFlux(1)
        Dy = self.diffusiveFlux(2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return self._dt / (2 * self.dx) * flux_x, self._dt / (2 * self.dy) * flux_y

    def stochasticFlux(self):

        Sx_E = np.zeros([3, self.Nx+2, self.Ny+2])
        Sx_W = np.zeros([3, self.Nx+2, self.Ny+2])
        Sy_N = np.zeros([3, self.Nx+2, self.Ny+2])
        Sy_S = np.zeros([3, self.Nx+2, self.Ny+2])

        corrX = self.dx / self.height.field[0]
        corrY = self.dy / self.height.field[0]

        # corrX = corrY = 1.

        Sx_E[1] = self.stoch_stress.field[0]
        Sx_E[2] = self.stoch_stress.field[5]
        Sx_W[1] = np.roll(self.stoch_stress.field[0], 1, axis=0)
        Sx_W[2] = np.roll(self.stoch_stress.field[5], 1, axis=0)

        Sy_N[1] = self.stoch_stress.field[5]
        Sy_N[2] = self.stoch_stress.field[1]
        Sy_S[1] = np.roll(self.stoch_stress.field[5], 1, axis=1)
        Sy_S[2] = np.roll(self.stoch_stress.field[1], 1, axis=1)

        return self._dt / self.dx * (Sx_E - Sx_W) * corrX, self._dt / self.dy * (Sy_N - Sy_S) * corrY
