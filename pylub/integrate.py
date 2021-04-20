"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
from mpi4py import MPI

from pylub.field import VectorField
from pylub.stress import SymStressField2D, SymStressField3D
from pylub.geometry import GapHeight
from pylub.eos import EquationOfState


class ConservedField(VectorField):

    def __init__(self, disc, BC, geometry, material, numerics, q_init=None, t_init=None):

        super().__init__(disc)

        self.BC = BC
        self.material = material

        self.stokes = bool(numerics["stokes"])
        self.numFlux = str(numerics["numFlux"])
        self.adaptive = bool(numerics["adaptive"])

        self.x0 = np.array(list(self.BC["x0"]))
        self.x1 = np.array(list(self.BC["x1"]))
        self.y0 = np.array(list(self.BC["y0"]))
        self.y1 = np.array(list(self.BC["y1"]))

        self.dx = disc["dx"]
        self.dy = disc["dy"]

        if "D" in self.x0:
            self.rhox0 = self.BC["rhox0"]
        if "D" in self.x1:
            self.rhox1 = self.BC["rhox1"]
        if "D" in self.y0:
            self.rhoy0 = self.BC["rhoy0"]
        if "D" in self.y1:
            self.rhoy1 = self.BC["rhoy1"]

        if q_init is not None:
            self.field[:, 1:-1, 1:-1] = q_init[:, self.without_ghost]
            self.fill_ghost_buffer()
        else:
            self.field[0] = float(material['rho0'])

        self.height = GapHeight(disc, geometry)

        if t_init is not None:
            self.time = t_init[0]
            self.dt = t_init[1]
        else:
            self.time = 0.
            self.dt = numerics["dt"]

        self.eps = 1.

        if self.adaptive:
            self.C = numerics["C"]

        self.viscous_stress = SymStressField2D(disc, geometry, material)
        self.upper_stress = SymStressField3D(disc, geometry, material)
        self.lower_stress = SymStressField3D(disc, geometry, material)

        self.neighbors = self.get_neighbors()

    @property
    def mass(self):
        local_mass = np.sum(self.inner[0] * self.height.inner[0] * self.dx * self.dy)
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_mass, recvbuf, op=MPI.SUM)

        return recvbuf[0]

    @property
    def vSound(self):
        local_vSound = EquationOfState(self.material).soundSpeed(self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vSound, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def vmax(self):
        local_vmax = np.amax(np.sqrt(self.inner[1]**2 + self.inner[2]**2) / self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vmax, recvbuf, op=MPI.MAX)

        return recvbuf[0]

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

        # Richtmyer two-step Lax-Wendroff
        elif self.numFlux == "LW":
            try:
                self.richtmyer()
            except NotImplementedError:
                print("Lax-Wendroff scheme not implemented. Exit!")
                quit()

        # 3rd order Runge-Kutta
        elif self.numFlux == "RK3":
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

        q0 = self.field.copy()

        for dir in directions:

            fX, fY = self.predictor_corrector(self.field, self.height.field, dir)
            src = self.getSource(self.field, self.height.field)

            self.field = self.field - fX - fY + self.dt * src

        self.field = 0.5 * (self.field + q0)
        self.fill_ghost_buffer()

        self.post_integrate(q0)

    def richtmyer(self):

        q0 = self.field.copy()

        # Step 1
        self.lax_step()
        # Step 2
        self.leap_frog(q0)

        self.fill_ghost_buffer()

        self.post_integrate(q0)

    def runge_kutta3(self):
        raise NotImplementedError

    def post_integrate(self, q0):
        self.time += self.dt

        if self.adaptive:
            self.dt = self.C * min(self.dx, self.dy) / (self.vmax + self.vSound)
        else:
            self.C = self.dt * (self.vmax + self.vSound) / min(self.dx, self.dy)

        local_diff = np.sum((self.inner[0] - q0[0, 1:-1, 1:-1])**2)
        local_denom = np.sum(q0[0, 1:-1, 1:-1]**2)

        diff = np.empty(1)
        denom = np.empty(1)

        self.comm.Allreduce(local_diff, diff, op=MPI.SUM)
        self.comm.Allreduce(local_denom, denom, op=MPI.SUM)

        self.eps = np.sqrt(diff[0] / denom[0]) / self.C

    def fill_ghost_buffer(self):

        (ls, ld), (rs, rd), (bs, bd), (ts, td) = self.neighbors

        # Send to left, receive from right
        recvbuf = np.ascontiguousarray(self.field[:, -1, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, 1, :]), ld, recvbuf=recvbuf, source=ls)

        if ls >= 0:
            self.field[:, -1, :] = recvbuf
        else:
            self.field[self.x1 == "D", -1, :] = 2. * self.rhox1 - self.field[self.x1 == "D", -2, :]
            self.field[self.x1 == "N", -1, :] = self.field[self.x1 == "N", -2, :]

        # Send to right, receive from left
        recvbuf = np.ascontiguousarray(self.field[:, 0, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, -2, :]), rd, recvbuf=recvbuf, source=rs)

        if rs >= 0:
            self.field[:, 0, :] = recvbuf
        else:
            self.field[self.x0 == "D", 0, :] = 2. * self.rhox0 - self.field[self.x0 == "D", 1, :]
            self.field[self.x0 == "N", 0, :] = self.field[self.x0 == "N", 1, :]

        # Send to bottom, receive from top
        recvbuf = np.ascontiguousarray(self.field[:, :, -1])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, 1]), bd, recvbuf=recvbuf, source=bs)

        if bs >= 0:
            self.field[:, :, -1] = recvbuf
        else:
            self.field[self.y1 == "D", :, -1] = 2. * self.rhoy1 - self.field[self.y1 == "D", :, -2]
            self.field[self.y1 == "N", :, -1] = self.field[self.y1 == "N", :, -2]

        # Send to top, receive from bottom
        recvbuf = np.ascontiguousarray(self.field[:, :, 0])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, -2]), td, recvbuf=recvbuf, source=ts)

        if ts >= 0:
            self.field[:, :, 0] = recvbuf
        else:
            self.field[self.y0 == "D", :, 0] = 2. * self.rhoy0 - self.field[self.y0 == "D", :, 1]
            self.field[self.y0 == "N", :, 0] = self.field[self.y0 == "N", :, 1]

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

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx) * self.dt / self.dx
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy) * self.dt / self.dy

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

        self.field = self.verticeNE - self.dt / (2. * self.dx) * (FE - FW) - self.dt / (2. * self.dy) * (FN - FS) + src * self.dt / 2.

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

        self.field = q0 - self.dt / self.dx * (FE - FW) - self.dt / self.dy * (FN - FS) + src * self.dt

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

    def diffusiveCD(self):

        Dx = self.diffusiveFlux(1)
        Dy = self.diffusiveFlux(2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return self.dt / (2 * self.dx) * flux_x, self.dt / (2 * self.dy) * flux_y
