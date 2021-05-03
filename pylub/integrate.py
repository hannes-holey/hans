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
from pylub.material import Material


class ConservedField(VectorField):

    def __init__(self, disc, bc, geometry, material, numerics, q_init=None, t_init=None):
        """
        This class contains the field of conserved variable densities (rho, jx, jy),
        and the methods to update them. Derived from VectorField.

        Parameters
        ----------
        disc : dict
            Discretization parameters.
        bc : dict
            Boundary condition parameters.
        geometry : dict
            Geometry parameters.
        material : dict
            Material parameters.
        numerics : dict
            numerics parameters
        q_init : np.array
            Inital field in case of a restart (the default is None)
        t_init : tuple
            Initial time and time step in case of a restart (the default is None).

        """

        super().__init__(disc)

        self.disc = disc
        self.bc = bc
        self.material = material
        self.numerics = numerics

        # initialize gap height field
        self.height = GapHeight(disc, geometry)

        # intialize field and time
        if q_init is not None:
            self.field[:, 1:-1, 1:-1] = q_init[:, self.without_ghost[0], self.without_ghost[1]]
            self.fill_ghost_buffer()
            self.time = t_init[0]
            self.dt = t_init[1]
        else:
            self.field[0] = material['rho0']
            self.time = 0.
            self.dt = numerics["dt"]

        self.viscous_stress = SymStressField2D(disc, geometry, material)
        self.upper_stress = SymStressField3D(disc, geometry, material)
        self.lower_stress = SymStressField3D(disc, geometry, material)

    @property
    def mass(self):
        area = self.disc["dx"] * self.disc["dy"]
        local_mass = np.sum(self.inner[0] * self.height.inner[0] * area)
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_mass, recvbuf, op=MPI.SUM)

        return recvbuf[0]

    @property
    def vSound(self):
        local_vSound = Material(self.material).eos_sound_speed(self.inner[0])
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
        """
        Wrapper function for the explicit time update of the solution field.

        Parameters
        ----------
        i : int
            time step

        """

        integrator = self.numerics["numFlux"]

        # MacCormack forward backward
        if integrator == "MC" or integrator == "MC_fb":
            try:
                self.mac_cormack(0)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Mac Cormack backward forward
        elif integrator == "MC_bf":
            try:
                self.mac_cormack(1)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # MacCormack alternating
        elif integrator == "MC_alt":
            try:
                self.mac_cormack(i % 2)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Richtmyer two-step Lax-Wendroff
        elif integrator == "LW":
            try:
                self.richtmyer()
            except NotImplementedError:
                print("Lax-Wendroff scheme not implemented. Exit!")
                quit()

        # 3rd order Runge-Kutta
        elif integrator == "RK3":
            try:
                self.runge_kutta3()
            except NotImplementedError:
                print("Runge-Kutta 3 scheme not implemented. Exit!")
                quit()

    def mac_cormack(self, switch):
        """
        Explicit time update using MacCormack's predictor corrector scheme with source term.
        Field is updated in place.

        Parameters
        ----------
        switch : int
            Determine if predictor step is forwards (0) or backwards (1).

        """

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        if switch == 0:
            directions = [-1, 1]
        elif switch == 1:
            directions = [1, -1]

        q0 = self.field.copy()

        for dir in directions:

            fX, fY = self.predictor_corrector(self.field, self.height.field, dir)
            src = self.get_source(self.field, self.height.field)

            self.field = self.field - self.dt * (fX / dx + fY / dy - src)
            self.fill_ghost_buffer()

        self.field = 0.5 * (self.field + q0)

        self.post_integrate(q0)

    def richtmyer(self):
        """
        Explicit time update using Richtmyer's two-step scheme with source term.
        Field is updated in place.

        """

        q0 = self.field.copy()

        # Step 1
        self.lax_step()
        self.fill_ghost_buffer()

        # Step 2
        self.leap_frog(q0)
        self.fill_ghost_buffer()

        self.post_integrate(q0)

    def runge_kutta3(self):
        raise NotImplementedError

    def post_integrate(self, q0):
        """
        Compute relative change of solution after time update.
        Update time, and, if adaptive time-stepping is used, also the time step.

        Parameters
        ----------
        q0 : np.array
            Copy of the solution field from the previous step

        """
        min_dist = min(self.disc["dx"], self.disc["dy"])

        self.time += self.dt

        if bool(self.numerics["adaptive"]):
            CFL = self.numerics["C"]
            self.dt = CFL * min_dist / (self.vmax + self.vSound)
        else:
            CFL = self.dt * (self.vmax + self.vSound) / min_dist

        local_diff = np.sum((self.inner[0] - q0[0, 1:-1, 1:-1])**2)
        local_denom = np.sum(q0[0, 1:-1, 1:-1]**2)

        diff = np.empty(1)
        denom = np.empty(1)

        self.comm.Allreduce(local_diff, diff, op=MPI.SUM)
        self.comm.Allreduce(local_denom, denom, op=MPI.SUM)

        self.eps = np.sqrt(diff[0] / denom[0]) / CFL

    def fill_ghost_buffer(self):
        """
        Communicate results from adjacent domains to ghost buffers.
        Fill ghost cells at the boundary according to the applied BCs.

        """

        x0 = self.bc["x0"]
        x1 = self.bc["x1"]
        y0 = self.bc["y0"]
        y1 = self.bc["y1"]

        (ls, ld), (rs, rd), (bs, bd), (ts, td) = self.get_neighbors()

        # Send to left, receive from right
        recvbuf = np.ascontiguousarray(self.field[:, -1, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, 1, :]), ld, recvbuf=recvbuf, source=ls)

        if ls >= 0:
            self.field[:, -1, :] = recvbuf
        else:
            self.field[x1 == "D", -1, :] = 2. * self.bc["rhox1"] - self.field[x1 == "D", -2, :]
            self.field[x1 == "N", -1, :] = self.field[x1 == "N", -2, :]

        # Send to right, receive from left
        recvbuf = np.ascontiguousarray(self.field[:, 0, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, -2, :]), rd, recvbuf=recvbuf, source=rs)

        if rs >= 0:
            self.field[:, 0, :] = recvbuf
        else:
            self.field[x0 == "D", 0, :] = 2. * self.bc["rhox0"] - self.field[x0 == "D", 1, :]
            self.field[x0 == "N", 0, :] = self.field[x0 == "N", 1, :]

        # Send to bottom, receive from top
        recvbuf = np.ascontiguousarray(self.field[:, :, -1])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, 1]), bd, recvbuf=recvbuf, source=bs)

        if bs >= 0:
            self.field[:, :, -1] = recvbuf
        else:
            self.field[y1 == "D", :, -1] = 2. * self.bc["rhoy1"] - self.field[y1 == "D", :, -2]
            self.field[y1 == "N", :, -1] = self.field[y1 == "N", :, -2]

        # Send to top, receive from bottom
        recvbuf = np.ascontiguousarray(self.field[:, :, 0])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, -2]), td, recvbuf=recvbuf, source=ts)

        if ts >= 0:
            self.field[:, :, 0] = recvbuf
        else:
            self.field[y0 == "D", :, 0] = 2. * self.bc["rhoy0"] - self.field[y0 == "D", :, 1]
            self.field[y0 == "N", :, 0] = self.field[y0 == "N", :, 1]

    def get_source(self, q, h):
        """
        Compute the source term from the unknown and gap height.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).

        Returns
        -------
        np.array
            Source term

        """

        self.viscous_stress.set(q, h)
        self.upper_stress.set(q, h, "top")
        self.lower_stress.set(q, h, "bottom")

        out = np.zeros_like(q)

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = (-q[1] * h[1] - q[2] * h[2]) / h[0]

        if bool(self.numerics["stokes"]):
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

    def predictor_corrector(self, q, h, dir):
        """
        Compute predictor/corrector fluxes fro MacCormack's method.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).
        dir : int
            Direction of the substep (-1: forwards / 1: backwards differencing)

        Returns
        -------
        tuple
            Total numerical flux in x- and y-directions respectively.

        """

        Fx = self.hyperbolicFlux(q, 1) - self.diffusiveFlux(q, h, 1)
        Fy = self.hyperbolicFlux(q, 2) - self.diffusiveFlux(q, h, 2)

        flux_x = -dir * (np.roll(Fx, dir, axis=1) - Fx)
        flux_y = -dir * (np.roll(Fy, dir, axis=2) - Fy)

        return flux_x, flux_y

    def hyperbolicFlux(self, q, ax):
        """Compute hyperbolic flux.

        Parameters
        ----------
        q : np.array
            Solution field.
        ax : int
            Axis parameter, 1 == x and 2 == y

        Returns
        -------
        np.array
            Hyperbolic flux field.

        """

        p = Material(self.material).eos_pressure(q[0])
        inertialess = bool(self.numerics["stokes"])

        F = np.zeros_like(q)
        if ax == 1:
            if inertialess:
                F[0] = q[1]
                F[1] = p
            else:
                F[0] = q[1]
                F[1] = q[1] * q[1] / q[0] + p
                F[2] = q[2] * q[1] / q[0]

        elif ax == 2:
            if inertialess:
                F[0] = q[2]
                F[2] = p
            else:
                F[0] = q[2]
                F[1] = q[1] * q[2] / q[0]
                F[2] = q[2] * q[2] / q[0] + p

        return F

    def diffusiveFlux(self, q, h, ax):
        """Compute diffusive flux.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).
        ax : int
            Axis parameter, 1 == x and 2 == y

        Returns
        -------
        np.array
            Diffusive flux field

        """

        self.viscous_stress.set(q, h)

        D = np.zeros_like(q)
        if ax == 1:
            D[1] = self.viscous_stress.field[0]
            D[2] = self.viscous_stress.field[2]

        elif ax == 2:
            D[1] = self.viscous_stress.field[2]
            D[2] = self.viscous_stress.field[1]

        return D

    def lax_step(self):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        FE = self.hyperbolicFlux(self.edgeE, 1) - self.diffusiveFlux(self.edgeE, self.height.edgeE, 1)
        FN = self.hyperbolicFlux(self.edgeN, 2) - self.diffusiveFlux(self.edgeN, self.height.edgeN, 2)
        FW = self.hyperbolicFlux(self.edgeW, 1) - self.diffusiveFlux(self.edgeW, self.height.edgeW, 1)
        FS = self.hyperbolicFlux(self.edgeS, 2) - self.diffusiveFlux(self.edgeS, self.height.edgeS, 2)
        src = self.get_source(self.field, self.height.field)

        self.field = self.field - self.dt / 2. * ((FE - FW) / dx + (FN - FS) / dy - src)

    def leap_frog(self, q0):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        FE = self.hyperbolicFlux(self.edgeE, 1) - self.diffusiveFlux(self.edgeE, self.height.edgeE, 1)
        FN = self.hyperbolicFlux(self.edgeN, 2) - self.diffusiveFlux(self.edgeN, self.height.edgeN, 2)
        FW = self.hyperbolicFlux(self.edgeW, 1) - self.diffusiveFlux(self.edgeW, self.height.edgeW, 1)
        FS = self.hyperbolicFlux(self.edgeS, 2) - self.diffusiveFlux(self.edgeS, self.height.edgeS, 2)
        src = self.get_source(self.field, self.height.field)

        self.field = q0 - self.dt * ((FE - FW) / dx + (FN - FS) / dy - src)

    def diffusiveCD(self):

        Dx = self.diffusiveFlux(1)
        Dy = self.diffusiveFlux(2)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return self.dt / (2 * self.dx) * flux_x, self.dt / (2 * self.dy) * flux_y
