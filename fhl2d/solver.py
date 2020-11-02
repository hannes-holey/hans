import numpy as np

from .eos import EquationOfState
from .geometry import Analytic
from .field import VectorField
from .flux import Flux
from .stress import Newtonian


class Solver:

    def __init__(self, disc, geometry, numerics, material, q_init):

        self.type = str(geometry['type'])
        self.numFlux = str(numerics['numFlux'])
        self.adaptive = bool(numerics['adaptive'])
        self.dt = float(numerics['dt'])
        self.C = numerics['C']
        self.fluct = bool(numerics['Fluctuating'])

        self.material = material

        self.time = 0

        # Gap height
        self.height = VectorField(disc)

        if self.type == 'journal':
            self.height.field[0] = Analytic(disc, geometry).journalBearing(self.height.xx, self.height.yy, axis=0)
        elif self.type == 'inclined':
            self.height.field[0] = Analytic(disc, geometry).linearSlider(self.height.xx, self.height.yy)
        elif self.type == 'step':
            self.height.field[0] = Analytic(disc, geometry).doubleStep(self.height.xx, self.height.yy, axis=0)

        self.height.getGradients()

        # P0 = float(material['P0'])
        rho0 = float(material['rho0'])

        self.q = VectorField(disc)

        if q_init is not None:
            self.q.field = q_init
        else:
            self.q.field[0] = rho0

        # if self.type == 'inclined':
        #     self.q.field[0][0,:] = EquationOfState(self.material).isoT_density(P0)
        #     self.q.field[0][-1,:] = EquationOfState(self.material).isoT_density(P0)
        # elif self.type == 'poiseuille':
        #     self.q.field[0][-1,:] = EquationOfState(self.material).isoT_density(P0)
        #     self.q.field[0][0,:] = EquationOfState(self.material).isoT_density(2. * P0)
        # elif self.type == 'droplet':
        #     self.q.fill_circle(EquationOfState(self.material).isoT_density(2. * P0), 0)
        # elif self.type == 'wavefront':
        #     self.q.fill_line(EquationOfState(self.material).isoT_density(2. * P0), 0, 0)

        self.Flux = Flux(disc, geometry, numerics, material)
        self.Newtonian = Newtonian(disc, geometry, numerics, material)

        self.vSound = EquationOfState(self.material).soundSpeed(rho0)

    def solve(self, i):

        self.vmax = self.vSound + max(np.amax(np.sqrt(self.q.field[1]**2 + self.q.field[2]**2) / self.q.field[0]), 1e-3)

        if self.adaptive is True:
            if i == 0:
                self.dt = self.dt
            else:
                self.dt = self.C * min(self.q.dx, self.q.dy) / self.vmax

        # viscousStress, stress, cov3, p = self.Newtonian.stress_avg(self.q, self.height, self.dt)

        # if self.fluct is True:
        #     stress.addNoise_FH(cov3)

        # if self.numFlux == 'LF':
        #     fXE = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, -1, 0)
        #     fXW = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, 1, 0)
        #     fYN = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, -1, 1)
        #     fYS = self.Flux.getFlux_LF(self.q, self.height, stress, self.dt, 1, 1)
        #
        #     rhs = -1. / self.q.dx * (fXE.field - fXW.field) - 1. / self.q.dy * (fYN.field - fYS.field)
        #     source = self.Flux.getSource(viscousStress, self.q, self.height, self.dt)
        #     rhs += source.field
        #     self.q.field += self.dt * rhs

        if self.numFlux == 'LW':
            self.q = self.Flux.Richtmyer(self.q, self.height, self.dt)

        elif self.numFlux == 'MC_old':
            self.q = self.Flux.MacCormack_total(self.q, self.height, self.dt)

        elif self.numFlux == 'MC':
            self.q = self.Flux.MacCormack(self.q, self.height, self.dt, i)

        elif self.numFlux == 'RK3':
            self.q = self.Flux.RungeKutta3(self.q, self.height, self.dt, i)

        # some scalar output
        self.mass = np.sum(self.q.field[0] * self.height.field[0] * self.q.dx * self.q.dy)
        self.time += self.dt

        self.vSound = EquationOfState(self.material).soundSpeed(self.q.field[0])
        vmax_new = self.vSound + np.amax(np.sqrt(self.q.field[1]**2 + self.q.field[2]**2) / self.q.field[0])
        self.eps = abs(vmax_new - self.vmax) / self.vmax / self.C
        # self.eps = 1.
