import numpy as np

from .eos import EquationOfState
from .geometry import Analytic
from .field import VectorField
from .flux import Flux


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

        rho0 = float(material['rho0'])

        self.q = VectorField(disc)

        if q_init is not None:
            self.q.field = q_init
        else:
            self.q.field[0] = rho0

        if self.type == 'droplet':
            self.q.fill_circle(1.05 * rho0, 0, radius=1e-7)
        elif self.type == 'wavefront':
            self.q.fill_line(1.05 * rho0, 0, 0)

        self.Flux = Flux(disc, geometry, numerics, material)

        self.vSound = EquationOfState(self.material).soundSpeed(rho0)

    def solve(self, i):

        self.vmax = self.vSound + max(np.amax(np.sqrt(self.q.field[1]**2 + self.q.field[2]**2) / self.q.field[0]), 1e-3)

        if self.adaptive is True:
            if i == 0:
                self.dt = self.dt
            else:
                self.dt = self.C * min(self.q.dx, self.q.dy) / self.vmax

        if self.numFlux == 'LW':
            self.q = self.Flux.Richtmyer(self.q, self.height, self.dt)

        elif self.numFlux == 'MC':
            self.q = self.Flux.MacCormack(self.q, self.height, self.dt)

        # some scalar output
        self.mass = np.sum(self.q.field[0] * self.height.field[0] * self.q.dx * self.q.dy)
        self.time += self.dt

        self.vSound = EquationOfState(self.material).soundSpeed(self.q.field[0])
        vmax_new = self.vSound + np.amax(np.sqrt(self.q.field[1]**2 + self.q.field[2]**2) / self.q.field[0])
        self.eps = abs(vmax_new - self.vmax) / self.vmax / self.C
        # self.eps = 1.
