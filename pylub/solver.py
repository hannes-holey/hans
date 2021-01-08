import numpy as np

from .eos import EquationOfState
from .geometry import Analytic
from .field import VectorField
from .flux import Flux


class Solver:

    def __init__(self, disc, geometry, numerics, material, q_init):

        self.numFlux = str(numerics['numFlux'])
        self.adaptive = bool(numerics['adaptive'])
        self.dt = float(numerics['dt'])
        self.C = float(numerics['C'])

        self.material = material

        self.time = 0

        # Gap height: journal bearing geometry
        self.height = VectorField(disc)
        self.height.field[0] = Analytic(disc, geometry).journalBearing(self.height.xx, self.height.yy, axis=0)
        self.height.getGradients()

        rho0 = float(material['rho0'])

        self.q = VectorField(disc)

        if q_init is not None:
            self.q.field = q_init
        else:
            self.q.field[0] = rho0

        self.Flux = Flux(disc, geometry, material)

        self.vSound = EquationOfState(material).soundSpeed(rho0)

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
