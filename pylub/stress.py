import numpy as np

from pylub.field import VectorField, TensorField
from pylub.eos import EquationOfState


class SymStressField2D(VectorField):

    def __init__(self, disc, geometry, material):

        super().__init__(disc)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h):

        U = self.geo['U']
        V = self.geo['V']
        eta = EquationOfState(self.mat).viscosity(U, V, q[0], h[0])
        zeta = self.mat['bulk']

        visc_1 = zeta + 4 / 3 * eta
        visc_2 = zeta - 2 / 3 * eta

        # origin bottom, U_top = 0, U_bottom = U
        self._field[0] = -3 * ((U * q[0] - q[1]) * visc_1 * h[1] + (V * q[0] - q[2]) * visc_2 * h[2]) / (h[0] * q[0])
        self._field[1] = -3 * ((V * q[0] - q[2]) * visc_1 * h[2] + (U * q[0] - q[1]) * visc_2 * h[1]) / (h[0] * q[0])
        self._field[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material):

        super().__init__(disc)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h, bound):

        U = self.geo['U']
        V = self.geo['V']
        eta = EquationOfState(self.mat).viscosity(U, V, q[0], h[0])
        zeta = self.mat['bulk']

        visc_1 = zeta + 4 / 3 * eta
        visc_2 = zeta - 2 / 3 * eta

        if bound == "top":

            # origin bottom, U_top = 0, U_bottom = U
            self._field[0] = -2 * ((U * q[0] - 3 * q[1]) * visc_1 * h[1] + (V * q[0] - 3 * q[2]) * visc_2 * h[2]) / (h[0] * q[0])
            self._field[1] = -2 * ((V * q[0] - 3 * q[2]) * visc_1 * h[2] + (U * q[0] - 3 * q[1]) * visc_2 * h[1]) / (h[0] * q[0])
            self._field[2] = -2 * visc_2 * ((U * q[0] - 3 * q[1]) * h[1] + (V * q[0] - 3 * q[2]) * h[2]) / (h[0] * q[0])
            self._field[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self._field[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * h[0])
            self._field[5] = -2 * eta * ((V * q[0] - 3 * q[2]) * h[1] + h[2] * (U * q[0] - 3 * q[1])) / (q[0] * h[0])

        elif bound == "bottom":

            # origin bottom, U_top = 0, U_bottom = U
            self._field[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self._field[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])


class SymStochStressField3D(TensorField):

    def __init__(self, disc, geometry, material):

        super().__init__(disc)

        self.disc = disc
        self.material = material

    def set(self, W_field_A, W_field_B, h, dt, stage):
        KB = 1.380649e-23

        dx = float(self.disc['dx'])
        dy = float(self.disc['dy'])
        dz = h[0]

        eta = float(self.material['shear'])
        zeta = float(self.material['bulk'])
        T = float(self.material['T0'])

        a_coeff = np.sqrt(2 * KB * T * eta / (dx * dy * dz * dt))
        b_coeff = np.sqrt(KB * T * zeta / (3 * dx * dy * dz * dt))

        # weights
        # 1RNG
        # weight = {1: [0., 1.], 2: [0., 1.], 3: [0., 1.]}

        # 2RNG_V1
        # weight = {1: [1., -np.sqrt(3)], 2: [1., np.sqrt(3)], 3: [1., 0.]}

        # 2RNG_V2
        weight = {1: [1., (2 * np.sqrt(2) - np.sqrt(3)) / 5],
                  2: [1., (-4 * np.sqrt(2) - 3 * np.sqrt(3)) / 5],
                  3: [1., (np.sqrt(2) + 2 * np.sqrt(3)) / 10]}

        # 2RNG_V3
        # weight = {1: [1., (2 * np.sqrt(2) + np.sqrt(3)) / 5],
        #           2: [1., (-4 * np.sqrt(2) + 3 * np.sqrt(3)) / 5],
        #           3: [1., (np.sqrt(2) - 2 * np.sqrt(3)) / 10]}

        W_field = weight[stage][0] * W_field_A + weight[stage][1] * W_field_B

        W_field_sym = (W_field + np.transpose(W_field, axes=(1, 0, 2, 3))) / np.sqrt(2)

        self._field[0] = W_field_sym[0, 0, :, :]
        self._field[1] = W_field_sym[1, 1, :, :]
        self._field[2] = W_field_sym[2, 2, :, :]
        self._field[3] = W_field_sym[1, 2, :, :]
        self._field[4] = W_field_sym[0, 2, :, :]
        self._field[5] = W_field_sym[0, 1, :, :]

        W_trace = np.sum(self._field[:3], axis=0)

        self._field[:3] -= W_trace / 3
        self._field *= a_coeff
        self._field[:3] += b_coeff * W_trace
