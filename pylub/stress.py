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
