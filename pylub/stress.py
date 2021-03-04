from pylub.field import VectorField, TensorField
from pylub.eos import EquationOfState


class SymStressField2D(VectorField):

    def __init__(self, disc, geometry, material, grid=False):

        super().__init__(disc, grid)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h):

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = EquationOfState(self.mat).viscosity(U, V, q[0], h[0])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        # origin bottom, U_top = 0, U_bottom = U
        self._field[0] = -((U * q[0] - 3 * q[1]) * (lam + 2 * eta) * h[1] + (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
        self._field[1] = -((V * q[0] - 3 * q[2]) * (lam + 2 * eta) * h[2] + (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
        self._field[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material, grid=False):

        super().__init__(disc, grid)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h, bound):

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = EquationOfState(self.mat).viscosity(U, V, q[0], h[0])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        if bound == "top":

            # origin bottom, U_top = 0, U_bottom = U
            self._field[0] = (-2 * (U * q[0] - 3 * q[1]) * (2 * eta + lam) * h[1] - 2 * (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
            self._field[1] = (-2 * (V * q[0] - 3 * q[2]) * (2 * eta + lam) * h[2] - 2 * (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
            self._field[2] = -2 * lam * ((U * q[0] - 3 * q[1]) * h[1] + (V * q[0] - 3 * q[2]) * h[2]) / (q[0] * h[0])
            self._field[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self._field[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * h[0])
            self._field[5] = -2 * eta * ((V * q[0] - 3 * q[2]) * h[1] + h[2] * (U * q[0] - 3 * q[1])) / (q[0] * h[0])

        elif bound == "bottom":

            # origin bottom, U_top = 0, U_bottom = U
            self._field[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self._field[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])
