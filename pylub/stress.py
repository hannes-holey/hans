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
        eta = EquationOfState(self.mat).viscosity(q[0])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        # origin bottom, U_top = 0, U_bottom = U
        self._field[0] = -((U * q[0] - 3 * q[1]) * (lam + 2 * eta) * h[1] + (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
        self._field[1] = -((V * q[0] - 3 * q[2]) * (lam + 2 * eta) * h[2] + (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
        self._field[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])

        # origin center, U_top = U, U_bottom = 0
        # out.field[0] = (-3 * (lam + 2 * eta) * (U * rho - 2 * j_x) * hx - 3 * lam * (V * rho - 2 * j_y) * hy) / (2 * h0 * rho)
        # out.field[1] = (-3 * (V * rho - 2 * j_y) * (lam + 2 * eta) * hy - 3 * lam * (U * rho - 2 * j_x) * hx) / (2 * h0 * rho)
        # out.field[2] = -(3 * eta * ((V * rho - 2 * j_y) * hx + hy * (U * rho - 2 * j_x))) / (2 * h0 * rho)

        # origin center, U_top = U/2, U_bottom = - U/2
        # out.field[0] = (6 * j_x * (eta + lam / 2) * hx + 3 * lam * j_y * hy) / (h0 * rho)
        # out.field[1] = (6 * j_y * (eta + lam / 2) * hy + 3 * lam * j_x * hx) / (h0 * rho)
        # out.field[2] = 3 * eta * (j_x * hy + j_y * hx) / (h0 * rho)


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material, grid=False):

        super().__init__(disc, grid)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h, bound):

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = EquationOfState(self.mat).viscosity(q[0])
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

            # origin center, U_top = U, U_bottom = 0
            # out.field[0] = (-6 * (U * rho - 2 * j_x) * (eta + lam / 2) * hx - 3 * lam * (V * rho - 2 * j_y) * hy) / (2 * h0 * rho)
            # out.field[1] = (-6 * (V * rho - 2 * j_y) * (eta + lam / 2) * hy - 3 * lam * (U * rho - 2 * j_x) * hx) / (2 * h0 * rho)
            # out.field[2] = (-3 * lam * (U * rho - 2 * j_x) * hx + hy * (V * rho - 2 * j_y)) / (2 * rho * h0)
            # out.field[3] = 3 * eta * (V * rho - 2 * j_y) / (rho * h0)
            # out.field[4] = 3 * eta * (U * rho - 2 * j_x) / (rho * h0)
            # out.field[5] = -3 * eta * ((V * rho - 2 * j_y) * hx + hy * (U * rho - 2 * j_x)) / (2 * rho * h0)

            # origin center, U_top = U/2, U_bottom = - U/2
            # out.field[0] = (-2 * (U * rho - 6 * j_x) * (eta + lam / 2) * hx - lam * (V * rho - 6 * j_y) * hy) / (2 * h0 * rho)
            # out.field[1] = (-2 * (V * rho - 6 * j_y) * (eta + lam / 2) * hy - lam * (U * rho - 6 * j_x) * hx) / (2 * h0 * rho)
            # out.field[2] = -lam * ((U * rho - 6 * j_x) * hx + (V * rho - 6 * j_y) * hy) / (2 * rho * h0)
            # out.field[3] = eta * (V * rho - 6 * j_y) / (rho * h0)
            # out.field[4] = eta * (U * rho - 6 * j_x) / (rho * h0)
            # out.field[5] = -eta * ((V * rho - 6 * j_y) * hx + hy * (U * rho - 6 * j_x)) / (2 * rho * h0)

        elif bound == "bottom":

            # origin bottom, U_top = 0, U_bottom = U
            self._field[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self._field[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])

            # origin center, U_top = U, U_bottom = 0
            # out.field[0] = (-6 * (U * rho - 2 * j_x) * (eta + lam / 2) * hx - 3 * hy * lam * (V * rho - 2 * j_y)) / (2 * h0 * rho)
            # out.field[1] = (-6 * (V * rho - 2 * j_y) * (eta + lam / 2) * hy - 3 * hx * lam * (U * rho - 2 * j_x)) / (2 * h0 * rho)
            # out.field[2] = -3 * lam * ((U * rho - 2 * j_x) * hx + hy * (V * rho - 2 * j_y)) / (2 * rho * h0)
            # out.field[3] = -3 * eta * (V * rho - 2 * j_y) / (rho * h0)
            # out.field[4] = -3 * eta * (U * rho - 2 * j_x) / (rho * h0)
            # out.field[5] = -3 * eta * ((V * rho - 2 * j_y) * hx + hy * (U * rho - 2 * j_x)) / (2 * rho * h0)

            # origin center, U_top = U/2, U_bottom = - U/2
            # out.field[0] = (2 * (U * rho + 6 * j_x) * (eta + lam / 2) * hx - lam * (V * rho + 6 * j_y) * hy) / (2 * h0 * rho)
            # out.field[1] = (2 * (V * rho + 6 * j_y) * (eta + lam / 2) * hy - lam * (U * rho + 6 * j_x) * hx) / (2 * h0 * rho)
            # out.field[2] = lam * ((U * rho + 6 * j_x) * hx + (V * rho + 6 * j_y) * hy) / (2 * rho * h0)
            # out.field[3] = eta * (V * rho + 6 * j_y) / (rho * h0)
            # out.field[4] = eta * (U * rho + 6 * j_x) / (rho * h0)
            # out.field[5] = eta * ((V * rho + 6 * j_y) * hx + hy * (U * rho + 6 * j_x)) / (2 * rho * h0)
