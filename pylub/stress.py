from .eos import EquationOfState
from .field import VectorField, TensorField


class Deterministic:

    def __init__(self, disc, geometry, material):

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def viscousStress_avg(self, q, h, dt):

        out = VectorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = float(self.mat["shear"])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        # origin bottom, U_top = 0, U_bottom = U
        out.field[0] = -((U * rho - 3 * j_x) * (lam + 2 * eta) * hx + (V * rho - 3 * j_y) * lam * hy) / (h0 * rho)
        out.field[1] = -((V * rho - 3 * j_y) * (lam + 2 * eta) * hy + (U * rho - 3 * j_x) * lam * hx) / (h0 * rho)
        out.field[2] = -eta * ((V * rho - 3 * j_y) * hx + (U * rho - 3 * j_x) * hy) / (h0 * rho)

        # origin center, U_top = U, U_bottom = 0
        # out.field[0] = (-3 * (lam + 2 * eta) * (U * rho - 2 * j_x) * hx - 3 * lam * (V * rho - 2 * j_y) * hy) / (2 * h0 * rho)
        # out.field[1] = (-3 * (V * rho - 2 * j_y) * (lam + 2 * eta) * hy - 3 * lam * (U * rho - 2 * j_x) * hx) / (2 * h0 * rho)
        # out.field[2] = -(3 * eta * ((V * rho - 2 * j_y) * hx + hy * (U * rho - 2 * j_x))) / (2 * h0 * rho)

        # origin center, U_top = U/2, U_bottom = - U/2
        # out.field[0] = (6 * j_x * (eta + lam / 2) * hx + 3 * lam * j_y * hy) / (h0 * rho)
        # out.field[1] = (6 * j_y * (eta + lam / 2) * hy + 3 * lam * j_x * hx) / (h0 * rho)
        # out.field[2] = 3 * eta * (j_x * hy + j_y * hx) / (h0 * rho)

        return out

    def stress_avg(self, q, h, dt):

        viscStress = self.viscousStress_avg(q, h, dt)
        stress = VectorField(self.disc)

        pressure = EquationOfState(self.mat).isoT_pressure(q.field[0])

        stress.field[0] = viscStress.field[0] - pressure
        stress.field[1] = viscStress.field[1] - pressure

        return stress, viscStress

    def pressure(self, q):

        return EquationOfState(self.mat).isoT_pressure(q[0])

    def viscousStress_wall(self, q, h, dt, bound):

        out = TensorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = float(self.mat["shear"])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        if bound == "top":

            # origin bottom, U_top = 0, U_bottom = U
            out.field[0] = (-2 * (U * rho - 3 * j_x) * (2 * eta + lam) * hx - 2 * (V * rho - 3 * j_y) * lam * hy) / (h0 * rho)
            out.field[1] = (-2 * (V * rho - 3 * j_y) * (2 * eta + lam) * hy - 2 * (U * rho - 3 * j_x) * lam * hx) / (h0 * rho)
            out.field[2] = -2 * lam * ((U * rho - 3 * j_x) * hx + (V * rho - 3 * j_y) * hy) / (rho * h0)
            out.field[3] = 2 * eta * (V * rho - 3 * j_y) / (rho * h0)
            out.field[4] = 2 * eta * (U * rho - 3 * j_x) / (rho * h0)
            out.field[5] = -2 * eta * ((V * rho - 3 * j_y) * hx + hy * (U * rho - 3 * j_x)) / (rho * h0)

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
            out.field[3] = -2 * eta * (2 * V * rho - 3 * j_y) / (rho * h0)
            out.field[4] = -2 * eta * (2 * U * rho - 3 * j_x) / (rho * h0)

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

        return out
