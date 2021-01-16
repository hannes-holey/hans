from pylub.field import VectorField, TensorField
from pylub.eos import EquationOfState


class SymmetricStressField(VectorField):

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
        self.field[0] = -((U * q[0] - 3 * q[1]) * (lam + 2 * eta) * h[1] + (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
        self.field[1] = -((V * q[0] - 3 * q[2]) * (lam + 2 * eta) * h[2] + (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
        self.field[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])

        # origin center, U_top = U, U_bottom = 0
        # out[0] = (-3 * (lam + 2 * eta) * (U * q[0] - 2 * q[1]) * h[1] - 3 * lam * (V * q[0] - 2 * q[2]) * h[2]) / (2 * h[0] * q[0])
        # out[1] = (-3 * (V * q[0] - 2 * q[2]) * (lam + 2 * eta) * h[2] - 3 * lam * (U * q[0] - 2 * q[1]) * h[1]) / (2 * h[0] * q[0])
        # out[2] = -(3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1]))) / (2 * h[0] * q[0])

        # origin center, U_top = U/2, U_bottom = - U/2
        # out[0] = (6 * q[1] * (eta + lam / 2) * h[1] + 3 * lam * q[2] * h[2]) / (h[0] * q[0])
        # out[1] = (6 * q[2] * (eta + lam / 2) * h[2] + 3 * lam * q[1] * h[1]) / (h[0] * q[0])
        # out[2] = 3 * eta * (q[1] * h[2] + q[2] * h[1]) / (h[0] * q[0])


class StressField(TensorField):

    def __init__(self, disc, geometry, material, grid=False):

        super().__init__(disc, grid)

        self.disc = disc
        self.geo = geometry
        self.mat = material

    def set(self, q, h, bound):

        # out = np.zeros((6, *q.shape[1:]))

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = EquationOfState(self.mat).viscosity(q[0])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        if bound == "top":

            # origin bottom, U_top = 0, U_bottom = U
            self.field[0] = (-2 * (U * q[0] - 3 * q[1]) * (2 * eta + lam) * h[1] - 2 * (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
            self.field[1] = (-2 * (V * q[0] - 3 * q[2]) * (2 * eta + lam) * h[2] - 2 * (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
            self.field[2] = -2 * lam * ((U * q[0] - 3 * q[1]) * h[1] + (V * q[0] - 3 * q[2]) * h[2]) / (q[0] * h[0])
            self.field[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self.field[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * h[0])
            self.field[5] = -2 * eta * ((V * q[0] - 3 * q[2]) * h[1] + h[2] * (U * q[0] - 3 * q[1])) / (q[0] * h[0])

            # origin center, U_top = U, U_bottom = 0
            # out[0] = (-6 * (U * q[0] - 2 * q[1]) * (eta + lam / 2) * h[1] - 3 * lam * (V * q[0] - 2 * q[2]) * h[2]) / (2 * h[0] * q[0])
            # out[1] = (-6 * (V * q[0] - 2 * q[2]) * (eta + lam / 2) * h[2] - 3 * lam * (U * q[0] - 2 * q[1]) * h[1]) / (2 * h[0] * q[0])
            # out[2] = (-3 * lam * (U * q[0] - 2 * q[1]) * h[1] + h[2] * (V * q[0] - 2 * q[2])) / (2 * q[0] * h[0])
            # out[3] = 3 * eta * (V * q[0] - 2 * q[2]) / (q[0] * h[0])
            # out[4] = 3 * eta * (U * q[0] - 2 * q[1]) / (q[0] * h[0])
            # out[5] = -3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1])) / (2 * q[0] * h[0])

            # origin center, U_top = U/2, U_bottom = - U/2
            # out[0] = (-2 * (U * q[0] - 6 * q[1]) * (eta + lam / 2) * h[1] - lam * (V * q[0] - 6 * q[2]) * h[2]) / (2 * h[0] * q[0])
            # out[1] = (-2 * (V * q[0] - 6 * q[2]) * (eta + lam / 2) * h[2] - lam * (U * q[0] - 6 * q[1]) * h[1]) / (2 * h[0] * q[0])
            # out[2] = -lam * ((U * q[0] - 6 * q[1]) * h[1] + (V * q[0] - 6 * q[2]) * h[2]) / (2 * q[0] * h[0])
            # out[3] = eta * (V * q[0] - 6 * q[2]) / (q[0] * h[0])
            # out[4] = eta * (U * q[0] - 6 * q[1]) / (q[0] * h[0])
            # out[5] = -eta * ((V * q[0] - 6 * q[2]) * h[1] + h[2] * (U * q[0] - 6 * q[1])) / (2 * q[0] * h[0])

        elif bound == "bottom":

            # origin bottom, U_top = 0, U_bottom = U
            self.field[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self.field[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])

            # origin center, U_top = U, U_bottom = 0
            # out[0] = (-6 * (U * q[0] - 2 * q[1]) * (eta + lam / 2) * h[1] - 3 * h[2] * lam * (V * q[0] - 2 * q[2])) / (2 * h[0] * q[0])
            # out[1] = (-6 * (V * q[0] - 2 * q[2]) * (eta + lam / 2) * h[2] - 3 * h[1] * lam * (U * q[0] - 2 * q[1])) / (2 * h[0] * q[0])
            # out[2] = -3 * lam * ((U * q[0] - 2 * q[1]) * h[1] + h[2] * (V * q[0] - 2 * q[2])) / (2 * q[0] * h[0])
            # out[3] = -3 * eta * (V * q[0] - 2 * q[2]) / (q[0] * h[0])
            # out[4] = -3 * eta * (U * q[0] - 2 * q[1]) / (q[0] * h[0])
            # out[5] = -3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1])) / (2 * q[0] * h[0])

            # origin center, U_top = U/2, U_bottom = - U/2
            # out[0] = (2 * (U * q[0] + 6 * q[1]) * (eta + lam / 2) * h[1] - lam * (V * q[0] + 6 * q[2]) * h[2]) / (2 * h[0] * q[0])
            # out[1] = (2 * (V * q[0] + 6 * q[2]) * (eta + lam / 2) * h[2] - lam * (U * q[0] + 6 * q[1]) * h[1]) / (2 * h[0] * q[0])
            # out[2] = lam * ((U * q[0] + 6 * q[1]) * h[1] + (V * q[0] + 6 * q[2]) * h[2]) / (2 * q[0] * h[0])
            # out[3] = eta * (V * q[0] + 6 * q[2]) / (q[0] * h[0])
            # out[4] = eta * (U * q[0] + 6 * q[1]) / (q[0] * h[0])
            # out[5] = eta * ((V * q[0] + 6 * q[2]) * h[1] + h[2] * (U * q[0] + 6 * q[1])) / (2 * q[0] * h[0])


# class Deterministic:
#
#     def __init__(self, disc, geometry, material):
#
#         self.disc = disc
#         self.geo = geometry
#         self.mat = material
#
#     def viscousStress_avg(self, q, h):
#
#         out = np.zeros_like(q)
#
#         U = float(self.geo['U'])
#         V = float(self.geo['V'])
#         eta = EquationOfState(self.mat).viscosity(q[0])
#         zeta = float(self.mat['bulk'])
#         lam = zeta - 2 / 3 * eta
#
#         # origin bottom, U_top = 0, U_bottom = U
#         out[0] = -((U * q[0] - 3 * q[1]) * (lam + 2 * eta) * h[1] + (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
#         out[1] = -((V * q[0] - 3 * q[2]) * (lam + 2 * eta) * h[2] + (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
#         out[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])
#
#         # origin center, U_top = U, U_bottom = 0
#         # out[0] = (-3 * (lam + 2 * eta) * (U * q[0] - 2 * q[1]) * h[1] - 3 * lam * (V * q[0] - 2 * q[2]) * h[2]) / (2 * h[0] * q[0])
#         # out[1] = (-3 * (V * q[0] - 2 * q[2]) * (lam + 2 * eta) * h[2] - 3 * lam * (U * q[0] - 2 * q[1]) * h[1]) / (2 * h[0] * q[0])
#         # out[2] = -(3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1]))) / (2 * h[0] * q[0])
#
#         # origin center, U_top = U/2, U_bottom = - U/2
#         # out[0] = (6 * q[1] * (eta + lam / 2) * h[1] + 3 * lam * q[2] * h[2]) / (h[0] * q[0])
#         # out[1] = (6 * q[2] * (eta + lam / 2) * h[2] + 3 * lam * q[1] * h[1]) / (h[0] * q[0])
#         # out[2] = 3 * eta * (q[1] * h[2] + q[2] * h[1]) / (h[0] * q[0])
#
#         return out
#
#     def stress_avg(self, q, h):
#
#         viscStress = self.viscousStress_avg(q, h)
#         stress = np.zeros_like(q)
#
#         pressure = EquationOfState(self.mat).isoT_pressure(q[0])
#
#         stress[0] = viscStress[0] - pressure
#         stress[1] = viscStress[1] - pressure
#
#         return stress, viscStress
#
#     def pressure(self, q):
#
#         return EquationOfState(self.mat).isoT_pressure(q[0])
#
#     def viscousStress_wall(self, q, h, bound):
#
#         out = np.zeros((6, *q.shape[1:]))
#
#         U = float(self.geo['U'])
#         V = float(self.geo['V'])
#         eta = EquationOfState(self.mat).viscosity(q[0])
#         zeta = float(self.mat['bulk'])
#         lam = zeta - 2 / 3 * eta
#
#         if bound == "top":
#
#             # origin bottom, U_top = 0, U_bottom = U
#             out[0] = (-2 * (U * q[0] - 3 * q[1]) * (2 * eta + lam) * h[1] - 2 * (V * q[0] - 3 * q[2]) * lam * h[2]) / (h[0] * q[0])
#             out[1] = (-2 * (V * q[0] - 3 * q[2]) * (2 * eta + lam) * h[2] - 2 * (U * q[0] - 3 * q[1]) * lam * h[1]) / (h[0] * q[0])
#             out[2] = -2 * lam * ((U * q[0] - 3 * q[1]) * h[1] + (V * q[0] - 3 * q[2]) * h[2]) / (q[0] * h[0])
#             out[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * h[0])
#             out[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * h[0])
#             out[5] = -2 * eta * ((V * q[0] - 3 * q[2]) * h[1] + h[2] * (U * q[0] - 3 * q[1])) / (q[0] * h[0])
#
#             # origin center, U_top = U, U_bottom = 0
#             # out[0] = (-6 * (U * q[0] - 2 * q[1]) * (eta + lam / 2) * h[1] - 3 * lam * (V * q[0] - 2 * q[2]) * h[2]) / (2 * h[0] * q[0])
#             # out[1] = (-6 * (V * q[0] - 2 * q[2]) * (eta + lam / 2) * h[2] - 3 * lam * (U * q[0] - 2 * q[1]) * h[1]) / (2 * h[0] * q[0])
#             # out[2] = (-3 * lam * (U * q[0] - 2 * q[1]) * h[1] + h[2] * (V * q[0] - 2 * q[2])) / (2 * q[0] * h[0])
#             # out[3] = 3 * eta * (V * q[0] - 2 * q[2]) / (q[0] * h[0])
#             # out[4] = 3 * eta * (U * q[0] - 2 * q[1]) / (q[0] * h[0])
#             # out[5] = -3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1])) / (2 * q[0] * h[0])
#
#             # origin center, U_top = U/2, U_bottom = - U/2
#             # out[0] = (-2 * (U * q[0] - 6 * q[1]) * (eta + lam / 2) * h[1] - lam * (V * q[0] - 6 * q[2]) * h[2]) / (2 * h[0] * q[0])
#             # out[1] = (-2 * (V * q[0] - 6 * q[2]) * (eta + lam / 2) * h[2] - lam * (U * q[0] - 6 * q[1]) * h[1]) / (2 * h[0] * q[0])
#             # out[2] = -lam * ((U * q[0] - 6 * q[1]) * h[1] + (V * q[0] - 6 * q[2]) * h[2]) / (2 * q[0] * h[0])
#             # out[3] = eta * (V * q[0] - 6 * q[2]) / (q[0] * h[0])
#             # out[4] = eta * (U * q[0] - 6 * q[1]) / (q[0] * h[0])
#             # out[5] = -eta * ((V * q[0] - 6 * q[2]) * h[1] + h[2] * (U * q[0] - 6 * q[1])) / (2 * q[0] * h[0])
#
#         elif bound == "bottom":
#
#             # origin bottom, U_top = 0, U_bottom = U
#             out[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
#             out[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])
#
#             # origin center, U_top = U, U_bottom = 0
#             # out[0] = (-6 * (U * q[0] - 2 * q[1]) * (eta + lam / 2) * h[1] - 3 * h[2] * lam * (V * q[0] - 2 * q[2])) / (2 * h[0] * q[0])
#             # out[1] = (-6 * (V * q[0] - 2 * q[2]) * (eta + lam / 2) * h[2] - 3 * h[1] * lam * (U * q[0] - 2 * q[1])) / (2 * h[0] * q[0])
#             # out[2] = -3 * lam * ((U * q[0] - 2 * q[1]) * h[1] + h[2] * (V * q[0] - 2 * q[2])) / (2 * q[0] * h[0])
#             # out[3] = -3 * eta * (V * q[0] - 2 * q[2]) / (q[0] * h[0])
#             # out[4] = -3 * eta * (U * q[0] - 2 * q[1]) / (q[0] * h[0])
#             # out[5] = -3 * eta * ((V * q[0] - 2 * q[2]) * h[1] + h[2] * (U * q[0] - 2 * q[1])) / (2 * q[0] * h[0])
#
#             # origin center, U_top = U/2, U_bottom = - U/2
#             # out[0] = (2 * (U * q[0] + 6 * q[1]) * (eta + lam / 2) * h[1] - lam * (V * q[0] + 6 * q[2]) * h[2]) / (2 * h[0] * q[0])
#             # out[1] = (2 * (V * q[0] + 6 * q[2]) * (eta + lam / 2) * h[2] - lam * (U * q[0] + 6 * q[1]) * h[1]) / (2 * h[0] * q[0])
#             # out[2] = lam * ((U * q[0] + 6 * q[1]) * h[1] + (V * q[0] + 6 * q[2]) * h[2]) / (2 * q[0] * h[0])
#             # out[3] = eta * (V * q[0] + 6 * q[2]) / (q[0] * h[0])
#             # out[4] = eta * (U * q[0] + 6 * q[1]) / (q[0] * h[0])
#             # out[5] = eta * ((V * q[0] + 6 * q[2]) * h[1] + h[2] * (U * q[0] + 6 * q[1])) / (2 * q[0] * h[0])
#
#         return out
