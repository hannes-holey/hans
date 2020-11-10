import numpy as np
import scipy.constants as const
from .eos import EquationOfState
from .field import VectorField, TensorField


class Deterministic:

    def __init__(self, disc, geometry, numerics, material):

        self.disc = disc
        self.geo = geometry
        self.mat = material
        self.num = numerics

    def viscousStress_avg(self, q, h, dt):

        out = VectorField(self.disc)

        U = float(self.geo['U'])
        V = float(self.geo['V'])
        eta = float(self.mat['shear'])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        if bool(self.num['Rey']) is False:
            # origin bottom
            out.field[0] = (-4 * (U * rho - (3 * j_x) / 2) * (eta + lam / 2) * hx - 2 * lam * (V * rho - (3 * j_y) / 2) * hy) / (h0 * rho)
            out.field[1] = (-4 * (V * rho - (3 * j_y) / 2) * (eta + lam / 2) * hy - 2 * lam * (U * rho - (3 * j_x) / 2) * hx) / (h0 * rho)
            out.field[2] = -2 * eta * ((V * rho - (3 * j_y) / 2) * hx + hy * (U * rho - (3 * j_x) / 2)) / (h0 * rho)

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
        eta = float(self.mat['shear'])
        zeta = float(self.mat['bulk'])
        lam = zeta - 2 / 3 * eta

        rho = q.field[0]
        j_x = q.field[1]
        j_y = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        if bound == "top":
            if bool(self.num['Rey']) is False:
                out.field[0] = (-8 * (eta + lam / 2) * (U * rho - (3 * j_x) / 2) * hx - 4 * hy * lam * (V * rho - (3 * j_y) / 2)) / (h0 * rho)
                out.field[1] = (-8 * (eta + lam / 2) * (V * rho - (3 * j_y) / 2) * hy - 4 * hx * lam * (U * rho - (3 * j_x) / 2)) / (h0 * rho)
                out.field[2] = -4 * lam * ((U * rho - (3 * j_x) / 2) * hx + hy * (V * rho - (3 * j_y) / 2)) / (h0 * rho)
                out.field[3] = 2 * eta * (2 * V * rho - 3 * j_y) / (h0 * rho)
                out.field[4] = 2 * eta * (2 * U * rho - 3 * j_x) / (h0 * rho)
                out.field[5] = -4 * ((V * rho - (3 * j_y) / 2) * hx + hy * (U * rho - (3 * j_x) / 2)) * eta / (h0 * rho)
            else:
                out.field[3] = 2 * eta * (2 * V * rho - 3 * j_y) / (h0 * rho)
                out.field[4] = 2 * eta * (2 * U * rho - 3 * j_x) / (h0 * rho)
        elif bound == "bottom":
            out.field[3] = -2 * eta * (V * rho - 3 * j_y) / (h0 * rho)
            out.field[4] = -2 * eta * (U * rho - 3 * j_x) / (h0 * rho)

        return out


class Stochastic:

    def __init__(self, disc, material):

        self.disc = disc
        self.material = material

    def full_tensor(self, W_field_A, W_field_B, h, dt, stage):

        dx = float(self.disc['dx'])
        dy = float(self.disc['dy'])
        dz = h.field[0]

        eta = float(self.material['shear'])
        zeta = float(self.material['bulk'])
        T = float(self.material['T0'])

        a_coeff = np.sqrt(2 * const.k * T * eta / (dx * dy * dz * dt))
        b_coeff = np.sqrt(const.k * T * zeta / (3 * dx * dy * dz * dt))

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

        W_field_sym = (W_field + np.transpose(W_field, axes=(1,0,2,3))) / np.sqrt(2)

        W_field = TensorField(self.disc)

        W_field.field[0] = W_field_sym[0,0,:,:]
        W_field.field[1] = W_field_sym[1,1,:,:]
        W_field.field[2] = W_field_sym[2,2,:,:]
        W_field.field[3] = W_field_sym[1,2,:,:]
        W_field.field[4] = W_field_sym[0,2,:,:]
        W_field.field[5] = W_field_sym[0,1,:,:]

        diag = np.array([1, 1, 1, 0, 0, 0])

        W_trace = np.sum(W_field.field[diag], axis=0)

        W_field.field[diag] -= W_trace / 3
        W_field.field *= a_coeff
        W_field.field[diag] += b_coeff * W_trace

        return W_field.field
