"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from hans.field import VectorField, TensorField
from hans.material import Material


class SymStressField2D(VectorField):

    def __init__(self, disc, geometry, material):

        super().__init__(disc)

        self.geometry = geometry
        self.material = material

    def set(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        # slip top
        self.field[0] = (-3*v1*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                         - 3*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2
                                        + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

        self.field[1] = (-3*v2*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                         - 3*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2
                                        + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

        self.field[2] = -eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                              + ((U*q[0] - 3*q[1]) * h[0]**2
                                 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material):

        super().__init__(disc)

        self.geometry = geometry
        self.material = material

    def set(self, q, h, Ls, bound):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        if bound == "top":

            # slip top
            self.field[0] = (-6*v1*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                             - 6*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                            + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            self.field[1] = (-6*v2*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                             - 6*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                            + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            self.field[2] = -2*v2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1] +
                                   h[2]*((V*q[0] - 3*q[2])*h[0]**2
                                         + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2])))/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            self.field[3] = 2*(V*q[0] - 3*q[2])*eta/(q[0]*(4*Ls + h[0]))

            self.field[4] = 2*(U*q[0] - 3*q[1])*eta/(q[0]*(4*Ls + h[0]))

            self.field[5] = -2*eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                                    + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0]
                                       + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

        elif bound == "bottom":

            # slip top
            self.field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
            self.field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))


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

        self.field[0] = W_field_sym[0, 0, :, :]
        self.field[1] = W_field_sym[1, 1, :, :]
        self.field[2] = W_field_sym[2, 2, :, :]
        self.field[3] = W_field_sym[1, 2, :, :]
        self.field[4] = W_field_sym[0, 2, :, :]
        self.field[5] = W_field_sym[0, 1, :, :]

        W_trace = np.sum(self.field[:3], axis=0)

        self.field[:3] -= W_trace / 3
        self.field *= a_coeff
        self.field[:3] += b_coeff * W_trace
