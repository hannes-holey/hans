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
        self.field[0] = -3 * ((U * q[0] - q[1]) * visc_1 * h[1] + (V * q[0] - q[2]) * visc_2 * h[2]) / (h[0] * q[0])
        self.field[1] = -3 * ((V * q[0] - q[2]) * visc_1 * h[2] + (U * q[0] - q[1]) * visc_2 * h[1]) / (h[0] * q[0])
        self.field[2] = -eta * ((V * q[0] - 3 * q[2]) * h[1] + (U * q[0] - 3 * q[1]) * h[2]) / (h[0] * q[0])


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
            self.field[0] = -2 * ((U * q[0] - 3 * q[1]) * visc_1 * h[1] + (V * q[0] - 3 * q[2]) * visc_2 * h[2]) / (h[0] * q[0])
            self.field[1] = -2 * ((V * q[0] - 3 * q[2]) * visc_1 * h[2] + (U * q[0] - 3 * q[1]) * visc_2 * h[1]) / (h[0] * q[0])
            self.field[2] = -2 * visc_2 * ((U * q[0] - 3 * q[1]) * h[1] + (V * q[0] - 3 * q[2]) * h[2]) / (h[0] * q[0])
            self.field[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self.field[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * h[0])
            self.field[5] = -2 * eta * ((V * q[0] - 3 * q[2]) * h[1] + h[2] * (U * q[0] - 3 * q[1])) / (q[0] * h[0])

        elif bound == "bottom":

            # origin bottom, U_top = 0, U_bottom = U
            self.field[3] = -2 * eta * (2 * V * q[0] - 3 * q[2]) / (q[0] * h[0])
            self.field[4] = -2 * eta * (2 * U * q[0] - 3 * q[1]) / (q[0] * h[0])
