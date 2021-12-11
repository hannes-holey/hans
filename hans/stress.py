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

    def __init__(self, disc, geometry, material, surface=None):

        super().__init__(disc)

        self.geometry = geometry
        self.material = material
        self.surface = surface

        try:
            self.n = self.material["PLindex"]
        except KeyError:
            self.n = 1

    def set(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        if self.surface is None or self.surface["type"] == "full":
            self.field[0] = (-3*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v1*h[1]
                             - 3*((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                  + 6*Ls**2*(V*q[0] - 2*q[2])) * h[2]*v2)/(3*h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

            self.field[1] = (-3*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v2*h[1]
                             - 3*((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                  + 6*Ls**2*(V*q[0] - 2*q[2])) * h[2]*v1)/(3*h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

            self.field[2] = -eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls ** 2*(V*q[0] - 2*q[2]))*h[1]
                                  + ((U*q[0] - 3*q[1])*h[0] ** 2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))
                                  * h[2])/(h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

        else:
            # all other types than 'full' consider slip only at the top surface
            self.field[0] = (-3*v1*((U*q[0] - 3*q[1])*h[0]**2
                                    + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                             - 3*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                            + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            self.field[1] = (-3*v2*((U*q[0] - 3*q[1])*h[0]**2
                                    + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                             - 3*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                            + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            self.field[2] = -eta*(((V*q[0] - 3*q[2])*h[0]**2
                                   + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                                  + ((U*q[0] - 3*q[1]) * h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0]
                                     + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material, surface):

        super().__init__(disc)

        self.geometry = geometry
        self.material = material
        self.surface = surface

        try:
            self.n = self.material["PLindex"]
        except KeyError:
            self.n = 1

    def set(self, q, h, Ls, bound):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        if bound == "top":
            if self.surface is None or self.surface["type"] == "full":
                # all other types than 'full' consider slip only at the top surface
                s0 = 2 * (((q[1]*(2+4*self.n)-(1+self.n) * q[0] * U) * h[1])/(self.n * q[0] * h[0]))
                self.field[0] = eta * np.sign(s0) * np.abs(s0)**self.n

                s1 = 2 * (((q[2]*(2+4*self.n)-(1+self.n) * q[0] * V) * h[2])/(self.n * q[0] * h[0]))
                self.field[1] = eta * np.sign(s1) * np.abs(s1)**self.n

                s3 = ((-2 * q[2]*(1 + 2 * self.n) + (1 + self.n) * q[0] * V)/(self.n * q[0] * h[0]))
                self.field[3] = eta * np.sign(s3) * np.abs(s3)**self.n

                s4 = ((-2 * q[1]*(1 + 2 * self.n) + (1 + self.n) * q[0] * U)/(self.n * q[0] * h[0]))
                self.field[4] = eta * np.sign(s4) * np.abs(s4)**self.n

                s5 = (((q[1] * (2 + 4 * self.n) - (1+self.n) * q[0] * U) * h[2] +
                       (q[2] * (2 + 4 * self.n) - (1+self.n) * q[0] * V) * h[2]) / (self.n * q[0] * h[0]))
                self.field[5] = eta * np.sign(s5) * np.abs(s5)**self.n
            else:
                # all other types than 'full' consider slip only at the top surface
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
            if self.surface is None or self.surface["type"] == "full":
                # surface type "full" means that both surface are slippery with equal slip length
                s3 = ((q[2] * (2 + 4 * self.n) - (1 + 3 * self.n) * q[0] * V)/(self.n * q[0] * h[0]))
                self.field[3] = eta * np.sign(s3) * np.abs(s3)**self.n

                s4 = ((q[1] * (2 + 4 * self.n) - (1 + 3 * self.n) * q[0] * U)/(self.n * q[0] * h[0]))
                self.field[4] = eta * np.sign(s4) * np.abs(s4)**self.n
            else:
                # all other types than 'full' consider slip only at the top surface
                self.field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
                self.field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))
