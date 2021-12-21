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

    def num_param(self, Un, Vn):

        zmaxu = -1e5 * np.ones_like(Un)
        zmaxv = 0.5 * np.ones_like(Vn)

        zmaxu[Un < 1.9] = (2. + 1./(Un[Un < 1.9] - 2.)) / 3.
        zmaxv[Vn < 1.9] = (2. + 1./(Vn[Vn < 1.9] - 2.)) / 3.

        zmaxu[Un >= 1.9] = -1e5
        zmaxv[Vn >= 1.9] = 0.5

        return zmaxu, zmaxv

    def set(self, q, h, Ls, bound):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        n = self.n

        zeroflux_x = np.isclose(q[1], 0)
        zeroflux_y = np.isclose(q[2], 0)

        meanu = q[1] / q[0]
        meanv = q[2] / q[0]

        # meanu[zeroflux_x] = U / 2
        # meanv[zeroflux_y] = V / 2

        if U == 0.:
            U_nondim = np.zeros_like(q[0])
        else:
            U_nondim = U / meanu

        if V == 0.:
            V_nondim = np.zeros_like(q[0])
        else:
            V_nondim = V / meanv

        UV_crit = (1 + 2 * n) / (1 + n)

        # get numeric parameters
        zmaxu, zmaxv = self.num_param(U_nondim, V_nondim)

        print(np.amax(zmaxu), np.amax(U_nondim))

        zmaxu *= h[0]
        zmaxv *= h[0]

        # case 1: U_nondim <= UV_crit; V_nondim <= UV_crit
        maskU1 = U_nondim <= UV_crit
        maskV1 = V_nondim <= UV_crit

        # case 2: U_nondim > UV_crit; V_nondim > UV_crit
        maskU2 = U_nondim > UV_crit
        maskV2 = V_nondim > UV_crit

        if bound == "top":
            if self.surface is None or self.surface["type"] == "full":
                # all other types than 'full' consider slip only at the top surface

                # case 1
                s0_1 = -2 * ((1+2*n)*power(-zmaxu+h[0], 1/n)*(q[0]*U*zmaxu-q[1]*h[0])*h[1]) / \
                    (n*q[0]*(power(zmaxu, 2+1/n)+power(-zmaxu+h[0], 2+1/n)))

                s1_1 = -2 * ((1+2*n)*power(-zmaxv+h[0], 1/n)*(q[0]*V*zmaxv-q[2]*h[0])*h[2]) / \
                    (n*q[0]*(power(zmaxv, 2+1/n)+power(-zmaxv+h[0], 2+1/n)))

                s3_1 = ((1+2*n)*power(-zmaxv+h[0], 1/n)*(q[0]*V*zmaxv-q[2]*h[0]))/(n*q[0]*(power(zmaxv, 2+1/n)+power(-zmaxv+h[0], 2+1/n)))

                s4_1 = ((1+2*n)*power(-zmaxu+h[0], 1/n)*(q[0]*U*zmaxu-q[1]*h[0]))/(n*q[0]*(power(zmaxu, 2+1/n)+power(-zmaxu+h[0], 2+1/n)))

                s5_u1 = -s4_1 * h[2]
                s5_v1 = -s3_1 * h[1]

                # case 2
                s0_2 = -2*((1+2*n)*(q[1]*q[0]-U)*(power(zmaxu, 1+1/n)-power(zmaxu-h[0], 1+1/n)) *
                           (power(zmaxu, 2+1/n)*n-power(zmaxu - h[0], 1+1/n)
                            * (zmaxu*n+(1+n)*h[0]))*h[1])/(n*power(zmaxu-h[0], 2+1/n) + power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0]))**2

                s1_2 = -2*((1+2*n)*(q[2]*q[0]-V)*(power(zmaxv, 1+1/n)-power(zmaxv-h[0], 1+1/n)) *
                           (power(zmaxv, 2+1/n)*n-power(zmaxv - h[0], 1+1/n)
                            * (zmaxv*n+(1+n)*h[0]))*h[2])/(n*power(zmaxv-h[0], 2+1/n) + power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0]))**2

                s3_2 = ((1+1/n)*(1+2*n)*(q[2]*q[0]-V)*power(zmaxv-h[0], 1/n)*h[0]) / \
                    (n*power(zmaxv-h[0], 2+1/n)+power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0]))

                s4_2 = ((1+1/n)*(1+2*n)*(q[1]*q[0]-V)*power(zmaxu-h[0], 1/n)*h[0]) / \
                    (n*power(zmaxu-h[0], 2+1/n)+power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0]))

                s5_u2 = (1+2*n)*(-(((1+2*n)*(q[1]*q[0]-U)*(power(zmaxu, 1+1/n)
                                                           - power(zmaxu-h[0], 1+1/n))**2 * h[0]*h[2]) /
                                   (n*power(zmaxu-h[0], 2+1/n) + power(zmaxu, 1+1/n)*(-zmaxu * n+(1+2*n)*h[0]))**2)
                                 + ((q[1]*q[0]-U)*(power(zmaxu, 1+1/n)-power(zmaxu-h[0], 1+1/n))*h[2]) /
                                 (n*power(zmaxu-h[0], 2+1/n)+power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0])))

                s5_v2 = (1+2*n)*(-(((1+2*n)*(q[2]*q[0]-V)*(power(zmaxv, 1+1/n)
                                                           - power(zmaxv-h[0], 1+1/n))**2 * h[0]*h[1]) /
                                   (n*power(zmaxv-h[0], 2+1/n) + power(zmaxv, 1+1/n)*(-zmaxv * n+(1+2*n)*h[0]))**2)
                                 + ((q[2]*q[0]-V)*(power(zmaxv, 1+1/n)-power(zmaxv-h[0], 1+1/n))*h[1]) /
                                 (n*power(zmaxv-h[0], 2+1/n)+power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0])))

                self.field[0, maskU1] = eta * power(s0_1[maskU1], n)
                self.field[0, maskU2] = eta * power(s0_2[maskU2], n)

                self.field[1, maskV1] = eta * power(s1_1[maskV1], n)
                self.field[1, maskV2] = eta * power(s1_2[maskV2], n)

                self.field[3, maskV1] = eta * power(s3_1[maskV1], n)
                self.field[3, maskV2] = eta * power(s3_2[maskV2], n)

                self.field[4, maskU1] = eta * power(s4_1[maskU1], n)
                self.field[4, maskU2] = eta * power(s4_2[maskU2], n)

                self.field[5, maskU1] = eta * power(s5_u1[maskU1], n)
                self.field[5, maskU2] = eta * power(s5_u2[maskU2], n)

                self.field[5, maskV1] += eta * power(s5_v1[maskV1], n)
                self.field[5, maskV2] += eta * power(s5_v2[maskV2], n)

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

                # case 1:
                s3_1 = -(((1+2*n)*power(zmaxv, 1/n)*(q[0]*V*zmaxv-q[2]*h[0]))/(n*q[0]*(power(zmaxv, 2+1/n)+power(-zmaxv+h[0], 2+1/n))))

                s4_1 = -(((1+2*n)*power(zmaxu, 1/n)*(q[0]*U*zmaxu-q[1]*h[0]))/(n*q[0]*(power(zmaxu, 2+1/n)+power(-zmaxu+h[0], 2+1/n))))

                # case 2:
                s3_2 = (power(zmaxv, 1/n)*(1+1/n)*(1+2*n)*(q[2]*q[0]-V)*h[0]) / (n*power(zmaxv-h[0], 2+1/n) +
                                                                                 power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0]))

                s4_2 = (power(zmaxu, 1/n)*(1+1/n)*(1+2*n)*(q[1]*q[0]-U)*h[0]) / (n*power(zmaxu-h[0], 2+1/n) +
                                                                                 power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0]))

                self.field[3, maskV1] = eta * power(s3_1[maskV1], n)
                self.field[3, maskV2] = eta * power(s3_2[maskV2], n)

                self.field[4, maskU1] = eta * power(s4_1[maskU1], n)
                self.field[4, maskU2] = eta * power(s4_2[maskU2], n)

            else:
                # all other types than 'full' consider slip only at the top surface
                self.field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
                self.field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))


def power(base, exponent, mode='real'):

    complex_base = np.ones_like(base, dtype=complex)
    complex_base *= base

    result = np.power(complex_base, exponent)

    return result.real
    # return np.absolute(result)
