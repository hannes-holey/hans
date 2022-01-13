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
        """This class contains the averaged viscous stress tensor components (xx, yy, xy).
        Derived from VectorField.

        Parameters
        ----------
        disc : dict
            Discretization parameters.
        geometry : dict
            Geometry parameters.
        material : dict
            Material parameters.
        surface : dict
            Surface parameters (the default is None).

        """

        super().__init__(disc)

        self.geometry = geometry
        self.material = material
        self.surface = surface

        try:
            self.n = self.material["PLindex"]
        except KeyError:
            self.n = 1

    def set(self, q, h, Ls):
        """Set method for Newtonian stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        Ls : numpy.ndarray
            Field of slip lengths.

        """

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

    def __init__(self, disc, geometry, material, surface=None):
        """This class contains all viscous stress tensor components
        evaluated at the top and bottom respectively. Derived from TensorField.

        Parameters
        ----------
        disc : dict
            Discretization parameters.
        geometry : dict
            Geometry parameters.
        material : dict
            Material parameters.
        surface : dict
            Surface parameters (the default is None).

        """

        super().__init__(disc)

        self.geometry = geometry
        self.material = material
        self.surface = surface

        try:
            self.n = self.material["PLindex"]
        except KeyError:
            self.n = 1

    def num_param(self, Um, Vm):

        # values for b
        # 0.5 -0.714512478467
        # 0.6 -0.610142431834
        # 0.7 -0.537878481715
        # 0.8 -0.485438588613
        # 0.9 -0.446312057535
        # 1.0 -0.416666666667
        # 1.1 -0.393528444050
        # 1.2 -0.375088307644
        # 1.3 -0.363378160640
        # 1.4 -0.353426202841
        # 1.5 -0.337945369360
        # 1.6 -0.340883666078
        # 1.7 -0.337488427417
        # 1.8 -0.335670825927
        # TODO: automatic selection, more careful fitting

        # fitted parameters for n=1 (make sure that material.PLindex = 1.)
        a = 0.50
        b = -0.41666667
        c = 2.5

        # set zmax to -1000 for Um<U/2, TODO: negative Um, Vm
        zmaxu = -1e3 * np.ones_like(Um)
        zmaxv = -1e3 * np.ones_like(Vm)

        zmaxu[Um > c] = a + b / (Um[Um > c] - c)
        zmaxv[Vm > c] = a + b / (Vm[Vm > c] - c)

        return zmaxu, zmaxv

    def set(self, q, h, Ls, bound):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        n = self.n

        meanu = q[1] / q[0]
        meanv = q[2] / q[0]

        zmaxu, zmaxv = self.num_param(meanu, meanv)

        zmaxu *= h[0]
        zmaxv *= h[0]

        U_crit = meanu * (1 + 2 * n) / (1 + n)
        V_crit = meanv * (1 + 2 * n) / (1 + n)

        # case 1: U_nondim <= UV_crit; V_nondim <= UV_crit
        maskU1 = U <= U_crit
        maskV1 = V <= V_crit

        # case 2: U_nondim > UV_crit; V_nondim > UV_crit
        maskU2 = U > U_crit
        maskV2 = V > V_crit

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
                s0_2 = -2*((1+2*n)*(q[1]-U*q[0])*(power(zmaxu, 1+1/n)-power(zmaxu-h[0], 1+1/n)) *
                           (power(zmaxu, 2+1/n)*n-power(zmaxu - h[0], 1+1/n)
                            * (zmaxu*n+(1+n)*h[0]))*h[1])/(q[0]*(n*power(zmaxu-h[0], 2+1/n) + power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0]))**2)

                s1_2 = -2*((1+2*n)*(q[2]-V*q[0])*(power(zmaxv, 1+1/n)-power(zmaxv-h[0], 1+1/n)) *
                           (power(zmaxv, 2+1/n)*n-power(zmaxv - h[0], 1+1/n)
                            * (zmaxv*n+(1+n)*h[0]))*h[2])/(q[0]*(n*power(zmaxv-h[0], 2+1/n) + power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0]))**2)

                s3_2 = ((1+1/n)*(1+2*n)*(q[2]-V*q[0])*power(zmaxv-h[0], 1/n)*h[0]) / \
                    (q[0]*(n*power(zmaxv-h[0], 2+1/n)+power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0])))

                s4_2 = ((1+1/n)*(1+2*n)*(q[1]-U*q[0])*power(zmaxu-h[0], 1/n)*h[0]) / \
                    (q[0]*(n*power(zmaxu-h[0], 2+1/n)+power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0])))

                s5_u2 = (1+2*n)/q[0]*(-(((1+2*n)*(q[1]-U*q[0])*(power(zmaxu, 1+1/n)
                                                                - power(zmaxu-h[0], 1+1/n))**2 * h[0]*h[2]) /
                                        (n*power(zmaxu-h[0], 2+1/n) + power(zmaxu, 1+1/n)*(-zmaxu * n+(1+2*n)*h[0]))**2)
                                      + ((q[1]-U*q[0])*(power(zmaxu, 1+1/n)-power(zmaxu-h[0], 1+1/n))*h[2]) /
                                      (n*power(zmaxu-h[0], 2+1/n)+power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0])))

                s5_v2 = (1+2*n)/q[0]*(-(((1+2*n)*(q[2]-V*q[0])*(power(zmaxv, 1+1/n)
                                                                - power(zmaxv-h[0], 1+1/n))**2 * h[0]*h[1]) /
                                        (n*power(zmaxv-h[0], 2+1/n) + power(zmaxv, 1+1/n)*(-zmaxv * n+(1+2*n)*h[0]))**2)
                                      + ((q[2]-V*q[0])*(power(zmaxv, 1+1/n)-power(zmaxv-h[0], 1+1/n))*h[1]) /
                                      (n*power(zmaxv-h[0], 2+1/n)+power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0])))

                self.field[0, maskU1] = eta * power(s0_1[maskU1], n)
                self.field[0, maskU2] = eta * power(s0_2[maskU2], n)

                self.field[1, maskV1] = eta * power(s1_1[maskV1], n)
                self.field[1, maskV2] = eta * power(s1_2[maskV2], n)

                self.field[3, maskV1] = eta * power(s3_1[maskV1], n)
                self.field[3, maskV2] = eta * power(s3_2[maskV2], n)

                # TODO: probably wrong wall shear stress
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
                s3_2 = (power(zmaxv, 1/n)*(1+1/n)*(1+2*n)*(q[2]-V*q[0])*h[0]) / (q[0]*(n*power(zmaxv-h[0], 2+1/n) +
                                                                                       power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0])))

                s4_2 = (power(zmaxu, 1/n)*(1+1/n)*(1+2*n)*(q[1]-U*q[0])*h[0]) / (q[0]*(n*power(zmaxu-h[0], 2+1/n) +
                                                                                       power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0])))

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

    result = np.float_power(complex_base, exponent)

    return result.real
