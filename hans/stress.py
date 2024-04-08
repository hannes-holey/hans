#
# Copyright 2020, 2024 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np

from hans.field import VectorField, TensorField, DoubleTensorField
from hans.material import Material
from hans.special.powerlaw_fluid import solve_zmax, approximate_zmax
from hans.tools import power
from hans.multiscale.gp import GP_stress, GP_stress2D


class SymStressField2D(VectorField):

    def __init__(self, disc, geometry, material, surface=None, gp=None):
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
        gp : dict
            Gaussaian process parameters (the default is None).
        """

        super().__init__(disc)

        self.geometry = geometry
        self.material = material
        self.surface = surface
        self.gp = gp

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

        if self.gp is not None:
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

    def get(self, q, h, Ls):
        """Set method for Newtonian stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        Ls : numpy.ndarray
            Field of slip lengths.
                gp : dict
            Gaussaian process parameters (the default is None).

        """

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros_like(q)

        if self.surface is None or self.surface["type"] == "full":
            field[0] = (-3*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v1*h[1]
                        - 3*((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                             + 6*Ls**2*(V*q[0] - 2*q[2])) * h[2]*v2)/(3*h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

            field[1] = (-3*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v2*h[1]
                        - 3*((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                             + 6*Ls**2*(V*q[0] - 2*q[2])) * h[2]*v1)/(3*h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

            field[2] = -eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls ** 2*(V*q[0] - 2*q[2]))*h[1]
                             + ((U*q[0] - 3*q[1])*h[0] ** 2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))
                             * h[2])/(h[0]*q[0]*(h[0] + 6*Ls)*(h[0] + 2*Ls))

        else:
            # all other types than 'full' consider slip only at the top surface
            field[0] = (-3*v1*((U*q[0] - 3*q[1])*h[0]**2
                               + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                        - 3*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            field[1] = (-3*v2*((U*q[0] - 3*q[1])*h[0]**2
                               + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                        - 3*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            field[2] = -eta*(((V*q[0] - 3*q[2])*h[0]**2
                              + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                             + ((U*q[0] - 3*q[1]) * h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0]
                                + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

        return field


class SymStressField3D(TensorField):

    def __init__(self, disc, geometry, material, surface=None, gp=None):
        """This class contains all viscous stress tensor components
        evaluated at the top and bottom respectively. Derived from TensorField.

        This is legacy code. Used to store the wall stress in separate instances
        for top and bottom wall. Now replaced by single instance of WallStressField3D. 
        Currently analytic expressions for the stress of a power law fluid  at the
        solid-liquid interface is not implemented elsewhere.

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

        self.gp = gp
        # self.gp = None

        try:
            self.n = self.material["PLindex"]
            self.PL = True
        except KeyError:
            self.PL = False

    def set(self, q, h, Ls, bound):
        """Wrapper function around set methods for wall stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        Ls : numpy.ndarray
            Field of slip lengths.
        bound : str
            Flag for bottom or top wall.

        """

        if self.PL:
            self.set_PowerLaw(q, h, bound)
        else:
            self.set_Newton(q, h, Ls, bound)

    def set_Newton(self, q, h, Ls, bound):
        """Set method for Newtonian stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        Ls : numpy.ndarray
            Field of slip lengths.
        bound : str
            Flag for bottom or top wall.

        """

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        if bound == "top":
            if self.surface is None or self.surface["type"] == "full":
                # all other types than 'full' consider slip only at the top surface
                self.field[0] = -2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v1*h[1]
                                    + ((V*q[0] - 3*q[2]) * h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v2) * (h[0] + Ls)/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[1] = -2*(h[0] + Ls)*(((U*q[0] - 3*q[1])*h[0]**2
                                                 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v2*h[1]
                                                + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                                    + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v1) / (h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[2] = -2*(h[0] + Ls)*v2*(((U*q[0] - 3*q[1])*h[0]**2
                                                    + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*h[1]
                                                   + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                                      + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[3] = 2*(q[0]*V*h[0] - 3*h[0]*q[2] - 6*q[2]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

                self.field[4] = 2*(q[0]*U*h[0] - 3*h[0]*q[1] - 6*q[1]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

                self.field[5] = -2*eta*(h[0] + Ls)*(((V*q[0] - 3*q[2])*h[0]**2
                                                     + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls**2*(V*q[0] - 2*q[2]))*h[1]
                                                    + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0]
                                                       + 6*Ls**2*(U*q[0] - 2*q[1]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))
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
                self.field[0] = -2*(((U*q[0] - 3*q[1])*h[0]**2
                                     + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*v1*h[1]
                                    + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v2)*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

                self.field[1] = -2*(((U*q[0] - 3*q[1])*h[0]**2
                                     + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*v2*h[1]
                                    + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*(zeta + (4*eta)/3))*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

                self.field[2] = -2*v2*Ls*(((U*q[0] - 3*q[1])*h[0]**2
                                           + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*h[1]
                                          + ((V*q[0] - 3*q[2])*h[0] ** 2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                             + 6*Ls ** 2*(V*q[0] - 2*q[2]))*h[2])/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

                self.field[3] = -2*(2*q[0]*V*h[0] + 6*q[0]*V*Ls - 3*h[0]*q[2] - 6*q[2]*Ls) * \
                    eta/(q[0]*(h[0] ** 2 + 8*Ls*h[0] + 12*Ls ** 2))

                self.field[4] = -2*(2*q[0]*U*h[0] + 6*q[0]*U*Ls - 3*h[0]*q[1] - 6*q[1]*Ls) * \
                    eta/(q[0]*(h[0] ** 2 + 8*Ls*h[0] + 12*Ls ** 2))

                self.field[5] = -2*eta*(((V*q[0] - 3*q[2])*h[0]**2
                                         + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls**2*(V*q[0] - 2*q[2]))*h[1]
                                        + ((U*q[0] - 3*q[1])*h[0] ** 2 + 3*Ls*(U*q[0] - 4*q[1])*h[0]
                                           + 6*Ls ** 2*(U*q[0] - 2*q[1]))*h[2])*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))
            else:
                # all other types than 'full' consider slip only at the top surface
                self.field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
                self.field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))

    def set_PowerLaw(self, q, h, bound):
        """Set method for power law stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        bound : str
            Flag for bottom or top wall.

        """

        U = self.geometry['U']
        V = self.geometry['V']
        eta = self.material["shear"]

        n = self.n

        u_mean = q[1] / q[0]
        v_mean = q[2] / q[0]

        if self.material["PLmethod"] == "approx":
            zmaxu, zmaxv = approximate_zmax(U, V, u_mean, v_mean, n)
        else:
            zmaxu, zmaxv = solve_zmax(U, V, u_mean, v_mean, n)

        zmaxu *= h[0]
        zmaxv *= h[0]

        U_crit = u_mean * (1 + 2 * n) / (1 + n)
        V_crit = v_mean * (1 + 2 * n) / (1 + n)

        # case 1: U <= U_crit; V <= V_crit
        maskU1 = U <= U_crit
        maskV1 = V <= V_crit

        # case 2: U > U_crit; V > V_crit
        maskU2 = U > U_crit
        maskV2 = V > V_crit

        if bound == "top":
            # case 1
            s0_1 = -2 * ((1+2*n)*power(-zmaxu+h[0], 1/n)*(q[0]*U*zmaxu-q[1]*h[0])*h[1]) / (n*q[0]*(power(zmaxu, 2+1/n)
                                                                                                   + power(-zmaxu+h[0], 2+1/n)))

            s1_1 = -2 * ((1+2*n)*power(-zmaxv+h[0], 1/n)*(q[0]*V*zmaxv-q[2]*h[0])*h[2]) / (n*q[0]*(power(zmaxv, 2+1/n)
                                                                                                   + power(-zmaxv+h[0], 2+1/n)))

            s3_1 = ((1+2*n)*power(-zmaxv+h[0], 1/n)*(q[0]*V*zmaxv-q[2]*h[0]))/(n*q[0]*(power(zmaxv, 2+1/n)+power(-zmaxv+h[0], 2+1/n)))

            s4_1 = ((1+2*n)*power(-zmaxu+h[0], 1/n)*(q[0]*U*zmaxu-q[1]*h[0]))/(n*q[0]*(power(zmaxu, 2+1/n)+power(-zmaxu+h[0], 2+1/n)))

            s5_u1 = -s4_1 * h[2]
            s5_v1 = -s3_1 * h[1]

            # case 2
            s0_2 = -2*((1+2*n)*(q[1]-U*q[0])*(power(zmaxu, 1+1/n)-power(zmaxu-h[0], 1+1/n)) *
                       (power(zmaxu, 2+1/n)*n-power(zmaxu - h[0], 1+1/n)
                        * (zmaxu*n+(1+n)*h[0]))*h[1])/(q[0]*(n*power(zmaxu-h[0], 2+1/n)
                                                             + power(zmaxu, 1+1/n)*(-zmaxu*n+(1+2*n)*h[0]))**2)

            s1_2 = -2*((1+2*n)*(q[2]-V*q[0])*(power(zmaxv, 1+1/n)-power(zmaxv-h[0], 1+1/n)) *
                       (power(zmaxv, 2+1/n)*n-power(zmaxv - h[0], 1+1/n)
                        * (zmaxv*n+(1+n)*h[0]))*h[2])/(q[0]*(n*power(zmaxv-h[0], 2+1/n)
                                                             + power(zmaxv, 1+1/n)*(-zmaxv*n+(1+2*n)*h[0]))**2)

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

            self.field[4, maskU1] = eta * power(s4_1[maskU1], n)
            self.field[4, maskU2] = eta * power(s4_2[maskU2], n)

            self.field[5, maskU1] = eta * power(s5_u1[maskU1], n)
            self.field[5, maskU2] = eta * power(s5_u2[maskU2], n)

            self.field[5, maskV1] += eta * power(s5_v1[maskV1], n)
            self.field[5, maskV2] += eta * power(s5_v2[maskV2], n)

        elif bound == "bottom":
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


class WallStressField3D(DoubleTensorField):

    def __init__(self, disc, geometry, material, surface=None, gp=None):
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

        self.gp = gp

        if "PLindex" in self.material.keys():
            raise NotImplementedError

        self.ncalls = 0

    def init_gp(self, q, db):

        if q.shape[-1] > 3:
            # 2D
            self.GP = GP_stress2D(db, self.gp)
        else:
            # 1D
            self.GP = GP_stress(db, self.gp)

        self.GP.setup(q)

    @property
    def lower(self):
        return self.field[:6]

    @property
    def upper(self):
        return self.field[6:]

    def gp_wall_stress(self, q, h, Ls):

        sb = self.get_bot(q, h, Ls)  # [4]
        st = self.get_top(q, h, Ls)  # [4]

        out = np.vstack([sb, st])

        return out

    def set(self, q, h, Ls):
        """Setter method for wall stress tensor components.

        Parameters
        ----------
        q : numpy.ndarray
            Field of conserved variables.
        h : numpy.ndarray
            Field of height and height gradients.
        Ls : numpy.ndarray
            Field of slip lengths.
        """

        if self.gp is not None:

            if self.ncalls % 2 == 0:
                mean, cov = self.GP.active_learning_step(q)
            else:
                mean, cov = self.GP.predict()

            if self.GP.ndim == 2:
                mean[0] *= np.sign(q[2])
                mean[2] *= np.sign(q[2])

            self.field[self.GP.Ymask[1:]] = mean

            self.ncalls += 1

        else:
            self.field[:6] = self.get_bot(q, h, Ls)
            self.field[6:] = self.get_top(q, h, Ls)
            self.ncalls += 1

    def get_top(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

        if self.surface is None or self.surface["type"] == "full":

            # all other types than 'full' consider slip only at the top surface
            field[0] = -2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v1*h[1]
                           + ((V*q[0] - 3*q[2]) * h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                              + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v2) * (h[0] + Ls)/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

            field[1] = -2*(h[0] + Ls)*(((U*q[0] - 3*q[1])*h[0]**2
                                        + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v2*h[1]
                                       + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                          + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v1) / (h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

            field[2] = -2*(h[0] + Ls)*v2*(((U*q[0] - 3*q[1])*h[0]**2
                                           + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*h[1]
                                          + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                             + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

            field[3] = 2*(q[0]*V*h[0] - 3*h[0]*q[2] - 6*q[2]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

            field[4] = 2*(q[0]*U*h[0] - 3*h[0]*q[1] - 6*q[1]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

            field[5] = -2*eta*(h[0] + Ls)*(((V*q[0] - 3*q[2])*h[0]**2
                                            + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls**2*(V*q[0] - 2*q[2]))*h[1]
                                           + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0]
                                              + 6*Ls**2*(U*q[0] - 2*q[1]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

        else:
            field[0] = (-6*v1*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                        - 6*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            field[1] = (-6*v2*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                        - 6*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            field[2] = -2*v2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1] +
                              h[2]*((V*q[0] - 3*q[2])*h[0]**2
                                    + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2])))/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

            field[3] = 2*(V*q[0] - 3*q[2])*eta/(q[0]*(4*Ls + h[0]))

            field[4] = 2*(U*q[0] - 3*q[1])*eta/(q[0]*(4*Ls + h[0]))

            field[5] = -2*eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                               + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0]
                                  + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

        return field

    def get_bot(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

        if self.surface is None or self.surface["type"] == "full":

            field[0] = -2*(((U*q[0] - 3*q[1])*h[0]**2
                            + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*v1*h[1]
                           + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                              + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v2)*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

            field[1] = -2*(((U*q[0] - 3*q[1])*h[0]**2
                            + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*v2*h[1]
                           + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                              + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*(zeta + (4*eta)/3))*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

            field[2] = -2*v2*Ls*(((U*q[0] - 3*q[1])*h[0]**2
                                  + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls ** 2*(U*q[0] - 2*q[1]))*h[1]
                                 + ((V*q[0] - 3*q[2])*h[0] ** 2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                    + 6*Ls ** 2*(V*q[0] - 2*q[2]))*h[2])/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

            field[3] = -2*(2*q[0]*V*h[0] + 6*q[0]*V*Ls - 3*h[0]*q[2] - 6*q[2]*Ls)*eta/(q[0]*(h[0] ** 2 + 8*Ls*h[0] + 12*Ls ** 2))

            field[4] = -2*(2*q[0]*U*h[0] + 6*q[0]*U*Ls - 3*h[0]*q[1] - 6*q[1]*Ls)*eta/(q[0]*(h[0] ** 2 + 8*Ls*h[0] + 12*Ls ** 2))

            field[5] = -2*eta*(((V*q[0] - 3*q[2])*h[0]**2
                                + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls**2*(V*q[0] - 2*q[2]))*h[1]
                               + ((U*q[0] - 3*q[1])*h[0] ** 2 + 3*Ls*(U*q[0] - 4*q[1])*h[0]
                                  + 6*Ls ** 2*(U*q[0] - 2*q[1]))*h[2])*Ls/(h[0]*(h[0] + 2*Ls) ** 2*q[0]*(h[0] + 6*Ls))

        else:
            field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
            field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))

        return field
