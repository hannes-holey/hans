#
# Copyright 2020, 2022 Hannes Holey
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
from scipy.optimize import fsolve

from hans.field import VectorField, TensorField, DoubleTensorField
from hans.material import Material
from hans.gp import GP_stress


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

        # 1D only
        # Xtest = np.vstack([h[0, :, 1], q[0, :, 1], q[1, :, 1]]).T

        # mean, cov = model.predict(Xtest)

        # self.field[0] = mean[:, 0, None] / 2.
        # self.field[1] = mean[:, 1, None] / 2.

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

        self.ncalls = 0

        # Saving and loading model consistently across python versions:
        # m_load = GPRegression(np.array([[], [], []]).T, np.array([[], []]).T, initialize=False)
        # m_load.update_model(False)
        # m_load.initialize_parameter()
        # m_load[:] = np.load('model_save.npy')
        # m_load.update_model(True)
        # self.model = m_load

        # Pickle is inconsitent across python versions!
        # self.model = GPRegression.load_model('/home/hannes/data/2023-05_hans_gpr_stress/shear_50000.0.json.zip')
        # model_diag = GPRegression.load_model('/home/hannes/data/2023-05_hans_gpr_stress/diag_0.1.json.zip')

    def init_gp(self, height, sol, wall):

        if self.gp is not None:

            # 1D only (centerline)
            h = height[0, :, 1]
            dh = height[1, :, 1]
            q = sol[:, :, 1]

            active_learning = {'max_iter': self.disc['Nx'], 'threshold': self.gp['tol']}
            kernel_dict = {'type': 'Mat32',
                           'init_params': [self.gp['var'], self.gp['lh'], self.gp['lrho'], self.gp['lj']],
                           'ARD': True}

            optimizer = {'type': 'bfgs', 'num_restarts': self.gp['num_restarts'], 'verbose': bool(self.gp['verbose'])}

            self.GP = GP_stress(h, dh, self.gp_wall_stress, {}, [wall], active_learning, kernel_dict, optimizer)

            init_ids = [1, ]

            # Initialize
            self.GP.setup(q, init_ids)

    def gp_wall_stress(self, q, h, Ls):

        sb = self.get_bot(q, h, Ls)[4]
        st = self.get_top(q, h, Ls)[4]

        out = np.vstack([sb, st])

        return out

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

        if self.gp is not None:

            if self.ncalls % 2 == 0:
                self.GP.active_learning_step(q[:, :, 1])
            mean, cov = self.GP.predict()

            self.field[4] = mean[:, 0, None]
            self.ncalls += 1

        else:
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

    def get_top(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

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

        return field

    def get_bot(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

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

        return field

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
            zmaxu, zmaxv = self.approximate_zmax(U, V, u_mean, v_mean)
        else:
            zmaxu, zmaxv = self.solve_zmax(U, V, u_mean, v_mean)

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

    def solve_zmax(self, U, V, um, vm):
        """Calculate the location of maximum velocity in z-direction to compute the power law stresses.

        Parameters
        ----------
        U : float
            Sliding velocity of the lower surface in x-direction.
        V : float
            Sliding velocity of the lower surface in y-direction.
        Um : numpy.ndarray
            Array of mean velocities in x-direction (jx/rho).
        Vm : numpy.ndarray
            Array of mean velocities in y-direction (jy/rho).

        Returns
        -------
        numpy.ndarray
            Nondimensional z-coordinate of velocity maximum in x-direction
        numpy.ndarray
            Nondimensional z-coordinate of velocity maximum in y-direction
        """

        Nx, Ny = vm.shape

        if V == 0:
            zmaxv = np.ones_like(vm) * 0.5
        else:
            vn = vm / V
            zmaxv = np.ones_like(vm)
            init = 0.5 - 1 / (12*self.n*(vn - 0.5))

            for i in range(Nx):
                for j in range(Ny):
                    if vn[i, j] >= (1 + self.n) / (1 + 2 * self.n):
                        zmaxv[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(self.n, vn[i, j]))[0]
                    else:
                        zmaxv[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(self.n, vn[i, j]))[0]

        if U == 0:
            zmaxu = np.ones_like(um) * 0.5
        else:
            un = um / U
            zmaxu = np.ones_like(um)
            init = 0.5 - 1 / (12*self.n*(un - 0.5))

            for i in range(Nx):
                for j in range(Ny):
                    if un[i, j] >= (1 + self.n) / (1 + 2 * self.n):
                        zmaxu[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(self.n, un[i, j]))[0]
                    else:
                        zmaxu[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(self.n, un[i, j]))[0]

        # upper and lower bound for asymptotic behavior
        zmaxu = np.minimum(zmaxu, 1e5)
        zmaxu = np.maximum(zmaxu, -1e5)
        zmaxv = np.minimum(zmaxv, 1e5)
        zmaxv = np.maximum(zmaxv, -1e5)

        return zmaxu, zmaxv

    def approximate_zmax(self, U, V, Um, Vm):
        """Approximate the location of maximum velocity in z-direction to compute the power law stresses.
        Exact values have been found numerically for both cases, but a good approximation
        is given by a hyperbola with vertical asymptote at jx/rho = U/2 (Couette profile).

        Parameters
        ----------
        U : float
            Sliding velocity of the lower surface in x-direction.
        V : float
            Sliding velocity of the lower surface in y-direction.
        Um : numpy.ndarray
            Array of mean velocities in x-direction (jx/rho).
        Vm : numpy.ndarray
            Array of mean velocities in y-direction (jy/rho).

        Returns
        -------
        numpy.ndarray
            Nondimensional z-coordinate of velocity maximum in x-direction
        numpy.ndarray
            Nondimensional z-coordinate of velocity maximum in y-direction
        """

        a = 0.5
        bu = -U / (12 * self.n)
        cu = U / 2

        bv = -V / (12 * self.n)
        cv = V / 2

        if V == 0:
            zmaxv = np.ones_like(Vm) * 0.5
        else:
            zmaxv = a + bv / (Vm - cv)

        if U == 0:
            zmaxu = np.ones_like(Um) * 0.5
        else:
            zmaxu = a + bu / (Um - cu)

        # upper and lower bound for asymptotic behavior
        zmaxu = np.minimum(zmaxu, 1e5)
        zmaxu = np.maximum(zmaxu, -1e5)
        zmaxv = np.minimum(zmaxv, 1e5)
        zmaxv = np.maximum(zmaxv, -1e5)

        return zmaxu, zmaxv


def zmax_nleq_case1(zmax, n, un):
    """Definition of nonlinear equation (Case 1) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    # Case1
    return power((1+n)/(n * (-power(zmax, 1+1/n)
                             + power(1. - zmax, 1+1/n))), n) - power(((1+2*n)*(un - zmax))/(n*(power(zmax, 2+1/n)
                                                                                               + power(1. - zmax, 2+1/n))), n)


def zmax_nleq_case2(zmax, n, un):
    """Definition of nonlinear equation (Case 2) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    return 1. - (un*(1+2*n)*(power(zmax, 1+1/n)-power(zmax-1, 1+1/n))) / (power(zmax, 2+1/n)*n - power(zmax-1, 1+1/n)*(zmax*n+(1+n)))


def power(base, exponent):

    complex_base = base.astype('complex128')
    result = np.float_power(complex_base, exponent)

    return result.real


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
        # self.gp = None

        if "PLindex" in self.material.keys():
            raise NotImplementedError

        self.ncalls = 0

        # Saving and loading model consistently across python versions:
        # m_load = GPRegression(np.array([[], [], []]).T, np.array([[], []]).T, initialize=False)
        # m_load.update_model(False)
        # m_load.initialize_parameter()
        # m_load[:] = np.load('model_save.npy')
        # m_load.update_model(True)
        # self.model = m_load

        # Pickle is inconsitent across python versions!
        # self.model = GPRegression.load_model('/home/hannes/data/2023-05_hans_gpr_stress/shear_50000.0.json.zip')
        # model_diag = GPRegression.load_model('/home/hannes/data/2023-05_hans_gpr_stress/diag_0.1.json.zip')

    def init_gp(self, height, sol):

        if self.gp is not None:

            # 1D only (centerline)
            h = height[0, :, 1]
            dh = height[1, :, 1]
            q = sol[:, :, 1]

            active_learning = {'max_iter': self.disc['Nx'], 'threshold': self.gp['tol']}
            kernel_dict = {'type': 'Mat32',
                           'init_params': [self.gp['var'], self.gp['lh'], self.gp['lrho'], self.gp['lj']],
                           'ARD': True}

            optimizer = {'type': 'bfgs', 'num_restarts': self.gp['num_restarts'], 'verbose': bool(self.gp['verbose'])}

            self.GP = GP_stress(h, dh, self.gp_wall_stress, {}, [0, 1], active_learning, kernel_dict, optimizer)

            init_ids = [1, ]

            # Initialize
            self.GP.setup(q, init_ids)

    @property
    def lower(self):
        return self.field[:6]

    @property
    def upper(self):
        return self.field[6:]

    def gp_wall_stress(self, q, h, Ls):

        sb = self.get_bot(q, h, Ls)[4]
        st = self.get_top(q, h, Ls)[4]

        out = np.vstack([sb, st])

        return out

    def set(self, q, h, Ls):
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
        self.set_Newton(q, h, Ls)

    def set_Newton(self, q, h, Ls):
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

        if self.gp is not None:

            if self.ncalls % 2 == 0:
                self.GP.active_learning_step(q[:, :, 1])

            mean, cov = self.GP.predict()

            self.field[4] = mean[:, 0, None]
            self.field[10] = mean[:, 1, None]

            self.ncalls += 1

        else:
            U = self.geometry['U']
            V = self.geometry['V']
            eta = Material(self.material).viscosity(U, V, q[0], h[0])
            zeta = self.material['bulk']

            v1 = zeta + 4 / 3 * eta
            v2 = zeta - 2 / 3 * eta

            if self.surface is None or self.surface["type"] == "full":
                # all other types than 'full' consider slip only at the top surface

                # Bottom: indices 0-5

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

                # Top wall: indices 6-11

                self.field[6] = -2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v1*h[1]
                                    + ((V*q[0] - 3*q[2]) * h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                       + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v2) * (h[0] + Ls)/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[7] = -2*(h[0] + Ls)*(((U*q[0] - 3*q[1])*h[0]**2
                                                 + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*v2*h[1]
                                                + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                                    + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2]*v1) / (h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[8] = -2*(h[0] + Ls)*v2*(((U*q[0] - 3*q[1])*h[0]**2
                                                    + 3*Ls*(U*q[0] - 4*q[1])*h[0] + 6*Ls**2*(U*q[0] - 2*q[1]))*h[1]
                                                   + ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 4*q[2])*h[0]
                                                      + 6*Ls**2*(V*q[0] - 2*q[2]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))

                self.field[9] = 2*(q[0]*V*h[0] - 3*h[0]*q[2] - 6*q[2]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

                self.field[10] = 2*(q[0]*U*h[0] - 3*h[0]*q[1] - 6*q[1]*Ls)*eta/(q[0]*(h[0]**2 + 8*Ls*h[0] + 12*Ls**2))

                self.field[11] = -2*eta*(h[0] + Ls)*(((V*q[0] - 3*q[2])*h[0]**2
                                                      + 3*Ls*(V*q[0] - 4*q[2])*h[0] + 6*Ls**2*(V*q[0] - 2*q[2]))*h[1]
                                                     + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 4*q[1])*h[0]
                                                        + 6*Ls**2*(U*q[0] - 2*q[1]))*h[2])/(h[0]*(h[0] + 2*Ls)**2*q[0]*(h[0] + 6*Ls))
            else:
                # all other types than 'full' consider slip only at the top surface

                # Bottom: indices 0-5

                self.field[3] = -2*(6*q[0]*Ls*V + 2*q[0]*V*h[0] - 6*q[2]*Ls - 3*h[0]*q[2])*eta/(h[0]*q[0]*(4*Ls + h[0]))
                self.field[4] = -2*(6*q[0]*Ls*U + 2*q[0]*U*h[0] - 6*q[1]*Ls - 3*h[0]*q[1])*eta/(h[0]*q[0]*(4*Ls + h[0]))

                # Top wall: indices 6-11

                self.field[6] = (-6*v1*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                                 - 6*v2*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                                + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

                self.field[7] = (-6*v2*((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1]
                                 - 6*v1*h[2] * ((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0]
                                                + 6*Ls**2*(V*q[0] - q[2])))/(3*h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

                self.field[8] = -2*v2*(((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0] + 6*Ls**2*(U*q[0] - q[1]))*h[1] +
                                       h[2]*((V*q[0] - 3*q[2])*h[0]**2
                                             + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2])))/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

                self.field[9] = 2*(V*q[0] - 3*q[2])*eta/(q[0]*(4*Ls + h[0]))

                self.field[10] = 2*(U*q[0] - 3*q[1])*eta/(q[0]*(4*Ls + h[0]))

                self.field[11] = -2*eta*(((V*q[0] - 3*q[2])*h[0]**2 + 3*Ls*(V*q[0] - 2*q[2])*h[0] + 6*Ls**2*(V*q[0] - q[2]))*h[1]
                                         + ((U*q[0] - 3*q[1])*h[0]**2 + 3*Ls*(U*q[0] - 2*q[1])*h[0]
                                            + 6*Ls**2*(U*q[0] - q[1]))*h[2])/(h[0]*(4*Ls + h[0])*q[0]*(Ls + h[0]))

    def get_top(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

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

        return field

    def get_bot(self, q, h, Ls):

        U = self.geometry['U']
        V = self.geometry['V']
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material['bulk']

        v1 = zeta + 4 / 3 * eta
        v2 = zeta - 2 / 3 * eta

        field = np.zeros((6,)+q.shape[1:])

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

        return field
