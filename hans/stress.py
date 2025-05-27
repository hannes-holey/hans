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
from unittest.mock import Mock

from hans.field import VectorField, TensorField, DoubleTensorField
from hans.material import Material

from hans.models.viscous_stress_newton import stress_bottom, stress_top, stress_avg
from hans.models.viscous_stress_powerlaw import stress_powerlaw_bottom, stress_powerlaw_top
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
            U = self.geometry["U"]
            V = self.geometry["V"]
            eta = Material(self.material).viscosity(U, V, q[0], h[0])
            zeta = self.material["bulk"]

            if self.surface is None or self.surface["type"] == "full":
                self.field = stress_avg(q, h, U, V, eta, zeta, Ls, slip="both")
            else:
                self.field = stress_avg(q, h, U, V, eta, zeta, Ls, slip="top")

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

        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material["bulk"]

        if self.surface is None or self.surface["type"] == "full":
            return stress_avg(q, h, U, V, eta, zeta, Ls, slip="both")
        else:
            return stress_avg(q, h, U, V, eta, zeta, Ls, slip="top")


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

            active_learning = {"max_iter": self.disc["Nx"], "threshold": self.gp["tol"]}
            kernel_dict = {"type": "Mat32", "init_params": [
                self.gp["var"], self.gp["lh"], self.gp["lrho"], self.gp["lj"]], "ARD": True}

            optimizer = {"type": "bfgs", "num_restarts": self.gp["num_restarts"], "verbose": bool(self.gp["verbose"])}

            self.GP = GP_stress(h, dh, self.gp_wall_stress, {}, [wall], active_learning, kernel_dict, optimizer)

            init_ids = [self.disc["Nx"] // 4, self.disc["Nx"] // 2, 3 * self.disc["Nx"] // 4]

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
            U = self.geometry["U"]
            V = self.geometry["V"]
            eta = Material(self.material).viscosity(U, V, q[0], h[0])
            zeta = self.material["bulk"]

            if bound == "top":
                if self.surface is None or self.surface["type"] == "full":
                    # all other types than 'full' consider slip only at the top surface
                    self.field = stress_top(q, h, U, V, eta, zeta, Ls, slip="both")
                else:
                    self.field = stress_top(q, h, U, V, eta, zeta, Ls, slip="top")

            elif bound == "bottom":
                if self.surface is None or self.surface["type"] == "full":
                    # surface type "full" means that both surface are slippery with equal slip length
                    self.field = stress_bottom(q, h, U, V, eta, zeta, Ls, slip="both")
                else:
                    self.field = stress_bottom(q, h, U, V, eta, zeta, Ls, slip="top")

    def get_top(self, q, h, Ls):
        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material["bulk"]

        # only no slip
        return stress_top(q, h, U, V, eta, zeta, Ls, slip="both")

    def get_bot(self, q, h, Ls):

        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material["bulk"]

        # only no slip
        return stress_bottom(q, h, U, V, eta, zeta, Ls, slip="both")

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

        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = self.material["shear"]
        n = self.n

        if bound == "top":
            self.field = stress_powerlaw_top(q, h, U, V, eta, n, method=self.material['PLmethod'])

        elif bound == "bottom":
            self.field = stress_powerlaw_bottom(q, h, U, V, eta, n, method=self.material['PLmethod'])


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

    def init_gp(self, q, db):

        if self.gp is not None:

            active_learning = {
                "max_iter": self.disc["Nx"],
                "threshold": self.gp["tol"],
                "start": self.gp["start"],
                "Ninit": self.gp["Ninit"],
                "alpha": self.gp["alpha"],
                "sampling": self.gp["sampling"],
            }

            kernel_dict = {"type": "Mat32", "init_params": None, "ARD": True}

            noise = {"type": "Gaussian", "fixed": bool(self.gp["fix"]), "variance": self.gp["sns"]}

            optimizer = {"type": "bfgs", "num_restarts": self.gp["num_restarts"], "verbose": bool(self.gp["verbose"])}

            if q.shape[-1] > 3:
                # 2D
                self.GP = GP_stress2D(db, active_learning, kernel_dict, optimizer, noise)
                # q = sol[:, :, :]  # 1D only
            else:
                # 1D
                self.GP = GP_stress(db, active_learning, kernel_dict, optimizer, noise)
                # q = sol[:, :, 1]

            self.GP.setup(q)
        else:
            self.GP = Mock()
            self.GP.dbsize = 0

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

        if self.gp is not None and self.ncalls > 2 * self.gp["start"]:

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

        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material["bulk"]

        if self.surface is None or self.surface["type"] == "full":
            return stress_top(q, h, U, V, eta, zeta, Ls, slip="both")
        else:
            return stress_top(q, h, U, V, eta, zeta, Ls, slip="top")

    def get_bot(self, q, h, Ls):

        U = self.geometry["U"]
        V = self.geometry["V"]
        eta = Material(self.material).viscosity(U, V, q[0], h[0])
        zeta = self.material["bulk"]

        if self.surface is None or self.surface["type"] == "full":
            return stress_bottom(q, h, U, V, eta, zeta, Ls, slip="both")
        else:
            return stress_bottom(q, h, U, V, eta, zeta, Ls, slip="top")
