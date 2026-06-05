#
# Copyright 2025 Hannes Holey
#           2025 Christoph Huber
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
import numpy.typing as npt
import jax.numpy as jnp
from jax import vmap, grad, jit
from jax import Array
from typing import Optional, Tuple, Any
from muGrid import Field

from .gp import GaussianProcessSurrogate
from .gp import multi_in_single_out, multi_in_multi_out
from .pressure import eos_pressure, eos_rho
from .viscous import (stress_bottom, stress_top, stress_avg,
                      stress_top_xz, stress_bottom_xz,
                      stress_top_yz, stress_bottom_yz,
                      get_shear_viscosity)
from .sound import eos_sound_velocity
from .viscosity import piezoviscosity, shear_thinning_factor, shear_rate_avg

NDArray = npt.NDArray[np.floating]
JAXArray = Array


class WallStress(GaussianProcessSurrogate):
    """
    Wall stress model (wall shear/stress in xz or yz direction).

    This class can operate in two modes:
    - Deterministic: compute wall/boundary stresses from viscous models.
    - GP-based surrogate: train/predict wall stress using GaussianProcessSurrogate.

    """

    def __init__(
        self,
        fc: Any,
        prop: dict,
        geo: dict,
        direction: str = 'x',
        data: Optional[Any] = None,
        gp: Optional[dict] = None
    ) -> None:
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        direction : {'x', 'y'}, optional
            Direction of the wall stress ('x' -> xz component (default), 'y' -> yz component).
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = fc.real_field(f'wall_stress_{direction}z', (12,))
        self.__pressure = Field(fc.get_real_field('pressure'))
        self.__x = Field(fc.get_real_field('x'))
        self.__y = Field(fc.get_real_field('y'))

        self.geo = geo
        self.prop = prop
        self.name = f'{direction}z'

        self._out_index = {'x': 4, 'y': 3}[direction]

        if gp is not None:
            self.active_dims = {'x': gp.get('active_dims_x', [0, 1, 3]),
                                'y': gp.get('active_dims_y', [0, 2, 3])}[direction]

            self.__field_variance = fc.real_field(f'wall_stress_{direction}z_var')

            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.pause_steps = gp['pause_steps']
            self.is_gp_model = True
            self.use_active_learning = gp['active_learning']
            self.build_gp = multi_in_multi_out
        else:
            self.is_gp_model = False
            self.use_active_learning = False

        super().__init__(fc, data)

    # -------------------------
    # Properties
    # -------------------------
    @property
    def full(self) -> NDArray:
        """
        Full wall stress field including upper and lower components.

        Returns
        -------
        ndarray
            Array holding 12 components for wall stress (6 lower + 6 upper).
        """
        return self.__field.pg

    @property
    def upper(self) -> NDArray:
        """
        Upper-wall stress slice.

        Returns
        -------
        ndarray
            Upper half of the wall stress components (6 entries).
        """
        return self.__field.pg[6:]

    @property
    def lower(self) -> NDArray:
        """
        Lower-wall stress slice.

        Returns
        -------
        ndarray
            Lower half of the wall stress components (6 entries).
        """
        return self.__field.pg[:6]

    @property
    def variance(self) -> NDArray:
        """
        Variance of the shear stress field

        """
        return self.__field_variance.pg

    @property
    def pressure(self) -> NDArray:
        """
        Local pressure field.

        Returns
        -------
        ndarray
            Pressure field from the field collection.
        """
        return self.__pressure.pg

    @property
    def dp_dx(self) -> NDArray:
        """
        Partial derivative of pressure with respect to x (∂p/∂x).

        Returns
        -------
        ndarra
            Gradient along x computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__x.pg[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """
        Partial derivative of pressure with respect to y (∂p/∂y).

        Returns
        -------
        ndarray
            Gradient along y computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__y.pg[0, :], axis=1)

    @property
    def Xtest(self) -> Tuple[JAXArray, JAXArray]:
        """
        Test inputs for GP prediction.

        Returns
        -------
        tuple of jax.Array
            (X, flag) where X contains duplicated and scaled features and flag
            indicates which output (lower/upper) the sample corresponds to.
        """
        X = jnp.concatenate([(self._Xtest / self.database.X_scale)[:, self.active_dims],
                             (self._Xtest / self.database.X_scale)[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def Xtrain(self) -> Tuple[JAXArray, JAXArray]:
        """
        Training inputs for GP (duplicated to account for two outputs).

        Returns
        -------
        tuple of jax.Array
            (X, flag) where X contains duplicated input rows and flag labels.
        """
        X = jnp.concatenate([self.database.Xtrain[:, self.active_dims],
                             self.database.Xtrain[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def _Ytrain(self) -> JAXArray:
        """
        Training outputs for GP corresponding to the lower and upper wall stress.

        Returns
        -------
        jax.Array
            Concatenated array of training outputs (lower then upper).
        """
        return jnp.concatenate([self.database._Ytrain[:self.last_fit_train_size, self._out_index + 1],
                                self.database._Ytrain[:self.last_fit_train_size, self._out_index + 7]])

    @property
    def Ytrain(self) -> JAXArray:
        """
        Normalized training outputs for GP corresponding to the lower and upper wall stress.

        Returns
        -------
        jax.Array
            Concatenated array of training outputs (lower then upper).
        """
        return self._Ytrain / self.Yscale

    @property
    def Yscale(self) -> JAXArray:
        """
        Output scaling factor used for normalization.

        Returns
        -------
        jax.Array
            Scalar-like array representing the maximum of selected Y scales.
        """
        indices = jnp.array([self._out_index + 1,
                             self._out_index + 7], dtype=int)

        return jnp.max(self.database.Y_scale[indices])

    @property
    def Yerr(self) -> JAXArray:
        """
        Observational error (normalized by Yscale).

        Returns
        -------
        jax.Array
            Observation noise standard deviation normalized by Yscale.
        """

        Yerr_all = jnp.concatenate([self.database._Ytrain_err[:self.last_fit_train_size, self._out_index + 1],
                                    self.database._Ytrain_err[:self.last_fit_train_size, self._out_index + 7]])

        return jnp.mean(Yerr_all / self.Yscale)

    @property
    def kernel_variance(self) -> JAXArray:
        """Return kernel variance (JAX scalar or array)."""
        return self.gp.kernel.kernels[0].kernel1.value

    @property
    def kernel_lengthscale(self) -> JAXArray:
        """Return kernel lengthscale(s)."""
        return self.gp.kernel.kernels[0].kernel2.scale

    @property
    def obs_stddev(self) -> JAXArray:
        """Observation standard deviation (normalized)."""
        return self.Yerr

    # -------------------------
    # Update
    # -------------------------
    def init(self) -> None:
        """Run first training and inference."""
        if self.is_gp_model:
            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.std(self.Xtrain[0], axis=0))
            }

            self._train()
            self._infer()

    def update(self,
               predictor: bool = False,
               compute_var: bool = False) -> None:
        """
        Update wall stress: compute deterministic stresses and, if enabled,
        perform GP prediction and place predicted mean and variance into the
        appropriate field entries.

        Parameters
        ----------
        predictor : bool, optional
            Whether this update is part of the predictor stage.
        compute_var : bool, optional
            Flag for re-computing the variance (the default is False which uses
            the stored variance from previous steps).
        """

        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure if not self.prop['EOS'] == 'Bayada' else self.solution[0],
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        np.hypot(self.geo['U_bot'], self.geo['V_bot']),
                                        np.hypot(self.geo['U_top'], self.geo['V_top']),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        s_bot = stress_bottom(self.solution,
                              self.height_and_slopes,
                              self.geo['U_bot'],
                              self.geo['V_bot'],
                              self.geo['U_top'],
                              self.geo['V_top'],
                              shear_viscosity,
                              self.prop['bulk'],
                              0.0, self.extra  # Ls_bot=0, Ls_top=extra
                              )

        s_top = stress_top(self.solution,
                           self.height_and_slopes,
                           self.geo['U_bot'],
                           self.geo['V_bot'],
                           self.geo['U_top'],
                           self.geo['V_top'],
                           shear_viscosity,
                           self.prop['bulk'],
                           0.0, self.extra  # Ls_bot=0, Ls_top=extra
                           )

        self.__field.pg[:3] = s_bot[:3] / 2.
        self.__field.pg[6:9] = s_top[:3] / 2.

        self.__field.pg[5] = s_bot[-1] / 2.
        self.__field.pg[11] = s_top[-1] / 2.

        if self.is_gp_model:
            mean, var = self.predict(predictor=predictor,
                                     compute_var=self.use_active_learning or compute_var)

            self.__field.pg[self._out_index] = mean[0, :, :]
            self.__field.pg[self._out_index + 6] = mean[1, :, :]
            self.__field_variance.pg[:] = var[0, :, :]
        else:
            self.__field.pg[self._out_index] = s_bot[self._out_index]
            self.__field.pg[self._out_index + 6] = s_top[self._out_index]

    def build_grad(self) -> None:
        """Build JIT-compiled gradient functions for wall stress."""

        if self.is_gp_model:
            raise NotImplementedError("Gradient of GP-based wall stress not implemented.")

        else:
            dir = self.name[0]  # 'x' or 'y'
            stress_top_fn = globals()[f'stress_top_{dir}z']
            stress_bot_fn = globals()[f'stress_bottom_{dir}z']
            der_vars = ['rho', 'j' + dir, 'theta']
            der_arg_idx = [0, 1 if dir == 'x' else 2, 10]

            # central functions: only argument difference for x/y is dh; tau_bot required for energy
            def _tau(rho, jx, jy, h, dh, U_bot, V_bot, U_top, V_top, Ls, theta, dp_dx, dp_dy):
                p = eos_pressure(rho, self.prop)
                eta = get_shear_viscosity(self, p, dp_dx, dp_dy, h)
                q = jnp.array([rho, jx / (1.0 - theta), jy / (1.0 - theta)])
                h_arr = jnp.array([h, dh])
                tau_top = stress_top_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                tau_bot = stress_bot_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                return (1.0 - theta) * (tau_top - tau_bot)

            def _tau_bot(rho, jx, jy, h, dh, U_bot, V_bot, U_top, V_top, Ls, theta, dp_dx, dp_dy):
                p = eos_pressure(rho, self.prop)
                eta = get_shear_viscosity(self, p, dp_dx, dp_dy, h)
                q = jnp.array([rho, jx / (1.0 - theta), jy / (1.0 - theta)])
                h_arr = jnp.array([h, dh])
                tau_bot = stress_bot_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                return (1.0 - theta) * tau_bot

            vmap2 = lambda f: vmap(vmap(f))  # no map_axes required since we broadcast all args to quad fields

            # tau and its gradients
            setattr(self, 'tau', jit(vmap2(_tau)))
            for i, var in enumerate(der_vars):
                idx = der_arg_idx[i]
                setattr(self, f'dtau_d{var}', jit(vmap2(grad(_tau, argnums=idx))))

            # tau_bot and its gradients
            setattr(self, 'tau_bot', jit(vmap2(_tau_bot)))
            for i, var in enumerate(der_vars):
                idx = der_arg_idx[i]
                setattr(self, f'dtau_bot_d{var}', jit(vmap2(grad(_tau_bot, argnums=idx))))


class BulkStress(GaussianProcessSurrogate):
    """
    Bulk (gap-averaged) viscous stress model.

    This model currently operates deterministically (no GP surrogate).
    """

    name = "bulk"

    def __init__(self,
                 fc: Any,
                 prop: dict,
                 geo: dict,
                 data: Optional[Any] = None,
                 gp: Optional[dict] = None) -> None:
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.__pressure = Field(fc.get_real_field('pressure'))
        self.__x = Field(fc.get_real_field('x'))
        self.__y = Field(fc.get_real_field('y'))

        self.geo = geo
        self.prop = prop
        self.is_gp_model = False

        super().__init__(fc, data)

    @property
    def stress(self) -> NDArray:
        """Return the bulk viscous stress field."""
        return self.__field.pg

    @property
    def pressure(self) -> NDArray:
        """Return the pressure field."""
        return self.__pressure.pg

    @property
    def dp_dx(self) -> NDArray:
        """Return ∂p/∂x."""
        return np.gradient(self.pressure, self.__x.pg[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """Return ∂p/∂y."""
        return np.gradient(self.pressure, self.__y.pg[0, :], axis=1)

    def update(self) -> None:
        """Compute and store bulk viscous stress using viscous model."""

        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure if not self.prop['EOS'] == 'Bayada' else self.solution[0],
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        np.hypot(self.geo['U_bot'], self.geo['V_bot']),
                                        np.hypot(self.geo['U_top'], self.geo['V_top']),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        self.__field.pg[:] = stress_avg(self.solution,
                                        self.height_and_slopes,
                                        self.geo['U_bot'],
                                        self.geo['V_bot'],
                                        self.geo['U_top'],
                                        self.geo['V_top'],
                                        shear_viscosity,
                                        self.prop['bulk'],
                                        0.0, self.extra)


class Pressure(GaussianProcessSurrogate):
    """
    Pressure model.

    Supports deterministic pressure via lookup equation of state or a GP surrogate.
    """

    name = "zz"

    def __init__(self,
                 fc: Any,
                 prop: dict,
                 geo: dict,
                 data: Optional[Any] = None,
                 gp: Optional[dict] = None) -> None:
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = Field(fc.get_real_field('pressure'))
        self.geo = geo
        self.prop = prop

        if gp is not None:
            self.active_dims = gp.get('active_dims', [0, 3])
            self.__field_variance = fc.real_field('pressure_var')
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.pause_steps = gp['pause_steps']
            self.is_gp_model = True
            self.use_active_learning = gp['active_learning']
            self.build_gp = multi_in_single_out
        else:
            self.is_gp_model = False
            self.use_active_learning = False

        super().__init__(fc, data)

    @property
    def pressure(self) -> NDArray:
        """Pressure field."""
        return self.__field.pg

    @property
    def variance(self) -> NDArray:
        """Variance of the pressure field."""
        return self.__field_variance.pg

    @property
    def v_sound(self) -> NDArray | JAXArray:
        """
        Effective sound speed computed from the GP-based eos (if available)
        or from the analytic eos_sound_velocity.

        Returns
        -------
        jax.Array or scalar-like
            Sound speed (may be a JAX array/scalar).
        """
        if self.is_gp_model:
            eos_grad = vmap(grad(self.eos))
            vsound_squared = eos_grad(self.Xtest)[:, 0].max() * self.Yscale / self.database.X_scale[0]
            vsound = jnp.sqrt(vsound_squared)
            return vsound
        else:
            return eos_sound_velocity(self.solution[0], self.prop).max()

    @property
    def Xtest(self) -> JAXArray:
        """Test inputs for pressure GP (not normalized)."""
        return (self._Xtest / self.database.X_scale)[:, self.active_dims]

    @property
    def Xtrain(self) -> JAXArray:
        """Training inputs for pressure GP (normalized)."""
        return self.database.Xtrain[:, self.active_dims]

    @property
    def _Ytrain(self) -> JAXArray:
        """Training outputs for pressure GP (normalized)."""
        return self.database._Ytrain[:self.last_fit_train_size, 0]

    @property
    def Ytrain(self) -> JAXArray:
        """Training outputs for pressure GP (normalized)."""
        return self._Ytrain / self.Yscale

    @property
    def Yscale(self) -> JAXArray:
        """Output scale for pressure (scalar-like jax.Array)."""
        return self.database.Y_scale[0]

    @property
    def Yerr(self) -> float:
        """Observation noise (normalized) for pressure."""
        return jnp.mean(self.database.Ytrain_err[:self.last_fit_train_size, 0])

    @property
    def kernel_variance(self) -> JAXArray:
        """Kernel variance for pressure GP."""
        return self.gp.kernel.kernel1.value

    @property
    def kernel_lengthscale(self) -> JAXArray:
        """Kernel lengthscale for pressure GP."""
        return self.gp.kernel.kernel2.scale

    @property
    def obs_stddev(self) -> JAXArray:
        """Observation standard deviation for pressure GP."""
        return self.Yerr

    def init(self) -> None:
        """Run first training and inference."""
        if self.is_gp_model:
            # for sound speed
            self.eos = lambda x: self.gp.predict(self.Ytrain, x[None, :]).squeeze()

            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.std(self.Xtrain, axis=0))
            }

            self._train()
            self._infer()

    def update(self,
               predictor: bool = False,
               compute_var: bool = False) -> None:
        """
        Update pressure: compute deterministic stresses and, if enabled,
        perform GP prediction and place predicted mean and variance into the
        appropriate field entries.

        Parameters
        ----------
        predictor : bool, optional
            Whether this update is part of the predictor stage.
        compute_var : bool, optional
            Flag for re-computing the variance (the default is False which uses
            the stored variance from previous steps).
        """
        if self.is_gp_model:
            mean, var = self.predict(predictor=predictor,
                                     compute_var=self.use_active_learning or compute_var)
            self.__field.pg[:] = mean
            self.__field_variance.pg[:] = var
        else:
            self.__field.pg[:] = eos_pressure(self.solution[0], self.prop)

    def build_grad(self) -> None:
        if self.is_gp_model:
            raise NotImplementedError("Gradient of GP-based EOS not implemented.")
        else:
            vmap2 = lambda f: jit(vmap(vmap(f, in_axes=0), in_axes=0))

            f0 = lambda rho: eos_pressure(rho, self.prop)
            f1 = grad(f0)
            f2 = grad(f1)

            self.p_from_rho = vmap2(f0)
            self.dp_drho    = vmap2(f1)
            self.d2p_drho2  = vmap2(f2)
            self.drho_dp    = vmap2(lambda rho: 1.0 / f1(rho))


class Viscosity():

    def __init__(self,
                 fc: Any,
                 prop: dict) -> None:
        self.__field = fc.real_field('shear_viscosity')
        self.prop = prop

    @property
    def shear_viscosity(self) -> NDArray:
        """Shear viscosity field."""
        return self.__field.pg

    def update(self,
               pressure: NDArray,
               dp_dx: NDArray,
               dp_dy: NDArray,
               height: NDArray,
               U_bot: float,
               V_bot: float,
               U_top: float = 0.0,
               V_top: float = 0.0) -> None:
        """Update shear viscosity field using piezoviscosity and shear-thinning."""
        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(pressure,
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(dp_dx,
                                        dp_dy,
                                        height,
                                        np.hypot(U_bot, V_bot),
                                        np.hypot(U_top, V_top),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        # Handle scalar case (constant viscosity)
        if np.isscalar(shear_viscosity):
            self.__field.pg[:] = shear_viscosity
        else:
            self.__field.pg[:] = shear_viscosity

    @property
    def eta(self) -> NDArray:
        """Alias for shear_viscosity for cleaner access."""
        return self.__field.pg
