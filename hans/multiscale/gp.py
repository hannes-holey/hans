#
# Copyright 2023-2024 Hannes Holey
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

import abc
import GPy
import numpy as np
import os
from copy import deepcopy


class GaussianProcess:

    __metaclass__ = abc.ABCMeta

    name: str
    ndim: int

    def __init__(self,
                 active_dim,
                 db,
                 gp,
                 Xmask,
                 Ymask,
                 atol,
                 kernel_init_var,
                 kernel_init_scale,
                 noise_variance):

        self.tol = atol
        self.atol = atol
        self.rtol = gp['rtol']

        self.active_dim = active_dim
        self.db = db
        self.Xmask = Xmask
        self.Ymask = Ymask

        # Kernel
        self.kern = GPy.kern.Matern32(active_dim, ARD=True)
        self.kernel_init_var = kernel_init_var
        self.kernel_init_scale = kernel_init_scale

        if len(self.kernel_init_scale) == self.active_dim:
            self.kern.lengthscale = kernel_init_scale

        self.kern_last_success_params = None

        # Noise
        self.heteroscedastic_noise = gp['heteroscedastic']
        self.noise_fixed = gp['noiseFixed']
        self.noise_variance = noise_variance

        # Common parameters
        self.options = gp
        self.step = 0
        self.maxvar = np.inf
        self._reset = False

        self._init_outfile()

    def __del__(self):

        self._fit()
        self.model.save_model(f'gp_{self.name}.json', compress=True)
        self._write_history()
        self.file.close()

    @property
    def dbsize(self):
        return self.db.size

    @property
    def trusted(self):
        return self._trusted

    @property
    def reset(self):
        return self._reset

    def reset_reset(self):
        self._reset = False

    def increment(self):
        self.step += 1

    def setup(self, q):
        nx, ny = q.shape[1:]
        self.rshape = (nx, ny if ny > 3 else 1)

        self._set_solution(q)

        if self.dbsize < self.options['Ninit']:
            self.db.initialize(self.sol,
                               self.options['Ninit'],
                               self.options['sampling'])

        self.last_fit_dbsize = self.dbsize
        self._fit()
        Xtest = self._get_test_input()
        self._raw_predict(Xtest)
        self._write_history()

    def _raw_predict(self, Xtest):

        mean, cov = self.model.predict_noiseless(Xtest, full_cov=False)

        self.maxvar = np.amax(cov)
        self.tol = max(self.atol, (self.rtol * (np.amax(mean) - np.amin(mean)))**2)

        self._trusted = self.maxvar < self.tol

        if self._trusted and not self._reset:
            self.kern_last_success_params = np.copy(self.kern.param_array)

        if self.ndim == 1:
            return mean.T[:, :, np.newaxis], cov
        else:
            return np.transpose(mean.reshape(*self.rshape, -1), (2, 0, 1)), cov

    def predict(self, q, in_predictor):

        self._set_solution(q)
        Xtest = self._get_test_input()
        old_maxvar = deepcopy(self.maxvar)

        if in_predictor and self.last_fit_dbsize != self.dbsize:
            # Predictor step: Refit after database has grown
            self.gp_print()
            self._fit()
            mean, cov = self._raw_predict(Xtest)
            self._write_history()
            self.gp_print(f'>>>> {"Reason":<18}: {"DB changed"}')
            self.gp_print(f'>>>> {"Max. var. (old)":<18}: {old_maxvar:.3e}')
            self.gp_print(f'>>>> {"Max. var. (new)":<18}: {self.maxvar:.3e}')
            self.gp_print(f'>>>> {"Tolerance":<18}: {self.tol:.3e}')
            suffix = ' (waiting)' if self._reset else ''
            self.gp_print(f'>>>> {"Accepted":<18}: {self.trusted} {suffix}')
        else:
            # Corrector step: Only use additional training point w/o hyperparameter optimization
            self._build_model()
            mean, cov = self._raw_predict(Xtest)

        if in_predictor and not self._reset:
            counter = 0
            while (not self._trusted) and (counter < self.options['maxSteps']):
                found_newX = self._active_learning_step(mean, cov)
                counter += 1
                level = 2 - np.sum(counter <= np.arange(1, 3) * self.options['maxSteps'] // 3 + self.options['maxSteps'] % 3)
                old_maxvar = deepcopy(self.maxvar)
                if found_newX:
                    self.gp_print()
                    self._fit(level)
                    mean, cov = self._raw_predict(Xtest)
                    self._write_history()
                    self.gp_print(f'>>>> {"Reason":<18}: AL {counter}/{self.options["maxSteps"]} ({level})')
                    self.gp_print(f'>>>> {"Max. var. (old)":<18}: {old_maxvar:.3e}')
                    self.gp_print(f'>>>> {"Max. var. (new)":<18}: {self.maxvar:.3e}')
                    self.gp_print(f'>>>> {"Tolerance":<18}: {self.tol:.3e}')
                    self.gp_print(f'>>>> {"Accepted":<18}: {self.trusted}')
                else:
                    self.gp_print(f'>>>> No data added ({counter}/{self.options["maxSteps"]})')

            if not self.trusted:
                self._reset = True

        return mean, cov

    def predict_gradient(self):
        Xtest = self._get_test_input()
        dq_dX, dv_dX = self.model.predictive_gradients(Xtest)

        if self.ndim == 1:
            return dq_dX.T[0], dv_dX
        else:
            return np.transpose(dq_dX.reshape(*self.rshape, -1), (2, 0, 1)), dv_dX

    def _active_learning_step(self, mean, cov):
        success = False

        xnext = np.argsort(cov[:, 0])
        _, nx, ny = mean.shape

        # try to find a point not too close to previous ones
        for i, x in enumerate(xnext[::-1]):
            ix, iy = np.unravel_index(x, (nx, ny))
            success = not self._similarity_check(ix, iy if self.ndim == 2 else 1)

            if success:
                # Add to training data
                self._update_database(ix, iy)
                break

        return success

    def _init_outfile(self):

        fname = f'gp_{self.name}.out'
        if os.path.exists(fname):
            os.remove(fname)
        self.file = open(fname, 'w', buffering=1)
        self.file.write(f"# Gaussian process: {self.name}\n# Step DB_size Kernel_params[*] maxvar tol nmll\n")

    def _write_history(self):

        per_step = [self.step, self.dbsize]
        [per_step.append(param) for param in self.kern.param_array]

        if self.heteroscedastic_noise:
            per_step.append(np.mean(self.model.het_Gauss.variance))
        else:
            per_step.append(self.model.Gaussian_noise.variance[0])

        per_step.append(self.maxvar)
        per_step.append(self.tol)
        per_step.append(-self.model.log_likelihood())

        fmt = ["{:8d}", "{:8d}"] + (len(per_step) - 2) * ["{:8e}"]
        per_step = [f.format(item) for f, item in zip(fmt, per_step)]
        out_str = " ".join(per_step) + '\n'

        self.file.write(out_str)

    def _build_model(self):

        if self.heteroscedastic_noise:
            self.model = GPy.models.GPHeteroscedasticRegression(self.db.Xtrain[self.Xmask, :].T,
                                                                self.db.Ytrain[self.Ymask, :].T,
                                                                self.kern)
        else:
            self.model = GPy.models.GPRegression(self.db.Xtrain[self.Xmask, :].T,
                                                 self.db.Ytrain[self.Ymask, :].T,
                                                 self.kern)
        self._set_noise()

    def _initial_guess_kernel_params(self, level):

        # three different levels of initial kernel lengthscale
        if level == 0 and len(self.kernel_init_scale) == self.active_dim:
            # 0) use initial scales from input (if given)
            l0 = np.array(self.kernel_init_scale)
            v0 = self.kernel_init_var if self.kernel_init_var > 0. else self.atol * 100.
        elif level == 1 and self.kern_last_success_params is not None:
            # 1) try with fitted scales from previous succesful step
            v0 = np.array(self.kern_last_success_params[0]) * 100
            l0 = np.array(self.kern_last_success_params[1:]) * 100
        else:
            # 2) use completely random lengthscales in reasonable interval
            l0 = np.random.uniform(.1, 100., size=self.active_dim)
            v0 = 1.

        return v0, l0

    def _fit(self, level=0):

        self._build_model()

        v0, l0 = self._initial_guess_kernel_params(level)

        best_mll = np.inf
        best_model = self.model
        num_restarts = self.options['optRestarts']

        # TODO: test vs. GPy.model.optimize_restarts()
        for i in range(num_restarts):
            try:
                # Randomize hyperparameters
                self.kern.lengthscale = l0 * np.exp(np.random.normal(size=self.active_dim))
                self.kern.variance = v0 * np.exp(np.random.normal())
                self.model.optimize('lbfgsb', max_iters=1000)
            except np.linalg.LinAlgError:
                print('LinAlgError: Skip optimization step')

            current_mll = -self.model.log_likelihood()

            if current_mll < best_mll:
                best_model = self.model.copy()
                best_mll = np.copy(current_mll)

            to_stdout = f'>>>> GP_{self.name} {self.step + 1:8d} | DB size {self.dbsize:3d} | '
            to_stdout += f'Fit {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}'
            self.gp_print(to_stdout, flush=True, end='\r' if i < num_restarts - 1 else '\n')

        self.last_fit_dbsize = self.dbsize

        self.model = best_model
        self.model.save_model(os.path.join(self.options['local'],
                                           f'gp_{self.name}-{self.dbsize}.json'), compress=True)

        # Also save parameters as numpy array.
        # This is more consistent across python versions (avoids pickle)
        np.save(os.path.join(self.options['local'], f'gp_{self.name}-{self.dbsize}.npy'), self.model.param_array)

        # To load it later:
        # m_load = GPRegression(np.array([[], [], []]).T, np.array([[], []]).T, initialize=False)
        # m_load.update_model(False)
        # m_load.initialize_parameter()
        # m_load[:] = np.load('model_save.npy')
        # m_load.update_model(True)

    def _set_noise(self):

        index = 0 if self.name == 'press' else 1

        if self.heteroscedastic_noise:
            self.model.het_Gauss.variance = (self.db.Yerr[index, :].T)[:, None]
            if self.noise_fixed:
                self.model.het_Gauss.variance.fix()
        else:
            self.model.Gaussian_noise.variance = np.mean(self.db.Yerr[index, :].T)
            if self.noise_fixed:
                self.model.Gaussian_noise.variance.fix()

    def _similarity_check(self, ix, iy=1):

        X_new = self.sol[:, ix, iy, None]
        Hnew = self.db.gap_height(ix, iy)

        Xnew = np.vstack([Hnew, X_new])

        similar_point_exists = False

        for j in range(self.db.Xtrain.shape[1]):

            var = self.kern.variance
            similarity = (self.kern.K(Xnew[self.Xmask, 0, None].T, self.db.Xtrain[self.Xmask, j, None].T) / var)[0][0]

            if np.isclose(similarity, 1., rtol=0., atol=1e-8):
                similar_point_exists = True
                break

        return similar_point_exists

    def _update_database(self, ix, iy):

        X_new = self.sol[:, ix, iy]

        if X_new.ndim == 1:
            X_new = X_new[:, None]

        self.gp_print(f'\n>>>> Database update in step {self.step + 1} ({self.name}):')

        self.db.update(X_new, ix, iy)

    def gp_print(self, msg='', **kwargs):
        if self.options['verbose']:
            print(msg, **kwargs)

    @ abc.abstractmethod
    def _get_test_input(self):
        """
        Assemble test input data
        """
        raise NotImplementedError

    def _set_solution(self, q):

        raise NotImplementedError


class GP_stress(GaussianProcess):

    ndim = 1
    name = 'shear'

    def __init__(self, db, gp):

        atol = gp['atolShear']
        kernel_init_var = gp['varShear']
        kernel_init_scale = gp['scaleShear']
        noise_variance = gp['noiseShear']

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True

        Ymask = 13 * [False]
        Ymask[5] = True
        Ymask[11] = True

        super().__init__(3, db, gp, Xmask, Ymask, atol, kernel_init_var, kernel_init_scale, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        return np.vstack([self.db.h[0, :, 1], self.sol[0, :, 1], self.sol[1, :, 1]]).T

    def _set_solution(self, q):
        self.sol = q  # [:, :, 1]


class GP_stress2D(GaussianProcess):

    ndim = 2
    name = 'shear'

    def __init__(self, db, gp):

        atol = gp['atolShear']
        kernel_init_var = gp['varShear']
        kernel_init_scale = gp['scaleShear']
        noise_variance = gp['noiseShear']

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True
        Xmask[5] = True

        Ymask = 13 * [False]
        Ymask[4] = True
        Ymask[5] = True
        Ymask[10] = True
        Ymask[11] = True

        super().__init__(4, db, gp, Xmask, Ymask, atol, kernel_init_var, kernel_init_scale, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        Xtest_raw = np.dstack([self.db.h[0], self.sol[0], self.sol[1], self.sol[2]])
        Xtest = np.reshape(np.transpose(Xtest_raw, (2, 0, 1)), (self.active_dim, -1)).T

        return Xtest

    def _set_solution(self, q):

        self.sol = deepcopy(q)
        self.sol[2] *= np.sign(q[2])


class GP_pressure(GaussianProcess):

    ndim = 1
    name = 'press'

    def __init__(self, db, gp):

        atol = gp['atolPress']
        kernel_init_var = gp['varPress']
        kernel_init_scale = gp['scalePress']
        noise_variance = gp['noisePress']

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True

        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(3, db, gp, Xmask, Ymask, atol, kernel_init_var, kernel_init_scale, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction:
        For pressure, only density is required (not with MD!)
        """
        return np.vstack([self.db.h[0, :, 1], self.sol[0, :, 1], self.sol[1, :, 1]]).T

    def _set_solution(self, q):
        self.sol = q  # [:, :, 1]


class GP_pressure2D(GaussianProcess):

    ndim = 2
    name = 'press'

    def __init__(self, db, gp):

        atol = gp['atolPress']
        kernel_init_var = gp['varPress']
        kernel_init_scale = gp['scalePress']
        noise_variance = gp['noisePress']

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True
        Xmask[5] = True

        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(4, db, gp, Xmask, Ymask, atol, kernel_init_var, kernel_init_scale, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction:
        For pressure, only density is required (not with MD!)
        """

        Xtest_raw = np.dstack([self.db.h[0], self.sol[0], self.sol[1], self.sol[2]])
        Xtest = np.reshape(np.transpose(Xtest_raw, (2, 0, 1)), (self.active_dim, -1)).T

        return Xtest

    def _set_solution(self, q):

        self.sol = deepcopy(q)
        self.sol[2] *= np.sign(q[2])
