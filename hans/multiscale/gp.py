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

    def __init__(self, active_dim, db, Xmask, Ymask,
                 active_learning={'max_iter': 10, 'threshold': .1, 'alpha': 0.05, 'start': 20, 'Ninit': 5, 'sampling': 'lhc'},
                 kernel_dict={'type': 'Mat32', 'init_params': None, 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        self.Xmask = Xmask
        self.Ymask = Ymask

        self.active_dim = active_dim
        self.db = db

        self.kernel_dict = kernel_dict
        self.optimizer = optimizer
        self.active_learning = active_learning
        self.noise = noise

        self.maxvar = np.inf
        self.tol = active_learning['threshold']

        self.kern = GPy.kern.Matern32(active_dim, ARD=kernel_dict['ARD'])

        if kernel_dict['init_params'] is not None:
            self.kern.variance = kernel_dict['init_params'][0]
            self.kern.lengthscale = kernel_dict['init_params'][1:active_dim+1]

        self.step = 0
        self.wait = -1

        self._init_outfile()

    def __del__(self):
        self.model.save_model(f'gp_{self.name}.json', compress=True)

        self._write_history()
        self.file.close()

    @property
    def dbsize(self):
        return self.db.size

    def setup(self, q):
        nx, ny = q.shape[1:]
        self.rshape = (nx, ny if ny > 3 else 1)

        self._set_solution(q)

        if self.dbsize < self.active_learning['Ninit']:
            self._initialize_database()

        self._fit()
        self._write_history()

    def predict(self, skips=0):

        refit = False
        # Check if dbsize changed and refit
        if self.last_fit_dbsize != self.dbsize:
            print('')
            print(f'Step {self.step} - GP_{self.name}' + (f': (skipped {skips+1} inputs)' if skips > 0 else ':'))
            print(f'Max. variance before fit: {self.maxvar:.3e}')
            self._fit()
            refit = True

        Xtest = self._get_test_input()
        mean, cov = self.model.predict_noiseless(Xtest, full_cov=False)

        self.maxvar = np.amax(cov)
        meandiff = np.amax(mean) - np.amin(mean)
        self.tol = max(self.active_learning['threshold'], (self.active_learning['alpha'] * meandiff)**2)

        if refit:
            self._write_history()
            print(f'Max. variance after fit: {self.maxvar:.3e}')
            print('')

        if self.ndim == 1:
            return mean.T[:, :, np.newaxis], cov
        else:
            return np.transpose(mean.reshape(*self.rshape, -1), (2, 0, 1)), cov

    def active_learning_step(self, q):

        self._set_solution(q)
        mean, cov = self.predict()

        # TODO: in input
        counter = 0
        maxcount = 5
        self.wait -= 1

        while self.maxvar > self.tol and self.wait <= 0:
            success = False

            # sort indices with increasing variance
            xnext = np.argsort(cov[:, 0])
            _, nx, ny = mean.shape

            # try to find a point not too close to previous ones
            for i, x in enumerate(xnext[::-1]):

                ix, iy = np.unravel_index(x, (nx, ny))

                if not self._similarity_check(ix, iy):
                    # Add to training data
                    print(f'Active learning: GP_{self.name} in step {self.step + 1} ({(counter + 1)}/{maxcount})...')
                    self._update_database(ix, iy)
                    mean, cov = self.predict(skips=i)
                    success = True
                    break

            if not success:
                # If not possible, use the max variance point anyways
                ix, iy = np.unravel_index(xnext[-1], (nx, ny))
                self._update_database(ix, iy)
                mean, cov = self.predict(skips=i)

            counter += 1
            if counter == maxcount:
                self.wait = 10
                print(f'Active learning seems to stall. Wait for {self.wait} steps...')

        self.step += 1

        return mean, cov

    def _init_outfile(self):

        fname = f'gp_{self.name}.out'
        if os.path.exists(fname):
            os.remove(fname)
        self.file = open(fname, 'w', buffering=1)
        self.file.write(f"# Gaussian process: {self.name}\n# Step DB_size Kernel_params[*] maxvar tol nmll\n")

    def _write_history(self):

        per_step = [self.step, self.dbsize]
        [per_step.append(param) for param in self.kern.param_array]
        per_step.append(self.model.Gaussian_noise.variance[0])
        per_step.append(self.maxvar)
        per_step.append(self.tol)
        per_step.append(-self.model.log_likelihood())

        fmt = ["{:8d}", "{:8d}"] + (self.active_dim + 5) * ["{:8e}"]
        per_step = [f.format(item) for f, item in zip(fmt, per_step)]
        out_str = " ".join(per_step) + '\n'

        self.file.write(out_str)

    def _build_model(self):

        self.model = GPy.models.GPRegression(self.db.Xtrain[self.Xmask, :].T,
                                             self.db.Ytrain[self.Ymask, :].T,
                                             self.kern)
        self._fix_noise()

        self.last_fit_dbsize = self.dbsize

    def _fit(self):

        # Initial guess hyperparameters
        l0 = np.ones(self.active_dim + 1)
        l0[0] = self.tol

        if self.active_dim == 3:
            Xrange = (np.amax(self.db.Xtrain, axis=1) - np.amin(self.db.Xtrain, axis=1))[[0, 3, 4]]
            l0[1:] = np.maximum(Xrange, [1., 1e-3, 1.])
        elif self.active_dim == 4:
            Xrange = (np.amax(self.db.Xtrain, axis=1) - np.amin(self.db.Xtrain, axis=1))[[0, 3, 4, 5]]
            l0[1:] = np.maximum(Xrange, [1., 1e-3, 1., 1.])

        self._build_model()

        best_mll = np.inf
        best_model = self.model

        num_restarts = self.optimizer['num_restarts']

        for i in range(num_restarts):

            try:
                # Randomize hyperparameters
                self.kern.lengthscale = l0[1:] * np.exp(np.random.normal(size=self.active_dim))
                self.kern.variance = l0[0] * np.exp(np.random.normal())
                self.model.optimize('bfgs', max_iters=100)
            except np.linalg.LinAlgError:
                print('LinAlgError: Skip optimization step')

            current_mll = -self.model.log_likelihood()

            if current_mll < best_mll:
                best_model = self.model.copy()
                best_mll = np.copy(current_mll)

            if self.optimizer['verbose']:
                to_stdout = f'DB size {self.dbsize:3d} | '
                to_stdout += f'Fit {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}'
                print(to_stdout, flush=True, end='\r' if i < num_restarts - 1 else '\n')

        self.model = best_model
        self.model.save_model(os.path.join(self.db.gp['local'],
                                           f'gp_{self.name}-{self.dbsize}.json'), compress=True)

        # Also save parameters as numpy array.
        # This is more consistent across python versions (avoids pickle)
        np.save(os.path.join(self.db.gp['local'], f'gp_{self.name}-{self.dbsize}.npy'), self.model.param_array)

        # To load it later:
        # m_load = GPRegression(np.array([[], [], []]).T, np.array([[], []]).T, initialize=False)
        # m_load.update_model(False)
        # m_load.initialize_parameter()
        # m_load[:] = np.load('model_save.npy')
        # m_load.update_model(True)

    def _fix_noise(self):

        if self.noise['fixed']:
            self.model.Gaussian_noise.variance = self.noise['variance']
            self.model.Gaussian_noise.variance.fix()

    # def _get_solution(self):
    #     return self.sol

    def _similarity_check(self, ix, iy=1):

        X_new = self.sol[:, ix, iy, None]
        Hnew = self.db.gap_height(ix, iy)

        Xnew = np.vstack([Hnew, X_new])

        similar_point_exists = False

        for j in range(self.db.Xtrain.shape[1]):

            var = self.kern.variance
            similarity = (self.kern.K(Xnew[self.Xmask, 0, None].T, self.db.Xtrain[self.Xmask, j, None].T) / var)[0][0]

            if np.isclose(similarity, 1.):
                similar_point_exists = True
                break

        return similar_point_exists

    def _initialize_database(self):

        # Xsol = self._get_solution()
        self.db.sampling(self.sol,
                         self.active_learning['Ninit'],
                         self.active_learning['sampling'])

    def _update_database(self, ix, iy):

        X_new = self.sol[:, ix, iy]

        if X_new.ndim == 1:
            X_new = X_new[:, None]

        self.db.update(X_new, ix, iy)

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

    def __init__(self, db,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.], 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True

        Ymask = 13 * [False]
        Ymask[5] = True
        Ymask[11] = True

        super().__init__(3, db, Xmask, Ymask, active_learning, kernel_dict, optimizer, noise)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        return np.vstack([self.db.h[0, :, 1], self.sol[0], self.sol[1]]).T

    def _set_solution(self, q):
        self.sol = q[:, :, 1]


class GP_stress2D(GaussianProcess):

    ndim = 2
    name = 'shear'

    def __init__(self, db,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.], 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

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

        super().__init__(4, db, Xmask, Ymask, active_learning, kernel_dict, optimizer, noise)

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

    def __init__(self, db,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.], 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        # self.Ytrain = np.empty((1, 0))

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True

        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(3, db, Xmask, Ymask, active_learning, kernel_dict, optimizer, noise)

    def _get_test_input(self):
        """
        Test data as input for GP prediction:
        For pressure, only density is required (not with MD!)
        """
        return np.vstack([self.db.h[0, :, 1], self.sol[0], self.sol[1]]).T

    def _set_solution(self, q):
        self.sol = q[:, :, 1]


class GP_pressure2D(GaussianProcess):

    ndim = 2
    name = 'press'

    def __init__(self, db,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.], 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        # self.Ytrain = np.empty((1, 0))

        Xmask = 6 * [False]
        Xmask[0] = True
        Xmask[3] = True
        Xmask[4] = True
        Xmask[5] = True

        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(4, db, Xmask, Ymask, active_learning, kernel_dict, optimizer, noise)

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
