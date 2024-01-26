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


class GaussianProcess:

    __metaclass__ = abc.ABCMeta

    name: str

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
        self.counter = -1

        self._init_outfile()

    def __del__(self):
        self.model.save_model(f'gp_{self.name}.json', compress=True)

        self._write_history()
        self.file.close()

    @property
    def dbsize(self):
        return self.db.Xtrain.shape[1]

    def setup(self, q):
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
            print(f'GP_{self.name}' + (f': (skipped {skips+1} inputs)' if skips > 0 else ':'))
            print(f'Max. variance before fit: {self.maxvar:.3e}')
            self._fit()
            refit = True

        Xtest = self._get_test_input()
        mean, cov = self.model.predict_noiseless(Xtest, full_cov=True)
        self.maxvar = np.amax(np.diag(cov))
        meandiff = np.amax(mean) - np.amin(mean)
        self.tol = max(self.active_learning['threshold'], (self.active_learning['alpha'] * meandiff)**2)

        if refit:
            self._write_history()
            print(f'Max. variance after fit: {self.maxvar:.3e}')
            print('')

        return mean, cov

    def active_learning_step(self, q):

        self._set_solution(q)
        mean, cov = self.predict()

        while self.maxvar > self.tol:
            success = False

            # sort indices with increasing variance
            xnext = np.argsort(np.diag(cov))[1:-1]

            # try to find a point not too close to previous ones
            for i, x in enumerate(xnext[::-1]):
                if not(self._similarity_check(x)):
                    # Add to training data
                    self._update_database(x)
                    mean, cov = self.predict(skips=i)
                    success = True
                    break

            if not(success):
                # If not possible, use the max variance point anyways
                self._update_database(xnext[-1])
                mean, cov = self.predict(skips=i)

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
        per_step.append(self.maxvar)
        per_step.append(self.tol)
        per_step.append(-self.model.log_likelihood())

        fmt = ["{:8d}", "{:8d}"] + (self.active_dim + 4) * ["{:8e}"]
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

        # Use initial kernel parameters if:
        # [database is small, likelihood is low, time step is low, ...]
        if self.dbsize < self.active_learning['Ninit'] + 5:
            l0 = self.kernel_dict['init_params'][:]
        else:
            l0 = np.copy(self.kern.param_array[:])

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

    def _set_solution(self, q):
        self.sol = q

    def _get_solution(self):
        return self.sol

    def _similarity_check(self, index):

        Xsol = self._get_solution()

        X_new = Xsol[:, index, None]

        Hnew = self.db.gap_height(index)
        Xnew = np.vstack([Hnew, X_new])

        # Only compare height, density and mass flux
        input_mask = [True, False, False, True, True, False]

        similar_point_exists = False

        for j in range(self.db.Xtrain.shape[1]):

            var = self.kern.variance
            similarity = (self.kern.K(Xnew[input_mask, 0, None].T, self.db.Xtrain[input_mask, j, None].T) / var)[0][0]

            if np.isclose(similarity, 1.):
                similar_point_exists = True
                break

        return similar_point_exists

    def _initialize_database(self):

        Xsol = self._get_solution()
        self.db.sampling(Xsol, self.active_learning['Ninit'], self.active_learning['sampling'])

    def _update_database(self, next_id):

        Xsol = self._get_solution()

        X_new = Xsol[:, next_id]

        if X_new.ndim == 1:
            X_new = X_new[:, None]

        self.db.update(X_new, next_id)

    @abc.abstractmethod
    def _get_test_input(self):
        """
        Assemble test input data
        """
        raise NotImplementedError


class GP_stress(GaussianProcess):

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


class GP_pressure(GaussianProcess):

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
