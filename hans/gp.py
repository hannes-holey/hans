#
# Copyright 2023 Hannes Holey
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


class GaussianProcess:

    __metaclass__ = abc.ABCMeta

    name: str

    def __init__(self, input_dim, active_dim, func, func_args,
                 active_learning={'max_iter': 10, 'threshold': .1, 'start': 20},
                 kernel_dict={'type': 'Mat32', 'init_params': None, 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        # self.threshold = active_learning['threshold']
        self.func = func
        self.func_args = func_args
        self.input_dim = input_dim
        self.active_dim = active_dim

        self.kernel_dict = kernel_dict
        self.optimizer = optimizer
        self.active_learning = active_learning
        self.maxvar = np.inf

        self.noise = noise

        # Initial training data
        self.Xtrain = np.empty((active_dim, 0))

        self.kern = GPy.kern.Matern32(active_dim, ARD=kernel_dict['ARD'])

        if kernel_dict['init_params'] is not None:
            self.kern.variance = kernel_dict['init_params'][0]
            self.kern.lengthscale = kernel_dict['init_params'][1:active_dim+1]

        self.step = 0
        self.dbsize = 0
        self.history = []

        self.file = open(f'gp_{self.name}.out', 'a+')

    def __del__(self):
        self.model.save_model(f'gp_{self.name}.json', compress=True)
        self.file.close()

    def setup(self, q, init_ids):
        self._set_solution(q)
        self._update_database(init_ids)
        self._build_model()

    def predict(self):
        Xtest = self._get_solution()

        mean, cov = self.model.predict_noiseless(Xtest, full_cov=True)

        self.maxvar = np.amax(np.diag(cov))

        return mean, cov

    def active_learning_step(self, q):

        if self.step > self.active_learning['start']:

            self._set_solution(q)
            new_ids = []
            for i in range(self.active_learning['max_iter']):

                mean, cov = self.predict()

                if self.maxvar > self.active_learning['threshold']:
                    # AL
                    xnext = np.argsort(np.diag(cov))
                    aid = -1
                    next_id = xnext[aid]

                    while next_id in new_ids:
                        aid -= 1
                        next_id = xnext[aid]

                    self._update_database(next_id)
                    new_ids.append(next_id)

                    self._fit()

                else:
                    break

        self._write_history()
        self.step += 1

    def _write_history(self):

        per_step = [self.step, self.dbsize]
        for param in self.kern.param_array:
            per_step.append(param)

        per_step = [str(item) for item in per_step]

        out_str = " ".join(per_step) + '\n'

        self.file.write(out_str)

    def _build_model(self):

        self.model = GPy.models.GPRegression(self.Xtrain.T, self.Ytrain.T, self.kern)
        self._fix_noise()

    def _fit(self):

        l0 = self.kernel_dict['init_params'][1:self.active_dim + 1]

        self._build_model()

        best_mll = np.inf
        best_model = self.model

        end = {True: '\n', False: '\r'}
        num_restarts = self.optimizer['num_restarts']

        for i in range(num_restarts):

            try:
                # Randomize hyperparameters parameters (lengthscale only)
                self.kern.lengthscale = l0 * np.exp(np.random.normal(size=self.active_dim))
                self.model.optimize('bfgs', max_iters=100)
            except np.linalg.LinAlgError:
                print('LinAlgError: Skip optimization step')

            current_mll = -self.model.log_likelihood()

            if current_mll < best_mll:
                best_model = self.model.copy()
                best_mll = np.copy(current_mll)

            if self.optimizer['verbose']:
                to_stdout = f'DB size {self.dbsize:3d} | '
                to_stdout += f'Fit ({self.name}) {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}'
                print(to_stdout, flush=True, end=end[i == num_restarts-1])

        self.model = best_model

    def _fix_noise(self):

        if self.noise['fixed']:
            self.model.Gaussian_noise.variance = self.noise['variance']
            self.model.Gaussian_noise.variance.fix()

    def _set_solution(self, q):
        self.sol = q

    @abc.abstractmethod
    def _get_solution(self):
        """
        Assemble test data
        """

    @abc.abstractmethod
    def _update_database(self, next_id):
        """
        Update training data
        """


class GP_stress(GaussianProcess):

    name = 'shear'

    def __init__(self, h, dh, func, func_args, out_ids,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.], 'ARD': True},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        self.h = h
        self.dh = dh
        self.out_ids = out_ids

        self.Ytrain = np.empty((len(out_ids), 0))

        super().__init__(4, 3, func, func_args, active_learning, kernel_dict, optimizer, noise)

    def _get_solution(self):
        """
        Format raw solution for function evaluation.
        """

        return np.vstack([self.h, self.sol[0], self.sol[1]]).T

    def _update_database(self, next_id):

        try:
            size = len(next_id)
        except TypeError:
            size = 1

        Xsol = self._get_solution()

        H = np.vstack([np.array(Xsol[next_id, 0]), np.ones(size) * self.dh[0], np.zeros(size)])
        Q = np.vstack([np.array(Xsol[next_id, 1]), np.array(Xsol[next_id, 2]), np.zeros(size)])

        Y_new = self.func(Q, H, 0., **self.func_args)
        Y_new += np.random.normal(0, np.sqrt(self.noise['variance']), size=(2, size))

        X_new = Xsol[next_id].T
        if X_new.ndim == 1:
            X_new = X_new[:, None]

        self.Xtrain = np.hstack([self.Xtrain, X_new])
        self.Ytrain = np.hstack([self.Ytrain, Y_new])

        self.dbsize += size


class GP_pressure(GaussianProcess):

    name = 'press'

    def __init__(self, func, func_args,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-2, ], 'ARD': False},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True},
                 noise={'type': 'Gaussian',  'fixed': True, 'variance': 0.}):

        self.Ytrain = np.empty((1, 0))

        super().__init__(1, 1, func, func_args, active_learning, kernel_dict, optimizer, noise)

    def _get_solution(self):

        return self.sol[:, None]

    def _update_database(self, next_id):

        try:
            size = len(next_id)
        except TypeError:
            size = 1

        Xsol = self._get_solution()

        X_new = Xsol[next_id].T

        if X_new.ndim == 1:
            X_new = X_new[:, None]

        Y_new = self.func(X_new, **self.func_args)
        Y_new += np.random.normal(0, np.sqrt(self.noise['variance']), size=(1, size))

        self.Xtrain = np.hstack([self.Xtrain, X_new])
        self.Ytrain = np.hstack([self.Ytrain, Y_new])

        self.dbsize += size
