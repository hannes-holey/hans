import GPy
import numpy as np


class GaussianProcess_stress:
    """
    TODO:   - think about member variables (which are really necessary, access via getter)
            - store history
            - write to file (json)
            - plot?
    """

    def __init__(self, h, dh, func, func_args, out_ids,
                 active_learning={'max_iter': 10, 'threshold': .1},
                 kernel_dict={'type': 'Mat32', 'init_params': [1e6, 1e-5, 1e-2, 100.]},
                 optimizer={'type': 'bfgs', 'restart': True, 'num_restarts': 10, 'verbose': True}):

        self.threshold = active_learning['threshold']
        self.func = func
        self.func_args = func_args
        self.out_ids = out_ids

        self.kernel_dict = kernel_dict
        self.optimizer = optimizer
        self.active_learning = active_learning
        self.maxvar = np.inf

        self.h = h
        self.dh = dh

        # Initial training data
        self.Xtrain = np.array([[], [], [], []])

        self.kern = GPy.kern.Matern32(3, variance=kernel_dict['init_params'][0],
                                      lengthscale=kernel_dict['init_params'][1:], ARD=True)

        self.dbsize = 0
        self.alstep = 0

    def setup(self, q, init_ids):
        self._set_solution(q)
        self._update_database(init_ids)
        self._build_model()

    def predict(self):
        Xtest = self._get_solution()
        mean, cov = self.model.predict(Xtest, full_cov=True)

        self.maxvar = np.amax(np.diag(cov))

        return mean, cov

    def active_learning_step(self, q):

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

        self.alstep += 1

    def _build_model(self):

        X, Y = self._assemble_data()
        self.model = GPy.models.GPRegression(X, Y.T, self.kern)
        self._fix_noise(0.)

    def _fit(self):

        l0 = self.kernel_dict['init_params'][1:]  # [1e-5, 1e-2, 100.]

        self._build_model()

        best_mll = np.inf
        best_model = self.model

        end = {True: '\n', False: '\r'}
        num_restarts = self.optimizer['num_restarts']

        for i in range(num_restarts):

            try:
                # Randomize hyperparameters parameters (lengthscale only)
                self.kern.lengthscale = l0 * np.exp(np.random.normal(size=3))
                # self.kern.variance *= np.exp(np.random.normal(size=1))
                # self.model.likelihood.variance *= np.exp(np.random.normal(size=1))

                self.model.optimize('bfgs', max_iters=100)
            except np.linalg.LinAlgError:
                print('LinAlgError: Skip optimization step')

            current_mll = -self.model.log_likelihood()

            if current_mll < best_mll:
                best_model = self.model.copy()
                best_mll = np.copy(current_mll)

            if self.optimizer['verbose']:
                print(f'Wall {self.out_ids[0]:1d} | DB size {self.dbsize:3d} | Optimize {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}',
                      flush=True, end=end[i == num_restarts-1])

        self.model = best_model

    def _fix_noise(self, noise_variance):

        self.model.Gaussian_noise.variance = noise_variance
        self.model.Gaussian_noise.variance.fix()

    def _assemble_data(self):
        H0_train, H1_train, Q0_train, Q1_train = self.Xtrain
        H = np.vstack([H0_train, H1_train, np.zeros_like(H1_train)])
        Q = np.vstack([Q0_train, Q1_train, np.zeros_like(Q1_train)])

        X = np.vstack([H0_train, Q0_train, Q1_train]).T
        Y = self.func(Q, H, 0., **self.func_args)
        indices = [index in self.out_ids for index in range(Y.shape[0])]
        Y = Y[indices]

        return X, Y

    def _set_solution(self, q):
        self.sol = q

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

        H0_train, H1_train, Q0_train, Q1_train = self.Xtrain

        Xsol = self._get_solution()

        H0_train = np.hstack([H0_train, np.array(Xsol[next_id, 0])])
        H1_train = np.hstack([H1_train, np.ones(size) * self.dh[0]])
        Q0_train = np.hstack([Q0_train, np.array(Xsol[next_id, 1])])
        Q1_train = np.hstack([Q1_train, np.array(Xsol[next_id, 2])])

        self.Xtrain = np.vstack([H0_train, H1_train, Q0_train, Q1_train])

        self.dbsize += size
