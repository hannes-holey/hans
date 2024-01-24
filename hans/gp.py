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
import os
import dtoolcore
import lammps
from ruamel.yaml import YAML
from dateutil.relativedelta import relativedelta
from getpass import getuser
from socket import gethostname
from datetime import datetime, date
from dtool_lookup_api import query
from scipy.stats import qmc


from hans.tools import progressbar, bordered_text, abort
from hans.md.mpi import mpi_run


class GaussianProcess:

    __metaclass__ = abc.ABCMeta

    name: str

    def __init__(self, active_dim, db, Xmask, Ymask,  # func, func_args,
                 active_learning={'max_iter': 10, 'threshold': .1, 'start': 20, 'Ninit': 5, 'sampling': 'lhc'},
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

        if refit:
            self._write_history()
            print(f'Max. variance after fit: {self.maxvar:.3e}')
            print('')

        return mean, cov

    def active_learning_step(self, q):

        threshold = self.active_learning['threshold']

        self._set_solution(q)
        mean, cov = self.predict()

        while self.maxvar > threshold:
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
        self.file.write(f"# Gaussian process: {self.name}\n# Step DB_size Kernel_params[*] maxvar nmll\n")

    def _write_history(self):

        per_step = [self.step, self.dbsize]
        [per_step.append(param) for param in self.kern.param_array]
        per_step.append(self.maxvar)
        per_step.append(-self.model.log_likelihood())

        fmt = ["{:8d}", "{:8d}"] + (self.active_dim + 2) * ["{:8e}"] + ["{:+8e}"]
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
        np.save(f'gp_{self.name}-{self.dbsize}.npy', self.model.param_array)

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

        # Xtest = self.sol[0]
        # if Xtest.ndim == 1:
        #     Xtest = Xtest[:, None]

        # return Xtest


class Database:

    def __init__(self, height, eos_func, tau_func, gp):

        self.eos_func = eos_func
        self.tau_func = tau_func
        self.h = height
        self.gp = gp

        input_dim = 6
        # h, dh_dx, dh_dy, rho, jx, jy

        output_dim = 13
        # p, tau_lower (6), tau_upper (6)

        if bool(self.gp['remote']):
            self._init_remote(input_dim, output_dim)
        else:
            self._init_local(input_dim, output_dim)

    def _init_remote(self, input_dim, output_dim):
        # uses dtool_lookup_server configuration from your dtool config
        # TODO: Possibility to pass a txtfile w/ uuids or query string
        query_dict = {"readme.description": {"$regex": "Dummy"}}
        remote_ds_list = query(query_dict)

        remote_ds = [dtoolcore.DataSet.from_uri(ds['uri'])
                     for ds in progressbar(remote_ds_list,
                                           prefix="Loading remote datasets based on dtool query: ")]

        if len(remote_ds) > 0:

            yaml = YAML()
            Xtrain = []
            Ytrain = []

            for ds in remote_ds:
                readme_string = ds.get_readme_content()
                rm = yaml.load(readme_string)

                Xtrain.append(rm['X'])
                Ytrain.append(rm['Y'])

            self.Xtrain = np.array(Xtrain).T
            self.Ytrain = np.array(Ytrain).T
        else:
            print("No matching dtool datasets found. Start with empty database.")
            self.Xtrain = np.empty((input_dim, 0))
            self.Ytrain = np.empty((output_dim, 0))

    def _init_local(self, input_dim, output_dim):
        # All datsets in local directory

        if not os.path.exists(self.gp['local']):
            os.makedirs(self.gp['local'])

        # Python 3.9+
        # str.removeprefix (prefix = f'file://{gethostname()}')
        readme_list = [os.path.join(os.sep, *ds.uri.split(os.sep)[3:], 'README.yml')
                       for ds in dtoolcore.iter_datasets_in_base_uri(self.gp['local'])]

        if len(readme_list) > 0:
            print(f"Loading {len(readme_list)} local datasets in '{self.gp['local']}'.")

            yaml = YAML()
            Xtrain = []
            Ytrain = []

            for readme in readme_list:
                with open(readme, 'r') as instream:
                    rm = yaml.load(instream)

                Xtrain.append(rm['X'])
                Ytrain.append(rm['Y'])

            self.Xtrain = np.array(Xtrain).T
            self.Ytrain = np.array(Ytrain).T
        else:
            print("No matching dtool datasets found. Start with empty database.")
            self.Xtrain = np.empty((input_dim, 0))
            self.Ytrain = np.empty((output_dim, 0))

    def update(self, Qnew, next_id):
        self._update_inputs(Qnew, next_id)

    def sampling(self, Q, Ninit, sampling='lhc'):
        """Build initial database with different sampling strategies.

        Select points in two-dimensional space (gap height, flux) to
        initialize training database. Choose between random, latin hypercube, 
        and Sobol sampling.


        Parameters
        ----------
        Q : [type]
            [description]
        Ninit : [type]
            [description]
        """

        # Bounds for quasi random sampling of initial database
        l_bounds = [np.amin(self.h[0]), 0.0]
        u_bounds = [np.amax(self.h[0]), 2. * np.mean(Q[1, :])]

        # Sampling
        if sampling == 'random':
            h_init = l_bounds[0] + np.random.random_sample([Ninit, ]) * (u_bounds[1] - l_bounds[0])
            jx_init = l_bounds[1] + np.random.random_sample([Ninit, ]) * (u_bounds[1] - l_bounds[0])
        elif sampling == 'lhc':

            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=Ninit)

            scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
            h_init = scaled_samples[:, 0]
            jx_init = scaled_samples[:, 1]

        elif sampling == 'sobol':
            sampler = qmc.Sobol(d=2)

            m = int(np.log2(Ninit))
            if int(2**m) != Ninit:
                m = int(np.ceil(np.log2(Ninit)))
                print(f'Sample size should be a power of 2 for Sobol sampling. Use Ninit={2**m}.')
            sample = sampler.random_base2(m=m)

            scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
            h_init = scaled_samples[:, 0]
            jx_init = scaled_samples[:, 1]

        # Remaining inputs (constant)
        rho_init = np.ones_like(jx_init) * np.mean(Q[0, :])
        jy_init = np.zeros_like(jx_init)
        h_gradx_init = np.ones_like(h_init) * np.mean(self.h[1, :, 1])
        h_grady_init = np.ones_like(h_init) * np.mean(self.h[2, :, 1])

        # Assemble
        Hnew = np.vstack([h_init, h_gradx_init, h_grady_init])
        Qnew = np.vstack([rho_init, jx_init, jy_init])

        Xnew = np.vstack([Hnew, Qnew])
        self.Xtrain = np.hstack([self.Xtrain, Xnew])

        self._update_outputs(Xnew)

    def _update_inputs(self, Qnew, next_id):

        Hnew = self.gap_height(next_id)
        Xnew = np.vstack([Hnew, Qnew])

        self.Xtrain = np.hstack([self.Xtrain, Xnew])

        self._update_outputs(Xnew)

    def _update_outputs(self, Xnew):

        Ynew = np.zeros((13, Xnew.shape[1]))

        for i in range(Xnew.shape[1]):

            num = self.Ytrain.shape[1] + i
            base_uri = self.gp['local']
            ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_dataset-{num:03}'
            proto_ds = dtoolcore.create_proto_dataset(name=ds_name, base_uri=base_uri)
            proto_datapath = os.path.join(base_uri, ds_name, 'data')

            # Run LAMMPS...
            if bool(self.gp['lmp']):
                kw_args = dict(gap_height=Xnew[0, i],
                               vWall=0.12,  # TODO: not hard-coded
                               density=Xnew[3, i],
                               mass_flux=Xnew[4, i],
                               slabfile="slab111-S.lammps")

                # Run MD with fixed number of cores in proto dataset
                nworker = self.gp['ncpu']
                wdir = os.getcwd()
                os.chdir(proto_datapath)

                text = f"""Run next MD simulation in: {proto_datapath}
---
Gap height: {Xnew[0, i]}
Mass density: {Xnew[3, i]}
Mass flux: {Xnew[4, i]}
"""

                print(bordered_text(text))

                # Run
                mpi_run(nworker, kw_args)

                # Get stress
                step, pL_t, tauL_t, pU_t, tauU_t = np.loadtxt('stress_wall.dat', unpack=True)
                press = np.mean(pL_t + pU_t) / 2.
                tauL = np.mean(tauL_t)
                tauU = np.mean(tauU_t)

                os.chdir(wdir)

                Ynew[0, i] = press
                Ynew[5, i] = tauL
                Ynew[11, i] = tauU

            # ... or use a (possibly noisy) constitutive law
            else:
                Ynew[0, i] = self.eos_func(Xnew[3:, i]) + pnoise[:, i]
                Ynew[1:, i, None] = self.tau_func(Xnew[3:, i, None], Xnew[:3, i, None], 0.) + snoise[:, i, None]

            self.write_readme(Xnew[:, i], Ynew[:, i], os.path.join(base_uri, ds_name))
            proto_ds.freeze()

            if self.gp['remote']:
                dtoolcore.copy(proto_ds.uri, self.gp['storage'])

        self.Ytrain = np.hstack([self.Ytrain, Ynew])

    def write_readme(self, Xnew, Ynew, path):

        readme_template = """
        project: Multiscale Simulation of Lubrication
        description: Automatically generated MD run of confined fluid for multiscale simulations
        owners:
          - name: Hannes Holey
            email: hannes.holey@kit.edu
            username: hannes
            orcid: 0000-0002-4547-8791
        funders:
          - organization: Deutsche Forschungsgemeinschaft (DFG)
            program: Graduiertenkolleg
            code: GRK 2450
        creation_date: DATE
        expiration_date: EXPIRATION_DATE
        software_packages:
          - name: LAMMPS
            version: version
            website: https://lammps.sandia.gov/
            repository: https://github.com/lammps/lammps
        MD:
          - cutoff: 2.5
          - temp: 1.0
          - vel_wall: 0.12
        GP:
          - active_dims: [0, 3, 4]
          - out_dims_1: [0]
          - out_dims_2: [5, 11]

        """

        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent(mapping=4, sequence=4, offset=2)
        metadata = yaml.load(readme_template)

        # Update metadata
        metadata["owners"][0].update(dict(username=getuser()))
        metadata["creation_date"] = date.today()
        metadata["expiration_date"] = metadata["creation_date"] + relativedelta(years=10)
        metadata["software_packages"][0]["version"] = str(lammps.__version__)

        out_fname = os.path.join(path, 'README.yml')

        X = [float(item) for item in Xnew]
        Y = [float(item) for item in Ynew]

        metadata['X'] = X
        metadata['Y'] = Y

        with open(out_fname, 'w') as outfile:
            yaml.dump(metadata, outfile)

    def gap_height(self, next_id):

        Hnew = self.h[:, next_id, 1]
        if Hnew.ndim == 1:
            Hnew = Hnew[:, None]

        return Hnew
