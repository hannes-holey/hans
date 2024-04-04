#
# Copyright 2024 Hannes Holey
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
import os
import numpy as np
import dtoolcore
try:
    import lammps
except ImportError:
    pass
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from getpass import getuser
from ruamel.yaml import YAML
from dtool_lookup_api import query
from scipy.stats import qmc


from hans.tools import progressbar, bordered_text
from hans.multiscale.md import run, mpirun


class Database:

    def __init__(self, gp, md, height, eos_func, tau_func):

        self.eos_func = eos_func
        self.tau_func = tau_func
        self.h = height
        self.gp = gp
        self.md = md

        input_dim = 6
        # h, dh_dx, dh_dy, rho, jx, jy

        output_dim = 13
        # p, tau_lower (6), tau_upper (6)

        if bool(self.gp['remote']):
            self._init_remote(input_dim, output_dim)
        else:
            self._init_local(input_dim, output_dim)

    def __del__(self):
        np.save('Xtrain.npy', self.Xtrain)
        np.save('Ytrain.npy', self.Ytrain)

    @property
    def size(self):
        return self.Xtrain.shape[1]

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
            Yerr = []

            for readme in readme_list:
                with open(readme, 'r') as instream:
                    rm = yaml.load(instream)

                Xtrain.append(rm['X'])
                Ytrain.append(rm['Y'])

                if self.gp['heteroscedastic']:
                    if 'Yerr' in rm.keys():
                        Yerr.append(rm['Yerr'])
                    else:
                        Yerr.append([0., 0.])

            self.Xtrain = np.array(Xtrain).T
            self.Ytrain = np.array(Ytrain).T
            if self.gp["heteroscedastic"]:
                self.Yerr = np.array(Yerr).T

        else:
            print("No matching dtool datasets found. Start with empty database.")
            self.Xtrain = np.empty((input_dim, 0))
            self.Ytrain = np.empty((output_dim, 0))

            if self.gp['heteroscedastic']:
                self.Yerr = np.empty((2, 0))

    def update(self, Qnew, ix, iy):
        self._update_inputs(Qnew, ix, iy)

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

        Nsample = Ninit - self.size

        # Bounds for quasi random sampling of initial database
        l_bounds = [np.amin(self.h[0]), 0.0]
        u_bounds = [np.amax(self.h[0]), 2. * np.mean(Q[1, :])]

        # Sampling
        if sampling == 'random':
            h_init = l_bounds[0] + np.random.random_sample([Nsample, ]) * (u_bounds[1] - l_bounds[0])
            jx_init = l_bounds[1] + np.random.random_sample([Nsample, ]) * (u_bounds[1] - l_bounds[0])
        elif sampling == 'lhc':

            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=Nsample)

            scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
            h_init = scaled_samples[:, 0]
            jx_init = scaled_samples[:, 1]
            # jy_init = scaled_samples[:, 2]

        elif sampling == 'sobol':
            sampler = qmc.Sobol(d=2)

            m = int(np.log2(Nsample))
            if int(2**m) != Nsample:
                m = int(np.ceil(np.log2(Nsample)))
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

    def _update_inputs(self, Qnew, ix, iy):

        Hnew = self.gap_height(ix, iy)
        Xnew = np.vstack([Hnew, Qnew])

        self.Xtrain = np.hstack([self.Xtrain, Xnew])

        self._update_outputs(Xnew)

    def _update_outputs(self, Xnew):

        Ynew = np.zeros((13, Xnew.shape[1]))

        # In current version of heteroscedastic multi-output GP (single kernel):
        # one error for normal and one for shear stress.
        # Otherwise need individual kernels for tau_xy and tau_xy,
        # and possibly correlations between outputs.
        Yerrnew = np.zeros((2, Xnew.shape[1]))

        for i in range(Xnew.shape[1]):

            num = self.Ytrain.shape[1] + i + 1
            base_uri = self.gp['local']
            ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_dataset-{num:03}'
            proto_ds = dtoolcore.create_proto_dataset(name=ds_name, base_uri=base_uri)
            proto_datapath = os.path.join(base_uri, ds_name, 'data')

            # Run LAMMPS...
            if self.md is not None:

                # Run MD with fixed number of cores in proto dataset
                nworker = self.md['ncpu']
                basedir = os.getcwd()

                kw_args = dict(gap_height=Xnew[0, i],
                               vWall=self.md['vWall'],
                               density=Xnew[3, i],
                               mass_flux_x=Xnew[4, i],
                               mass_flux_y=Xnew[5, i],
                               wallfile=os.path.join(basedir, self.md['wallfile']),
                               inputfile=os.path.join(basedir, self.md['infile']))

                text = f"""Run next MD simulation in: {proto_datapath}
---
Gap height: {Xnew[0, i]:.5f}
Mass density: {Xnew[3, i]:.5f}
Mass flux: ({Xnew[4, i]:.5f}, {Xnew[5, i]:.5f})
"""

                print(bordered_text(text))

                # Run
                os.chdir(proto_datapath)

                if self.md['ncpu'] > 1:
                    mpirun('slab', kw_args, nworker)
                else:
                    run('slab', kw_args)

                # Move inputfiles to proto dataset
                proto_ds.put_item(os.path.join(basedir, self.md['wallfile']), os.path.basename(self.md['wallfile']))
                proto_ds.put_item(os.path.join(basedir, self.md['infile']), os.path.basename(self.md['infile']))

                # Get stress
                md_data = np.loadtxt('stress_wall.dat')

                if md_data.shape[1] == 5:
                    # 1D
                    # step, pL_t, tauL_t, pU_t, tauU_t = np.loadtxt('stress_wall.dat', unpack=True)
                    press = np.mean(md_data[:, 1] + md_data[:, 3]) / 2.
                    tauL = np.mean(md_data[:, 2])
                    tauU = np.mean(md_data[:, 4])

                    Ynew[0, i] = press
                    Ynew[5, i] = tauL
                    Ynew[11, i] = tauU

                    if self.gp['heteroscedastic']:
                        pL_err = variance_of_mean(md_data[:, 1])
                        pU_err = variance_of_mean(md_data[:, 3])

                        tauxzL_err = variance_of_mean(md_data[:, 2])
                        tauxzU_err = variance_of_mean(md_data[:, 4])

                        Yerrnew[0, i] = (pL_err + pU_err) / 2.
                        Yerrnew[1, i] = (tauxzL_err + tauxzU_err) / 2.

                elif md_data.shape[1] == 7:
                    # 2D
                    # step, pL_t, tauxzL_t, pU_t, tauxzU_t, tauyzL_t, tauyzU_t = np.loadtxt('stress_wall.dat', unpack=True)
                    press = np.mean(md_data[:, 1] + md_data[:, 3]) / 2.
                    tauxzL = np.mean(md_data[:, 2])
                    tauxzU = np.mean(md_data[:, 4])
                    tauyzL = np.mean(md_data[:, 5])
                    tauyzU = np.mean(md_data[:, 6])

                    Ynew[0, i] = press
                    Ynew[4, i] = tauyzL
                    Ynew[5, i] = tauxzL
                    Ynew[10, i] = tauyzU
                    Ynew[11, i] = tauxzU

                    if self.gp['heteroscedastic']:
                        pL_err = variance_of_mean(md_data[:, 1])
                        pU_err = variance_of_mean(md_data[:, 3])

                        tauxzL_err = variance_of_mean(md_data[:, 2])
                        tauxzU_err = variance_of_mean(md_data[:, 4])
                        tauyzL_err = variance_of_mean(md_data[:, 5])
                        tauyzU_err = variance_of_mean(md_data[:, 6])

                        Yerrnew[0, i] = (pL_err + pU_err) / 2.
                        Yerrnew[1, i] = (tauxzL_err + tauxzU_err + tauyzL_err + tauyzU_err) / 4.

                os.chdir(basedir)

            # ... or use a (possibly noisy) constitutive law
            else:
                # Artificial noise
                pnoise = np.random.normal(0., np.sqrt(self.gp['snp']), size=(1, Xnew.shape[1]))
                snoise = np.random.normal(0., np.sqrt(self.gp['sns']), size=(1, Xnew.shape[1]))
                Ynew[0, i] = self.eos_func(Xnew[3, i]) + pnoise[:, i]
                Ynew[1:, i, None] = self.tau_func(Xnew[3:, i, None], Xnew[:3, i, None], 0.) + snoise[:, i, None]

                if self.gp['heteroscedastic']:
                    Yerrnew[0, i] = self.gp['snp']
                    Yerrnew[1, i] = self.gp['sns']

            self.write_readme(os.path.join(base_uri, ds_name), Xnew[:, i], Ynew[:, i], Yerrnew[:, i])

            proto_ds.freeze()

            if self.gp['remote']:
                dtoolcore.copy(proto_ds.uri, self.gp['storage'])

        self.Ytrain = np.hstack([self.Ytrain, Ynew])
        if self.gp["heteroscedastic"]:
            self.Yerr = np.hstack([self.Yerr, Yerrnew])

    def write_readme(self, path, Xnew, Ynew, Yerrnew):

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
        creation_date: {DATE}
        expiration_date: {EXPIRATION_DATE}
        software_packages:
          - name: LAMMPS
            version: {version}
            website: https://lammps.sandia.gov/
            repository: https://github.com/lammps/lammps
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
        if self.md is not None:
            metadata['parameters'] = {k: self.md[k] for k in ['cutoff', 'temp', 'vWall']}

        out_fname = os.path.join(path, 'README.yml')

        X = [float(item) for item in Xnew]
        Y = [float(item) for item in Ynew]

        metadata['X'] = X
        metadata['Y'] = Y
        if self.gp['heteroscedastic']:
            Yerr = [float(item) for item in Yerrnew]
            metadata['Yerr'] = Yerr

        with open(out_fname, 'w') as outfile:
            yaml.dump(metadata, outfile)

    def gap_height(self, ix, iy=1):

        Hnew = self.h[:, ix, iy]
        if Hnew.ndim == 1:
            Hnew = Hnew[:, None]

        return Hnew


def autocorr_func_1d(x):
    """

    Compute autocorrelation function of 1D time series.

    Parameters
    ----------
    x : numpy.ndarry
        The time series

    Returns
    -------
    numpy.ndarray
        Normalized time autocorrelation function
    """
    n = len(x)

    # unbias and rescale
    x -= np.mean(x)
    x /= np.std(x)

    # pad with zeros
    ext_size = 2 * n - 1
    fsize = 2**np.ceil(np.log2(ext_size)).astype('int')

    # ACF from FFT
    x_f = np.fft.fft(x, fsize)
    C = np.fft.ifft(x_f * x_f.conjugate())[:n] / (n - np.arange(n))

    return C.real


def statistical_inefficiency(corr, mintime):
    """
    see e.g. Chodera et al. J. Chem. Theory Comput., Vol. 3, No. 1, 2007

    Parameters
    ----------
    corr : numpy.ndarray
        autocorrelation_function
    mintime : int
        Minimum time lag to calculate correlation time

    Returns
    -------
    float
        Statisitical inefficiency parameter
    """
    N = corr.size
    C_t = autocorr_func_1d(corr)
    t_grid = np.arange(N).astype('float')
    g_t = 2.0 * C_t * (1.0 - t_grid / float(N))
    ind = np.where((C_t <= 0) & (t_grid > mintime))[0][0]
    g = 1.0 + g_t[1:ind].sum()
    return max(1.0, g)


def variance_of_mean(timeseries, mintime=1):
    """

    Compute the variance of the mean value for a correlated time series

    Parameters
    ----------
    timeseries : numpy.ndarray
        The time series
    mintime : int, optional
        Minimum time lag to calculate correlation time

    Returns
    -------
    float
        Variance of the mean
    """

    g = statistical_inefficiency(timeseries, mintime)
    n = len(timeseries)
    var = np.var(timeseries) / n * g

    return var
