import GPy
import numpy as np
from scipy.interpolate import interp1d


def GPRegression(file):

    # NOTE: training in 1/nanoseconds - mPas
    srate, visc, visc_err = np.loadtxt(file, delimiter=",", usecols=(0, 3, 4), unpack=True)

    # TODO: other input: homo/hetero, (un)fixed noise, kernel, prior, output (mean or sample) ...

    xtrain = srate[:, None]
    ytrain = visc[:, None]

    # TODO: do not recompute GP every time viscosity is called (stoer m somewhere)
    k = GPy.kern.RBF(1)
    m = GPy.models.GPRegression(xtrain, ytrain, k)

    # heteroscedastic
    # m = GPy.models.GPHeteroscedasticRegression(srate, visc, k)
    # m['.*het_Gauss.variance'] = visc_err    # set the noise to noise of training data
    # m.het_Gauss.variance.fix()              # fix noise parameter before optimization

    m.optimize('bfgs', max_iters=100)

    return m


def interpolate(shear_rate, material):
    m = material["model"]

    # rescale shear rate
    shear_rate /= 1e9

    nx, ny = shear_rate.shape

    shear_rate = shear_rate.flatten()[:, None]

    mean, cov = m.predict(shear_rate, full_cov=True)

    # rescale posterior mean and covariance
    mean *= 1e-3
    cov *= 1e-6

    if material["sampling"] == "mean":
        out = mean[:, 0]
    elif material["sampling"] == "random":
        np.random.seed(material["seed"])

        # sample = np.random.multivariate_normal(mean[:, 0], cov)
        # new in version 1.18.0
        # methods to compute factor matrix: svd, eigh, cholesky (increasing speed)
        rng = np.random.default_rng()
        out = rng.multivariate_normal(mean[:, 0], cov, method="cholesky")

    return interp1d(shear_rate[:, 0] * 1e9, out)

    # return sample.reshape((nx, ny))
