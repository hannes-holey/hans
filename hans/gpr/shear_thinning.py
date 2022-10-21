import GPy
import numpy as np


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
