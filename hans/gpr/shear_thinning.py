#
# Copyright 2022 Hannes Holey
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
import GPy
import numpy as np
from scipy.interpolate import interp1d


def GPRegression(file):

    # NOTE: training in 1/nanoseconds - mPas

    # TODO: should only depend on input file
    # glycosylated
    # srate, visc, visc_err, _, _ = np.loadtxt(file, delimiter=",", unpack=True)

    # non-glycosylated
    srate, _, _, visc, visc_err = np.loadtxt(file, delimiter=",", unpack=True)

    # TODO: other input: homo/hetero, (un)fixed noise, kernel, prior, output (mean or sample) ...
    xtrain = srate[:, None]
    ytrain = visc[:, None]
    yerr = visc_err[:, None]**2

    # kernel
    k = GPy.kern.RBF(1)

    # homoscedastic
    m = GPy.models.GPRegression(xtrain, ytrain, k)

    # heteroscedastic
    # m = GPy.models.GPHeteroscedasticRegression(xtrain, ytrain, k)
    # m['.*het_Gauss.variance'] = yerr     # set the noise to noise of training data
    # m.het_Gauss.variance.fix()              # fix noise parameter before optimization

    m.optimize('bfgs', max_iters=100)

    return m


def interpolate(shear_rate, material):
    m = material["model"]

    # rescale shear rate
    shear_rate /= 1e9

    nx, ny = shear_rate.shape

    shear_rate = shear_rate.flatten()[:, None]

    mean, cov = m.predict(shear_rate, full_cov=True, include_likelihood=True)

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
        out = rng.multivariate_normal(mean[:, 0], cov, method="svd")

    # rescale
    return interp1d(shear_rate[:, 0] * 1e9, out)

    # return interp1d(shear_rate[:, 0], out)
