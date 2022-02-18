#
# Copyright 2020, 2022 Hannes Holey
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
import pytest

from hans.input import Input
from hans.plottools import DatasetSelector


@pytest.fixture(scope="session", params=["MC", "RK3", "LW"])
def setup(tmpdir_factory, request):
    config_file = os.path.join("tests", "journal-bearing1D_x_incompressible.yaml")
    tmp_dir = tmpdir_factory.mktemp("tmp")

    myTestProblem = Input(config_file).getProblem()
    myTestProblem.numerics["integrator"] = request.param
    myTestProblem.run(out_dir=tmp_dir)

    ds = DatasetSelector(tmp_dir, mode="name", fname=[tmp_dir.join(os.path.basename(myTestProblem.outpath))])

    data = ds.get_centerline()[0]
    mass = ds.get_scalar(key="mass")[0]

    yield data, mass


def test_pressure(setup):
    p_ref = np.loadtxt(os.path.join("tests", "journal-bearing1D_eps0.7_incompressible.dat"), usecols=(1,))

    data, _ = setup

    xdata, ydata = data
    p = ydata["p"] / 1e6
    p_ref /= 1e6

    np.testing.assert_almost_equal(p, p_ref, decimal=1)


def test_density(setup):
    rho_ref = np.loadtxt(os.path.join("tests", "journal-bearing1D_eps0.7_incompressible.dat"), usecols=(0,))

    data, _ = setup

    xdata, ydata = data
    rho = ydata["rho"]

    np.testing.assert_almost_equal(rho, rho_ref, decimal=1)


def test_massConservation(setup):

    _, data = setup

    time, mass = data

    relDiff = abs(mass[-1] - mass[0]) / mass[0]
    assert relDiff < 0.001
