"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import netCDF4
import numpy as np
import pytest

from pylub.input import Input
from pylub.eos import EquationOfState


@pytest.fixture(scope="session")
def setup(tmpdir_factory):
    config_file = os.path.join("examples", "slider1D_DH.yaml")
    tmp_dir = tmpdir_factory.mktemp("tmp")

    myTestProblem = Input(config_file).getProblem()
    material = myTestProblem.material
    myTestProblem.run(out_dir=tmp_dir)

    ds = netCDF4.Dataset(tmp_dir.join(os.path.basename(myTestProblem.outpath)))
    rho_ref, p_ref = np.loadtxt(os.path.join("tests", "slider_2e-3_inf_DH_ref.dat"), unpack=True)

    yield ds, rho_ref, p_ref, material


def test_pressure(setup):
    ds, rho_ref, p_ref, material = setup
    rho = np.array(ds.variables["rho"])[-1]
    p = EquationOfState(material).isoT_pressure(rho)
    p = p[:, p.shape[1] // 2] / 1e6
    p_ref /= 1e6

    np.testing.assert_almost_equal(p, p_ref, decimal=0)


def test_density(setup):
    ds, rho_ref, p_ref, material = setup
    rho = np.array(ds.variables["rho"])[-1]
    rho = rho[:, rho.shape[1] // 2]

    np.testing.assert_almost_equal(rho, rho_ref, decimal=0)


def test_massConservation(setup):
    ds, rho_ref, p_ref, material = setup
    mass = np.array(ds.variables["mass"])
    relDiff = abs(mass[-1] - mass[0]) / mass[0]
    assert relDiff < 0.01
