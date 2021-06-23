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

from hans.input import Input
from hans.material import Material


def p_channel_nondimensional(x, b):
    """Reference solution, cf Savio et al., Sci Adv. 2 (2016)"""

    kappa = 5 * b / (2 + 5 * b)
    slope = 6 * kappa / 5

    return (abs(x - 1) * slope - slope / 2) / kappa


@pytest.fixture(scope="session", params=[1e-4, 1e-5, 1e-6])
def setup(tmpdir_factory, request):
    config_file = os.path.join("tests", "channel-slip1D_x_incompressible.yaml")
    tmp_dir = tmpdir_factory.mktemp("tmp")

    myTestProblem = Input(config_file).getProblem()
    myTestProblem.surface["lslip"] = request.param
    material = myTestProblem.material
    myTestProblem.run(out_dir=tmp_dir)

    ds = netCDF4.Dataset(tmp_dir.join(os.path.basename(myTestProblem.outpath)))

    yield ds, material


def test_pressure(setup):

    ds, material = setup
    rho = np.array(ds.variables["rho"])[-1]

    # reference solution
    x = 2 * np.arange(100) / 100 + 1 / 100
    b = float(ds.surface_lslip / ds.geometry_h1)
    p_ref = p_channel_nondimensional(x, b)

    kappa = 5 * b / (2 + 5 * b)
    scalef = ds.geometry_h2**2 / (ds.material_shear * ds.geometry_U * ds.disc_Lx/2)
    p = Material(material).eos_pressure(rho)[:, ds.disc_Ny // 2] * scalef / kappa

    np.testing.assert_almost_equal(p, p_ref, decimal=2)
