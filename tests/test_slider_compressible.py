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

    ds = netCDF4.Dataset(tmp_dir.join("slider1D_DH_0001.nc"))
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
