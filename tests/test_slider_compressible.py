import os
import shutil
import unittest
import netCDF4
import numpy as np

from pylub.input import Input
from pylub.eos import EquationOfState


class TestCompressibleInclinedSlider(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_file = os.path.join("examples", "slider_inf_DH.yaml")
        cls.tmp_dir = os.path.join("tests", "tmp")
        if not os.path.exists(cls.tmp_dir):
            os.makedirs(cls.tmp_dir)

        myTestProblem = Input(config_file).getProblem()
        cls.material = myTestProblem.material
        myTestProblem.run(out_dir=cls.tmp_dir)

        ds = netCDF4.Dataset(os.path.join(cls.tmp_dir, "slider_inf_DH_0001.nc"))
        cls.rho = ds.variables["rho"]
        cls.mass = ds.variables["mass"]
        cls.rho_ref, cls.p_ref = np.loadtxt(os.path.join("tests", "slider_2e-3_inf_DH_ref.dat"), unpack=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_pressure(self):
        p = EquationOfState(self.material).isoT_pressure(np.array(self.rho[-1]))
        p = p[:, p.shape[1] // 2] / 1e6
        self.p_ref /= 1e6
        # p_ref = np.loadtxt(os.path.join("tests", "incl_2e-3_inf_DH_ref.dat"), usecols=(1,)) / 1e6

        np.testing.assert_almost_equal(p, self.p_ref, decimal=0)

    def test_density(self):
        rho = np.array(self.rho[-1])
        rho = rho[:, rho.shape[1] // 2]
        # rho_ref = np.loadtxt(os.path.join("tests", "incl_2e-3_inf_DH_ref.dat"), usecols=(0,))

        np.testing.assert_almost_equal(rho, self.rho_ref, decimal=0)

    def test_massConservation(self):
        mass = np.array(self.mass)
        relDiff = abs(mass[-1] - mass[0]) / mass[0]
        self.assertLess(relDiff, 0.01)


if __name__ == "__main__":
    unittest.main()
