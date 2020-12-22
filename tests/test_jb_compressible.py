import os
import unittest
import netCDF4
import numpy as np

from pylub.problem import Input
from pylub.eos import EquationOfState


class TestCompressibleJournalBearing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_file = os.path.join("examples", "journal-bearing.yaml")
        cls.tmp_dir = os.path.join("tests", "tmp")
        os.makedirs(cls.tmp_dir)

        myTestProblem = Input(config_file).getProblem()
        myTestProblem.disc["Ny"] = 3
        cls.material = myTestProblem.material
        myTestProblem.run(out_dir=cls.tmp_dir)

        ds = netCDF4.Dataset(os.path.join(cls.tmp_dir, "journal-bearing_0001.nc"))
        cls.rho = ds.variables["rho"]

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(cls.tmp_dir, "journal-bearing_0001.nc"))
        os.removedirs(cls.tmp_dir)

    def test_pressure(self):

        p = EquationOfState(self.material).isoT_pressure(np.array(self.rho[-1]))
        p = p[:, p.shape[1] // 2] / 1e6
        p_ref = np.loadtxt(os.path.join("tests", "rey_comp_journal_0.7.dat"), usecols=(1,)) / 1e6

        np.testing.assert_almost_equal(p, p_ref, decimal=1)

    def test_density(self):

        rho = np.array(self.rho[-1])
        rho = rho[:, rho.shape[1] // 2]
        rho_ref = np.loadtxt(os.path.join("tests", "rey_comp_journal_0.7.dat"), usecols=(0,))

        np.testing.assert_almost_equal(rho, rho_ref, decimal=1)


if __name__ == "__main__":
    unittest.main()
