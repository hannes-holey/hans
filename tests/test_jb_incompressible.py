import os
import shutil
import unittest
import netCDF4
import numpy as np

from pylub.problem import Input
from pylub.eos import EquationOfState


class TestIncompressibleJournalBearing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_file = os.path.join("examples", "journal-bearing_incomp.yaml")
        cls.tmp_dir = os.path.join("tests", "tmp")
        if not os.path.exists(cls.tmp_dir):
            os.makedirs(cls.tmp_dir)

        myTestProblem = Input(config_file).getProblem()
        myTestProblem.disc["Ny"] = 3
        cls.material = myTestProblem.material
        myTestProblem.run(out_dir=cls.tmp_dir)

        ds = netCDF4.Dataset(os.path.join(cls.tmp_dir, "journal-bearing_incomp_0001.nc"))
        cls.rho = ds.variables["rho"]
        cls.mass = ds.variables["mass"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_pressure(self):
        p = EquationOfState(self.material).isoT_pressure(np.array(self.rho)[-1])
        p = p[:, p.shape[1] // 2] / 1e6
        p_ref = np.loadtxt(os.path.join("tests", "rey_incomp_journal_0.7.dat"), usecols=(1,)) / 1e6

        np.testing.assert_almost_equal(p, p_ref, decimal=1)

    def test_massConservation(self):
        mass = np.array(self.mass)
        relDiff = abs(mass[-1] - mass[0]) / mass[0]
        self.assertLess(relDiff, 0.001)


if __name__ == "__main__":
    unittest.main()
