import os
import shutil
import unittest
import netCDF4
import numpy as np

from pylub.problem import Input
from pylub.eos import EquationOfState


class TestIncompressibleJournalBearing(unittest.TestCase):

    def setUp(self):
        config_file = os.path.join("examples", "journal-bearing_incomp.yaml")
        self.tmp_dir = os.path.join("tests", "tmp")
        os.makedirs(self.tmp_dir)

        myTestProblem = Input(config_file).getProblem()
        myTestProblem.disc["Ny"] = 3
        self.material = myTestProblem.material

        myTestProblem.run(out_dir=self.tmp_dir)

    def test_pressure(self):
        ds = netCDF4.Dataset(os.path.join(self.tmp_dir, "journal-bearing_incomp_0001.nc"))
        rho = np.array(ds.variables["rho"])[-1]

        p = EquationOfState(self.material).isoT_pressure(rho)
        p = p[:, p.shape[1] // 2] / 1e6

        p_ref = np.loadtxt(os.path.join("tests", "rey_incomp_journal_0.7.dat"), usecols=(1,)) / 1e6

        np.testing.assert_almost_equal(p, p_ref, decimal=1)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        # os.remove(os.path.join(self.tmp_dir, "journal-bearing_incomp_0001.nc"))
        # os.removedirs(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()
