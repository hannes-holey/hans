import os
import pytest
from lammps import lammps


@pytest.fixture(scope='session')
def setup(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("tmp")
    logfile = os.path.join(tmpdir, "log.lammps")

    yield logfile


def test_lmp_serial(setup):

    logfile = setup

    nargs = ["-screen", logfile, "-log", logfile]

    lmp = lammps(cmdargs=nargs)
    lmp.file(os.path.join('tests', 'in.lj'))
    lmp.close()

    with open(logfile, 'r') as f:
        last = f.readlines()[-1]

    assert last.startswith("Total wall time:")
