import importlib
import lammps


def test_lmp_findable():
    lmp_spec = importlib.util.find_spec("lammps")

    assert lmp_spec is not None


def test_lmp_parallel():

    assert lammps.lammps.has_mpi_support


def test_lmp_extrafix():

    assert lammps.lammps(cmdargs=['-log', 'none', "-screen", 'none']).has_package(name='EXTRA-FIX')
