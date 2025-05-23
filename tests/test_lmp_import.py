import pytest
import importlib
import lammps



@pytest.mark.skip(reason="Skip tests that require LAMMPS for now")
def test_lmp_findable():
    lmp_spec = importlib.util.find_spec("lammps")

    assert lmp_spec is not None


@pytest.mark.skip(reason="Skip tests that require LAMMPS for now")
def test_lmp_parallel():

    assert lammps.lammps.has_mpi_support


@pytest.mark.skip(reason="Skip tests that require LAMMPS for now")
def test_lmp_extrafix():

    assert lammps.lammps(cmdargs=['-log', 'none', "-screen", 'none']).has_package(name='EXTRA-FIX')
