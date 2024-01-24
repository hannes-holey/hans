import os
import numpy as np
from lammps import lammps, formats


def run(system, cmdargs):
    """Wrapper function around implemented MD systems called by lmpworker.py

    Parameters
    ----------
    system : str
        Name of the system.
    cmdargs : dict
        Collection of keyword arguments for the system's corresponding run function

    """

    if system == "lj":
        run_lj(**cmdargs)
    elif system == "pen":
        run_pentane(**cmdargs)
    elif system == "slab":
        run_slab(**cmdargs)


def run_slab(gap_height=50., vWall=0.12, density=0.8, mass_flux=0.08, slabfile='slab111-S-R.lammps'):

    nargs = ["-log", "log.lammps"]

    lmp = lammps(cmdargs=nargs)

    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    # TODO: Other useful checks, might add here or somewhere else
    # lmp.has_mpi4py
    # lmp.has_mpi_support

    tmpdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates', 'rigid')
    slabfile = os.path.join(tmpdir, slabfile)
    inputfile = os.path.join(tmpdir, "rigid_gauss.in")

    # set variables
    lmp.command(f'variable input_gap equal {gap_height}')
    lmp.command(f'variable input_flux equal {mass_flux}')
    lmp.command(f'variable input_dens equal {density}')
    lmp.command(f'variable input_vWall equal {vWall}')
    lmp.command(f'variable slabfile index {slabfile}')

    # run LAMMPS
    lmp.file(inputfile)


def run_lj(temp=2., dens=0.452, Natoms=1000, Rcut=5., dt=0.005, tequi=10000, tsample=100000, logfile="log.lammps"):
    """Run Lennard-Jones fluid system. All quantities measured in LJ units.

    Parameters
    ----------
    temp : float
        Temperature (the default is 2.).
    dens : float
        Number density (the default is 0.452).
    Natoms : float
        Number of atoms (the default is 1000).
    Rcut : float
        Cutoff radius (the default is 5.).
    dt : float
        Timestep (the default is 0.005).
    tequi : int
        Eqilibration steps (the default is 10000).
    tsample : int
        Sampling steps (the default is 100000).
    logfile : str
        Name of the log file. If None, fallback to LAMMPS default 'log.lammps' (the default is None).

    """

    nargs = ["-log", logfile]

    lmp = lammps(cmdargs=nargs)

    try:
        length = np.power((Natoms / dens), 1 / 3)
        assert length/2 >= Rcut
    except AssertionError:
        print("Number of atoms too low for this density and cutoff radius")
        N_new = int(np.ceil((2 * Rcut)**3 * dens))
        print(f"Change number of atoms to {N_new}")
        Natoms = int(N_new)
        length = (Natoms / dens)**(1/3)

    # set variables
    lmp.command(f'variable length equal {length}')
    lmp.command(f'variable temp equal {temp}')
    lmp.command(f'variable ndens equal {dens}')
    lmp.command(f'variable cutoff equal {Rcut}')
    lmp.command(f'variable Natoms equal {Natoms}')
    lmp.command(f'variable dt equal {dt}')
    lmp.command(f'variable tsample equal {tsample}')
    lmp.command(f'variable tequi equal {tequi}')

    # run
    tmpdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')
    lmp.file(os.path.join(tmpdir, "01-lj_setup.in"))
    lmp.file(os.path.join(tmpdir, "02-lj_equi.in"))
    lmp.file(os.path.join(tmpdir, "03-lj_sample.in"))


# def run_pentane(temp=303., dens=720., Nmol=400, Rcut=10., dt=1., tequi=10000, tsample=10000, logfile="log.lammps"):
#     """Run n-pentane  system in LAMMPS. All quantities in real units.

#     Parameters
#     ----------
#     T : float
#         Temperature (K) (the default is 303.).
#     dens : type
#         Mass density (kg/m³) (the default is 720.).
#     Nmol : type
#         Number of molecules (the default is 400).
#     Rcut : type
#         Cutoff radius (Angstrom) (the default is 10.).
#     dt : type
#         Timestep (fs) (the default is 1.).
#     tequi : int
#         Eqilibration steps (the default is 10000).
#     tsample : int
#         Sampling steps (the default is 100000).
#     logfile : str
#         Name of the log file. If None, fallback to LAMMPS default 'log.lammps' (the default is None).

#     """

#     nargs = ["-log", logfile]

#     lmp = lammps(cmdargs=nargs)

#     # set variables
#     lmp.command(variable_equal("temp", temp))
#     lmp.command(variable_equal("rho", dens))
#     lmp.command(variable_equal("cutoff", Rcut))
#     lmp.command(variable_equal("nMol", Nmol))
#     lmp.command(variable_equal("dt", dt))
#     lmp.command(variable_equal("tsample", tsample))
#     lmp.command(variable_equal("tequi", tequi))

#     tmpdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'templates')
#     lmp.command(f"variable molfile index {os.path.join(tmpdir, 'pentane.mol')}")

#     # run
#     lmp.file(os.path.join(tmpdir, "01-pen_setup.in"))
#     lmp.file(os.path.join(tmpdir, "02-pen_equi.in"))
#     lmp.file(os.path.join(tmpdir, "03-pen_sample.in"))


# def extract_thermo(logfile):
#     """Extract thermodyanmic output.

#     Parameters
#     ----------
#     logfile : str
#         Name of the logfile

#     Returns
#     -------
#     list
#         List of dictionaries where each list entry stands for a single run command within a simulation.

#     """
#     log = formats.LogFile(logfile)
#     return [run for run in log.runs]
