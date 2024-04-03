#
# Copyright 2023-2024 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import sys
from mpi4py import MPI
import numpy as np
try:
    from lammps import lammps, formats
except ImportError:
    pass


def main():

    comm = MPI.Comm.Get_parent().Merge()

    comm.Barrier()

    assert comm != MPI.COMM_NULL

    kw_args = None
    kw_args = comm.bcast(kw_args, root=0)

    run(sys.argv[1], kw_args)

    comm.Barrier()
    comm.Free()


def mpirun(system, kw_args, nworker):

    worker_file = os.path.abspath(__file__)

    sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker_file, system],
                                   maxprocs=nworker)

    common_comm = sub_comm.Merge()
    common_comm.Barrier()

    if common_comm.Get_rank() == 0:
        kw_args = kw_args

    kw_args = common_comm.bcast(kw_args, root=0)

    # Wait for MD to complete and free spawned communicator
    common_comm.Barrier()
    common_comm.Free()


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


def run_slab(gap_height=50.,
             vWall=0.12,
             density=0.8,
             mass_flux_x=0.08,
             mass_flux_y=0.0,
             inputfile='in.lmp',
             wallfile='wall.lmp'):

    nargs = ["-log", "log.lammps"]

    lmp = lammps(cmdargs=nargs)

    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    # set variables
    lmp.command(f'variable input_gap equal {gap_height}')
    lmp.command(f'variable input_fluxX equal {mass_flux_x}')
    lmp.command(f'variable input_dens equal {density}')
    lmp.command(f'variable input_vWall equal {vWall}')
    lmp.command(f'variable slabfile index {wallfile}')
    lmp.command(f'variable input_fluxY equal {mass_flux_y}')

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


if __name__ == "__main__":
    # main is called by an individual spawned process for parallel MD runs
    main()
