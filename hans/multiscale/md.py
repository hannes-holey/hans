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
try:
    from lammps import lammps
except ImportError:
    pass


def main():

    comm = MPI.Comm.Get_parent()

    # Parameter broadcasting fails on some systems
    # kw_args = comm.bcast(None, root=0)

    run(sys.argv[1])

    comm.Barrier()
    comm.Free()


def mpirun(system, nworker):

    worker_file = os.path.abspath(__file__)

    sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker_file, system],
                                   maxprocs=nworker)

    # Parameter broadcasting fails on some systems
    # kw_args = sub_comm.bcast(kw_args, root=0)

    # Wait for MD to complete and free spawned communicator
    sub_comm.Barrier()
    sub_comm.Free()


def run(system):
    """Wrapper function around implemented MD systems called by lmpworker.py

    Parameters
    ----------
    system : str
        Name of the system.

    """

    assert system == 'slab'
    run_slab()

    # Select type of simulation, currently only 'slab' implemented
    # if system == "lj":
    #     run_lj()
    # elif system == "slab":
    #     run_slab()


def run_slab():

    nargs = ["-log", "log.lammps"]

    lmp = lammps(cmdargs=nargs)

    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    # Invoke parameters and wall definition
    lmp.command("include in.param")
    lmp.command("variable slabfile index in.wall")

    # run LAMMPS
    lmp.file("in.run")


# def run_lj(temp=2., dens=0.452, Natoms=1000, Rcut=5., dt=0.005, tequi=10000, tsample=100000, logfile="log.lammps"):
#     """Run Lennard-Jones fluid system. All quantities measured in LJ units.

#     Parameters
#     ----------
#     temp : float
#         Temperature (the default is 2.).
#     dens : float
#         Number density (the default is 0.452).
#     Natoms : float
#         Number of atoms (the default is 1000).
#     Rcut : float
#         Cutoff radius (the default is 5.).
#     dt : float
#         Timestep (the default is 0.005).
#     tequi : int
#         Eqilibration steps (the default is 10000).
#     tsample : int
#         Sampling steps (the default is 100000).
#     logfile : str
#         Name of the log file. If None, fallback to LAMMPS default 'log.lammps' (the default is None).

#     """

#     nargs = ["-log", logfile]

#     lmp = lammps(cmdargs=nargs)

#     try:
#         length = np.power((Natoms / dens), 1 / 3)
#         assert length/2 >= Rcut
#     except AssertionError:
#         print("Number of atoms too low for this density and cutoff radius")
#         N_new = int(np.ceil((2 * Rcut)**3 * dens))
#         print(f"Change number of atoms to {N_new}")
#         Natoms = int(N_new)
#         length = (Natoms / dens)**(1/3)

#     # set variables
#     lmp.command(f'variable length equal {length}')
#     lmp.command(f'variable temp equal {temp}')
#     lmp.command(f'variable ndens equal {dens}')
#     lmp.command(f'variable cutoff equal {Rcut}')
#     lmp.command(f'variable Natoms equal {Natoms}')
#     lmp.command(f'variable dt equal {dt}')
#     lmp.command(f'variable tsample equal {tsample}')
#     lmp.command(f'variable tequi equal {tequi}')

#     # run
#     tmpdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')
#     lmp.file(os.path.join(tmpdir, "01-lj_setup.in"))
#     lmp.file(os.path.join(tmpdir, "02-lj_equi.in"))
#     lmp.file(os.path.join(tmpdir, "03-lj_sample.in"))


if __name__ == "__main__":
    # main is called by an individual spawned process for parallel MD runs
    main()
