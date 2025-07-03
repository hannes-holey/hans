#
# Copyright 2023, 2025 Hannes Holey
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


def run_slab():

    nargs = ["-log", "log.lammps"]

    lmp = lammps(cmdargs=nargs)

    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    # Invoke parameters and wall definition
    lmp.command("include in.param")
    lmp.command("variable slabfile index in.wall")

    # run LAMMPS
    lmp.file("in.run")


if __name__ == "__main__":
    # main is called by an individual spawned process for parallel MD runs
    main()
