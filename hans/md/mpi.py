#
# Copyright 2023 Hannes Holey
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


def mpi_run(nworker, kw_args):

    worker_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lmpworker.py')

    sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker_file],
                                   maxprocs=nworker)

    print(sub_comm)

    common_comm = sub_comm.Merge()

    print("Merged communicator (manager)")

    common_comm.Barrier()

    print(common_comm)

    # MPI.pickle.PROTOCOL = 3

    if common_comm.Get_rank() == 0:
        system = "slab"
        kw_args = kw_args

        # kw_args = dict(gap_height=50.,
        #                vWall=0.12,
        #                density=0.8,
        #                mass_flux=0.08,
        #                slabfile="slab111-S.lammps")

    kw_args = common_comm.bcast(kw_args, root=0)
    system = common_comm.bcast(system, root=0)

    # Wait for MD to complete and free spawned communicator
    common_comm.Barrier()
    common_comm.Free()
