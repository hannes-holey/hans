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
from mpi4py import MPI
from hans.md.run import run


def main():

    comm = MPI.Comm.Get_parent().Merge()

    print("Merged communicator (worker)")
    comm.Barrier()

    assert comm != MPI.COMM_NULL

    kw_args = None
    system = None
    kw_args = comm.bcast(kw_args, root=0)
    system = comm.bcast(system, root=0)

    # system = "slab"
    # kw_args = dict(gap_height=50.,
    #                vWall=0.12,
    #                density=0.8,
    #                mass_flux=0.08,
    #                slabfile="slab111-S.lammps")

    # print('Broadcast', system)

    run(system, kw_args)

    comm.Barrier()
    comm.Free()

    # MPI.Finalize()


if __name__ == "__main__":
    main()
