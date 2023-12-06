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
