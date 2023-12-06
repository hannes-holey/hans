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
