
import sys
from mpi4py import MPI


def main():

    WORKER_COMMAND = 'worker'

    command = len(sys.argv) > 1 and sys.argv[1] or '1'
    if command != WORKER_COMMAND:
        worker_count = int(command)
        print('Manager: launching {} worker.'.format(worker_count))
        sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=[sys.argv[0], WORKER_COMMAND],
                                       maxprocs=worker_count)

        common_comm = sub_comm.Merge()
        common_comm.Barrier()
        print("Manager: merged workers.")

        for i in range(worker_count):
            msg = common_comm.recv(source=MPI.ANY_SOURCE)
            print(f"Manager: received {msg}.")

        print(f"Manager: finished with fleet size {common_comm.Get_size()}.")

        common_comm.Barrier()
        common_comm.Free()

    else:
        comm = MPI.Comm.Get_parent()
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f'Worker {rank}/{size}: launched.')
        print(f"Worker {rank}/{size}: got parent.")

        comm = comm.Merge()
        print(f"Worker {rank}/{size}: merged parent.")

        comm.Barrier()

        comm.send(rank, dest=0)

        print(f"Worker {rank}/{size}: finished")

        comm.Barrier()
        comm.Free()


if __name__ == '__main__':
    main()
