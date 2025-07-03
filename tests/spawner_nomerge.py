#
# Copyright 2024 Hannes Holey
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

        arg = {'a': 3, 'b': 5.}
        arg = sub_comm.bcast(arg, root=MPI.ROOT)

        print(f"Manager: finished with fleet size {sub_comm.Get_size()}.")

    else:
        comm = MPI.Comm.Get_parent()
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f'Worker {rank}/{size}: launched.')
        print(f"Worker {rank}/{size}: got parent.")

        arg = comm.bcast(None, root=0)

        assert arg == {'a': 3, 'b': 5.}

        print(f"Worker {rank}/{size}: finished")


if __name__ == '__main__':
    main()
