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
    SHOULD_MERGE = True
    SHOULD_DISCONNECT = False

    command = len(sys.argv) > 1 and sys.argv[1] or '1'
    if command != WORKER_COMMAND:
        worker_count = int(command)
        print('launching {} workers.'.format(worker_count))
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[sys.argv[0], WORKER_COMMAND],
                                   maxprocs=worker_count)
        print('launched workers.')
        if SHOULD_MERGE:
            comm = comm.Merge()
            print("Merged workers.")
        for i in range(worker_count):
            msg = comm.recv(source=MPI.ANY_SOURCE)
            print("Manager received {}.".format(msg))
        print("Manager finished with fleet size {}.".format(comm.Get_size()))
    else:
        print('worker launched.')
        comm = MPI.Comm.Get_parent()
        print("Got parent.")
        if SHOULD_MERGE:
            comm = comm.Merge()
            print("Merged parent.")
        size = comm.Get_size()
        rank = comm.Get_rank()
        comm.send(rank, dest=0)

        print("Finished worker: rank {} of {}".format(rank, size))

    if SHOULD_DISCONNECT:
        comm.Disconnect()
        print("Finished with command {}.".format(command))


if __name__ == '__main__':
    main()
