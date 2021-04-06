from mpi4py import MPI
import numpy as np


class Field:

    def __init__(self, disc, ndim):

        self.ndim = ndim
        self.disc = disc

        self.dx = float(disc['dx'])
        self.dy = float(disc['dy'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])
        periodicX = bool(disc['pX'])
        periodicY = bool(disc['pY'])

        self.Lx = self.dx * self.Nx
        self.Ly = self.dy * self.Ny

        self.comm, self.rank, size_x, size_y = self.get_2d_cartesian_communicator(MPI.COMM_WORLD, (periodicX, periodicY))
        mpix, mpiy = self.comm.Get_coords(self.rank)

        # local array sizes
        nx = self.Nx // size_x
        ny = self.Ny // size_y

        # fill right and topmost grid, s.t. local arrays sizes sum up to global size
        if mpix + 1 == size_x:
            nx += self.Nx % size_x
        if mpiy + 1 == size_y:
            ny += self.Ny % size_y

        # bounds of FD grid
        # left/bottom
        xleft = mpix * nx
        ybottom = mpiy * ny

        # right/top
        xright = xleft + nx - 1
        ytop = ybottom + ny - 1

        # adjust rightmost/topmost cell
        if mpix + 1 == size_x:
            xleft = self.Nx - nx
            xright = self.Nx - 1
        if mpiy + 1 == size_y:
            ybottom = self.Ny - ny
            ytop = self.Ny - 1

        idx = np.linspace(xleft - 1, xright + 1, nx + 2, dtype=int)
        idy = np.linspace(ybottom - 1, ytop + 1, ny + 2, dtype=int)

        self.wo_ghost_x = slice(idx[0] + 1, idx[-1])
        self.wo_ghost_y = slice(idy[0] + 1, idy[-1])

        idxx, idyy = np.meshgrid(idx, idy, indexing='ij')

        self.xx = idxx * (self.Lx + 2. * self.dx) / (self.Nx + 2) + self.dx / 2
        self.yy = idyy * (self.Ly + 2. * self.dy) / (self.Ny + 2) + self.dy / 2

        self._field = np.zeros(shape=(self.ndim, nx + 2, ny + 2), dtype=np.float64)

        self.ls, self.ld, self.rs, self.rd, self.bs, self.bd, self.ts, self.td = self.get_neighbors()

    def get_neighbors(self):

        # get MPI ranks of neighboring cells
        left_src, left_dst = self.comm.Shift(0, -1)
        right_src, right_dst = self.comm.Shift(0, 1)
        bottom_src, bottom_dst = self.comm.Shift(1, -1)
        top_src, top_dst = self.comm.Shift(1, 1)

        return left_src, left_dst, right_src, right_dst, bottom_src, bottom_dst, top_src, top_dst

    def get_2d_cartesian_communicator(self, comm, periods=(True, True)):
        """Creates a two-dimensional cartesian MPI communicator.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator object
        periods : Iterable
            Directions with periodic boundary conditions (the default is (True, True)).

        Returns
        -------
        cartcomm : mpi4py.MPI.Cartcomm
            2D cartesian communicator
        rank : int
            rank of the MPI process
        size_x : int
            size of MPI grid in x direction
        size_y : int
            size of MPI grid in y direction
        """

        # get rank and size
        rank = comm.Get_rank()
        size = comm.Get_size()

        # define MPI grd sizes from number of procs
        possible_grids = [(i, size // i) for i in range(size, 0, -1) if size % i == 0]
        grid_diff = [abs(i[0] - i[1]) for i in possible_grids]
        size_x, size_y = possible_grids[grid_diff.index(min(grid_diff))]

        # create cartesian communicator
        cartcomm = comm.Create_cart(dims=(size_x, size_y), periods=periods)

        return cartcomm, rank, size_x, size_y

    @property
    def field(self):
        return self._field

    @property
    def inner(self):
        return np.ascontiguousarray(self._field[:, 1:-1, 1:-1])

    @property
    def edgeE(self):
        return (self._field + np.roll(self._field, -1, axis=1)) / 2.

    @property
    def edgeN(self):
        return (self._field + np.roll(self._field, -1, axis=2)) / 2.

    @property
    def edgeW(self):
        return (self._field + np.roll(self._field, 1, axis=1)) / 2.

    @property
    def edgeS(self):
        return (self._field + np.roll(self._field, 1, axis=2)) / 2.

    @property
    def verticeNE(self):
        return (self._field +
                np.roll(self._field, -1, axis=1) +
                np.roll(self._field, -1, axis=2) +
                np.roll(self._field, (-1, -1), axis=(1, 2))) / 4.

    @property
    def verticeSW(self):
        return (self._field +
                np.roll(self._field, 1, axis=1) +
                np.roll(self._field, 1, axis=2) +
                np.roll(self._field, (1, 1), axis=(1, 2))) / 4.


class ScalarField(Field):

    def __init__(self, disc):
        self.ndim = 1
        super().__init__(disc, self.ndim)


class VectorField(Field):

    def __init__(self, disc):
        self.ndim = 3
        super().__init__(disc, self.ndim)


class TensorField(Field):

    def __init__(self, disc):
        self.ndim = 6
        super().__init__(disc, self.ndim)
