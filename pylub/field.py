import numpy as np


class Field:

    def __init__(self, disc, ndim, grid=True):

        self.ndim = ndim
        self.disc = disc

        self.dx = float(disc['dx'])
        self.dy = float(disc['dy'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.Lx = self.dx * self.Nx
        self.Ly = self.dy * self.Ny

        if grid:
            x = np.arange(self.Nx + 2) * (self.Lx + 2. * self.dx) / (self.Nx + 2) - self.dx / 2
            y = np.arange(self.Ny + 2) * (self.Ly + 2. * self.dy) / (self.Ny + 2) - self.dy / 2
            xx, yy = np.meshgrid(x, y)

            self.xx = xx.T
            self.yy = yy.T

        self._field = np.zeros(shape=(self.ndim, self.Nx + 2, self.Ny + 2), dtype=np.float64)

    @property
    def field(self):
        return self._field

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

    def __init__(self, disc, grid=True):
        self.ndim = 1
        super().__init__(disc, self.ndim, grid)


class VectorField(Field):

    def __init__(self, disc, grid=True):
        self.ndim = 3
        super().__init__(disc, self.ndim, grid)


class TensorField(Field):

    def __init__(self, disc, grid=True):
        self.ndim = 6
        super().__init__(disc, self.ndim, grid)
