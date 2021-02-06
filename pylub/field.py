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

    def fill_circle(self, value, comp, center=None, radius=None):

        if center is None:
            center = (self.Lx / 2, self.Ly / 2)
        if radius is None:
            radius = 0.25 * min(self.Lx / 2, self.Ly / 2)

        mask = (self.xx - center[0])**2 + (self.yy - center[1])**2 < radius**2
        self.field[comp][mask] = value

    def fill_line(self, value, ax, comp, loc=None, width=None):

        if loc is None:
            loc = (self.Lx / 2, self.Ly / 2)
        if width is None:
            width = 0.2 * (self.Lx, self.Ly)[ax]

        mask = abs(self.xx - loc[ax]) < width / 2.
        self.field[comp][mask] = value

    def fromFunctionXY(self, func, comp, **kwargs):
        self.field[comp] = func(self.xx, self.yy, kwargs)

    def edgesField(self):
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)

        x = np.linspace(0., self.Lx, self.Nx)
        y = np.linspace(0., self.Ly, self.Ny)
        xx, yy = np.meshgrid(x, y)

        self.xx = xx.T
        self.yy = yy.T

    def stagArray(self, dir, ax):

        out = Field(self.disc, self.ndim)
        out._field = 0.5 * (self.field + np.roll(self.field, dir, axis=ax))

        return out

    def cellAverage(self):

        out = Field(self.disc, self.ndim)

        E = self.stagArray(-1, 1)
        W = self.stagArray(1, 1)

        NE = E.stagArray(-1, 2)
        SE = E.stagArray(1, 2)
        NW = W.stagArray(-1, 2)
        SW = W.stagArray(1, 2)

        out.field = 0.25 * (NE.field + SE.field + NW.field + SW.field)

        return out


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
