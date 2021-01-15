import numpy as np
from pylub.field import VectorField


class GapHeight(VectorField):

    def __init__(self, disc, geometry, grid=True):

        super().__init__(disc, grid)

        self.geometry = geometry
        self.fill_analytic()
        self.fill_gradients()

    def fill_analytic(self):

        if self.geometry["type"] == "journal":
            CR = float(self.geometry['CR'])
            eps = float(self.geometry['eps'])

            Rb = self.Lx / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self.field[0] = c + e * np.cos(self.xx / Rb)

            print(f"Min. channel height: {np.amin(self.field[0, 1:-1,1:-1]):.2g} m")
            print(f"Max. channel height: {np.amax(self.field[0, 1:-1,1:-1]):.2g} m")

        elif self.geometry["type"] == "parabolic":
            hmin = float(self.geometry['hmin'])
            hmax = float(self.geometry['hmax'])
            self.field[0] = (hmax - hmin) / self.Lx**2 * (self.xx - self.Lx / 2)**2 + hmin

        elif self.geometry["type"] == "inclined":
            h1 = float(self.geometry['h1'])
            h2 = float(self.geometry['h2'])
            self.field[0] = h1 + (h2 - h1) / (self.Lx) * self.xx

        elif self.geometry["type"] == "step":
            h1 = float(self.geometry['h1'])
            h2 = float(self.geometry['h2'])

            lstep = self.Lx / 3
            rstep = 2. * lstep

            mask = np.logical_and(np.less(self.xx, rstep), np.greater(self.xx, lstep))
            self.field[0] = h1
            self.field[0][mask] = h2

    def fill_gradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"
        self.field[1:] = np.gradient(self.field[0], self.dx, self.dy, edge_order=2)
