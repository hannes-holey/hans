import numpy as np
from pylub.field import VectorField


class GapHeight(VectorField):

    def __init__(self, disc, geometry):

        super().__init__(disc)

        self.geometry = geometry
        self.fill_analytic()
        self.fill_gradients()

    def fill_analytic(self):

        if self.geometry["type"] == "journal":
            CR = self.geometry['CR']
            eps = self.geometry['eps']

            Rb = self.Lx / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self._field[0] = c + e * np.cos(self.xx / Rb)

            # print(f"Min. channel height: {np.amin(self.inner[0]):.2g} m", flush=True)
            # print(f"Max. channel height: {np.amax(self.inner[0]):.2g} m", flush=True)

        elif self.geometry["type"] == "parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']
            self._field[0] = 4 * (hmax - hmin) / self.Lx**2 * (self.xx - self.Lx / 2)**2 + hmin

        elif self.geometry["type"] == "twin_parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']

            right = np.greater(self.xx, self.Lx / 2)
            self._field[0] = 16 * (hmax - hmin) / self.Lx**2 * (self.xx - self.Lx / 4)**2 + hmin
            self._field[0][right] = 16 * (hmax - hmin) / self.Lx**2 * (self.xx[right] - 3 * self.Lx / 4)**2 + hmin

        elif self.geometry["type"] == "inclined":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            self._field[0] = h1 + (h2 - h1) / (self.Lx) * self.xx

        elif self.geometry["type"] == "inclined_pocket":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            hp = self.geometry['hp']
            c = self.geometry['c']
            l = self.geometry['l']
            w = self.geometry['w']

            self._field[0] = h1 + (h2 - h1) / (self.Lx) * self.xx

            xmask = np.logical_and(self.xx > c, self.xx <= c + l)
            ymask = np.logical_and(self.yy > (self.Ly - w) / 2., self.yy <= (self.Ly + w) / 2.)
            xymask = np.logical_and(xmask, ymask)

            self._field[0, xymask] += hp

        elif self.geometry["type"] == "half_sine":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self._field[0] = h0 - amp * np.sin(- 4 * np.pi * (self.xx - self.Lx / 2) * num / self.Lx)
            mask = np.greater(self.xx, self.Lx / 2)
            self._field[0][mask] = h0

        elif self.geometry["type"] == "half_sine_squared":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self._field[0] = h0 + amp * np.sin(- 4 * np.pi * (self.xx - self.Lx / 2) * num / self.Lx)**2
            mask = np.greater(self.xx, self.Lx / 2)
            self._field[0][mask] = h0

    def fill_gradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"
        self._field[1:] = np.gradient(self._field[0], self.dx, self.dy, edge_order=2)
