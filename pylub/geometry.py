import numpy as np


class Analytic:

    def __init__(self, disc, geometry):

        self.geometry = geometry

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

    def linearSlider(self, x, y, axis=0):
        "Linear height profile"

        h1 = float(self.geometry['h1'])
        h2 = float(self.geometry['h2'])

        sx = (h2 - h1) / (self.Lx)
        sy = 0.
        return h1 + sx * x + sy * y

    def parabolicSlider(self, x, y, axis=0):

        hmin = float(self.geometry['hmin'])
        hmax = float(self.geometry['hmax'])

        if axis == 0:
            L = self.Lx
            dir = x
        elif axis == 1:
            L = self.Ly
            dir = y

        return 4. * (hmax - hmin) / L**2 * (dir - L / 2)**2 + hmin

    def journalBearing(self, x, y, axis=0):

        CR = float(self.geometry['CR'])
        eps = float(self.geometry['eps'])

        if axis == 0:
            Rb = self.Lx / (2 * np.pi)
            dir = x
        elif axis == 1:
            Rb = self.Ly / (2 * np.pi)
            dir = y

        c = CR * Rb
        e = eps * c

        print(f"Min. channel height: {np.amin(c + e * np.cos(dir / Rb)):.2g} m")
        print(f"Max. channel height: {np.amax(c + e * np.cos(dir / Rb)):.2g} m")

        return c + e * np.cos(dir / Rb)

    def doubleStep(self, x, y, axis=0):

        h1 = float(self.geometry['h1'])
        h2 = float(self.geometry['h2'])

        if axis == 0:
            lstep = self.Lx / 3
            dir = x
        elif axis == 1:
            lstep = self.Ly / 3
            dir = y

        rstep = 2. * lstep

        mask = np.logical_and(np.less(dir, rstep), np.greater(dir, lstep))

        height = np.ones_like(x) * h1
        height[mask] = h2

        return height
