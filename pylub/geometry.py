import numpy as np


class Analytic:

    def __init__(self, disc, geometry):

        self.geometry = geometry

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

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
