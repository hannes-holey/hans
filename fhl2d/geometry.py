import numpy as np


class Analytic:

    def __init__(self, disc, geometry):

        self.geometry = geometry

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])

    def linearSlider(self, x, y, kwargs):
        "Linear height profile"

        h1 = float(self.geometry['h1'])
        h2 = float(self.geometry['h2'])

        sx = (h2 - h1) / (self.Lx)
        sy = 0.
        return h1 + sx * x + sy * y

    def journalBearing(self, x, y, kwargs):

        if len(kwargs) == 0:
            ax = 0
        else:
            ax = kwargs['axis']

        CR = float(self.geometry['CR'])
        eps = float(self.geometry['eps'])
        L = self.Lx

        if ax == 1:
            L = self.Ly
            x = y

        Rb = L / (2 * np.pi)

        c = CR * Rb
        e = eps * c

        print("Min. channel height: {:.2g} m".format(np.amin(c + e * np.cos(x / Rb))))
        print("Max. channel height: {:.2g} m".format(np.amax(c + e * np.cos(x / Rb))))

        return c + e * np.cos(x / Rb)
