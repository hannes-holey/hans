"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
from hans.field import ScalarField, VectorField


class GapHeight(VectorField):

    def __init__(self, disc, geometry):

        super().__init__(disc)

        self.geometry = geometry
        self.disc = disc

        self.fill_analytic()
        self.fill_gradients()

    def fill_analytic(self):

        Lx = self.disc["Lx"]
        Ly = self.disc["Ly"]
        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]
        dx = self.disc["dx"]
        dy = self.disc["dy"]

        idxx, idyy = self.id_grid

        ng = self.disc["nghost"]

        xx = idxx * (Lx + 2 * ng * dx) / (Nx + 2 * ng) + dx / 2
        yy = idyy * (Ly + 2 * ng * dy) / (Ny + 2 * ng) + dy / 2

        if self.geometry["type"] in ["journal", "journal_x"]:
            CR = self.geometry['CR']
            eps = self.geometry['eps']

            Rb = Lx / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self.field[0] = c + e * np.cos(xx / Rb)

        elif self.geometry["type"] == "journal_y":
            CR = self.geometry['CR']
            eps = self.geometry['eps']

            Rb = Ly / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self.field[0] = c + e * np.cos(yy / Rb)

        elif self.geometry["type"] == "parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']
            self.field[0] = 4 * (hmax - hmin) / Lx**2 * (xx - Lx / 2)**2 + hmin

        elif self.geometry["type"] == "twin_parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']

            right = np.greater(xx, Lx / 2)
            self.field[0] = 16 * (hmax - hmin) / Lx**2 * (xx - Lx / 4)**2 + hmin
            self.field[0][right] = 16 * (hmax - hmin) / Lx**2 * (xx[right] - 3 * Lx / 4)**2 + hmin

        elif self.geometry["type"] in ["inclined", "inclined_x"]:
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            self.field[0] = h1 + (h2 - h1) / (Lx) * xx

        elif self.geometry["type"] == "inclined_y":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            self.field[0] = h1 + (h2 - h1) / (Ly) * yy

        elif self.geometry["type"] == "inclined_pocket":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            hp = self.geometry['hp']
            cp = self.geometry['c']
            lp = self.geometry['l']
            wp = self.geometry['w']

            self.field[0] = h1 + (h2 - h1) / (Lx) * xx

            xmask = np.logical_and(xx > cp, xx <= cp + lp)
            ymask = np.logical_and(yy > (Ly - wp) / 2., yy <= (Ly + wp) / 2.)
            xymask = np.logical_and(xmask, ymask)

            self.field[0, xymask] += hp

        elif self.geometry["type"] == "half_sine":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self.field[0] = h0 - amp * np.sin(- 4 * np.pi * (xx - Lx / 2) * num / Lx)
            mask = np.greater(xx, Lx / 2)
            self.field[0][mask] = h0

        elif self.geometry["type"] == "half_sine_squared":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self.field[0] = h0 + amp * np.sin(- 4 * np.pi * (xx - Lx / 2) * num / Lx)**2
            mask = np.greater(xx, Lx / 2)
            self.field[0][mask] = h0

    def fill_gradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        self.field[1:] = np.gradient(self.field[0], dx, dy, edge_order=2)


class SlipLength(ScalarField):

    def __init__(self, disc, surface):

        super().__init__(disc)

        if surface is not None:

            self.surface = surface
            self.disc = disc

            self.fill()

    def fill(self):

        Lx = self.disc["Lx"]
        Ly = self.disc["Ly"]
        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]
        dx = self.disc["dx"]
        dy = self.disc["dy"]

        idxx, idyy = self.id_grid

        ng = self.disc["nghost"]

        xx = idxx * (Lx + 2 * ng * dx) / (Nx + 2 * ng) + dx / 2
        yy = idyy * (Ly + 2 * ng * dy) / (Ny + 2 * ng) + dy / 2

        if self.surface["type"] in ["stripes", "stripes_x"]:
            num = self.surface["num"]
            sign = self.surface["sign"]
            sin = sign * np.sin(2 * np.pi * xx * num / Lx)
            mask = np.greater(sin, 0)

        elif self.surface["type"] in ["stripes_y"]:
            num = self.surface["num"]
            sign = self.surface["sign"]
            sin = sign * np.sin(2 * np.pi * yy * num / Ly)
            mask = np.greater(sin, 0)

        elif self.surface["type"] == "checkerboard":
            right_x = np.greater(xx, Lx / 2)
            center_y = np.logical_and(np.greater(yy, Ly / 4), np.less(yy, 3 * Ly / 4))
            mask_1 = np.logical_and(right_x, center_y)
            mask_2 = np.logical_and(np.logical_not(right_x), np.logical_not(center_y))
            mask = np.logical_or(mask_1, mask_2)

        elif self.surface["type"] == "circle":
            center = (3 * Lx / 4, Ly / 2)
            radius = Lx / 4
            mask = np.less((xx - center[0])**2 + (yy - center[1])**2, radius**2)

        elif self.surface["type"] == "circle2":
            center_1 = (3 * Lx / 4, Ly/2)
            center_2 = (Lx / 4, Ly)
            center_3 = (Lx / 4, 0)
            radius = Lx / 4
            mask_1 = np.less((xx - center_1[0])**2 + (yy - center_1[1])**2, radius**2)
            mask_2 = np.less((xx - center_2[0])**2 + (yy - center_2[1])**2, radius**2)
            mask_3 = np.less((xx - center_3[0])**2 + (yy - center_3[1])**2, radius**2)

            mask = np.logical_or(np.logical_or(mask_1, mask_2), mask_3)

        elif self.surface["type"] == "square":
            right_x = np.greater(xx, Lx / 2)
            center_y = np.logical_and(np.greater(yy, Ly / 4), np.less(yy, 3 * Ly / 4))
            mask = np.logical_and(right_x, center_y)

        elif self.surface["type"] == "full":
            mask = None

        ls = self.surface["lslip"]
        self.field[0, mask] = ls
