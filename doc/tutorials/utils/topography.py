#
# Copyright 2025 Christoph Huber
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
"""On-the-fly topography generation for cavitation tutorial cases.

Each function accepts a fully initialised Problem instance (after from_yaml,
before run) and injects the computed height field via set_global_height so
that grid resolution can be changed freely in the YAML without regenerating
any .npy files.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _geo_grid_1d(L, N):
    """Cell-centre coordinates for a geometry grid of N points over [0, L]."""
    d = L / N
    return np.linspace(d / 2, L - d / 2, N)


def _geo_grid_2d(Lx, Ly, Nx, Ny):
    """Cell-centre coordinate arrays for a geometry grid of Nx×Ny over [0,Lx]×[0,Ly]."""
    return _geo_grid_1d(Lx, Nx), _geo_grid_1d(Ly, Ny)


def regen_conv_slider_pocket_1d(problem, Nx_geo=None):
    """Convergent slider with pocket — 1D (Bertocchi 2013).

    Linear convergent wedge (hmax=1.1 µm → hmin=1.0 µm) with a rectangular
    pocket (depth=0.4 µm) at x ∈ [4 mm, 10 mm], extruded uniformly in y.
    """
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmin, hmax, h_pock = 1.0e-6, 1.1e-6, 0.4e-6
    x_p_start, x_p_end = 4.0e-3, 10.0e-3

    h_geo = hmax - (hmax - hmin) * x_geo / Lx
    h_geo[(x_geo >= x_p_start) & (x_geo < x_p_end)] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)

    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def regen_conv_slider_pocket_1d_giacopini(problem, Nx_geo=None):
    """Convergent slider with pocket — 1D, Giacopini (2010) Table 3.

    Linear convergent wedge (K=0.01: hmax≈1.0101 µm → hmin=1.0 µm) with a
    rectangular pocket (depth=5 µm = 5*hmin) at x ∈ [2 mm, 5 mm].
    Total domain 10 mm: inlet b=2 mm, pocket c=3 mm, outlet d=5 mm.
    """
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmin = 1.0e-6
    K = 0.01
    hmax = K*hmin + hmin
    h_pock = 1.0 * hmin        # 5 µm
    x_p_start, x_p_end = 2.0e-3, 5.0e-3

    h_geo = hmax - (hmax - hmin) * x_geo / Lx
    h_geo[(x_geo >= x_p_start) & (x_geo < x_p_end)] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)

    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def regen_conv_slider_pocket_1d_hansen(problem, Nx_geo=None):
    """Convergent slider with pocket — 1D, Hansen (2022) geometry.

    Linear convergent wedge (hmax=1.1 µm → hmin=1.0 µm) with a rectangular
    pocket (depth=1.0 µm) at x ∈ [2 mm, 5 mm], total domain 10 mm.
    Inlet distance a=2 mm, pocket length b=3 mm (cf. Hansen et al. 2022).
    """
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmin, hmax, h_pock = 1.0e-6, 1.05e-6, 1.0e-6
    x_p_start, x_p_end = 2.0e-3, 5.0e-3

    h_geo = hmax - (hmax - hmin) * x_geo / Lx
    h_geo[(x_geo >= x_p_start) & (x_geo < x_p_end)] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)

    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def regen_conv_slider_pocket_2d(problem, Nx_geo=None, Ny_geo=None):
    """Convergent slider with pocket — 2D (Bertocchi 2013).

    Linear convergent wedge (hmax=1.1 µm → hmin=1.0 µm) with a rectangular
    pocket (depth=0.4 µm) at x ∈ [4 mm, 10 mm], y ∈ [2.5 mm, 7.5 mm].
    Pocket does not span the full domain width.
    """
    g = problem.grid
    Nx, Ny, Lx, Ly = g['Nx'], g['Ny'], g['Lx'], g['Ly']
    if Nx_geo is None:
        Nx_geo = Nx
    if Ny_geo is None:
        Ny_geo = Ny

    x_geo, y_geo = _geo_grid_2d(Lx, Ly, Nx_geo, Ny_geo)
    hmin, hmax, h_pock = 1.0e-6, 1.1e-6, 0.4e-6
    x_p_start, x_p_end = 4.0e-3, 10.0e-3
    y_p_start, y_p_end = 1.5e-3, 8.5e-3

    xx_geo, yy_geo = np.meshgrid(x_geo, y_geo, indexing='ij')
    h_geo = hmax - (hmax - hmin) * xx_geo / Lx
    pocket = (xx_geo >= x_p_start) & (xx_geo < x_p_end) & (yy_geo >= y_p_start) & (yy_geo < y_p_end)
    h_geo[pocket] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    y_fem = np.linspace(g['dy'] / 2, Ly - g['dy'] / 2, Ny)
    interp = RegularGridInterpolator((x_geo, y_geo), h_geo, method='linear',
                                     bounds_error=False, fill_value=None)
    xx_fem, yy_fem = np.meshgrid(x_fem, y_fem, indexing='ij')
    h2d = interp(np.stack([xx_fem.ravel(), yy_fem.ravel()], axis=-1)).reshape(Nx, Ny)

    problem.topo.set_global_height(h2d)


def regen_twin_parabolic_slider(problem, Nx_geo=None):
    """Twin parabolic slider — non-identical (Bayada et al., Fig. 7).

    Two asymmetric symmetric parabolas (hmin1=30 µm, hmin2=35 µm, hmax=60 µm)
    separated by a 10 mm flat region, followed by an 8 mm flat outlet.
    Domain Lx=0.09 m, constant in y.

    Region layout:
      slider1 [0, 36 mm] | flat_sep [36, 46 mm] | slider2 [46, 82 mm] | flat_outlet [82, 90 mm]
    """
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmax = 6.0e-5
    hmin1 = 3.0e-5
    hmin2 = 3.5e-5

    x_s1_start, x_s1_end = 0.000, 0.036
    x_s2_start, x_s2_end = 0.046, 0.082

    h_geo = np.full(Nx_geo, hmax)

    mask1 = (x_geo >= x_s1_start) & (x_geo < x_s1_end)
    x1_mid = 0.5 * (x_s1_start + x_s1_end)
    x1_half = 0.5 * (x_s1_end - x_s1_start)
    h_geo[mask1] = hmin1 + (hmax - hmin1) * ((x_geo[mask1] - x1_mid) / x1_half) ** 2

    mask2 = (x_geo >= x_s2_start) & (x_geo < x_s2_end)
    x2_mid = 0.5 * (x_s2_start + x_s2_end)
    x2_half = 0.5 * (x_s2_end - x_s2_start)
    h_geo[mask2] = hmin2 + (hmax - hmin2) * ((x_geo[mask2] - x2_mid) / x2_half) ** 2

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)

    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def regen_twin_parabolic_slider_id(problem, Nx_geo=None):
    """Twin identical parabolic slider (Sahlin / Giacopini 2010).

    Two identical symmetric parabolas (hmax=50.2 µm, hmin=25.4 µm), each
    occupying exactly half the domain. Domain Lx=0.1524 m, constant in y.
    """
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmax = 50.2e-6
    hmin = 25.4e-6
    L_slider = Lx / 2.0

    h_geo = np.full(Nx_geo, hmax)

    mask1 = x_geo < L_slider
    x1_mid = L_slider / 2.0
    h_geo[mask1] = hmin + (hmax - hmin) * ((x_geo[mask1] - x1_mid) / (L_slider / 2.0)) ** 2

    mask2 = x_geo >= L_slider
    x2_mid = L_slider + L_slider / 2.0
    h_geo[mask2] = hmin + (hmax - hmin) * ((x_geo[mask2] - x2_mid) / (L_slider / 2.0)) ** 2

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)

    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def regen_conv_slider_pocket_2d_wide(problem, Nx_geo=None, Ny_geo=None):
    """Convergent slider with pocket — 2D wide domain (Ly=300 mm).

    Same wedge and pocket geometry as the standard 2D case, but the domain
    width is 300 mm and the pocket spans 210 mm in y (centred).
    Pocket x-extent and all gap heights are unchanged.
    """
    g = problem.grid
    Nx, Ny, Lx, Ly = g['Nx'], g['Ny'], g['Lx'], g['Ly']
    if Nx_geo is None:
        Nx_geo = Nx
    if Ny_geo is None:
        Ny_geo = Ny

    x_geo, y_geo = _geo_grid_2d(Lx, Ly, Nx_geo, Ny_geo)
    hmin, hmax, h_pock = 1.0e-6, 1.1e-6, 0.4e-6
    x_p_start, x_p_end = 4.0e-3, 10.0e-3
    y_p_margin = (Ly - 0.210) / 2
    y_p_start = y_p_margin
    y_p_end = Ly - y_p_margin

    xx_geo, yy_geo = np.meshgrid(x_geo, y_geo, indexing='ij')
    h_geo = hmax - (hmax - hmin) * xx_geo / Lx
    pocket = (xx_geo >= x_p_start) & (xx_geo < x_p_end) & (yy_geo >= y_p_start) & (yy_geo < y_p_end)
    h_geo[pocket] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    y_fem = np.linspace(g['dy'] / 2, Ly - g['dy'] / 2, Ny)
    interp = RegularGridInterpolator((x_geo, y_geo), h_geo, method='linear',
                                     bounds_error=False, fill_value=None)
    xx_fem, yy_fem = np.meshgrid(x_fem, y_fem, indexing='ij')
    h2d = interp(np.stack([xx_fem.ravel(), yy_fem.ravel()], axis=-1)).reshape(Nx, Ny)

    problem.topo.set_global_height(h2d)
