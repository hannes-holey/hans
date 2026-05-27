"""
Plot pressure along x (y-averaged) for the two latest simulation results,
side-by-side in separate subplots.  Each subplot has the same dimensions as
the single-panel plot in plot_pressure.py (3.6 x 2.0 in).

Usage:
    python plot_pressure_compare.py
"""

import os
import glob
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def science_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.4,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

run_dirs = sorted(
    d for d in glob.glob(os.path.join(data_dir, '*_parabolic_slider_FB'))
    if os.path.isfile(os.path.join(d, 'sol.nc'))
)
if len(run_dirs) < 2:
    raise FileNotFoundError(
        f"Need at least 2 runs with sol.nc under {data_dir}, found {len(run_dirs)}")

dirs = run_dirs[-2:]
print(f"Run 1: {dirs[0]}")
print(f"Run 2: {dirs[1]}")

Lx = 0.0762  # m (from config)

science_style()

# Two subplots, each 3.6 x 2.0 in, with a 0.4 in gap between them.
subplot_w, subplot_h = 3.6, 2.0
gap = 0.4
fig, axes = plt.subplots(1, 2, figsize=(2 * subplot_w + gap, subplot_h))

for ax, run_dir in zip(axes, dirs):
    with nc.Dataset(os.path.join(run_dir, 'sol.nc')) as ds:
        p_all = ds.variables['pressure'][:]   # (frames, nx, ny)

    p_x = p_all[-1].mean(axis=1)
    Nx = p_x.shape[0]
    x_sim = np.linspace(0, Lx, Nx)

    label = os.path.basename(run_dir)

    p_bar = p_x / 1e5
    x_mm = x_sim * 1e3
    ax.fill_between(x_mm, 0, p_bar, color='#1f77b4', alpha=0.25)
    ax.plot(x_mm, p_bar, color='#1f77b4', lw=1.4)
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$p$ [bar]')

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

out_path = os.path.join(script_dir, 'pressure_compare.png')
fig.savefig(out_path, facecolor='white')
print(f"Saved: {out_path}")
plt.close(fig)
