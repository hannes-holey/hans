"""
Plot effective density rho*(1-theta) along x (y-averaged) from the latest
simulation result.

Usage:
    python plot_rho_eff.py
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
if not run_dirs:
    raise FileNotFoundError(f"No sol.nc found under {data_dir}")

latest_dir = run_dirs[-1]
print(f"Using: {latest_dir}")

Lx = 0.0762  # m (from config)

with nc.Dataset(os.path.join(latest_dir, 'sol.nc')) as ds:
    sol = ds.variables['solution'][-1, :, 0, :, :]  # (nb_comp, Nx, Ny)

nb_comp = sol.shape[0]
if nb_comp < 4:
    raise ValueError(
        f"Expected 4 solution components (rho, jx, jy, theta), got {nb_comp}. "
        "Run was probably without cavitation.")

rho   = sol[0]   # (Nx, Ny)
theta = sol[3]   # (Nx, Ny)

rho_eff = rho * (1.0 - theta)
rho_eff_x = rho_eff.mean(axis=1)   # average over y

Nx = rho_eff_x.shape[0]
x_sim = np.linspace(0, Lx, Nx)

science_style()
fig, ax = plt.subplots(figsize=(3.6, 2.0))

ax.fill_between(x_sim * 1e3, 0, rho_eff_x, color='#2ca02c', alpha=0.25)
ax.plot(x_sim * 1e3, rho_eff_x, color='#2ca02c', lw=1.4)

ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$\rho_\mathrm{eff}$ [kg/m³]')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

out_path = os.path.join(script_dir, 'rho_eff.png')
fig.savefig(out_path, facecolor='white')
print(f"Saved: {out_path}")
plt.close(fig)
