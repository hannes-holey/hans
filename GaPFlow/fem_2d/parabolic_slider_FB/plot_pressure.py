"""
Plot pressure along x (y-averaged) from the latest simulation result.
Optionally overlays the Bayada reference data from parabolic_slider/ if present.

Usage:
    python plot_pressure.py
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

with nc.Dataset(os.path.join(latest_dir, 'sol.nc')) as ds:
    p_all = ds.variables['pressure'][:]   # (frames, nx, ny)

p_last = p_all[-1]
p_x = p_last.mean(axis=1)

Nx = p_x.shape[0]
Lx = 0.0762
x_sim = np.linspace(0, Lx, Nx)

science_style()
fig, ax = plt.subplots(figsize=(3.6, 2.0))

ax.plot(x_sim * 1e3, p_x / 1e5, color='0.3', lw=1.4, label='FB (DH EOS)')

# Overlay Bayada reference data if available
ref_csv = os.path.join(script_dir, '..', 'parabolic_slider', 'parabolic_Bayada.csv')
if os.path.isfile(ref_csv):
    import pandas as pd
    ref = pd.read_csv(ref_csv, skipinitialspace=True)
    ref_x = ref['x'].values * Lx
    ref_p = ref['y'].values
    ax.scatter(ref_x * 1e3, ref_p / 1e5, color='#1f77b4', s=12, zorder=5,
               label='Bayada (ref.)')

ax.axhline(101325 / 1e5, color='0.6', lw=0.8, ls='--', label=r'$p_\mathrm{cav}$')
ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$p$ [bar]')
ax.legend(loc='best')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

out_path = os.path.join(script_dir, 'pressure_comparison.png')
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close(fig)
