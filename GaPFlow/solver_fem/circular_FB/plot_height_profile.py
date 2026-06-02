"""Plot the gap height profile along x (mid-y slice) from the latest result in ./data."""

import os
import glob
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import yaml

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def latest_run_dir(data_dir):
    runs = sorted(glob.glob(os.path.join(data_dir, '*_circular_FB')))
    if not runs:
        raise FileNotFoundError(f"No runs found in {data_dir}")
    return runs[-1]


def load_grid(run_dir):
    """Read Lx and Nx from the YAML config (ignoring numpy tags)."""
    yaml_path = os.path.join(os.path.dirname(__file__), 'circular_FB.yaml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg['grid']['Lx'], cfg['grid']['Nx']


run_dir = latest_run_dir(DATA_DIR)
print(f"Reading: {run_dir}")

Lx, Nx = load_grid(run_dir)
dx = Lx / Nx
x = (np.arange(Nx) + 0.5) * dx  # cell centres, inner grid

topo_path = os.path.join(run_dir, 'topo.nc')
with nc.Dataset(topo_path) as f:
    topo = f.variables['topography'][:]

h_last = topo[-1, 0, 0, :, :]
d_last = topo[-1, 3, 0, :, :]
iy_mid = h_last.shape[1] // 2
h_profile = h_last[:, iy_mid]
d_profile = d_last[:, iy_mid]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x * 1e3, h_profile * 1e6)
ax1.set_ylabel('h  [µm]')
ax1.set_title(f'Mid-y profile along x  (last frame)\n{os.path.basename(run_dir)}')

ax2.plot(x * 1e3, d_profile * 1e9)
ax2.set_xlabel('x  [mm]')
ax2.set_ylabel('deformation  [nm]')

plt.tight_layout()
plt.savefig('height_profile.png', facecolor='white')
plt.show()
