"""
Plot pressure along x (y-averaged) from the latest simulation result,
compared against digitised Bayada reference data.

Usage:
    python plot_pressure.py
"""

import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Locate latest run directory
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

run_dirs = sorted(
    d for d in glob.glob(os.path.join(data_dir, "*_parabolic_slider"))
    if os.path.isfile(os.path.join(d, "sol.nc"))
)
if not run_dirs:
    raise FileNotFoundError(f"No sol.nc found under {data_dir}")

latest_dir = run_dirs[-1]
print(f"Using: {latest_dir}")

# ---------------------------------------------------------------------------
# Load simulation pressure (last frame, averaged over y)
# ---------------------------------------------------------------------------
with nc.Dataset(os.path.join(latest_dir, "sol.nc")) as ds:
    p_all = ds.variables["pressure"][:]      # (frames, nx, ny)

p_last = p_all[-1]                           # (nx, ny)
p_x = p_last.mean(axis=1)                   # average over y

Nx = p_x.shape[0]
Lx = 0.0762                                 # m  (from config)
x_sim = np.linspace(0, Lx, Nx)

# ---------------------------------------------------------------------------
# Load Bayada reference data
# ---------------------------------------------------------------------------
csv_path = os.path.join(script_dir, "parabolic_Bayada.csv")
ref = pd.read_csv(csv_path, skipinitialspace=True)  # columns: x, y
ref_x = ref["x"].values * Lx               # normalised [0,1] -> physical [m]
ref_p = ref["y"].values                    # already in Pa

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x_sim * 1e3, p_x / 1e5, label="GaPFlow (FEM)", color="steelblue", lw=1.5)
ax.scatter(ref_x * 1e3, ref_p / 1e5, label="Bayada (reference)", color="firebrick",
           s=20, zorder=5)

ax.set_xlabel("x  [mm]")
ax.set_ylabel("Pressure  [bar]")
ax.set_title("Parabolic slider — pressure along x")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(script_dir, "pressure_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
