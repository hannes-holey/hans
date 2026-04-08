"""
Generate topography for the 1D convergent slider with pocket benchmark from:
  Bertocchi, L. et al. "Fluid film lubrication in the presence of cavitation:
  a mass-conserving two-dimensional formulation for compressible, piezoviscous
  and non-Newtonian fluids." Tribology International 67 (2013), 61–71.

Domain layout (Lx = 20 mm):
  Linear convergent wedge: h(x) = hmax - (hmax - hmin) * x / Lx
  Rectangular pocket superimposed at x in [x_p_start, x_p_end]:
    h(x) += h_pocket

Geometry parameters:
  hmin    = 1.0e-6 m   (outlet gap)
  hmax    = 1.1e-6 m   (inlet gap)
  h_pock  = 0.4e-6 m   (pocket depth)
  x_p_start = 4.0e-3 m
  x_p_end   = 10.0e-3 m  (pocket length = 6 mm)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Domain / grid parameters (must match the YAML config) ──────────────────
Lx = 0.020      # [m]  total domain length
Ly = 0.004      # [m]  small periodic width
Nx = 256
Ny = 4

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(dx / 2, Lx - dx / 2, Nx)   # cell centres

# ── Geometry parameters ────────────────────────────────────────────────────
hmin   = 1.0e-6   # [m]  gap at outlet
hmax   = 1.1e-6   # [m]  gap at inlet
h_pock = 0.4e-6   # [m]  pocket depth

x_p_start = 4.0e-3   # [m]  pocket start
x_p_end   = 10.0e-3  # [m]  pocket end  (length = 6 mm)

# ── Build 1-D height profile ───────────────────────────────────────────────
# Linear convergent wedge
h1d = hmax - (hmax - hmin) * x / Lx

# Add pocket
pocket = (x >= x_p_start) & (x < x_p_end)
h1d[pocket] += h_pock

# ── Extrude to 2-D (constant in y) ────────────────────────────────────────
h2d = np.tile(h1d[:, np.newaxis], (1, Ny))   # shape (Nx, Ny)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topography')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'conv_slider_pocket_1D.npy')
np.save(out_file, h2d)
print(f"Saved height field {h2d.shape} to: {out_file}")
print(f"  h_min = {h2d.min():.3e} m,  h_max = {h2d.max():.3e} m")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))

x_plot = x * 1e3
h_plot = h1d * 1e9   # nm

ax.fill_between(x_plot, 0, h_plot, color='steelblue', alpha=0.35)
ax.plot(x_plot, h_plot, color='darkgray', lw=1.5)
ax.axvline(x_p_start * 1e3, color='gray', ls='--', lw=0.8)
ax.axvline(x_p_end * 1e3,   color='gray', ls='--', lw=0.8)

ax.set_xlabel('x [mm]')
ax.set_ylabel('h [nm]')
ax.set_title('Convergent slider with pocket — height profile')
ax.set_xlim(x_plot[0], x_plot[-1])
ax.grid(True, linestyle=':', alpha=0.4)

fig.tight_layout()
plot_file = os.path.join(out_dir, 'conv_slider_pocket_1D.png')
fig.savefig(plot_file, dpi=150)
print(f"Saved plot to: {plot_file}")
plt.show()
