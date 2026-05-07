"""
Generate topography for the 2D convergent slider with pocket benchmark from:
  Bertocchi, L. et al. "Fluid film lubrication in the presence of cavitation:
  a mass-conserving two-dimensional formulation for compressible, piezoviscous
  and non-Newtonian fluids." Tribology International 67 (2013), 61–71.

Domain layout (Lx = 20 mm, Ly = 10 mm):
  Linear convergent wedge: h(x) = hmax - (hmax - hmin) * x / Lx
  Rectangular pocket superimposed at x in [x_p_start, x_p_end],
                                     y in [y_p_start, y_p_end]:
    h(x, y) += h_pocket

Geometry parameters:
  hmin      = 1.0e-6 m   (outlet gap)
  hmax      = 1.1e-6 m   (inlet gap)
  h_pock    = 0.4e-6 m   (pocket depth)
  x_p_start = 4.0e-3 m
  x_p_end   = 11.0e-3 m  (pocket length = 7 mm)
  y_p_start = 2.0e-3 m
  y_p_end   = 8.0e-3 m   (pocket width = 6 mm, centred in y)

Also prints the Bayada EOS cavitation pressure Pcav and the density
corresponding to p_amb = 1e5 Pa (used as Dirichlet BC at inlet/outlet).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Domain / grid parameters (must match the YAML config) ──────────────────
Lx = 0.020      # [m]  total domain length
Ly = 0.010      # [m]  domain width
Nx = 120
Ny = 60

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(dx / 2, Lx - dx / 2, Nx)   # cell centres in x
y = np.linspace(dy / 2, Ly - dy / 2, Ny)   # cell centres in y

# ── Geometry parameters ────────────────────────────────────────────────────
hmin   = 1.0e-6   # [m]  gap at outlet
hmax   = 1.1e-6   # [m]  gap at inlet
h_pock = 0.4e-6   # [m]  pocket depth

x_p_start = 4.0e-3    # [m]  pocket start in x
x_p_end   = 11.0e-3   # [m]  pocket end in x  (length = 7 mm)
y_p_start = 2.0e-3    # [m]  pocket start in y
y_p_end   = 8.0e-3    # [m]  pocket end in y  (width = 6 mm)

# ── Build 2-D height field ─────────────────────────────────────────────────
# Linear convergent wedge (x-dependent only)
xx, yy = np.meshgrid(x, y, indexing='ij')   # shape (Nx, Ny)
h2d = hmax - (hmax - hmin) * xx / Lx

# Add pocket in x-y rectangle
pocket = (
    (xx >= x_p_start) & (xx < x_p_end) &
    (yy >= y_p_start) & (yy < y_p_end)
)
h2d[pocket] += h_pock

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topography')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'conv_slider_pocket_2D_FB.npy')
np.save(out_file, h2d)
print(f"Saved height field {h2d.shape} to: {out_file}")
print(f"  h_min = {h2d.min():.3e} m,  h_max = {h2d.max():.3e} m")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 2D colour map
ax = axes[0]
im = ax.imshow(h2d.T * 1e9, origin='lower', aspect='auto',
               extent=[0, Lx * 1e3, 0, Ly * 1e3], cmap='viridis')
fig.colorbar(im, ax=ax, label='h [nm]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Height field h(x, y)')

# x-profiles at y = 0 (edge) and y = Ly/2 (centre)
ax2 = axes[1]
iy_edge   = 0
iy_centre = Ny // 2
ax2.plot(x * 1e3, h2d[:, iy_edge]   * 1e9, label=f'y = {y[iy_edge]*1e3:.1f} mm (edge)')
ax2.plot(x * 1e3, h2d[:, iy_centre] * 1e9, label=f'y = {y[iy_centre]*1e3:.1f} mm (centre)', ls='--')
ax2.axvline(x_p_start * 1e3, color='gray', ls=':', lw=0.9)
ax2.axvline(x_p_end   * 1e3, color='gray', ls=':', lw=0.9)
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('h [nm]')
ax2.set_title('Height profiles')
ax2.legend(fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.4)

fig.tight_layout()
plot_file = os.path.join(out_dir, 'conv_slider_pocket_2D_FB.png')
fig.savefig(plot_file, dpi=150)
print(f"Saved plot to: {plot_file}")
plt.show()
