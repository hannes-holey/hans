"""
Generate topography for the twin identical parabolic slider (FB variant).

Same geometry as twin_parabolic_slider_id — only the EOS and cavitation
model differ. Run this once before running run.py.

Domain layout (Lx = 0.1524 m):
  [slider1 (parabola) | slider2 (parabola)]
   0.0762 m             0.0762 m

Both sliders are identical symmetric parabolas: hmax at edges, hmin at centre.

Geometry parameters:
  hmax = 50.2 µm
  hmin = 25.4 µm
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Domain / grid parameters (must match the YAML config) ──────────────────
Lx = 0.1524     # [m]  bearing length
Ly = 0.004      # [m]  periodic width
Nx = 400
Ny = 2

dx = Lx / Nx
x = np.linspace(dx / 2, Lx - dx / 2, Nx)

# ── Geometry parameters ────────────────────────────────────────────────────
hmax = 50.2e-6   # [m]
hmin = 25.4e-6   # [m]

# Each slider occupies half the domain
L_slider = Lx / 2.0

# ── Build 1-D height profile ───────────────────────────────────────────────
h1d = np.full(Nx, hmax)

# Slider 1: x in [0, L_slider]
mask1 = x < L_slider
x1_mid = L_slider / 2.0
x1_half = L_slider / 2.0
h1d[mask1] = hmin + (hmax - hmin) * ((x[mask1] - x1_mid) / x1_half)**2

# Slider 2: x in [L_slider, Lx]
mask2 = x >= L_slider
x2_mid = L_slider + L_slider / 2.0
x2_half = L_slider / 2.0
h1d[mask2] = hmin + (hmax - hmin) * ((x[mask2] - x2_mid) / x2_half)**2

# ── Extrude to 2-D (constant in y) ────────────────────────────────────────
h2d = np.tile(h1d[:, np.newaxis], (1, Ny))

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topography')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'twin_parabolic_slider_id_FB.npy')
np.save(out_file, h2d)
print(f"Saved height field {h2d.shape} to: {out_file}")
print(f"  h_min = {h2d.min():.3e} m,  h_max = {h2d.max():.3e} m")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
h_plot = h1d * 1e6
x_plot = x * 1e3
h_top = hmax * 1e6

ax.fill_between(x_plot, 0, h_plot, color='steelblue', alpha=0.35)
ax.fill_between(x_plot, h_plot, h_top * 1.25, color='lightgray', alpha=0.8)
ax.plot(x_plot, h_plot, color='darkgray', lw=1.5)
ax.axvline(L_slider * 1e3, color='gray', ls='--', lw=0.8)

ax.set_xlabel('x [mm]')
ax.set_ylabel('h [µm]')
ax.set_title('Twin identical parabolic slider FB — height profile')
ax.set_xlim(x_plot[0], x_plot[-1])
ax.set_ylim(0, h_top * 1.25)
ax.grid(True, linestyle=':', alpha=0.4)

fig.tight_layout()
plot_file = os.path.join(out_dir, 'twin_parabolic_slider_id_FB.png')
fig.savefig(plot_file, dpi=150)
print(f"Saved plot to: {plot_file}")
plt.show()
