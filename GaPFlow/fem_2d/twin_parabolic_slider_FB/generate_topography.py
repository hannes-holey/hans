"""
Generate topography for the twin parabolic slider bearing from:
  Bayada, G. et al. "From a compressible fluid model to new mass conserving
  cavitation algorithms" (Fig. 7).

Domain layout (Lx = 0.09 m):
  [slider1 (parabola) | flat_sep | slider2 (parabola) | flat_outlet]
   0.036 m              0.010 m    0.036 m               0.008 m

Each slider is a symmetric parabola: hmax at both edges, hmin at centre.
Flat regions (separator, outlet) are at hmax.

Geometry parameters:
  hmax  = 6.0e-5 m  (both sliders and flat regions)
  hmin1 = 3.0e-5 m  (slider 1 minimum)
  hmin2 = 3.5e-5 m  (slider 2 minimum)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Domain / grid parameters (must match the YAML config) ──────────────────
Lx = 0.09       # [m]  total domain length
Ly = 0.004      # [m]  small periodic width
Nx = 800
Ny = 2

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(dx / 2, Lx - dx / 2, Nx)   # cell centres

# ── Geometry parameters ────────────────────────────────────────────────────
hmax  = 6.0e-5   # [m]
hmin1 = 3.0e-5   # [m]  slider 1
hmin2 = 3.5e-5   # [m]  slider 2

# Region boundaries [m]
x_s1_start = 0.000
x_s1_end   = x_s1_start + 0.036   # 0.036

x_flat_sep_start = x_s1_end
x_flat_sep_end   = x_flat_sep_start + 0.010   # 0.046

x_s2_start = x_flat_sep_end        # 0.046
x_s2_end   = x_s2_start + 0.036   # 0.082

# flat outlet: 0.082 → 0.090  (0.008 m)

# ── Build 1-D height profile ───────────────────────────────────────────────
h1d = np.full(Nx, hmax)

# Slider 1 — symmetric parabola: hmin1 at centre, hmax at both edges.
mask1 = (x >= x_s1_start) & (x < x_s1_end)
x1_cells = x[mask1]
x1_mid = 0.5 * (x_s1_start + x_s1_end)
x1_half = 0.5 * (x_s1_end - x_s1_start)
h1d[mask1] = hmin1 + (hmax - hmin1) * ((x1_cells - x1_mid) / x1_half)**2

# Slider 2 — symmetric parabola: hmin2 at centre, hmax at both edges.
mask2 = (x >= x_s2_start) & (x < x_s2_end)
x2_cells = x[mask2]
x2_mid = 0.5 * (x_s2_start + x_s2_end)
x2_half = 0.5 * (x_s2_end - x_s2_start)
h1d[mask2] = hmin2 + (hmax - hmin2) * ((x2_cells - x2_mid) / x2_half)**2

# ── Extrude to 2-D (constant in y) ────────────────────────────────────────
h2d = np.tile(h1d[:, np.newaxis], (1, Ny))   # shape (Nx, Ny)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topography')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'twin_parabolic_slider_FB.npy')
np.save(out_file, h2d)
print(f"Saved height field {h2d.shape} to: {out_file}")
print(f"  h_min = {h2d.min():.3e} m,  h_max = {h2d.max():.3e} m")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: full 1-D profile
ax = axes[0]
h_plot = h1d * 1e6
x_plot = x * 1e3
h_top = hmax * 1e6

ax.fill_between(x_plot, 0, h_plot, color='steelblue', alpha=0.35)
ax.fill_between(x_plot, h_plot, h_top * 1.25, color='lightgray', alpha=0.8)
ax.plot(x_plot, h_plot, color='darkgray', lw=1.5)

for xv in [x_s1_end, x_s2_start, x_s2_end]:
    ax.axvline(xv * 1e3, color='gray', ls='--', lw=0.8)

ax.annotate(f'hmin₁ = {hmin1*1e6:.0f} µm', xy=(x_s1_start * 1e3 + 18, hmin1 * 1e6),
            xytext=(x_s1_start * 1e3 + 18, hmin1 * 1e6 - 6),
            fontsize=9, color='steelblue', ha='center')
ax.annotate(f'hmin₂ = {hmin2*1e6:.0f} µm', xy=(x_s2_start * 1e3 + 18, hmin2 * 1e6),
            xytext=(x_s2_start * 1e3 + 18, hmin2 * 1e6 - 6),
            fontsize=9, color='steelblue', ha='center')

ax.set_xlabel('x [mm]')
ax.set_ylabel('h [µm]')
ax.set_title('Twin parabolic slider — height profile')
ax.set_xlim(x_plot[0], x_plot[-1])
ax.set_ylim(0, h_top * 1.25)
ax.grid(True, linestyle=':', alpha=0.4)

# Right: 2-D colour map
ax2 = axes[1]
im = ax2.pcolormesh(
    np.linspace(0, Lx * 1e3, Nx + 1),
    np.linspace(0, Ly * 1e3, Ny + 1),
    h2d.T * 1e6,
    cmap='viridis_r', shading='flat'
)
fig.colorbar(im, ax=ax2, label='h [µm]')
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('y [mm]')
ax2.set_title('2-D topography (constant in y)')

fig.tight_layout()
plot_file = os.path.join(out_dir, 'twin_parabolic_slider_FB.png')
fig.savefig(plot_file, dpi=150)
print(f"Saved plot to: {plot_file}")
plt.show()
