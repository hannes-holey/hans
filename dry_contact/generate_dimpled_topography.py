"""
Generate a flat surface with four square dimples (sharp height jumps) for
texturing + cavitation tests.  Each dimple is a rectangular well with a
near-step edge profile (smoothed over ~1 cell via tanh) to avoid spurious
Gibbs-type noise while keeping the jump visually sharp.

Saves the height field as a .npy file in the topography/ subfolder and plots it.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Grid parameters (must match the config) ---
Nx = 128
Ny = 128
Lx = 0.002   # [m]
Ly = 0.002   # [m]

# --- Flat baseline height ---
h_flat = 5.0e-6   # [m]

# --- Dimple parameters ---
dimple_depth  = 3.0e-6   # [m]   extra height inside each dimple
dimple_half_w = 2.5e-4   # [m]   half-width (and half-height) of the square well
# Transition width: ~1.5 cell widths for a near-step but well-resolved edge
dx = Lx / Nx
transition = 1.5 * dx    # [m]

def square_well(XX, YY, x0, y0, hw, depth, t):
    """Return depth * smooth_square_indicator centred at (x0,y0) with half-width hw."""
    # Product of two tanh "ramp-down" functions along x and y
    wx = 0.5 * (np.tanh((XX - (x0 - hw)) / t) - np.tanh((XX - (x0 + hw)) / t))
    wy = 0.5 * (np.tanh((YY - (y0 - hw)) / t) - np.tanh((YY - (y0 + hw)) / t))
    return depth * wx * wy

# Four dimple centres at (±Lx/4, ±Ly/4)
cx = Lx / 4.0
cy = Ly / 4.0
centres = [
    ( cx,  cy),
    (-cx,  cy),
    ( cx, -cy),
    (-cx, -cy),
]

# --- Build coordinate arrays (cell centres) ---
dy = Ly / Ny
x = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
y = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, Ny)
XX, YY = np.meshgrid(x, y, indexing='ij')   # shape (Nx, Ny)

# --- Compute height field ---
h = h_flat * np.ones((Nx, Ny))
for (x0, y0) in centres:
    h += square_well(XX, YY, x0, y0, dimple_half_w, dimple_depth, transition)

# Clip so height is never negative
h = np.maximum(h, 1.0e-8)

# --- Save ---
out_dir = os.path.join(os.path.dirname(__file__), 'topography')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'dimpled_flat.npy')
np.save(out_file, h)
print(f"Saved height field {h.shape} to: {out_file}")
print(f"  h_min = {h.min():.3e} m,  h_max = {h.max():.3e} m")

# --- Plot ---
# Downsample for the 3-D surface (full 128x128 is fine, but stride keeps it crisp)
stride = 1
xs = x[::stride] * 1e3        # mm
ys = y[::stride] * 1e3        # mm
hs = h[::stride, ::stride] * 1e6   # µm
XS, YS = np.meshgrid(xs, ys, indexing='ij')

fig = plt.figure(figsize=(16, 5))

# --- 3-D surface ---
ax3d = fig.add_subplot(131, projection='3d')
surf = ax3d.plot_surface(
    XS, YS, hs,
    cmap='viridis',
    rstride=1, cstride=1,
    linewidth=0, antialiased=True,
    vmin=hs.min(), vmax=hs.max(),
)
fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1, label='h [µm]')
ax3d.set_xlabel('x [mm]')
ax3d.set_ylabel('y [mm]')
ax3d.set_zlabel('')
ax3d.set_title('Height field (4 square dimples)')
ax3d.view_init(elev=30, azim=-50)
ax3d.xaxis.set_major_locator(plt.MaxNLocator(5))
ax3d.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3d.zaxis.set_major_locator(plt.MaxNLocator(4))
# Compress z-axis so the dimples look prominent despite the thin gap
z_range = hs.max() - hs.min()
ax3d.set_zlim(hs.min() - z_range * 0.5, hs.max() + z_range * 0.5)

# --- Top-view colour map ---
ax2d = fig.add_subplot(132)
im = ax2d.pcolormesh(xs, ys, hs.T, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax2d, label='h [µm]')
ax2d.set_xlabel('x [mm]')
ax2d.set_ylabel('y [mm]')
ax2d.set_title('Top view')
ax2d.set_aspect('equal')

# --- Cross-sections ---
ax1d = fig.add_subplot(133)
ax1d.plot(xs, hs[:, Ny // (2 * stride)] , label='y = 0  (x-cut)')
ax1d.plot(ys, hs[Nx // (2 * stride), :] , label='x = 0  (y-cut)', linestyle='--')
ax1d.set_xlabel('position [mm]')
ax1d.set_ylabel('h [µm]')
ax1d.set_title('Cross-sections')
ax1d.legend()
ax1d.grid(True)

fig.tight_layout()
plot_file = os.path.join(out_dir, 'dimpled_flat.png')
fig.savefig(plot_file, dpi=150)
print(f"Saved plot to: {plot_file}")
plt.show()
