"""Generate minimal geometry sketch for the twin parabolic slider (Bayada)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Geometry from generate_topography.py
Lx = 0.09
hmax = 6.0e-5
hmin1 = 3.0e-5
hmin2 = 3.5e-5

x_s1_start = 0.0
x_s1_end = 0.036
x_s2_start = 0.046
x_s2_end = 0.082

N = 500
x = np.linspace(0, Lx, N)
h = np.full(N, hmax)

mask1 = (x >= x_s1_start) & (x < x_s1_end)
x1_mid = 0.5 * (x_s1_start + x_s1_end)
x1_half = 0.5 * (x_s1_end - x_s1_start)
h[mask1] = hmin1 + (hmax - hmin1) * ((x[mask1] - x1_mid) / x1_half)**2

mask2 = (x >= x_s2_start) & (x < x_s2_end)
x2_mid = 0.5 * (x_s2_start + x_s2_end)
x2_half = 0.5 * (x_s2_end - x_s2_start)
h[mask2] = hmin2 + (hmax - hmin2) * ((x[mask2] - x2_mid) / x2_half)**2

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

fig, ax = plt.subplots(figsize=(3.6, 2.16))

ax.fill_between(x, 0, h, color='steelblue', alpha=0.35)
ax.fill_between(x, h, hmax * 1.3, color='lightgray', alpha=0.8)
ax.axhline(0, color='darkgray', lw=0.8)
ax.plot(x, h, color='darkgray', lw=1.2)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, hmax * 1.3)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.5)

out_path = script_dir / 'geometry.png'
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close(fig)
