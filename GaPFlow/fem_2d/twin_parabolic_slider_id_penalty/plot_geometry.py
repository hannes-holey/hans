"""Generate minimal geometry sketch for the twin identical parabolic slider (penalty)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Geometry from generate_topography.py
Lx = 0.1524
hmax = 50.2e-6
hmin = 25.4e-6

L_slider = Lx / 2.0
N = 500
x = np.linspace(0, Lx, N)
h = np.full(N, hmax)

mask1 = x < L_slider
x1_mid = L_slider / 2.0
x1_half = L_slider / 2.0
h[mask1] = hmin + (hmax - hmin) * ((x[mask1] - x1_mid) / x1_half)**2

mask2 = x >= L_slider
x2_mid = L_slider + L_slider / 2.0
x2_half = L_slider / 2.0
h[mask2] = hmin + (hmax - hmin) * ((x[mask2] - x2_mid) / x2_half)**2

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
ax.axvline(L_slider, color='gray', ls='--', lw=0.8)

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
