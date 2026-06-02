"""Generate minimal geometry sketch for the convergent slider with pocket."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Geometry from generate_topography.py / conv_slider_pocket_1D.yaml
Lx = 0.020
hmin = 1.0e-6
hmax = 1.1e-6
h_pock = 0.4e-6
x_p_start = 4.0e-3
x_p_end = 10.0e-3

N = 500
x = np.linspace(0, Lx, N)
h = hmax - (hmax - hmin) * x / Lx
pocket = (x >= x_p_start) & (x < x_p_end)
h[pocket] += h_pock

h_top = h.max()

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

fig, ax = plt.subplots(figsize=(3.6, 2.16))

ax.fill_between(x, 0, h, color='steelblue', alpha=0.35)
ax.fill_between(x, h, h_top * 1.3, color='lightgray', alpha=0.8)
ax.axhline(0, color='darkgray', lw=0.8)
ax.plot(x, h, color='darkgray', lw=1.2)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, h_top * 1.3)
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
