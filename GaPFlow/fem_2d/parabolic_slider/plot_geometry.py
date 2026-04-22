"""Generate minimal geometry sketch for the parabolic slider (Bayada)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Geometry from parabolic_slider.yaml
Lx = 0.0762
hmin, hmax = 2.54e-5, 5.08e-5

N = 500
x = np.linspace(0, Lx, N)
prefac = 4.0 / Lx**2 * (hmax - hmin)
h = prefac * (x - Lx / 2)**2 + hmin

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
