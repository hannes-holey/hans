"""Generate minimal geometry sketch for the Bernoulli venturi nozzle."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Geometry parameters (from notebook)
H_INLET, H_THROAT = 1.0e-3, 0.5e-3
X_RAMP_START, X_RAMP_END = 0.03, 0.07
THROAT_HALF_WIDTH = 0.005
Lx = 0.1

N = 500
x = np.linspace(0, Lx, N)

x_mid = (X_RAMP_START + X_RAMP_END) / 2
ramp_len = x_mid - THROAT_HALF_WIDTH - X_RAMP_START
dist = np.clip(np.abs(x - x_mid) - THROAT_HALF_WIDTH, 0, ramp_len)
xi = dist / ramp_len
h = np.where(
    (x >= X_RAMP_START) & (x <= X_RAMP_END),
    (H_INLET + H_THROAT) / 2 - (H_INLET - H_THROAT) / 2 * np.cos(np.pi * xi),
    H_INLET,
)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

fig, ax = plt.subplots(figsize=(3.6, 2.16))

ax.fill_between(x, 0, h, color='steelblue', alpha=0.35)
ax.fill_between(x, h, H_INLET * 1.3, color='lightgray', alpha=0.8)
ax.axhline(0, color='darkgray', lw=0.8)
ax.plot(x, h, color='darkgray', lw=1.2)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, H_INLET * 1.3)
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
