"""Generate minimal geometry sketches for the three benchmark cases."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

N = 500  # points for smooth profile


def inclined_slider():
    Lx = 0.1
    hmax, hmin = 6.6e-5, 1.0e-5
    x = np.linspace(0, Lx, N)
    slope = (hmin - hmax) / Lx
    h = hmax + slope * x
    return x, h, hmax


def journal_bearing():
    Lx = 100 * 1e-5  # Nx*dx = 100 * 1e-5
    CR, eps = 1e-2, 0.7
    freq = 2 * np.pi / Lx
    shift = CR / freq
    amp = eps * shift
    x = np.linspace(0, Lx, N)
    h = shift + amp * np.cos(freq * x)
    return x, h, h.max()


def parabolic_slider():
    Lx = 0.0762
    hmin, hmax = 2.54e-5, 5.08e-5
    prefac = 4.0 / Lx**2 * (hmax - hmin)
    x = np.linspace(0, Lx, N)
    h = prefac * (x - Lx / 2)**2 + hmin
    return x, h, hmax


CASES = [
    ('inclined_slider', inclined_slider),
    ('journal_bearing', journal_bearing),
    ('parabolic_slider', parabolic_slider),
]


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

    for case_name, geom_func in CASES:
        x, h, h_top = geom_func()

        fig, ax = plt.subplots(figsize=(3.6, 2.16))

        # Fluid gap (blue)
        ax.fill_between(x, 0, h, color='steelblue', alpha=0.35)
        # Solid upper wall (grey)
        ax.fill_between(x, h, h_top * 1.3, color='lightgray', alpha=0.8)
        # Bottom wall
        ax.axhline(0, color='darkgray', lw=0.8)
        # Profile line
        ax.plot(x, h, color='darkgray', lw=1.2)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, h_top * 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)

        out_path = DATA_DIR / f'{case_name}_geom.png'
        fig.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
