"""Plot Bernoulli venturi results from .npz file produced by run_and_save.py."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def science_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.4,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def main():
    science_style()

    npz_path = DATA_DIR / 'bernoulli_venturi.npz'
    if not npz_path.exists():
        print(f"Data not found: {npz_path}")
        return

    data = np.load(npz_path)
    x_mm = data['x'] * 1e3

    fig, (ax_p, ax_jx) = plt.subplots(
        2, 1, figsize=(3.6, 4.0), sharex=True,
        gridspec_kw={'hspace': 0.08},
    )

    # Pressure
    ax_p.plot(x_mm, data['p_theory'] / 1e3, color='0.7', lw=1.4,
              label='Bernoulli')
    ax_p.plot(x_mm, data['p_sim'] / 1e3, color='#1f77b4', ls='--', lw=1.8,
              label='FEM')
    ax_p.set_ylabel(r'$p$ [kPa]')
    ax_p.legend(loc='best')
    ax_p.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_p.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    # Mass flux
    ax_jx.plot(x_mm, data['jx_theory'], color='0.7', lw=1.4)
    ax_jx.plot(x_mm, data['jx_sim'], color='#1f77b4', ls='--', lw=1.8)
    ax_jx.set_xlabel(r'$x$ [mm]')
    ax_jx.set_ylabel(r'$j_x$ $\left[\dfrac{\mathrm{kg}}{\mathrm{m}^2 \cdot \mathrm{s}}\right]$')
    ax_jx.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    out_path = DATA_DIR / 'bernoulli_venturi.png'
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
