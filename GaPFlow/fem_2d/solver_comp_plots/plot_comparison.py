"""Plot solver comparison results from .npz files produced by run_solvers.py."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

CASES = [
    ('inclined_slider', 'Inclined Slider (Air, PL EOS)'),
    ('journal_bearing', 'Journal Bearing (Oil, DH EOS)'),
    ('parabolic_slider', 'Parabolic Slider (Oil, DH EOS)'),
]

SOLVERS = {
    'explicit': {'label': 'Explicit', 'color': '0.7', 'ls': '-', 'lw': 1.4},
    'fem_2d': {'label': 'FEM 2D (Taylor-Hood)', 'color': '#1f77b4', 'ls': '--', 'lw': 1.8},
}


def science_style():
    """Matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        # Axes
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        # Ticks
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
        # Lines
        'lines.linewidth': 1.4,
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def main():
    science_style()

    for case_file, _ in CASES:
        npz_path = DATA_DIR / f'{case_file}.npz'
        if not npz_path.exists():
            print(f"Skipping {case_file}: {npz_path} not found")
            continue

        data = np.load(npz_path)
        x = data['x_norm']

        fig, (ax_rho, ax_jx) = plt.subplots(
            2, 1, figsize=(3.6, 4.0), sharex=True,
            gridspec_kw={'hspace': 0.08},
        )

        for key, style in SOLVERS.items():
            rho = data[f'rho_{key}']
            jx = data[f'jx_{key}']
            ax_rho.plot(x, rho, color=style['color'], ls=style['ls'],
                        lw=style['lw'], label=style['label'])
            ax_jx.plot(x, jx, color=style['color'], ls=style['ls'],
                       lw=style['lw'])

        # Top subplot: density
        ax_rho.set_ylabel(r'$\rho$ $\left[\dfrac{\mathrm{kg}}{\mathrm{m}^3}\right]$')
        if case_file == 'inclined_slider':
            ax_rho.legend(loc='best')
        ax_rho.xaxis.set_major_formatter(ticker.NullFormatter())

        # Bottom subplot: mass flux
        ax_jx.set_xlabel(r'$x\,/\,L$')
        ax_jx.set_ylabel(r'$j_x$ $\left[\dfrac{\mathrm{kg}}{\mathrm{m}^2 \cdot \mathrm{s}}\right]$')

        # Use ScalarFormatter to avoid offset notation
        for ax in (ax_rho, ax_jx):
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

        out_path = DATA_DIR / f'{case_file}.png'
        fig.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
