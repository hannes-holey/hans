"""
Plot Newton residual (and scaled residual) vs. cumulative iteration.

Called from run.py after problem.run():
    from plot_newton_residual import plot_newton_residual
    plot_newton_residual(problem.solver, problem.outdir)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
        'lines.linewidth': 1.0,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def plot_newton_residual(solver, outdir: str) -> None:
    """Plot R_norm and R_scaled_norm histories from a completed FEMSolver run.

    Parameters
    ----------
    solver : FEMSolver
    outdir : str
        Directory where the PNG is saved.
    """
    R_hist = solver.R_norm_history           # list[list[float]]
    Rs_hist = solver.R_scaled_norm_history   # list[list[float]]

    if not R_hist:
        print("[plot_newton_residual] No residual history found, skipping.")
        return

    # Flatten to 1-D arrays, record timestep boundaries for vertical lines
    R_flat, Rs_flat = [], []
    step_boundaries = [0]   # cumulative iteration index where each new timestep starts
    cum = 0
    for step_norms, step_norms_s in zip(R_hist, Rs_hist):
        R_flat.extend(step_norms)
        Rs_flat.extend(step_norms_s)
        cum += len(step_norms)
        step_boundaries.append(cum)

    R_flat = np.array(R_flat)
    iters = np.arange(len(R_flat))

    have_scaled = len(Rs_flat) > 0 and any(len(s) > 0 for s in Rs_hist)
    if have_scaled:
        Rs_flat = np.array(Rs_flat)

    science_style()

    nrows = 2 if have_scaled else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(4.5, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Faint vertical lines at timestep boundaries (skip first and last)
    for ax in axes:
        for b in step_boundaries[1:-1]:
            ax.axvline(b, color='0.8', lw=0.5, zorder=0)

    axes[0].semilogy(iters, R_flat, color='0.4', lw=1.0)
    axes[0].set_ylabel(r'$\|R\|$')

    if have_scaled:
        Rs_iters = np.arange(len(Rs_flat))
        axes[1].semilogy(Rs_iters, Rs_flat, color='#1f77b4', lw=1.0)
        axes[1].set_ylabel(r'$\|R_\mathrm{scaled}\|$')

        # Mark iterations where the scaled residual grew vs. the previous iteration
        growing = np.where(np.diff(Rs_flat) > 0)[0] + 1
        if len(growing):
            axes[1].semilogy(Rs_iters[growing], Rs_flat[growing],
                             'r.', ms=3, zorder=5, label='growing')
            axes[1].legend(loc='upper right', fontsize=7)

    axes[-1].set_xlabel('cumulative Newton iteration')

    for ax in axes:
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    fig.tight_layout()

    out_path = os.path.join(outdir, 'newton_residual.png')
    fig.savefig(out_path, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close(fig)
