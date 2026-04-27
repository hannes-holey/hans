"""Plot Bayada-Chupin EoS: pressure, dp/dρ, d²p/dρ² for the original (unsmoothed) model.

Usage:
    python plot_bayada_eos.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax.numpy as jnp
from jax import grad, vmap, jit
from GaPFlow.models.pressure import bayada_chupin


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
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
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
rho_l, rho_v, c_l, c_v = 850.0, 0.019, 1600.0, 352.0

rho = np.linspace(rho_l - 5.0, rho_l + 5.0, 5000)
rho_jax = jnp.array(rho)

# ---------------------------------------------------------------------------
# Evaluate original EoS and its derivatives via JAX autodiff
# ---------------------------------------------------------------------------
f = lambda r: bayada_chupin(r, rho_l, rho_v, c_l, c_v)
d1 = grad(f)
d2 = grad(d1)

p_fn  = jit(vmap(f))
dp_fn = jit(vmap(d1))
d2p_fn = jit(vmap(d2))

p_arr   = np.array(p_fn(rho_jax))
dp_arr  = np.array(dp_fn(rho_jax))
d2p_arr = np.array(d2p_fn(rho_jax))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
science_style()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4.0, 6.0), sharex=True)

kw_line = dict(color='tab:blue', lw=1.4)
kw_ref  = dict(color='k', ls='--', lw=0.8)

ax1.plot(rho, p_arr / 1e6, **kw_line)
ax1.axvline(rho_l, **kw_ref, label=r'$\rho_\ell$')
ax1.set_ylabel(r'$p$ [MPa]')
ax1.legend(loc='best')
ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

ax2.semilogy(rho, np.abs(dp_arr), **kw_line)
ax2.axvline(rho_l, **kw_ref)
ax2.set_ylabel(r'$|dp/d\rho|$ [Pa·m³/kg]')

ax3.semilogy(rho, np.abs(d2p_arr), **kw_line)
ax3.axvline(rho_l, **kw_ref)
ax3.set_ylabel(r'$|d^2p/d\rho^2|$ [Pa·m⁶/kg²]')
ax3.set_xlabel(r'$\rho$ [kg/m³]')

fig.tight_layout()

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bayada_eos.png")
fig.savefig(out_path, facecolor='white')
print(f"Saved: {out_path}")
plt.close(fig)
