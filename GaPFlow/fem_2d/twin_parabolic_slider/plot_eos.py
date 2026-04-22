"""Plot Bayada-Chupin EoS: pressure, dp/drho, d2p/drho2, d3p/drho3.

Two smoothed variants around rho_l, both spanning (rho_l-eps_l, rho_l+eps_r):
  - septic Hermite (C3): matches value, dp/drho, d2p/drho2, d3p/drho3 at both ends
  - sigmoid blend: g_smooth = g(x)*|tau| + (1-|tau|)*(g0+g1)/2
    where tau is a sigmoid scaled to [-1,1] on y over the window.
    At the boundaries |tau|=1 so g_smooth=g(x); at rho_l tau=0 so g_smooth=(g0+g1)/2.
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, vmap, jit
from GaPFlow.models.pressure import bayada_chupin


rho_l, rho_v, c_l, c_v = 850.0, 0.019, 1600.0, 352.0
eps_l = 0.3   # blend window left of rho_l (into mixture), density units
eps_r = 0.3   # blend window right of rho_l (into liquid), density units

# Sensible range around cavitation point (rho_l = 850)
rho = np.linspace(rho_l - 5.0, rho_l + 5.0, 5000)
rho_jax = jnp.array(rho)


# ---------------------------------------------------------------------------
# Septic Hermite (C3) blend
# ---------------------------------------------------------------------------

def _p_mix(dens, rho_l, rho_v, c_l, c_v):
    """p_mix branch of Bayada-Chupin (mixture region)."""
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    alpha = (dens - rho_l) / (rho_v - rho_l)
    denom = rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha)
    return Pcav + N * jnp.log(rho_v * c_v**2 * dens / denom)


def _blend_bcs(rho_l, rho_v, c_l, c_v, eps_l, eps_r):
    """Boundary conditions for the septic Hermite blend."""
    args = (rho_l, rho_v, c_l, c_v)
    rho_left = rho_l - eps_l
    d1 = grad(_p_mix)
    d2 = grad(d1)
    d3 = grad(d2)
    f0 = _p_mix(rho_left, *args)
    f0p = d1(rho_left, *args)
    f0pp = d2(rho_left, *args)
    f0ppp = d3(rho_left, *args)

    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    f1 = Pcav + eps_r * c_l**2
    f1p = c_l**2
    f1pp = 0.0
    f1ppp = 0.0

    return f0, f0p, f0pp, f0ppp, f1, f1p, f1pp, f1ppp


def bayada_chupin_c3(dens, rho_l, rho_v, c_l, c_v, eps_l=1.0, eps_r=1.0):
    """Bayada-Chupin with septic Hermite blend (C3) around rho_l."""
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

    alpha = (dens - rho_l) / (rho_v - rho_l)
    p_mix = Pcav + N * jnp.log(
        rho_v * c_v**2 * dens
        / (rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha))
    )
    p_liq = Pcav + (dens - rho_l) * c_l**2
    p_vap = c_v**2 * dens

    f0, f0p, f0pp, f0ppp, f1, f1p, f1pp, f1ppp = _blend_bcs(
        rho_l, rho_v, c_l, c_v, eps_l, eps_r
    )

    h = eps_l + eps_r
    t = (dens - (rho_l - eps_l)) / h

    H00 = 20*t**7 - 70*t**6 + 84*t**5 - 35*t**4 + 1
    H10 = 10*t**7 - 36*t**6 + 45*t**5 - 20*t**4 + t
    H20 = 2*t**7 - 7.5*t**6 + 10*t**5 - 5*t**4 + 0.5*t**2
    H30 = t**7/6 - (2/3)*t**6 + t**5 - (2/3)*t**4 + t**3/6
    H01 = -20*t**7 + 70*t**6 - 84*t**5 + 35*t**4
    H11 = 10*t**7 - 34*t**6 + 39*t**5 - 15*t**4
    H21 = -2*t**7 + 6.5*t**6 - 7*t**5 + 2.5*t**4
    H31 = t**7/6 - 0.5*t**6 + 0.5*t**5 - t**4/6

    p_blend = (H00 * f0
               + H10 * (h * f0p)
               + H20 * (h**2 * f0pp)
               + H30 * (h**3 * f0ppp)
               + H01 * f1
               + H11 * (h * f1p)
               + H21 * (h**2 * f1pp)
               + H31 * (h**3 * f1ppp))

    p = jnp.where(
        dens < rho_l - eps_l,
        p_mix,
        jnp.where(dens < rho_l + eps_r, p_blend, p_liq),
    )
    return jnp.where(dens < rho_v, p_vap, p)


# ---------------------------------------------------------------------------
# Sigmoid blend
# ---------------------------------------------------------------------------

def bayada_chupin_sigmoid(dens, rho_l, rho_v, c_l, c_v, eps_l=1.0, eps_r=1.0):
    """Bayada-Chupin with sigmoid blend around rho_l.

    Within the window (rho_l-eps_l, rho_l+eps_r):
      tau(rho) = 2*sigmoid(12*(rho-rho_l)/(eps_l+eps_r)) - 1
    scaled so tau ~ -1 at rho_l-eps_l and tau ~ +1 at rho_l+eps_r.

      g_smooth = g(rho) * |tau| + (1 - |tau|) * (g0 + g1) / 2

    g(rho) is the original EoS: p_mix for rho < rho_l, p_liq for rho > rho_l.
    g0 = p_mix(rho_l - eps_l), g1 = p_liq(rho_l + eps_r).
    Outside the window the original EoS is unchanged.
    """
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

    alpha = (dens - rho_l) / (rho_v - rho_l)
    p_mix = Pcav + N * jnp.log(
        rho_v * c_v**2 * dens
        / (rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha))
    )
    p_liq = Pcav + (dens - rho_l) * c_l**2
    p_vap = c_v**2 * dens

    # g(rho): original EoS — p_mix left of rho_l, p_liq right of rho_l
    # In the window the left half uses p_mix, right half uses p_liq.
    g = jnp.where(dens < rho_l, p_mix, p_liq)

    # tau in [-1, 1]: sigmoid argument goes from -6 at rho_l-eps_l to +6 at rho_l+eps_r
    # centered at the midpoint of the window
    h = eps_l + eps_r
    rho_mid = rho_l - eps_l + h / 2.0   # = rho_l + (eps_r - eps_l) / 2

    # g_mid: original EoS at the window midpoint (target value where |tau|=0)
    g_mid = jnp.where(
        rho_mid < rho_l,
        _p_mix(rho_mid, rho_l, rho_v, c_l, c_v),
        Pcav + (rho_mid - rho_l) * c_l**2,
    )

    tau = 2.0 * jax_sigmoid(12.0 * (dens - rho_mid) / h) - 1.0

    p_blend = g * jnp.abs(tau) + (1.0 - jnp.abs(tau)) * g_mid

    p = jnp.where(
        dens < rho_l - eps_l,
        p_mix,
        jnp.where(dens < rho_l + eps_r, p_blend, p_liq),
    )
    return jnp.where(dens < rho_v, p_vap, p)


def jax_sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_evaluators(fn):
    d1 = grad(fn)
    d2 = grad(d1)
    d3 = grad(d2)
    return jit(vmap(fn)), jit(vmap(d1)), jit(vmap(d2)), jit(vmap(d3))


f_orig = lambda r: bayada_chupin(r, rho_l, rho_v, c_l, c_v)
f_c3 = lambda r: bayada_chupin_c3(r, rho_l, rho_v, c_l, c_v, eps_l=eps_l, eps_r=eps_r)
f_sig = lambda r: bayada_chupin_sigmoid(r, rho_l, rho_v, c_l, c_v, eps_l=eps_l, eps_r=eps_r)

print("Computing original...")
p_fn, dp_fn, d2p_fn, d3p_fn = make_evaluators(f_orig)
p_arr   = np.array(p_fn(rho_jax))
dp_arr  = np.array(dp_fn(rho_jax))
d2p_arr = np.array(d2p_fn(rho_jax))
d3p_arr = np.array(d3p_fn(rho_jax))

print("Computing C3 septic blend...")
p_fn3, dp_fn3, d2p_fn3, d3p_fn3 = make_evaluators(f_c3)
p_arr3   = np.array(p_fn3(rho_jax))
dp_arr3  = np.array(dp_fn3(rho_jax))
d2p_arr3 = np.array(d2p_fn3(rho_jax))
d3p_arr3 = np.array(d3p_fn3(rho_jax))

print("Computing sigmoid blend...")
p_fns, dp_fns, d2p_fns, d3p_fns = make_evaluators(f_sig)
p_arrs   = np.array(p_fns(rho_jax))
dp_arrs  = np.array(dp_fns(rho_jax))
d2p_arrs = np.array(d2p_fns(rho_jax))
d3p_arrs = np.array(d3p_fns(rho_jax))
print("Done.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

kw_orig = dict(color='k', lw=1, label='original')
kw_c3  = dict(color='tab:orange', lw=1, ls='--', label=f'septic C3 (ε_l={eps_l}, ε_r={eps_r})')
kw_sig = dict(color='tab:blue',   lw=1, ls=':',  label=f'sigmoid (ε_l={eps_l}, ε_r={eps_r})')
kw_ref = dict(color='r', ls='--', lw=0.5)

for ax, arrs, title, ylabel in [
    (ax1, (p_arr,   p_arr3,   p_arrs),   'C0: Pressure',                    'p(ρ)'),
    (ax2, (dp_arr,  dp_arr3,  dp_arrs),  'C1: First derivative (log scale)', '|dp/dρ|'),
    (ax3, (d2p_arr, d2p_arr3, d2p_arrs), 'C2: Second derivative (log scale)','|d²p/dρ²|'),
    (ax4, (d3p_arr, d3p_arr3, d3p_arrs), 'C3: Third derivative (log scale)', '|d³p/dρ³|'),
]:
    orig, c3, sig = arrs
    if ax is ax1:
        ax.plot(rho, orig, **kw_orig)
        ax.plot(rho, c3,   **kw_c3)
        ax.plot(rho, sig,  **kw_sig)
        ax.axvline(rho_l, **kw_ref, label='ρ_l')
        ax.legend(fontsize=8)
    else:
        ax.semilogy(rho, np.abs(orig), **kw_orig)
        ax.semilogy(rho, np.abs(c3),   **kw_c3)
        ax.semilogy(rho, np.abs(sig),  **kw_sig)
        ax.axvline(rho_l, **kw_ref)
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

ax4.set_xlabel('ρ [kg/m³]')
fig.tight_layout()
fig.savefig('bayada_eos_derivatives.png', dpi=150)
plt.show()
print('Saved bayada_eos_derivatives.png')
