"""Plot Bayada-Chupin EoS: pressure, dp/drho, d2p/drho2."""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, vmap, jit
from GaPFlow.models.pressure import bayada_chupin

rho_l, rho_v, c_l, c_v = 850.0, 0.019, 1600.0, 352.0

# Sensible range around cavitation point (rho_l = 850)
rho = np.linspace(rho_l - 5.0, rho_l + 5.0, 5000)

f = lambda r: bayada_chupin(r, rho_l, rho_v, c_l, c_v)
p_fn = jit(vmap(f))
dp_fn = jit(vmap(grad(f)))
d2p_fn = jit(vmap(grad(grad(f))))

import jax.numpy as jnp
rho_jax = jnp.array(rho)

print("Computing...")
p_arr = np.array(p_fn(rho_jax))
dp_arr = np.array(dp_fn(rho_jax))
d2p_arr = np.array(d2p_fn(rho_jax))
print("Done.")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

def auto_ylim(ax, arr):
    q01, q99 = np.nanpercentile(arr[np.isfinite(arr)], [1, 99])
    margin = max(abs(q01), abs(q99)) * 0.2
    ax.set_ylim(q01 - margin, q99 + margin)

ax1.plot(rho, p_arr, 'k-', lw=1)
ax1.set_ylabel('p(ρ)')
ax1.set_title('C0: Pressure')
ax1.axvline(rho_l, color='r', ls='--', lw=0.5, label='ρ_l')
ax1.legend()
auto_ylim(ax1, p_arr)

ax2.plot(rho, dp_arr, 'k-', lw=1)
ax2.set_ylabel("dp/dρ  (c²)")
ax2.set_title('C1: First derivative')
ax2.axvline(rho_l, color='r', ls='--', lw=0.5)
auto_ylim(ax2, dp_arr)

ax3.semilogy(rho, np.abs(d2p_arr), 'k-', lw=1)
ax3.set_ylabel('|d²p/dρ²|')
ax3.set_xlabel('ρ [kg/m³]')
ax3.set_title('C2: Second derivative (log scale)')
ax3.axvline(rho_l, color='r', ls='--', lw=0.5)

fig.tight_layout()
fig.savefig('bayada_eos_derivatives.png', dpi=150)
plt.show()
print('Saved bayada_eos_derivatives.png')
