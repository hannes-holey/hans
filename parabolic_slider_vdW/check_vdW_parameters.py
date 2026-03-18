"""
Van der Waals EOS parameter check for GaPFlow parabolic slider cavitation test.

Fluid: Argon (Ar)
Goal : Show the rho-P curve at T < T_c so that the spinodal / cavitation region
       is clearly visible, and identify operating conditions for the simulation.

VdW EOS (as implemented in GaPFlow, pressure.py):
    P = R*T * mol_dens / (1 - b_SI * mol_dens)  -  a_SI * mol_dens^2

    mol_dens = rho / M * 1000          [mol/m³]
    a_SI     = a / 10                  [m^6 Pa / mol^2]   (input a in L^2 bar/mol^2)
    b_SI     = b / 1000                [m^3 / mol]         (input b in L/mol)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── physical constants ────────────────────────────────────────────────────────
R = 8.314462618   # J / (mol K)

# ── Argon vdW parameters (GaPFlow defaults) ───────────────────────────────────
M   = 39.948       # g/mol  → molar mass
a   = 1.355        # L^2 bar / mol^2
b   = 0.03201      # L / mol

a_SI = a / 10.     # m^6 Pa / mol^2
b_SI = b / 1e3     # m^3 / mol

# ── critical point (analytical) ──────────────────────────────────────────────
T_c   = 8 * a_SI / (27 * R * b_SI)           # K
rho_c = M / (3 * b_SI) / 1000.               # kg/m³  (M in g/mol → /1000)
P_c   = a_SI / (27 * b_SI**2)               # Pa

print("=" * 55)
print("  Van der Waals parameters  –  Argon")
print("=" * 55)
print(f"  a       = {a:.4f}  L² bar / mol²")
print(f"  b       = {b:.5f} L / mol")
print(f"  M       = {M:.3f}  g / mol")
print(f"  T_c     = {T_c:.2f}  K")
print(f"  rho_c   = {rho_c:.2f}  kg/m³")
print(f"  P_c     = {P_c/1e6:.4f}  MPa")
print()

# ── chosen simulation temperature ────────────────────────────────────────────
T_sim = 130.   # K  (< T_c  →  two-phase / cavitation region)
print(f"  T_sim   = {T_sim:.1f}  K   (T/T_c = {T_sim/T_c:.3f})")

# ── helper: P(rho) via GaPFlow convention ────────────────────────────────────
def vdW_pressure(rho, T):
    mol_dens = rho / M * 1000.
    return R * T * mol_dens / (1. - b_SI * mol_dens) - a_SI * mol_dens**2

# density range (avoid singularity at rho = M/b_SI/1000)
rho_max_phys = M / (b_SI * 1000.) * 0.98   # 98 % of close-packing limit
rho_arr      = np.linspace(1., rho_max_phys, 4000)

# ── Maxwell construction to find saturation pressure at T_sim ─────────────────
# Locate spinodal densities via sign changes in dP/drho.
P_arr = vdW_pressure(rho_arr, T_sim)
dP    = np.gradient(P_arr, rho_arr)

sign_changes = np.where(np.diff(np.sign(dP)))[0]
rho_sp_v = rho_arr[sign_changes[0]]    # vapour-side spinodal  (local P minimum)
rho_sp_l = rho_arr[sign_changes[1]]    # liquid-side spinodal  (local P maximum)
P_local_min = vdW_pressure(rho_sp_v, T_sim)   # pressure at vapour spinodal
P_local_max = vdW_pressure(rho_sp_l, T_sim)   # pressure at liquid spinodal

# Equal-area (Maxwell) rule:
#   P_sat * (rho_l - rho_v) = integral_{rho_v}^{rho_l} P(rho) drho
# Rearranged residual: integral P drho - P_sat*(rho_l - rho_v) = 0
# We sweep P_sat in (P_local_min, P_local_max) and find the three rho-roots at
# each candidate, then evaluate the area condition.
def _roots_at(P_sat_candidate):
    """Return (rho_v, rho_l) roots of P(rho)=P_sat on the outer branches."""
    f = lambda r: vdW_pressure(r, T_sim) - P_sat_candidate
    # vapour branch: search between a very low density and just below rho_sp_v
    rho_lo = rho_arr[0]
    if np.sign(f(rho_lo)) == np.sign(f(rho_sp_v - 0.5)):
        return None, None   # no root on this branch
    rv = brentq(f, rho_lo, rho_sp_v - 0.5)
    # liquid branch: search between just above rho_sp_l and close-packing limit
    rho_hi = rho_max_phys - 0.5
    if np.sign(f(rho_sp_l + 0.5)) == np.sign(f(rho_hi)):
        return None, None
    rl = brentq(f, rho_sp_l + 0.5, rho_hi)
    return rv, rl

def maxwell_residual(P_sat_candidate):
    rv, rl = _roots_at(P_sat_candidate)
    if rv is None:
        return np.nan
    rho_int = np.linspace(rv, rl, 4000)
    P_int   = vdW_pressure(rho_int, T_sim)
    # area under curve minus rectangle
    return np.trapezoid(P_int, rho_int) - P_sat_candidate * (rl - rv)

# Scan to confirm sign change exists in the bracket
P_bracket = np.linspace(P_local_min * 1.001, P_local_max * 0.999, 200)
res_vals  = np.array([maxwell_residual(p) for p in P_bracket])
valid     = np.isfinite(res_vals)
idx_sign  = np.where(np.diff(np.sign(res_vals[valid])))[0]
P_lo = P_bracket[valid][idx_sign[0]]
P_hi = P_bracket[valid][idx_sign[0] + 1]

P_sat = brentq(maxwell_residual, P_lo, P_hi)
rho_v_sat, rho_l_sat = _roots_at(P_sat)

print(f"  P_sat   = {P_sat/1e6:.4f}  MPa")
print(f"  rho_v   = {rho_v_sat:.2f}  kg/m³  (saturated vapour)")
print(f"  rho_l   = {rho_l_sat:.2f}  kg/m³  (saturated liquid)")
print()

# ── simulation operating point ────────────────────────────────────────────────
# Use ambient (inlet) density slightly above rho_l_sat so the fluid is liquid
# at the boundaries but can cavitate in the diverging part of the slider.
rho0_sim = rho_l_sat * 1.01
P0_sim   = vdW_pressure(rho0_sim, T_sim)
print(f"  rho0    = {rho0_sim:.2f}  kg/m³  (inlet / reference)")
print(f"  P0      = {P0_sim/1e6:.4f}  MPa  (inlet pressure)")
print()

# speed of sound at operating point  c = sqrt(dP/drho)
drho  = rho0_sim * 1e-5
c_sim = np.sqrt((vdW_pressure(rho0_sim + drho, T_sim) -
                 vdW_pressure(rho0_sim - drho, T_sim)) / (2 * drho))
print(f"  c_sound = {c_sim:.1f}  m/s  (at rho0)")
print("=" * 55)

# ── plot ──────────────────────────────────────────────────────────────────────
T_values = [150., T_c, T_sim, 110.]
colors   = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50']
labels   = [f'T = 150 K  (T/Tc = {150/T_c:.2f})',
            f'T = Tc = {T_c:.1f} K',
            f'T = {T_sim} K  [simulation]  (T/Tc = {T_sim/T_c:.2f})',
            f'T = 110 K  (T/Tc = {110/T_c:.2f})']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── left: full rho-P curves ───────────────────────────────────────────────────
ax = axes[0]
for T_val, col, lab in zip(T_values, colors, labels):
    P_plot = vdW_pressure(rho_arr, T_val)
    ax.plot(rho_arr, P_plot / 1e6, color=col, lw=1.8, label=lab)

# mark critical point
ax.plot(rho_c, P_c / 1e6, 'k*', ms=12, zorder=5, label=f'Critical point')

# mark saturation densities at T_sim
ax.axhline(P_sat / 1e6, color='#E91E63', lw=1.0, ls=':', alpha=0.7)
ax.plot([rho_v_sat, rho_l_sat], [P_sat / 1e6, P_sat / 1e6],
        'o', color='#E91E63', ms=7, zorder=6, label='Saturation (Maxwell)')

# mark operating point
ax.plot(rho0_sim, P0_sim / 1e6, 's', color='black', ms=8, zorder=7,
        label=f'Inlet: ρ₀ = {rho0_sim:.0f} kg/m³')

ax.set_xlim(0, min(rho_max_phys, 1800))
ax.set_ylim(-5, 25)
ax.set_xlabel('ρ  [kg/m³]')
ax.set_ylabel('P  [MPa]')
ax.set_title('Van der Waals EOS  –  Argon')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
ax.axhline(0, color='k', lw=0.7, ls='--')

# ── right: zoom around the cavitation / two-phase region ──────────────────────
ax2 = axes[1]
rho_zoom = np.linspace(rho_v_sat * 0.3, rho_l_sat * 1.05, 3000)
P_zoom   = vdW_pressure(rho_zoom, T_sim)
ax2.plot(rho_zoom, P_zoom / 1e6, color='#E91E63', lw=2.2,
         label=f'T = {T_sim} K')
ax2.axhline(P_sat / 1e6, color='gray', lw=1.2, ls='--', label=f'P_sat = {P_sat/1e6:.3f} MPa')
ax2.fill_between(rho_zoom, P_zoom / 1e6, P_sat / 1e6,
                 where=(rho_zoom > rho_v_sat) & (rho_zoom < rho_l_sat),
                 alpha=0.15, color='#E91E63', label='Spinodal region')
ax2.plot(rho_v_sat, P_sat / 1e6, 'o', color='#2196F3', ms=9,
         label=f'ρ_v = {rho_v_sat:.1f} kg/m³')
ax2.plot(rho_l_sat, P_sat / 1e6, 'o', color='#4CAF50', ms=9,
         label=f'ρ_l = {rho_l_sat:.1f} kg/m³')
ax2.plot(rho0_sim, P0_sim / 1e6, 's', color='black', ms=9,
         label=f'Inlet ρ₀ = {rho0_sim:.0f} kg/m³')
ax2.axhline(0, color='k', lw=0.7, ls='--')
ax2.set_xlabel('ρ  [kg/m³]')
ax2.set_ylabel('P  [MPa]')
ax2.set_title(f'Zoom: cavitation region at T = {T_sim} K')
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.4)

fig.tight_layout()
plt.savefig('vdW_eos_check.png', dpi=150)
print("Saved: vdW_eos_check.png")
plt.show()
