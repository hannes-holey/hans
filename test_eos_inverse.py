"""Validation script for the pressure-based EoS inverse (rho_of_p).

Checks, for Dowson-Higginson and Bayada-Chupin:
  1. rho -> p -> rho round-trip accuracy (over a wide rho range).
  2. p -> rho -> p round-trip accuracy (over the matching p range).
  3. Jacobian consistency:  dp/drho(rho) * drho/dp(p) == 1  at matching points.
  4. Continuity of the Bayada inverse at the region boundaries (rho_l, rho_v).

All derivatives are obtained via jax.grad from the forward EoS (models/pressure.py)
and the proposed analytic inverses (defined here).
"""

import os
import sys

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GaPFlow.models.pressure import (
    dowson_higginson, bayada_chupin,
    rho_of_p_dh, rho_of_p_bayada,
    eos_rho, eos_drho_dp,
)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _report(label, err):
    err_np = np.asarray(err)
    worst = int(np.argmax(err_np))
    med = float(np.median(err_np))
    print(f"  {label:<40s} max = {err_np.max():.3e}   median = {med:.3e}   (worst idx {worst})")


def roundtrip_forward_inverse(forward_fn, inverse_fn, rhos, label):
    p = forward_fn(rhos)
    rho_back = inverse_fn(p)
    err = jnp.abs(rho_back - rhos) / jnp.maximum(jnp.abs(rhos), 1e-30)
    _report(f"{label} rho→p→rho", err)


def roundtrip_inverse_forward(forward_fn, inverse_fn, ps, label):
    rho = inverse_fn(ps)
    p_back = forward_fn(rho)
    err = jnp.abs(p_back - ps) / jnp.maximum(jnp.abs(ps), 1e-30)
    _report(f"{label} p→rho→p", err)


def jacobian_consistency(forward_fn, inverse_fn, rhos, label):
    dp_drho = vmap(grad(forward_fn))
    drho_dp = vmap(grad(inverse_fn))
    p = forward_fn(rhos)
    err = jnp.abs(dp_drho(rhos) * drho_dp(p) - 1.0)
    _report(f"{label} dp/drho · drho/dp - 1", err)


def gradient_direct_check(forward_fn, inverse_fn, ps, label):
    """Cross-check: drho_dp(p) from jax.grad(inverse) matches 1/dp_drho(rho_of_p(p))
    computed via jax.grad(forward) at the consistent rho.  Relative comparison."""
    drho_dp_from_inv = vmap(grad(inverse_fn))(ps)
    rhos = vmap(inverse_fn)(ps)
    dp_drho_from_fwd = vmap(grad(forward_fn))(rhos)
    err = jnp.abs(drho_dp_from_inv - 1.0 / dp_drho_from_fwd) / jnp.maximum(jnp.abs(1.0 / dp_drho_from_fwd), 1e-30)
    _report(f"{label} drho_dp vs 1/dp_drho rel", err)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_dowson_higginson():
    print("=== Dowson-Higginson ===")
    rho0, P0, C1, C2 = 877.7007, 101325., 3.5e8, 1.23

    # Unwrap to scalar-input jax functions for grad/vmap
    fwd = jit(lambda rho: dowson_higginson(rho, rho0, P0, C1, C2))
    inv = jit(lambda p:   rho_of_p_dh(p,       rho0, P0, C1, C2))

    # Dense sweep just below the forward's clamp 0.99 * C2 * rho0.
    rhos = jnp.linspace(0.50 * rho0, 1.215 * rho0, 2000)
    roundtrip_forward_inverse(fwd, inv, rhos, "DH ρ→p→ρ")

    # p sweep: 1 Pa to 1 GPa (includes gauge pressures near P0 where forward
    # has catastrophic cancellation — shown separately).
    ps_full = jnp.linspace(1.0, 1.0e9, 2000)
    roundtrip_inverse_forward(fwd, inv, ps_full, "DH p→ρ→p (full)")

    # Operating range: 1 MPa upwards (where lubricants actually live).
    ps_op = jnp.linspace(1.0e6, 1.0e9, 2000)
    roundtrip_inverse_forward(fwd, inv, ps_op, "DH p→ρ→p (≥1MPa)")

    jacobian_consistency(fwd, inv, rhos, "DH")
    gradient_direct_check(fwd, inv, ps_full, "DH")


def test_bayada_chupin():
    print("\n=== Bayada-Chupin ===")
    rho_l, rho_v, c_l, c_v = 850., 0.019, 1500., 400.

    fwd = jit(lambda rho: bayada_chupin(rho, rho_l, rho_v, c_l, c_v))
    inv = jit(lambda p:   rho_of_p_bayada(p,   rho_l, rho_v, c_l, c_v))

    # Three rho regions — sample densely in each.
    rhos_vap = jnp.linspace(1e-4,          0.98 * rho_v, 500)
    rhos_mix = jnp.linspace(1.02 * rho_v,  0.998 * rho_l, 1000)
    rhos_liq = jnp.linspace(1.002 * rho_l, 1.50 * rho_l, 500)
    rhos = jnp.concatenate([rhos_vap, rhos_mix, rhos_liq])

    roundtrip_forward_inverse(fwd, inv, rhos, "Bayada ρ→p→ρ (all)")

    # p region bounds (scalar values for sampling)
    N    = rho_v*c_v**2 * rho_l*c_l**2 * (rho_v - rho_l) / (rho_v**2*c_v**2 - rho_l**2*c_l**2)
    Pcav = float(rho_v*c_v**2 - N * jnp.log(rho_v**2*c_v**2 / (rho_l**2*c_l**2)))
    P_vt = float(rho_v * c_v**2)
    print(f"  (Pcav = {Pcav:.3e} Pa, P_vt = {P_vt:.3e} Pa)")

    ps_vap = jnp.linspace(1e-6,          0.98 * P_vt, 500)
    ps_mix = jnp.linspace(1.02 * P_vt,   0.998 * Pcav, 1000)
    ps_liq = jnp.linspace(1.002 * Pcav,  1.0e9,       500)
    ps_all = jnp.concatenate([ps_vap, ps_mix, ps_liq])

    roundtrip_inverse_forward(fwd, inv, ps_all, "Bayada p→ρ→p (all)")

    # Report per-region separately to find any exceptions
    roundtrip_inverse_forward(fwd, inv, ps_vap, "Bayada p→ρ→p (vap)")
    roundtrip_inverse_forward(fwd, inv, ps_mix, "Bayada p→ρ→p (mix)")
    roundtrip_inverse_forward(fwd, inv, ps_liq, "Bayada p→ρ→p (liq)")

    jacobian_consistency(fwd, inv, rhos, "Bayada (all)")
    jacobian_consistency(fwd, inv, rhos_vap, "Bayada (vap)")
    jacobian_consistency(fwd, inv, rhos_mix, "Bayada (mix)")
    jacobian_consistency(fwd, inv, rhos_liq, "Bayada (liq)")

    gradient_direct_check(fwd, inv, ps_all, "Bayada")

    # Exact boundary check: feeding p(rho_v) and p(rho_l) back should recover
    # the original rho to high accuracy.
    rho_boundary = jnp.array([rho_v, rho_l])
    p_boundary   = fwd(rho_boundary)
    rho_back     = inv(p_boundary)
    print("  boundary points:")
    print(f"    rho_v={float(rho_boundary[0]):.6e}  →  p={float(p_boundary[0]):.6e}  →  rho={float(rho_back[0]):.6e}")
    print(f"    rho_l={float(rho_boundary[1]):.6e}  →  p={float(p_boundary[1]):.6e}  →  rho={float(rho_back[1]):.6e}")


def test_dispatcher():
    """Exercise the eos_rho / eos_drho_dp dispatchers to confirm they route
    correctly and produce the same values as the direct-branch calls."""
    print("\n=== Dispatcher (eos_rho / eos_drho_dp) ===")

    prop_dh = {'EOS': 'DH', 'rho0': 877.7007, 'P0': 101325.,
               'C1': 3.5e8, 'C2': 1.23}
    prop_bc = {'EOS': 'Bayada', 'rho_l': 850., 'rho_v': 0.019,
               'c_l': 1500., 'c_v': 400.}

    ps_dh = jnp.linspace(1.0e6, 1.0e9, 200)
    rho_direct = rho_of_p_dh(ps_dh, prop_dh['rho0'], prop_dh['P0'],
                              prop_dh['C1'], prop_dh['C2'])
    rho_disp = eos_rho(ps_dh, prop_dh)
    err = jnp.max(jnp.abs(rho_direct - rho_disp) / jnp.abs(rho_direct))
    print(f"  DH eos_rho vs direct: max rel err = {float(err):.3e}")

    drho_dp_disp = eos_drho_dp(ps_dh, prop_dh)
    drho_dp_ref = vmap(grad(lambda p: rho_of_p_dh(p, prop_dh['rho0'],
                                                    prop_dh['P0'],
                                                    prop_dh['C1'],
                                                    prop_dh['C2'])))(ps_dh)
    err = jnp.max(jnp.abs(drho_dp_disp - drho_dp_ref)
                  / jnp.maximum(jnp.abs(drho_dp_ref), 1e-30))
    print(f"  DH eos_drho_dp vs grad: max rel err = {float(err):.3e}")

    ps_bc = jnp.linspace(1e-3, 1.0e9, 200)
    rho_direct = rho_of_p_bayada(ps_bc, prop_bc['rho_l'], prop_bc['rho_v'],
                                  prop_bc['c_l'], prop_bc['c_v'])
    rho_disp = eos_rho(ps_bc, prop_bc)
    err = jnp.max(jnp.abs(rho_direct - rho_disp)
                  / jnp.maximum(jnp.abs(rho_direct), 1e-30))
    print(f"  Bayada eos_rho vs direct: max rel err = {float(err):.3e}")

    drho_dp_disp = eos_drho_dp(ps_bc, prop_bc)
    drho_dp_ref = vmap(grad(lambda p: rho_of_p_bayada(
        p, prop_bc['rho_l'], prop_bc['rho_v'],
        prop_bc['c_l'], prop_bc['c_v'])))(ps_bc)
    err = jnp.max(jnp.abs(drho_dp_disp - drho_dp_ref)
                  / jnp.maximum(jnp.abs(drho_dp_ref), 1e-30))
    print(f"  Bayada eos_drho_dp vs grad: max rel err = {float(err):.3e}")


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    test_dowson_higginson()
    test_bayada_chupin()
    test_dispatcher()
