"""
solver_fem integration tests.

Solver comparison
-----------------
Compares solver_fem steady-state solutions against pre-saved explicit reference data
(tests/data/explicit_<case>.npz).  Explicit runs are expensive (~minutes each)
so their results are committed once and re-used by CI.

To regenerate the reference data run:
    python tests/generate_explicit_reference.py

Cases:
  inclined_slider  - PL EOS, air, Dirichlet density BC, linear gap
  journal_bearing  - DH EOS, oil, periodic BC, cosine gap
  parabolic_slider - DH EOS, oil, Dirichlet density BC, parabolic gap

Analytic validation
-------------------
  Poiseuille  - body force, periodic x, no-slip y walls
  Couette     - wall velocity, periodic x, no-slip y walls
  Bernoulli   - venturi nozzle, inertia terms, quasi-1D
"""

import numpy as np
import pytest
from pathlib import Path

from GaPFlow.problem import Problem
from GaPFlow.models.pressure import eos_pressure

CONFIG_DIR = Path(__file__).parent / "configs"
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Helpers
# =============================================================================

def load_template(name: str) -> str:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return f.read()


def compute_l2_error(numerical, analytical):
    return np.sqrt(((numerical - analytical) ** 2).mean()) / analytical.max()


def compare_solutions(sol1, sol2, bc_type="periodic", rtol=0.05, field_name="field"):
    """Point-wise comparison; Dirichlet cases use inner-region clipping."""
    if bc_type == "periodic":
        np.testing.assert_allclose(sol1, sol2, rtol=rtol,
                                   err_msg=f"{field_name}: point-wise mismatch")
    else:
        N = len(sol1)
        inner = slice(int(0.05 * N), int(0.95 * N))
        np.testing.assert_allclose(
            sol1[inner], sol2[inner], rtol=rtol,
            err_msg=f"{field_name}: inner region (5%-95%) mismatch")
        int1, int2 = np.trapezoid(sol1), np.trapezoid(sol2)
        np.testing.assert_allclose(
            int1, int2, rtol=0.02,
            err_msg=f"{field_name}: integral mismatch ({int1:.4f} vs {int2:.4f})")
        np.testing.assert_allclose(
            sol1[inner].max(), sol2[inner].max(), rtol=rtol,
            err_msg=f"{field_name}: inner-region peak mismatch")


# =============================================================================
# Solver comparison: solver_fem vs saved explicit reference
# =============================================================================

FEM2D_DEFAULTS = {
    "Ny": 4,
    "solver": "fem",
    "dt": 1e-2,
    "max_it": 100,
    "adaptive": 0,
    "CFL": 0.5,
    "pressure_stab_alpha": 0,
    "momentum_stab_alpha": 0,
}

COMPARISON_CASES = {
    "inclined_slider": {
        "template": "inclined_slider",
        "bc_type": "dirichlet",
        "rho_bc": 1.0,
    },
    "journal_bearing": {
        "template": "journal_bearing",
        "bc_type": "periodic",
        "rho_bc": 1.0,
    },
    "parabolic_slider": {
        "template": "parabolic_slider",
        "bc_type": "dirichlet",
        "rho_bc": 850.0,
    },
}


def _run_fem2d(case_name: str):
    case = COMPARISON_CASES[case_name]
    template = load_template(case["template"])
    params = {**FEM2D_DEFAULTS, "rho_bc": case["rho_bc"]}
    problem = Problem.from_string(template.format(**params))
    problem.run()
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    return rho, jx


def _load_explicit_reference(case_name: str):
    path = DATA_DIR / f"explicit_{case_name}.npz"
    if not path.exists():
        pytest.skip(f"Reference data not found: {path}. Run generate_explicit_reference.py.")
    data = np.load(path)
    return data["rho"], data["jx"]


@pytest.fixture(scope="module", params=list(COMPARISON_CASES.keys()))
def comparison(request):
    case_name = request.param
    rho_ref, jx_ref = _load_explicit_reference(case_name)
    rho_fem, jx_fem = _run_fem2d(case_name)
    bc_type = COMPARISON_CASES[case_name]["bc_type"]
    return dict(rho_ref=rho_ref, jx_ref=jx_ref, rho_fem=rho_fem, jx_fem=jx_fem,
                bc_type=bc_type, case=case_name)


def test_fem2d_vs_explicit_rho(comparison):
    r = comparison
    compare_solutions(r["rho_ref"], r["rho_fem"], bc_type=r["bc_type"],
                      field_name=f"rho ({r['case']})")


def test_fem2d_vs_explicit_jx(comparison):
    r = comparison
    compare_solutions(r["jx_ref"], r["jx_fem"], bc_type=r["bc_type"],
                      field_name=f"jx ({r['case']})")



# =============================================================================
# Analytic: 2D Poiseuille flow
# =============================================================================

@pytest.fixture(scope="module")
def poiseuille_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "poiseuille_2d_body_force.yaml"))
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_poiseuille_velocity_profile(poiseuille_problem):
    """In-plane viscous diffusion (R23xy, R23yx) driven by body force.

    jx/jy Dirichlet BCs use the mirror formula: effective wall at the cell face
    (y=0, y=Ly), not at the ghost cell center.
    """
    p = poiseuille_problem
    rho = p.q[0][1:-1, 1:-1]
    jx = p.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = p.grid["Ly"], p.grid["Ny"]
    dy = Ly / Ny
    mu = p.prop["shear"]
    f_x = p.prop["force_x"]
    rho_m = rho.mean()

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_ana = (rho_m * f_x) / (2 * mu) * y * (Ly - y)

    l2_err = compute_l2_error(vx, u_ana)
    assert l2_err < 0.01, f"Poiseuille L2 error = {l2_err:.2e} > 0.01"


# =============================================================================
# Analytic: 2D Couette flow
# =============================================================================

@pytest.fixture(scope="module")
def couette_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "couette_2d_wall_velocity.yaml"))
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_couette_velocity_profile(couette_problem):
    """In-plane viscous diffusion (R23xy, R23yx) driven by wall motion.

    jx/jy Dirichlet BCs use the mirror formula: effective wall at the cell face
    (y=0, y=Ly), not at the ghost cell center.
    """
    p = couette_problem
    rho = p.q[0][1:-1, 1:-1]
    jx = p.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = p.grid["Ly"], p.grid["Ny"]
    dy = Ly / Ny
    rho_m = rho.mean()
    U_wall = p.grid["bc_yN_D_val"][1] / rho_m

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_ana = U_wall * y / Ly

    l2_err = compute_l2_error(vx, u_ana)
    assert l2_err < 0.01, f"Couette L2 error = {l2_err:.2e} > 0.01"


# =============================================================================
# Analytic: Bernoulli venturi nozzle
# =============================================================================

_H_INLET = 1.0e-3
_H_THROAT = 0.5e-3
_X_RAMP_START = 0.03
_X_RAMP_END = 0.07
_THROAT_HALF_WIDTH = 0.005
_RHO0 = 1000.0
_V_INLET = 5.0


def _venturi_topography(xx):
    x_mid = (_X_RAMP_START + _X_RAMP_END) / 2
    ramp_len = x_mid - _THROAT_HALF_WIDTH - _X_RAMP_START
    dist = np.clip(np.abs(xx - x_mid) - _THROAT_HALF_WIDTH, 0, ramp_len)
    xi = dist / ramp_len
    return np.where(
        (xx >= _X_RAMP_START) & (xx <= _X_RAMP_END),
        (_H_INLET + _H_THROAT) / 2 - (_H_INLET - _H_THROAT) / 2 * np.cos(np.pi * xi),
        _H_INLET,
    )


@pytest.fixture(scope="module")
def bernoulli_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "bernoulli_venturi.yaml"))
    problem.topo.set_mapped_height(_venturi_topography(problem.topo.xx))
    problem.q[1][:] = _RHO0 * _V_INLET
    problem.run()
    return problem


def test_bernoulli_velocity_ratio(bernoulli_problem):
    """Velocity doubles at throat (mass conservation, h_inlet / h_throat = 2)."""
    p = bernoulli_problem
    rho_b = p.q[0][1:-1, 1:-1]
    jx_b = p.q[1][1:-1, 1:-1]
    dx = p.grid["dx"]
    x_b = np.arange(rho_b.shape[0]) * dx + dx / 2
    j_center = rho_b.shape[1] // 2
    vx_b = jx_b[:, j_center] / rho_b[:, j_center]

    inlet = x_b < _X_RAMP_START
    throat = (x_b >= 0.045) & (x_b <= 0.055)
    v_ratio = vx_b[throat].mean() / vx_b[inlet].mean()

    assert abs(v_ratio - 2.0) < 0.01, \
        f"Bernoulli velocity ratio = {v_ratio:.4f}, expected 2.0"


def test_bernoulli_pressure_drop(bernoulli_problem):
    """Pressure drop at throat matches Bernoulli prediction within 1%."""
    p = bernoulli_problem
    rho_b = p.q[0][1:-1, 1:-1]
    jx_b = p.q[1][1:-1, 1:-1]
    dx = p.grid["dx"]
    x_b = np.arange(rho_b.shape[0]) * dx + dx / 2
    j_center = rho_b.shape[1] // 2
    p_b = np.asarray(eos_pressure(rho_b[:, j_center], p.prop))

    inlet = x_b < _X_RAMP_START
    throat = (x_b >= 0.045) & (x_b <= 0.055)
    dp_sim = p_b[inlet].mean() - p_b[throat].mean()

    v_throat_theory = _V_INLET * _H_INLET / _H_THROAT
    dp_theory = 0.5 * _RHO0 * (v_throat_theory ** 2 - _V_INLET ** 2)

    rel_err = abs(dp_sim - dp_theory) / dp_theory
    assert rel_err < 0.01, \
        f"Bernoulli Δp error = {rel_err*100:.2f}%, sim={dp_sim:.1f} Pa, theory={dp_theory:.1f} Pa"
