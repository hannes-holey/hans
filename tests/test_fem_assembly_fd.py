#
# Copyright 2026 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# flake8: noqa: W503
"""
Finite difference verification of assemble_matrix against assemble_rhs.

Tests that the analytic Jacobian (assemble_matrix) matches the central-FD
Jacobian of the residual (assemble_rhs) for all terms with exact derivatives.

Excluded terms:
  - R24x/y, R24x/y_fb  : τ_xz/yz frozen in Jacobian; FD ~1e-3 error expected
  - Energy terms        : material properties (k, cp) frozen in der_funs
  - OSS terms           : intentionally approximate Jacobian for stability

Two fixtures:
  solver_noncav  : Bayada EOS, no cavitation, 3×2 grid, periodic_y BCs.
                   p ramps through P_cav to exercise the dp/dρ correction terms.
  solver_cav     : DH EOS + p_cav=0, cavitation=True, 4×3 grid, periodic_y BCs.
                   p/theta set to cover all three FB regimes simultaneously.
                   jx/jy vary spatially so (jx,theta) pairs differ within stencils.
                   4×3 ensures no inner node sits at the domain center so both
                   dh_dx and dh_dy are nonzero everywhere (activates *S terms).

Why DH for cavitation fixture: Bayada EOS encodes cavitation in the EOS itself;
the Fischer-Burmeister approach is designed for smooth EOS models.

Why periodic_y BCs: Dirichlet in x exercises BC-adjacent assembly (ghost rows at
p_inlet/p_outlet); periodic in y gives clean cross-derivative tests (R22yx etc.).
"""
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_fem.solver_fem import FEMSolver
from GaPFlow.solver_fem.terms import (
    R11x, R11y, R11x_corr, R11y_corr,
    R21x, R21y, R2Tx, R2Ty,
    R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx, R23xx, R23yy,

)
from GaPFlow.solver_fem.terms_theta import (
    R11x_fb, R11y_fb, R11Sx_fb, R11Sy_fb,
    R11x_fb_corr, R11y_fb_corr, R11Sx_fb_corr, R11Sy_fb_corr,
    R_cav,
)
R_FB = R_cav  # R_cav is a single Term instance

# =============================================================================
# YAML config templates
# =============================================================================

_BASE = """
options:
    output: /tmp/fem_{label}
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {Nx}
    Ny: {Ny}
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 877.7007
    xW_D: 877.7007
    yS_D: 877.7007
    yN_D: 877.7007

geometry:
    type: parabolic_2d
    hmax: 2e-5
    hmin: 1e-5
    U: 1.0
    V: 2.0

numerics:
    solver: fem
    dt: 1e-8
    tol: 1e-6
    max_it: 100

fem_solver:
    type: newton_alpha
    equations:
        energy: False
        cavitation: {cavitation}
    physics:
        gap_shear: False
        plane_shear: {plane_shear}
        inertia: {inertia}
        body_force: {body_force}
"""

_PROPS_BAYADA = """
properties:
    EOS: Bayada
    rho0: 850.0
    rho_l: 850.0
    rho_v: 0.019
    c_l: 1600.0
    c_v: 352.0
    shear: 0.039
    bulk: 0.0
"""


_PROPS_DH_CAV = """
properties:
    EOS: DH
    rho0: 877.7007
    P0: 101325
    C1: 3.5e10
    C2: 1.23
    shear: 0.0794
    bulk: 0.0
    p_cav: 0.0
"""

# =============================================================================
# Term groups — each list of Term instances is tested as one Jacobian check.
# Correction terms (R11x_corr etc.) MUST be grouped with their base term:
# FD sees the implicit p→ρ→dp/dρ chain; the corr term encodes that dependency.
# =============================================================================

# Each entry is (group, tol). Linear/constant-coefficient terms use 1e-6;
# nonlinear terms (EOS-dependent, FB) use 1e-4.
NON_CAV_TERM_GROUPS = [
    ([R11x, R11x_corr], 1e-4),
    ([R11y, R11y_corr], 1e-4),
    # R11Sx, R11Sy: need correction terms (R11Sx_corr, R11Sy_corr) — not yet implemented;
    # without them the Bayada dp_drho(p) chain causes a large FD mismatch
    ([R21x], 1e-6),
    ([R21y], 1e-6),
    ([R2Tx], 1e-6),
    ([R2Ty], 1e-6),
]

NON_CAV_INERTIA_GROUPS = [
    ([R22xx], 1e-4),
    ([R22xxS], 1e-4),
    ([R22yx], 1e-4),
    ([R22yxS], 1e-4),
    ([R22xy], 1e-4),
    ([R22xyS], 1e-4),
    ([R22yy], 1e-4),
    ([R22yyS], 1e-4),
]

NON_CAV_VISCOUS_GROUPS = [
    ([R23xy], 1e-4),
    ([R23yx], 1e-4),
    ([R23xx], 1e-4),
    ([R23yy], 1e-4),
]


CAV_TERM_GROUPS = [
    ([R11x_fb, R11x_fb_corr], 1e-4),
    ([R11y_fb, R11y_fb_corr], 1e-4),
    ([R11Sx_fb, R11Sx_fb_corr], 1e-4),
    ([R11Sy_fb, R11Sy_fb_corr], 1e-4),
    ([R_FB], 1e-4),
    ([R21x], 1e-6),
    ([R21y], 1e-6),
    ([R2Tx], 1e-6),
    ([R2Ty], 1e-6),
]

# =============================================================================
# Helpers
# =============================================================================


def _make_problem(props: str, Nx: int, Ny: int, cavitation: bool = False,
                  plane_shear: bool = False, inertia: bool = False,
                  body_force: bool = False, label: str = 'test') -> tuple:
    config = _BASE.format(
        label=label, Nx=Nx, Ny=Ny,
        cavitation=str(cavitation).lower(),
        plane_shear=str(plane_shear).lower(),
        inertia=str(inertia).lower(),
        body_force=str(body_force).lower(),
    ) + props
    problem = Problem.from_string(config)
    problem._pre_run()
    solver = problem.solver

    return problem, solver


def _pcav_bayada(prop: dict) -> float:
    rho_l, rho_v = prop['rho_l'], prop['rho_v']
    c_l, c_v = prop['c_l'], prop['c_v']
    N = (rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
         / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2))
    return rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))


def _set_and_sync(solver: FEMSolver, q: np.ndarray) -> None:
    solver.update_q_nodal(q)
    solver.update_quad()


def _init_noncav(solver: FEMSolver) -> None:
    """Set p to a ramp straddling P_cav so the dp/dρ correction terms are nonzero."""
    Pcav = _pcav_bayada(solver.problem.prop)
    delta = 0.05 * abs(Pcav)
    q = solver.get_q_nodal().copy()
    p_sl = solver._sol_slices['p']
    n_p = p_sl.stop - p_sl.start
    q[p_sl] = np.linspace(Pcav - delta, Pcav + delta, n_p)
    _set_and_sync(solver, q)


def _init_cav(solver: FEMSolver) -> None:
    """Set p/theta to cover all three FB regimes.

    p: ramp from -delta to +delta around p_cav=0.
    theta: complementary ramp — a<0: θ proportional to |a|; a≥0: θ=eps.
    jx/jy: ±30% ramp on P2 nodes so (jx, theta) vary within each assembly stencil.

    Regime coverage:
      a > 0, theta ~ eps  — full film
      a < 0, theta > 0    — cavitation
      a ~ 0, theta ~ 0    — transition (exercises _fb_denom singularity guard)
    """
    p_cav = float(solver.problem.prop['p_cav'])
    delta = 1e4
    q = solver.get_q_nodal().copy()

    p_sl = solver._sol_slices['p']
    n_p = p_sl.stop - p_sl.start
    p_vals = np.linspace(p_cav - delta, p_cav + delta, n_p)
    q[p_sl] = p_vals

    theta_sl = solver._sol_slices['theta']
    n_th = theta_sl.stop - theta_sl.start
    a = np.linspace(-delta, delta, n_th)
    theta_vals = np.where(a < 0, -a / delta * 0.5, np.finfo(float).eps)
    q[theta_sl] = theta_vals

    jx_sl = solver._sol_slices['jx']
    n_jx = jx_sl.stop - jx_sl.start
    jx_base = q[jx_sl].copy()
    q[jx_sl] = jx_base * (1.0 + 0.3 * np.linspace(-1, 1, n_jx))

    jy_sl = solver._sol_slices['jy']
    n_jy = jy_sl.stop - jy_sl.start
    jy_base = q[jy_sl].copy()
    q[jy_sl] = jy_base * (1.0 + 0.3 * np.linspace(-1, 1, n_jy))

    _set_and_sync(solver, q)


def compute_fd_jacobian(solver: FEMSolver, eps: float = 1e-6) -> np.ndarray:
    """Central finite difference Jacobian of get_R_ w.r.t. nodal DOFs.

    Step size: eps_j = eps * max(1, |q0[j]|)  (relative perturbation).
    Restores original state before returning.
    """
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J_fd = np.zeros((n, n))

    for j in range(n):
        eps_j = eps * max(1.0, abs(q0[j]))

        q_plus = q0.copy()
        q_plus[j] += eps_j
        _set_and_sync(solver, q_plus)
        R_plus = solver.get_R_().copy()

        q_minus = q0.copy()
        q_minus[j] -= eps_j
        _set_and_sync(solver, q_minus)
        R_minus = solver.get_R_().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2.0 * eps_j)

    _set_and_sync(solver, q0)
    return J_fd


def rel_err(M_anal: np.ndarray, M_fd: np.ndarray) -> float:
    return np.linalg.norm(M_anal - M_fd) / (np.linalg.norm(M_fd) + 1e-15)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def solver_noncav():
    _, solver = _make_problem(_PROPS_BAYADA, Nx=3, Ny=2, label='noncav')
    _init_noncav(solver)
    return solver


@pytest.fixture(scope='module')
def solver_inertia():
    _, solver = _make_problem(_PROPS_BAYADA, Nx=3, Ny=2, inertia=True, label='inertia')
    _init_noncav(solver)
    return solver


@pytest.fixture(scope='module')
def solver_viscous():
    _, solver = _make_problem(_PROPS_BAYADA, Nx=3, Ny=2, plane_shear=True, label='viscous')
    _init_noncav(solver)
    return solver


@pytest.fixture(scope='module')
def solver_cav():
    _, solver = _make_problem(_PROPS_DH_CAV, Nx=4, Ny=3, cavitation=True, label='cav')
    _init_cav(solver)
    return solver


# =============================================================================
# Tests
# =============================================================================

def _check_group(solver: FEMSolver, group: list, tol: float) -> None:
    all_terms = solver.assembly.assembly_terms
    try:
        solver.assembly.assembly_terms = group
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver)
        names = '+'.join(t.name for t in group)
        assert np.linalg.norm(J) > 1e-10, f"{names}: FD Jacobian is essentially zero"
        err = rel_err(M, J)
        assert err < tol, f"{names}: rel_err={err:.2e} >= {tol:.0e}"
    finally:
        solver.assembly.assembly_terms = all_terms


def _ids(groups):
    return ['+'.join(t.name for t in g) for g, _ in groups]


@pytest.mark.parametrize('group,tol', NON_CAV_TERM_GROUPS, ids=_ids(NON_CAV_TERM_GROUPS))
def test_jacobian_noncav(solver_noncav, group, tol):
    _check_group(solver_noncav, group, tol)


@pytest.mark.parametrize('group,tol', NON_CAV_INERTIA_GROUPS, ids=_ids(NON_CAV_INERTIA_GROUPS))
def test_jacobian_inertia(solver_inertia, group, tol):
    _check_group(solver_inertia, group, tol)


@pytest.mark.parametrize('group,tol', NON_CAV_VISCOUS_GROUPS, ids=_ids(NON_CAV_VISCOUS_GROUPS))
def test_jacobian_viscous(solver_viscous, group, tol):
    _check_group(solver_viscous, group, tol)


@pytest.mark.parametrize('group,tol', CAV_TERM_GROUPS, ids=_ids(CAV_TERM_GROUPS))
def test_jacobian_cav(solver_cav, group, tol):
    _check_group(solver_cav, group, tol)


# =============================================================================
# Dev utilities (not run by pytest)
# =============================================================================

def check_all_terms(solver: FEMSolver, groups: list, eps: float = 1e-6) -> dict:
    """Check each term group, return dict of group_label -> rel_err."""
    all_terms = solver.assembly.assembly_terms
    results = {}
    for group in groups:
        label = '+'.join(t.name for t in group)
        solver.assembly.assembly_terms = group
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver, eps=eps)
        results[label] = rel_err(M, J)
    solver.assembly.assembly_terms = all_terms
    return results


def compare_matrices(solver: FEMSolver, group: list, block: str,
                     eps: float = 1e-6, fmt: str = '11.2e') -> None:
    """Print analytic and FD Jacobian side-by-side for one group, one dep-var block."""
    all_terms = solver.assembly.assembly_terms
    try:
        solver.assembly.assembly_terms = group
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver, eps=eps)
    finally:
        solver.assembly.assembly_terms = all_terms

    col_sl = solver._sol_slices.get(block)
    if col_sl is None:
        raise ValueError(f"block={block!r} not in {list(solver._sol_slices)}")

    res_names = sorted({t.res for t in group}, key=lambda r: solver._res_slices[r].start)
    row_indices = []
    for res in res_names:
        rs = solver._res_slices[res]
        row_indices.extend(range(rs.start, rs.stop))
    col_indices = list(range(col_sl.start, col_sl.stop))

    M_blk = M[np.ix_(row_indices, col_indices)]
    J_blk = J[np.ix_(row_indices, col_indices)]
    diff = M_blk - J_blk

    label = '+'.join(t.name for t in group)
    rel = np.linalg.norm(diff) / (np.linalg.norm(J_blk) + 1e-15)
    print(f"\nterm={label}  block=d/d{block}  eps={eps:.0e}  rel_err={rel:.3e}")

    entry_w = len(f'{0:{fmt}}')
    gap = '  '
    col_w = entry_w + len(gap)
    row_label_w = max(4, len(str(max(row_indices))))
    col_hdr = ' ' * (row_label_w + 2) + ''.join(f'{c:>{col_w}d}' for c in col_indices)
    sep_line = '-' * len(col_hdr)

    def _row_str(mat, r):
        return ''.join(
            gap + (' ' * entry_w if mat[r, c] == 0.0 else f'{mat[r, c]:{fmt}}')
            for c in range(mat.shape[1])
        )

    for title, mat in [('analytic (M)', M_blk), ('FD (J)', J_blk), ('diff (M-J)', diff)]:
        print(f"\n  {title}")
        print(col_hdr)
        print(sep_line)
        for r, global_r in enumerate(row_indices):
            print(f"  {global_r:<{row_label_w}}{_row_str(mat, r)}")
