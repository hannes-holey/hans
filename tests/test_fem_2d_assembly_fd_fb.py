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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Finite difference Jacobian verification for the Fischer-Burmeister cavitation terms.

EOS: Dowson-Higginson (DH).  The FB approach is designed for smooth EOS models
where cavitation is not baked into the EOS — the FB condition enforces p >= p_cav
as a complementarity constraint.  The Bayada EOS already encodes cavitation in the
EOS itself and is therefore not the right pairing.

Terms under test:
  R11x_fb, R11y_fb   — flux divergence x/y (Elrod-Adams IBP form)
  R11Sx_fb, R11Sy_fb — flux divergence height source x/y
  R_FB               — Fischer-Burmeister complementarity condition

State design — all combinations of relevant interactions must be nonzero:
  (a) jx != 0, jy != 0, spatially non-uniform
                             — nonzero fluxes so d/dtheta of R11*_fb is nonzero;
                               spatial variation ensures different (jx, theta)
                               combinations appear in the same assembly stencil
  (b) theta in (0, 1)        — (1-theta) factor keeps d/djx, d/djy nonzero;
                               theta>0 makes d/dtheta of R11*_fb nonzero
  (c) p straddling p_cav     — exercises all three FB regimes simultaneously:
                                 a>0, theta~eps  (full film)
                                 a<0, theta>0    (cavitation)
                                 a~0, theta~0    (transition, _fb_denom guard)
  (d) dh_dx != 0, dh_dy != 0 — parabolic_2d paraboloid geometry; with a 4x3 grid
                                no inner node sits at the domain center so both
                                gradients are nonzero everywhere, activating
                                R11Sx_fb and R11Sy_fb
  (e) dp_drho nonzero, mildly varying — DH EOS; dp_drho varies slowly with p so
                                the d(dp_drho)/dp correction is negligible.
                                NOTE: if a future run shows a systematic ~1e-3
                                relative error on the d/dp block of R11x_fb, it
                                is likely from the dropped correction term, not a
                                bug in the Jacobian implementation.

The test grid is 4x3 (inner) with periodic_y BCs.  The x-direction has Dirichlet
pressure BCs so boundary ghost rows have theta=0 (physical boundary condition),
which means those nodes are in the full-film regime — this exercises the case
where theta_ghost=0 coexists with interior theta>0 in the same assembly stencil.
"""
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2d

# =============================================================================
# YAML config template
# =============================================================================

_COMMON_GRID = """
options:
    output: /tmp/fem2d_fd_test_fb_{label}
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {Nx}
    Ny: {Ny}
    xE: {xE}
    xW: {xW}
    yS: {yS}
    yN: {yN}
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
        cavitation: True
        {term_list_entry}
"""

_PROPS_DH_CAV = """
properties:
    EOS: DH
    rho0: 877.7007
    P0: 101325
    C1: 3.5e10
    C2: 1.23
    p_cav: 101325
    shear: 0.0794
    bulk: 0.0
"""

# =============================================================================
# Boundary condition presets
# =============================================================================

BC_CONFIGS = {
    'periodic_y': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
    },
}

# =============================================================================
# Helpers
# =============================================================================

_PCAV = 101325.0  # p_cav = P0 for DH EOS


def make_problem(Nx: int, Ny: int, bc: str = 'periodic_y',
                 term_list: list = None) -> tuple:
    """Create and pre-run a (problem, solver) pair with cavitation active."""
    bc_cfg = BC_CONFIGS[bc]
    term_list_entry = f"term_list: {term_list}" if term_list is not None else ""

    config = (_COMMON_GRID + _PROPS_DH_CAV).format(
        label='fb',
        Nx=Nx, Ny=Ny,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
        term_list_entry=term_list_entry,
    )

    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()

    _init_fb_straddling(solver)
    return problem, solver


def _init_fb_straddling(solver: FEMSolver2d) -> None:
    """Set initial (p, theta, jx, jy) state covering all relevant interaction regimes.

    State design:
      p  : linear ramp from (Pcav - delta) to (Pcav + delta) across inner P1 nodes,
           so half the domain has a < 0 (cavitation) and half has a > 0 (full film).
           The midpoint node has a ~ 0 (transition, exercises _fb_denom guard).

      theta : complementary ramp — where a < 0: theta = -a/delta * 0.5 (positive,
              up to 0.5); where a >= 0: theta = machine eps.
              This gives nonzero theta exactly where p < p_cav, so:
                - d/dtheta of R11*_fb is nonzero (requires theta > 0)
                - d/djx, d/djy of R11*_fb are nonzero ((1-theta) in (0.5, 1))
                - d/dp and d/dtheta of R_FB both nonzero in all three regimes

      jx, jy : problem-initialised base value (rho0 * U/2.1 and rho0 * V/2.1,
               nonzero since U=1.0, V=2.0) plus a ±30% linear ramp across the
               P2 inner nodes. The ramp index is the flat Newton ordering
               (y-major due to Fortran order), which doesn't align with any
               particular physical direction — that is intentional: it ensures
               different (jx, theta) value combinations appear in the same
               assembly stencil without requiring a specific spatial pattern.
    """
    delta = 0.05 * abs(_PCAV)
    q = solver.get_q_nodal().copy()

    p_sl = solver._sol_slices['p']
    theta_sl = solver._sol_slices['theta']
    jx_sl = solver._sol_slices['jx']
    jy_sl = solver._sol_slices['jy']
    n_p = p_sl.stop - p_sl.start

    p_ramp = np.linspace(_PCAV - delta, _PCAV + delta, n_p)
    q[p_sl] = p_ramp

    a = p_ramp - _PCAV
    q[theta_sl] = np.where(a < 0, -a / delta * 0.5, np.finfo(float).eps)

    for sl in (jx_sl, jy_sl):
        n = sl.stop - sl.start
        base = q[sl].mean()
        q[sl] = base * (1.0 + 0.3 * np.linspace(-1.0, 1.0, n))

    _set_and_sync(solver, q)


def _set_and_sync(solver: FEMSolver2d, q: np.ndarray) -> None:
    """Mirror the solver's own set→sync_rho→exchange→update_quad sequence."""
    solver.set_q_nodal(q)
    solver._sync_rho_before_exchange()
    solver.exchange_ghosts()
    solver.update_quad()


def build_scale_array(solver: FEMSolver2d, scales: dict = None) -> np.ndarray:
    """Build a per-DOF scale array from per-variable characteristic scales."""
    if scales is None:
        scales = {}
    q0 = solver.get_q_nodal()
    scale = np.ones(len(q0))
    for var, sl in solver._sol_slices.items():
        scale[sl] = scales.get(var, 1.0)
    return scale


def compute_fd_jacobian(solver: FEMSolver2d, eps: float = 1e-6,
                        scale: np.ndarray = None) -> np.ndarray:
    """Central finite difference Jacobian of get_R_ w.r.t. all nodal DOFs."""
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J_fd = np.zeros((n, n))

    for j in range(n):
        eps_j = eps * scale[j] if scale is not None else eps * max(1.0, abs(q0[j]))

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
# Term groups
# =============================================================================

# Each group is tested as a unit; a group may contain multiple terms when their
# Jacobians cannot be separated (but for FB terms all are self-contained).
_TERM_GROUPS = [
    ['R11x_fb'],
    ['R11y_fb'],
    ['R11Sx_fb'],
    ['R11Sy_fb'],
    ['R11x_fb_corr'],
    ['R11y_fb_corr'],
    ['R11Sx_fb_corr'],
    ['R11Sy_fb_corr'],
    ['R_FB'],
    ['R1T'],
    ['R21x'],
    ['R21y'],
    ['R2Tx'],
    ['R2Ty'],
    ['R24x'],
    ['R24y'],
]


# =============================================================================
# check_all_terms / print / eps_sensitivity
# =============================================================================

def check_all_terms(
    Nx: int = 4,
    Ny: int = 3,
    bc: str = 'periodic_y',
    eps: float = 1e-6,
    scale: np.ndarray = None,
) -> dict:
    """Check each FB term group against its FD Jacobian.

    Returns nested dict: results[group_label][dep_var] = rel_err,
    plus results[group_label]['total'].
    """
    _, solver = make_problem(Nx, Ny, bc=bc)
    all_terms = solver.terms
    term_by_name = {t.name: t for t in all_terms}
    sol_slices = solver._sol_slices

    results = {}
    for group in _TERM_GROUPS:
        members = [term_by_name[n] for n in group if n in term_by_name]
        if not members:
            continue
        label = '+'.join(t.name for t in members)
        print(f"  eps={eps:.0e}  {label}", flush=True)
        solver.terms = members
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver, eps=eps, scale=scale)
        diff = M - J

        block_errs = {}
        for var, sl in sol_slices.items():
            J_col = J[:, sl]
            d_col = diff[:, sl]
            J_n = np.linalg.norm(J_col)
            d_n = np.linalg.norm(d_col)
            block_errs[var] = 0.0 if (J_n == 0.0 and d_n == 0.0) else d_n / (J_n + 1e-15)
        entry = dict(block_errs)
        entry['total'] = max(block_errs.values())
        results[label] = entry

    solver.terms = all_terms
    return results


def print_term_check(results: dict, solver: FEMSolver2d = None,
                     dep_vars: list = None) -> None:
    """Pretty-print the output of check_all_terms."""
    if dep_vars is None:
        dep_vars = ['jx', 'jy', 'p', 'theta']
    header = f"{'term':<30s}  {'total':>8s}" + ''.join(f"  {'d/d'+v:>10s}" for v in dep_vars)
    print(header)
    print('-' * len(header))
    for label, entry in results.items():
        total = entry['total']
        flag = '' if total < 1e-4 else ' ✗'
        row = f"{label:<30s}  {total:>8.2e}"
        for v in dep_vars:
            row += f"  {entry.get(v, float('nan')):>10.2e}"
        print(row + flag)


def eps_sensitivity(
    Nx: int = 4,
    Ny: int = 3,
    bc: str = 'periodic_y',
    eps_list: list = None,
    var_scales: dict = None,
) -> None:
    """Run check_all_terms for each eps in eps_list."""
    if eps_list is None:
        eps_list = [1e-4, 1e-6, 1e-8]

    _, solver = make_problem(Nx, Ny, bc=bc)
    dep_vars = list(solver._sol_slices.keys())
    scale = build_scale_array(solver, var_scales) if var_scales is not None else None

    col_w = 10
    header = (f"  {'term':<28s}  {'total':>{col_w}s}"
              + ''.join(f"  {'d/d'+v:>{col_w}s}" for v in dep_vars))
    sep = '  ' + '-' * (len(header) - 2)

    print(f"\ngrid={Nx}x{Ny}  bc={bc}"
          + (f"  scales={var_scales}" if var_scales else "  scales=relative"))

    for eps in eps_list:
        r = check_all_terms(Nx, Ny, bc=bc, eps=eps, scale=scale)
        print(f"\n  eps = {eps:.0e}")
        print(header)
        print(sep)
        for label, entry in r.items():
            total = entry['total']
            flag = ' ✗' if total > 1e-4 else ''
            row = (f"  {label:<28s}  {total:>{col_w}.2e}"
                   + ''.join(f"  {entry.get(v, float('nan')):>{col_w}.2e}"
                              for v in dep_vars))
            print(row + flag)


# =============================================================================
# Diagnostic: element-wise matrix comparison for a single term/block/eps
# =============================================================================

def compare_matrices(
    term: str,
    block: str,
    eps: float = 1e-6,
    Nx: int = 4,
    Ny: int = 3,
    bc: str = 'periodic_y',
    var_scales: dict = None,
    fmt: str = '11.2e',
) -> None:
    """Print analytic and FD Jacobian side-by-side for one FB term, one dep-var block."""
    _, solver = make_problem(Nx, Ny, bc=bc)
    scale = build_scale_array(solver, var_scales) if var_scales is not None else None

    term_by_name = {t.name: t for t in solver.terms}
    names = [n.strip() for n in term.split('+')]
    members = [term_by_name[n] for n in names if n in term_by_name]
    missing = [n for n in names if n not in term_by_name]
    if missing:
        raise ValueError(f"Terms not found: {missing}. Available: {sorted(term_by_name)}")

    solver.terms = members
    M = solver.get_M_dense()
    J = compute_fd_jacobian(solver, eps=eps, scale=scale)

    col_sl = solver._sol_slices.get(block)
    if col_sl is None:
        raise ValueError(f"block={block!r} not in {list(solver._sol_slices)}")

    res_names = sorted({t.res for t in members}, key=lambda r: solver._res_slices[r].start)
    row_indices = []
    for res in res_names:
        rs = solver._res_slices[res]
        row_indices.extend(range(rs.start, rs.stop))
    col_indices = list(range(col_sl.start, col_sl.stop))

    M_blk = M[np.ix_(row_indices, col_indices)]
    J_blk = J[np.ix_(row_indices, col_indices)]
    diff = M_blk - J_blk

    n_rows, n_cols = M_blk.shape
    entry_w = len(f'{0:{fmt}}')
    gap = '  '
    col_w = entry_w + len(gap)
    row_label_w = max(4, len(str(max(row_indices))))

    print(f"\nterm={term}  block=d/d{block}  eps={eps:.0e}  grid={Nx}x{Ny}  bc={bc}")
    print(f"rows: {', '.join(f'{r}[{solver._res_slices[r].start}:{solver._res_slices[r].stop}]' for r in res_names)}  "
          f"cols: {block}[{col_sl.start}:{col_sl.stop}]")
    rel = np.linalg.norm(diff) / (np.linalg.norm(J_blk) + 1e-15)
    print(f"rel_err = {rel:.3e}")

    col_hdr = ' ' * (row_label_w + 2) + ''.join(f'{c:>{col_w}d}' for c in col_indices)
    sep_line = '-' * len(col_hdr)

    def _row_str(mat, r):
        return ''.join(
            gap + (' ' * entry_w if mat[r, c] == 0.0 else f'{mat[r, c]:{fmt}}')
            for c in range(n_cols)
        )

    for title, mat in [('analytic (M)', M_blk), ('FD (J)', J_blk), ('diff (M-J)', diff)]:
        print(f"\n  {title}")
        print(col_hdr)
        print(sep_line)
        for r, global_r in enumerate(row_indices):
            print(f"  {global_r:<{row_label_w}}{_row_str(mat, r)}")


# =============================================================================
# Diagnostic: state verification
# =============================================================================

def print_state_coverage(Nx: int = 4, Ny: int = 3, bc: str = 'periodic_y') -> None:
    """Print a summary of the (a, theta) state at each inner P1 node.

    Useful for verifying that the straddling initialiser covers all three regimes.
    Columns: node index, p, a = p-p_cav, theta, regime label.
    """
    _, solver = make_problem(Nx, Ny, bc=bc)
    q = solver.get_q_nodal()
    p_sl = solver._sol_slices['p']
    theta_sl = solver._sol_slices['theta']
    p_vals = q[p_sl]
    theta_vals = q[theta_sl]
    a_vals = p_vals - _PCAV

    print(f"\nState coverage  grid={Nx}x{Ny}  p_cav={_PCAV:.4e}")
    print(f"{'node':>5s}  {'p':>12s}  {'a=p-pcav':>12s}  {'theta':>12s}  regime")
    print('-' * 60)
    transition_eps = 0.01 * abs(_PCAV)
    for i, (a, th) in enumerate(zip(a_vals, theta_vals)):
        if abs(a) < transition_eps:
            regime = 'transition'
        elif a > 0:
            regime = 'full-film'
        else:
            regime = 'cavitation'
        print(f"  {i:3d}  {p_vals[i]:12.4e}  {a:12.4e}  {th:12.4e}  {regime}")

    n_ff  = np.sum(a_vals >  transition_eps)
    n_cav = np.sum(a_vals < -transition_eps)
    n_tr  = len(a_vals) - n_ff - n_cav
    print(f"\nSummary: {n_ff} full-film  {n_cav} cavitation  {n_tr} transition")
    assert n_ff  > 0, "No full-film nodes — increase Nx or delta"
    assert n_cav > 0, "No cavitation nodes — increase Nx or delta"
    assert n_tr  > 0, "No transition nodes — increase Nx or adjust ramp"


_SCALES = {'p': _PCAV, 'jx': 1.0, 'jy': 1.0, 'theta': 0.5}

if __name__ == '__main__':
    #print_state_coverage(Nx=4, Ny=3)
    #eps_sensitivity(Nx=4, Ny=3, bc='periodic_y', var_scales=_SCALES)
    compare_matrices('R_FB',          'p',     eps=1e-6, var_scales=_SCALES)
    compare_matrices('R_FB',          'theta', eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11x_fb',       'jx',    eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11x_fb',       'theta', eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11Sx_fb',      'jx',    eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11Sx_fb',      'theta', eps=1e-6, var_scales=_SCALES)
    # Correction terms must be tested combined with their parent term so that
    # FD sees the implicit p-dependence through dp_drho(rho(p)).
    compare_matrices('R11x_fb  + R11x_fb_corr',  'p', eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11y_fb  + R11y_fb_corr',  'p', eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11Sx_fb + R11Sx_fb_corr', 'p', eps=1e-6, var_scales=_SCALES)
    compare_matrices('R11Sy_fb + R11Sy_fb_corr', 'p', eps=1e-6, var_scales=_SCALES)
