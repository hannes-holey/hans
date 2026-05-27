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
Finite difference verification of assemble_matrix and assemble_rhs.

Assumes the pressure-based Taylor-Hood P2/P1 formulation throughout.
Two EOS configurations:
  - DH  (Dowson-Higginson): smooth, no cavitation
  - Bayada: cavitation EOS, tests terms near the mixture/liquid boundary
"""
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2d

# =============================================================================
# YAML config templates
# =============================================================================

_COMMON_GRID = """
options:
    output: /tmp/fem2d_fd_test_{label}
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
    xE_D: 1.1
    xW_D: 1.0
    yS_D: 1.05
    yN_D: 1.05

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
        {term_list_entry}
"""

_PROPS_DH = """
properties:
    EOS: DH
    rho0: 877.7007
    P0: 101325
    C1: 3.5e10
    C2: 1.23
    shear: 0.0794
    bulk: 0.
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

_CONFIG_DH = _COMMON_GRID + _PROPS_DH
_CONFIG_BAYADA = _COMMON_GRID + _PROPS_BAYADA

# Cavitation variant: same grid/props but with cavitation: True in the equations block.
# p_cav is added to properties so the FB term has a reference.
_COMMON_GRID_CAV = """
options:
    output: /tmp/fem2d_fd_test_{label}
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
    xE_D: 1.1
    xW_D: 1.0
    yS_D: 1.05
    yN_D: 1.05

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

_PROPS_BAYADA_CAV = """
properties:
    EOS: Bayada
    rho0: 850.0
    rho_l: 850.0
    rho_v: 0.019
    c_l: 1600.0
    c_v: 352.0
    shear: 0.039
    bulk: 0.0
    p_cav: 0.0
"""

_CONFIG_BAYADA_CAV = _COMMON_GRID_CAV + _PROPS_BAYADA_CAV

# =============================================================================
# Boundary condition presets
# =============================================================================

BC_CONFIGS = {
    'periodic': {
        'xE': "['P', 'P', 'P']",
        'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
    },
    'dirichlet': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
    },
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

def make_problem(eos: str, Nx: int, Ny: int, bc: str = 'periodic',
                 term_list: list = None, cavitation: bool = False) -> tuple:
    """Create and pre-run a (problem, solver) pair.

    Parameters
    ----------
    eos : {'DH', 'Bayada'}
    Nx, Ny : grid dimensions
    bc : key into BC_CONFIGS
    term_list : restrict active terms to this subset (passed through YAML)
    cavitation : if True, use the cavitation config (Bayada EOS + cavitation: True)
    """
    if cavitation:
        template = _CONFIG_BAYADA_CAV
    elif eos == 'DH':
        template = _CONFIG_DH
    else:
        template = _CONFIG_BAYADA
    bc_cfg = BC_CONFIGS[bc]
    term_list_entry = f"term_list: {term_list}" if term_list is not None else ""

    config = template.format(
        label=eos.lower() + ('_cav' if cavitation else ''),
        Nx=Nx, Ny=Ny,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
        term_list_entry=term_list_entry,
    )

    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()

    if eos == 'Bayada' or cavitation:
        _init_bayada_straddling(solver, cavitation=cavitation)

    return problem, solver


def _pcav(prop: dict) -> float:
    """Cavitation pressure for the Bayada EOS."""
    rho_l, rho_v = prop['rho_l'], prop['rho_v']
    c_l, c_v = prop['c_l'], prop['c_v']
    N = (rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
         / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2))
    return rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))


def _init_bayada_straddling(solver: FEMSolver2d,
                            cavitation: bool = False) -> None:
    """Set p as a ramp straddling Pcav so correction terms are nonzero.

    When cavitation=True, also sets theta to a smooth ramp in (0.1, 0.9) so
    the dtau/dtheta Jacobian block is exercised at non-trivial theta values.
    """
    Pcav = _pcav(solver.problem.prop)
    delta = 0.05 * abs(Pcav)
    q = solver.get_q_nodal().copy()
    p_sl = solver._sol_slices['p']
    n_p = p_sl.stop - p_sl.start
    q[p_sl] = np.linspace(Pcav - delta, Pcav + delta, n_p)
    if cavitation and 'theta' in solver._sol_slices:
        theta_sl = solver._sol_slices['theta']
        n_th = theta_sl.stop - theta_sl.start
        q[theta_sl] = np.linspace(0.1, 0.9, n_th)
    _set_and_sync(solver, q)


def _set_and_sync(solver: FEMSolver2d, q: np.ndarray) -> None:
    """Mirror the solver's own set→sync_rho→exchange→update_quad sequence."""
    solver.set_q_nodal(q)
    solver._sync_rho_before_exchange()
    solver.exchange_ghosts()
    solver.update_quad()


def build_scale_array(solver: FEMSolver2d,
                      scales: dict = None) -> np.ndarray:
    """Build a per-DOF scale array from per-variable characteristic scales.

    Parameters
    ----------
    scales : dict, optional
        Maps variable name to characteristic scale, e.g.
        {'p': 1e5, 'jx': 1.0, 'jy': 1.0}.
        Defaults to 1.0 for any variable not listed.
    """
    if scales is None:
        scales = {}
    q0 = solver.get_q_nodal()
    scale = np.ones(len(q0))
    for var, sl in solver._sol_slices.items():
        scale[sl] = scales.get(var, 1.0)
    return scale


def compute_fd_jacobian(solver: FEMSolver2d, eps: float = 1e-6,
                        scale: np.ndarray = None) -> np.ndarray:
    """Central finite difference Jacobian of get_R_ w.r.t. nodal DOFs.

    The step for DOF j is  eps_j = eps * scale[j].
    If scale is None, falls back to a relative perturbation:
      eps_j = eps * max(1, |q0[j]|).
    Restores original state before returning.
    """
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


# Correction terms have zero residual contribution; they only complete the
# analytic Jacobian for the implicit p→ρ→dp/dρ dependence.  Testing a base
# term without its correction will always fail the FD check.
_TERM_GROUPS = [
    ['R11x',  'R11x_corr'],
    ['R11y',  'R11y_corr'],
    ['R11Sx', 'R11Sx_corr'],
    ['R11Sy', 'R11Sy_corr'],
    ['R1T'],
    ['R21x'],
    ['R21y'],
    ['R2Tx'],
    ['R2Ty'],
    ['R24x'],
    ['R24y'],
]

# Term groups that require cavitation=True (theta DOF present)
_TERM_GROUPS_CAV = [
    ['R24x_fb'],
    ['R24y_fb'],
]


def check_all_terms(
    eos: str = 'Bayada',
    Nx: int = 3,
    Ny: int = 2,
    bc: str = 'periodic_y',
    eps: float = 1e-6,
    scale: np.ndarray = None,
) -> dict:
    """Check each term group against its FD Jacobian, broken down by dep-var column block.

    Creates one problem, then swaps solver.terms for each group.
    Returns a nested dict: results[group_label][dep_var] = rel_err,
    plus results[group_label]['total'] = overall rel_err.
    """
    _, solver = make_problem(eos, Nx, Ny, bc=bc)
    all_terms = solver.terms
    term_by_name = {t.name: t for t in all_terms}
    sol_slices = solver._sol_slices   # {'jx': slice, 'jy': slice, 'p': slice}

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


def print_term_check(results: dict) -> None:
    """Pretty-print the output of check_all_terms."""
    dep_vars = ['jx', 'jy', 'p']
    header = f"{'term':<30s}  {'total':>8s}" + ''.join(f"  {'d/d'+v:>8s}" for v in dep_vars)
    print(header)
    print('-' * len(header))
    for label, entry in results.items():
        total = entry['total']
        flag = '' if total < 1e-4 else ' ✗'
        row = f"{label:<30s}  {total:>8.2e}"
        for v in dep_vars:
            row += f"  {entry.get(v, float('nan')):>8.2e}"
        print(row + flag)


def eps_sensitivity(
    eos: str = 'Bayada',
    Nx: int = 3,
    Ny: int = 2,
    bc: str = 'periodic_y',
    eps_list: list = None,
    var_scales: dict = None,
) -> None:
    """Run check_all_terms for each eps in eps_list.

    Prints one sub-table per eps value showing total error and per-dep-var
    column block errors (d/djx, d/djy, d/dp).

    Parameters
    ----------
    eps_list : list of float
        FD step sizes to sweep. Default: [1e-4, 1e-5, 1e-6, 1e-7, 1e-8].
    var_scales : dict, optional
        Per-variable characteristic scales for build_scale_array,
        e.g. {'p': 1e5, 'jx': 1.0, 'jy': 1.0}.
        If None, falls back to the relative-perturbation default.
    """
    if eps_list is None:
        eps_list = [1e-4, 1e-6, 1e-8]

    _, solver = make_problem(eos, Nx, Ny, bc=bc)
    scale = build_scale_array(solver, var_scales) if var_scales is not None else None

    dep_vars = ['jx', 'jy', 'p']
    col_w = 9
    header = (f"  {'term':<28s}  {'total':>{col_w}s}"
              + ''.join(f"  {'d/d'+v:>{col_w}s}" for v in dep_vars))
    sep = '  ' + '-' * (len(header) - 2)

    print(f"\nEOS={eos}  grid={Nx}x{Ny}  bc={bc}"
          + (f"  scales={var_scales}" if var_scales else "  scales=relative"))

    for eps in eps_list:
        r = check_all_terms(eos, Nx, Ny, bc=bc, eps=eps, scale=scale)
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
    eos: str = 'Bayada',
    Nx: int = 3,
    Ny: int = 2,
    bc: str = 'periodic_y',
    var_scales: dict = None,
    fmt: str = '11.2e',
) -> None:
    """Print analytic and FD Jacobian side-by-side for one term, one dep-var column block.

    Parameters
    ----------
    term : str
        Name of a single term (e.g. 'R11x') or a '+'-joined group (e.g. 'R11x+R11x_corr').
        Correction terms must be included together with their base term.
    block : str
        Dependent variable whose column block to show: 'jx', 'jy', or 'p'.
    eps : float
        FD step size.
    eos : {'DH', 'Bayada'}
    Nx, Ny : grid dimensions
    bc : key into BC_CONFIGS
    var_scales : dict, optional
        Per-variable scales for build_scale_array.
    fmt : str
        Python format spec for each matrix entry (default '11.2e').
    """
    _, solver = make_problem(eos, Nx, Ny, bc=bc)
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

    # rows: union of res_slices for all residuals the selected terms write to
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

    print(f"\nterm={term}  block=d/d{block}  eps={eps:.0e}  "
          f"EOS={eos}  grid={Nx}x{Ny}  bc={bc}")
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
# Diagnostic: plot initial pressure field
# =============================================================================

def plot_initial_p(eos: str = 'Bayada', Nx: int = 6, Ny: int = 4,
                   bc: str = 'periodic_y') -> None:
    """Plot the initial p field after make_problem initialisation.

    For Bayada, the ramp straddling Pcav is visible as a smooth gradient
    with the cavitation pressure marked as a contour line.
    Run directly:  python tests/test_fem_2d_assembly_fd.py
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _, solver = make_problem(eos, Nx, Ny, bc=bc)

    gi = solver.grid_idx
    Nx_p = gi.Nx_p_inner
    Ny_p = gi.Ny_p_inner

    q = solver.get_q_nodal()
    p_sl = solver._sol_slices['p']
    p2d = q[p_sl].reshape((Nx_p, Ny_p), order='F') / 1e5   # MPa

    Lx = solver.problem.decomp.grid['Lx']
    Ly = solver.problem.decomp.grid['Ly']
    x = np.linspace(0, Lx * 1e3, Nx_p)   # mm
    y = np.linspace(0, Ly * 1e3, Ny_p)   # mm

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'axes.labelsize': 10, 'figure.dpi': 150,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=(5.0, 3.0), layout='constrained')
    im = ax.pcolormesh(x, y, p2d.T, shading='auto', cmap='RdBu_r')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$p$ [bar]')

    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.set_title(f'Initial $p$ — {eos}, {Nx}×{Ny}, {bc}')
    plt.show()


if __name__ == '__main__':
    #plot_initial_p(eos='Bayada', Nx=6, Ny=4, bc='periodic')
    #eps_sensitivity(eos='Bayada', Nx=5, Ny=3, bc='periodic_y')

    #compare_matrices('R11x', 'p', eps=1e-6)
    compare_matrices('R11x+R11x_corr', 'p', eps=1e-6)
    #compare_matrices('R2Tx', 'jx', eps=1e-6, eos='DH')
    #compare_matrices('R24x', 'p', eps=1e-6, eos='Bayada')
