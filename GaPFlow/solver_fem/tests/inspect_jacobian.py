"""Interactive Jacobian inspector.

Compares analytical assemble_matrix with finite-difference Jacobian
for a chosen term subset, grid size, and BC configuration.

Usage (synthetic config):
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/solver_fem/tests/inspect_jacobian.py

Usage (from YAML):
    Set RUN_FROM_YAML = True and YAML_PATH below, then run as above.

Edit the CONFIG / YAML CONFIG sections at the bottom to change parameters.
"""

import os
import numpy as np
import yaml
from GaPFlow.problem import Problem

# =============================================================================
# Config — edit these
# =============================================================================

Nx        = 2
Ny        = 2
Lx        = 10
Ly        = 10
term_list = ['R21x', 'R21y']        # e.g. ['R1T'], ['R11x'], ['R1T','R11x','R11y']
bc        = 'dirichlet'    # 'dirichlet' | 'periodic' | 'periodic_y'
block     = ('momentum_x', 'rho')  # (residual, variable) block to print; None = full matrix
fd_eps    = 1e-6

# =============================================================================
# BC presets
# =============================================================================

BC_CONFIGS = {
    'dirichlet': {
        'xE': "['D', 'N', 'N']", 'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']", 'yN': "['D', 'N', 'N']",
    },
    'periodic': {
        'xE': "['P', 'P', 'P']", 'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']", 'yN': "['P', 'P', 'P']",
    },
    'periodic_y': {
        'xE': "['D', 'N', 'N']", 'xW': "['D', 'N', 'N']",
        'yS': "['P', 'P', 'P']", 'yN': "['P', 'P', 'P']",
    },
}

_CONFIG_TEMPLATE = """
options:
    output: /tmp/inspect_jac
    write_freq: 1000
    silent: True
grid:
    Lx: {Lx}
    Ly: {Ly}
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
    type: inclined
    hmax: 1e-5
    hmin: 1e-5
    U: 0.0
    V: 0.0
numerics:
    solver: fem
    dt: 1e-3
    tol: 1e-6
    max_it: 100
properties:
    EOS: PL
    rho0: 1.0
    shear: 10
    bulk: 0.
    P0: 101325
    alpha: 0.
fem_solver:
    type: newton_alpha
    equations:
        energy: False
        {term_list_entry}
"""

# =============================================================================
# YAML CONFIG — used when RUN_FROM_YAML = True
# =============================================================================

RUN_FROM_YAML = True

YAML_PATH = os.path.join(
    os.path.dirname(__file__),
    '../twin_parabolic_slider/twin_parabolic_slider.yaml',
)

# Overrides applied on top of the YAML before loading.
# Use nested dicts matching the YAML structure, e.g.:
#   {'grid': {'Nx': 8, 'Ny': 4}, 'fem_solver': {'scaling': False}}
YAML_OVERRIDES = {
    'grid': {'Nx': 4, 'Ny': 3},
    'options': {'output': '/tmp/inspect_jac_yaml', 'write_freq': 1000,
                'save_output': False, 'output_plots': False,
                'residual_analysis': False},
    'fem_solver': {'newton_debug': False, 'scaling': False},
}

# Block and FD settings for the YAML run (reuse same vars as synthetic)
# Set YAML_block = None to skip per-block matrix printout (only summary shown)
# ('momentum_x', 'jx')  
YAML_block  = ('momentum_x', 'rho')
YAML_fd_eps = 1e-9


# =============================================================================
# Build problem
# =============================================================================

def _deep_update(base, overrides):
    """Recursively merge overrides into base dict (in-place)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def make_solver_from_yaml(yaml_path, overrides=None):
    """Load a Problem from a YAML file, apply overrides, return solver."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)
    # Write patched config to a temp string and load via from_string so that
    # relative paths in the YAML are resolved from the YAML's own directory.
    import tempfile
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', dir=yaml_dir, delete=False)
    try:
        yaml.dump(cfg, tmp)
        tmp.close()
        problem = Problem.from_yaml(tmp.name)
    finally:
        os.unlink(tmp.name)
    solver = problem.solver
    solver.pre_run()
    return solver


def make_solver(Nx, Ny, Lx, Ly, bc, term_list):
    bc_cfg = BC_CONFIGS[bc]
    term_entry = f'term_list: {term_list}' if term_list else ''
    config = _CONFIG_TEMPLATE.format(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
        term_list_entry=term_entry,
    )
    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()
    return solver


# =============================================================================
# FD Jacobian
# =============================================================================

def compute_fd_jacobian(solver, eps=1e-8):
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J = np.zeros((n, n))
    for j in range(n):
        eps_j = max(eps, eps * abs(q0[j]))
        q_p = q0.copy(); q_p[j] += eps_j
        solver.set_q_nodal(q_p); solver.exchange_ghosts(); solver.update_quad()
        Rp = solver.get_R().copy()
        q_m = q0.copy(); q_m[j] -= eps_j
        solver.set_q_nodal(q_m); solver.exchange_ghosts(); solver.update_quad()
        Rm = solver.get_R().copy()
        J[:, j] = (Rp - Rm) / (2.0 * eps_j)
    solver.set_q_nodal(q0); solver.exchange_ghosts(); solver.update_quad()
    return J


# =============================================================================
# Axis labels
# =============================================================================

def node_coords(solver, name):
    """Return list of (ix, iy) for each DOF in block `name` (res or var).

    F-order flatten: x varies fastest, so index i -> ix = i % Nx, iy = i // Nx.
    """
    from GaPFlow.solver_fem.assembly import DOF_GRID
    grid = DOF_GRID[name]
    if grid == 'v':
        Nx = solver.grid_idx.Nx_v_inner
        Ny = solver.grid_idx.Ny_v_inner
    else:
        Nx = solver.grid_idx.Nx_p_inner
        Ny = solver.grid_idx.Ny_p_inner
    n = Nx * Ny
    return [(i % Nx, i // Nx) for i in range(n)]


def _col_header(coords, col_w=10):
    """Single-line column header: '(ix,iy)' centred in col_w chars."""
    parts = [f'{"("+str(ix)+","+str(iy)+")":^{col_w}}' for ix, iy in coords]
    return ' ' * col_w + ''.join(parts)


def _row_label(ix, iy, col_w=10):
    """Row label left-padded to col_w chars."""
    return f'{"("+str(ix)+","+str(iy)+")":>{col_w}}'


# =============================================================================
# Pretty print
# =============================================================================

def print_matrix(label, M, row_coords=None, col_coords=None):
    scale = max(np.abs(M).max(), 1e-30)
    threshold = scale * 1e-4
    col_w = 10
    print(f'\n{label}:')
    if col_coords is not None:
        print(_col_header(col_coords, col_w))
    for i, row in enumerate(M):
        parts = []
        for v in row:
            if abs(v) <= threshold:
                parts.append('         0')
            else:
                parts.append(f'{v:10.1f}')
        prefix = _row_label(*row_coords[i]) if row_coords is not None else '  '
        print(prefix + ''.join(parts))


def print_ratio(M_anal, M_fd, row_coords=None, col_coords=None):
    print('\nRatio M_anal / M_fd  (. = both ~zero):')
    scale = max(np.abs(M_fd).max(), np.abs(M_anal).max(), 1e-30)
    threshold = scale * 1e-4
    col_w = 10
    if col_coords is not None:
        print(_col_header(col_coords, col_w))
    for i in range(M_anal.shape[0]):
        parts = []
        for j in range(M_anal.shape[1]):
            if abs(M_fd[i, j]) <= threshold and abs(M_anal[i, j]) <= threshold:
                parts.append('         .')
            else:
                parts.append(f'{M_anal[i, j] / (M_fd[i, j] + 1e-30):10.4f}')
        prefix = _row_label(*row_coords[i]) if row_coords is not None else '  '
        print(prefix + ''.join(parts))


def print_summary(M_anal, M_fd):
    scale = max(np.abs(M_fd).max(), np.abs(M_anal).max(), 1e-30)
    threshold = scale * 1e-4
    nz = np.abs(M_fd) > threshold
    rel_err = (np.linalg.norm(M_anal - M_fd) /
               (np.linalg.norm(M_fd) + 1e-15))
    diag_fd = np.diag(M_fd)
    diag_nz = np.abs(diag_fd) > threshold
    diag_ratio = np.diag(M_anal)[diag_nz] / diag_fd[diag_nz]
    print(f'\nrel_err = {rel_err:.4e}')
    print(f'diag ratio (unique): {np.unique(np.round(diag_ratio, 4))}')
    off_ratios = M_anal[nz] / M_fd[nz]
    print(f'all nonzero ratios (unique): {np.unique(np.round(off_ratios, 4))}')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    if RUN_FROM_YAML:
        print(f'Loading from YAML: {YAML_PATH}')
        print(f'Overrides: {YAML_OVERRIDES}')
        solver = make_solver_from_yaml(YAML_PATH, YAML_OVERRIDES)
        block = YAML_block
        fd_eps = YAML_fd_eps
    else:
        print(f'Grid: {Nx}x{Ny}  Lx={Lx} Ly={Ly}  bc={bc}  terms={term_list}')
        solver = make_solver(Nx, Ny, Lx, Ly, bc, term_list)

    print(f'Variables : {solver.variables}')
    print(f'Residuals : {solver.residuals}')
    print(f'DOF sizes : '
          + ', '.join(f'{v}={solver.assembly._sol_slices[v].stop - solver.assembly._sol_slices[v].start}'
                      for v in solver.variables))

    print('\nComputing FD Jacobian...', end='', flush=True)
    J_fd = compute_fd_jacobian(solver, eps=fd_eps)
    print(' done.')

    M_anal = solver.get_M_dense()

    if block is not None:
        res_name, var_name = block
        rs = solver.assembly._res_slices[res_name]
        ss = solver.assembly._sol_slices[var_name]
        M_b = M_anal[rs, ss]
        J_b = J_fd[rs, ss]
        label_suffix = f'  block M[{res_name}, {var_name}]'
        row_coords = node_coords(solver, res_name)
        col_coords = node_coords(solver, var_name)
    else:
        M_b = M_anal
        J_b = J_fd
        label_suffix = '  (full matrix)'
        row_coords = None
        col_coords = None

    if block is not None:
        print_matrix('Analytical' + label_suffix, M_b, row_coords, col_coords)
        print_matrix('FD        ' + label_suffix, J_b, row_coords, col_coords)
        print_ratio(M_b, J_b, row_coords, col_coords)
        print_summary(M_b, J_b)

    # Check all blocks: rel_err per block
    print(f'\n{"="*70}')
    print('Block-wise comparison (analytical vs FD):')
    print(f'{"="*70}')
    all_ok = True
    for res in solver.residuals:
        for var in solver.variables:
            rs2 = solver.assembly._res_slices[res]
            ss2 = solver.assembly._sol_slices[var]
            M_b2 = M_anal[rs2, ss2]
            J_b2 = J_fd[rs2, ss2]
            norm_fd = np.linalg.norm(J_b2)
            norm_anal = np.linalg.norm(M_b2)
            if norm_fd > 1e-10:
                err = np.linalg.norm(M_b2 - J_b2) / norm_fd
                status = 'OK' if err < 1e-6 else 'FAIL'
                print(f'  M[{res:12s}, {var:4s}]: rel_err={err:.2e}  {status}')
            elif norm_anal > 1e-10:
                print(f'  M[{res:12s}, {var:4s}]: FD=0 but anal={norm_anal:.2e}  FAIL')
                status = 'FAIL'
            else:
                print(f'  M[{res:12s}, {var:4s}]: both zero  OK')
                status = 'OK'
            if status == 'FAIL':
                all_ok = False
    print(f'{"="*70}')
    print(f'Overall: {"ALL BLOCKS OK" if all_ok else "SOME BLOCKS FAILED"}')
