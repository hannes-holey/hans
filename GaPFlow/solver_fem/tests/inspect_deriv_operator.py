"""Interactive derivative operator inspector.

Sets up a problem, overwrites rho/jx/jy with fields of known spatial
derivatives, then applies get_deriv_dx / get_deriv_dy and compares the
quad-point output against the expected analytical values.

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/solver_fem/tests/inspect_deriv_operator.py

Edit the CONFIG section at the bottom to change parameters.
"""

import numpy as np
from GaPFlow.problem import Problem

# =============================================================================
# Config — edit these
# =============================================================================

Nx = 3
Ny = 2
Lx = 3
Ly = 2
bc = 'periodic'

# Which fields to test: 'rho' (P1, coarse), 'jx' or 'jy' (P2, fine)
test_fields = ['rho', 'jx']

# Slope of the linear field:  f(x,y) = f0 + slope_x * x + slope_y * y
f0 = 1.0
slope_x = 3.0
slope_y = -2.0

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
}

_CONFIG_TEMPLATE = """
options:
    output: /tmp/inspect_deriv
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
    shear: 1e-3
    bulk: 0.
    P0: 101325
    alpha: 0.
fem_solver:
    type: newton_alpha
    equations:
        energy: False
        term_list: ['R1T']
"""


# =============================================================================
# Build problem
# =============================================================================

def make_solver(Nx, Ny, Lx, Ly, bc):
    bc_cfg = BC_CONFIGS[bc]
    config = _CONFIG_TEMPLATE.format(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
    )
    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()
    return problem, solver


# =============================================================================
# Set field to known linear function
# =============================================================================

def set_linear_field(solver, field_name, f0, slope_x, slope_y):
    """Overwrite nodal field with f(x,y) = f0 + slope_x*x + slope_y*y.

    Returns the nodal field array (padded) for inspection.
    """
    dx = solver.dx
    dy = solver.dy
    qm = solver.quad_mgr
    nf = qm.nodal_fields[field_name]
    pg = nf.pg[0]  # shape (Nx_pad, Ny_pad)

    Nx_pad, Ny_pad = pg.shape

    # Determine physical coordinates of each node.
    # For P2 (jx, jy): fine grid with spacing dx/2, dy/2
    # For P1 (rho): coarse grid with spacing dx, dy
    if field_name in ('jx', 'jy'):
        hx = dx / 2
        hy = dy / 2
        ghost = 2
    else:
        hx = dx
        hy = dy
        ghost = 1

    # Node (i,j) in padded array -> physical (x, y)
    # Inner nodes start at index `ghost`, so physical origin at ghost offset
    for i in range(Nx_pad):
        for j in range(Ny_pad):
            x = (i - ghost) * hx
            y = (j - ghost) * hy
            pg[i, j] = f0 + slope_x * x + slope_y * y

    return pg


# =============================================================================
# Main
# =============================================================================

def print_sq_array(arr, Nx_sq, Ny_sq, label=''):
    """Print a (nb_sq, nb_quad_sq) array matching nodal field layout.

    Rows = x (top to bottom), columns = y (left to right).
    get_quad returns y-major ordering: sq_idx = sy * Nx_sq + sx.
    """
    if label:
        print(f'\n{label}')
    # Build 2D lookup: grid[sx][sy] = arr[sq_idx]
    grid = [[None] * Ny_sq for _ in range(Nx_sq)]
    for idx in range(arr.shape[0]):
        sy = idx // Nx_sq
        sx = idx % Nx_sq
        grid[sx][sy] = arr[idx]

    # Header
    col_w = 8
    hdr = ' ' * 10 + ''.join(f'{"sy="+str(sy):^{col_w * arr.shape[1]}}' for sy in range(Ny_sq))
    print(hdr)

    for sx in range(Nx_sq):
        parts = []
        for sy in range(Ny_sq):
            vals = grid[sx][sy]
            parts.append(' '.join(f'{v:7.3f}' for v in vals))
        print(f'  sx={sx:<3d}  ' + '  |  '.join(parts))


def run_test(solver, field_name, f0, slope_x, slope_y):
    """Test derivative operator for one field."""
    qm = solver.quad_mgr

    # Square grid dimensions from placeholder shape
    ph = qm._deriv_placeholder.pg
    Nx_sq = ph.shape[1] - 1
    Ny_sq = ph.shape[2] - 1

    grid_type = 'P2 (fine)' if field_name in ('jx', 'jy') else 'P1 (coarse)'
    print(f'\n{"#"*60}')
    print(f'# Field: {field_name}  ({grid_type})')
    print(f'# f(x,y) = {f0} + {slope_x}*x + {slope_y}*y')
    print(f'# Expected: df/dx = {slope_x},  df/dy = {slope_y}')
    print(f'# Nx_sq={Nx_sq}, Ny_sq={Ny_sq}')
    print(f'{"#"*60}')

    # --- Overwrite field ---
    pg = set_linear_field(solver, field_name, f0, slope_x, slope_y)
    print(f'\nNodal field (padded array):')
    print(pg)

    # --- Apply derivative operators ---
    deriv_dx = qm.get_quad_dx_sq(field_name)
    deriv_dy = qm.get_quad_dy_sq(field_name)

    # --- Compare against analytical ---
    err_dx = deriv_dx - slope_x
    err_dy = deriv_dy - slope_y

    print(f'\n{"="*60}')
    print(f'd/dx results  (expected: {slope_x})')
    print(f'{"="*60}')
    print(f'  min = {deriv_dx.min():.10f}')
    print(f'  max = {deriv_dx.max():.10f}')
    print(f'  max |error| = {np.abs(err_dx).max():.2e}')
    print_sq_array(deriv_dx, Nx_sq, Ny_sq)

    print(f'\n{"="*60}')
    print(f'd/dy results  (expected: {slope_y})')
    print(f'{"="*60}')
    print(f'  min = {deriv_dy.min():.10f}')
    print(f'  max = {deriv_dy.max():.10f}')
    print(f'  max |error| = {np.abs(err_dy).max():.2e}')
    print_sq_array(deriv_dy, Nx_sq, Ny_sq)

    # --- Also check interpolation ---
    qm.interpolate_nodal_to_quad(field_name)
    quad_vals = qm.get_quad_sq(field_name)
    print(f'\n{"="*60}')
    print(f'Interpolated quad values')
    print(f'{"="*60}')
    print_sq_array(quad_vals, Nx_sq, Ny_sq)

    # --- Summary ---
    dx_ok = np.abs(err_dx).max() < 1e-8
    dy_ok = np.abs(err_dy).max() < 1e-8
    print(f'\n  d/dx: {"PASS" if dx_ok else "FAIL"}  (max err = {np.abs(err_dx).max():.2e})')
    print(f'  d/dy: {"PASS" if dy_ok else "FAIL"}  (max err = {np.abs(err_dy).max():.2e})')

    return dx_ok, dy_ok


if __name__ == '__main__':
    np.set_printoptions(precision=6, linewidth=120)

    print(f'Grid: {Nx}x{Ny}  Lx={Lx} Ly={Ly}  bc={bc}')
    print(f'f(x,y) = {f0} + {slope_x}*x + {slope_y}*y')
    print()

    problem, solver = make_solver(Nx, Ny, Lx, Ly, bc)
    print(f'dx = {solver.dx},  dy = {solver.dy}')

    results = {}
    for field in test_fields:
        dx_ok, dy_ok = run_test(solver, field, f0, slope_x, slope_y)
        results[field] = (dx_ok, dy_ok)

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    for field, (dx_ok, dy_ok) in results.items():
        print(f'  {field:4s}  d/dx: {"PASS" if dx_ok else "FAIL"}   d/dy: {"PASS" if dy_ok else "FAIL"}')
    print(f'{"="*60}')
