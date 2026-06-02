"""3D scatter plot of interpolated quad-point values.

Sets up a problem, overwrites a field with a known function,
interpolates to quad points, and plots them in 3D to check smoothness.

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/solver_fem/tests/inspect_interpolation_op.py

Edit the CONFIG section to change parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from GaPFlow.problem import Problem

# =============================================================================
# Config — edit these
# =============================================================================

Nx = 4
Ny = 3
Lx = 4
Ly = 3
bc = 'periodic'

# Which field to plot: 'rho' (P1, coarse), 'jx' or 'jy' (P2, fine)
test_field = 'jx'

# Field function: f(x,y) = f0 + slope_x * x + slope_y * y + curve * x*y
f0 = 1.0
slope_x = 3.0
slope_y = -2.0
curve = 0.5    # set nonzero for a non-linear field

# =============================================================================
# BC presets
# =============================================================================

BC_CONFIGS = {
    'periodic': {
        'xE': "['P', 'P', 'P']", 'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']", 'yN': "['P', 'P', 'P']",
    },
}

_CONFIG_TEMPLATE = """
options:
    output: /tmp/inspect_interp
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
# Helpers
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


def set_field(solver, field_name, f0, slope_x, slope_y, curve):
    """Overwrite nodal field with f(x,y) = f0 + slope_x*x + slope_y*y + curve*x*y."""
    dx = solver.dx
    dy = solver.dy
    qm = solver.quad_mgr
    nf = qm.nodal_fields[field_name]
    pg = nf.pg[0]

    Nx_pad, Ny_pad = pg.shape

    if field_name in ('jx', 'jy'):
        hx = dx / 2
        hy = dy / 2
        ghost = 2
    else:
        hx = dx
        hy = dy
        ghost = 1

    for i in range(Nx_pad):
        for j in range(Ny_pad):
            x = (i - ghost) * hx
            y = (j - ghost) * hy
            pg[i, j] = f0 + slope_x * x + slope_y * y + curve * x * y


def quad_physical_coords(solver, Nx_sq, Ny_sq):
    """Compute physical (x, y) for every quad point in get_quad ordering.

    Returns x_coords, y_coords of shape (nb_sq, 6).
    get_quad ordering: sq_idx = sy * Nx_sq + sx.
    6 quad points per square: tri0 q0,q1,q2, tri1 q0,q1,q2.

    The padded grid includes one ghost layer of squares on each side,
    so sx=0 corresponds to the ghost square at x = -dx (P1 ghost depth 1).
    Inner squares start at sx=1, sy=1.
    """
    dx = solver.dx
    dy = solver.dy
    quad_ref = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])

    # Ghost offset: P1 ghost depth = 1 square
    ghost = 1

    nb_sq = Nx_sq * Ny_sq
    x_coords = np.empty((nb_sq, 6))
    y_coords = np.empty((nb_sq, 6))

    for idx in range(nb_sq):
        sy = idx // Nx_sq
        sx = idx % Nx_sq
        # Square origin in physical space, accounting for ghost offset
        x0 = (sx - ghost) * dx
        y0 = (sy - ghost) * dy

        for qi in range(3):
            xi, eta = quad_ref[qi]
            # tri0: lower-left triangle
            x_coords[idx, qi] = x0 + xi * dx
            y_coords[idx, qi] = y0 + eta * dy
            # tri1: upper-right triangle (flipped: 1-xi, 1-eta)
            x_coords[idx, 3 + qi] = x0 + (1 - xi) * dx
            y_coords[idx, 3 + qi] = y0 + (1 - eta) * dy

    return x_coords, y_coords


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print(f'Grid: {Nx}x{Ny}  Lx={Lx} Ly={Ly}  bc={bc}')
    print(f'Field: {test_field}')
    print(f'f(x,y) = {f0} + {slope_x}*x + {slope_y}*y + {curve}*x*y')

    _, solver = make_solver(Nx, Ny, Lx, Ly, bc)
    dx = solver.dx
    dy = solver.dy
    qm = solver.quad_mgr

    # Square grid dimensions
    ph = qm._deriv_placeholder.pg
    Nx_sq = ph.shape[1] - 1
    Ny_sq = ph.shape[2] - 1

    print(f'dx={dx}, dy={dy}, Nx_sq={Nx_sq}, Ny_sq={Ny_sq}')

    # Set field and interpolate
    set_field(solver, test_field, f0, slope_x, slope_y, curve)
    qm.interpolate_nodal_to_quad(test_field)
    quad_vals = qm.get_quad_sq(test_field)

    # Physical coordinates of quad points
    x_coords, y_coords = quad_physical_coords(solver, Nx_sq, Ny_sq)

    # Expected values
    expected = f0 + slope_x * x_coords + slope_y * y_coords + curve * x_coords * y_coords
    error = quad_vals - expected

    print(f'max |error| = {np.abs(error).max():.2e}')

    # --- 3D plot ---
    fig = plt.figure(figsize=(14, 5))

    # Plot 1: interpolated values
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(x_coords.flat, y_coords.flat, quad_vals.flat,
                      c=quad_vals.flat, cmap='viridis', s=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('value')
    ax1.set_title(f'Interpolated ({test_field})')
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    # Plot 2: expected
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(x_coords.flat, y_coords.flat, expected.flat,
                      c=expected.flat, cmap='viridis', s=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('value')
    ax2.set_title('Expected')
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    # Plot 3: error
    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(x_coords.flat, y_coords.flat, error.flat,
                      c=error.flat, cmap='RdBu', s=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('error')
    ax3.set_title('Error')
    fig.colorbar(sc3, ax=ax3, shrink=0.5)

    plt.savefig('inspect_interpolation.png', dpi=150, bbox_inches='tight')
    print('Saved: inspect_interpolation.png')
    plt.show()
