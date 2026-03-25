"""Interactive residual inspector.

Set individual jx nodal values and see the resulting residual.

Usage:
    python GaPFlow/fem_2d/tests/interactive_residual.py

At the prompt, type:
    ix,iy,val   — set jx at inner node (ix,iy) to val
    r           — reset all jx to zero
    q           — quit
"""

import numpy as np
from GaPFlow.problem import Problem

# =============================================================================
# Config
# =============================================================================

Nx        = 3
Ny        = 3
Lx        = 10
Ly        = 10
term_list = ['R23xx']
bc        = 'periodic_y'

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
    output: /tmp/interactive_res
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
# Setup
# =============================================================================

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

gi = solver.grid_idx
Nx_v = gi.Nx_v_inner
Ny_v = gi.Ny_v_inner
Nx_v_pad = gi.Nx_v_padded
Ny_v_pad = gi.Ny_v_padded
ghost = 2

jx_slice = solver._sol_slices['jx']
q_base = solver.get_q_nodal().copy()

print(f'\nGrid: {Nx}x{Ny}, Lx={Lx}, Ly={Ly}, bc={bc}, terms={term_list}')
print(f'P2 inner: {Nx_v}x{Ny_v}, P2 padded: {Nx_v_pad}x{Ny_v_pad}')
print(f'jx DOF range: [{jx_slice.start}, {jx_slice.stop})')
print(f'Inner node (ix,iy) -> flat index: ix + iy*{Nx_v}')
print()

# =============================================================================
# Helpers
# =============================================================================

def print_field_2d(label, field_flat, Nx, Ny):
    """Print a 2D field from F-order flat array."""
    arr = field_flat.reshape((Nx, Ny), order='F')
    print(f'\n{label} (rows=iy, cols=ix):')
    header = '     ' + ''.join(f'{ix:>10d}' for ix in range(Nx))
    print(header)
    for iy in range(Ny):
        row = ''.join(f'{arr[ix, iy]:10.4f}' for ix in range(Nx))
        print(f'iy={iy:2d} {row}')

def print_padded_field(label, solver, var='jx'):
    """Print the full padded nodal field."""
    pg = solver.quad_mgr.nodal_fields[var].pg[0]
    Nx_p, Ny_p = pg.shape
    print(f'\n{label} padded (rows=iy, cols=ix):')
    header = '     ' + ''.join(f'{ix:>10d}' for ix in range(Nx_p))
    print(header)
    for iy in range(Ny_p):
        row = ''.join(f'{pg[ix, iy]:10.4f}' for ix in range(Nx_p))
        marker = ''
        if iy < ghost or iy >= Ny_p - ghost:
            marker = '  <- ghost'
        print(f'iy={iy:2d} {row}{marker}')

def print_residual(solver):
    """Print residual per equation, with node coordinates."""
    from GaPFlow.fem_2d.assembly import DOF_GRID
    R = solver.get_R()
    for res in solver.residuals:
        rs = solver.assembly._res_slices[res]
        R_block = R[rs.start:rs.stop]
        grid = DOF_GRID[res]
        nx = gi.Nx_v_inner if grid == 'v' else gi.Nx_p_inner
        ny = gi.Ny_v_inner if grid == 'v' else gi.Ny_p_inner
        arr = R_block.reshape((nx, ny), order='F')
        print(f'\nR[{res}] (rows=iy, cols=ix):')
        header = '     ' + ''.join(f'{ix:>10d}' for ix in range(nx))
        print(header)
        for iy in range(ny):
            row = ''.join(f'{arr[ix, iy]:10.4f}' for ix in range(nx))
            print(f'iy={iy:2d} {row}')

# =============================================================================
# Interactive loop
# =============================================================================

# Start from zero jx
q = q_base.copy()
q[jx_slice.start:jx_slice.stop] = 0.0

print('\nCommands:')
print('  ix,iy,val  — set jx(ix,iy) = val')
print('  r          — reset all jx to 0')
print('  p          — print padded jx field (with ghosts)')
print('  q          — quit')

while True:
    print()
    cmd = input('> ').strip()
    if not cmd:
        continue
    if cmd == 'q':
        break
    if cmd == 'r':
        q[jx_slice.start:jx_slice.stop] = 0.0
        print('Reset jx to zero.')
    elif cmd == 'p':
        solver.set_q_nodal(q)
        solver.exchange_ghosts()
        print_padded_field('jx', solver, 'jx')
        continue
    else:
        try:
            parts = cmd.split(',')
            ix, iy, val = int(parts[0]), int(parts[1]), float(parts[2])
            if ix < 0 or ix >= Nx_v or iy < 0 or iy >= Ny_v:
                print(f'Out of range: ix in [0,{Nx_v-1}], iy in [0,{Ny_v-1}]')
                continue
            flat_idx = ix + iy * Nx_v
            q[jx_slice.start + flat_idx] = val
            print(f'Set jx({ix},{iy}) = {val}')
        except (ValueError, IndexError):
            print('Invalid input. Use: ix,iy,val  or  r  or  p  or  q')
            continue

    # Update and show
    solver.set_q_nodal(q)
    solver.exchange_ghosts()
    solver.update_quad()

    jx_inner = q[jx_slice.start:jx_slice.stop]
    print_field_2d('jx (inner)', jx_inner, Nx_v, Ny_v)
    print_residual(solver)
