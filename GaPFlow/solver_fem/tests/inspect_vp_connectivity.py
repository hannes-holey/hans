"""Inspect v->p and p->v connectivity in the assembly.

For a small grid, prints all unique contribution pairs
from the element loop for both block types:
  v->p : jx -> mass  (e.g. R11x term)
  p->v : rho -> momentum_x  (e.g. R21x term)

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/solver_fem/tests/inspect_vp_connectivity.py
"""

import numpy as np
from collections import defaultdict
from GaPFlow.problem import Problem

# =============================================================================
# Config
# =============================================================================

Nx = 3
Ny = 3
Lx = 3
Ly = 2
bc = 'dirichlet'

# =============================================================================
# Setup
# =============================================================================

BC_CONFIGS = {
    'periodic': {
        'xE': "['P', 'P', 'P']", 'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']", 'yN': "['P', 'P', 'P']",
    },
    'dirichlet': {
        'xE': "['D', 'N', 'N']", 'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']", 'yN': "['D', 'N', 'N']",
    },
}

_CONFIG_TEMPLATE = """
options:
    output: /tmp/inspect_conn
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


def make_solver():
    bc_cfg = BC_CONFIGS[bc]
    config = _CONFIG_TEMPLATE.format(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
    )
    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()
    return solver


def collect_pairs(gi, res_element, var_element, TO, FROM):
    """Collect all (res_inner_idx, var_padded_idx) pairs from the element loop."""
    pair_count = defaultdict(int)
    for sq_idx in range(gi.nb_sq):
        res_nodes = TO[sq_idx]
        var_nodes = FROM[sq_idx]
        for tri_idx in (0, 1):
            res_tri = res_nodes[res_element.idx_to_std[tri_idx]]
            var_tri = var_nodes[var_element.idx_to_std[tri_idx]]
            for vnode in var_tri:
                for rnode in res_tri:
                    if rnode >= 0 and vnode >= 0:
                        pair_count[(int(rnode), int(vnode))] += 1
    return pair_count


def print_connectivity(pair_count, res_name, var_name, Nx_res, Nx_var):
    """Print forward and reverse connectivity tables."""
    # Forward: var -> res
    print(f'\n{"="*60}')
    print(f'{var_name} -> {res_name} connectivity')
    print(f'{"="*60}')

    var_to_res = defaultdict(set)
    for (rnode, vnode), count in pair_count.items():
        r_ix, r_iy = rnode % Nx_res, rnode // Nx_res
        var_to_res[vnode].add((r_ix, r_iy))

    for vnode in sorted(var_to_res.keys()):
        targets = sorted(var_to_res[vnode])
        target_str = '  '.join(f'({x},{y})' for x, y in targets)
        print(f'  {var_name}[{vnode:2d}] -> {res_name}: {target_str}')

    # Reverse: res <- var
    print(f'\n{"="*60}')
    print(f'{res_name} <- {var_name} connectivity')
    print(f'{"="*60}')

    res_from_var = defaultdict(set)
    for (rnode, vnode), count in pair_count.items():
        res_from_var[rnode].add(vnode)

    for rnode in sorted(res_from_var.keys()):
        r_ix, r_iy = rnode % Nx_res, rnode // Nx_res
        sources = sorted(res_from_var[rnode])
        print(f'  {res_name}({r_ix},{r_iy}) [idx={rnode}] <- {var_name}: '
              f'{sources}  ({len(sources)} connections)')


if __name__ == '__main__':
    solver = make_solver()
    gi = solver.grid_idx
    elem = solver.elements
    P1 = elem.P1
    P2 = elem.P2

    Nx_p, Ny_p = gi.Nx_p_inner, gi.Ny_p_inner
    Nx_P2, Ny_P2 = gi.Nx_v_inner, gi.Ny_v_inner

    print(f'Grid: {Nx}x{Ny}, bc={bc}')
    print(f'P1 inner: {Nx_p}x{Ny_p},  P2 inner: {Nx_P2}x{Ny_P2}')
    print(f'Squares: {gi.sq_per_row}x{gi.sq_per_col} = {gi.nb_sq}')

    # Index masks
    print(f'\nP1 inner mask (rows=x, cols=y):')
    print(gi.index_mask_inner_local_P1)
    print(f'\nP2 inner mask:')
    print(gi.index_mask_inner_local_P2)

    # ---- v->p block: jx -> mass (e.g. R11x) ----
    print(f'\n{"#"*60}')
    print(f'# v->p block: jx -> mass')
    print(f'{"#"*60}')

    TO_p = gi.sq_TO_inner_P1
    FROM_P2 = gi.sq_FROM_padded_P2('jx')
    vp_pairs = collect_pairs(gi, P1, P2, TO_p, FROM_P2)
    print_connectivity(vp_pairs, 'mass', 'jx', Nx_p, Nx_P2)

    # ---- p->v block: rho -> momentum_x (e.g. R21x) ----
    print(f'\n{"#"*60}')
    print(f'# p->v block: rho -> momentum_x')
    print(f'{"#"*60}')

    TO_P2 = gi.sq_TO_inner_P2
    FROM_p = gi.sq_FROM_padded_P1('rho')
    pv_pairs = collect_pairs(gi, P2, P1, TO_P2, FROM_p)
    print_connectivity(pv_pairs, 'mom_x', 'rho', Nx_P2, Nx_p)
