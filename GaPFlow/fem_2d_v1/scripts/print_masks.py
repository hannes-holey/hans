"""Print P1 and P2 index masks for a 4x4 grid.

Edit the BC flags at the top to test different configurations.
"""

import numpy as np
from functools import lru_cache, cached_property

# =============================================================================
# Configuration — edit here
# =============================================================================

Nx = 4   # number of P1 cells in x
Ny = 4   # number of P1 cells in y

# Boundary condition flags (True = this boundary exists and is non-periodic)
periodic_x = False
periodic_y = True

# Dirichlet flags per boundary (only relevant when not periodic)
dirichlet_xW = True
dirichlet_xE = True
dirichlet_yS = True   # no-slip south wall
dirichlet_yN = True   # no-slip north wall

# Neumann flags (forwarding ghost → inner)
neumann_xW = False
neumann_xE = False
neumann_yS = False
neumann_yN = False

# =============================================================================
# Derived sizes
# =============================================================================

# Ghost depth: P1=1, P2=2
# Padded sizes
Nx_p_inner = Nx
Ny_p_inner = Ny
Nx_p_padded = Nx_p_inner + 2     # ghost depth 1 on each side
Ny_p_padded = Ny_p_inner + 2

Nx_v_inner = 2 * Nx - 1          # P2 fine grid inner (2*Nx-1 nodes)
Ny_v_inner = 2 * Ny - 1
Nx_v_padded = Nx_v_inner + 4     # ghost depth 2 on each side
Ny_v_padded = Ny_v_inner + 4

bc_at_W = not periodic_x
bc_at_E = not periodic_x
bc_at_S = not periodic_y
bc_at_N = not periodic_y

# =============================================================================
# P1 masks
# =============================================================================

def make_inner_p():
    mask = np.full((Nx_p_padded, Ny_p_padded), -1, dtype=np.int32)
    nb = Nx_p_inner * Ny_p_inner
    mask[1:-1, 1:-1] = np.arange(nb).reshape((Nx_p_inner, Ny_p_inner), order='F')
    return mask

def make_padded_p(neumann_W=neumann_xW, neumann_E=neumann_xE,
                  neumann_S=neumann_yS, neumann_N=neumann_yN):
    mask = make_inner_p()

    # Periodic wrapping
    if periodic_x:
        mask[0, :]  = mask[Nx_p_padded - 2, :]
        mask[-1, :] = mask[1, :]
    if periodic_y:
        mask[:, 0]  = mask[:, Ny_p_padded - 2]
        mask[:, -1] = mask[:, 1]

    # Assign new indices to remaining ghost nodes that aren't Dirichlet
    cur = Nx_p_inner * Ny_p_inner
    for x in range(Nx_p_padded):
        for y in range(Ny_p_padded):
            if mask[x, y] == -1:
                is_dir = False
                if x == 0 and bc_at_W and dirichlet_xW: is_dir = True
                if x == Nx_p_padded-1 and bc_at_E and dirichlet_xE: is_dir = True
                if y == 0 and bc_at_S and dirichlet_yS: is_dir = True
                if y == Ny_p_padded-1 and bc_at_N and dirichlet_yN: is_dir = True
                if not is_dir:
                    mask[x, y] = cur
                    cur += 1

    # Neumann forwarding
    if bc_at_W and neumann_W:
        mask[0, :] = mask[1, :]
    if bc_at_E and neumann_E:
        mask[-1, :] = mask[-2, :]
    if bc_at_S and neumann_S:
        mask[:, 0] = mask[:, 1]
    if bc_at_N and neumann_N:
        mask[:, -1] = mask[:, -2]

    return mask

# =============================================================================
# P2 masks
# =============================================================================

def is_dirichlet_v(x, y):
    if x in (0, 1) and bc_at_W and dirichlet_xW: return True
    if x in (Nx_v_padded-1, Nx_v_padded-2) and bc_at_E and dirichlet_xE: return True
    if y in (0, 1) and bc_at_S and dirichlet_yS: return True
    if y in (Ny_v_padded-1, Ny_v_padded-2) and bc_at_N and dirichlet_yN: return True
    return False

def make_inner_v():
    mask = np.full((Nx_v_padded, Ny_v_padded), -1, dtype=np.int32)
    nb = Nx_v_inner * Ny_v_inner
    mask[2:-2, 2:-2] = np.arange(nb).reshape((Nx_v_inner, Ny_v_inner), order='F')
    return mask

def make_padded_v(neumann_W=neumann_xW, neumann_E=neumann_xE,
                  neumann_S=neumann_yS, neumann_N=neumann_yN):
    mask = make_inner_v()

    # Periodic wrapping (depth 2)
    if periodic_x:
        mask[0, :]  = mask[Nx_v_padded - 4, :]
        mask[1, :]  = mask[Nx_v_padded - 3, :]
        mask[-2, :] = mask[2, :]
        mask[-1, :] = mask[3, :]
    if periodic_y:
        mask[:, 0]  = mask[:, Ny_v_padded - 4]
        mask[:, 1]  = mask[:, Ny_v_padded - 3]
        mask[:, -2] = mask[:, 2]
        mask[:, -1] = mask[:, 3]

    # Assign new indices to remaining ghost nodes that aren't Dirichlet
    cur = Nx_v_inner * Ny_v_inner
    for x in range(Nx_v_padded):
        for y in range(Ny_v_padded):
            if mask[x, y] == -1 and not is_dirichlet_v(x, y):
                mask[x, y] = cur
                cur += 1

    # Neumann forwarding (depth 2 → inner)
    if bc_at_W and neumann_W:
        mask[1, :] = mask[2, :]
        mask[0, :] = mask[2, :]
    if bc_at_E and neumann_E:
        mask[-2, :] = mask[-3, :]
        mask[-1, :] = mask[-3, :]
    if bc_at_S and neumann_S:
        mask[:, 1] = mask[:, 2]
        mask[:, 0] = mask[:, 2]
    if bc_at_N and neumann_N:
        mask[:, -2] = mask[:, -3]
        mask[:, -1] = mask[:, -3]

    return mask

# =============================================================================
# Print helper
# =============================================================================

def print_mask(name, mask):
    Nx_, Ny_ = mask.shape
    header = f'--- {name}  shape=({Nx_},{Ny_})  [x=col, y=row, origin bottom-left] ---'
    print(header)
    # Print y from top (Ny-1) to bottom (0) so the picture is spatially correct
    for y in range(Ny_ - 1, -1, -1):
        row = ' '.join(f'{mask[x, y]:3d}' for x in range(Nx_))
        print(f'  y={y:2d}: {row}')
    print()

# =============================================================================
# Run
# =============================================================================

print(f'Grid: Nx={Nx}, Ny={Ny}')
print(f'BCs: periodic_x={periodic_x}, periodic_y={periodic_y}')
print(f'     dirichlet yS={dirichlet_yS}, yN={dirichlet_yN}')
print(f'     neumann   yS={neumann_yS},  yN={neumann_yN}')
print()

print_mask('P1 inner (rho)',        make_inner_p())
print_mask('P1 padded (rho)',       make_padded_p())
print_mask('P2 inner (jx)',         make_inner_v())
print_mask('P2 padded (jx, Dirichlet yS/yN)', make_padded_v())
print_mask('P2 padded (jx, Neumann yS/yN)',   make_padded_v(neumann_S=True, neumann_N=True))
