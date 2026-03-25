"""Standalone test: sq_TO_inner and sq_FROM_padded node mapping.

Verifies that for every square and triangle:
  - sq_TO_inner_p/v returns the correct inner-node indices
  - sq_FROM_padded_p/v returns the correct contributor indices
  - idx_to_std selects the right triangle corners

Run:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/fem_2d/tests/test_sq_node_mapping.py

Produces: sq_node_mapping.png in the current directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from GaPFlow.fem_2d.elements import TaylorHoodP2P1
from GaPFlow.fem_2d.grid_index import GridIndexManager


# =============================================================================
# Stub decomp (Dirichlet, serial, non-periodic)
# =============================================================================

class StubDecomp:
    def __init__(self, Nx_p, Ny_p):
        Nx_v, Ny_v = 2*Nx_p - 1, 2*Ny_p - 1
        self.periodic_x = False
        self.periodic_y = False
        self.is_at_xW = True
        self.is_at_xE = True
        self.is_at_yS = True
        self.is_at_yN = True
        nb_vars = 3
        self.grid = {
            'bc_xW': ['D'] * nb_vars, 'bc_xE': ['D'] * nb_vars,
            'bc_yS': ['D'] * nb_vars, 'bc_yN': ['D'] * nb_vars,
        }
        self.nb_subdomain_grid_pts = (Nx_p, Ny_p)
        self.nb_subdomain_grid_pts_v = (Nx_v, Ny_v)
        mp = np.full((Nx_p + 2, Ny_p + 2), -1, dtype=np.int32)
        mp[1:-1, 1:-1] = np.arange(Nx_p * Ny_p).reshape(Nx_p, Ny_p, order='F')
        self.index_mask_padded_global = mp
        mv = np.full((Nx_v + 4, Ny_v + 4), -1, dtype=np.int32)
        mv[2:-2, 2:-2] = np.arange(Nx_v * Ny_v).reshape(Nx_v, Ny_v, order='F')
        self.index_mask_padded_global_v = mv

    @property
    def bc_at_W(self): return self.is_at_xW and not self.periodic_x
    @property
    def bc_at_E(self): return self.is_at_xE and not self.periodic_x
    @property
    def bc_at_S(self): return self.is_at_yS and not self.periodic_y
    @property
    def bc_at_N(self): return self.is_at_yN and not self.periodic_y
    @property
    def has_full_x(self): return True
    @property
    def has_full_y(self): return True


# =============================================================================
# Analytic reference: build expected sq_TO_inner_p from scratch
# =============================================================================

def expected_sq_to_inner_p(gi):
    """Ground truth: for each square, look up the 4 P1 corner inner indices.

    Corner layout:
      tl=(sx, sy+1)  tr=(sx+1, sy+1)
      bl=(sx, sy  )  br=(sx+1, sy  )
    Returns shape (nb_sq, 4) with order [bl, br, tl, tr].
    """
    m = gi.index_mask_inner_local_p
    result = np.empty((gi.nb_sq, 4), dtype=np.int32)
    for sq_idx in range(gi.nb_sq):
        sx = gi.sq_x_arr_p[sq_idx]
        sy = gi.sq_y_arr_p[sq_idx]
        result[sq_idx] = [
            m[sx,     sy    ],   # bl
            m[sx + 1, sy    ],   # br
            m[sx,     sy + 1],   # tl
            m[sx + 1, sy + 1],   # tr
        ]
    return result


def expected_sq_from_padded_p(gi, var='rho'):
    """Ground truth: for each square, look up the 4 P1 corner contributor indices."""
    m = gi.index_mask_padded_local_p(var)
    result = np.empty((gi.nb_sq, 4), dtype=np.int32)
    for sq_idx in range(gi.nb_sq):
        sx = gi.sq_x_arr_p[sq_idx]
        sy = gi.sq_y_arr_p[sq_idx]
        result[sq_idx] = [
            m[sx,     sy    ],
            m[sx + 1, sy    ],
            m[sx,     sy + 1],
            m[sx + 1, sy + 1],
        ]
    return result


def expected_sq_to_inner_v(gi):
    """Ground truth: for each square, look up the 9 P2 node inner indices.

    Node layout (fine-grid offsets from sq origin (sx_v, sy_v)):
      idx  fine offset  name
       0   (0,0)        bl
       1   (2,0)        br
       2   (0,2)        tl
       3   (2,2)        tr
       4   (0,1)        ml
       5   (1,0)        bm
       6   (1,1)        mm
       7   (1,2)        tm
       8   (2,1)        mr
    """
    m = gi.index_mask_inner_local_v
    offsets = [(0,0),(2,0),(0,2),(2,2),(0,1),(1,0),(1,1),(1,2),(2,1)]
    result = np.empty((gi.nb_sq, 9), dtype=np.int32)
    for sq_idx in range(gi.nb_sq):
        sx = gi.sq_x_arr_v[sq_idx]
        sy = gi.sq_y_arr_v[sq_idx]
        for k, (dx, dy) in enumerate(offsets):
            result[sq_idx, k] = m[sx + dx, sy + dy]
    return result


def expected_sq_from_padded_v(gi, var='jx'):
    """Ground truth: for each square, look up the 9 P2 node contributor indices."""
    m = gi.index_mask_padded_local_v(var)
    offsets = [(0,0),(2,0),(0,2),(2,2),(0,1),(1,0),(1,1),(1,2),(2,1)]
    result = np.empty((gi.nb_sq, 9), dtype=np.int32)
    for sq_idx in range(gi.nb_sq):
        sx = gi.sq_x_arr_v[sq_idx]
        sy = gi.sq_y_arr_v[sq_idx]
        for k, (dx, dy) in enumerate(offsets):
            result[sq_idx, k] = m[sx + dx, sy + dy]
    return result


# =============================================================================
# Triangle node extraction: the lines under test
# =============================================================================

def extract_tri_nodes(sq_to_nodes, idx_to_std, sq_idx, tri_idx):
    """Reproduce the two lines from _build_nnz / _build_nnz_res:

        res_nodes_on_sq  = res_sq_to_nodes[sq_idx]
        res_nodes_on_tri = res_nodes_on_sq[res_element.idx_to_std[tri_idx]]

    Returns the node indices for the triangle.
    """
    nodes_on_sq  = sq_to_nodes[sq_idx]
    nodes_on_tri = nodes_on_sq[idx_to_std[tri_idx]]
    return nodes_on_tri


# =============================================================================
# Verification
# =============================================================================

def verify_p1(gi, element):
    print('\n--- P1 sq_TO_inner_p ---')
    ref = expected_sq_to_inner_p(gi)
    actual = gi.sq_TO_inner_p
    match = np.array_equal(ref, actual)
    print(f'  sq_TO_inner_p matches reference: {match}')
    if not match:
        for sq_idx in range(gi.nb_sq):
            if not np.array_equal(ref[sq_idx], actual[sq_idx]):
                print(f'  sq {sq_idx}: ref={ref[sq_idx]}  actual={actual[sq_idx]}')

    print('\n--- P1 sq_FROM_padded_p (rho, Dirichlet) ---')
    ref_f = expected_sq_from_padded_p(gi, 'rho')
    actual_f = gi.sq_FROM_padded_p('rho')
    match_f = np.array_equal(ref_f, actual_f)
    print(f'  sq_FROM_padded_p matches reference: {match_f}')
    if not match_f:
        for sq_idx in range(gi.nb_sq):
            if not np.array_equal(ref_f[sq_idx], actual_f[sq_idx]):
                print(f'  sq {sq_idx}: ref={ref_f[sq_idx]}  actual={actual_f[sq_idx]}')

    print('\n--- P1 triangle node extraction ---')
    idx_to_std = element.P1.idx_to_std
    ok = True
    for sq_idx in range(gi.nb_sq):
        for tri_idx in range(2):
            tri_nodes = extract_tri_nodes(actual, idx_to_std, sq_idx, tri_idx)
            tri_nodes_ref = extract_tri_nodes(ref, idx_to_std, sq_idx, tri_idx)
            if not np.array_equal(tri_nodes, tri_nodes_ref):
                print(f'  MISMATCH sq={sq_idx} tri={tri_idx}: '
                      f'ref={tri_nodes_ref}  actual={tri_nodes}')
                ok = False
    if ok:
        print('  All triangle node extractions match reference.')

    print('\n--- P1 triangle: inner nodes are a subset of sq corners ---')
    ok2 = True
    for sq_idx in range(gi.nb_sq):
        sq_x, sq_y = gi.sq_x_arr_p[sq_idx], gi.sq_y_arr_p[sq_idx]
        for tri_idx in range(2):
            inner_tri = extract_tri_nodes(gi.sq_TO_inner_p, idx_to_std, sq_idx, tri_idx)
            from_tri  = extract_tri_nodes(gi.sq_FROM_padded_p('rho'), idx_to_std, sq_idx, tri_idx)
            # inner nodes with index >= 0 must correspond to FROM nodes at same position
            for k in range(3):
                if inner_tri[k] >= 0 and from_tri[k] >= 0:
                    # inner and from at same position should agree for non-ghost nodes
                    inner_pos = np.argwhere(gi.index_mask_inner_local_p == inner_tri[k])
                    from_pos  = np.argwhere(gi.index_mask_padded_local_p('rho') == from_tri[k])
                    if inner_pos.size > 0 and from_pos.size > 0:
                        if not np.any(np.all(inner_pos == from_pos, axis=1)):
                            print(f'  MISMATCH sq={sq_idx} tri={tri_idx} k={k}: '
                                  f'inner_node={inner_tri[k]} at {inner_pos}, '
                                  f'from_node={from_tri[k]} at {from_pos}')
                            ok2 = False
    if ok2:
        print('  Inner/FROM positions are consistent for all (sq, tri, k).')

    return match and match_f


def verify_p2(gi, element):
    print('\n--- P2 sq_TO_inner_v ---')
    ref = expected_sq_to_inner_v(gi)
    actual = gi.sq_TO_inner_v
    match = np.array_equal(ref, actual)
    print(f'  sq_TO_inner_v matches reference: {match}')
    if not match:
        for sq_idx in range(gi.nb_sq):
            if not np.array_equal(ref[sq_idx], actual[sq_idx]):
                print(f'  sq {sq_idx}: ref={ref[sq_idx]}  actual={actual[sq_idx]}')

    print('\n--- P2 sq_FROM_padded_v (jx, Dirichlet) ---')
    ref_f = expected_sq_from_padded_v(gi, 'jx')
    actual_f = gi.sq_FROM_padded_v('jx')
    match_f = np.array_equal(ref_f, actual_f)
    print(f'  sq_FROM_padded_v matches reference: {match_f}')
    if not match_f:
        for sq_idx in range(gi.nb_sq):
            if not np.array_equal(ref_f[sq_idx], actual_f[sq_idx]):
                print(f'  sq {sq_idx}: ref={ref_f[sq_idx]}  actual={actual_f[sq_idx]}')

    print('\n--- P2 triangle node extraction ---')
    idx_to_std = element.P2.idx_to_std
    ok = True
    for sq_idx in range(gi.nb_sq):
        for tri_idx in range(2):
            tri_nodes = extract_tri_nodes(actual, idx_to_std, sq_idx, tri_idx)
            tri_nodes_ref = extract_tri_nodes(ref, idx_to_std, sq_idx, tri_idx)
            if not np.array_equal(tri_nodes, tri_nodes_ref):
                print(f'  MISMATCH sq={sq_idx} tri={tri_idx}: '
                      f'ref={tri_nodes_ref}  actual={tri_nodes}')
                ok = False
    if ok:
        print('  All P2 triangle node extractions match reference.')

    return match and match_f


# =============================================================================
# Plot
# =============================================================================

def _p1_node_phys_pos(gi, node_idx):
    """Physical (x, y) position of P1 inner node node_idx (using unit cell)."""
    pos = np.argwhere(gi.index_mask_inner_local_p == node_idx)
    if len(pos) == 0:
        return None
    ix, iy = pos[0]
    return (ix - 1, iy - 1)   # shift so inner region starts at (0,0)


def _p2_node_phys_pos(gi, node_idx):
    """Physical (x, y) position of P2 inner node node_idx (fine grid, unit half-cell)."""
    pos = np.argwhere(gi.index_mask_inner_local_v == node_idx)
    if len(pos) == 0:
        return None
    ix, iy = pos[0]
    return ((ix - 2) * 0.5, (iy - 2) * 0.5)


def _draw_p1_panel(ax, gi, element, Nx_p, Ny_p, highlight_sq, tri_idx):
    """Draw one P1 panel: all squares, highlighted square + triangle."""
    dx_p = 1.0
    dy_p = 1.0
    idx_to_std_p1 = element.P1.idx_to_std

    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.set_title(f'P1 (coarse)  sq{highlight_sq}  tri{tri_idx}', fontsize=10)

    # All squares
    for sq_i in range(gi.nb_sq):
        sx, sy = gi.sq_x_arr_p[sq_i], gi.sq_y_arr_p[sq_i]
        x0, y0 = (sx - 1) * dx_p, (sy - 1) * dy_p
        color = '#d0e8ff' if sq_i == highlight_sq else 'white'
        ax.add_patch(plt.Rectangle((x0, y0), dx_p, dy_p,
                                   facecolor=color, edgecolor='gray', lw=0.8))
        ax.text(x0 + dx_p/2, y0 + dy_p/2, f'sq{sq_i}',
                ha='center', va='center', fontsize=7, color='gray')

    # Highlighted triangle
    sx, sy = gi.sq_x_arr_p[highlight_sq], gi.sq_y_arr_p[highlight_sq]
    sq_corners = np.array([
        [(sx - 1)*dx_p, (sy - 1)*dy_p],   # bl
        [(sx    )*dx_p, (sy - 1)*dy_p],   # br
        [(sx - 1)*dx_p, (sy    )*dy_p],   # tl
        [(sx    )*dx_p, (sy    )*dy_p],   # tr
    ])
    tri_xy = sq_corners[idx_to_std_p1[tri_idx]]
    ax.add_patch(plt.Polygon(tri_xy, closed=True,
                             facecolor='#aaddaa', edgecolor='green', lw=2, alpha=0.7))
    ax.text(tri_xy[:, 0].mean(), tri_xy[:, 1].mean(), f'tri{tri_idx}',
            ha='center', va='center', fontsize=8, color='darkgreen', fontweight='bold')

    # All P1 nodes
    m_inner = gi.index_mask_inner_local_p
    m_pad   = gi.index_mask_padded_local_p('rho')
    for ix in range(m_inner.shape[0]):
        for iy in range(m_inner.shape[1]):
            xp, yp = (ix - 1)*dx_p, (iy - 1)*dy_p
            ii, pi = m_inner[ix, iy], m_pad[ix, iy]
            if ii >= 0:
                ax.plot(xp, yp, 'ko', ms=7, zorder=5)
                ax.text(xp + 0.07, yp + 0.07, f'{ii}', fontsize=8,
                        color='black', fontweight='bold')
            elif pi >= 0:
                ax.plot(xp, yp, 'bs', ms=5, zorder=4)
                ax.text(xp + 0.07, yp + 0.07, f'({pi})', fontsize=6, color='blue')
            else:
                ax.plot(xp, yp, 'rx', ms=5, zorder=3)

    # Node labels on triangle corners
    inner_tri = gi.sq_TO_inner_p[highlight_sq][idx_to_std_p1[tri_idx]]
    from_tri  = gi.sq_FROM_padded_p('rho')[highlight_sq][idx_to_std_p1[tri_idx]]
    for k, (xc, yc) in enumerate(tri_xy):
        ax.annotate(f'N{k}\nin={inner_tri[k]}\nfr={from_tri[k]}',
                    xy=(xc, yc), xytext=(xc + 0.15, yc + 0.18),
                    fontsize=7, color='darkgreen',
                    arrowprops=dict(arrowstyle='-', color='green', lw=0.8))

    ax.set_xlim(-1.3*dx_p, (Nx_p + 0.3)*dx_p)
    ax.set_ylim(-1.3*dy_p, (Ny_p + 0.3)*dy_p)
    for v in (0, Nx_p*dx_p):
        ax.axvline(v, color='k', lw=0.5, ls='--')
    for v in (0, Ny_p*dy_p):
        ax.axhline(v, color='k', lw=0.5, ls='--')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(handles=[
        mpatches.Patch(facecolor='#d0e8ff', edgecolor='gray', label=f'sq{highlight_sq}'),
        mpatches.Patch(facecolor='#aaddaa', edgecolor='green', label=f'tri{tri_idx}'),
        plt.Line2D([0],[0], marker='o', color='k', ls='', label='inner P1'),
        plt.Line2D([0],[0], marker='s', color='b', ls='', label='padded'),
        plt.Line2D([0],[0], marker='x', color='r', ls='', label='Dirichlet'),
    ], fontsize=7, loc='upper right')


def _draw_p2_panel(ax, gi, element, Nx_p, Ny_p, highlight_sq, tri_idx):
    """Draw one P2 panel: fine grid, highlighted square + triangle."""
    dx_p = 1.0
    dy_p = 1.0
    dx_v = 0.5
    dy_v = 0.5
    idx_to_std_p2 = element.P2.idx_to_std
    p2_offsets = [(0,0),(2,0),(0,2),(2,2),(0,1),(1,0),(1,1),(1,2),(2,1)]
    p2_names   = ['bl','br','tl','tr','ml','bm','mm','tm','mr']

    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.set_title(f'P2 (fine)  sq{highlight_sq}  tri{tri_idx}', fontsize=10)

    # Coarse square outlines
    for sq_i in range(gi.nb_sq):
        sx, sy = gi.sq_x_arr_p[sq_i], gi.sq_y_arr_p[sq_i]
        ax.add_patch(plt.Rectangle(((sx-1)*dx_p, (sy-1)*dy_p), dx_p, dy_p,
                                   facecolor='none', edgecolor='lightgray',
                                   lw=0.5, ls='--'))

    # Highlighted P2 square
    sx_v = gi.sq_x_arr_v[highlight_sq]
    sy_v = gi.sq_y_arr_v[highlight_sq]
    x0_v = (sx_v - 2) * dx_v
    y0_v = (sy_v - 2) * dy_v
    ax.add_patch(plt.Rectangle((x0_v, y0_v), dx_p, dy_p,
                                facecolor='#d0e8ff', edgecolor='blue', lw=1.5, alpha=0.4))

    sq_nodes_phys = np.array([
        (x0_v + ox*dx_v, y0_v + oy*dy_v) for ox, oy in p2_offsets
    ])

    # Highlighted P2 triangle
    tri_xy_v = sq_nodes_phys[idx_to_std_p2[tri_idx]]
    ax.add_patch(plt.Polygon(tri_xy_v, closed=True,
                             facecolor='#aaddaa', edgecolor='green', lw=2, alpha=0.7))
    ax.text(tri_xy_v[:, 0].mean(), tri_xy_v[:, 1].mean(), f'tri{tri_idx}',
            ha='center', va='center', fontsize=8, color='darkgreen', fontweight='bold')

    # All P2 nodes
    m_inner_v = gi.index_mask_inner_local_v
    m_pad_v   = gi.index_mask_padded_local_v('jx')
    for ix in range(m_inner_v.shape[0]):
        for iy in range(m_inner_v.shape[1]):
            xp, yp = (ix - 2)*dx_v, (iy - 2)*dy_v
            ii, pi = m_inner_v[ix, iy], m_pad_v[ix, iy]
            if ii >= 0:
                ax.plot(xp, yp, 'ko', ms=5, zorder=5)
                ax.text(xp + 0.04, yp + 0.04, f'{ii}', fontsize=6,
                        color='black', fontweight='bold')
            elif pi >= 0:
                ax.plot(xp, yp, 'bs', ms=4, zorder=4)
                ax.text(xp + 0.04, yp + 0.04, f'({pi})', fontsize=5, color='blue')
            else:
                ax.plot(xp, yp, 'rx', ms=4, zorder=3)

    # Square node position labels
    for k, (xp, yp) in enumerate(sq_nodes_phys):
        ax.text(xp - 0.06, yp - 0.06, p2_names[k], fontsize=6,
                color='navy', ha='right', va='top')

    # Triangle node annotations
    inner_tri = gi.sq_TO_inner_v[highlight_sq][idx_to_std_p2[tri_idx]]
    from_tri  = gi.sq_FROM_padded_v('jx')[highlight_sq][idx_to_std_p2[tri_idx]]
    for k, (xc, yc) in enumerate(tri_xy_v):
        ax.annotate(f'N{k}\nin={inner_tri[k]}\nfr={from_tri[k]}',
                    xy=(xc, yc), xytext=(xc + 0.12, yc + 0.12),
                    fontsize=6, color='darkgreen',
                    arrowprops=dict(arrowstyle='-', color='green', lw=0.8))

    ax.set_xlim(-1.3*dx_p, (Nx_p + 0.3)*dx_p)
    ax.set_ylim(-1.3*dy_p, (Ny_p + 0.3)*dy_p)
    for v in (0, Nx_p*dx_p):
        ax.axvline(v, color='k', lw=0.5, ls='--')
    for v in (0, Ny_p*dy_p):
        ax.axhline(v, color='k', lw=0.5, ls='--')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(handles=[
        mpatches.Patch(facecolor='#d0e8ff', edgecolor='blue', label=f'sq{highlight_sq}'),
        mpatches.Patch(facecolor='#aaddaa', edgecolor='green', label=f'tri{tri_idx}'),
        plt.Line2D([0],[0], marker='o', color='k', ls='', label='inner P2'),
        plt.Line2D([0],[0], marker='s', color='b', ls='', label='padded'),
        plt.Line2D([0],[0], marker='x', color='r', ls='', label='Dirichlet'),
    ], fontsize=7, loc='upper right')


def plot_grids(gi, element, Nx_p, Ny_p, highlight_sq=0,
               out_file='sq_node_mapping.png'):
    """2×2 subplot grid: rows = tri0, tri1; cols = P1, P2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             facecolor='white')

    for tri_idx in range(2):
        _draw_p1_panel(axes[tri_idx, 0], gi, element, Nx_p, Ny_p,
                       highlight_sq, tri_idx)
        _draw_p2_panel(axes[tri_idx, 1], gi, element, Nx_p, Ny_p,
                       highlight_sq, tri_idx)

    fig.suptitle(f'Square {highlight_sq} node mapping  '
                 f'({Nx_p}×{Ny_p} inner P1,  {gi.Nx_v_inner}×{gi.Ny_v_inner} inner P2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=120, facecolor='white')
    print(f'\nPlot saved to: {out_file}')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Nx_p=3 -> 2x2 inner P1 squares, 5x5 inner P2 nodes
    Nx_p, Ny_p = 3, 3
    highlight_sq  = 4   # (x=1,y=1): interior square with all four P1 corners inner

    decomp   = StubDecomp(Nx_p, Ny_p)
    gi       = GridIndexManager(decomp, ['jx', 'jy', 'rho'])
    element  = TaylorHoodP2P1(dx=1.0, dy=1.0)

    print(f'Grid: {Nx_p}x{Ny_p} inner P1 nodes')
    print(f'  Nx_p_padded={gi.Nx_p_padded}, Ny_p_padded={gi.Ny_p_padded}')
    print(f'  sq_per_row={gi.sq_per_row}, sq_per_col={gi.sq_per_col}, nb_sq={gi.nb_sq}')
    print(f'  Nx_v_padded={gi.Nx_v_padded}, Ny_v_padded={gi.Ny_v_padded}')
    print()
    print('P1 inner mask (index_mask_inner_local_p):')
    print(gi.index_mask_inner_local_p)
    print()
    print('P1 padded mask (index_mask_padded_local_p, rho/Dirichlet):')
    print(gi.index_mask_padded_local_p('rho'))
    print()
    print('Square ordering:')
    for sq_idx in range(gi.nb_sq):
        sx, sy = gi.sq_x_arr_p[sq_idx], gi.sq_y_arr_p[sq_idx]
        print(f'  sq{sq_idx}: (x={sx}, y={sy})  '
              f'TO_inner={gi.sq_TO_inner_p[sq_idx]}  '
              f'FROM_pad={gi.sq_FROM_padded_p("rho")[sq_idx]}')
    print()
    print('P1 idx_to_std:', element.P1.idx_to_std)
    print('P2 idx_to_std:', element.P2.idx_to_std)
    print()

    print('Highlighted square+triangle:')
    sx, sy = gi.sq_x_arr_p[highlight_sq], gi.sq_y_arr_p[highlight_sq]
    print(f'  sq{highlight_sq}: x={sx}, y={sy}')
    for tri_idx in range(2):
        inner_tri = gi.sq_TO_inner_p[highlight_sq][element.P1.idx_to_std[tri_idx]]
        from_tri  = gi.sq_FROM_padded_p('rho')[highlight_sq][element.P1.idx_to_std[tri_idx]]
        print(f'  P1 tri{tri_idx}: inner_nodes={inner_tri}  from_nodes={from_tri}')
    for tri_idx in range(2):
        inner_tri = gi.sq_TO_inner_v[highlight_sq][element.P2.idx_to_std[tri_idx]]
        from_tri  = gi.sq_FROM_padded_v('jx')[highlight_sq][element.P2.idx_to_std[tri_idx]]
        print(f'  P2 tri{tri_idx}: inner_nodes={inner_tri}  from_nodes={from_tri}')

    # Verification
    ok_p1 = verify_p1(gi, element)
    ok_p2 = verify_p2(gi, element)
    print(f'\n=== RESULT: P1 arrays correct={ok_p1},  P2 arrays correct={ok_p2} ===')

    # Plot
    plot_grids(gi, element, Nx_p, Ny_p,
               highlight_sq=highlight_sq,
               out_file='sq_node_mapping.png')
