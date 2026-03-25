"""
Benchmark: P2 interpolation to quadrature points.

Two approaches compared:
  - muGrid:    GenericLinearOperator convolution on the fine nodal grid,
               results extracted at even cells (2i, 2j).
  - numpy:     Explicit gather of the 6 local nodes per cell via advanced
               indexing, then einsum with the N matrix. No Python loops.

Both produce output shape (nb_cells_x, nb_cells_y, 2*nb_quad).

Results are verified to agree before timing.
"""

import time
import numpy as np
import muGrid
from GaPFlow.fem_2d_new.elements import TaylorHoodP2P1

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------

def make_fields(nx, ny):
    """Create muGrid field collection for a grid of nx*ny P2 cells.

    The fine nodal grid has (2*nx+1) x (2*ny+1) nodes. We allocate
    (2*nx+1) x (2*ny+1) pixels with 1 ghost on each right side so the
    stencil (max offset 2) can reach the last column/row.
    """
    fc = muGrid.GlobalFieldCollection(
        [2 * nx + 1, 2 * ny + 1],
        nb_sub_pts={'quad': 6},
        nb_ghosts_left=[0, 0],
        nb_ghosts_right=[2, 2],
    )
    nf = fc.real_field('nodal', sub_pt='pixel')
    qf = fc.real_field('quad',  sub_pt='quad')
    return fc, nf, qf


MUGRID_TO_N_PERM = [0, 2, 1, 3, 5, 4]  # reorder muGrid sub-pts to match N[q,k] ordering

def mugrid_interp(op, nf, qf, nx, ny):
    """Apply muGrid operator, extract even cells, reorder sub-pts to match N ordering."""
    op.apply(nf, qf)
    # Even cells (2i, 2j) are the P2 cell origins; last node row/col excluded
    return qf.p[MUGRID_TO_N_PERM, :2*nx:2, :2*ny:2]   # shape (6, nx, ny)


def numpy_interp(nodal, N, idx_to_std, offsets, nx, ny):
    """Gather local node values and contract with N.

    nodal : (2*nx+ghost, 2*ny+ghost) fine grid values
    N     : (nb_quad, nb_nodes)  shape functions at quad pts
    Returns (6, nx, ny)
    """
    nb_q, nb_nodes = N.shape
    nb_tri = 2

    # Gather: node_vals[t, k, row, col] = nodal[2*row + oy, 2*col + ox]
    # nodal axes are (row, col) = (y, x); muGrid field axes match this convention.
    node_vals = np.empty((nb_tri, nb_nodes, ny, nx))
    for t in range(nb_tri):
        for k in range(nb_nodes):
            sq = idx_to_std[t, k]
            ox, oy = offsets[sq]
            node_vals[t, k] = nodal[oy:2*ny+oy:2, ox:2*nx+ox:2]  # (ny, nx) matching muGrid axes

    # Contract: result[t*nb_q + q, row, col] = sum_k N[q,k] * node_vals[t,k,row,col]
    result = np.einsum('qk, tkrc -> tqrc', N, node_vals)  # (2, nb_q, ny, nx)
    return result.reshape(nb_tri * nb_q, ny, nx)


# ---------------------------------------------------------------
# Correctness check on a small grid
# ---------------------------------------------------------------

def check_correctness(nx=4, ny=4):
    elem = TaylorHoodP2P1(dx=1.0, dy=1.0)
    _, nf, qf = make_fields(nx, ny)

    np.random.seed(0)
    nf.p[:] = np.random.rand(*nf.p.shape)

    ref   = mugrid_interp(elem.P2.interpolation_operator, nf, qf, nx, ny)
    numpy_res = numpy_interp(
        nf.p, elem.P2.N,
        elem.P2.idx_to_std, elem.P2.square_node_offsets,
        nx, ny,
    )

    max_diff = np.max(np.abs(ref - numpy_res))
    print(f"Correctness check ({nx}x{ny} cells): max |diff| = {max_diff:.2e}",
          "✓" if max_diff < 1e-12 else "✗ MISMATCH")


# ---------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------

def benchmark(nx=512, ny=512, repeats=20):
    elem = TaylorHoodP2P1(dx=1.0, dy=1.0)
    _, nf, qf = make_fields(nx, ny)
    np.random.seed(1)
    nf.p[:] = np.random.rand(*nf.p.shape)
    nodal = nf.p.copy()

    op  = elem.P2.interpolation_operator
    N   = elem.P2.N
    idt = elem.P2.idx_to_std
    off = elem.P2.square_node_offsets

    # Warm up
    mugrid_interp(op, nf, qf, nx, ny)
    numpy_interp(nodal, N, idt, off, nx, ny)

    t0 = time.perf_counter()
    for _ in range(repeats):
        mugrid_interp(op, nf, qf, nx, ny)
    t_mugrid = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        numpy_interp(nodal, N, idt, off, nx, ny)
    t_numpy = (time.perf_counter() - t0) / repeats

    print(f"\nBenchmark ({nx}x{ny} cells, {repeats} runs each):")
    print(f"  muGrid (conv + extract):  {t_mugrid*1e3:.2f} ms")
    print(f"  numpy  (gather + einsum): {t_numpy*1e3:.2f} ms")
    print(f"  speedup (numpy/muGrid):   {t_numpy/t_mugrid:.2f}x")


if __name__ == '__main__':
    check_correctness(nx=10, ny=10)
    benchmark(nx=2000, ny=2000, repeats=10)
