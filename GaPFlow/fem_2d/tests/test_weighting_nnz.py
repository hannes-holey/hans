"""
Diagnostic: verify that _build_weighting and _build_nnz produce entries in
the same order — i.e., entry k of the shape weighting corresponds to the same
(res_node, var_node) pair as entry k of the nnz template.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/qd5728/fem_taylor_hood/GaPFlow')

from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2d
from GaPFlow.fem_2d.assembly import DOF_GRID

# =============================================================================
# Setup
# =============================================================================

CONFIG = """
options:
    output: /tmp/test_assembly
    write_freq: 1000
    silent: True
grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 4
    Ny: 3
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 1.0
    xW_D: 1.0
    yS_D: 1.0
    yN_D: 1.0
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

problem = Problem.from_string(CONFIG)
solver = problem.solver
solver.pre_run()

assembly = solver.assembly
grid_idx = solver.grid_idx
element = solver.elements

# The R1T term is mass/rho with deriv_combo='none' -> P1-P1 block
res = 'mass'
var = 'rho'
dc = 'none'
key = (res, var, dc)

sw = assembly.assembly_templates[key]['w']
nnz = assembly.assembly_templates[key]['nnz']
entries_per_quad = assembly.assembly_templates[key]['entries_per_quad']

res_element = element.P1
var_element = element.P1
nodes_tri_res = res_element.nodes_per_tri  # 3
nodes_tri_var = var_element.nodes_per_tri  # 3
n_tri = element.n_tri  # 2
quad_per_tri = element.Quadrature.nb_points  # 3
nb_quad_sq = n_tri * quad_per_tri  # 6
nb_sq = grid_idx.nb_sq

print("=" * 80)
print("BASIC DIMENSIONS")
print("=" * 80)
print(f"nodes_tri_res = {nodes_tri_res}")
print(f"nodes_tri_var = {nodes_tri_var}")
print(f"entries_per_quad = {entries_per_quad}  (= {nodes_tri_res} * {nodes_tri_var})")
print(f"n_tri = {n_tri}")
print(f"quad_per_tri = {quad_per_tri}")
print(f"nb_quad_sq = {nb_quad_sq}")
print(f"nb_sq = {nb_sq}")
print(f"sw.shape = {sw.shape}  (expected: {nb_quad_sq * entries_per_quad})")
print(f"nnz.shape = {nnz.shape}  (expected: {nb_sq * n_tri * entries_per_quad})")
print()

# =============================================================================
# Part 1: Shape function values for verification
# =============================================================================

print("=" * 80)
print("P1 SHAPE FUNCTIONS AT QUADRATURE POINTS")
print("=" * 80)
N_tri = res_element.N  # (3, 3)
weights = element.Quadrature.weights
area = element.sq_area
coords = element.Quadrature.coordinates

print(f"Quadrature coords: {coords}")
print(f"Quadrature weights: {weights}")
print(f"Square area: {area}")
print(f"N_tri (shape {N_tri.shape}):")
for q in range(quad_per_tri):
    print(f"  quad {q}: N = {N_tri[q]}")
print()

# =============================================================================
# Part 2: Examine sw (shape weighting) structure
# =============================================================================

print("=" * 80)
print("SHAPE WEIGHTING (sw) STRUCTURE")
print("=" * 80)
print(f"sw has {len(sw)} entries = {nb_quad_sq} quad_pts * {entries_per_quad} entries_per_quad")
print()

# Reshape to (nb_quad_sq, entries_per_quad) to see per-quad-point entries
sw_2d = sw.reshape(nb_quad_sq, entries_per_quad)

# For each quad point, show which (res_local, var_local) pair each entry maps to
# Within one quad point: flat index k = var_local * nodes_tri_res + res_local
print("Ordering within one quad point (k = var_local * nodes_tri_res + res_local):")
for k in range(entries_per_quad):
    var_local = k // nodes_tri_res
    res_local = k % nodes_tri_res
    print(f"  k={k}: res_local={res_local}, var_local={var_local}")
print()

# Show sw values per triangle (after quad summation)
print("sw values per triangle (summed over 3 quad points):")
for tri in range(n_tri):
    q_start = tri * quad_per_tri
    q_end = q_start + quad_per_tri
    sw_tri = sw_2d[q_start:q_end, :]  # (3, entries_per_quad)
    sw_tri_sum = sw_tri.sum(axis=0)   # (entries_per_quad,)
    print(f"  Triangle {tri} (quad points {q_start}..{q_end-1}):")
    for k in range(entries_per_quad):
        var_local = k // nodes_tri_res
        res_local = k % nodes_tri_res
        print(f"    k={k} (res_local={res_local}, var_local={var_local}): "
              f"sw_sum = {sw_tri_sum[k]:.10e}")
print()

# =============================================================================
# Part 3: Manually compute expected mass matrix entries
# =============================================================================

print("=" * 80)
print("MANUAL VERIFICATION: EXPECTED MASS MATRIX ENTRIES")
print("=" * 80)

# For tri0: N is evaluated at the standard quad points
# For tri1: N is tiled (same values as tri0 for 'none' deriv_combo)
N_full = np.tile(N_tri, (2, 1))  # (6, 3) — rows 0-2 = tri0, rows 3-5 = tri1

print("Expected integral for each (res_local, var_local) on triangle 0:")
print("  integral = sum_q w_q * area * N_res(q) * N_var(q)")
for res_local in range(nodes_tri_res):
    for var_local in range(nodes_tri_var):
        val = 0.0
        for q in range(quad_per_tri):
            val += weights[q] * area * N_tri[q, res_local] * N_tri[q, var_local]
        k = var_local * nodes_tri_res + res_local
        print(f"  res_local={res_local}, var_local={var_local} (k={k}): "
              f"expected = {val:.10e}")
print()

# Compare with sw
print("Comparison: sw (after quad sum) vs expected:")
for tri in range(n_tri):
    q_start = tri * quad_per_tri
    sw_tri_sum = sw_2d[q_start:q_start + quad_per_tri, :].sum(axis=0)
    print(f"  Triangle {tri}:")
    for k in range(entries_per_quad):
        var_local = k // nodes_tri_res
        res_local = k % nodes_tri_res
        # Expected: same for both triangles (N is tiled, weights are the same)
        expected = sum(weights[q] * area * N_tri[q, res_local] * N_tri[q, var_local]
                       for q in range(quad_per_tri))
        diff = sw_tri_sum[k] - expected
        print(f"    k={k} (r={res_local}, v={var_local}): "
              f"sw={sw_tri_sum[k]:.10e}  expected={expected:.10e}  diff={diff:.2e}")
print()

# =============================================================================
# Part 4: Examine nnz structure for specific squares and triangles
# =============================================================================

print("=" * 80)
print("NNZ STRUCTURE")
print("=" * 80)

# nnz shape: (nb_sq * n_tri * entries_per_quad,)
nnz_3d = nnz.reshape(nb_sq, n_tri, entries_per_quad)

# P1 square node connectivity
TO_p = grid_idx.sq_TO_inner_p   # (nb_sq, 4) — inner indices
FROM_p = grid_idx.sq_FROM_padded_p('rho')  # (nb_sq, 4) — padded indices

# P1 idx_to_std: which square nodes form each triangle
#   tri0: [0, 1, 2] -> bl, br, tl
#   tri1: [3, 2, 1] -> tr, tl, br
idx_to_std = res_element.idx_to_std
print(f"P1 idx_to_std:")
print(f"  tri0: {idx_to_std[0]} -> square nodes {idx_to_std[0]}")
print(f"  tri1: {idx_to_std[1]} -> square nodes {idx_to_std[1]}")
print()

# Show connectivity for a few squares
for sq_idx in range(min(3, nb_sq)):
    print(f"--- Square {sq_idx} ---")
    print(f"  TO_p (inner/res):   {TO_p[sq_idx]}")
    print(f"  FROM_p (padded/var): {FROM_p[sq_idx]}")
    for tri in range(n_tri):
        res_nodes_on_tri = TO_p[sq_idx][idx_to_std[tri]]
        var_nodes_on_tri = FROM_p[sq_idx][idx_to_std[tri]]
        print(f"  Triangle {tri}:")
        print(f"    res nodes (tri-local order): {res_nodes_on_tri}")
        print(f"    var nodes (tri-local order): {var_nodes_on_tri}")
        for k in range(entries_per_quad):
            var_local = k // nodes_tri_res
            res_local = k % nodes_tri_res
            res_node = res_nodes_on_tri[res_local]
            var_node = var_nodes_on_tri[var_local]
            nnz_val = nnz_3d[sq_idx, tri, k]
            print(f"    k={k} (r_loc={res_local}, v_loc={var_local}): "
                  f"res_node={res_node:3d}, var_node={var_node:3d} -> nnz_idx={nnz_val}")
    print()

# =============================================================================
# Part 5: Verify that nnz matches the (res, var) pairs from the weighting
# =============================================================================

print("=" * 80)
print("CONSISTENCY CHECK: nnz vs weighting ordering")
print("=" * 80)

# After assemble_matrix does:
#   quad_val_vec = np.repeat(res_quad_field.flatten(), entries_per_quad)
#   sw_vec = np.tile(sw, nb_sq)
#   q_vec = quad_val_vec * sw_vec
#   ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)
#   np.add.at(self._nnz_buf, nnz, ele_vec)
#
# ele_vec has shape (nb_sq * n_tri * entries_per_quad,) = (nb_sq * n_tri, entries_per_quad) flattened.
# nnz has the same shape.
#
# ele_vec[sq * n_tri * entries_per_quad + tri * entries_per_quad + k]
# should go to the same (res_node, var_node) as
# nnz[sq * n_tri * entries_per_quad + tri * entries_per_quad + k]

# Let's trace the quad summation manually.
# sw_vec = np.tile(sw, nb_sq), shape = (nb_sq * nb_quad_sq * entries_per_quad,)
# If we imagine res_quad_field = ones(nb_sq, nb_quad_sq), then
# quad_val_vec = ones(nb_sq * nb_quad_sq * entries_per_quad,)
# q_vec = sw_vec
# q_vec.reshape(-1, quad_per_tri, entries_per_quad) groups every 3 consecutive
# quad-point blocks.

print("Tracing the reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1) logic:")
print()

# sw_vec has structure: [sw for sq0, sw for sq1, ...]
# sw itself has structure: [q0_tri0, q1_tri0, q2_tri0, q0_tri1, q1_tri1, q2_tri1]
# Each qi has entries_per_quad entries.
# So sw_vec for one square = 6 * entries_per_quad entries
# After reshape(-1, 3, entries_per_quad):
#   block 0 (sq0): rows 0,1,2 = q0_tri0, q1_tri0, q2_tri0  -> sum = tri0 of sq0
#   block 1 (sq0): rows 3,4,5 = q0_tri1, q1_tri1, q2_tri1  -> sum = tri1 of sq0
#   block 2 (sq1): rows 0,1,2 of sq1 = q0_tri0, ...
#   etc.
# After sum(axis=1): shape = (nb_sq * n_tri, entries_per_quad)
# Flattened: [sq0_tri0_k0, sq0_tri0_k1, ..., sq0_tri1_k0, ..., sq1_tri0_k0, ...]

# This is the same order as nnz which is built as:
#   for sq in range(nb_sq):
#     for tri in (0, 1):
#       for var_node in var_nodes_on_tri:      # var slow
#         for res_node in res_nodes_on_tri:    # res fast
# i.e. [sq0_tri0_k0, sq0_tri0_k1, ..., sq0_tri1_k0, ..., sq1_tri0_k0, ...]

# The KEY question: does q_vec.reshape(-1, 3, entries_per_quad) correctly group
# quad points 0,1,2 as tri0 and quad points 3,4,5 as tri1?

# Let's verify by looking at the res_quad_field ordering.
# res_quad_field from get_quad has shape (nb_sq, nb_quad_sq).
# The quad ordering from the operator is stored as
# pg shape (nb_quad_sq, Ny_pad, Nx_pad), and get_quad does:
#   sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])
# So res_quad_field[sq, q] where q runs 0..5:
#   q=0: t0_q0, q=1: t0_q1, q=2: t0_q2, q=3: t1_q0, q=4: t1_q1, q=5: t1_q2
# (This is because the operator writes: out_pg[:, :ny, :nx] = result.reshape(nb_q*nb_tri, ny, nx)
#  where result shape before reshape is (nb_tri, nb_q, ny, nx) from the einsum.
#  reshape(nb_q*nb_tri, ny, nx) will interleave as t0_q0, t0_q1, t0_q2, t1_q0, ...
#  Actually: reshape from (2, 3, ny, nx) to (6, ny, nx) goes:
#  [t0_q0, t0_q1, t0_q2, t1_q0, t1_q1, t1_q2]
#  Yes — C-order reshape, so first axis (tri) is slowest.)

# After np.repeat(res_quad_field.flatten(), entries_per_quad):
# The flatten gives [sq0_q0, sq0_q1, sq0_q2, sq0_q3, sq0_q4, sq0_q5, sq1_q0, ...]
# repeat gives each quad value repeated entries_per_quad times.
# So the structure is:
# [sq0_q0, sq0_q0, ...(9x), sq0_q1, sq0_q1, ...(9x), ..., sq0_q5, ...(9x), sq1_q0, ...]
# Total length: nb_sq * nb_quad_sq * entries_per_quad

# np.tile(sw, nb_sq) gives [sw, sw, sw, ...] nb_sq times.
# sw has structure: [q0_t0, q1_t0, q2_t0, q0_t1, q1_t1, q2_t1] each with entries_per_quad entries
# Total length: nb_sq * nb_quad_sq * entries_per_quad

# q_vec = quad_val_vec * sw_vec
# q_vec.reshape(-1, quad_per_tri, entries_per_quad):
#   The total length is nb_sq * nb_quad_sq * entries_per_quad
#   = nb_sq * 6 * 9
#   reshape to (-1, 3, 9) gives shape (nb_sq * 2, 3, 9)
#   Blocks of 3 consecutive quad points:
#     Block 0: positions 0..26 = quad points q0,q1,q2 of sq0 = tri0 of sq0
#     Block 1: positions 27..53 = quad points q3,q4,q5 of sq0 = tri1 of sq0
#     Block 2: positions 54..80 = quad points q0,q1,q2 of sq1 = tri0 of sq1
#     ...

# So the ordering DOES match: ele_vec[i] and nnz[i] refer to the same (sq, tri, k).

# Let's verify this numerically by assembling with a known field.

print("NUMERICAL VERIFICATION: assemble with rho=1 everywhere")
print()

# With R1T (mass equation, rho term), evaluate returns just rho at quad points.
# If rho=1 everywhere, then res_quad_field = ones(nb_sq, nb_quad_sq).
# The assembled result should be: for each (res_node, var_node) pair,
# sum over all squares and triangles containing that pair of
# sum_q w_q * area * N_res(q) * N_var(q).

# For a periodic grid, every inner node is equivalent.
# The diagonal entry (i, i) should get contributions from all triangles containing node i.
# The off-diagonal entry (i, j) should get contributions from triangles containing both i and j.

# First, let's just verify the raw ele_vec vs nnz alignment.
# We'll manually do the assembly steps and check.

# Get the quad field values (should be rho at quad points)
qf = solver._build_all_quad_fields()
rho_quad = qf['rho']
print(f"rho quad field shape: {rho_quad.shape}")
print(f"rho quad field (first 2 squares):")
for sq in range(min(2, nb_sq)):
    print(f"  sq {sq}: {rho_quad[sq]}")
print()

# Manually replicate assemble_matrix logic
quad_val_vec = np.repeat(rho_quad.flatten(), entries_per_quad)
sw_vec = np.tile(sw, nb_sq)
q_vec = quad_val_vec * sw_vec
ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

print(f"ele_vec shape: {ele_vec.shape} = ({nb_sq * n_tri}, {entries_per_quad}) flattened")
print(f"nnz shape:     {nnz.shape}")
print()

# For each square and triangle, show the alignment
print("DETAILED ELEMENT-LEVEL COMPARISON (first 3 squares):")
print()

ele_vec_3d = ele_vec.reshape(nb_sq, n_tri, entries_per_quad)

for sq_idx in range(min(3, nb_sq)):
    print(f"--- Square {sq_idx} ---")
    for tri in range(n_tri):
        res_nodes_on_tri = TO_p[sq_idx][idx_to_std[tri]]
        var_nodes_on_tri = FROM_p[sq_idx][idx_to_std[tri]]
        print(f"  Triangle {tri}:")
        print(f"    res_nodes = {res_nodes_on_tri}, var_nodes = {var_nodes_on_tri}")
        for k in range(entries_per_quad):
            var_local = k // nodes_tri_res
            res_local = k % nodes_tri_res
            res_node = res_nodes_on_tri[res_local]
            var_node = var_nodes_on_tri[var_local]
            nnz_val = nnz_3d[sq_idx, tri, k]
            ele_val = ele_vec_3d[sq_idx, tri, k]

            # What does this nnz index map to in the COO pattern?
            # nnz_val is the flat index into the _nnz_buf
            block = assembly.block_order[(res, var)]
            nnz_start = block['nnz_idx_start']
            if 0 <= nnz_val < len(assembly.nnz_local_to):
                coo_res = assembly.nnz_local_to[nnz_val]
                coo_var = assembly.nnz_local_from[nnz_val]
            else:
                coo_res = -999
                coo_var = -999

            match = "OK" if (coo_res == res_node and coo_var == var_node) else "MISMATCH!"
            print(f"    k={k} (r_loc={res_local}, v_loc={var_local}): "
                  f"from_nnz: res={coo_res:3d}, var={coo_var:3d}  |  "
                  f"from_sq:  res={res_node:3d}, var={var_node:3d}  |  "
                  f"ele_val={ele_val:.8e}  |  {match}")
    print()

# =============================================================================
# Part 6: Check for over-counting on diagonal
# =============================================================================

print("=" * 80)
print("OVER-COUNTING CHECK: HOW MANY TIMES IS EACH (res, var) PAIR ACCUMULATED?")
print("=" * 80)

# Count how many times each nnz index appears in the nnz array
nnz_flat = nnz.flatten()
unique_nnz, counts = np.unique(nnz_flat[nnz_flat >= 0], return_counts=True)

print(f"Total nnz entries: {len(nnz_flat)}")
print(f"Unique nnz indices used: {len(unique_nnz)}")
print(f"Count distribution:")
for c in sorted(set(counts)):
    n = np.sum(counts == c)
    print(f"  count={c}: {n} nnz indices")
print()

# For a few specific nnz indices, show which (sq, tri) contribute
print("Detail for a few diagonal nnz indices:")
nb_inner_p = grid_idx.Nx_p_inner * grid_idx.Ny_p_inner
for inner_node in range(min(4, nb_inner_p)):
    # Find nnz index for diagonal entry (inner_node, inner_node)
    nnz_idx = assembly.lookup_nnz(res,
                                   np.array([inner_node], dtype=np.int32),
                                   var,
                                   np.array([inner_node], dtype=np.int32))
    if nnz_idx[0] < 0:
        print(f"  Node {inner_node}: diagonal not in sparsity pattern")
        continue

    nnz_i = nnz_idx[0]
    # Find all occurrences of this nnz index in the nnz array
    locations = np.argwhere(nnz_flat == nnz_i).flatten()
    # Decode to (sq, tri, k)
    decoded = []
    for loc in locations:
        sq = loc // (n_tri * entries_per_quad)
        rem = loc % (n_tri * entries_per_quad)
        tri = rem // entries_per_quad
        k = rem % entries_per_quad
        var_local = k // nodes_tri_res
        res_local = k % nodes_tri_res
        decoded.append((sq, tri, res_local, var_local, k))

    print(f"  Inner node {inner_node}: nnz_idx={nnz_i}, "
          f"appears {len(locations)} times:")
    for sq, tri, rl, vl, k in decoded:
        # Show the ele_vec value at this location
        ev = ele_vec_3d[sq, tri, k]
        print(f"    sq={sq}, tri={tri}, res_local={rl}, var_local={vl} (k={k}), "
              f"ele_val={ev:.8e}")

    total_accumulated = sum(ele_vec_3d[sq, tri, k] for sq, tri, _, _, k in decoded)
    print(f"    TOTAL accumulated: {total_accumulated:.8e}")
print()

# =============================================================================
# Part 7: Compare assembled matrix diagonal with expected
# =============================================================================

print("=" * 80)
print("ASSEMBLED MATRIX vs EXPECTED (diagonal)")
print("=" * 80)

# Actually assemble
M = solver.get_M()
print(f"Assembled M shape: {M.shape}")

# Expected diagonal for periodic grid with rho=1:
# Each P1 inner node belongs to some number of triangles.
# For a 4x3 periodic grid, all P1 nodes are inner, each belongs to 6 triangles.
# The diagonal contribution from one triangle is:
#   sum_q w_q * area * N_i(q)^2
#   = area * (1/6) * [N_i(q0)^2 + N_i(q1)^2 + N_i(q2)^2]
#
# For node 0 (local index 0 in the triangle):
#   N_0 = 1-x-y, at q0=(1/6,1/6): N_0=2/3, at q1=(2/3,1/6): N_0=1/6, at q2=(1/6,2/3): N_0=1/6
#   integral = area/6 * [(2/3)^2 + (1/6)^2 + (1/6)^2] = area/6 * [4/9 + 1/36 + 1/36]
#            = area/6 * [16/36 + 1/36 + 1/36] = area/6 * 18/36 = area/6 * 1/2 = area/12

diag_one_tri_node0 = area * sum(weights[q] * N_tri[q, 0]**2 for q in range(quad_per_tri))
diag_one_tri_node1 = area * sum(weights[q] * N_tri[q, 1]**2 for q in range(quad_per_tri))
diag_one_tri_node2 = area * sum(weights[q] * N_tri[q, 2]**2 for q in range(quad_per_tri))

print(f"Diagonal contribution per triangle:")
print(f"  Node 0 (1-x-y): {diag_one_tri_node0:.10e}")
print(f"  Node 1 (x):     {diag_one_tri_node1:.10e}")
print(f"  Node 2 (y):     {diag_one_tri_node2:.10e}")
print(f"  Sum (should = area/6 = {area/6:.10e}): "
      f"{diag_one_tri_node0 + diag_one_tri_node1 + diag_one_tri_node2:.10e}")
print()

# For a periodic 4x3 grid, each P1 node sits at a corner shared by 4 squares,
# each square has 2 triangles, so 8 triangles. But the node appears as different
# local indices in different triangles.
# Let's count from the nnz array instead.

print("Assembled diagonal values for first few P1 inner nodes:")
block = assembly.block_order[(res, var)]
nnz_start = block['nnz_idx_start']
for inner_node in range(min(6, nb_inner_p)):
    nnz_idx = assembly.lookup_nnz(res,
                                   np.array([inner_node], dtype=np.int32),
                                   var,
                                   np.array([inner_node], dtype=np.int32))
    if nnz_idx[0] >= 0:
        assembled_val = M[nnz_idx[0]]

        # Count contributions from nnz array
        locations = np.argwhere(nnz_flat == nnz_idx[0]).flatten()

        # Expected: count occurrences and their local indices
        expected_val = 0.0
        for loc in locations:
            sq = loc // (n_tri * entries_per_quad)
            rem = loc % (n_tri * entries_per_quad)
            tri = rem // entries_per_quad
            k = rem % entries_per_quad
            var_local = k // nodes_tri_res
            res_local = k % nodes_tri_res
            # For rho=1, ele_val = area * sum_q w_q * N_res * N_var
            expected_val += sum(weights[q] * area * N_tri[q, res_local] * N_tri[q, var_local]
                                for q in range(quad_per_tri))

        print(f"  Node {inner_node}: assembled={assembled_val:.8e}, "
              f"expected={expected_val:.8e}, "
              f"ratio={assembled_val/expected_val if expected_val != 0 else float('inf'):.6f}, "
              f"n_contributions={len(locations)}")

print()

# =============================================================================
# Part 8: Off-diagonal check
# =============================================================================

print("=" * 80)
print("ASSEMBLED MATRIX vs EXPECTED (off-diagonal, first few)")
print("=" * 80)

# Pick a few off-diagonal pairs
for inner_res in range(min(3, nb_inner_p)):
    for inner_var in range(min(3, nb_inner_p)):
        if inner_res == inner_var:
            continue
        nnz_idx = assembly.lookup_nnz(res,
                                       np.array([inner_res], dtype=np.int32),
                                       var,
                                       np.array([inner_var], dtype=np.int32))
        if nnz_idx[0] < 0:
            continue

        assembled_val = M[nnz_idx[0]]
        if abs(assembled_val) < 1e-20:
            continue

        locations = np.argwhere(nnz_flat == nnz_idx[0]).flatten()
        expected_val = 0.0
        for loc in locations:
            sq = loc // (n_tri * entries_per_quad)
            rem = loc % (n_tri * entries_per_quad)
            tri = rem // entries_per_quad
            k = rem % entries_per_quad
            var_local = k // nodes_tri_res
            res_local = k % nodes_tri_res
            expected_val += sum(weights[q] * area * N_tri[q, res_local] * N_tri[q, var_local]
                                for q in range(quad_per_tri))

        ratio = assembled_val / expected_val if expected_val != 0 else float('inf')
        print(f"  ({inner_res}, {inner_var}): assembled={assembled_val:.8e}, "
              f"expected={expected_val:.8e}, "
              f"ratio={ratio:.6f}, n_contributions={len(locations)}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
