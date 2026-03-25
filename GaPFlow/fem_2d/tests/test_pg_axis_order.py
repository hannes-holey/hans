"""
Standalone diagnostic: determine muGrid pg axis ordering for 2D FieldCollections
and verify that get_quad produces the correct square ordering.

Key questions answered:
  1. Does field.pg have shape (sub_pt, y, x) or (sub_pt, x, y)?
  2. Does get_quad produce x-fast square ordering consistent with sq_x_arr_p / sq_y_arr_p?
"""
import numpy as np
from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2d

# ============================================================================
# Setup
# ============================================================================

CONFIG = """
options:
    output: /tmp/test_pg_axis
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
solver = FEMSolver2d(problem.fem_solver, problem)
solver.pre_run()

qm = solver.quad_mgr
gi = solver.grid_idx

print("=" * 70)
print("PART 1: Basic shapes and grid dimensions")
print("=" * 70)
print(f"Nx_p_inner  = {gi.Nx_p_inner}")
print(f"Ny_p_inner  = {gi.Ny_p_inner}")
print(f"Nx_p_padded = {gi.Nx_p_padded}")
print(f"Ny_p_padded = {gi.Ny_p_padded}")
print(f"sq_per_row  = {gi.sq_per_row}")
print(f"sq_per_col  = {gi.sq_per_col}")
print(f"nb_sq       = {gi.nb_sq}")

# Pixel field shape (nodal)
rho_nodal = qm.nodal_fields['rho']
print(f"\nrho_nodal.pg shape = {rho_nodal.pg.shape}")
print(f"  -> sub_pts x dim1 x dim2 = {rho_nodal.pg.shape}")

# Quad field shape
rho_quad = qm.quad_fields['rho']
print(f"rho_quad.pg shape  = {rho_quad.pg.shape}")
print(f"  -> sub_pts x dim1 x dim2 = {rho_quad.pg.shape}")

print("\n" + "=" * 70)
print("PART 2: Direct pg axis probing (pixel field)")
print("=" * 70)
print("Setting rho_nodal to uniform 1.0, then placing 99.0 at one position.")
print("Testing whether pg indexing is [sub_pt, x, y] or [sub_pt, y, x].")

# Save original
rho_orig = rho_nodal.pg.copy()

# Set uniform
rho_nodal.pg[:] = 1.0

# We want to set the pixel at padded-grid position (ix, iy) = (2, 1)
# which corresponds to inner position (ix-1, iy-1) = (1, 0)
target_ix_padded = 2
target_iy_padded = 1

print(f"\nTarget padded position: ix={target_ix_padded}, iy={target_iy_padded}")

# Hypothesis A: pg is indexed as [sub_pt, x, y]
rho_nodal.pg[:] = 1.0
rho_nodal.pg[0, target_ix_padded, target_iy_padded] = 99.0
val_A = rho_nodal.pg[0].copy()

# Read back using .p (inner region only, F-order)
inner_A = rho_nodal.p[0].copy()

print(f"\nHypothesis A: pg[0, ix, iy] = pg[0, {target_ix_padded}, {target_iy_padded}] = 99")
print(f"  pg[0] array (full padded):")
print(f"  {val_A}")
print(f"  inner .p[0] (F-order flat):")
print(f"  {inner_A}")

# Hypothesis B: pg is indexed as [sub_pt, y, x]
rho_nodal.pg[:] = 1.0
rho_nodal.pg[0, target_iy_padded, target_ix_padded] = 99.0
val_B = rho_nodal.pg[0].copy()
inner_B = rho_nodal.p[0].copy()

print(f"\nHypothesis B: pg[0, iy, ix] = pg[0, {target_iy_padded}, {target_ix_padded}] = 99")
print(f"  pg[0] array (full padded):")
print(f"  {val_B}")
print(f"  inner .p[0] (F-order flat):")
print(f"  {inner_B}")

# Check which hypothesis is consistent with get_nodal_sol_val
# get_nodal_sol_val does: self.nodal_fields[var].p[0].flatten(order='F')
# The inner region .p[0] has shape (Nx_p_inner, Ny_p_inner)
# With F-order flatten, x varies fastest.
# Inner position (1, 0) -> flat index = 1 * 1 + 0 * Nx_p_inner = 1
# Wait, F-order: column-major, so index = ix + iy * Nx_inner
# For (ix_inner, iy_inner) = (1, 0): flat_idx = 1 + 0 * Nx_p_inner = 1

print(f"\n.p[0] shape = {rho_nodal.p[0].shape}")

print("\n" + "=" * 70)
print("PART 3: Verify via index_mask_inner_local_p")
print("=" * 70)
mask = gi.index_mask_inner_local_p
print(f"index_mask_inner_local_p shape = {mask.shape}")
print(f"index_mask_inner_local_p =\n{mask}")
print(f"\nThis mask is indexed as [x, y] by construction (see grid_index.py).")
print(f"The inner node at padded (2,1) = inner (1,0) has mask value = {mask[target_ix_padded, target_iy_padded]}")

# Now check which hypothesis gives 99 at the right inner node
# Under hypothesis A (pg indexed [sub_pt, x, y]):
#   pg[0, 2, 1] = 99 means padded position (x=2, y=1)
#   inner position = (1, 0), mask value = mask[2, 1]

# Under hypothesis B (pg indexed [sub_pt, y, x]):
#   pg[0, 1, 2] = 99 means padded position (y=1, x=2)
#   inner position = (1, 0), mask value = mask[2, 1]
# But then pg axis 1 = y, axis 2 = x

print("\n" + "=" * 70)
print("PART 4: Interpolate to quad and check get_quad")
print("=" * 70)

# Restore and set up a clean perturbation
rho_nodal.pg[:] = 1.0
# Use hypothesis A first: pg[sub_pt, x, y]
rho_nodal.pg[0, target_ix_padded, target_iy_padded] = 99.0

# Interpolate
qm.interpolate_nodal_to_quad('rho')
gq = qm.get_quad('rho')
print(f"get_quad('rho') shape = {gq.shape}  (nb_sq, nb_quad_sq)")

# Which squares see the perturbation?
sq_has_perturb = np.any(np.abs(gq - 1.0) > 0.1, axis=1)
perturbed_sqs = np.where(sq_has_perturb)[0]
print(f"\nWith pg[0, {target_ix_padded}, {target_iy_padded}] = 99 (Hyp A: pg[sub, x, y]):")
print(f"  Perturbed squares: {perturbed_sqs}")
for sq in perturbed_sqs:
    print(f"    sq {sq}: sq_x={gi.sq_x_arr_p[sq]}, sq_y={gi.sq_y_arr_p[sq]}, "
          f"quad values={gq[sq]}")

# Which squares SHOULD contain the node at padded (ix=2, iy=1)?
# A node at padded (ix, iy) is a corner of squares whose origins are:
#   (ix-1, iy-1), (ix, iy-1), (ix-1, iy), (ix, iy) -- as bottom-left corners
# But only if those origins are valid (0 <= sx < sq_per_row, 0 <= sy < sq_per_col)
expected_sqs_A = []
for dsx, dsy in [(-1, -1), (0, -1), (-1, 0), (0, 0)]:
    sx = target_ix_padded + dsx
    sy = target_iy_padded + dsy
    if 0 <= sx < gi.sq_per_row and 0 <= sy < gi.sq_per_col:
        sq_idx = sx + sy * gi.sq_per_row  # x-fast
        expected_sqs_A.append(sq_idx)
        # wait, sq_idx = sx % sq_per_row is just sx since sx < sq_per_row
        # But actual formula: sq_x = sq_idx % sq_per_row, sq_y = sq_idx // sq_per_row
        # So sq_idx = sy * sq_per_row + sx  -- NO, that's y-major
        # Actually: sq_idx from code is just np.arange(nb_sq), and
        # sq_x = sq_idx % sq_per_row, sq_y = sq_idx // sq_per_row
        # So sq_idx = sq_y * sq_per_row + sq_x = sy * sq_per_row + sx

# Redo properly
expected_sqs_A = []
for dsx, dsy in [(-1, -1), (0, -1), (-1, 0), (0, 0)]:
    sx = target_ix_padded + dsx
    sy = target_iy_padded + dsy
    if 0 <= sx < gi.sq_per_row and 0 <= sy < gi.sq_per_col:
        sq_idx = sy * gi.sq_per_row + sx
        expected_sqs_A.append(sq_idx)
expected_sqs_A = sorted(expected_sqs_A)
print(f"  Expected squares (if pg is [sub, x, y]): {expected_sqs_A}")

# Now check via sq_TO_inner_p
target_inner_idx = mask[target_ix_padded, target_iy_padded]
print(f"\n  Target inner index (from mask[{target_ix_padded},{target_iy_padded}]) = {target_inner_idx}")
sq_corners = gi.sq_TO_inner_p  # (nb_sq, 4)
sqs_from_connectivity = np.where(np.any(sq_corners == target_inner_idx, axis=1))[0]
print(f"  Squares containing inner node {target_inner_idx} (from sq_TO_inner_p): {sqs_from_connectivity}")

match_A = set(perturbed_sqs) == set(sqs_from_connectivity)
print(f"\n  Hypothesis A match: {match_A}")

# Now try hypothesis B
rho_nodal.pg[:] = 1.0
rho_nodal.pg[0, target_iy_padded, target_ix_padded] = 99.0
qm.interpolate_nodal_to_quad('rho')
gq_B = qm.get_quad('rho')
sq_has_perturb_B = np.any(np.abs(gq_B - 1.0) > 0.1, axis=1)
perturbed_sqs_B = np.where(sq_has_perturb_B)[0]

print(f"\nWith pg[0, {target_iy_padded}, {target_ix_padded}] = 99 (Hyp B: pg[sub, y, x]):")
print(f"  Perturbed squares: {perturbed_sqs_B}")
for sq in perturbed_sqs_B:
    print(f"    sq {sq}: sq_x={gi.sq_x_arr_p[sq]}, sq_y={gi.sq_y_arr_p[sq]}, "
          f"quad values={gq_B[sq]}")

match_B = set(perturbed_sqs_B) == set(sqs_from_connectivity)
print(f"  Hypothesis B match: {match_B}")

print("\n" + "=" * 70)
print("PART 5: Second probe point to disambiguate")
print("=" * 70)

# Use a non-symmetric point: padded (3, 1) = inner (2, 0)
probe2_ix = 3
probe2_iy = 1
target_inner_idx2 = mask[probe2_ix, probe2_iy]
sqs_from_conn2 = np.where(np.any(sq_corners == target_inner_idx2, axis=1))[0]
print(f"Probe at padded ({probe2_ix}, {probe2_iy}), inner idx = {target_inner_idx2}")
print(f"Expected squares from sq_TO_inner_p: {sqs_from_conn2}")

# Hypothesis A
rho_nodal.pg[:] = 1.0
rho_nodal.pg[0, probe2_ix, probe2_iy] = 99.0
qm.interpolate_nodal_to_quad('rho')
gq2A = qm.get_quad('rho')
perturbed2A = np.where(np.any(np.abs(gq2A - 1.0) > 0.1, axis=1))[0]
match2A = set(perturbed2A) == set(sqs_from_conn2)
print(f"Hyp A (pg[sub, x, y]): perturbed squares = {perturbed2A}, match = {match2A}")

# Hypothesis B
rho_nodal.pg[:] = 1.0
rho_nodal.pg[0, probe2_iy, probe2_ix] = 99.0
qm.interpolate_nodal_to_quad('rho')
gq2B = qm.get_quad('rho')
perturbed2B = np.where(np.any(np.abs(gq2B - 1.0) > 0.1, axis=1))[0]
match2B = set(perturbed2B) == set(sqs_from_conn2)
print(f"Hyp B (pg[sub, y, x]): perturbed squares = {perturbed2B}, match = {match2B}")

print("\n" + "=" * 70)
print("PART 6: Third probe — clearly asymmetric")
print("=" * 70)

# padded (1, 3) = inner (0, 2)  -- different from (3, 1)
probe3_ix = 1
probe3_iy = 3
if probe3_ix < gi.Nx_p_padded and probe3_iy < gi.Ny_p_padded:
    target_inner_idx3 = mask[probe3_ix, probe3_iy]
    sqs_from_conn3 = np.where(np.any(sq_corners == target_inner_idx3, axis=1))[0]
    print(f"Probe at padded ({probe3_ix}, {probe3_iy}), inner idx = {target_inner_idx3}")
    print(f"Expected squares from sq_TO_inner_p: {sqs_from_conn3}")

    # Hypothesis A
    rho_nodal.pg[:] = 1.0
    rho_nodal.pg[0, probe3_ix, probe3_iy] = 99.0
    qm.interpolate_nodal_to_quad('rho')
    gq3A = qm.get_quad('rho')
    perturbed3A = np.where(np.any(np.abs(gq3A - 1.0) > 0.1, axis=1))[0]
    match3A = set(perturbed3A) == set(sqs_from_conn3)
    print(f"Hyp A (pg[sub, x, y]): perturbed squares = {perturbed3A}, match = {match3A}")

    # Hypothesis B
    rho_nodal.pg[:] = 1.0
    rho_nodal.pg[0, probe3_iy, probe3_ix] = 99.0
    qm.interpolate_nodal_to_quad('rho')
    gq3B = qm.get_quad('rho')
    perturbed3B = np.where(np.any(np.abs(gq3B - 1.0) > 0.1, axis=1))[0]
    match3B = set(perturbed3B) == set(sqs_from_conn3)
    print(f"Hyp B (pg[sub, y, x]): perturbed squares = {perturbed3B}, match = {match3B}")
else:
    print(f"Probe ({probe3_ix}, {probe3_iy}) out of range for padded grid "
          f"({gi.Nx_p_padded}, {gi.Ny_p_padded})")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

# Gather all results
all_A = []
all_B = []
all_A.append(match_A)
all_B.append(match_B)
all_A.append(match2A)
all_B.append(match2B)
if probe3_ix < gi.Nx_p_padded and probe3_iy < gi.Ny_p_padded:
    all_A.append(match3A)
    all_B.append(match3B)

if all(all_A) and not all(all_B):
    print("RESULT: pg is indexed as [sub_pt, x, y]")
    print("  -> get_quad transpose(2,1,0) maps (x, y, sub_pt) -> correct x-fast ordering")
elif all(all_B) and not all(all_A):
    print("RESULT: pg is indexed as [sub_pt, y, x]")
    print("  -> get_quad transpose(2,1,0) swaps axes, need to verify correctness")
elif all(all_A) and all(all_B):
    print("RESULT: AMBIGUOUS -- both hypotheses match (grid may be too symmetric)")
else:
    print("RESULT: NEITHER hypothesis matches cleanly -- something else is wrong")
    print(f"  Hypothesis A results: {all_A}")
    print(f"  Hypothesis B results: {all_B}")

# Restore
rho_nodal.pg[:] = rho_orig
