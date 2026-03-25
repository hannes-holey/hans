# Taylor-Hood P2P1 Implementation Plan

## Status of `fem_2d/`

- `elements.py` — `TaylorHoodP2P1` with P1 and P2 inner classes, stencils, `QuadOperator`. Done.
- `tests/test_elements.py` — shape function and quadrature tests. Done.
- `plan.md` — architectural design. Settled.
- `tests/test_decomp.py` — Phase 1 tests. Done. ✅

Everything else needs to be built from scratch, in a deliberate order so that each component can be tested before the next depends on it.

---

## Architecture Map: What Changes vs. What Stays

### Reused as-is (with extension)
- `parallel.py` `DomainDecomposition` — extended to hold a second internal `CartesianDecomposition` for the velocity fine grid (ghost depth 2, non-uniform subdomain sizes). The external interface remains a single instance; the second `CartesianDecomposition` is an implementation detail.
- `fem_2d/petsc_system.py`, `fem_2d/scipy_system.py` — linear solvers, pure linear algebra
- `NonLinearTerm` contract (name, res, dep_vars, fun, der_funs) — PDEs don't change
- Newton iteration loop in `solver_fem_2d.py`

### Replaced entirely
| Old | New | Reason |
|---|---|---|
| `fem_2d_old/elements.py` | `fem_2d/elements.py` | P2 shape functions, two-grid operators |
| `fem_2d_old/grid_index.py` | `fem_2d/grid_index.py` | Two separate index masks (mass flux/pressure) |
| `fem_2d_old/assembly.py` | `fem_2d/assembly.py` | Four stencil types, new shape-weighting scheme |
| `fem_2d_old/quad_fields.py` | `fem_2d/quad_fields.py` | P2 nodal→quad path for mass flux, P1 for pressure |
| `solver_fem_2d.py` | `fem_2d/solver.py` | Wires new components, same Newton skeleton |

### Adapted / new
- `fem_2d/terms.py` — copy of `fem_2d_old/terms.py` with PSPG/GLS stabilization terms removed; all physical terms unchanged
- `fem_2d/global_matrix.py` — new point-interleaved global ordering formula (mass-flux-primary, density at even-even positions)

---

## Known Issues / Risks Before Starting

### 0. ~~Stripe decomposition hardcoded in `parallel.py`~~ ✅ Resolved in Phase 1
Block decomposition (`floor(sqrt(n)) × n//floor(sqrt(n))`) is now implemented and tested. `FFTDomainTranslation` still assumes 1D Y-decomposition — this is a known limitation but does not affect the FEM solver.

### 1. ~~`stencil_even_odd` and `stencil_odd_even` appear asymmetric~~ ✅ Resolved in Phase 0
All three non-even-even stencils were incomplete (9 entries instead of 19). Fixed in `elements.py`. The correct stencils are geometrically identical for interior nodes — all node types have the same 19-entry neighbourhood.

### 2. ~~Ghost depth of 2 is not native to `DomainDecomposition`~~ ✅ Resolved in Phase 1
`self._decomp_v` with ghost depth 2 is implemented, conditioned on `numerics['solver'] == 'fem'`. muGrid's `CartesianDecomposition` accepts `nb_ghost_left/right > 1` as confirmed. BC application for both ghost layers is implemented in `communicate_ghost_buffers_v` / `_apply_mass_flux_bcs`.

### 3. Grid-specific properties across the two grids *(keep in mind)*
Properties like `local_shape_padded`, `nb_subdomain_grid_pts`, `subdomain_locations`, `index_mask_padded_global`, and `communicate_ghosts` are currently written for a single grid. With two grids inside one `DomainDecomposition`, each needs a velocity-grid variant.

The straightforward approach — consistent `_v` suffix for all velocity-grid properties, with a clear docstring convention — is sufficient for two grids and keeps the interface transparent. A heavier abstraction (e.g. a grid descriptor object) would only pay off with three or more grids, which is speculative. Stick with `_v` suffix, document it prominently, and revisit only if a third grid is ever needed.

### 3. ~~Global ordering formula is non-trivial and untested~~ ✅ Resolved in Phase 3
The corrected formula `((r+1)//2)*M + ((c+1)//2 if r%2==0 else 0)` is implemented in `global_matrix.py` and verified by 40 tests (round-trips on 6 grid sizes, manual spot-checks, energy variants). plan.md updated with the correct formula.

### 4. ~~Non-uniform subdomain sizes for velocity field~~ ✅ Resolved in Phase 1
Verified empirically for 4-process 2×2 decomposition on multiple grid sizes. muGrid's ceiling-division strategy combined with the divisibility constraint (`Nx % nx_splits == 0`) produces the correct non-uniform split (SW owns extra interface nodes per `plan.md` convention). `v_loc == 2*p_loc` holds for all ranks.

### 5. ~~Dirichlet treatment for depth-2 ghost nodes~~ ✅ Resolved in Phase 1
Both ghost layers are set to the Dirichlet value in `_apply_mass_flux_bcs`. Confirmed correct — does not impose flatness. Uncertainty note in `plan.md` line 26 should be removed.

### 6. No existing P2P1 terms / physics
All terms in `fem_2d/terms.py` were written for P1P1 with PSPG/GLS stabilization. For Taylor-Hood, stabilization is not needed (inf-sup stable). Terms need to be rewritten from scratch — this is not a blocker but must be scoped.

---

## Implementation Phases

---

### Phase 0: Stencil Verification ✅ Complete

**Test:** `tests/test_stencils.py` — 11 passed.
- `derive_stencil(col, row)`: enumerates all even-even square SW corners near origin, checks triangle membership, collects offsets. This is the ground truth.
- All four stencil match tests: derived == hardcoded for even-even, odd-odd, even-odd, odd-even.
- Symmetry tests: even-even and odd-odd are closed under negation; no duplicates in any stencil.
- Visual debug plot: saved to `/tmp/stencils_all.png` (blue=correct, red=hardcoded-only, green=missing).

**Bugs found and fixed in `elements.py`:**
- `stencil_odd_odd`, `stencil_even_odd`, `stencil_odd_even` were each missing 10 entries — all nodes reachable via the outer triangle ring at offset ±2. The correct full stencil has 19 entries (same shape as even-even). All three non-even-even stencils are geometrically identical for an interior node.

**Acceptance:** ✅ All 11 tests pass.

---

### Phase 1: Extend `DomainDecomposition` for P2 mass flux grid ✅ Complete

**Changes to `parallel.py`:**
- `__init__` accepts `numerics=None`; `_decomp_v` only initialised when `numerics['solver'] == 'fem'`.
- Block decomposition (`floor(sqrt(n)) × n//floor(sqrt(n))`) replacing hardcoded stripe.
- Divisibility check raises `ValueError` when `Nx % nx_splits != 0 or Ny % ny_splits != 0`.
- Added `_v` properties: `nb_domain_grid_pts_v`, `nb_subdomain_grid_pts_v`, `subdomain_locations_v`, `local_shape_inner_v`, `local_shape_padded_v`, `icoordsg_v`, `index_mask_padded_global_v`.
- Added `communicate_ghosts_v(field)`, `communicate_ghost_buffers_v(problem, jx, jy)`, `_apply_mass_flux_bcs`, `_get_bc_slices_v`.
- All `_v` members assert `_is_fem` if called without `solver='fem'`.

**Test:** `tests/test_decomp.py` — serial (49 passed), 3-process (25 passed, 37 skipped), 4-process (61 passed, 1 skipped).
- Block decomposition coverage, subdivision counts, divisibility check.
- Mass flux subdomain tiling for multiple grid sizes and process counts.
- Ghost exchange correctness at depth 1 (pressure) and depth 2 (mass flux).
- Neumann and Dirichlet BC application on both ghost layers.
- `_is_fem` gating: non-FEM decomposition has no `_decomp_v`.

**Acceptance:** ✅ All tests pass.

---

### Phase 2: `GridIndexManager` (P2P1 version) ✅ Complete

**New file:** `fem_2d/grid_index.py`

**Design:**
- Two separate index mask sets (`_p` / `_v` suffix) for pressure (ghost depth 1) and mass flux (ghost depth 2).
- `index_mask_inner_local_{p,v}`: inner nodes sequential (F-order), ghost nodes -1.
- `index_mask_padded_local_{p,v}(var)`: inter-subdomain ghosts get new indices; Neumann ghosts forwarded to inner neighbour (both layers for `_v`); Dirichlet ghosts stay -1.
- `l2g_list_{p,v}`: local contributor → global index.
- `sq_TO_inner_{p,v}`, `sq_FROM_padded_{p,v}`: square corner connectivity (4 nodes for P1, 9 nodes for P2).
- P2 square origins at `2 * P1` square origins on the padded fine grid.

**Test:** `tests/test_grid_index.py` — 84 passed.
- Inner masks: sequential, correct shape, F-order, boundary -1.
- Padded masks: equals inner on inner nodes; Neumann forwarding (both layers for P2); Dirichlet stays -1; mixed BC (per-variable).
- l2g round-trip for both grids.
- Square connectivity shapes, interior squares all-positive, boundary squares have -1 in correct positions, P2 origin = 2×P1 origin, index ranges.
- `nb_contributors` for Neumann and Dirichlet (both grids): equals inner nodes only in serial (physical boundary ghosts excluded from index assignment; Neumann forwarding reuses inner indices).
- Periodic wrapping (serial): ghost wraps to opposite inner for P1 (depth 1) and P2 (depth 2), no -1 in fully periodic mask, l2g round-trip, semi-periodic (x only) correctness.
- MPI (4-process): interior ranks have `nb_contributors > nb_inner` due to inter-subdomain ghost index assignment; l2g round-trip holds on all ranks for both grids.

**Acceptance:** ✅ All tests pass.

---

### Phase 3: Global Matrix Ordering ✅ Complete

**New file:** `fem_2d/global_matrix.py`

**Design:**
- `_n_density_before(k, cols_v, M)`: count of density nodes emitted before mass-flux block k.
  Formula: `((r+1)//2)*M + ((c+1)//2 if r%2==0 else 0)` where `r=k//cols_v, c=k%cols_v`.
  Note: plan.md had a sign error in the column count (`c//2+1` should be `(c+1)//2` which equals the number of even values in {0..c-1}).
- `field_to_global(field_idx, res_type, cols_v, M, energy=False) -> int`
- `global_to_field(global_idx, cols_v, M, energy=False) -> (field_idx, res_type)`

**Test:** `tests/test_global_matrix.py` — 40 passed.
- `TestNDensityBefore`: 7 direct tests of the count formula.
- `TestRoundTrip`: `global_to_field → field_to_global` and `field_to_global → global_to_field` round-trips for M ∈ {2,3,4,5} with and without energy.
- `TestSpotCheck`: 10 spot-checks for M=3, cols_v=5, no energy.
- `TestSpotCheckEnergy`: 9 spot-checks for M=2, cols_v=3, energy=True.
- `TestValidation`: invalid res_type and energy=3 without energy=True.

**Acceptance:** ✅ All 40 tests pass.

---

### Phase 4: `QuadFieldManager` (P2P1 version)
**Goal:** Correct nodal→quadrature interpolation using P2 operators for velocity and P1 operators for pressure.

**New file:** `fem_2d_new/quad_fields.py`

**Design:**
- Velocity fields use `P2.interpolation_operator`, `P2.dx_operator`, `P2.dy_operator`.
- Pressure field uses `P1.interpolation_operator`, `P1.dx_operator`, `P1.dy_operator`.
- Output shape: `(n_tri * n_quad, sq_per_row, sq_per_col)` for both.
- Keep `get(name)`, `get_deriv_dx(name)`, `get_deriv_dy(name)` interface.

**Test:** `tests/test_quad_fields.py`
- For a linear velocity field `u(x,y) = ax + by`, interpolated values at quadrature points should match the analytical `ax + by` exactly (P2 is exact for polynomials up to degree 2).
- For a quadratic velocity field, interpolated values should be exact.
- Derivative operators: for `u = ax + by`, `du/dx = a` exactly everywhere.
- Pressure (P1): same exactness tests for linear fields.

**Acceptance:** All exactness tests pass to machine precision.

---

### Phase 5: `Assembly` (P2P1 version)
**Goal:** Correct `nnz` structure, shape-weighting precomputation, and COO injection.

**New file:** `fem_2d_new/assembly.py`

**Design** (following `plan.md`):
- Compile one contiguous `nnz` array covering all blocks (ρ→Rρ, ρ→Rjx, jx→Rρ, jx→Rjx, etc.).
- For each block: precompute `shape_weighting` array (length `2 * n_quad * n_contr`) and `nnz_index` list.
- For Dirichlet boundaries: compile nine separate `shape_weighting` variants (inner, N, S, E, W, NE, NW, SE, SW).
- The `folded` reduction: `contribs.reshape(-1, n_tri, n_contr).sum(axis=1)`.
- `nnz_index` list length: `n_sq_x * n_sq_y * n_tri * n_contr`.

**Test:** `tests/test_assembly.py`
- **Structural test:** For a small grid, build the COO pattern and assert no duplicate (row, col) entries, no out-of-range indices, and that the sparsity pattern is symmetric for symmetric problems.
- **Zero-sum test:** For a constant field, all residuals should be zero (translation invariance). Assert `||R|| ≈ 0`.
- **Consistency test (patch test):** For a linear velocity field on a uniform mesh with no body forces, momentum residual should be zero to machine precision. This is the fundamental correctness test for any FEM assembly.
- **Block test:** Perturb one DOF, finite-difference the residual, and compare with the assembled tangential matrix column. Do this for at least one (res, dep_var) pair per block type.

**Acceptance:** Patch test passes; finite-difference Jacobian check passes to `O(h²)` with perturbation size `h`.

---

### Phase 6: `Terms` (P2P1 physics)
**Goal:** Remove stabilization terms; keep all physical terms unchanged.

**New file:** `fem_2d_new/terms.py`

**Design:**
- Copy `fem_2d/terms.py` and remove the PSPG/GLS stabilization terms. The physical PDE terms (mass conservation, momentum x/y, pressure gradient, stress, convection) are identical and do not change.
- No new term logic needs to be written.

**Test:** `tests/test_terms.py`
- For each retained term, finite-difference `fun` and compare with `der_funs`. Test at multiple quadrature point values.
- This is pure function testing — no grid needed.

**Acceptance:** Finite-difference derivatives match `der_funs` to `O(h²)`.

---

### Phase 7: Solver Integration
**Goal:** Wire all components into a working Newton solver.

**New file:** `fem_2d_new/solver.py`

**Design:**
- Reuse Newton iteration skeleton from `solver_fem_2d.py`.
- Replace grid index, quad fields, assembly, and terms with new P2P1 versions.
- Use existing `PETScSystem` or `ScipySystem` unchanged.

**Test:** `tests/test_solver.py`
- **Stokes flow (manufactured solution):** Lid-driven cavity or Poiseuille flow with known analytic solution. Measure L2 error vs. mesh refinement.
  - P2 velocity should converge at O(h³).
  - P1 pressure should converge at O(h²).
- **Pressure robustness:** With a constant pressure field and zero body force, velocity should be zero. Assert `||u|| < tol`.

**Acceptance:** Convergence rates match theoretical P2P1 rates on at least two refinement levels.

---

## File Creation Order

```
fem_2d/
├── elements.py            ✅ Phase 0 (done)
├── plan.md                ✅ done
├── implementation_plan.md ✅ done
├── tests/
│   ├── test_elements.py   ✅ Phase 0 (done)
│   ├── test_stencils.py   ✅ Phase 0 (done)
│   ├── test_decomp.py     ✅ Phase 1 (done)
│   ├── test_grid_index.py ✅ Phase 2 (done)
│   ├── test_global_matrix.py ✅ Phase 3 (done)
│   ├── test_quad_fields.py   Phase 4
│   ├── test_assembly.py   Phase 5
│   ├── test_terms.py      Phase 6
│   └── test_solver.py     Phase 7
├── grid_index.py          ✅ Phase 2 (done)
├── global_matrix.py       ✅ Phase 3 (done)
├── quad_fields.py         Phase 4
├── assembly.py            Phase 5
├── terms.py               Phase 6
└── solver.py              Phase 7
```

---

## Notes on Testing Strategy

Each phase has a self-contained test that does not depend on subsequent phases. This means failures are localized and each component can be developed and green-lit independently. The patch test in Phase 5 is the single most important correctness check — if that passes, the element, indexing, and assembly are all consistent. The convergence rate test in Phase 7 confirms the full solver is working correctly.

Do not proceed to Phase 5 (assembly) before Phase 0 (stencil verification) is confirmed — a wrong stencil cannot be caught by assembly-level tests alone.
