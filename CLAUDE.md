# Project Overview

GaPFlow is a 2D height-averaged Navier-Stokes (lubrication) solver. The active backend is
the Taylor-Hood P2/P1 FEM solver in `GaPFlow/solver_fem/`. P2 elements for mass flux `jx/jy`,
P1 for density `rho`. Variables: `['jx', 'jy', 'rho']` (+ `'E'` with energy). Equations:
mass continuity + x/y momentum (+ energy). Terms are modular `NonLinearTerm` instances
in `terms.py` (R1x=mass, R2x=momentum, R3x=energy).

## Key files

- `GaPFlow/solver_fem/elements.py` — `TaylorHoodP2P1`: P1/P2 shape functions, quadrature (3pts/tri),
  each cell split into 2 triangles. Stencils for P1/P2 sparsity connectivity.
- `GaPFlow/solver_fem/grid_index.py` — `GridIndexManager`: index masks for P1 (ghost=1) and P2
  (ghost=2) padded grids, L2G maps, square connectivity arrays `sq_TO_inner_*` / `sq_FROM_padded_*`.
- `GaPFlow/solver_fem/global_matrix.py` — `field_to_global`: maps local block indices to interleaved
  global DOF ordering for PETSc/SciPy.
- `GaPFlow/solver_fem/assembly.py` — `Assembly`: COO sparsity, `AssemblyTemplate` (shape_weighting +
  nnz_index), `assemble_matrix` (Jacobian) and `assemble_rhs` (residual). Fully vectorized,
  no Python loop per cell.
- `GaPFlow/solver_fem/quad_fields.py` — `QuadFieldManager`: muGrid-backed nodal/quad fields,
  interpolation to quad pts, physics updates. `sync_from_problem_q` uses `scipy.ndimage.zoom`
  (order=1) to upsample `jx/jy` from P1 to P2 grid.
- `GaPFlow/solver_fem/terms.py` — `NonLinearTerm` instances + `get_active_terms()`.
- `GaPFlow/solver_fem/petsc_system.py` / `scipy_system.py` — linear solver wrappers (same interface).
- `GaPFlow/solver_fem/scaling.py` — `build_scaling` row-column preconditioner.
- `GaPFlow/solver_fem.py` — `FEMSolver`: Newton loop wiring all components.
- `GaPFlow/parallel.py` — `DomainDecomposition`: MPI decomp, ghost exchange, BC application.
  `BCContext` passed to user BC callbacks. `_apply_field_bcs` handles P1_cell, P1_nodal, P2_nodal.

## Grids

- P1 (coarse): corners only. Inner: `Nx×Ny`. Padded: `(Nx+2)×(Ny+2)` (ghost depth 1).
- P2 (fine): corners + edge midpoints + cell center. Non-periodic: `(2Nx-1)×(2Ny-1)` globally.
  Padded: `(2Nx-1+4)×(2Ny-1+4)` (ghost depth 2).
- `problem.q` lives on the P1 padded grid for all fields.

## Assembly pipeline (per Newton iteration)

1. `update_quad()` → `update_physics()` → `update_nodal_to_quad()` → `update_quad_computed()`
2. `_build_all_quad_fields()` → dict of `(nb_sq, nb_quad_sq)` arrays
3. `assemble_matrix(qf, terms)` → COO values `(n_nnz,)`
4. `assemble_rhs(qf, terms)` → residual `(res_size,)`
5. `linear_solver.assemble` + `solve` → Newton update `dq`

## BC callbacks

User BC callbacks receive a `BCContext` and return an array. For P2 fields (`jx`, `jy`),
if the callback returns a P1-shaped array (e.g. built from `problem.q`), `_apply_field_bcs`
stretches it to the P2 ghost shape using `scipy.ndimage.zoom(..., order=1)`.

# Code Style

- Do not align `=` signs in blocks (no extra spaces to vertically align assignment operators).
- Do not change docstrings or comments unless explicitly asked to.

# Tests

`tests/test_solver_fem_jacobian.py` and `tests/test_solver_fem_analytic.py` target the old
solver_fem_v1 assembly and are NOT valid for the current Taylor-Hood P2P1 code in `solver_fem/`.
The relevant test for the new code is `tests/test_solver_fem_assembly_fd.py`.

# FD Test Design (`tests/test_solver_fem_assembly_fd.py`)

Progressive finite difference verification of `assemble_matrix` and `assemble_rhs`.
Uses `Problem.from_string()` with a minimal YAML config. Infrastructure:

- `make_problem(Nx, Ny, bc, term_list)` — creates and fully initializes `(problem, solver)`
- `compute_fd_jacobian(solver)` — central FD of `get_R`, relative perturbation
- Tests are structured in levels 1–6, starting with a single term on a 3×2 grid

Term subset isolation: pass `term_list: [R1T]` etc. in the YAML config under
`fem_solver.equations.term_list`. This goes through `get_active_terms()` normally.

## Resolved: assemble_matrix diagonal over-count (was factor 3/2)

This bug has been fixed. The diagonal over-counting in `assemble_matrix` is resolved.

## Quad field axis order fix (already applied)

`get_quad` / `get_deriv_dx` / `get_deriv_dy` in `quad_fields.py` now return shape
`(nb_sq, nb_quad_sq)` via `.transpose(2, 1, 0).reshape(-1, sq.shape[0])`.
muGrid stores fields as `(nb_quad_sq, Nx_padded, Ny_padded)` — the transpose converts
to the x-major square ordering that matches `sq_x_arr_p`/`sq_y_arr_p`.

## TODO (medium–long term)

- **Scaling blocks**: implement block-aware scaling in `scaling.py` that treats the P2/P1
  sub-blocks separately, rather than applying a single row/column scaling across the full
  interleaved system.

- **Solution checker** (as in the previous solver version): after computing the Newton
  update `dq`, check whether applying it would produce unphysical values (e.g. negative
  densities); if so, reduce `alpha` on the current update (no retry of the linear solve)
  until the update is safe.

- **Checkpoint-restart**: implement `problem.save_checkpoint(path)` and
  `problem.load_checkpoint(path)` (or a `from_checkpoint` classmethod) so simulations
  can be resumed. Minimal state to persist (per MPI rank, as `.npz`):
  - `problem.q` (local padded array), `step`, `simtime`, `dt`
  - `kinetic_energy_old` — needed for the first residual after restart
  - `residual_buffer` (deque of 5) — needed for the `converged` check
  - Elastic only: `topo.u_prev`, `topo._h0_prev`, PID controller `_prev`/`_integral`
  On load: restore the above after `_pre_run()`, then call `set_q_nodal` +
  `update_quad()` + `store_prev_values()` so `_prev` quad fields are initialised
  from the loaded `q` before the first Newton step. Each MPI rank writes/reads its own
  shard (e.g. `checkpoint_rank{rank}.npz`); rank layout must match between save and
  restart. Hook the save into the `SIGUSR1` signal path in `Problem._receive_signal`.
