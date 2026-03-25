# Code Style

- Do not align `=` signs in blocks (no extra spaces to vertically align assignment operators).
- Do not change docstrings or comments unless explicitly asked to.

# Tests

`tests/test_fem_2d_jacobian.py` and `tests/test_fem_2d_analytic.py` target the old
fem_2d_v1 assembly and are NOT valid for the current Taylor-Hood P2P1 code in `fem_2d/`.
The relevant test for the new code is `tests/test_fem_2d_assembly_fd.py`.

# FD Test Design (`tests/test_fem_2d_assembly_fd.py`)

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
