# Fischer-Burmeister Cavitation — Implementation Plan

## Physical Model

Elrod-Adams: replace flux `j` with `(1−θ)·j` in the mass equation.
θ is the void fraction (P1 DOF, same grid as p):
- full film: θ=0, p>p_cav
- cavitation: θ>0, p=p_cav

`∂p/∂t = dp/drho · ∇·((1−θ)j) − (1/h) · ∇h · (1−θ)j`

No ∂θ/∂t in the mass equation. θ is determined implicitly at each time step
by the coupled (mass, FB) system. Momentum terms are unchanged.

---

## New DOF: theta (P1)

- `DOF_GRID['theta'] = 'p'` — add to module-level dict in `assembly.py`
- DOF index for theta is **computed dynamically** inside `Assembly.__init__`
  from the ordered variables list (slot 2 = p, slot 3 = E if active, then theta).
  Cannot be a static constant because the slot depends on which combination
  of P1 DOFs is active.
- `field_to_global`: replace `bEnergy: bool` with `p_factor: int`; change
  P1-branch check from `res_type == 2 or res_type == 3` to `res_type >= 2`.
- `p_factor` computed in `Assembly.__init__` as the count of P1 variables:
  `p_factor = len([v for v in variables if DOF_GRID[v] == 'p'])`
- `quad_fields`: add `theta` at quadrature points (no gradient needed — IBP
  removes derivative from theta in R11x_fb/R11y_fb)

---

## New Residual: fb (P1)

- `DOF_GRID['fb'] = 'p'` — add to module-level dict in `assembly.py`
- Residual slot index computed dynamically (same logic as theta DOF index)
- Standard Galerkin, no spatial derivatives

---

## New Mass Terms (Elrod-Adams variants)

The original R11x, R11y, R11Sx, R11Sy terms are kept unchanged.
New `_fb`-suffixed variants are added and selected instead when `cavitation: true`.
The original `_corr` terms are also dropped when cavitation is active (dp/drho
is weakly p-dependent for fluid EoS — correction is negligible).
`R_cav_pen` is superseded by the FB approach and excluded when `cavitation: true`.

### R11x_fb / R11y_fb — IBP (weak form)

**Motivation**: moving the derivative to the test function via IBP gives a
complete Jacobian for (mass, theta) with no correction terms, and avoids
needing `d_dx_theta` in quad_fields.

```python
R11x_fb = NonLinearTerm(
    name='R11x_fb', res='mass', dep_vars=['jx', 'theta'], dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jx, theta:  ctx['dp_drho']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta:  ctx['dp_drho']() * (1 - theta),  # ∂/∂jx
        lambda ctx: lambda jx, theta: -ctx['dp_drho']() * jx,           # ∂/∂theta
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun='x')
```

R11y_fb: identical with `jy` and `der_testfun='y'`.

Jacobian blocks produced (both complete, no correction terms):
- `(mass, jx,    'none', 'x')`: `∫ ∂Nᵢ/∂x · dp_drho·(1−θ) · Nⱼᴾ² dΩ`
- `(mass, theta, 'none', 'x')`: `∫ ∂Nᵢ/∂x · (−dp_drho·jx) · Nⱼᴾ¹ dΩ`

### R11Sx_fb / R11Sy_fb — no IBP needed

jx/jy already appear as values (d_dx_resfun=False), so theta is simply
added to dep_vars:

```python
R11Sx_fb = NonLinearTerm(
    name='R11Sx_fb', res='mass', dep_vars=['jx', 'theta'], dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta:  ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * jx,
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)
```

R11Sy_fb: identical with `jy` and `dh_dy`.

### get_active_terms / _term_names_from_physics

When `physics.get('cavitation', False)` is True:
- Remove `R11x`, `R11y`, `R11Sx`, `R11Sy` and all `_corr` variants
- Add `R11x_fb`, `R11y_fb`, `R11Sx_fb`, `R11Sy_fb`, `R_FB`

---

## New FB Term

```python
def _fb_denom(a, b):
    d = np.sqrt(a**2 + b**2)
    return np.where(d == 0.0, np.finfo(float).eps, d)

R_FB = NonLinearTerm(
    name='R_FB', description='Fischer-Burmeister complementarity condition',
    res='fb', dep_vars=['p', 'theta'], dep_vals=['p_cav'],
    fun=lambda ctx: lambda p, theta: (
        lambda a: np.sqrt(a**2 + theta**2) - a - theta
    )(p - ctx['p_cav']()),
    der_funs=[
        lambda ctx: lambda p, theta: (lambda a: a     / _fb_denom(a, theta) - 1.0)(p - ctx['p_cav']()),
        lambda ctx: lambda p, theta: (lambda a: theta / _fb_denom(a, theta) - 1.0)(p - ctx['p_cav']()),
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)
```

Singularity at (p−p_cav, θ)=(0,0): bump denominator to machine epsilon
(not sqrt regularisation — see literature note).

---

## Global DOF Structure

Current interleaving at each P1 corner: `[jx(0), jy(1), p(2)]`

| Active flags         | Interleaving                          | p_factor |
|----------------------|---------------------------------------|----------|
| base                 | `[jx(0), jy(1), p(2)]`               | 1        |
| energy               | `[jx(0), jy(1), p(2), E(3)]`         | 2        |
| cavitation           | `[jx(0), jy(1), p(2), theta(3)]`     | 2        |
| energy + cavitation  | `[jx(0), jy(1), p(2), E(3), theta(4)]` | 3      |

`field_to_global` signature: `p_factor: int` replaces `bEnergy: bool`.
P1-branch condition: `res_type >= 2` (was `== 2 or == 3`).

---

## quad_fields additions

| Field   | Notes                                    |
|---------|------------------------------------------|
| `theta` | P1 nodal field interpolated to quad pts  |

No gradient fields needed (IBP removes ∂θ/∂x from all terms).

---

## Solver wiring

### problem.py — solution field component count

`problem.py` line 158 creates the solution field with a fixed component count.
When cavitation is active this must be 4:

```python
nb_sol = 3 + int(self.fem_solver['equations'].get('cavitation', False))
self.__field = self.fc.real_field('solution', (nb_sol,))
```

`self.fem_solver` is parsed from YAML before this line — no ordering issue.

### solver_fem_2d.py

- Add `'theta': 'p'` to `_VAR_TO_GRID` and `'fb': 'p'` to `_RES_TO_GRID`.
- `_init_accessors`: read `cavitation` flag; append `'theta'` / `'fb'` to
  `variables` / `residuals`.
- `_build_assembly`: drop `energy=` kwarg (Assembly derives p_factor from
  the variables list).
- `_exchange_ghosts`: add `(theta_field, 'P1')` to `exchange_specs`; apply
  Dirichlet θ=0 at physical boundaries in a new
  `_fill_theta_at_physical_boundaries()` method (not via user callback).
- Newton loop `update_dynamic`: apply θ floor after each update (see BCs section).

### theta: problem.q and ghost exchange

theta lives in `problem.q[3]` (P1 padded, same layout as rho in `problem.q[0]`).
Ghost exchange follows the identical path as rho — no new MPI machinery needed.

```
problem.q[0]  rho
problem.q[1]  jx  (coarse)
problem.q[2]  jy  (coarse)
problem.q[3]  theta          ← new
```

`sync_to_problem_q`: add `p.q[3] = self.nodal_fields['theta'].pg[0]` (guarded by
`if self.cavitation`).

`sync_from_problem_q`: copy `p.q[3]` into `nodal_fields['theta'].pg[0]` directly
(no EoS derivation — theta is a Newton DOF). Initial value is zeros (full-film IC);
`problem.q[3]` is zero-initialised when the solution field is allocated with `nb_sol=4`.

### QuadFieldManager

- New P1 nodal field `theta_nodal` and quad field `theta` (same path as `p`).
- `CAVITATION_FIELDS = {'theta'}` OR'd into `_needed_fields()` when active.
- `update_nodal_to_quad`: add `'theta'` to `coarse_interp` when active.
- `store_prev_values`: no change — theta has no `_prev` field and the
  existing guard (`prev_key in self.quad_fields`) silently skips it.

### theta floor placement

The θ floor (`q[theta_sl] = np.maximum(...)`) is a **post-step operation in
`update_dynamic`**, applied after accepting the Newton update. It must NOT be
applied inside `solver_step_fun`, `get_R`, or the assembly path. The FD test
perturbs theta DOFs symmetrically around the current state (including slightly
negative theta values in the minus-step), so the floor must not interfere with
the raw residual evaluation during Jacobian verification.

## theta BCs and initialisation

- Initial value: θ=0 everywhere (start in full-film)
- **All physical boundaries**: Dirichlet θ=0, applied automatically — no user
  callback required. Covers the common case (full-film inlet/outlet, walls).
- θ floor applied after each Newton step — must be strictly positive to avoid
  the FB Jacobian singularity at (p−p_cav, θ)=(0,0):
  ```python
  theta_min = fem_solver.get('theta_min', np.finfo(float).eps)
  q[theta_sl] = np.maximum(q[theta_sl], theta_min)
  ```
  Default `theta_min = np.finfo(float).eps` (~2.2e-16), consistent with `_fb_denom`.

---

### p_cav property key

`_build_terms` uses `ctx['p_cav'] = lambda: p.prop['p_cav']`. Confirm that the
YAML `properties` block populates `p_cav` explicitly (or that the EOS parser
inserts it). The Bayada EOS config in the FD test currently uses `P0` for ambient
pressure — add `p_cav: <value>` explicitly to any test YAML that exercises FB
terms, matching the value used in `_pcav()`.

---

## FD Jacobian test (`tests/test_fem_2d_assembly_fd.py`)

The existing test infrastructure (`compute_fd_jacobian`, `check_all_terms`,
`compare_matrices`) generalises to any number of DOFs automatically — no
structural changes needed. What must be added:

### YAML template

A `_CONFIG_BAYADA_CAV` template: identical to `_CONFIG_BAYADA` but with
`cavitation: True` under `fem_solver.equations` and an explicit `p_cav`
property matching the Bayada `_pcav()` value. A `cavitation` bool parameter
to `make_problem` selects which template to use.

### Straddling initialiser

`_init_fb_straddling(solver)` — sets the initial `(p, θ)` state to cover all
three regimes simultaneously:

```python
def _init_fb_straddling(solver):
    Pcav = _pcav(solver.problem.prop)
    delta = 0.05 * abs(Pcav)
    q = solver.get_q_nodal().copy()

    p_sl    = solver._sol_slices['p']
    theta_sl = solver._sol_slices['theta']
    n_p = p_sl.stop - p_sl.start

    # p ramp straddling p_cav (same as _init_bayada_straddling)
    p_ramp = np.linspace(Pcav - delta, Pcav + delta, n_p)
    q[p_sl] = p_ramp

    # theta complementary to p: high where p < p_cav, near-zero where p > p_cav
    # Uses the same linear index ordering as p (both P1 inner nodes)
    a = p_ramp - Pcav                        # signed distance from p_cav
    theta_ramp = np.where(a < 0, -a / delta * 0.5, np.finfo(float).eps)
    q[theta_sl] = theta_ramp

    _set_and_sync(solver, q)
```

This guarantees:
- Some nodes with `a > 0, θ ≈ eps` (full-film, FB residual ≈ 0)
- Some nodes with `a < 0, θ > 0` (cavitation zone, FB residual ≈ 0 only if θ = −a)
- The transition region near `(a, θ) ≈ (0, 0)` where `_fb_denom` guard is relevant

### Term groups

Extend `_TERM_GROUPS` with:

```python
['R11x_fb'],
['R11y_fb'],
['R11Sx_fb'],
['R11Sy_fb'],
['R_FB'],
```

Note: `R11x_fb` etc. have no `_corr` companions — the IBP formulation gives a
complete Jacobian without chain-rule corrections. `R_FB` produces Jacobian blocks
in columns `p` and `theta` only (no `jx`/`jy` columns).

### Column-block reporting

`dep_vars` in `print_term_check`, `eps_sensitivity`, and `check_all_terms` must
be read from `solver._sol_slices.keys()` rather than hardcoded as
`['jx', 'jy', 'p']`, so `theta` is included automatically when cavitation is active.

### `_set_and_sync` remains unchanged

`_sync_rho_before_exchange` only touches `rho`/`p`; `exchange_ghosts()` handles
`theta` via the extended `exchange_specs`. No changes needed to the test helper.

---

## Open questions

- R21 pressure gradient: standard `−∂p/∂x` unchanged (FB pins p=p_cav in
  cavitation zone, so ∂p/∂x→0 naturally).

## Resolved

- PSPG stabilisation terms: not used — P2P1 discretisation provides inf-sup
  stability without PSPG.
