# Pressure-Based Mass Conservation — Migration Notes

Working document for the switch from density-based to pressure-based mass
conservation in the Taylor-Hood P2P1 FEM solver (`GaPFlow/fem_2d/`).

This file captures decisions as they are made. Sections marked **TBD** are not
yet agreed.

---

## 1. Goal

Replace `rho` with `p` as the P1 (coarse-grid) primary variable in the Newton
system. The mass conservation equation is reformulated in terms of `p`.

Motivation: improved system conditioning (mass and momentum residuals both
live in pressure-rate units) and elimination of the d²p/dρ² Jacobian
correction terms (`R21x_corr`, `R21y_corr`, `R1PSPG_P*2`).

---

## 2. Core design decision

**Option A — `p` is the DOF, `rho` is a derived field at quad points.**

- Newton variables: `['jx', 'jy', 'p']` (+ `'E'` if energy active).
- `rho` is computed from `p` via the inverse EoS (`rho_of_p`) and stored as a
  read-only nodal + quad field, updated each Newton iteration.
- Existing momentum / energy / wall-stress terms that consume `rho` keep using
  it; they just read it from the derived field instead of from a DOF.
- The Jacobian of those terms w.r.t. the new DOF `p` is obtained by chain rule:
  `∂f/∂p = (∂f/∂ρ) · (dρ/dp)`.

Rationale: least invasive path — most term algebra is preserved, only the
mass equation and the pressure-gradient term (R21) are restructured.

---

## 3. Mass conservation equation

### Strong form

Original (density-based):

```
∂ρ/∂t + ∇·j + (1/h)(j·∇h) = 0
```

New (pressure-based) — obtained by multiplying every term by `dp/dρ`:

```
∂p/∂t + c²·∇·j + c²·(1/h)(j·∇h) = 0        where c² := dp/dρ
```

The first term follows from the chain rule: `(dp/dρ)·∂ρ/∂t = ∂p/∂t`.

### Discretisation choice for the time derivative

For the `R1T` term we use the direct pressure form

```
∂p/∂t  ≈  (p - p_prev) / dt
```

rather than the equivalent `(dp/dρ)·(ρ(p) - ρ(p_prev))/dt`. This is
equivalent to O(Δt) but has a much cleaner Jacobian:

- Direct form:  `∂R1T/∂p = -1/dt`  (constant, no correction term).
- Formal form:  `∂R1T/∂p = -[d²p/dρ²·(dρ/dp)·Δρ/dt + 1/dt]`  — requires a
  d²p/dρ² Jacobian correction term analogous to `R21_corr`.

The direct form removes the need for any `R1T_corr` term.

### Weak form of the new R1 terms

For each term the `(dp/dρ)` factor enters as a quad-point coefficient field
(not as a derivative). Structurally every R1 term is the density-based term
multiplied by `(dp/dρ)` at quad points — except R1T, which is written
directly as `Δp/Δt`.

| Term      | Density form (current)         | Pressure form (new)                        |
|-----------|--------------------------------|--------------------------------------------|
| `R1T`     | `-(ρ - ρ_prev)/dt`             | `-(p - p_prev)/dt`                         |
| `R11x`    | `-jx`                          | `-(dp/dρ) · jx`                            |
| `R11y`    | `-jy`                          | `-(dp/dρ) · jy`                            |
| `R11Sx`   | `-(1/h)(dh/dx) · jx`           | `-(dp/dρ)·(1/h)(dh/dx) · jx`               |
| `R11Sy`   | `-(1/h)(dh/dy) · jy`           | `-(dp/dρ)·(1/h)(dh/dy) · jy`               |

`dp/dρ` is evaluated at quad points from the current `p` (via EoS). The PSPG
terms `R1PSPG_*` and the mass-Laplacian `R1Lx/Ly` will be decided separately
(see §5 open items).

### Residual units

R1 residual units change from `[ρ]/[t]` to `[p]/[t]`. The scaling machinery
(`scaling.py`) will need `p_ref` for the mass row instead of `ρ_ref`.

---

## 4. Rho-dependent terms

### Rewrite rule for terms currently taking `rho` as a DOF

For every term with `'rho'` in `dep_vars` (R22*, R23*, R24*, R25*, R31*, R32*,
R34, R35*, R36, R1PSPG_W*):

1. In `dep_vars`, replace `'rho'` with `'p'` at the same position (keeps
   positional signatures aligned).
2. Add `'drho_dp'` to `dep_vals`.
3. Leave `fun` untouched — it reads `rho` from `ctx` (quad field), which is
   already consistent with the new DOF.
4. In `der_funs`, multiply the p-slot (formerly rho-slot) by `ctx['drho_dp']()`.
   Other slots (jx, jy, E) unchanged.

Exceptions: `R21x/R21y` and `R1T` — `p` enters directly, no chain rule.
`R21x_corr` / `R21y_corr` are deleted (no `d²p/dρ²` term exists with `p` as DOF).

### Reference: R21x (pressure gradient, simplified)

```python
R21x = NonLinearTerm(
    name='R21x', res='momentum_x',
    dep_vars=['p'], dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    d_dx_resfun=True, der_testfun=False)
```

### Reference: R24x (wall stress, chain rule on p-slot)

```python
R24x = NonLinearTerm(
    name='R24x', res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[
        # ∂f/∂p = (∂f/∂ρ) · (dρ/dp)
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx'](),
    ],
    d_dx_resfun=False, der_testfun=False)
```

---

## 5. Viscosity / wall-stress Jacobian — lagged approximation

### Preserved from the current solver

In `stress.py:build_grad`, the JAX-traced `get_tau(rho, jx, jy, h, ...)` pulls
`eta = get_shear_viscosity(self)` from the surrounding scope **before** JAX
starts tracing. So `eta` is a leaf constant in the Jacobian — the term
`∂tau/∂eta · ∂eta/∂(DOF)` is **not** part of the current Jacobian.

This is a deliberate linearisation-lagging approximation: `eta` is refreshed
once per Newton iteration, but held frozen during the Jacobian assembly of
that iteration. The migration **keeps this approximation**.

### No hidden p↔rho round-trip

With `p` as the DOF:

- `p.pressure.update()` (current rho→p call) is replaced by the inverse
  `rho = rho_of_p(p)` step — one EoS evaluation per iteration.
- `p.viscosity.update(pressure, ...)` keeps its existing signature. Its input
  is now directly the DOF, so there is no intermediate rho→p inversion
  (previously implicit inside `pressure.update()`).
- The JAX `get_tau(rho, jx, jy, ...)` trace still takes `rho` as input. Since
  `rho` at quad points is already available from the forward EoS, no bounce
  inside the trace. JAX returns `∂tau/∂rho`; the term der_funs multiply by
  `drho_dp` at quad (the chain-rule step, §4 rule 4).

### Quad-field update order (pressure-based)

```
1.  p nodal                    (DOF — just set by Newton)
2.  rho nodal     = rho_of_p(p_nodal)
3.  drho_dp, d2rho_dp2 nodal   (from EoS inverse)
4.  eta = viscosity.update(p_nodal, dp/dx, dp/dy, h, …)     (eta frozen during Jacobian)
5.  Interpolate {p, rho, drho_dp, eta, h, ...} → quad
6.  tau_xz, tau_yz etc. = JAX(rho_quad, jx_quad, …, eta_frozen)
7.  dtau/drho quad fields from JAX grad
8.  Term der_funs multiply the p-slot by drho_dp at quad
```

### Fields that disappear / appear

- **Removed quad fields:** `dp_drho`, `d2p_drho2`, `d_dx_rho`, `d_dy_rho`,
  `rho_prev`.
- **New quad fields:** `drho_dp`, `d_dx_p`, `d_dy_p`, `p_prev`.
  Nodal `rho` remains (derived, not a DOF).
  `d2rho_dp2` is **not needed** in the current term set (no pressure-based
  term carries a `d²ρ/dp²` Jacobian correction) — add only if a future term
  requires it.

### Future work point

Include `∂tau/∂eta · ∂eta/∂p` (and analogous `∂(·)/∂eta · ∂eta/∂p` for the
bulk-viscous terms R23*) in the Jacobian. Requires either tracing `eta`
inside the JAX graph, or adding an explicit chain-rule term. Not part of
this migration — document as a known linearisation error of O(dη/dp).

---

## 6. EoS inverse — implementation plan

### Scope

Only the two EoS variants actively used by the test cases are implemented
first:

- **Dowson-Higginson** — closed-form analytic inverse.
- **Bayada-Chupin** — closed-form analytic inverse in all three regions
  (liquid / mixture / vapor). Non-obvious: the mixture branch is a Möbius
  function because the denominator `D(ρ) = A + B·ρ` is affine in ρ, so the
  log inversion gives `ρ = q·A / (1 − q·B)` with `q = exp((p−Pcav)/N) /
  (ρ_v c_v²)`.

All other EoS variants (`power_law`, `murnaghan_tait`, `cubic`,
`van_der_waals`, `bwr`) raise `NotImplementedError` until needed.

### API

Add to `models/pressure.py`:

- `eos_rho(pressure, prop)` — dispatcher mirroring `eos_pressure`.
- `eos_drho_dp(pressure, prop)` — obtained via `jax.grad(rho_of_p_<branch>)`,
  consistent with the forward EoS by construction.

Per-branch inverses are jit/vmapped scalar functions.

### Gradient safety inside `jnp.where` — mandatory

In the Bayada mixture branch, `exp((p − Pcav)/N)` overflows for
liquid-region `p` (where `p − Pcav` can reach tens of MPa and `N ≈ ρ_v c_v²
≈ 3000`). `jax.grad` through `jnp.where` evaluates **all** branches and can
be poisoned by NaN / Inf intermediates even when the primal selector picks
the finite branch. Clamp `p` inside the mixture formula:

```python
p_mix = jnp.clip(p, P_vt, Pcav)
q     = jnp.exp((p_mix - Pcav) / N) / (rho_v * c_v**2)
rho_mix = q * A / (1.0 - q * B)
```

The outer `jnp.where` still selects the correct branch for the primal; the
clamp only matters for gradient-safety on the inactive branch.

### Verified accuracy (test_eos_inverse.py)

Dense sweep (2000 points each) confirmed:

- `rho_of_p` value accuracy: max `5.1e-16` (DH), `5.3e-15` (Bayada) —
  machine precision across the whole range, no exceptions.
- Jacobian consistency `dp/dρ · dρ/dp − 1`: max `2.8e-14` (DH), `5.9e-14`
  (Bayada), machine precision per region.
- `p → ρ → p` round-trip degrades outside the physical operating range
  (DH at sub-kPa gauge pressures, Bayada at the mixture-liquid boundary).
  Root cause is catastrophic cancellation in the **forward** EoS
  (`r − 1` with `r ≈ 1`, `ln(arg)` with `arg ≈ 1`), not in the inverse.
  Solver never performs `p → ρ → p`, so this is a diagnostic finding only.

### Why use `jax.grad` rather than hand-derivatives

The analytic `drho_dp` formulas are easy to derive but error-prone under
future EoS tweaks. `jax.grad(rho_of_p)` guarantees consistency with the
forward by construction and matches the existing pattern in
`stress.py.build_grad` / `energy.py`.

---

## 7. `problem.q` layout and ghost exchange

### Decision: `problem.q` stays ρ-based; FEM2D owns `p` internally

- `problem.q = [ρ, jx, jy]` is preserved. No change visible to
  `solver_explicit`, `solver_fem_1d`, IO, checkpoints, or user BC callbacks.
- The FEM2D solver holds `p` in its own `nodal_fields['p']` as the P1 DOF.
  `nodal_fields['rho']` remains as a derived field, set from `p` each Newton
  iteration via `eos_rho(p)`.
- Sync gate is the single translation boundary between "Newton state" and
  "problem state":
  - `sync_from_problem_q`: seed `nodal_fields['p'] = eos_pressure(q[0])`, and
    `nodal_fields['rho'] = q[0]` directly (the inverse call is unnecessary —
    they are consistent by construction at sync time).
  - `sync_to_problem_q`: `q[0] = nodal_fields['rho']`. The rho nodal field is
    already current (it was recomputed from `p` earlier in the quad-field
    update), so no EoS call is needed in `sync_to`.

### User-facing API — unchanged

- Initial condition: users still specify `rho_init` (or equivalent) in YAML;
  `sync_from_problem_q` derives the initial `p`. Keeps existing configs
  working without edits.
- BC callbacks: user callbacks continue to return ρ values. The FEM2D solver
  does the ρ → p translation at physical-boundary ghosts (see below).
- Pressure-valued BC callbacks are a possible future extension; not part of
  this migration.

### Internal variable naming and BC alias

- FEM2D internals use `variables = ['jx', 'jy', 'p']`. `DOF_GRID` / `DOF_IDX`
  in `assembly.py` use `'p'` in place of `'rho'`.
- The user-facing BC list order in `grid_index.py._BC_LIST_ORDER` stays
  `['rho', 'jx', 'jy']` (user config format unchanged).
- Add a single alias inside `GridIndexManager._bc_neumann` construction so
  that a query for `'p'` resolves to the `'rho'` BC entry. Invisible to users.

### Ghost exchange — Approach B

MPI-exchange **both `p` and `ρ`** on the P1 channel. Fill `p` at
physical-boundary ghosts from the BC-applied `ρ` via EoS forward.

Concrete `_exchange_ghosts` structure inside `FEMSolver2d`:

```python
decomp.update_ghosts(
    exchange_specs=[
        (rho_field, 'P1'),
        (p_field,   'P1'),          # NEW
        (jx, 'P2'), (jy, 'P2'),
    ],
    bc_specs=[
        (rho_field.pg[0], 'rho', 'P1_nodal'),   # user BC callback fills rho
        (jx.pg[0],        'jx',  'P2_nodal'),
        (jy.pg[0],        'jy',  'P2_nodal'),
    ],
    problem=p,
)
# After update_ghosts:
#   - Rank-boundary ghosts: p and rho both filled by MPI, mutually consistent
#     (neighbour set rho = eos_rho(p) before sending).
#   - Physical-boundary ghosts: rho filled by user BC; p is still uninitialised.
# Small post-step: fill p at the four physical-boundary ghost strips from rho.
self._fill_p_at_physical_boundaries()  # ~5 lines using decomp.is_at_x{W,E}/y{S,N}
```

Rationale over the alternative (exchange `p` only, derive `ρ` at rank-boundary
ghosts):

- Extra P1 message doubles the P1 channel but P1 is negligible compared to
  the P2 `jx`/`jy` messages. In typical runs this is inconsequential.
- Post-step logic is reduced to a per-boundary slice loop (~5 lines) rather
  than the ~20-line interior-vs-physical ghost mask reconciliation.
- No new `DomainDecomposition` primitive. The existing
  `update_ghosts(exchange_specs, bc_specs, ...)` handles everything.

### Consistency invariants

- After every `_exchange_ghosts` call: nodal `p` and nodal `rho` are
  consistent on all cells (interior + ghosts) via `rho = eos_rho(p)`. The
  physical-boundary post-step preserves this because `p_ghost =
  eos_pressure(rho_ghost)` uses the forward EoS, matching the invariant
  modulo `O(ε_float)`.
- `q[0]` (problem-level ρ) is only refreshed at `sync_to_problem_q`. Between
  syncs it may lag the Newton state, which is the existing contract.

### Implementation checklist

- [ ] Add `nodal_fields['p']` in `QuadFieldManager._init_fields` (P1 coarse
      grid, same field collection as rho).
- [ ] Rewrite `sync_from_problem_q` / `sync_to_problem_q` per above.
- [ ] Rewrite `FEMSolver2d._exchange_ghosts` per above, plus the
      `_fill_p_at_physical_boundaries` helper.
- [ ] Add `'p' → 'rho'` alias in `GridIndexManager._bc_neumann` build.
- [ ] Rename `DOF_GRID['rho']` / `DOF_IDX['rho']` to `...['p']` and update
      `solver_fem_2d.py._VAR_TO_GRID`.

---

## 8. Agreed so far

- [x] Switch to pressure-based mass conservation, Option A (rho derived from p).
- [x] Mass equation = density-based equation multiplied by `dp/dρ` term-wise.
- [x] `R1T` uses the direct `Δp/Δt` form (no d²p/dρ² correction needed).
- [x] Rho-dependent term rewrite rule (§4): `rho → p` in `dep_vars`, add
      `drho_dp` to `dep_vals`, multiply p-slot `der_fun` by `drho_dp`.
- [x] `R21x_corr` / `R21y_corr` deleted.
- [x] `rho` is carried as **both** nodal and quad field (derived from `p`).
- [x] Viscosity Jacobian stays lagged (η frozen during assembly); `∂(·)/∂η`
      linearisation is a future work item.
- [x] EoS inverse: DH and Bayada-Chupin analytic, added to `models/pressure.py`,
      gradients via `jax.grad`. `d2rho_dp2` deferred. `jnp.clip`-for-gradient-
      safety is mandatory inside the Bayada mixture branch.
- [x] `problem.q` stays ρ-based; FEM2D owns `p` internally (Option B).
      User API unchanged (rho_init, ρ-valued BC callbacks). Ghost exchange
      via Approach B (MPI both p and ρ on P1, fill p physical-boundary ghosts
      post-exchange via EoS forward). No new `DomainDecomposition` primitive.
      Variable name `'p'` inside FEM2D; BC-table alias `'p' → 'rho'`.
- [x] Cavitation guards deactivated for the pressure-based solver. The
      `_detect_rho_*`, `cavitation_guard`, `smooth_rho` methods in
      `solver_fem_2d.py` are left in the source (unmodified, for future
      reference or re-enable) but their call sites in `update_dynamic` are
      disabled. Expectation: with `p` as the DOF, branch-switch oscillations
      at `ρ_l` are much less likely — the Newton state no longer lives on
      the ill-conditioned side of the piecewise EoS. If cavitation cases
      turn out to need the guards back, translate to p-space then (Option α
      of the earlier discussion: threshold at `Pcav`, `δ_p = c_l²·δ_ρ`).
- [x] PSPG terms deactivated. `R1PSPG_Px/Py/Px2/Py2/Tx/Ty/Wx/Wy` remain
      defined in `terms.py` (unmodified) but `get_active_terms()` will not
      include them — the `physics.pspg` flag stays `False` by default and no
      pressure-form translation is done. Expectation: with `p` as the DOF,
      pressure-velocity coupling no longer needs PSPG stabilisation (Taylor-
      Hood P2/P1 is already inf-sup stable for this mix). Reintroducing
      PSPG, if needed, is a separate work item — translation rule is
      Option 1 from the discussion: absorb `(dp/dρ)` into a redefined
      `τ_pspg`, and delete the `_Px2 / _Py2` corrections.
- [x] Mass-Laplacian `R1Lx / R1Ly` deactivated. `physics.mass_diffusion`
      stays `False` by default; terms remain in `terms.py` unmodified. The
      adaptive-α machinery in `update_dynamic` (`mass_diffusion_adaptive`)
      is left in place but dormant. If later needed, reintroduce as a direct
      pressure-Laplacian (`fun=α·p`, `der=α`, `dep_vars=['p']`) rather than
      the ρ-Laplacian with chain rule.
- [x] Scaling: `char_scales['p'] := problem.prop['P0']` (reference pressure
      from the EoS config). For EoS variants without a native `P0` (e.g.
      Bayada), the user must supply `P0` in `prop` — no silent default. At
      solver initialisation, print the chosen value:
      `[FEMSolver2d] Using reference pressure p_ref = {P0:.3e} Pa for scaling`.
      One-line change in `scaling.py::compute_characteristic_scales`:
      `scales = {'p': prop['P0'], 'jx': j_ref, 'jy': j_ref}`.
- [x] BC list order: user-facing `_BC_LIST_ORDER = ['rho', 'jx', 'jy']` in
      `grid_index.py` stays unchanged. FEM2D queries for `'p'` resolve via
      the `'p' → 'rho'` alias in `GridIndexManager._bc_neumann`
      (per §7). No change needed in `parallel.py`'s BC-related code paths.
- [x] FD tests: keep current structure of `tests/test_fem_2d_assembly_fd.py`
      (levels 1–6). `compute_fd_jacobian` and `make_problem` are DOF-
      agnostic, no harness changes needed. Per-level YAML `term_list:` drops
      references to deleted terms (`R21x_corr`, `R21y_corr`, `R1PSPG_Px2`,
      `R1PSPG_Py2`). `rho_init` in test YAML preserved — initialisation
      flows through `sync_from_problem_q`. No new "level 0" for EoS;
      `test_eos_inverse.py` already covers that layer.

---

## 9. Open items (TBD)

*(none — all design decisions settled.)*

---

## 10. Implementation plan

Three stages. Stage 1 is additive and lands independently. Stage 2 is a
coordinated atomic change (solver is broken mid-stage; keep the commit
self-contained). Stage 3 validates.

### Stage 1 — EoS inverse (additive, ships alone) ✅ **done**

- `models/pressure.py`: added `rho_of_p_dh`, `rho_of_p_bayada` (with
  `jnp.clip(p, P_vt, Pcav)` gradient-safety in the Bayada mixture branch).
- Dispatchers `eos_rho(pressure, prop)` and `eos_drho_dp(pressure, prop)`
  mirror `eos_pressure`. `eos_drho_dp` uses `jax.grad(rho_of_p_<branch>)` —
  no hand derivatives.
- Other EoS variants (`power_law`, `murnaghan_tait`, `cubic`,
  `van_der_waals`, `bwr`) raise `NotImplementedError` in the dispatchers;
  `user` forwards to `prop['EOS_user_inv']`.
- Validation: `test_eos_inverse.py` confirms machine-precision inverse
  values and gradients across both EoS variants, including a dispatcher-
  path check.

### Stage 2 — Atomic migration of the FEM2D solver ✅ **done (first-iteration scope)**

Status: Level 1–5 FD tests of `tests/test_fem_2d_assembly_fd.py` all pass at
machine precision with the default pressure-based term set (R11*, R1T,
R21*, R2T*, R24*). Level 4b / 9 / 11 tests fail as expected — they
exercise dormant terms (R23*, R1L*, R21_corr) that were intentionally left
in ρ-form.

**Additional findings during implementation:**

- `nodal_fields['p']` must be a freshly-created 3D muGrid field
  (`fc.real_field('p_nodal', 1, 'pixel')`), not a wrapper around
  `fc.get_real_field('pressure')`. The pre-existing 'pressure' field has a
  2D `.pg` shape (no leading component axis), incompatible with the
  standard `.pg[0]` indexing used across QuadFieldManager. A helper
  `_push_p_to_pressure_field()` is called after each nodal-p update to
  keep `problem.pressure.pressure` in sync for downstream readers
  (topography, IO, other backends).
- Power-law EoS (`PL`) inverse added to `models/pressure.py` — needed
  because existing FD tests use `EOS: PL` with `alpha=0` as their
  canonical linear EoS. `rho_of_p_pl(p) = rho_0 · (p/P_0)^(1 - alpha/2)`.
- `dp_drho` quad field **kept** (not removed as originally planned) — it's
  the pressure-form prefactor for R11*, R11S*. Populated via
  `p.pressure.dp_drho` (the existing JAX-jitted forward-EoS gradient).
  `d2p_drho2`, `d_dx_rho`, `d_dy_rho`, `rho_prev` removed as planned.
- `plane_shear` default flipped `True → False` in **both**
  `terms.py::_term_names_from_physics` **and** `io.py` (there are two
  independent physics-default dicts — missed the second one on the first
  pass).
- Dormant terms (R22*, R23*, R25*, R31*, R32*, R34, R35*, R36,
  R1PSPG_W*) kept in density form for now; §4 rewrite deferred until
  they're needed.

**First-iteration scope (active terms):**

- Mass equation: all terms — `R11x`, `R11y`, `R11Sx`, `R11Sy`, `R1T`.
- Momentum: `R2Tx`, `R2Ty`, `R21x`, `R21y`, `R24x`, `R24y` only.
- All stabilisers off (cavitation guards, PSPG, mass-Laplacian — already
  decided in §8).
- In-plane viscous diffusion `R23*` off — flip the `plane_shear` default
  from `True` to `False` in `_term_names_from_physics` (`terms.py`). This
  is the only code-wide deactivation change. Convection `R22*`, body force
  `R25*`, energy terms already default-off.
- Other rho-dependent terms (`R22*`, `R23*`, `R25*`, `R31*`, `R32*`,
  `R34`, `R35*`, `R36`, `R1PSPG_W*`) still get their §4 rewrite so the
  code is internally consistent — they're just not in the active set for
  the first-iteration validation.

**Validation for this first iteration**: the classical thin-film
lubrication system `∂p/∂t + c²∇·j + c²(1/h)(j·∇h) = 0`, `∂j/∂t + ∇p =
τ_wall/h`. Sufficient for `parabolic_slider` / `bernoulli_venturi`
baselines in the isoviscous, steady-state limit.

The DOF change, term rewrites, and plumbing updates must land together —
no useful intermediate state exists. Edit order within the commit:

1. **`fem_2d/quad_fields.py`**
   - Add `nodal_fields['p']` (P1 coarse grid, same field collection as rho).
   - Remove `dp_drho`, `d2p_drho2`, `d_dx_rho`, `d_dy_rho`, `rho_prev` from
     `_needed_fields`. Add `drho_dp`, `d_dx_p`, `d_dy_p`, `p_prev`.
   - Rewrite `sync_from_problem_q`: `p = eos_pressure(q[0])`, `rho = q[0]`.
   - Rewrite `sync_to_problem_q`: `q[0] = rho` (already current).
   - Quad-field update pipeline: `p nodal → rho = eos_rho(p) nodal →
     drho_dp nodal → eta via viscosity.update(p_nodal, …) → interpolate
     {p, rho, drho_dp, eta, h, …} to quad → tau etc. via JAX`.
   - `store_prev_values` uses `p_prev` instead of `rho_prev`.

2. **`fem_2d/assembly.py`, `fem_2d/grid_index.py`, `solver_fem_2d.py`**
   - `DOF_GRID` / `DOF_IDX` in `assembly.py`: replace `'rho'` with `'p'`.
   - `_VAR_TO_GRID` in `solver_fem_2d.py`: replace `'rho'` with `'p'`.
   - `_init_accessors`: `variables = ['jx', 'jy', 'p']` (+ `'E'` if energy).
   - `GridIndexManager.__init__`: keep `_BC_LIST_ORDER = ['rho', 'jx', 'jy']`
     but add alias so queries for `'p'` resolve to the `'rho'` BC entry in
     the `_bc_neumann` dict.

3. **`solver_fem_2d.py::_exchange_ghosts`** — Approach B
   - `exchange_specs = [(rho, 'P1'), (p, 'P1'), (jx, 'P2'), (jy, 'P2')]`.
   - `bc_specs` unchanged (user BC callback still fills ρ at physical
     boundaries).
   - Add `_fill_p_at_physical_boundaries()` helper: for each of the four
     physical-boundary ghost strips, `p_ghost = eos_pressure(rho_ghost)`.

4. **`fem_2d/terms.py`**
   - Rewrite `R21x`, `R21y` per §4 reference: `dep_vars=['p']`, `fun=-p`,
     `der=-1`.
   - Delete `R21x_corr`, `R21y_corr` from `term_list` (remove entirely;
     they have no pressure-form analogue).
   - Rewrite `R1T` per §3: `dep_vars=['p']`, `fun=-(p - p_prev)/dt`,
     `der=-1/dt`, uses `p_prev`.
   - Apply §4 rewrite rule to: `R22xx`, `R22xxS`, `R22yx`, `R22yxS`,
     `R22xy`, `R22xyS`, `R22yy`, `R22yyS`, `R23xx`, `R23yy`, `R23xy`,
     `R23yx`, `R24x`, `R24y`, `R25x`, `R25y`, `R31x`, `R31y`, `R31Sx`,
     `R31Sy`, `R32x`, `R32y`, `R32Sx`, `R32Sy`, `R34`, `R35x`, `R35y`,
     `R36`, `R1PSPG_Wx`, `R1PSPG_Wy`.
   - Leave `R1Lx`, `R1Ly`, `R1PSPG_Px/Py/Tx/Ty` (dormant; physics flags
     stay `False` by default).
   - Delete `R1PSPG_Px2`, `R1PSPG_Py2` from `term_list` (no pressure-form
     analogue).

5. **`fem_2d/scaling.py`**
   - `scales['p'] = problem.prop['P0']` (raise if `P0` not set for the
     chosen EoS; no silent default).
   - At `FEMSolver2d` init, after `scaling` is built: print
     `[FEMSolver2d] Using reference pressure p_ref = {P0:.3e} Pa for scaling`.

6. **`solver_fem_2d.py::update_dynamic`** — deactivate dormant stabilisers
   - Comment out / gate the `cavitation_guard` and `smooth_rho` call sites.
     Keep method bodies unchanged (for future re-enable in p-space).
   - Keep `linearization_guard` — it's trivially satisfied now (no
     chain-rule linearisation error on the p diagonal), but cheap to
     leave in.

### Stage 3 — Validation

1. `tests/test_fem_2d_assembly_fd.py`: drop deleted term names from each
   level's YAML `term_list:`. Run levels 1–6. Expect machine-precision
   Jacobian match (tolerances should pass unchanged; tighten if desired).
2. Regression: run example cases `fem_2d/parabolic_slider/run.py` and
   `fem_2d/bernoulli_venturi/run_and_save.py`. Compare to stored `.npz`
   baselines.

### Rollback

Stage 1 is always safe (additive only). Stage 2 is atomic — if broken and
not quickly fixable, revert to the pre-migration commit; `fem_2d_old/` and
`fem_2d_v1/` remain untouched as historical references.
