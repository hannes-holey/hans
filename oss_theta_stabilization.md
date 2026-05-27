# OSS Stabilization for θ Oscillations — Project Plan

## Status of existing stabilizers

| Method | Physics flag | Residual-consistent | Notes |
|---|---|---|---|
| `R1UWx/y` | `upwind_theta` | No | Donor-cell, axis-split, O(dx) dissipation |
| `R1STx/y` | `theta_stab` | No | Isotropic Laplacian on θ, direction-blind |
| `R1FBpx/py/tx/ty` | `pspg_fb` | Yes (w.r.t. FB residual) | Tests ∇Nᵢ against ∇φ_FB, not ∇L_mass |
| SUPG (`R1SUPG*`) | `supg_theta` | Yes (intended) | **Unreliable — only diffusive sub-group stable** |

### Why SUPG failed

The full residual-consistent SUPG term
```
τ · ∫ (j·∇Nᵢ) · L_mass dΩ
```
decomposes into Groups 1–4 (see `project_supg_theta_plan.md`). In practice:
- Groups 2 and 4 (flux-divergence and height-source pieces) introduce extra
  cross-coupling between θ, jx/jy that destabilises the Newton iteration.
- Group 1 alone (θ-advection, the purely diffusive piece in streamline direction)
  is stable but reduces to a streamline-direction Laplacian on θ — effectively
  the same physics as `R1STx/y` but anisotropic.
- The full group gives no reliable improvement and the Jacobian is expensive.

Artificial diffusion (`R1STx/y`, `R1UWx/y`) works but changes the physics and
is hard to justify in publications.

---

## Next approach: Orthogonal Subscale Stabilization (OSS)

OSS (Codina 2000) is a residual-based stabilization that projects the fine-scale
residual onto the complement of the FE space before feeding it back. The
cross-wind diffusion is zero by construction (only the subscale residual drives
the stabilization, not a full Laplacian), and the method is consistent and
conservative.

### Core idea

Split the solution into a resolved part `θ_h` and a subscale `θ'`:
```
θ = θ_h + θ'
```
OSS approximates the subscale by:
```
θ' ≈ -τ · P^⊥(L_mass)
```
where `P^⊥` is the L²-projection onto the orthogonal complement of the FE space
(i.e. `P^⊥(f) = f − Π_h f`, with `Π_h` the nodal L²-projection).

Substituting back into the Galerkin form and linearising gives the additional
stabilization term:
```
+ τ · ∫ (j·∇Nᵢ) · P^⊥(L_mass) dΩ
```

Because `P^⊥(L_mass)` has zero FE-component by construction, the method:
- Is consistent: `P^⊥(L_mass) → 0` at the exact solution.
- Adds no spurious cross-wind diffusion (unlike isotropic Laplacians).
- Has better-conditioned Jacobians than full SUPG because the projection
  smooths out the sharp residual gradients that destabilise Groups 2/4.

### Difference from SUPG

SUPG uses the raw strong-form residual `L_mass`. OSS uses `P^⊥(L_mass) = L_mass − Π_h(L_mass)`.
The projected part `Π_h(L_mass)` is the component already captured by the
Galerkin test space — subtracting it removes the component that Galerkin
handles correctly and concentrates the stabilization where it is needed
(high-frequency / unresolved modes near the cavitation front).

In smooth regions `P^⊥(L_mass) ≈ 0`, so OSS is less dissipative than SUPG
outside sharp layers.

---

## Implementation plan

### Step 1: L²-projection of L_mass onto P1 FE space

`Π_h(L_mass)` is the nodal L²-projection:
```
M_p · (Π_h L_mass)_nodes = ∫ Nᵢ · L_mass dΩ
```
where `M_p` is the P1 mass matrix (assembled once, lumped for efficiency).

In practice: assemble the RHS of the mass equation with a Galerkin test (standard
`assemble_rhs` call), then solve with the lumped P1 mass matrix to get
`(Π_h L_mass)_nodes`, then interpolate back to quad points to get `Π_h(L_mass)_quad`.

### Step 2: Compute the residual-orthogonal part at quad points
```
P^⊥_L_mass = L_mass_quad − Π_h_L_mass_quad
```
This is a pure quad-point field, computed outside the assembly loop.

### Step 3: Add OSS term to mass residual

```
+ τ · ∫ (j·∇Nᵢ) · P^⊥_L_mass dΩ
```

This splits into `x`- and `y`-stream parts (same test-function structure as
`R1SUPGTx/y`), but the "residual" field `P^⊥_L_mass` is a **dep_val** (frozen
at the current Newton iterate, not differentiated through the projection):

| name | dep_vars | der_testfun | dep_vals |
|---|---|---|---|
| `R1OSSx` | `['p']` (dummy) | `'x'` | `tau_jx`, `oss_L_mass` |
| `R1OSSy` | `['p']` (dummy) | `'y'` | `tau_jy`, `oss_L_mass` |

`fun = lambda p: -tau_jk * oss_L_mass`  (returns dep_val directly, dep_var unused)
`der_funs[0] = lambda p: 0`             (Jacobian omitted — frozen-coefficient approximation)

The Jacobian is omitted (like `R_Lpx/y`), consistent with the "lagged" OSS
approach of Codina (2002) where `P^⊥` is updated every Newton step but not
differentiated through.

### Step 4: τ field

Use the same τ already computed for SUPG:
```
tau_jx = supg_theta_alpha * h_elem / (2*(|j| + eps)) * jx
tau_jy = supg_theta_alpha * h_elem / (2*(|j| + eps)) * jy
```
These are already quad-point fields in `quad_fields` when `supg_theta: true`.
OSS can share the same flag and coefficient (`supg_theta_alpha`) — the physics
flag selects between full-SUPG and OSS via a new sub-option.

---

## Changes required

### `quad_fields.py`
- Add `oss_L_mass` computation in `update_quad_computed()`:
  1. Evaluate `L_mass_quad` at quad points from existing quad fields.
  2. Assemble lumped P1 mass matrix once during `pre_run()`.
  3. Each Newton step: project → interpolate → subtract → store as `oss_L_mass`.

### `terms.py`
- Add `R1OSSx`, `R1OSSy` (two new `NonLinearTerm` instances).
- Add `oss_theta` to `_term_names_from_physics` gated on `cavitation` and the new
  `physics.oss_theta` flag.

### `io.py` / `sanitize_fem_solver()`
- Register `oss_theta_alpha` (or reuse `supg_theta_alpha`).

### `solver_fem_2d.py`
- Assemble lumped P1 mass matrix once in `pre_run()`.
- Each Newton step: call the `oss_L_mass` update before `_build_all_quad_fields`.

---

## Open questions

1. **Lumped vs consistent mass matrix**: Lumping (row-sum) avoids a solve per
   Newton step but introduces O(h²) projection error. For P1 this is acceptable
   and widely used. Alternative: use the diagonal of the consistent mass matrix.

2. **Lagging the projection**: Codina (2002) proposes updating `Π_h(L_mass)`
   once per time step (not per Newton iteration) to reduce cost. Try per-Newton
   first for correctness, then lag if convergence allows.

3. **Coefficient**: Start with `supg_theta_alpha ≈ 0.1–1.0`. OSS is typically
   less sensitive than SUPG to the choice of τ because the projection removes the
   resolved-scale part automatically.

4. **Interaction with R_FB Jacobian**: The OSS term has a zero Jacobian by the
   frozen-coefficient approximation. If Newton convergence stalls, a simplified
   analytic Jacobian for `d/dp(P^⊥ L_mass)` may be needed — but defer this.

5. **Test case**: Use `parabolic_slider_FB` (1D, easy to visualize θ profiles)
   and `circular_FB` (2D, captures cross-wind behavior). Compare θ profile
   smoothness and Newton iteration count against `R1STx/y` baseline.

---

## References

- Codina, R. (2000). *Stabilization of incompressible flow problems by finite
  calculus and subgrid scale models*, IJNME.
- Codina, R. (2002). *Stabilized finite element approximation of transient
  incompressible flows using orthogonal subscales*, CMAME 191, 4295–4321.
- Hughes, T.J.R., Feijóo, G.R., Mazzei, L., Quincy, J.-B. (1998).
  *The variational multiscale method — a paradigm for computational mechanics*,
  CMAME 166, 3–24.
