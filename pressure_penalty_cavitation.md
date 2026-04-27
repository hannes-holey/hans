# Penalty Method for Cavitation — Findings & Design Notes

## Background

Current GaPFlow cavitation approach (Bayada EOS): three-region piecewise EOS with a
liquid/mixture boundary. `d²p/dρ²` is discontinuous at that boundary, which destabilises
the Newton Jacobian via the `R11x/y_corr` correction terms. The penalty method replaces
EOS-based cavitation enforcement with an explicit pressure constraint.

## Penalty Formulation (Wu 1986, referenced in Habchi)

Add a penalty contribution to the mass (continuity) equation weak form:

    R_pen(v) = ε · ∫_Ω max(p_cav − p, 0) · φ_P1 dΩ

where `p_cav` is the cavitation pressure (typically `P0`), and `ε` is the penalty
strength. In the full-film region (`p ≥ p_cav`) the term is zero; in the cavitated
region it creates a source that resists further pressure drop.

## EOS with Penalty

Use a smooth single-branch EOS (e.g. Dowson-Higginson) for the entire domain. The
penalty handles the cavitation constraint — no mixture region, no `d²p/dρ²`
discontinuity, no correction terms needed.

Consequence: density does not drop below `ρ(p_cav)` in the cavitation zone. This is a
full-film/no-fill-fraction approximation (not mass-conserving JFO), which is standard in
EHL penalty approaches.

## Jacobian (Heaviside) — Open Question

The penalty Jacobian entry is:

    ∂R_pen/∂p = −ε · H(p_cav − p)   (Heaviside: 1 where p < p_cav, 0 elsewhere)

**Current implementation**: raw Heaviside step (non-smooth). Mathematically this
corresponds to using a Clarke subgradient, which is the basis of semi-smooth Newton
methods. In practice, the cavitation front rarely sits exactly at a DOF during Newton
iteration, so the non-smooth point is seldom hit.

**Status**: open. The literature (Wu 1986) uses the positive-part function directly and
couples with semi-smooth Newton. Habchi's specific treatment of the Jacobian at the
non-smooth point could not be confirmed from available sources. A smooth approximation
(e.g. softplus / sigmoid ramp) is a pragmatic alternative if Newton stalls near the
front, but is not confirmed as literature-standard.

**TODO**: Check Wu (1986) original paper and/or Habchi book §cavitation for explicit
Jacobian treatment. Consider smooth ramp with width parameter δ if raw Heaviside causes
Newton instability.

## Penalty Strength ε

Scaling not yet addressed. Needs to be calibrated relative to the (mass, p) diagonal
contribution from the time term (`drho_dp / dt`). Exposed as `fem_solver.pen_eps` in
the YAML config.

## Implementation in GaPFlow

- **Term**: `R_cav_pen` in `terms.py` — `res='mass'`, `dep_vars=['p']`, no spatial
  derivatives, no test-function derivative.
- **ctx entries**: `p_cav` (from `prop['P0']`) and `pen_eps` (from
  `fem_solver['pen_eps']`) added in `_build_terms` in `solver_fem_2d.py`.
- **Correction terms**: not needed when using a single-branch smooth EOS (DH). Excluded
  from the explicit term list for the penalty use case.

## First Test Case

Parabolic slider with Dowson-Higginson EOS (`rho0=850`, `P0=1e5`, `C1=3.5e8`,
`C2=1.23`) and explicit term list including `R_cav_pen`. Located in
`GaPFlow/fem_2d/parabolic_slider_penalty/`.
