#
# Copyright 2026 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# flake8: noqa: E501

"""OSS (Orthogonal Subscale Stabilization) terms for the Elrod-Adams / FB mass equation.

The projected quantity is the full flux divergence of the Elrod-Adams term:

    L(θ) = dp_drho · ∇·((1−θ)·j)
          = dp_drho · (a_vec·∇θ  +  (1−θ)·∇·j)   (a_vec = dp_drho·j, frozen)

ξ_h = P_h(L(θ)) is its L²-projection onto the P1 FE space, solved monolithically:

    R_oss:  (η_h, a_vec·∇θ) + (η_h, dp_drho·(1−θ)·∇·j) − (η_h, ξ_h) = 0

The stabilization term added to the mass residual is:

    S = (a_vec·∇q_h,  τ · P_h⊥(L(θ)))
      = (a_vec·∇q_h,  τ · L(θ))     ← Laplacian-type (R_OSS_mass_xx/yy, R_OSS_mass_div_x/y)
      − (a_vec·∇q_h,  τ · ξ_h)      ← correction (R_OSS_corr*)

a_vec, dp_drho, and ∇·j are all frozen in Jacobians (no blocks w.r.t. p, jx, jy).

Quad fields a_vec_x, a_vec_y, tau_a_x, tau_a_y are computed in
QuadFieldManager.update_quad_computed() when oss_theta is active.

τ = 1 / (2/h_elem · max(|a_vec|, 0.1) + |s·j|),   h_elem = min(dx, dy)
tau_a_x = oss_theta_alpha · τ · a_vec_x  (alpha knob already baked in)

Assembly sign conventions (from _build_weighting):
  ('x', False)  → +∫ N_res · fun · ∂(dep_var)/∂x dΩ
  ('x', 'x')   → −∫ (∂N_res/∂x) · (∂N_var/∂x) · fun dΩ
  ('x', 'y')   → −∫ (∂N_res/∂y) · (∂N_var/∂x) · fun dΩ
  ('none', 'x') → −∫ (∂N_res/∂x) · N_var · fun dΩ
  ('none', False) → +∫ N_res · N_var · fun dΩ
"""

import numpy as np
from .terms import NonLinearTerm


# ---------------------------------------------------------------------------
# R_oss residual — projection equation: (η_h, a_vec·∇θ_h) − (η_h, ξ_h) = 0
# ---------------------------------------------------------------------------

R_OSS_advx = NonLinearTerm(
    name='R_OSS_advx',
    description='OSS projection eq x: +(η_h, a_vec_x · ∂θ/∂x)',
    res='R_oss',
    dep_vars=['theta'],
    dep_vals=['a_vec_x'],
    fun=lambda ctx: lambda theta: ctx['a_vec_x']() * theta,
    der_funs=[lambda ctx: lambda theta: ctx['a_vec_x']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R_OSS_advy = NonLinearTerm(
    name='R_OSS_advy',
    description='OSS projection eq y: +(η_h, a_vec_y · ∂θ/∂y)',
    res='R_oss',
    dep_vars=['theta'],
    dep_vals=['a_vec_y'],
    fun=lambda ctx: lambda theta: ctx['a_vec_y']() * theta,
    der_funs=[lambda ctx: lambda theta: ctx['a_vec_y']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R_OSS_proj = NonLinearTerm(
    name='R_OSS_proj',
    description='OSS projection eq: −(η_h, ξ_h)',
    res='R_oss',
    dep_vars=['xi'],
    dep_vals=[],
    fun=lambda ctx: lambda xi: -xi,
    der_funs=[lambda ctx: lambda xi: np.full_like(xi, -1.0)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R_OSS_div = NonLinearTerm(
    name='R_OSS_div',
    description='OSS projection eq div: +(η_h, dp_drho·(1−θ)·(∂jx/∂x + ∂jy/∂y))',
    res='R_oss',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'd_dx_jx', 'd_dy_jy'],
    fun=lambda ctx: lambda theta: ctx['dp_drho']() * (1.0 - theta) * (ctx['d_dx_jx']() + ctx['d_dy_jy']()),
    der_funs=[lambda ctx: lambda theta: -ctx['dp_drho']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)


# ---------------------------------------------------------------------------
# mass residual — Laplacian-type terms: +(a_vec·∇q_h, τ·a_vec·∇θ_h)
#
# Assembly ('x','x') gives −∫ ∂N_res/∂x · ∂N_var/∂x · fun dΩ, so fun
# carries a minus sign to yield the desired positive contribution.
# ---------------------------------------------------------------------------

R_OSS_mass_xx = NonLinearTerm(
    name='R_OSS_mass_xx',
    description='OSS mass Laplacian xx: +(∂q/∂x, tau_a_x · a_vec_x · ∂θ/∂x)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'a_vec_x'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_x']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_x']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R_OSS_mass_yy = NonLinearTerm(
    name='R_OSS_mass_yy',
    description='OSS mass Laplacian yy: +(∂q/∂y, tau_a_y · a_vec_y · ∂θ/∂y)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'a_vec_y'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_y']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_y']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

R_OSS_mass_xy = NonLinearTerm(
    name='R_OSS_mass_xy',
    description='OSS mass Laplacian xy: +(∂q/∂y, tau_a_y · a_vec_x · ∂θ/∂x)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'a_vec_x'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_x']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_x']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='y')

R_OSS_mass_yx = NonLinearTerm(
    name='R_OSS_mass_yx',
    description='OSS mass Laplacian yx: +(∂q/∂x, tau_a_x · a_vec_y · ∂θ/∂y)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'a_vec_y'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_y']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_y']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='x')


# ---------------------------------------------------------------------------
# mass residual — divergence-type terms: +(a_vec·∇q_h, τ·(1−θ)·∇·j)
#
# Assembly ('none','x') gives −∫ ∂N_res/∂x · N_var · fun dΩ, so fun carries
# a minus sign to yield the desired positive contribution.
# dep_var is theta (differentiated); d_dx_jx, d_dy_jy and tau_a_* are frozen.
# ---------------------------------------------------------------------------

R_OSS_mass_div_x = NonLinearTerm(
    name='R_OSS_mass_div_x',
    description='OSS mass div x: +(∂q/∂x, tau_a_x · dp_drho · (1−θ) · ∇·j)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'dp_drho', 'd_dx_jx', 'd_dy_jy'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['dp_drho']() * (1.0 - theta) * (ctx['d_dx_jx']() + ctx['d_dy_jy']()),
    der_funs=[lambda ctx: lambda theta: ctx['tau_a_x']() * ctx['dp_drho']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R_OSS_mass_div_y = NonLinearTerm(
    name='R_OSS_mass_div_y',
    description='OSS mass div y: +(∂q/∂y, tau_a_y · dp_drho · (1−θ) · ∇·j)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'dp_drho', 'd_dx_jx', 'd_dy_jy'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['dp_drho']() * (1.0 - theta) * (ctx['d_dx_jx']() + ctx['d_dy_jy']()),
    der_funs=[lambda ctx: lambda theta: ctx['tau_a_y']() * ctx['dp_drho']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')


# ---------------------------------------------------------------------------
# mass residual — divq terms: +(dp_drho·(1−θ)·∇·j · q_h, τ·a_vec·∇θ)
#
# Test side: dp_drho·(1−θ)·∇·j frozen as coefficient via dep_val 'one_minus_theta'.
# Trial side: ∂θ/∂x or ∂θ/∂y via d_dx_resfun/d_dy_resfun.
# Pattern ('x', False) → +∫ N_res · fun · ∂θ/∂x dΩ.
# ---------------------------------------------------------------------------

R_OSS_mass_divq_x = NonLinearTerm(
    name='R_OSS_mass_divq_x',
    description='OSS mass divq x: +(dp_drho·(1−θ)·∇·j · q, tau_a_x · ∂θ/∂x)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'dp_drho', 'one_minus_theta', 'd_dx_jx', 'd_dy_jy'],
    fun=lambda ctx: lambda theta: ctx['tau_a_x']() * ctx['dp_drho']() * ctx['one_minus_theta']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']()) * theta,
    der_funs=[lambda ctx: lambda theta: ctx['tau_a_x']() * ctx['dp_drho']() * ctx['one_minus_theta']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']())],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R_OSS_mass_divq_y = NonLinearTerm(
    name='R_OSS_mass_divq_y',
    description='OSS mass divq y: +(dp_drho·(1−θ)·∇·j · q, tau_a_y · ∂θ/∂y)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'dp_drho', 'one_minus_theta', 'd_dx_jx', 'd_dy_jy'],
    fun=lambda ctx: lambda theta: ctx['tau_a_y']() * ctx['dp_drho']() * ctx['one_minus_theta']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']()) * theta,
    der_funs=[lambda ctx: lambda theta: ctx['tau_a_y']() * ctx['dp_drho']() * ctx['one_minus_theta']() * (ctx['d_dx_jx']() + ctx['d_dy_jy']())],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)


# ---------------------------------------------------------------------------
# mass residual — correction terms: −(a_vec·∇q_h, τ·ξ_h)
#
# Assembly ('none','x') gives −∫ ∂N_res/∂x · N_var · fun dΩ, so fun = +tau_a_x · xi
# to yield the desired −∫ ∂N_res/∂x · tau_a_x · ξ dΩ.
# ---------------------------------------------------------------------------

R_OSS_corrx = NonLinearTerm(
    name='R_OSS_corrx',
    description='OSS mass correction x: −(∂q/∂x, oss_correction_alpha · tau_a_x · ξ)',
    res='mass',
    dep_vars=['xi'],
    dep_vals=['tau_a_x', 'oss_correction_alpha'],
    fun=lambda ctx: lambda xi: ctx['oss_correction_alpha']() * ctx['tau_a_x']() * xi,
    der_funs=[lambda ctx: lambda xi: ctx['oss_correction_alpha']() * ctx['tau_a_x']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R_OSS_corry = NonLinearTerm(
    name='R_OSS_corry',
    description='OSS mass correction y: −(∂q/∂y, oss_correction_alpha · tau_a_y · ξ)',
    res='mass',
    dep_vars=['xi'],
    dep_vals=['tau_a_y', 'oss_correction_alpha'],
    fun=lambda ctx: lambda xi: ctx['oss_correction_alpha']() * ctx['tau_a_y']() * xi,
    der_funs=[lambda ctx: lambda xi: ctx['oss_correction_alpha']() * ctx['tau_a_y']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')


def get_oss_terms():
    """Return all OSS term instances."""
    return [
        R_OSS_advx, R_OSS_advy, R_OSS_div, R_OSS_proj,
        R_OSS_mass_xx, R_OSS_mass_yy, R_OSS_mass_xy, R_OSS_mass_yx,
        R_OSS_mass_div_x, R_OSS_mass_div_y,
        R_OSS_mass_divq_x, R_OSS_mass_divq_y,
        R_OSS_corrx, R_OSS_corry,
    ]


OSS_TERM_NAMES = [
    'R_OSS_advx', 'R_OSS_advy', 'R_OSS_proj', #'R_OSS_div', 
    'R_OSS_mass_xx', 'R_OSS_mass_yy', #'R_OSS_mass_xy', 'R_OSS_mass_yx',  # theta-theta cross
    #'R_OSS_mass_div_x', 'R_OSS_mass_div_y',   # a_vec·∇q vs dp_drho·(1−θ)·∇·j
    #'R_OSS_mass_divq_x', 'R_OSS_mass_divq_y', # dp_drho·(1−θ)·∇·j·q vs a_vec·∇θ
    #'R_OSS_corrx', 'R_OSS_corry',
]
