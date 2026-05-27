#
# Copyright 2025 Christoph Huber
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

"""SUPG stabilization terms for the mass (Elrod-Adams / FB) equation.

Adds the streamline-upwind Petrov-Galerkin term:

    τ · ∫ (j·∇Nᵢ) · L_mass dΩ  =  0

with strong-form residual:

    L_mass = drho_dp·(p − p_prev)/dt
           − dp_drho · [(1−θ)·∇·j  −  j·∇θ]
           − dp_drho · (1/h)·∇h·(1−θ)·j

τ and the coefficients tau_jx = τ·jx, tau_jy = τ·jy are computed as
quad fields in update_quad_computed (QuadFieldManager), with:

    τ = supg_theta_alpha · min(dx,dy) / (2·(|j| + eps))

supg_theta_alpha is a dimensionless O(1) tuning parameter. The dp_drho
factor from L_mass is included explicitly in each group 1/2/4 term lambda
so that alpha=1 corresponds to standard SUPG scaling.

All terms use res='mass'. Activated by physics flag supg_theta: true in the YAML.

Assembly conventions
--------------------
- assemble_rhs:    always calls term.evaluate() with field substitution
                   (d_dx_v substituted for v when depvar_deriv_for(v) != 'none')
- assemble_matrix: always calls term.evaluate_deriv() with plain quad_fields[v]
- Spatial derivatives of dep_vars needed inside der_funs are fetched via
  ctx['d_xi_v']() from dep_vals — not passed as arguments.

Groups
------
Group 1: θ-advection    τ·(jk·∂Nᵢ/∂xk) · dp_drho · ji · ∂θ/∂xi
Group 2: flux-divergence τ·(jk·∂Nᵢ/∂xk) · dp_drho · (1−θ) · ∂ji/∂xi
Group 3: time term       τ·(jk·∂Nᵢ/∂xk) · drho_dp/dt · (p − p_prev)
Group 4: height source   τ·(jk·∂Nᵢ/∂xk) · dp_drho · (1/h) · ∂h/∂xm · (1−θ) · jm
"""

import numpy as np


def _get_NonLinearTerm():
    from .terms import NonLinearTerm
    return NonLinearTerm


# ===========================================================================
# Group 1: θ-advection  τ·(jk·∂Nᵢ/∂xk) · dp_drho · ji · ∂θ/∂xi
#
# dep_vars=['ji','theta']: ji plain, theta differentiated (rhs substitutes d_xi_theta).
# fun receives (ji, d_xi_theta); der_funs receive plain (ji, theta).
# tau_jk and dp_drho fetched lazily every evaluation.
# ===========================================================================

def _make_supg_g1():
    NLT = _get_NonLinearTerm()

    R1SUPGxx = NLT(
        name='R1SUPGxx',
        description='SUPG mass group1 xx | τ·(jx·∂Nᵢ/∂x)·dp_drho·jx·∂θ/∂x',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jx', 'dp_drho'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']:
                lambda jx, dtheta: -_g() * _c() * jx * dtheta
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda jx, dtheta: -_g() * _c() * dtheta)(),
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda jx, dtheta: -_g() * _c() * jx)(),
        ],
        d_dx_resfun=[False, True],
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGyy = NLT(
        name='R1SUPGyy',
        description='SUPG mass group1 yy | τ·(jy·∂Nᵢ/∂y)·dp_drho·jy·∂θ/∂y',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jy', 'dp_drho'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']:
                lambda jy, dtheta: -_g() * _c() * jy * dtheta
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda jy, dtheta: -_g() * _c() * dtheta)(),
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda jy, dtheta: -_g() * _c() * jy)(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=[False, True],
        der_testfun='y')

    R1SUPGxy = NLT(
        name='R1SUPGxy',
        description='SUPG mass group1 xy | τ·(jy·∂Nᵢ/∂y)·dp_drho·jx·∂θ/∂x',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jy', 'dp_drho'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']:
                lambda jx, dtheta: -_g() * _c() * jx * dtheta
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda jx, dtheta: -_g() * _c() * dtheta)(),
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda jx, dtheta: -_g() * _c() * jx)(),
        ],
        d_dx_resfun=[False, True],
        d_dy_resfun=False,
        der_testfun='y')

    R1SUPGyx = NLT(
        name='R1SUPGyx',
        description='SUPG mass group1 yx | τ·(jx·∂Nᵢ/∂x)·dp_drho·jy·∂θ/∂y',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jx', 'dp_drho'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']:
                lambda jy, dtheta: -_g() * _c() * jy * dtheta
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda jy, dtheta: -_g() * _c() * dtheta)(),
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda jy, dtheta: -_g() * _c() * jy)(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=[False, True],
        der_testfun='x')

    return [R1SUPGxx, R1SUPGyy, R1SUPGxy, R1SUPGyx]


# ===========================================================================
# Group 2: flux-divergence  τ·(jk·∂Nᵢ/∂xk) · dp_drho · (1−θ) · ∂ji/∂xi
#
# dep_vars=['ji','theta']: ji differentiated (rhs substitutes d_xi_ji),
#                          theta plain.
# d_xi_ji fetched lazily for use in der_funs[1].
# fun receives (d_xi_ji, theta); der_funs receive plain (ji, theta).
# ===========================================================================

def _make_supg_g2():
    NLT = _get_NonLinearTerm()

    R1SUPGdivxx = NLT(
        name='R1SUPGdivxx',
        description='SUPG mass group2 divxx | τ·(jx·∂Nᵢ/∂x)·dp_drho·(1−θ)·∂jx/∂x',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jx', 'dp_drho', 'd_dx_jx'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']:
                lambda djx, theta: -_g() * _c() * (1 - theta) * djx
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda djx, theta: -_g() * _c() * (1 - theta))(),
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _d=ctx['d_dx_jx']: lambda djx, theta: _g() * _c() * _d())(),
        ],
        d_dx_resfun=[True, False],
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGdivyy = NLT(
        name='R1SUPGdivyy',
        description='SUPG mass group2 divyy | τ·(jy·∂Nᵢ/∂y)·dp_drho·(1−θ)·∂jy/∂y',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jy', 'dp_drho', 'd_dy_jy'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']:
                lambda djy, theta: -_g() * _c() * (1 - theta) * djy
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda djy, theta: -_g() * _c() * (1 - theta))(),
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _d=ctx['d_dy_jy']: lambda djy, theta: _g() * _c() * _d())(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=[True, False],
        der_testfun='y')

    R1SUPGdivyx = NLT(
        name='R1SUPGdivyx',
        description='SUPG mass group2 divyx | τ·(jx·∂Nᵢ/∂x)·dp_drho·(1−θ)·∂jy/∂x',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jx', 'dp_drho', 'd_dx_jy'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']:
                lambda djy, theta: -_g() * _c() * (1 - theta) * djy
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho']: lambda djy, theta: -_g() * _c() * (1 - theta))(),
            lambda ctx: (lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _d=ctx['d_dx_jy']: lambda djy, theta: _g() * _c() * _d())(),
        ],
        d_dx_resfun=[True, False],
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGdivxy = NLT(
        name='R1SUPGdivxy',
        description='SUPG mass group2 divxy | τ·(jy·∂Nᵢ/∂y)·dp_drho·(1−θ)·∂jx/∂y',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jy', 'dp_drho', 'd_dy_jx'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']:
                lambda djx, theta: -_g() * _c() * (1 - theta) * djx
        )(),
        der_funs=[
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho']: lambda djx, theta: -_g() * _c() * (1 - theta))(),
            lambda ctx: (lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _d=ctx['d_dy_jx']: lambda djx, theta: _g() * _c() * _d())(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=[True, False],
        der_testfun='y')

    return [R1SUPGdivxx, R1SUPGdivyy, R1SUPGdivyx, R1SUPGdivxy]


# ===========================================================================
# Group 3: time term  τ·(jk·∂Nᵢ/∂xk) · drho_dp/dt · (p − p_prev)
#
# dep_vars=['p'], no spatial derivative on trial function.
# tau_jk, drho_dp, p_prev fetched lazily every evaluation.
# No dp_drho factor here — the strong-form time term carries drho_dp directly.
# ===========================================================================

def _make_supg_g3():
    NLT = _get_NonLinearTerm()

    R1SUPGTx = NLT(
        name='R1SUPGTx',
        description='SUPG mass group3 Tx | τ·(jx·∂Nᵢ/∂x)·drho_dp/dt·(p−p_prev)',
        res='mass',
        dep_vars=['p'],
        dep_vals=['tau_jx', 'drho_dp', 'dt', 'p_prev'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _dr=ctx['drho_dp'], _dt=ctx['dt'], _pp=ctx['p_prev']:
                lambda p: -_g() * _dr() / _dt() * (p - _pp())
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jx'], _dr=ctx['drho_dp'], _dt=ctx['dt']:
                    lambda p: -_g() * _dr() / _dt() * np.ones_like(p)
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGTy = NLT(
        name='R1SUPGTy',
        description='SUPG mass group3 Ty | τ·(jy·∂Nᵢ/∂y)·drho_dp/dt·(p−p_prev)',
        res='mass',
        dep_vars=['p'],
        dep_vals=['tau_jy', 'drho_dp', 'dt', 'p_prev'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _dr=ctx['drho_dp'], _dt=ctx['dt'], _pp=ctx['p_prev']:
                lambda p: -_g() * _dr() / _dt() * (p - _pp())
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jy'], _dr=ctx['drho_dp'], _dt=ctx['dt']:
                    lambda p: -_g() * _dr() / _dt() * np.ones_like(p)
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='y')

    return [R1SUPGTx, R1SUPGTy]


# ===========================================================================
# Group 4: height source  τ·(jk·∂Nᵢ/∂xk) · dp_drho · (1/h) · ∂h/∂xm · (1−θ) · jm
#
# 4 terms. No spatial derivative on trial functions.
# tau_jk, dp_drho, h, dh_dx/dy, jm all fetched lazily.
# dep_vars=['jm','theta']: jm plain, theta plain.
# ===========================================================================

def _make_supg_g4():
    NLT = _get_NonLinearTerm()

    R1SUPGhxx = NLT(
        name='R1SUPGhxx',
        description='SUPG mass group4 hxx | τ·(jx·∂Nᵢ/∂x)·dp_drho·(1/h)·∂h/∂x·(1−θ)·jx',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jx', 'dp_drho', 'h', 'dh_dx', 'jx'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx']:
                lambda jx, theta: -_g() * _c() / _h() * _dh() * (1 - theta) * jx
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx']:
                    lambda jx, theta: -_g() * _c() / _h() * _dh() * (1 - theta)
            )(),
            lambda ctx: (
                lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx'], _jx=ctx['jx']:
                    lambda jx, theta: _g() * _c() / _h() * _dh() * _jx()
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGhxy = NLT(
        name='R1SUPGhxy',
        description='SUPG mass group4 hxy | τ·(jx·∂Nᵢ/∂x)·dp_drho·(1/h)·∂h/∂y·(1−θ)·jy',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jx', 'dp_drho', 'h', 'dh_dy', 'jy'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy']:
                lambda jy, theta: -_g() * _c() / _h() * _dh() * (1 - theta) * jy
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy']:
                    lambda jy, theta: -_g() * _c() / _h() * _dh() * (1 - theta)
            )(),
            lambda ctx: (
                lambda _g=ctx['tau_jx'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy'], _jy=ctx['jy']:
                    lambda jy, theta: _g() * _c() / _h() * _dh() * _jy()
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='x')

    R1SUPGhyx = NLT(
        name='R1SUPGhyx',
        description='SUPG mass group4 hyx | τ·(jy·∂Nᵢ/∂y)·dp_drho·(1/h)·∂h/∂x·(1−θ)·jx',
        res='mass',
        dep_vars=['jx', 'theta'],
        dep_vals=['tau_jy', 'dp_drho', 'h', 'dh_dx', 'jx'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx']:
                lambda jx, theta: -_g() * _c() / _h() * _dh() * (1 - theta) * jx
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx']:
                    lambda jx, theta: -_g() * _c() / _h() * _dh() * (1 - theta)
            )(),
            lambda ctx: (
                lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dx'], _jx=ctx['jx']:
                    lambda jx, theta: _g() * _c() / _h() * _dh() * _jx()
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='y')

    R1SUPGhyy = NLT(
        name='R1SUPGhyy',
        description='SUPG mass group4 hyy | τ·(jy·∂Nᵢ/∂y)·dp_drho·(1/h)·∂h/∂y·(1−θ)·jy',
        res='mass',
        dep_vars=['jy', 'theta'],
        dep_vals=['tau_jy', 'dp_drho', 'h', 'dh_dy', 'jy'],
        fun=lambda ctx: (
            lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy']:
                lambda jy, theta: -_g() * _c() / _h() * _dh() * (1 - theta) * jy
        )(),
        der_funs=[
            lambda ctx: (
                lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy']:
                    lambda jy, theta: -_g() * _c() / _h() * _dh() * (1 - theta)
            )(),
            lambda ctx: (
                lambda _g=ctx['tau_jy'], _c=ctx['dp_drho'], _h=ctx['h'], _dh=ctx['dh_dy'], _jy=ctx['jy']:
                    lambda jy, theta: _g() * _c() / _h() * _dh() * _jy()
            )(),
        ],
        d_dx_resfun=False,
        d_dy_resfun=False,
        der_testfun='y')

    return [R1SUPGhxx, R1SUPGhxy, R1SUPGhyx, R1SUPGhyy]


def get_supg_terms():
    """Return all SUPG term instances (all four groups)."""
    return (
        _make_supg_g1() +
        _make_supg_g4() +
        _make_supg_g2() +
        _make_supg_g3()
    )


SUPG_TERM_NAMES = [
    # Group 1
    'R1SUPGxx', 'R1SUPGyy', 'R1SUPGxy', 'R1SUPGyx',
    # Group 4
    'R1SUPGhxx', 'R1SUPGhxy', 'R1SUPGhyx', 'R1SUPGhyy',
    # Group 2
    'R1SUPGdivxx', 'R1SUPGdivyy', 'R1SUPGdivyx', 'R1SUPGdivxy',
    # Group 3
    'R1SUPGTx', 'R1SUPGTy',
]
