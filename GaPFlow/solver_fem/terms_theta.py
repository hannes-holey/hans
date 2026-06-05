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

import numpy as np

from .terms import Term


def _fb_denom(a, b):
    """sqrt(a²+b²), bumped to machine epsilon at (0,0) to avoid singularity."""
    d = np.sqrt(a**2 + b**2)
    return np.where(d == 0.0, np.finfo(float).eps, d)


# -----------------------------------------------------------------------------
# Elrod-Adams flux divergence (replaces R11x/y/Sx/Sy when cavitation: true)
# -----------------------------------------------------------------------------

R11x_fb = Term(
    name='R11x_fb',
    description='flux divergence x Elrod-Adams (IBP)',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jx, theta: -ctx['dp_drho']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['dp_drho']() * (1 - theta),
        lambda ctx: lambda jx, theta:  ctx['dp_drho']() * jx,
    ],
    test_deriv='x')

R11y_fb = Term(
    name='R11y_fb',
    description='flux divergence y Elrod-Adams (IBP)',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jy, theta: -ctx['dp_drho']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta: -ctx['dp_drho']() * (1 - theta),
        lambda ctx: lambda jy, theta:  ctx['dp_drho']() * jy,
    ],
    test_deriv='y')

R11Sx_fb = Term(
    name='R11Sx_fb',
    description='flux divergence height source x Elrod-Adams',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta:  ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * jx,
    ])

R11Sy_fb = Term(
    name='R11Sy_fb',
    description='flux divergence height source y Elrod-Adams',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['h', 'dh_dy', 'dp_drho'],
    fun=lambda ctx: lambda jy, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * (1 - theta),
        lambda ctx: lambda jy, theta:  ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * jy,
    ])

# Jacobian correction for the implicit p-dependence of dp_drho in R11*_fb.
# Zero residual contribution — Jacobian-only.
R11x_fb_corr = Term(
    name='R11x_fb_corr',
    description='flux divergence x FB Jacobian correction (d(dp_drho)/dp, theta-weighted)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jx', 'theta'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) * ctx['jx']()],
    test_deriv='x')

R11y_fb_corr = Term(
    name='R11y_fb_corr',
    description='flux divergence y FB Jacobian correction (d(dp_drho)/dp, theta-weighted)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jy', 'theta'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) * ctx['jy']()],
    test_deriv='y')

R11Sx_fb_corr = Term(
    name='R11Sx_fb_corr',
    description='flux divergence height source x FB Jacobian correction',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jx', 'h', 'dh_dx', 'theta'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) / ctx['h']() * ctx['dh_dx']() * ctx['jx']()])

R11Sy_fb_corr = Term(
    name='R11Sy_fb_corr',
    description='flux divergence height source y FB Jacobian correction',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jy', 'h', 'dh_dy', 'theta'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) / ctx['h']() * ctx['dh_dy']() * ctx['jy']()])


# -----------------------------------------------------------------------------
# Fischer-Burmeister complementarity condition
# -----------------------------------------------------------------------------

R_fb = Term(
    name='R_FB',
    description='Fischer-Burmeister complementarity condition (p normalized by P0)',
    res='fb',
    dep_vars=['p', 'theta'],
    dep_vals=[],
    fun=lambda ctx: lambda p, theta: (
        lambda a_nd: np.sqrt(a_nd**2 + theta**2) - a_nd - theta
    )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
    der_funs=[
        lambda ctx: lambda p, theta: (
            lambda a_nd: (a_nd / _fb_denom(a_nd, theta) - 1.0) / ctx['fb_p_ref']()
        )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
        lambda ctx: lambda p, theta: (
            lambda a_nd: theta / _fb_denom(a_nd, theta) - 1.0
        )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
    ])


# -----------------------------------------------------------------------------
# Artificial diffusion stabilization (stabilization.ad: true)
# -----------------------------------------------------------------------------

R1STx = Term(
    name='R1STx',
    description='theta diffusion stabilization in mass equation x',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jx', 'jy', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: -(ctx['ad_alpha']() * ctx['dp_drho']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * ctx['d_dx_theta']()),
    der_funs=[lambda ctx: lambda theta: -(ctx['ad_alpha']() * ctx['dp_drho']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    trial_deriv='x',
    test_deriv='x')

R1STy = Term(
    name='R1STy',
    description='theta diffusion stabilization in mass equation y',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jx', 'jy', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: -(ctx['ad_alpha']() * ctx['dp_drho']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * ctx['d_dy_theta']()),
    der_funs=[lambda ctx: lambda theta: -(ctx['ad_alpha']() * ctx['dp_drho']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    trial_deriv='y',
    test_deriv='y')


# -----------------------------------------------------------------------------
# Wall stress with theta-dependent effective density (replaces R24x/y when cavitation: true)
# -----------------------------------------------------------------------------

R24x_fb = Term(
    name='R24x_fb',
    description='wall stress x with theta-dependent effective density',
    res='momentum_x',
    dep_vars=['p', 'jx', 'theta'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'dtau_xz_dtheta', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_dtheta'](),
    ])

R24y_fb = Term(
    name='R24y_fb',
    description='wall stress y with theta-dependent effective density',
    res='momentum_y',
    dep_vars=['p', 'jy', 'theta'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'dtau_yz_dtheta', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_dtheta'](),
    ])


THETA_TERMS_MASS = [
    R11x_fb, R11y_fb, R11Sx_fb, R11Sy_fb,
    R11x_fb_corr, R11y_fb_corr, R11Sx_fb_corr, R11Sy_fb_corr,
    R1STx, R1STy,
]

R_fb = [R_fb]

THETA_TERMS_WALL_STRESS = [R24x_fb, R24y_fb]
