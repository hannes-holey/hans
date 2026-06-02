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

from typing import Callable
from typing import List
import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class NonLinearTerm():
    def __init__(self,
                 name: str,
                 description: str,
                 res: str,
                 dep_vars: list[str],
                 dep_vals: list[str],
                 fun: Callable,
                 der_funs: list[Callable],
                 trial_deriv: 'str | list[str | None] | None' = None,
                 test_deriv: 'str | None' = None):
        self.name = name
        self.description = description
        self.res = res
        self.dep_vars = dep_vars
        self.dep_vals = dep_vals
        self.fun_ = fun
        self.der_funs_ = der_funs
        self.trial_deriv = trial_deriv
        self.test_deriv = test_deriv
        self.built = False

    def build(self, ctx: dict) -> None:
        self.fun = self.fun_(ctx)
        self.der_funs = [der_fun_(ctx) for der_fun_ in self.der_funs_]
        self.built = True

    def evaluate(self, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built")
        return self.fun(*args)

    def evaluate_deriv(self, dep_var: str, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built")
        i = self.dep_vars.index(dep_var)
        return self.der_funs[i](*args)

    def depvar_deriv_for(self, var: str) -> str:
        """Direction of spatial derivative acting on dep_var 'var': 'none', 'x', or 'y'."""
        i = self.dep_vars.index(var)
        d = self.trial_deriv[i] if isinstance(self.trial_deriv, list) else self.trial_deriv
        return d if d is not None else 'none'

    @property
    def depvar_deriv(self):
        """Direction of spatial derivative acting on dep_var: 'none', 'x', or 'y'.
        All dep_vars must share the same derivative direction (use depvar_deriv_for otherwise)."""
        td = self.trial_deriv
        if isinstance(td, list):
            directions = [d for d in td if d is not None]
            return directions[0] if directions else 'none'
        return td if td is not None else 'none'

    @property
    def deriv_key(self):
        """Return (depvar_deriv, test_deriv) tuple for template lookup.
        Only valid when all dep_vars share the same derivative direction."""
        return (self.depvar_deriv, self.test_deriv)


from .terms_theta import (  # noqa: F401
    R11x_fb, R11y_fb, R11Sx_fb, R11Sy_fb,
    R11x_fb_corr, R11y_fb_corr, R11Sx_fb_corr, R11Sy_fb_corr,
    R_FB, R1STx, R1STy, R24x_fb, R24y_fb,
)
from .terms_oss import _OSS_TERMS  # noqa: F401

# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

R11x = NonLinearTerm(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']()],
    trial_deriv='x')

R11y = NonLinearTerm(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']()],
    trial_deriv='y')

R11Sx = NonLinearTerm(
    name='R11Sx',
    description='flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']()])

R11Sy = NonLinearTerm(
    name='R11Sy',
    description='flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy', 'dp_drho'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']()])

R11x_corr = NonLinearTerm(
    name='R11x_corr',
    description='flux divergence x Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dx_jx'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dx_jx']()])

R11y_corr = NonLinearTerm(
    name='R11y_corr',
    description='flux divergence y Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dy_jy'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dy_jy']()])

R1T = NonLinearTerm(
    name='R1T',
    description='time derivative',
    res='mass',
    dep_vars=['p'],
    dep_vals=['drho_dp'],
    fun=lambda ctx: lambda p: - (p - ctx['p_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: - np.ones_like(p) / ctx['dt']()])

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

R21x = NonLinearTerm(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='x')

R21y = NonLinearTerm(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='y')

R22xx = NonLinearTerm(
    name='R22xx',
    description='convective momentum flux jx*jx in x (IBP)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx: -(jx * jx) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: (jx * jx) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: -2 * jx / ctx['rho']()
    ],
    test_deriv='x')

R22xxS = NonLinearTerm(
    name='R22xxS',
    description='convective momentum flux jx*jx height source',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'dh_dx', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: -1 / ctx['h']() * ctx['dh_dx']() * 2 * jx / ctx['rho']()
    ])

R22yx = NonLinearTerm(
    name='R22yx',
    description='convective momentum flux jx*jy in y (for momentum_x, IBP)',
    res='momentum_x',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -(jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -jx / ctx['rho']()
    ],
    test_deriv='y')

R22yxS = NonLinearTerm(
    name='R22yxS',
    description='convective momentum flux jx*jy height source (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['h', 'dh_dy', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jx / ctx['rho']()
    ])

R22xy = NonLinearTerm(
    name='R22xy',
    description='convective momentum flux jx*jy in x (for momentum_y, IBP)',
    res='momentum_y',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -(jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -jx / ctx['rho']()
    ],
    test_deriv='x')

R22xyS = NonLinearTerm(
    name='R22xyS',
    description='convective momentum flux jx*jy height source (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['h', 'dh_dx', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jx / ctx['rho']()
    ])

R22yy = NonLinearTerm(
    name='R22yy',
    description='convective momentum flux jy*jy in y (IBP)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jy: -(jy * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: (jy * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: -2 * jy / ctx['rho']()
    ],
    test_deriv='y')

R22yyS = NonLinearTerm(
    name='R22yyS',
    description='convective momentum flux jy*jy height source',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'dh_dy', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: -1 / ctx['h']() * ctx['dh_dy']() * 2 * jy / ctx['rho']()
    ])

R23xy = NonLinearTerm(
    name='R23xy',
    description='shear viscous stress tau_xy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * jx / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R23yx = NonLinearTerm(
    name='R23yx',
    description='shear viscous stress tau_xy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * jy / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23xx = NonLinearTerm(
    name='R23xx',
    description='shear viscous stress tau_xx in x (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * jx / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23yy = NonLinearTerm(
    name='R23yy',
    description='shear viscous stress tau_yy in y (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * jy / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R24x = NonLinearTerm(
    name='R24x',
    description='wall stress x',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()])

R24y = NonLinearTerm(
    name='R24y',
    description='wall stress y',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()])

R25x = NonLinearTerm(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_x'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_x'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_x']()])

R25y = NonLinearTerm(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_y'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_y'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_y']()])

R2Tx = NonLinearTerm(
    name='R2Tx',
    description='time derivative momentum_x',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']()])

R2Ty = NonLinearTerm(
    name='R2Ty',
    description='time derivative momentum_y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']()])

# -----------------------------------------------------------------------------
# Energy equation terms (R3*)
# -----------------------------------------------------------------------------

R31x = NonLinearTerm(
    name='R31x',
    description='energy convection x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E,
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E,
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jx, E: - (jx / rho)],
    trial_deriv='x')

R31y = NonLinearTerm(
    name='R31y',
    description='energy convection y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E,
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E,
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jy, E: - (jy / rho)],
    trial_deriv='y')

R31Sx = NonLinearTerm(
    name='R31Sx',
    description='energy convection height source x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']())])

R31Sy = NonLinearTerm(
    name='R31Sy',
    description='energy convection height source y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']())])

R32x = NonLinearTerm(
    name='R32x',
    description='pressure work x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho),
    der_funs=[lambda ctx: lambda rho, jx: - (ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho)],
    trial_deriv='x')

R32y = NonLinearTerm(
    name='R32y',
    description='pressure work y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho),
    der_funs=[lambda ctx: lambda rho, jy: - (ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho)],
    trial_deriv='y')

R32Sx = NonLinearTerm(
    name='R32Sx',
    description='pressure work height source x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx: - ((ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)) * (1 / ctx['h']() * ctx['dh_dx']())),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dx']())])

R32Sy = NonLinearTerm(
    name='R32Sy',
    description='pressure work height source y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy: - ((ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)) * (1 / ctx['h']() * ctx['dh_dy']())),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dy']())])

R34 = NonLinearTerm(
    name='R34',
    description='wall stress work',
    res='energy',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'tau_xz_bot', 'tau_yz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx',
              'dtau_yz_bot_drho', 'dtau_yz_bot_djy', 'U_bot', 'V_bot'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * (ctx['tau_xz_bot']() * ctx['U_bot']() + ctx['tau_yz_bot']() * ctx['V_bot']()),
    der_funs=[lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * (ctx['dtau_xz_bot_drho']() * ctx['U_bot']() + ctx['dtau_yz_bot_drho']() * ctx['V_bot']()),
              lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dtau_xz_bot_djx']() * ctx['U_bot'](),
              lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dtau_yz_bot_djy']() * ctx['V_bot']()])

R35x = NonLinearTerm(
    name='R35x',
    description='thermal diffusion x',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    test_deriv='x')

R35y = NonLinearTerm(
    name='R35y',
    description='thermal diffusion y',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    test_deriv='y')

R36 = NonLinearTerm(
    name='R36',
    description='wall heat balance',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: ctx['S'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: ctx['dS_drho'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djx'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djy'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_dE']()])

R3T = NonLinearTerm(
    name='R3T',
    description='energy time derivative',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']()])

# Master list of all physical terms.
term_list = [
    # Mass equation
    R11x, R11y, R11Sx, R11Sy,
    R11x_corr, R11y_corr,
    R1T,
    # Elrod-Adams / FB cavitation (replaces R11* when cavitation: true)
    R11x_fb, R11y_fb, R11Sx_fb, R11Sy_fb,
    R11x_fb_corr, R11y_fb_corr, R11Sx_fb_corr, R11Sy_fb_corr,
    R_FB, R1STx, R1STy,
    # Momentum equation
    R21x, R21y,
    R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx, R23xx, R23yy,
    R24x, R24y,
    R24x_fb, R24y_fb,
    R25x, R25y,
    R2Tx, R2Ty,
    # Energy equation
    R3T,
    R34,
    R31x, R31y, R31Sx, R31Sy,
    R32x, R32y, R32Sx, R32Sy,
    R35x, R35y,
    R36,
    # OSS stabilization
    *_OSS_TERMS,
]


def _term_names_from_physics(fem_solver: dict) -> List[str]:
    """Build term name list from physics flags.

    Physics flags (in fem_solver['physics']):
    - gap_shear:          Gap-averaged wall shear τ/h (R24x, R24y; R24x_fb, R24y_fb with cavitation)
    - plane_shear:        In-plane viscous diffusion (R23xy, R23yx)
    - inertia:            Momentum convection (R22*)
    - body_force:         Body force (R25x, R25y)
    - energy:             Energy equation master switch, subflags below default to True
    - energy_convection:  Energy advection (R31*)
    - pressure_work:      Pressure-volume work (R32*)
    - thermal_diffusion:  Heat conduction (R35x, R35y)
    - wall_heat_balance:  Wall heat flux BC (R36)
    - wall_shear_work:    Wall stress work / shear heating (R34)
    """

    physics = fem_solver['physics']
    equations = fem_solver['equations']
    cavitation = equations['cavitation']

    if cavitation:
        terms = [
            'R11x_fb', 'R11y_fb', 'R11Sx_fb', 'R11Sy_fb',
            'R11x_fb_corr', 'R11y_fb_corr', 'R11Sx_fb_corr', 'R11Sy_fb_corr',
            'R_FB',
            'R1T',
            'R21x', 'R21y', 'R2Tx', 'R2Ty',
        ]
    else:
        terms = [
            'R11x', 'R11y', 'R11Sx', 'R11Sy',
            'R11x_corr', 'R11y_corr', 'R11Sx_corr', 'R11Sy_corr',
            'R1T',
            'R21x', 'R21y', 'R2Tx', 'R2Ty',
        ]

    if cavitation and physics['theta_stab']:
        terms.extend(['R1STx', 'R1STy'])

    if cavitation and physics['oss_theta']:
        from .terms_oss import OSS_TERM_NAMES
        terms.extend(OSS_TERM_NAMES)

    if physics['gap_shear']:
        if cavitation:
            terms.extend(['R24x_fb', 'R24y_fb'])
        else:
            terms.extend(['R24x', 'R24y'])

    if physics['plane_shear']:
        terms.extend(['R23xy', 'R23yx', 'R23xx', 'R23yy'])

    if physics['inertia']:
        terms.extend(['R22xx', 'R22xxS', 'R22yx', 'R22yxS', 'R22xy', 'R22xyS', 'R22yy', 'R22yyS'])

    if physics['body_force']:
        terms.extend(['R25x', 'R25y'])

    if physics['energy']:
        terms.append('R3T')

        if physics['wall_shear_work']:
            terms.append('R34')

        if physics['energy_convection']:
            terms.extend(['R31x', 'R31y', 'R31Sx', 'R31Sy'])

        if physics['pressure_work']:
            terms.extend(['R32x', 'R32y', 'R32Sx', 'R32Sy'])

        if physics['thermal_diffusion']:
            terms.extend(['R35x', 'R35y'])

        if physics['wall_heat_balance']:
            terms.append('R36')

    return terms


def get_active_terms(fem_solver: dict) -> List['NonLinearTerm']:
    """Return active NonLinearTerm instances based on fem_solver config.

    If fem_solver['equations']['term_list'] is set, use that explicit list.
    Otherwise auto-select from physics flags via _term_names_from_physics().
    """
    user_terms = fem_solver['equations']['term_list']
    if user_terms is not None:
        requested = set(user_terms)
    else:
        requested = set(_term_names_from_physics(fem_solver))

    return [t for t in term_list if t.name in requested]
