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


class Term():
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



# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

R11x = Term(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['dp_drho', 'd_dx_jx'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * ctx['d_dx_jx'](),
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']()],
    trial_deriv='x')

R11y = Term(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['dp_drho', 'd_dy_jy'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * ctx['d_dy_jy'](),
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']()],
    trial_deriv='y')

R11Sx = Term(
    name='R11Sx',
    description='flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']()])

R11Sy = Term(
    name='R11Sy',
    description='flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy', 'dp_drho'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']()])

R11x_corr = Term(
    name='R11x_corr',
    description='flux divergence x Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dx_jx'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dx_jx']()])

R11y_corr = Term(
    name='R11y_corr',
    description='flux divergence y Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dy_jy'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dy_jy']()])

R1T_old = Term(
    name='R1T',
    description='time derivative',
    res='mass',
    dep_vars=['p'],
    dep_vals=['p_prev'],
    fun=lambda ctx: lambda p: - (p - ctx['p_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: - np.ones_like(p) / ctx['dt']()])

R1T = Term(
    name='R1T',
    description='time derivative — density space, no h factor, pair with R1Th_working2',
    res='mass',
    dep_vars=['p'],
    dep_vals=['dp_drho', 'dp_drho_before', 'd2p_drho2', 'drho_dp', 'rho', 'rho_before'],
    fun=lambda ctx: lambda p: -0.5 * (ctx['dp_drho']() + ctx['dp_drho_before']()) * (ctx['rho']() - ctx['rho_before']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: -(0.5 * ctx['d2p_drho2']() * ctx['drho_dp']() * (ctx['rho']() - ctx['rho_before']()) + 0.5 * (ctx['dp_drho']() + ctx['dp_drho_before']()) * ctx['drho_dp']()) / ctx['dt']()])

R1Th = Term(
    name='R1Th_simplified',
    description='squeeze source — dp_drho_avg * rho_new / h_before * dh_dt (exact, pairs with plain R1T)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['dp_drho', 'dp_drho_before', 'd2p_drho2', 'drho_dp', 'rho', 'h_before', 'dh_dt'],
    fun=lambda ctx: lambda p: -0.5 * (ctx['dp_drho']() + ctx['dp_drho_before']()) * ctx['rho']() / ctx['h_before']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda p: -(0.5 * ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['rho']() +
                                      0.5 * (ctx['dp_drho']() + ctx['dp_drho_before']()) * ctx['drho_dp']()) / ctx['h_before']() * ctx['dh_dt']()])

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

R21x = Term(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='x')

R21y = Term(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='y')

R22xx = Term(
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

R22xxS = Term(
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

R22yx = Term(
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

R22yxS = Term(
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

R22xy = Term(
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

R22xyS = Term(
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

R22yy = Term(
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

R22yyS = Term(
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

R23xy = Term(
    name='R23xy',
    description='shear viscous stress tau_xy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dy_jx'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * ctx['d_dy_jx']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R23yx = Term(
    name='R23yx',
    description='shear viscous stress tau_xy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dx_jy'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * ctx['d_dx_jy']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23xx = Term(
    name='R23xx',
    description='shear viscous stress tau_xx in x (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dx_jx'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * ctx['d_dx_jx']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23yy = Term(
    name='R23yy',
    description='shear viscous stress tau_yy in y (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dy_jy'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * ctx['d_dy_jy']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R24x = Term(
    name='R24x',
    description='wall stress x',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()])

R24y = Term(
    name='R24y',
    description='wall stress y',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()])

R25x = Term(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_x'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_x'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_x']()])

R25y = Term(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_y'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_y'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_y']()])

R2Tx = Term(
    name='R2Tx',
    description='time derivative momentum_x',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=['jx_prev'],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']()])

R2Ty = Term(
    name='R2Ty',
    description='time derivative momentum_y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=['jy_prev'],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']()])

R2Thx = Term(
    name='R2Thx',
    description='squeeze source momentum_x (height rate of change)',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dt'],
    fun=lambda ctx: lambda jx: - jx / ctx['h']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda jx: - ctx['dh_dt']() / ctx['h']()])

R2Thy = Term(
    name='R2Thy',
    description='squeeze source momentum_y (height rate of change)',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dt'],
    fun=lambda ctx: lambda jy: - jy / ctx['h']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda jy: - ctx['dh_dt']() / ctx['h']()])

# -----------------------------------------------------------------------------
# Energy equation terms (R3*)
# -----------------------------------------------------------------------------

R31x = Term(
    name='R31x',
    description='energy convection x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['d_dx_rho', 'd_dx_jx', 'd_dx_E'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E,
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E,
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jx, E: - (jx / rho)],
    trial_deriv='x')

R31y = Term(
    name='R31y',
    description='energy convection y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=['d_dy_rho', 'd_dy_jy', 'd_dy_E'],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E,
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E,
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jy, E: - (jy / rho)],
    trial_deriv='y')

R31Sx = Term(
    name='R31Sx',
    description='energy convection height source x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']())])

R31Sy = Term(
    name='R31Sy',
    description='energy convection height source y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']())])

R32x = Term(
    name='R32x',
    description='pressure work x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['p', 'dp_drho', 'd_dx_rho', 'd_dx_jx'],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho),
    der_funs=[lambda ctx: lambda rho, jx: - (ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho)],
    trial_deriv='x')

R32y = Term(
    name='R32y',
    description='pressure work y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=['p', 'dp_drho', 'd_dy_rho', 'd_dy_jy'],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho),
    der_funs=[lambda ctx: lambda rho, jy: - (ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho)],
    trial_deriv='y')

R32Sx = Term(
    name='R32Sx',
    description='pressure work height source x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx', 'p', 'dp_drho'],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx: - ((ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)) * (1 / ctx['h']() * ctx['dh_dx']())),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dx']())])

R32Sy = Term(
    name='R32Sy',
    description='pressure work height source y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy', 'p', 'dp_drho'],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy: - ((ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)) * (1 / ctx['h']() * ctx['dh_dy']())),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dy']())])

R34 = Term(
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

R35x = Term(
    name='R35x',
    description='thermal diffusion x',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=['T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE'],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    test_deriv='x')

R35y = Term(
    name='R35y',
    description='thermal diffusion y',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=['T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE'],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    test_deriv='y')

R36 = Term(
    name='R36',
    description='wall heat balance',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=['S', 'dS_drho', 'dS_djx', 'dS_djy', 'dS_dE'],
    fun=lambda ctx: lambda rho, jx, jy, E: ctx['S'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: ctx['dS_drho'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djx'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djy'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_dE']()])

R3T = Term(
    name='R3T',
    description='energy time derivative',
    res='energy',
    dep_vars=['E'],
    dep_vals=['E_prev'],
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']()])

def get_active_terms(fem_solver: dict) -> List[Term]:
    """Return active Term instances based on fem_solver config.

    Physics flags (in fem_solver['physics']):
    - gap_shear:          Gap-averaged wall shear τ/h (R24x, R24y; R24x_fb, R24y_fb with cavitation)
    - plane_shear:        In-plane viscous diffusion (R23xy, R23yx)
    - inertia:            Momentum convection (R22*)
    - body_force:         Body force (R25x, R25y)
    - squeeze:            Height rate-of-change source in mass and momentum (R1Th, R2Thx, R2Thy)
    - energy:             Energy equation master switch, subflags below default to True
    - energy_convection:  Energy advection (R31*)
    - pressure_work:      Pressure-volume work (R32*)
    - thermal_diffusion:  Heat conduction (R35x, R35y)
    - wall_heat_balance:  Wall heat flux BC (R36)
    - wall_shear_work:    Wall stress work / shear heating (R34)
    """
    from .terms_theta import CAV_MASS_TERMS, THETA_TERMS_AD, R_cav, THETA_TERMS_WALL_STRESS, R1Th_cav
    from .terms_oss import OSS_TERMS, FC_TERMS

    physics = fem_solver['physics']
    stab = fem_solver['stabilization']
    cavitation = fem_solver['equations']['cavitation']

    if cavitation:
        terms = [*CAV_MASS_TERMS, R21x, R21y, R2Tx, R2Ty, R_cav]
    else:
        terms = [R11x, R11y, R11Sx, R11Sy, R11x_corr, R11y_corr, R1T, 
                 R21x, R21y, R2Tx, R2Ty]

    if cavitation and stab['ad']:
        terms += THETA_TERMS_AD

    if cavitation and stab['oss']:
        terms += OSS_TERMS

    if cavitation and stab['fc']:
        terms += FC_TERMS

    if physics['gap_shear']:
        terms += THETA_TERMS_WALL_STRESS if cavitation else [R24x, R24y]

    if physics['plane_shear']:
        terms += [R23xy, R23yx, R23xx, R23yy]

    if physics['inertia']:
        terms += [R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS]

    if physics['body_force']:
        terms += [R25x, R25y]

    if physics['squeeze']:
        terms += [R2Thx, R2Thy]
        terms += [R1Th_cav] if cavitation else [R1Th]

    if physics['energy']:
        terms.append(R3T)

        if physics['wall_shear_work']:
            terms.append(R34)

        if physics['energy_convection']:
            terms += [R31x, R31y, R31Sx, R31Sy]

        if physics['pressure_work']:
            terms += [R32x, R32y, R32Sx, R32Sy]

        if physics['thermal_diffusion']:
            terms += [R35x, R35y]

        if physics['wall_heat_balance']:
            terms.append(R36)

    return terms

def collect_required_fields(terms: List[Term]):
    """Collect all quad field keys required by a list of active terms."""
    plain = set()
    der = set()
    for term in terms:
        for key in term.dep_vals:
            if key.startswith('d_d'):
                der.add(key)
            else:
                plain.add(key)
    return plain, der
