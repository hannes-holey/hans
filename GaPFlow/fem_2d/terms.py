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

"""Height-Averaged Navier Stokes Terms

Each NonLinearTerm carries:
  name        : identifier
  description : human-readable label
  res         : residual equation name ('mass', 'momentum_x', 'momentum_y', 'energy')
  dep_vars    : field variables (DOF)
  dep_vals    : extra context fields such as height, which is not a DOF
  fun         : function expression
  der_funs    : partial derivatives w.r.t. field variables
  d_dx_resfun : bool  — True if fun contains ∂/∂x acting on a dep_var
  d_dy_resfun : bool  — True if fun contains ∂/∂y acting on a dep_var
  der_testfun : False | 'x' | 'y'  — direction of test-function derivative (IBP)

Derivative combinations (depvar_deriv, testfun_deriv):
  ('none', False)     — standard Galerkin: ∫ Nᵢ · f dΩ
  ('x', False)        — dep_var derivative only: fun receives ∂var/∂x
  ('y', False)        — dep_var derivative only: fun receives ∂var/∂y
  ('none', 'x')       — test-fun derivative only (PSPG): ∫ (∂Nᵢ/∂x) · f dΩ
  ('none', 'y')       — test-fun derivative only (PSPG): ∫ (∂Nᵢ/∂y) · f dΩ
  ('x', 'x')          — both (diffusion xx): ∫ (∂Nᵢ/∂x) · f(∂var/∂x) dΩ
  ('y', 'y')          — both (diffusion yy): ∫ (∂Nᵢ/∂y) · f(∂var/∂y) dΩ
  ('x', 'y')          — cross (diffusion xy): ∫ (∂Nᵢ/∂y) · f(∂var/∂x) dΩ
  ('y', 'x')          — cross (diffusion yx): ∫ (∂Nᵢ/∂x) · f(∂var/∂y) dΩ
"""

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
                 d_dx_resfun: bool = False,
                 d_dy_resfun: bool = False,
                 der_testfun=False):
        self.name = name
        self.description = description
        self.res = res
        self.dep_vars = dep_vars
        self.dep_vals = dep_vals
        self.fun_ = fun
        self.der_funs_ = der_funs
        self.d_dx_resfun = d_dx_resfun
        self.d_dy_resfun = d_dy_resfun
        self.der_testfun = der_testfun
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
    
    @property
    def depvar_deriv(self):
        """Direction of spatial derivative acting on dep_var: 'none', 'x', or 'y'."""
        if self.d_dx_resfun:
            return 'x'
        elif self.d_dy_resfun:
            return 'y'
        else:
            return 'none'

    @property
    def testfun_deriv(self):
        """Direction of test-function derivative: False, 'x', or 'y'."""
        return self.der_testfun

    @property
    def deriv_key(self):
        """Return (depvar_deriv, testfun_deriv) tuple for template lookup."""
        return (self.depvar_deriv, self.testfun_deriv)


# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

# R11: Flux divergence
R11x = NonLinearTerm(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: -jx,
    der_funs=[lambda ctx: lambda jx: np.full_like(jx, -1.0)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R11y = NonLinearTerm(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: -jy,
    der_funs=[lambda ctx: lambda jy: np.full_like(jy, -1.0)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R11Sx = NonLinearTerm(
    name='R11Sx',
    description='flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11Sy = NonLinearTerm(
    name='R11Sy',
    description='flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda jy: -1 / ctx['h']() * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -1 / ctx['h']() * ctx['dh_dy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R1T: Time derivative
R1T = NonLinearTerm(
    name='R1T',
    description='time derivative',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: - (rho - ctx['rho_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda rho: - np.full_like(rho, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

# R21: Pressure gradient
R21x = NonLinearTerm(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda *args: -ctx['p'](),
    der_funs=[lambda ctx: lambda *args: -ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R21y = NonLinearTerm(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda *args: -ctx['p'](),
    der_funs=[lambda ctx: lambda *args: -ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

# R22: Convective momentum flux
R22xx = NonLinearTerm(
    name='R22xx',
    description='convective momentum flux jx*jx in x',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx: -(jx * jx) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: (jx * jx) / (rho ** 2),
        lambda ctx: lambda rho, jx: -2 * jx / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R22xxS = NonLinearTerm(
    name='R22xxS',
    description='convective momentum flux jx*jx height source',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / (rho ** 2),
        lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dh_dx']() * 2 * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22yx = NonLinearTerm(
    name='R22yx',
    description='convective momentum flux jx*jy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy: -(jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -jy / rho,
        lambda ctx: lambda rho, jx, jy: -jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R22yxS = NonLinearTerm(
    name='R22yxS',
    description='convective momentum flux jx*jy height source (for momentum_x)',
    res='momentum_x',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jy / rho,
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22xy = NonLinearTerm(
    name='R22xy',
    description='convective momentum flux jx*jy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy: -(jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -jy / rho,
        lambda ctx: lambda rho, jx, jy: -jx / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R22xyS = NonLinearTerm(
    name='R22xyS',
    description='convective momentum flux jx*jy height source (for momentum_y)',
    res='momentum_y',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jy / rho,
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22yy = NonLinearTerm(
    name='R22yy',
    description='convective momentum flux jy*jy in y',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy: -(jy * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: (jy * jy) / (rho ** 2),
        lambda ctx: lambda rho, jy: -2 * jy / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R22yyS = NonLinearTerm(
    name='R22yyS',
    description='convective momentum flux jy*jy height source',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / (rho ** 2),
        lambda ctx: lambda rho, jy: -1 / ctx['h']() * ctx['dh_dy']() * 2 * jy / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R23: In-plane shear stress (viscous diffusion, integrated by parts)
# Weak form: ∫ (∂N_i/∂y) * η * ∂(jx/ρ)/∂y dΩ  for momentum_x
# f(rho, jx) = -η * jx/ρ, dep_var derivative ∂/∂y, test function derivative ∂/∂y
R23xy = NonLinearTerm(
    name='R23xy',
    description='shear viscous stress tau_xy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jx: -ctx['eta']() * jx / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: ctx['eta']() * jx / (rho ** 2),
        lambda ctx: lambda rho, jx: -ctx['eta']() / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# Weak form: ∫ (∂N_i/∂x) * η * ∂(jy/ρ)/∂x dΩ  for momentum_y
# f(rho, jy) = -η * jy/ρ, dep_var derivative ∂/∂x, test function derivative ∂/∂x
R23yx = NonLinearTerm(
    name='R23yx',
    description='shear viscous stress tau_xy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jy: -ctx['eta']() * jy / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: ctx['eta']() * jy / (rho ** 2),
        lambda ctx: lambda rho, jy: -ctx['eta']() / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R23xx = NonLinearTerm(
    name='R23xx',
    description='shear viscous stress tau_xx in x (for momentum_x)',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jx: - ctx['eta']() * jx / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: ctx['eta']() * jx / (rho ** 2),
        lambda ctx: lambda rho, jx: - ctx['eta']() / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R23yy = NonLinearTerm(
    name='R23yy',
    description='shear viscous stress tau_yy in y (for momentum_y)',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jy: - ctx['eta']() * jy / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: ctx['eta']() * jy / (rho ** 2),
        lambda ctx: lambda rho, jy: - ctx['eta']() / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# R24: Wall stress
R24x = NonLinearTerm(
    name='R24x',
    description='wall stress x',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R24y = NonLinearTerm(
    name='R24y',
    description='wall stress y',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R25: Body force
R25x = NonLinearTerm(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['rho'],
    dep_vals=['h', 'force_x'],
    fun=lambda ctx: lambda rho: ctx['h']() * rho * ctx['force_x'](),
    der_funs=[lambda ctx: lambda rho: ctx['h']() * ctx['force_x']() * np.ones_like(rho)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R25y = NonLinearTerm(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['rho'],
    dep_vals=['h', 'force_y'],
    fun=lambda ctx: lambda rho: ctx['h']() * rho * ctx['force_y'](),
    der_funs=[lambda ctx: lambda rho: ctx['h']() * ctx['force_y']() * np.ones_like(rho)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R2T: Time derivative
R2Tx = NonLinearTerm(
    name='R2Tx',
    description='time derivative momentum_x',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R2Ty = NonLinearTerm(
    name='R2Ty',
    description='time derivative momentum_y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# -----------------------------------------------------------------------------
# Energy equation terms (R3*)
# -----------------------------------------------------------------------------

# R31: Energy convection
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
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

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
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R31Sx = NonLinearTerm(
    name='R31Sx',
    description='energy convection height source x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R31Sy = NonLinearTerm(
    name='R31Sy',
    description='energy convection height source y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R32: Pressure work
R32x = NonLinearTerm(
    name='R32x',
    description='pressure work x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho),
    der_funs=[lambda ctx: lambda rho, jx: - (ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R32y = NonLinearTerm(
    name='R32y',
    description='pressure work y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho),
    der_funs=[lambda ctx: lambda rho, jy: - (ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R32Sx = NonLinearTerm(
    name='R32Sx',
    description='pressure work height source x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx: - ((ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)) * (1 / ctx['h']() * ctx['dh_dx']())),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R32Sy = NonLinearTerm(
    name='R32Sy',
    description='pressure work height source y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy: - ((ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)) * (1 / ctx['h']() * ctx['dh_dy']())),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R34: Wall stress work
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
              lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dtau_yz_bot_djy']() * ctx['V_bot']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R35: Thermal diffusion (integrated by parts)
# ∫ (∂N_i/∂x) * k*T dΩ  for energy
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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

# R36: Wall heat balance
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
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_dE']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R3T: Time derivative
R3T = NonLinearTerm(
    name='R3T',
    description='energy time derivative',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# Master list of all physical terms
term_list = [
    # Mass equation
    R11x, R11y, R11Sx, R11Sy, R1T,
    # Momentum equation
    R21x, R21y,
    R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx, R23xx, R23yy,
    R24x, R24y,
    R25x, R25y,
    R2Tx, R2Ty,
    # Energy equation
    R31x, R31y, R31Sx, R31Sy,
    R32x, R32y, R32Sx, R32Sy,
    R34, R35x, R35y, R36,
    R3T
]


def _term_names_from_physics(fem_solver: dict) -> List[str]:
    """Build term name list from physics flags.

    Physics flags (in fem_solver['physics']):
    - gap_shear:          Gap-averaged wall shear τ/h (R24x, R24y)
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

    physics = fem_solver.get('physics', {})

    # Mass conservation and pressure gradient always included
    terms = [
        'R11x', 'R11y', 'R11Sx', 'R11Sy', 'R1T',
        'R21x', 'R21y', 'R2Tx', 'R2Ty',
    ]

    if physics.get('gap_shear', True):
        terms.extend(['R24x', 'R24y'])

    if physics.get('plane_shear', True):
        terms.extend(['R23xy', 'R23yx', 'R23xx', 'R23yy'])

    if physics.get('inertia', False):
        terms.extend(['R22xx', 'R22yy', 'R22xy', 'R22yx',
                      'R22xxS', 'R22yyS', 'R22xyS', 'R22yxS'])

    if physics.get('body_force', False):
        terms.extend(['R25x', 'R25y'])

    if physics.get('energy', False):
        terms.append('R3T')

        if physics.get('wall_shear_work', True):
            terms.append('R34')

        if physics.get('energy_convection', True):
            terms.extend(['R31x', 'R31y', 'R31Sx', 'R31Sy'])

        if physics.get('pressure_work', True):
            terms.extend(['R32x', 'R32y', 'R32Sx', 'R32Sy'])

        if physics.get('thermal_diffusion', True):
            terms.extend(['R35x', 'R35y'])

        if physics.get('wall_heat_balance', True):
            terms.append('R36')

    return terms


def get_active_terms(fem_solver: dict) -> List['NonLinearTerm']:
    """Return active NonLinearTerm instances based on fem_solver config.

    If fem_solver['equations']['term_list'] is set, use that explicit list.
    Otherwise auto-select from physics flags via _term_names_from_physics().
    """
    user_terms = fem_solver['equations'].get('term_list')
    if user_terms is not None:
        requested = set(user_terms)
    else:
        requested = set(_term_names_from_physics(fem_solver))

    term_obj_list = [t for t in term_list if t.name in requested]

    return term_obj_list
