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
                 d_dx_resfun: 'bool | list[bool]' = False,
                 d_dy_resfun: 'bool | list[bool]' = False,
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
    
    def depvar_deriv_for(self, var: str) -> str:
        """Direction of spatial derivative acting on dep_var 'var': 'none', 'x', or 'y'."""
        i = self.dep_vars.index(var)
        dx = self.d_dx_resfun[i] if isinstance(self.d_dx_resfun, list) else self.d_dx_resfun
        dy = self.d_dy_resfun[i] if isinstance(self.d_dy_resfun, list) else self.d_dy_resfun
        if dx:
            return 'x'
        if dy:
            return 'y'
        return 'none'

    @property
    def depvar_deriv(self):
        """Direction of spatial derivative acting on dep_var: 'none', 'x', or 'y'.
        All dep_vars must share the same derivative direction (use depvar_deriv_for otherwise)."""
        if self.d_dx_resfun if not isinstance(self.d_dx_resfun, list) else any(self.d_dx_resfun):
            return 'x'
        elif self.d_dy_resfun if not isinstance(self.d_dy_resfun, list) else any(self.d_dy_resfun):
            return 'y'
        else:
            return 'none'

    @property
    def testfun_deriv(self):
        """Direction of test-function derivative: False, 'x', or 'y'."""
        return self.der_testfun

    @property
    def deriv_key(self):
        """Return (depvar_deriv, testfun_deriv) tuple for template lookup.
        Only valid when all dep_vars share the same derivative direction."""
        return (self.depvar_deriv, self.testfun_deriv)


# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

# R11: Flux divergence (pressure form: picks up (dp/drho) prefactor from
# the mass-equation × (dp/drho) reformulation — see pressure_formulation.md §3).
R11x = NonLinearTerm(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R11y = NonLinearTerm(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['dp_drho'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R11Sx = NonLinearTerm(
    name='R11Sx',
    description='flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11Sy = NonLinearTerm(
    name='R11Sy',
    description='flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy', 'dp_drho'],
    fun=lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['dp_drho']() * (1 / ctx['h']()) * ctx['dh_dy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R11x_corr / R11y_corr / R11Sx_corr / R11Sy_corr: Jacobian correction for the
# implicit p-dependence of dp/drho in R11x/R11y/R11Sx/R11Sy.
#
# R11x residual is  -∂/∂x(dp/drho(rho(p)) · jx).  Taking d/dp:
#   d/dp[dp/drho · jx] = d(dp/drho)/dp · jx = (d²p/drho² · drho/dp) · jx
# The correction Jacobian is therefore  -(d²p/drho² · drho/dp) · jx,
# with the same spatial derivative structure as the parent term (d_dx_resfun).
# Zero residual contribution — Jacobian-only.
R11x_corr = NonLinearTerm(
    name='R11x_corr',
    description='flux divergence x Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dx_jx'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dx_jx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11y_corr = NonLinearTerm(
    name='R11y_corr',
    description='flux divergence y Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'd_dy_jy'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * ctx['d_dy_jy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11Sx_corr = NonLinearTerm(
    name='R11Sx_corr',
    description='flux divergence height source x Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jx', 'h', 'dh_dx'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 / ctx['h']()) * ctx['dh_dx']() * ctx['jx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11Sy_corr = NonLinearTerm(
    name='R11Sy_corr',
    description='flux divergence height source y Jacobian correction (d(dp/drho)/dp)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jy', 'h', 'dh_dy'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 / ctx['h']()) * ctx['dh_dy']() * ctx['jy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R_cav_pen: Penalty enforcement of p >= p_cav (cavitation constraint).
# Adds ε·max(p_cav−p, 0) to the mass residual — zero in full-film, positive source
# in the cavitated zone. Jacobian is the raw Heaviside step (see notes in
# pressure_penalty_cavitation.md; Heaviside treatment is an open question).
# Requires ctx['p_cav'] (= prop['P0']) and ctx['pen_eps'] (= fem_solver['pen_eps']).
R_cav_pen = NonLinearTerm(
    name='R_cav_pen',
    description='cavitation penalty: enforces p >= p_cav in mass eq.',
    res='mass',
    dep_vars=['p'],
    dep_vals=['p_cav', 'pen_eps'],
    fun=lambda ctx: lambda p: ctx['pen_eps']() * np.maximum(ctx['p_cav']() - p, 0.0),
    der_funs=[lambda ctx: lambda p: -ctx['pen_eps']() * (p < ctx['p_cav']()).astype(float)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R1T: Time derivative (pressure form: -(drho/dp) * (p - p_prev)/dt).
# Chain rule: ∂ρ/∂t = (dρ/dp) · ∂p/∂t. drho_dp is evaluated at current p
# (quadrature field), so the Jacobian is simply -drho_dp/dt.
# The second-order correction -d²ρ/dp²·(p-p_prev)/dt is negligible for small
# dt and smooth EOS (DH), so it is omitted.
R1T = NonLinearTerm(
    name='R1T',
    description='time derivative',
    res='mass',
    dep_vars=['p'],
    dep_vals=['drho_dp'],
    fun=lambda ctx: lambda p: - (p - ctx['p_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: - np.ones_like(p) / ctx['dt']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R_Lpx / R_Lpy: theta-weighted pressure Laplacian stabilization for mass equation.
# Weak form: +alpha * theta * ∫ (∂Nᵢ/∂x)(∂p/∂x) dΩ  (and y-direction).
# Assembly applies -1/dx² for (d_dx_resfun=True, der_testfun='x'), so fun carries
# a minus sign: fun = -alpha*theta*p → net +alpha*theta*∫ ∂Nᵢ/∂x · ∂p/∂x dΩ.
# theta is a dep_val: Jacobian is approximate (d/dtheta omitted) which is
# acceptable for a stabilization term.
# Activated by physics flag lap_pressure: true.
# Coefficient set via fem_solver.lap_pressure_alpha (default 0.0, dimensionless).
R_Lpx = NonLinearTerm(
    name='R_Lpx',
    description='theta-weighted pressure Laplacian stabilization x',
    res='mass',
    dep_vars=['p'],
    dep_vals=['theta'],
    fun=lambda ctx: lambda p: -ctx['lap_p_alpha']() * ctx['theta']() * p,
    der_funs=[lambda ctx: lambda p: -ctx['lap_p_alpha']() * ctx['theta']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R_Lpy = NonLinearTerm(
    name='R_Lpy',
    description='theta-weighted pressure Laplacian stabilization y',
    res='mass',
    dep_vars=['p'],
    dep_vals=['theta'],
    fun=lambda ctx: lambda p: -ctx['lap_p_alpha']() * ctx['theta']() * p,
    der_funs=[lambda ctx: lambda p: -ctx['lap_p_alpha']() * ctx['theta']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# R1Lx / R1Ly: Simple Laplacian density diffusion for mass equation.
# Adds α_stab * ∫ (∂Nᵢ/∂x_k)(∂ρ/∂x_k) dΩ with a fixed, tunable coefficient.
# Activated by physics flag mass_diffusion: true.
# Coefficient set via fem_solver.mass_diffusion_alpha (default 1e-3).
R1Lx = NonLinearTerm(
    name='R1Lx',
    description='Laplacian density diffusion x',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: ctx['mass_diff_alpha']() * rho,
    der_funs=[lambda ctx: lambda rho: np.full_like(rho, ctx['mass_diff_alpha']())],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R1Ly = NonLinearTerm(
    name='R1Ly',
    description='Laplacian density diffusion y',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: ctx['mass_diff_alpha']() * rho,
    der_funs=[lambda ctx: lambda rho: np.full_like(rho, ctx['mass_diff_alpha']())],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# -----------------------------------------------------------------------------
# PSPG stabilization terms (mass equation, test-function derivatives)
# Activated by physics flag pspg: true.
# Uses tau_pspg (Tezduyar 1992, compressible) for stabilization parameter.
# -----------------------------------------------------------------------------

# PSPG pressure gradient: tau * ∫ (∂Nᵢ/∂x_k) · (-∂p/∂x_k) dΩ
# Strong-form momentum residual has -∂p/∂x_k.  Both test function and
# dep_var carry a spatial derivative → Laplacian stencil (d_d{x,y}_resfun + der_testfun).
R1PSPG_Px = NonLinearTerm(
    name='R1PSPG_Px',
    description='PSPG pressure gradient x',
    res='mass',
    dep_vars=['rho'],
    dep_vals=['tau_pspg', 'p', 'dp_drho'],
    fun=lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R1PSPG_Py = NonLinearTerm(
    name='R1PSPG_Py',
    description='PSPG pressure gradient y',
    res='mass',
    dep_vars=['rho'],
    dep_vals=['tau_pspg', 'p', 'dp_drho'],
    fun=lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# PSPG pressure gradient Jacobian correction:
# d/drho(-tau * dp/drho * drho/dx) has a second term  -tau * d2p/drho2 * drho/dx * Nj
# that the main R1PSPG_P{x,y} terms miss (they only give -tau * dp/drho * dNj/dx).
# Zero residual contribution — Jacobian-only.
R1PSPG_Px2 = NonLinearTerm(
    name='R1PSPG_Px2',
    description='PSPG pressure gradient x Jacobian correction',
    res='mass',
    dep_vars=['rho'],
    dep_vals=['tau_pspg', 'd2p_drho2', 'd_dx_rho'],
    fun=lambda ctx: lambda rho: 0.0 * rho,
    der_funs=[lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['d2p_drho2']() * ctx['d_dx_rho']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R1PSPG_Py2 = NonLinearTerm(
    name='R1PSPG_Py2',
    description='PSPG pressure gradient y Jacobian correction',
    res='mass',
    dep_vars=['rho'],
    dep_vals=['tau_pspg', 'd2p_drho2', 'd_dy_rho'],
    fun=lambda ctx: lambda rho: 0.0 * rho,
    der_funs=[lambda ctx: lambda rho: -ctx['tau_pspg']() * ctx['d2p_drho2']() * ctx['d_dy_rho']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

# PSPG temporal: tau * ∫ (∂Nᵢ/∂x_k) · (-(j - j_prev)/dt) dΩ
R1PSPG_Tx = NonLinearTerm(
    name='R1PSPG_Tx',
    description='PSPG temporal x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['tau_pspg', 'jx_prev'],
    fun=lambda ctx: lambda jx: -(ctx['tau_pspg']() / ctx['dt']()) * (jx - ctx['jx_prev']()),
    der_funs=[lambda ctx: lambda jx: -ctx['tau_pspg']() / ctx['dt']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R1PSPG_Ty = NonLinearTerm(
    name='R1PSPG_Ty',
    description='PSPG temporal y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['tau_pspg', 'jy_prev'],
    fun=lambda ctx: lambda jy: -(ctx['tau_pspg']() / ctx['dt']()) * (jy - ctx['jy_prev']()),
    der_funs=[lambda ctx: lambda jy: -ctx['tau_pspg']() / ctx['dt']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

# PSPG wall shear: tau * ∫ (∂Nᵢ/∂x_k) · (tau_xz/h) dΩ
R1PSPG_Wx = NonLinearTerm(
    name='R1PSPG_Wx',
    description='PSPG wall shear x',
    res='mass',
    dep_vars=['rho', 'jx'],
    dep_vals=['tau_pspg', 'tau_xz', 'h', 'dtau_xz_drho', 'dtau_xz_djx'],
    fun=lambda ctx: lambda rho, jx: ctx['tau_pspg']() * ctx['tau_xz']() / ctx['h'](),
    der_funs=[
        lambda ctx: lambda rho, jx: ctx['tau_pspg']() * ctx['dtau_xz_drho']() / ctx['h'](),
        lambda ctx: lambda rho, jx: ctx['tau_pspg']() * ctx['dtau_xz_djx']() / ctx['h'](),
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R1PSPG_Wy = NonLinearTerm(
    name='R1PSPG_Wy',
    description='PSPG wall shear y',
    res='mass',
    dep_vars=['rho', 'jy'],
    dep_vals=['tau_pspg', 'tau_yz', 'h', 'dtau_yz_drho', 'dtau_yz_djy'],
    fun=lambda ctx: lambda rho, jy: ctx['tau_pspg']() * ctx['tau_yz']() / ctx['h'](),
    der_funs=[
        lambda ctx: lambda rho, jy: ctx['tau_pspg']() * ctx['dtau_yz_drho']() / ctx['h'](),
        lambda ctx: lambda rho, jy: ctx['tau_pspg']() * ctx['dtau_yz_djy']() / ctx['h'](),
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

# R21: Pressure gradient, IBP form: ∫ (∂Nᵢ^P2/∂x) · p dΩ
# Derivative on the P2 test function (IBP of ∫ Nᵢ · ∂p/∂x dΩ, boundary term dropped).
# fun = -p so that assembly factor -1/dx from der_testfun='x' gives net +∫ (∂Nᵢ/∂x)·p dΩ,
# i.e. the momentum residual contribution is -∂p/∂x.
# p is the DOF, so der is the trivial -1 — no chain rule, no _corr term.
R21x = NonLinearTerm(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

R21y = NonLinearTerm(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

# R21x_corr / R21y_corr deleted in the pressure formulation: they existed to
# add the d²p/dρ² chain-rule contribution that is absent once p is the DOF.

# R22: Convective momentum flux (IBP form)
# Full height-averaged term: (1/h)·∂(h·jx²/ρ)/∂x = ∂(jx²/ρ)/∂x + (dh_dx/h)·(jx²/ρ)
# R22xx handles ∂(jx²/ρ)/∂x via IBP: ∫ Nᵢ·∂(−jx²/ρ)/∂x dΩ → −∫ ∂Nᵢ/∂x·(−jx²/ρ) dΩ
# Assembly with der_testfun='x' computes −∫ ∂Nᵢ/∂x · fun dΩ, so fun = −jx²/ρ.
# R22xxS handles the remaining height-source (dh_dx/h)·(jx²/ρ) separately.
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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='x')

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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

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
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun='y')

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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R23: In-plane shear stress (viscous diffusion, integrated by parts)
# Weak form: -∫ (∂N_i/∂y) * η/ρ * ∂jx/∂y dΩ  for momentum_x
# ρ is frozen (dep_val); p enters only via the Jacobian chain rule d/dp = d/dρ * dρ/dp.
# dep_var derivative: only jx gets ∂/∂y (list form); p enters undifferentiated.
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
    d_dx_resfun=[False, False],
    d_dy_resfun=[False, True],
    der_testfun='y')

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
    d_dx_resfun=[False, True],
    d_dy_resfun=[False, False],
    der_testfun='x')

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
    d_dx_resfun=[False, True],
    d_dy_resfun=[False, False],
    der_testfun='x')

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
    d_dx_resfun=[False, False],
    d_dy_resfun=[False, True],
    der_testfun='y')

# R24: Wall stress (chain rule on p-slot: ∂f/∂p = ∂f/∂ρ · dρ/dp)
R24x = NonLinearTerm(
    name='R24x',
    description='wall stress x',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R24y = NonLinearTerm(
    name='R24y',
    description='wall stress y',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R24x_fb / R24y_fb: Wall stress with theta-dependent effective density.
# Activating cavitation changes rho_eff = (1-theta)*rho inside get_tau, so the
# velocity seen by the wall stress model u = jx/rho_eff increases in cavitated cells.
# tau_xz already encodes the effect; the new Jacobian block d/dtheta comes from
# the chain rule through rho_eff.  The d/dp and d/jx blocks are unchanged in form
# but now evaluated at rho_eff (handled inside the JAX-traced tau functions).
# Activated when gap_shear: true AND cavitation: true (replaces R24x/R24y).
R24x_fb = NonLinearTerm(
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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R24y_fb = NonLinearTerm(
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
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R25: Body force
R25x = NonLinearTerm(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_x'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_x'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_x']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R25y = NonLinearTerm(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_y'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_y'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_y']()],
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
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R2Ty = NonLinearTerm(
    name='R2Ty',
    description='time derivative momentum_y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']()],
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
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# -----------------------------------------------------------------------------
# Elrod-Adams / Fischer-Burmeister cavitation terms (R11*_fb, R_FB)
# Activated when physics flag cavitation: true.
# Replace R11x, R11y, R11Sx, R11Sy (and their _corr variants) entirely.
# -----------------------------------------------------------------------------

def _fb_denom(a, b):
    """sqrt(a²+b²), bumped to machine epsilon at (0,0) to avoid singularity."""
    d = np.sqrt(a**2 + b**2)
    return np.where(d == 0.0, np.finfo(float).eps, d)


# IBP form: ∫ φ·∂(-c·jx)/∂x dΩ → +∫ ∂φ/∂x·c·jx dΩ by IBP.
# Assembly applies factor -1/d for der_testfun='x', so fun must carry a minus
# sign to cancel it: fun = -c·jx → assembled as +∫ ∂φ/∂x·c·jx dΩ.
# Jacobian blocks produced:
#   (mass, jx,    'none', 'x'): +∫ ∂Ni/∂x · dp_drho·(1−θ) · Nj^P2 dΩ
#   (mass, theta, 'none', 'x'): +∫ ∂Ni/∂x · dp_drho·jx · Nj^P1 dΩ
R11x_fb = NonLinearTerm(
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
    d_dx_resfun=False, d_dy_resfun=False, der_testfun='x')

R11y_fb = NonLinearTerm(
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
    d_dx_resfun=False, d_dy_resfun=False, der_testfun='y')

R11Sx_fb = NonLinearTerm(
    name='R11Sx_fb',
    description='flux divergence height source x Elrod-Adams',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['h', 'dh_dx', 'dp_drho'],
    fun=lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta:  ctx['dp_drho']() / ctx['h']() * ctx['dh_dx']() * jx,
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)

R11Sy_fb = NonLinearTerm(
    name='R11Sy_fb',
    description='flux divergence height source y Elrod-Adams',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['h', 'dh_dy', 'dp_drho'],
    fun=lambda ctx: lambda jy, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta: -ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * (1 - theta),
        lambda ctx: lambda jy, theta:  ctx['dp_drho']() / ctx['h']() * ctx['dh_dy']() * jy,
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)

# Jacobian correction for the implicit p-dependence of dp_drho in R11*_fb.
# R11x_fb uses IBP (der_testfun='x'): Rᵢ = ∫ ∂Nᵢ/∂x · (-dp_drho·(1-θ)·jx) dΩ
# so dRᵢ/dpⱼ = ∫ ∂Nᵢ/∂x · (-d²p/drho²·drho/dp·(1-θ)·jx·Nⱼ) dΩ
# → same der_testfun='x', dep_vals uses jx (not d_dx_jx).
# Zero residual contribution — Jacobian-only.
R11x_fb_corr = NonLinearTerm(
    name='R11x_fb_corr',
    description='flux divergence x FB Jacobian correction (d(dp_drho)/dp, theta-weighted)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jx', 'theta'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) * ctx['jx']()],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun='x')

R11y_fb_corr = NonLinearTerm(
    name='R11y_fb_corr',
    description='flux divergence y FB Jacobian correction (d(dp_drho)/dp, theta-weighted)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jy', 'theta'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) * ctx['jy']()],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun='y')

R11Sx_fb_corr = NonLinearTerm(
    name='R11Sx_fb_corr',
    description='flux divergence height source x FB Jacobian correction',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jx', 'h', 'dh_dx', 'theta'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) / ctx['h']() * ctx['dh_dx']() * ctx['jx']()],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)

R11Sy_fb_corr = NonLinearTerm(
    name='R11Sy_fb_corr',
    description='flux divergence height source y FB Jacobian correction',
    res='mass',
    dep_vars=['p'],
    dep_vals=['d2p_drho2', 'drho_dp', 'jy', 'h', 'dh_dy', 'theta'],
    fun=lambda ctx: lambda p: 0.0 * p,
    der_funs=[lambda ctx: lambda p: -ctx['d2p_drho2']() * ctx['drho_dp']() * (1 - ctx['theta']()) / ctx['h']() * ctx['dh_dy']() * ctx['jy']()],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)


# R1STx / R1STy: theta diffusion stabilization in the mass equation.
# Weak form: +tau_st * ∫ (∂Nᵢ/∂x)(∂theta/∂x) dΩ  (and y-direction),
# where tau_st = theta_stab_alpha * dp_drho * |j|.
# This adds cross-node coupling in theta via the mass equation, damping
# oscillations near the cavitation front. The coefficient dp_drho * |j|
# gives units consistent with the mass equation (Pa/m after IBP), and
# |j| = sqrt(jx²+jy²) makes the stabilization flow-aware.
# Jacobian is approximate: d/dtheta only, d/d|j| and d/dp dropped (frozen coefficients).
# Activated by physics flag theta_stab: true.
# Coefficient set via fem_solver.theta_stab_alpha (dimensionless, default 0.0).
R1STx = NonLinearTerm(
    name='R1STx',
    description='theta diffusion stabilization in mass equation x',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jx', 'jy', 'theta_stab_alpha'],
    fun=lambda ctx: lambda theta: -(ctx['theta_stab_alpha']() * ctx['dp_drho']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * theta),
    der_funs=[lambda ctx: lambda theta: -(ctx['theta_stab_alpha']() * ctx['dp_drho']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R1STy = NonLinearTerm(
    name='R1STy',
    description='theta diffusion stabilization in mass equation y',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jx', 'jy', 'theta_stab_alpha'],
    fun=lambda ctx: lambda theta: -(ctx['theta_stab_alpha']() * ctx['dp_drho']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * theta),
    der_funs=[lambda ctx: lambda theta: -(ctx['theta_stab_alpha']() * ctx['dp_drho']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# R1UWx / R1UWy: donor-cell upwind stabilization for theta in the mass equation.
# Derived from flux splitting of (1−θ)·j in the Elrod-Adams term:
#   θ_upwind = θ_Galerkin − sign(jx)·dx/2·∂θ/∂x
# The upwind correction contributes to the mass residual (before IBP):
#   +dp_drho·|jx|·dx/2 · ∂θ/∂x
# In weak form with ('x','x') (test and trial both differentiated, assembly
# applies 1/dx² scaling), the net integrand coefficient is dp_drho·|jx|·dx/2,
# which gives O(dx) relative to the main flux term — vanishes under refinement.
# Jacobian is approximate: d/dtheta only, d/d|j| and d/dp dropped (frozen coefficients).
# Activated by physics flag upwind_theta: true.
R1UWx = NonLinearTerm(
    name='R1UWx',
    description='donor-cell upwind stabilization for theta in mass equation x',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jx', 'dx'],
    fun=lambda ctx: lambda theta: -ctx['dp_drho']() * np.abs(ctx['jx']()) * ctx['dx']() / 2.0 * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['dp_drho']() * np.abs(ctx['jx']()) * ctx['dx']() / 2.0],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R1UWy = NonLinearTerm(
    name='R1UWy',
    description='donor-cell upwind stabilization for theta in mass equation y',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['dp_drho', 'jy', 'dy'],
    fun=lambda ctx: lambda theta: -ctx['dp_drho']() * np.abs(ctx['jy']()) * ctx['dy']() / 2.0 * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['dp_drho']() * np.abs(ctx['jy']()) * ctx['dy']() / 2.0],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

# R1FBpx / R1FBpy / R1FBtx / R1FBty: PSPG-FB stabilization for mass equation.
#
# Adds −τ·∇φ projected onto ∇Nᵢ, where φ is the FB residual. Since φ=0 at
# convergence, ∇φ=0 too — the term vanishes at the solution (residual-consistent).
#
# With φ = √(a_nd²+θ²) − a_nd − θ, a_nd = (p−p_cav)/P0:
#   ∂φ/∂x = cp·∂p/∂x + cθ·∂θ/∂x
#   cp = (a_nd/D − 1)/P0 ≤ 0,   cθ = θ/D − 1 ≤ 0,   D = √(a_nd²+θ²)
#
# Stabilization term (minus sign gives positive-definite diffusion since cp,cθ ≤ 0):
#   −τ · ∫ (∂Nᵢ/∂x)(cp·∂p/∂x + cθ·∂θ/∂x) dΩ  (x) + same for y
#
# τ = pspg_fb_alpha · P0 · dx²  (dimensionless knob × pressure scale × mesh area)
#
# Jacobian: cp, cθ, τ frozen as coefficients (not differentiated through).
# This is the standard PSPG approximation — consistent with how R_Lpx/R_Lpy work.
# Activated by physics flag pspg_fb: true.
# Coefficient set via fem_solver.pspg_fb_alpha (dimensionless, default 0.0).

def _fb_cp_ct(ctx):
    """Compute frozen FB gradient coefficients cp and cθ at quad points."""
    a_nd = (ctx['p']() - ctx['p_cav']()) / ctx['fb_p_ref']()
    theta = ctx['theta']()
    D = _fb_denom(a_nd, theta)
    cp = (a_nd / D - 1.0) / ctx['fb_p_ref']()
    ct = theta / D - 1.0
    tau = ctx['pspg_fb_alpha']() * ctx['fb_p_ref']() * ctx['dx']()**2
    return cp, ct, tau

R1FBpx = NonLinearTerm(
    name='R1FBpx',
    description='PSPG-FB stabilization mass eq x, dep_var=p',
    res='mass',
    dep_vars=['p'],
    dep_vals=['p_cav', 'fb_p_ref', 'p', 'theta', 'dx', 'pspg_fb_alpha'],
    fun=lambda ctx: lambda p: (
        lambda cp, ct, tau: tau * cp * p
    )(*_fb_cp_ct(ctx)),
    der_funs=[lambda ctx: lambda p: (
        lambda cp, ct, tau: tau * cp
    )(*_fb_cp_ct(ctx))],
    d_dx_resfun=True, d_dy_resfun=False, der_testfun='x')

R1FBpy = NonLinearTerm(
    name='R1FBpy',
    description='PSPG-FB stabilization mass eq y, dep_var=p',
    res='mass',
    dep_vars=['p'],
    dep_vals=['p_cav', 'fb_p_ref', 'p', 'theta', 'dx', 'pspg_fb_alpha'],
    fun=lambda ctx: lambda p: (
        lambda cp, ct, tau: tau * cp * p
    )(*_fb_cp_ct(ctx)),
    der_funs=[lambda ctx: lambda p: (
        lambda cp, ct, tau: tau * cp
    )(*_fb_cp_ct(ctx))],
    d_dx_resfun=False, d_dy_resfun=True, der_testfun='y')

R1FBtx = NonLinearTerm(
    name='R1FBtx',
    description='PSPG-FB stabilization mass eq x, dep_var=theta',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['p_cav', 'fb_p_ref', 'p', 'theta', 'dx', 'pspg_fb_alpha'],
    fun=lambda ctx: lambda theta: (
        lambda cp, ct, tau: tau * ct * theta
    )(*_fb_cp_ct(ctx)),
    der_funs=[lambda ctx: lambda theta: (
        lambda cp, ct, tau: tau * ct
    )(*_fb_cp_ct(ctx))],
    d_dx_resfun=True, d_dy_resfun=False, der_testfun='x')

R1FBty = NonLinearTerm(
    name='R1FBty',
    description='PSPG-FB stabilization mass eq y, dep_var=theta',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['p_cav', 'fb_p_ref', 'p', 'theta', 'dx', 'pspg_fb_alpha'],
    fun=lambda ctx: lambda theta: (
        lambda cp, ct, tau: tau * ct * theta
    )(*_fb_cp_ct(ctx)),
    der_funs=[lambda ctx: lambda theta: (
        lambda cp, ct, tau: tau * ct
    )(*_fb_cp_ct(ctx))],
    d_dx_resfun=False, d_dy_resfun=True, der_testfun='y')

# R_LTx / R_LTy: Laplacian theta stabilization for fb equation.
# Weak form: +alpha * ∫ (∂Nᵢ/∂x)(∂theta/∂x) dΩ  (and y-direction).
# Assembly applies -1/dx² for (d_dx_resfun=True, der_testfun='x'), so fun
# carries a minus sign: fun = -alpha*theta → net +alpha*∫ ∂Nᵢ/∂x·∂theta/∂x dΩ.
# Coefficient lap_theta_alpha has units consistent with R_FB (Pa) — tune directly.
# Activated by physics flag lap_theta: true.
R_LTx = NonLinearTerm(
    name='R_LTx',
    description='Laplacian theta stabilization x',
    res='fb',
    dep_vars=['theta'],
    dep_vals=[],
    fun=lambda ctx: lambda theta: -ctx['lap_theta_alpha']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['lap_theta_alpha']() * np.ones_like(theta)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun='x')

R_LTy = NonLinearTerm(
    name='R_LTy',
    description='Laplacian theta stabilization y',
    res='fb',
    dep_vars=['theta'],
    dep_vals=[],
    fun=lambda ctx: lambda theta: -ctx['lap_theta_alpha']() * theta,
    der_funs=[lambda ctx: lambda theta: -ctx['lap_theta_alpha']() * np.ones_like(theta)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun='y')

R_FB = NonLinearTerm(
    name='R_FB',
    description='Fischer-Burmeister complementarity condition (p normalized by P0)',
    res='fb',
    dep_vars=['p', 'theta'],
    dep_vals=['p_cav', 'fb_p_ref'],
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
    ],
    d_dx_resfun=False, d_dy_resfun=False, der_testfun=False)


# Master list of all physical terms.
term_list = [
    # Mass equation
    R11x, R11y, R11Sx, R11Sy,
    R11x_corr, R11y_corr, #R11Sx_corr, R11Sy_corr,
    R_cav_pen,
    R1T,
    # Elrod-Adams / FB cavitation (replaces R11* when cavitation: true)
    R11x_fb, R11y_fb, R11Sx_fb, R11Sy_fb,
    R11x_fb_corr, R11y_fb_corr, R11Sx_fb_corr, R11Sy_fb_corr,
    R_FB,
    # theta-weighted pressure Laplacian stabilization (dormant; physics.lap_pressure default False)
    R_Lpx, R_Lpy,
    # Theta diffusion stabilization in mass equation (dormant; physics.theta_stab default False)
    R1STx, R1STy,
    # Donor-cell upwind stabilization for theta (dormant; physics.upwind_theta default False)
    R1UWx, R1UWy,
    # PSPG-FB residual-consistent stabilization (dormant; physics.pspg_fb default False)
    R1FBpx, R1FBpy, R1FBtx, R1FBty,
    # Laplacian theta stabilization (dormant; physics.lap_theta default False)
    R_LTx, R_LTy,
    # Laplacian density diffusion (dormant; physics.mass_diffusion default False)
    R1Lx, R1Ly,
    # PSPG mass stabilization (dormant; physics.pspg default False)
    R1PSPG_Px, R1PSPG_Py,
    R1PSPG_Tx, R1PSPG_Ty, R1PSPG_Wx, R1PSPG_Wy,
    # Momentum equation
    R21x, R21y,
    R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx, R23xx, R23yy,
    R24x, R24y,
    R24x_fb, R24y_fb,
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
    - mass_diffusion:     Laplacian density diffusion (R1Lx, R1Ly)
    - pspg:               PSPG stabilization for mass eq (R1PSPG_*)
    - gap_shear:          Gap-averaged wall shear τ/h (R24x, R24y; R24x_fb, R24y_fb with cavitation)
    - plane_shear:        In-plane viscous diffusion (R23xy, R23yx)
    - inertia:            Momentum convection (R22*)
    - body_force:         Body force (R25x, R25y)
    - upwind_theta:       Donor-cell upwind stabilization for theta (R1UWx, R1UWy)
    - pspg_fb:            PSPG-FB residual-consistent stabilization (R1FBpx/py/tx/ty)
    - energy:             Energy equation master switch, subflags below default to True
    - energy_convection:  Energy advection (R31*)
    - pressure_work:      Pressure-volume work (R32*)
    - thermal_diffusion:  Heat conduction (R35x, R35y)
    - wall_heat_balance:  Wall heat flux BC (R36)
    - wall_shear_work:    Wall stress work / shear heating (R34)
    """

    physics = fem_solver.get('physics', {})
    equations = fem_solver.get('equations', {})
    cavitation = equations.get('cavitation', False)

    # Mass conservation and pressure gradient always included.
    # Cavitation: swap standard R11* (+ corr) for Elrod-Adams _fb variants.
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

    if physics.get('lap_pressure', False):
        terms.extend(['R_Lpx', 'R_Lpy'])

    if cavitation and physics.get('theta_stab', False):
        terms.extend(['R1STx', 'R1STy'])

    if cavitation and physics.get('upwind_theta', False):
        terms.extend(['R1UWx', 'R1UWy'])

    if cavitation and physics.get('pspg_fb', False):
        terms.extend(['R1FBpx', 'R1FBpy', 'R1FBtx', 'R1FBty'])

    if cavitation and physics.get('lap_theta', False):
        terms.extend(['R_LTx', 'R_LTy'])

    if cavitation and physics.get('supg_theta', False):
        terms.extend([
            'R1SUPGxx', 'R1SUPGyy', 'R1SUPGxy', 'R1SUPGyx',
            'R1SUPGhxx', 'R1SUPGhxy', 'R1SUPGhyx', 'R1SUPGhyy',
            # 'R1SUPGdivxx', 'R1SUPGdivyy', 'R1SUPGdivyx', 'R1SUPGdivxy',
            'R1SUPGTx', 'R1SUPGTy',
        ])

    if cavitation and physics.get('oss_theta', False):
        from .terms_oss import OSS_TERM_NAMES
        terms.extend(OSS_TERM_NAMES)

    if physics.get('mass_diffusion', False):
        terms.extend(['R1Lx', 'R1Ly'])

    if physics.get('pspg', False):
        terms.extend(['R1PSPG_Px', 'R1PSPG_Py',
                      'R1PSPG_Tx', 'R1PSPG_Ty',
                      'R1PSPG_Wx', 'R1PSPG_Wy'])

    if physics.get('gap_shear', True):
        if cavitation:
            terms.extend(['R24x_fb', 'R24y_fb'])
        else:
            terms.extend(['R24x', 'R24y'])

    # plane_shear default flipped to False for the pressure-based first
    # iteration. R23* terms themselves remain in density form in the
    # source (not yet migrated); enabling this flag would require their
    # §4-rewrite first.
    if physics.get('plane_shear', False):
        terms.extend(['R23xy', 'R23yx', 'R23xx', 'R23yy'])

    if physics.get('inertia', False):
        terms.extend(['R22xx', 'R22xxS', 'R22yx', 'R22yxS', 'R22xy', 'R22xyS', 'R22yy', 'R22yyS'])

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

    # SUPG terms (lazy import to avoid circular dependency)
    from .terms_supg import get_supg_terms, SUPG_TERM_NAMES
    if requested & set(SUPG_TERM_NAMES):
        supg_terms = {t.name: t for t in get_supg_terms()}
        term_obj_list += [supg_terms[n] for n in SUPG_TERM_NAMES if n in requested]

    # OSS terms (lazy import to avoid circular dependency)
    from .terms_oss import get_oss_terms, OSS_TERM_NAMES
    if requested & set(OSS_TERM_NAMES):
        oss_terms = {t.name: t for t in get_oss_terms()}
        term_obj_list += [oss_terms[n] for n in OSS_TERM_NAMES if n in requested]

    return term_obj_list
