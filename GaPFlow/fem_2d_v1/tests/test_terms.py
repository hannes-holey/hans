"""Phase 6 tests: NonLinearTerm.evaluate_deriv vs finite-difference of evaluate.

Strategy
--------
Each term's fun/der_funs are pure functions of (dep_vars...) given a ctx.
We build a minimal ctx with constant quad-point arrays, evaluate at a base
point, perturb each dep_var by eps, and check that

    der_fun_i(*args) ≈ (fun(*args_perturbed) - fun(*args)) / eps

No grid, no muGrid, no assembly — pure function testing.

Terms tested
------------
Mass equation    : R11x, R11y, R11Sx, R11Sy, R1T
Momentum x       : R21x, R22xx, R22xxS, R22yx, R22yxS, R23xy, R24x, R25x, R2Tx
Momentum y       : R21y, R22xy, R22xyS, R22yy, R22yyS, R23yx, R24y, R25y, R2Ty
Energy           : R31x, R31y, R31Sx, R31Sy, R32x, R32y, R32Sx, R32Sy,
                   R34, R35x, R35y, R36, R3T
"""
import pytest
import numpy as np

from GaPFlow.fem_2d.terms import (
    R11x, R11y, R11Sx, R11Sy, R1T,
    R21x, R21y,
    R22xx, R22xxS, R22yx, R22yxS,
    R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx,
    R24x, R24y,
    R25x, R25y,
    R2Tx, R2Ty,
    R31x, R31y, R31Sx, R31Sy,
    R32x, R32y, R32Sx, R32Sy,
    R34,
    R35x, R35y,
    R36,
    R3T,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS = 1e-6
ATOL = 1e-5    # finite-difference tolerance (loose — forward-diff O(eps))
N = 12          # quad-point array length (arbitrary)

RNG = np.random.default_rng(42)


def _const(val):
    """Return a lambda () -> constant array of shape (N,)."""
    arr = np.full(N, val)
    return lambda: arr


def _rand(lo, hi):
    """Return a lambda () -> random array in [lo, hi] of shape (N,)."""
    arr = RNG.uniform(lo, hi, N)
    return lambda: arr.copy()


def _build_and_check(term, ctx, base_vals, var_idx, *, eps=EPS, atol=ATOL):
    """Finite-difference check of der_funs[var_idx] against fun.

    base_vals : list of arrays, one per dep_var
    var_idx   : which dep_var to perturb
    """
    term.build(ctx)

    f0 = term.evaluate(*base_vals)

    perturbed = list(base_vals)
    perturbed[var_idx] = base_vals[var_idx] + eps
    f1 = term.evaluate(*perturbed)

    fd = (f1 - f0) / eps
    analytical = term.evaluate_deriv(term.dep_vars[var_idx], *base_vals)

    assert np.allclose(analytical, fd, atol=atol, rtol=1e-4), (
        f"Term {term.name!r}, d/d({term.dep_vars[var_idx]}): "
        f"max |analytic-fd| = {np.max(np.abs(analytical - fd)):.3e}"
    )


# ---------------------------------------------------------------------------
# Mass equation
# ---------------------------------------------------------------------------

class TestR11x:
    def setup_method(self):
        self.ctx = {}
        self.term = R11x

    def test_deriv_jx(self):
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jx], 0)


class TestR11y:
    def setup_method(self):
        self.ctx = {}
        self.term = R11y

    def test_deriv_jy(self):
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jy], 0)


class TestR11Sx:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dx': _rand(-0.3, 0.3)}
        self.term = R11Sx

    def test_deriv_jx(self):
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jx], 0)


class TestR11Sy:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dy': _rand(-0.3, 0.3)}
        self.term = R11Sy

    def test_deriv_jy(self):
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jy], 0)


class TestR1T:
    def setup_method(self):
        self.ctx = {'rho_prev': _rand(0.8, 1.5), 'dt': 0.01}
        self.term = R1T

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        _build_and_check(self.term, self.ctx, [rho], 0)


# ---------------------------------------------------------------------------
# Momentum x
# ---------------------------------------------------------------------------

class TestR21x:
    """R21x fun = -p(rho).  p depends on rho via dp_drho in ctx.
    We use a linearized p(rho) = p0 + dp_drho*(rho-rho0) for FD testing.
    """

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.p0   = RNG.uniform(1e4, 2e4, N)

        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
        }
        self.term = R21x

    def test_deriv_rho(self):
        """Analytic: -dp_drho.  FD with p(rho) = p0 + dp_drho*(rho-rho0): d(-p)/drho = -dp_drho."""
        self.term.build(self.ctx)
        rho = self.rho0.copy()
        analytic = self.term.evaluate_deriv('rho', rho)
        # FD: perturb rho → p changes by dp_drho*eps
        f0 = -(self.p0)
        f1 = -(self.p0 + self.dp_drho_arr * EPS)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)


class TestR22xx:
    def setup_method(self):
        self.ctx = {}
        self.term = R22xx

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 1)


class TestR22xxS:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dx': _rand(-0.3, 0.3)}
        self.term = R22xxS

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 1)


class TestR22yx:
    def setup_method(self):
        self.ctx = {}
        self.term = R22yx

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 1)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 2)


class TestR22yxS:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dy': _rand(-0.3, 0.3)}
        self.term = R22yxS

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 1)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 2)


class TestR23xy:
    def setup_method(self):
        self.ctx = {'eta': _rand(1e-3, 1e-2)}
        self.term = R23xy

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx], 1)


class TestR24x:
    """R24x fun = 1/h * tau_xz(rho, jx).
    Use linearized tau_xz for FD test.
    """

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.h_arr         = RNG.uniform(0.5, 2.0, N)
        self.tau0          = RNG.uniform(-0.1, 0.1, N)
        self.dtau_drho     = RNG.uniform(-0.05, 0.05, N)
        self.dtau_djx      = RNG.uniform(-0.05, 0.05, N)
        self.ctx = {
            'h':             lambda: self.h_arr,
            'tau_xz':        lambda: self.tau0.copy(),
            'dtau_xz_drho':  lambda: self.dtau_drho,
            'dtau_xz_djx':   lambda: self.dtau_djx,
        }
        self.term = R24x

    def _fun_at(self, rho, jx):
        tau = (self.tau0
               + self.dtau_drho * (rho - self.rho0)
               + self.dtau_djx  * (jx  - self.jx0))
        return 1 / self.h_arr * tau

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jx0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')


class TestR25x:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'force_x': _rand(-9.8, 9.8)}
        self.term = R25x

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        _build_and_check(self.term, self.ctx, [rho], 0)


class TestR2Tx:
    def setup_method(self):
        self.ctx = {'jx_prev': _rand(0.5, 2.0), 'dt': 0.01}
        self.term = R2Tx

    def test_deriv_jx(self):
        jx = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jx], 0)


# ---------------------------------------------------------------------------
# Momentum y
# ---------------------------------------------------------------------------

class TestR21y:
    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.p0   = RNG.uniform(1e4, 2e4, N)
        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
        }
        self.term = R21y

    def test_deriv_rho(self):
        self.term.build(self.ctx)
        rho = self.rho0.copy()
        analytic = self.term.evaluate_deriv('rho', rho)
        f0 = -self.p0
        f1 = -(self.p0 + self.dp_drho_arr * EPS)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)


class TestR22xy:
    def setup_method(self):
        self.ctx = {}
        self.term = R22xy

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 1)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 2)


class TestR22xyS:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dx': _rand(-0.3, 0.3)}
        self.term = R22xyS

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 0)

    def test_deriv_jx(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 1)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jx = RNG.uniform(0.5, 2.0, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jx, jy], 2)


class TestR22yy:
    def setup_method(self):
        self.ctx = {}
        self.term = R22yy

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 0)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 1)


class TestR22yyS:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dy': _rand(-0.3, 0.3)}
        self.term = R22yyS

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 0)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 1)


class TestR23yx:
    def setup_method(self):
        self.ctx = {'eta': _rand(1e-3, 1e-2)}
        self.term = R23yx

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 0)

    def test_deriv_jy(self):
        rho = RNG.uniform(0.8, 1.5, N)
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [rho, jy], 1)


class TestR24y:
    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.h_arr     = RNG.uniform(0.5, 2.0, N)
        self.tau0      = RNG.uniform(-0.1, 0.1, N)
        self.dtau_drho = RNG.uniform(-0.05, 0.05, N)
        self.dtau_djy  = RNG.uniform(-0.05, 0.05, N)
        self.ctx = {
            'h':            lambda: self.h_arr,
            'tau_yz':       lambda: self.tau0.copy(),
            'dtau_yz_drho': lambda: self.dtau_drho,
            'dtau_yz_djy':  lambda: self.dtau_djy,
        }
        self.term = R24y

    def _fun_at(self, rho, jy):
        tau = (self.tau0
               + self.dtau_drho * (rho - self.rho0)
               + self.dtau_djy  * (jy  - self.jy0))
        return 1 / self.h_arr * tau

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jy0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jy(self):  self._check(1, 'jy')


class TestR25y:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'force_y': _rand(-9.8, 9.8)}
        self.term = R25y

    def test_deriv_rho(self):
        rho = RNG.uniform(0.8, 1.5, N)
        _build_and_check(self.term, self.ctx, [rho], 0)


class TestR2Ty:
    def setup_method(self):
        self.ctx = {'jy_prev': _rand(0.5, 2.0), 'dt': 0.01}
        self.term = R2Ty

    def test_deriv_jy(self):
        jy = RNG.uniform(0.5, 2.0, N)
        _build_and_check(self.term, self.ctx, [jy], 0)


# ---------------------------------------------------------------------------
# Energy equation
# ---------------------------------------------------------------------------

class TestR31x:
    def setup_method(self):
        self.ctx = {}
        self.term = R31x

    def _vals(self):
        return [RNG.uniform(0.8, 1.5, N),
                RNG.uniform(0.5, 2.0, N),
                RNG.uniform(1e3, 2e3, N)]

    def test_deriv_rho(self):
        _build_and_check(self.term, self.ctx, self._vals(), 0)

    def test_deriv_jx(self):
        _build_and_check(self.term, self.ctx, self._vals(), 1)

    def test_deriv_E(self):
        _build_and_check(self.term, self.ctx, self._vals(), 2)


class TestR31y:
    def setup_method(self):
        self.ctx = {}
        self.term = R31y

    def _vals(self):
        return [RNG.uniform(0.8, 1.5, N),
                RNG.uniform(0.5, 2.0, N),
                RNG.uniform(1e3, 2e3, N)]

    def test_deriv_rho(self):
        _build_and_check(self.term, self.ctx, self._vals(), 0)

    def test_deriv_jy(self):
        _build_and_check(self.term, self.ctx, self._vals(), 1)

    def test_deriv_E(self):
        _build_and_check(self.term, self.ctx, self._vals(), 2)


class TestR31Sx:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dx': _rand(-0.3, 0.3)}
        self.term = R31Sx

    def _vals(self):
        return [RNG.uniform(0.8, 1.5, N),
                RNG.uniform(0.5, 2.0, N),
                RNG.uniform(1e3, 2e3, N)]

    def test_deriv_rho(self):
        _build_and_check(self.term, self.ctx, self._vals(), 0)

    def test_deriv_jx(self):
        _build_and_check(self.term, self.ctx, self._vals(), 1)

    def test_deriv_E(self):
        _build_and_check(self.term, self.ctx, self._vals(), 2)


class TestR31Sy:
    def setup_method(self):
        self.ctx = {'h': _rand(0.5, 2.0), 'dh_dy': _rand(-0.3, 0.3)}
        self.term = R31Sy

    def _vals(self):
        return [RNG.uniform(0.8, 1.5, N),
                RNG.uniform(0.5, 2.0, N),
                RNG.uniform(1e3, 2e3, N)]

    def test_deriv_rho(self):
        _build_and_check(self.term, self.ctx, self._vals(), 0)

    def test_deriv_jy(self):
        _build_and_check(self.term, self.ctx, self._vals(), 1)

    def test_deriv_E(self):
        _build_and_check(self.term, self.ctx, self._vals(), 2)


class TestR32x:
    """R32x fun = -p(rho) * (jx/rho).  p depends on rho; need linearized p for FD."""

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.p0         = RNG.uniform(1e4, 2e4, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
        }
        self.term = R32x

    def _p(self, rho):
        return self.p0 + self.dp_drho_arr * (rho - self.rho0)

    def _fun_at(self, rho, jx):
        return -self._p(rho) * (jx / rho)

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jx0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')


class TestR32y:
    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.p0         = RNG.uniform(1e4, 2e4, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
        }
        self.term = R32y

    def _p(self, rho):
        return self.p0 + self.dp_drho_arr * (rho - self.rho0)

    def _fun_at(self, rho, jy):
        return -self._p(rho) * (jy / rho)

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jy0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jy(self):  self._check(1, 'jy')


class TestR32Sx:
    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.p0         = RNG.uniform(1e4, 2e4, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.h_arr       = RNG.uniform(0.5, 2.0, N)
        self.dh_dx_arr   = RNG.uniform(-0.3, 0.3, N)
        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
            'h':       lambda: self.h_arr,
            'dh_dx':   lambda: self.dh_dx_arr,
        }
        self.term = R32Sx

    def _p(self, rho):
        return self.p0 + self.dp_drho_arr * (rho - self.rho0)

    def _fun_at(self, rho, jx):
        scale = 1 / self.h_arr * self.dh_dx_arr
        return -self._p(rho) * (jx / rho) * scale

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jx0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')


class TestR32Sy:
    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.p0         = RNG.uniform(1e4, 2e4, N)
        self.dp_drho_arr = RNG.uniform(100.0, 300.0, N)
        self.h_arr       = RNG.uniform(0.5, 2.0, N)
        self.dh_dy_arr   = RNG.uniform(-0.3, 0.3, N)
        self.ctx = {
            'p':       lambda: self.p0.copy(),
            'dp_drho': lambda: self.dp_drho_arr,
            'h':       lambda: self.h_arr,
            'dh_dy':   lambda: self.dh_dy_arr,
        }
        self.term = R32Sy

    def _p(self, rho):
        return self.p0 + self.dp_drho_arr * (rho - self.rho0)

    def _fun_at(self, rho, jy):
        scale = 1 / self.h_arr * self.dh_dy_arr
        return -self._p(rho) * (jy / rho) * scale

    def _check(self, idx, var_name):
        self.term.build(self.ctx)
        vals = [self.rho0.copy(), self.jy0.copy()]
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals); vals_p[idx] = vals[idx] + EPS
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / EPS
        assert np.allclose(analytic, fd, atol=ATOL, rtol=1e-4)

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jy(self):  self._check(1, 'jy')


class TestR34:
    """R34 fun = -1/h*(tau_xz_bot*U_bot + tau_yz_bot*V_bot).

    The dep_vars don't appear directly; the ctx-derived quantities
    (tau_xz_bot, tau_yz_bot, etc.) encode the nonlinear dependence.
    We test the analytic derivative by checking consistency with the
    contract: der_funs[i] == d(fun)/d(dep_vars[i]) when ctx fields
    carry the linearized information dtau_xz_bot_d*.

    Since fun does not literally depend on dep_vars in the closure, the
    'correct' numerical test is to verify that the analytic derivatives
    are self-consistent with the provided ctx derivative fields — not a
    naive FD of fun.  We do this by checking that evaluate_deriv returns
    the expected analytic expression directly.
    """

    def setup_method(self):
        self.h_arr     = RNG.uniform(0.5, 2.0, N)
        self.tau_xz    = RNG.uniform(-0.1, 0.1, N)
        self.tau_yz    = RNG.uniform(-0.1, 0.1, N)
        self.dtau_xz_drho = RNG.uniform(-0.05, 0.05, N)
        self.dtau_xz_djx  = RNG.uniform(-0.05, 0.05, N)
        self.dtau_yz_drho = RNG.uniform(-0.05, 0.05, N)
        self.dtau_yz_djy  = RNG.uniform(-0.05, 0.05, N)
        self.U_bot     = RNG.uniform(-1.0, 1.0, N)
        self.V_bot     = RNG.uniform(-1.0, 1.0, N)

        def _make_ctx():
            return {
                'h':               lambda: self.h_arr,
                'tau_xz_bot':      lambda: self.tau_xz,
                'tau_yz_bot':      lambda: self.tau_yz,
                'dtau_xz_bot_drho': lambda: self.dtau_xz_drho,
                'dtau_xz_bot_djx':  lambda: self.dtau_xz_djx,
                'dtau_yz_bot_drho': lambda: self.dtau_yz_drho,
                'dtau_yz_bot_djy':  lambda: self.dtau_yz_djy,
                'U_bot':           lambda: self.U_bot,
                'V_bot':           lambda: self.V_bot,
            }
        self.ctx = _make_ctx()
        self.term = R34

    def _vals(self):
        return [RNG.uniform(0.8, 1.5, N),
                RNG.uniform(0.5, 2.0, N),
                RNG.uniform(0.5, 2.0, N)]

    def test_deriv_rho_matches_ctx_expression(self):
        """der_funs[0] = -1/h*(dtau_xz_bot_drho*U_bot + dtau_yz_bot_drho*V_bot)"""
        self.term.build(self.ctx)
        vals = self._vals()
        result = self.term.evaluate_deriv('rho', *vals)
        expected = -1 / self.h_arr * (
            self.dtau_xz_drho * self.U_bot + self.dtau_yz_drho * self.V_bot)
        assert np.allclose(result, expected, atol=1e-12)

    def test_deriv_jx_matches_ctx_expression(self):
        """der_funs[1] = -1/h * dtau_xz_bot_djx * U_bot"""
        self.term.build(self.ctx)
        vals = self._vals()
        result = self.term.evaluate_deriv('jx', *vals)
        expected = -1 / self.h_arr * self.dtau_xz_djx * self.U_bot
        assert np.allclose(result, expected, atol=1e-12)

    def test_deriv_jy_matches_ctx_expression(self):
        """der_funs[2] = -1/h * dtau_yz_bot_djy * V_bot"""
        self.term.build(self.ctx)
        vals = self._vals()
        result = self.term.evaluate_deriv('jy', *vals)
        expected = -1 / self.h_arr * self.dtau_yz_djy * self.V_bot
        assert np.allclose(result, expected, atol=1e-12)


def _make_thermal_ctx(rho0, jx0, jy0, E0, k_arr, dT_drho_arr, dT_djx_arr,
                      dT_djy_arr, dT_dE_arr):
    """Build a ctx where T is a linear function of dep_vars.

    T(rho, jx, jy, E) = T0 + dT_drho*(rho-rho0) + dT_djx*(jx-jx0)
                            + dT_djy*(jy-jy0) + dT_dE*(E-E0)
    where T0 = 0 for simplicity.
    """
    T0 = np.zeros(N)

    def T_fn(rho, jx, jy, E):
        return (T0
                + dT_drho_arr * (rho - rho0)
                + dT_djx_arr  * (jx  - jx0)
                + dT_djy_arr  * (jy  - jy0)
                + dT_dE_arr   * (E   - E0))

    # ctx fields are evaluated at the base point, so T = T0 = 0
    return {
        'k':       lambda: k_arr,
        'T':       lambda: T0.copy(),
        'dT_drho': lambda: dT_drho_arr,
        'dT_djx':  lambda: dT_djx_arr,
        'dT_djy':  lambda: dT_djy_arr,
        'dT_dE':   lambda: dT_dE_arr,
    }, T_fn


class TestR35xFD:
    """FD test for R35x using a linearized T(dep_vars) to make fun depend on them.

    R35x fun = -k * T.  With T = dT_drho*(rho-rho0) + ..., we can FD-test
    because perturbing rho changes T and therefore fun.
    """

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.E0   = RNG.uniform(1e3, 2e3, N)

        self.k_arr      = RNG.uniform(0.1, 1.0, N)
        self.dT_drho    = RNG.uniform(-10.0, 10.0, N)
        self.dT_djx     = RNG.uniform(-1.0, 1.0, N)
        self.dT_djy     = RNG.uniform(-1.0, 1.0, N)
        self.dT_dE      = RNG.uniform(-0.01, 0.01, N)

        self.ctx, self.T_fn = _make_thermal_ctx(
            self.rho0, self.jx0, self.jy0, self.E0,
            self.k_arr, self.dT_drho, self.dT_djx, self.dT_djy, self.dT_dE)

        self.term = R35x

    def _base(self):
        return [self.rho0.copy(), self.jx0.copy(), self.jy0.copy(), self.E0.copy()]

    def _fun_at(self, rho, jx, jy, E):
        """Evaluate R35x fun with updated T."""
        return -self.k_arr * self.T_fn(rho, jx, jy, E)

    def _check(self, idx, var_name, eps=EPS, atol=ATOL):
        self.term.build(self.ctx)
        vals = self._base()

        # analytical
        analytic = self.term.evaluate_deriv(var_name, *vals)

        # FD
        f0 = self._fun_at(*vals)
        vals_p = list(vals)
        vals_p[idx] = vals[idx] + eps
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / eps

        assert np.allclose(analytic, fd, atol=atol, rtol=1e-4), (
            f"R35x d/d({var_name}): max|analytic-fd|={np.max(np.abs(analytic-fd)):.3e}")

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')
    def test_deriv_jy(self):  self._check(2, 'jy')
    def test_deriv_E(self):   self._check(3, 'E')


class TestR35yFD:
    """Same as TestR35xFD but for R35y — identical fun/der_funs."""

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.E0   = RNG.uniform(1e3, 2e3, N)

        self.k_arr   = RNG.uniform(0.1, 1.0, N)
        self.dT_drho = RNG.uniform(-10.0, 10.0, N)
        self.dT_djx  = RNG.uniform(-1.0, 1.0, N)
        self.dT_djy  = RNG.uniform(-1.0, 1.0, N)
        self.dT_dE   = RNG.uniform(-0.01, 0.01, N)

        self.ctx, self.T_fn = _make_thermal_ctx(
            self.rho0, self.jx0, self.jy0, self.E0,
            self.k_arr, self.dT_drho, self.dT_djx, self.dT_djy, self.dT_dE)

        self.term = R35y

    def _base(self):
        return [self.rho0.copy(), self.jx0.copy(), self.jy0.copy(), self.E0.copy()]

    def _fun_at(self, rho, jx, jy, E):
        return -self.k_arr * self.T_fn(rho, jx, jy, E)

    def _check(self, idx, var_name, eps=EPS, atol=ATOL):
        self.term.build(self.ctx)
        vals = self._base()
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals)
        vals_p[idx] = vals[idx] + eps
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / eps
        assert np.allclose(analytic, fd, atol=atol, rtol=1e-4), (
            f"R35y d/d({var_name}): max|analytic-fd|={np.max(np.abs(analytic-fd)):.3e}")

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')
    def test_deriv_jy(self):  self._check(2, 'jy')
    def test_deriv_E(self):   self._check(3, 'E')


class TestR36FD:
    """FD test for R36 using a linearized S(dep_vars).

    R36 fun = S.  With S = dS_drho*rho + ..., perturbing dep_vars changes S
    and we can FD-test.
    """

    def setup_method(self):
        self.rho0 = RNG.uniform(0.8, 1.5, N)
        self.jx0  = RNG.uniform(0.5, 2.0, N)
        self.jy0  = RNG.uniform(0.5, 2.0, N)
        self.E0   = RNG.uniform(1e3, 2e3, N)

        self.dS_drho = RNG.uniform(-10.0, 10.0, N)
        self.dS_djx  = RNG.uniform(-1.0, 1.0, N)
        self.dS_djy  = RNG.uniform(-1.0, 1.0, N)
        self.dS_dE   = RNG.uniform(-0.01, 0.01, N)
        S0 = np.zeros(N)

        def S_fn(rho, jx, jy, E):
            return (S0
                    + self.dS_drho * (rho - self.rho0)
                    + self.dS_djx  * (jx  - self.jx0)
                    + self.dS_djy  * (jy  - self.jy0)
                    + self.dS_dE   * (E   - self.E0))

        self.S_fn = S_fn
        self.ctx = {
            'S':      lambda: S0.copy(),
            'dS_drho': lambda: self.dS_drho,
            'dS_djx':  lambda: self.dS_djx,
            'dS_djy':  lambda: self.dS_djy,
            'dS_dE':   lambda: self.dS_dE,
        }
        self.term = R36

    def _base(self):
        return [self.rho0.copy(), self.jx0.copy(), self.jy0.copy(), self.E0.copy()]

    def _fun_at(self, rho, jx, jy, E):
        return self.S_fn(rho, jx, jy, E)

    def _check(self, idx, var_name, eps=EPS, atol=ATOL):
        self.term.build(self.ctx)
        vals = self._base()
        analytic = self.term.evaluate_deriv(var_name, *vals)
        f0 = self._fun_at(*vals)
        vals_p = list(vals)
        vals_p[idx] = vals[idx] + eps
        f1 = self._fun_at(*vals_p)
        fd = (f1 - f0) / eps
        assert np.allclose(analytic, fd, atol=atol, rtol=1e-4), (
            f"R36 d/d({var_name}): max|analytic-fd|={np.max(np.abs(analytic-fd)):.3e}")

    def test_deriv_rho(self): self._check(0, 'rho')
    def test_deriv_jx(self):  self._check(1, 'jx')
    def test_deriv_jy(self):  self._check(2, 'jy')
    def test_deriv_E(self):   self._check(3, 'E')


class TestR3T:
    def setup_method(self):
        self.ctx = {'E_prev': _rand(1e3, 2e3), 'dt': 0.01}
        self.term = R3T

    def test_deriv_E(self):
        E = RNG.uniform(1e3, 2e3, N)
        _build_and_check(self.term, self.ctx, [E], 0)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestNonLinearTermStructure:
    """Basic contract tests — not built/evaluate, just object structure."""

    @pytest.mark.parametrize("term", [
        R11x, R11y, R11Sx, R11Sy, R1T,
        R21x, R21y,
        R22xx, R22xxS, R22yx, R22yxS,
        R22xy, R22xyS, R22yy, R22yyS,
        R23xy, R23yx,
        R24x, R24y,
        R25x, R25y,
        R2Tx, R2Ty,
        R31x, R31y, R31Sx, R31Sy,
        R32x, R32y, R32Sx, R32Sy,
        R34,
        R35x, R35y,
        R36,
        R3T,
    ])
    def test_has_required_attributes(self, term):
        assert hasattr(term, 'name')
        assert hasattr(term, 'dep_vars')
        assert hasattr(term, 'dep_vals')
        assert hasattr(term, 'd_dx_resfun')
        assert hasattr(term, 'd_dy_resfun')
        assert hasattr(term, 'der_testfun')
        assert term.der_testfun in (False, 'x', 'y'), (
            f"Term {term.name!r} has invalid der_testfun={term.der_testfun!r}")

    @pytest.mark.parametrize("term", [
        R11x, R11y, R11Sx, R11Sy, R1T,
        R21x, R21y,
        R22xx, R22xxS, R22yx, R22yxS,
        R22xy, R22xyS, R22yy, R22yyS,
        R23xy, R23yx,
        R24x, R24y,
        R25x, R25y,
        R2Tx, R2Ty,
        R31x, R31y, R31Sx, R31Sy,
        R32x, R32y, R32Sx, R32Sy,
        R34,
        R35x, R35y,
        R36,
        R3T,
    ])
    def test_der_funs_count_matches_dep_vars(self, term):
        assert len(term.der_funs_) == len(term.dep_vars), (
            f"Term {term.name!r}: {len(term.der_funs_)} der_funs but "
            f"{len(term.dep_vars)} dep_vars")

    @pytest.mark.parametrize("term, expected_der_testfun", [
        (R23xy, 'y'),
        (R23yx, 'x'),
        (R35x,  'x'),
        (R35y,  'y'),
    ])
    def test_der_testfun_direction(self, term, expected_der_testfun):
        assert term.der_testfun == expected_der_testfun
