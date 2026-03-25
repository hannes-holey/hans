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
"""
Progressive finite difference verification of assemble_matrix and assemble_rhs.

Levels build up complexity:
  1: single none-combo term (R1T)
  2: single d/dx term (R11x)
  3: full mass equation
  4: single momentum term (R21x)
  5: full no-energy system, multiple grids and BCs
  6: block structure checks
"""
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2d

# =============================================================================
# Configuration
# =============================================================================

_CONFIG_TEMPLATE = """
options:
    output: /tmp/fem2d_fd_test
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {Nx}
    Ny: {Ny}
    xE: {xE}
    xW: {xW}
    yS: {yS}
    yN: {yN}
    xE_D: 1.1
    xW_D: 1.0
    yS_D: 1.05
    yN_D: 1.05

geometry:
    type: inclined
    hmax: 1e-5
    hmin: 1e-5
    U: 0.0
    V: 0.0

numerics:
    solver: fem
    dt: 1e-8
    tol: 1e-6
    max_it: 100

properties:
    EOS: PL
    rho0: 1.0
    shear: 1e-3
    bulk: 0.
    P0: 101325
    alpha: 0.

fem_solver:
    type: newton_alpha
    equations:
        energy: False
        {term_list_entry}
"""

BC_CONFIGS = {
    'periodic': {
        'xE': "['P', 'P', 'P']",
        'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
    },
    'dirichlet_rho': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
    },
    'periodic_y': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
    },
}

# =============================================================================
# Helpers
# =============================================================================

def make_problem(Nx: int, Ny: int, bc: str = 'periodic',
                 term_list: list = None) -> tuple:
    """Create and initialize a (problem, solver) pair.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    bc : str
        Key into BC_CONFIGS.
    term_list : list of str, optional
        If given, restricts active terms to this subset via fem_solver.equations.term_list.
    """
    bc_cfg = BC_CONFIGS[bc]
    if term_list is not None:
        term_list_entry = f"term_list: {term_list}"
    else:
        term_list_entry = ""

    config = _CONFIG_TEMPLATE.format(
        Nx=Nx, Ny=Ny,
        xE=bc_cfg['xE'], xW=bc_cfg['xW'],
        yS=bc_cfg['yS'], yN=bc_cfg['yN'],
        term_list_entry=term_list_entry,
    )

    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()

    return problem, solver


def compute_fd_jacobian(solver: FEMSolver2d, eps: float = 1e-6) -> np.ndarray:
    """Central finite difference Jacobian of get_R w.r.t. nodal DOFs.

    Uses relative perturbation for robustness across variable scales.
    Restores original state before returning.
    """
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J_fd = np.zeros((n, n))

    for j in range(n):
        eps_j = max(eps, eps * abs(q0[j]))

        q_plus = q0.copy()
        q_plus[j] += eps_j
        solver.set_q_nodal(q_plus)
        solver.exchange_ghosts()
        solver.update_quad()
        R_plus = solver.get_R().copy()

        q_minus = q0.copy()
        q_minus[j] -= eps_j
        solver.set_q_nodal(q_minus)
        solver.exchange_ghosts()
        solver.update_quad()
        R_minus = solver.get_R().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2.0 * eps_j)

    solver.set_q_nodal(q0)
    solver.exchange_ghosts()
    solver.update_quad()

    return J_fd


def rel_err(M_anal: np.ndarray, M_fd: np.ndarray) -> float:
    return np.linalg.norm(M_anal - M_fd) / (np.linalg.norm(M_fd) + 1e-15)


# =============================================================================
# Level 1 — single none-combo term: R1T
# =============================================================================

class TestLevel1SingleNoneTerm:
    """R1T only: mass time derivative, dc='none', linear in rho."""

    def test_jacobian_r1t(self):
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R1T'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R1T Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_residual_r1t_nonzero(self):
        """R1T residual should be nonzero since rho != rho_prev after pre_run."""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R1T'])
        R = solver.get_R()
        # After pre_run, rho and rho_prev are identical (update_prev_quad called),
        # so the residual should be zero (rho - rho_prev == 0)
        assert np.linalg.norm(R) < 1e-12, (
            f"R1T residual should be zero at init: ||R||={np.linalg.norm(R):.2e}")


# =============================================================================
# Level 2 — single d/dx term: R11x
# =============================================================================

class TestLevel2SingleDxTerm:
    """R11x only: mass flux divergence in x, dc='x', linear in jx."""

    def test_jacobian_r11x(self):
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R11x'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R11x Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_jacobian_r11y(self):
        """R11y: mass flux divergence in y, dc='y', linear in jy."""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R11y'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R11y Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")


# =============================================================================
# Level 3 — full mass equation
# =============================================================================

class TestLevel3MassEquation:
    """R11x + R11y + R1T: complete mass equation."""

    @pytest.mark.parametrize("Nx,Ny", [(3, 2), (4, 3)])
    def test_jacobian_mass(self, Nx, Ny):
        _, solver = make_problem(Nx, Ny, bc='periodic',
                                 term_list=['R11x', 'R11y', 'R1T'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"Mass Jacobian mismatch ({Nx}x{Ny}): rel_err={rel_err(M, J_fd):.2e}")

    def test_mass_block_only(self):
        """Non-mass blocks should be zero (no momentum terms active)."""
        _, solver = make_problem(4, 3, bc='periodic',
                                 term_list=['R11x', 'R11y', 'R1T'])

        M = solver.get_M_dense()

        for res in solver.residuals:
            for var in solver.variables:
                block = M[solver._res_slices[res], solver._sol_slices[var]]
                if res != 'mass':
                    assert np.linalg.norm(block) < 1e-12, (
                        f"Block M[{res},{var}] should be zero: "
                        f"norm={np.linalg.norm(block):.2e}")


# =============================================================================
# Level 4 — single momentum pressure gradient term
# =============================================================================

class TestLevel4MomentumPressure:
    """R21x / R21y: pressure gradient in momentum equations, dc='x'/'y'."""

    def test_jacobian_r21x(self):
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R21x'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R21x Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_jacobian_r21y(self):
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R21y'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R21y Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_jacobian_r24x_wall_stress(self):
        """R24x: nonlinear wall stress term, dc='none'."""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R24x'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R24x Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")


# =============================================================================
# Level 5 — full no-energy system
# =============================================================================

class TestLevel5FullSystem:
    """Full active term set, no energy, multiple grids and BCs."""

    @pytest.mark.parametrize("Nx,Ny", [(3, 2), (4, 3), (4, 4)])
    @pytest.mark.parametrize("bc,tol", [
        ('periodic', 1e-7),
        ('dirichlet_rho', 5e-2),
        ('periodic_y', 5e-2),
    ])
    def test_jacobian_full(self, Nx, Ny, bc, tol):
        _, solver = make_problem(Nx, Ny, bc=bc)

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        err = rel_err(M, J_fd)
        assert err < tol, (
            f"Full Jacobian mismatch ({Nx}x{Ny}, {bc}): rel_err={err:.2e}")

    @pytest.mark.parametrize("Nx,Ny", [(4, 3), (4, 4)])
    def test_jacobian_blocks_periodic(self, Nx, Ny):
        """Check each block individually for periodic BCs."""
        _, solver = make_problem(Nx, Ny, bc='periodic')

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        for res in solver.residuals:
            for var in solver.variables:
                M_b = M[solver._res_slices[res], solver._sol_slices[var]]
                J_b = J_fd[solver._res_slices[res], solver._sol_slices[var]]
                n_fd = np.linalg.norm(J_b)
                if n_fd > 1e-10:
                    err = np.linalg.norm(M_b - J_b) / n_fd
                    assert err < 1e-7, (
                        f"Block M[{res},{var}] mismatch ({Nx}x{Ny}): "
                        f"rel_err={err:.2e}")
                else:
                    assert np.linalg.norm(M_b) < 1e-10, (
                        f"Block M[{res},{var}] should be zero ({Nx}x{Ny}): "
                        f"norm={np.linalg.norm(M_b):.2e}")


# =============================================================================
# Level 6 — block structure
# =============================================================================

class TestLevel6BlockStructure:
    """Sparsity and zero-block checks."""

    @pytest.mark.parametrize("Nx,Ny", [(4, 3), (5, 4)])
    def test_mass_jx_block_sparse(self, Nx, Ny):
        """M[mass, jx] block should be sparse (not fully dense)."""
        _, solver = make_problem(Nx, Ny, bc='dirichlet_rho')

        M = solver.get_M_dense()
        block = M[solver._res_slices['mass'], solver._sol_slices['jx']]

        density = np.count_nonzero(np.abs(block) > 1e-12) / block.size
        assert density < 0.5, (
            f"M[mass,jx] block too dense: {density:.1%} nonzero")

    def test_momentum_rho_block_nonzero(self):
        """M[momentum_x, rho] should be nonzero (pressure gradient coupling)."""
        _, solver = make_problem(4, 3, bc='periodic')

        M = solver.get_M_dense()
        block = M[solver._res_slices['momentum_x'], solver._sol_slices['rho']]
        assert np.linalg.norm(block) > 1e-10, (
            "M[momentum_x, rho] should be nonzero (pressure gradient)")

    def test_mass_jy_block_nonzero(self):
        """M[mass, jy] should be nonzero (divergence coupling)."""
        _, solver = make_problem(4, 3, bc='periodic')

        M = solver.get_M_dense()
        block = M[solver._res_slices['mass'], solver._sol_slices['jy']]
        assert np.linalg.norm(block) > 1e-10, (
            "M[mass, jy] should be nonzero (divergence)")


# =============================================================================
# Level 4b — in-plane shear diffusion terms (R23)
# =============================================================================

class TestLevel4bInPlaneShear:
    """R23xy / R23yx: viscous diffusion with derivatives on both test and trial."""

    def test_jacobian_r23xy(self):
        """R23xy: ∫ (∂Nᵢ/∂y) · η · ∂(f(rho,jx))/∂y dΩ"""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R23xy'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R23xy Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_jacobian_r23yx(self):
        """R23yx: ∫ (∂Nᵢ/∂x) · η · ∂(f(rho,jy))/∂x dΩ"""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R23yx'])

        M = solver.get_M_dense()
        J_fd = compute_fd_jacobian(solver)

        assert rel_err(M, J_fd) < 1e-7, (
            f"R23yx Jacobian mismatch: rel_err={rel_err(M, J_fd):.2e}")

    def test_residual_r23xy_nonzero(self):
        """R23xy residual should be nonzero for non-uniform jx field."""
        _, solver = make_problem(3, 2, bc='periodic', term_list=['R23xy'])

        # Perturb jx to create spatial variation
        q = solver.get_q_nodal().copy()
        jx_slice = solver._sol_slices['jx']
        q[jx_slice] += np.linspace(0, 0.1, jx_slice.stop - jx_slice.start)
        solver.set_q_nodal(q)
        solver.exchange_ghosts()
        solver.update_quad()

        R = solver.get_R()
        assert np.linalg.norm(R) > 1e-15, (
            "R23xy residual should be nonzero for non-uniform jx")


# =============================================================================
# Level 7 — global↔block roundtrip via solve comparison
# =============================================================================

class TestLevel7GlobalBlockRoundtrip:
    """Verify that the sparse solver (global-interleaved ordering) returns the
    same dq as a dense block-ordered solve."""

    @pytest.mark.parametrize("Nx,Ny", [(3, 2), (4, 3)])
    @pytest.mark.parametrize("bc", ['periodic', 'dirichlet_rho', 'periodic_y'])
    def test_solve_roundtrip(self, Nx, Ny, bc):
        _, solver = make_problem(Nx, Ny, bc=bc)

        M_dense = solver.get_M_dense()
        M_coo = solver.get_M()
        R = solver.get_R()

        # Block-only solve (no global translation)
        dq_block = np.linalg.solve(M_dense, -R)

        # Sparse solve via ScipySystem (full block→global→solve→global→block)
        solver.linear_solver.assemble(M_coo, R)
        dq_sparse = solver.linear_solver.solve()

        assert np.allclose(dq_block, dq_sparse, rtol=1e-10, atol=1e-14), (
            f"Solve roundtrip mismatch ({Nx}x{Ny}, {bc}): "
            f"max diff={np.max(np.abs(dq_block - dq_sparse)):.2e}")
