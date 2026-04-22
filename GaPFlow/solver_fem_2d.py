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

"""Taylor-Hood P2P1 FEM Solver.

Wires GridIndexManager, QuadFieldManager, Assembly, and linear solver into a
Newton iteration loop.
"""

import time
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .fem_2d.elements import TaylorHoodP2P1
from .fem_2d.grid_index import GridIndexManager
from .fem_2d.quad_fields import QuadFieldManager
from .fem_2d.assembly import Assembly
from .fem_2d.terms import NonLinearTerm, get_active_terms
from .fem_2d.scaling import build_scaling
from .fem_2d.scipy_system import ScipySystem
from .fem_2d.solution_guards import linearization_guard  # noqa: F401 (apply_guards available but deactivated)

if TYPE_CHECKING:
    from .problem import Problem
    from .parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]

# Variable-to-grid mapping: which grid each variable lives on
_VAR_TO_GRID = {'jx': 'v', 'jy': 'v', 'rho': 'p', 'E': 'p'}
# Residual-to-grid mapping: which grid each residual equation lives on
_RES_TO_GRID = {
    'momentum_x': 'v',
    'momentum_y': 'v',
    'mass':       'p',
    'energy':     'p',
}


class FEMSolver2d:
    """FEM Solver for 2D Taylor-Hood P2P1 problems.

    Parameters
    ----------
    fem_spec : dict
        fem_solver config block from the Problem.
    problem : Problem
        Problem instance.
    """

    def __init__(self, fem_spec: dict, problem: "Problem") -> None:
        self.fem_spec = fem_spec
        self.problem  = problem
        self.R_norm_history: List[List[float]] = []

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_accessors(self) -> None:
        p = self.problem
        self.energy = p.fem_solver['equations'].get('energy', False)

        self.dx = p.grid['dx']
        self.dy = p.grid['dy']

        self.variables = ['jx', 'jy', 'rho']
        self.residuals = ['momentum_x', 'momentum_y', 'mass']
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')

        self.var_to_grid = {v: _VAR_TO_GRID[v] for v in self.variables}
        self.res_to_grid = {r: _RES_TO_GRID[r] for r in self.residuals}

        # Elements (carry dx, dy and shape function matrices)
        self.elements = TaylorHoodP2P1(self.dx, self.dy)

        # Grid index manager
        self.grid_idx = GridIndexManager(
            decomp=p.decomp,
            variables=self.variables,
        )

        # Per-variable inner sizes
        self.nb_inner_v = (self.grid_idx.Nx_v_inner *
                           self.grid_idx.Ny_v_inner)
        self.nb_inner_p = (self.grid_idx.Nx_p_inner *
                           self.grid_idx.Ny_p_inner)

        # Residual vector size (sum over all residuals)
        self.res_size = sum(
            self.nb_inner_v if self.res_to_grid[r] == 'v' else self.nb_inner_p
            for r in self.residuals
        )

    @property
    def _res_slices(self):
        return self.assembly._res_slices

    @property
    def _sol_slices(self):
        return self.assembly._sol_slices

    def _res_slice(self, res_name: str) -> slice:
        return self.assembly._res_slices[res_name]

    def _sol_slice(self, var_name: str) -> slice:
        return self.assembly._sol_slices[var_name]

    def _get_active_terms(self) -> None:
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_fields(self) -> None:
        self.quad_mgr = QuadFieldManager(
            problem=self.problem,
            energy=self.energy,
            variables=self.variables,
            elements=self.elements,
            decomp=self.problem.decomp,
        )

    def _build_assembly(self) -> None:
        self.assembly = Assembly(
            grid_idx=self.grid_idx,
            element=self.elements,
            variables=self.variables,
            residuals=self.residuals,
            energy=self.energy,
        )

    def _build_jit_functions(self) -> None:
        p = self.problem
        p.pressure.build_grad()
        p.wall_stress_xz.build_grad()
        p.wall_stress_yz.build_grad()
        if self.energy:
            p.energy.build_grad()

    def _build_terms(self) -> None:
        p = self.problem

        def make_getter(name):
            return lambda: self.quad_mgr.get_quad(name)

        all_field_names = self.quad_mgr._needed_fields()
        for term in self.terms:
            ctx = {name: make_getter(name) for name in all_field_names}
            ctx['dt'] = lambda: p.numerics['dt']
            ctx['mass_diff_alpha'] = lambda: p.fem_solver.get(
                'mass_diffusion_alpha', 1e-3)
            if self.energy:
                ctx['k'] = lambda: p.energy.k
            term.build(ctx)

    def _init_linear_solver(self) -> None:
        petsc_info = self.assembly.get_petsc_info(res_size=self.res_size)
        solver_type = self.problem.fem_solver.get('linear_solver', 'direct')

        try:
            from . import HAS_PETSC
        except ImportError:
            HAS_PETSC = False

        if HAS_PETSC:
            from .fem_2d.petsc_system import PETScSystem
            self.linear_solver = PETScSystem(petsc_info,
                                             solver_type=solver_type)
        else:
            if MPI.COMM_WORLD.size > 1:
                raise RuntimeError(
                    "PETSc required for parallel execution.")
            self.linear_solver = ScipySystem(petsc_info,
                                             solver_type=solver_type)

        self.scaling = build_scaling(
            self.problem, self.energy, self.variables, self.assembly)

        debug_from = self.problem.fem_solver.get('newton_debug', None)
        if debug_from is not None:
            from .fem_2d.newton_debug import NewtonDebugger
            self.debugger = NewtonDebugger(
                output_dir=self.problem.options['output'],
                variables=self.variables,
                residuals=self.residuals,
                res_slices=self.assembly._res_slices,
                sol_slices=self.assembly._sol_slices,
                Nx_p=self.grid_idx.Nx_p_inner,
                Ny_p=self.grid_idx.Ny_p_inner,
                Nx_v=self.grid_idx.Nx_v_inner,
                Ny_v=self.grid_idx.Ny_v_inner,
                terms=self.terms,
                problem=self.problem,
            )
            self._debug_from = debug_from
            self._debug_steps_done = 0
        else:
            self.debugger = None
            self._debug_from = None
            self._debug_steps_done = 0

    @property
    def _debug_active(self) -> bool:
        """True when newton_debug is enabled and the step threshold has been reached."""
        return (
            self.debugger is not None
            and self.problem.step >= self._debug_from
            and self._debug_steps_done < 5
        )

    # =========================================================================
    # Quadrature field update
    # =========================================================================

    def update_quad(self) -> None:
        self.quad_mgr.update_physics()
        self.quad_mgr.update_nodal_to_quad()
        self.quad_mgr.update_quad_computed()

    def update_prev_quad(self) -> None:
        self.quad_mgr.store_prev_values()

    # =========================================================================
    # Newton scatter / gather
    # =========================================================================

    def get_q_nodal(self) -> NDArray:
        """Gather inner nodal values into a flat solution vector."""
        q = np.zeros(self.res_size)
        for var in self.variables:
            q[self._sol_slice(var)] = self.quad_mgr.get_nodal_sol_val(var)
        return q

    def set_q_nodal(self, q: NDArray) -> None:
        """Scatter flat solution vector back to nodal fields."""
        for var in self.variables:
            self.quad_mgr.set_nodal_sol_val(var, q[self._sol_slice(var)])

    # =========================================================================
    # Ghost exchange
    # =========================================================================

    def _exchange_ghosts(self) -> None:
        """Exchange ghost cells for all grids after Newton update."""
        p = self.problem
        rho = self.quad_mgr.nodal_fields['rho']
        jx  = self.quad_mgr.nodal_fields['jx']
        jy  = self.quad_mgr.nodal_fields['jy']
        p.decomp.update_ghosts(
            exchange_specs=[(rho, 'P1'), (jx, 'P2'), (jy, 'P2')],
            bc_specs=[
                (rho.pg[0], 'rho', 'P1_nodal'),
                (jx.pg[0],  'jx',  'P2_nodal'),
                (jy.pg[0],  'jy',  'P2_nodal'),
            ],
            problem=p,
        )

    # =========================================================================
    # Assembly
    # =========================================================================

    def _build_all_quad_fields(self) -> dict:
        """Build the full quad_fields dict for all active terms.

        Includes plain values for all dep_vars and gradient fields
        ('d_dx_<var>', 'd_dy_<var>') for any variable that appears in a
        term with d_dx_resfun / d_dy_resfun set.
        """
        qf: dict = {}
        need_dx: set = set()
        need_dy: set = set()

        for term in self.terms:
            for v in term.dep_vars:
                if v not in qf:
                    qf[v] = self.quad_mgr.get_quad(v)
            if term.d_dx_resfun:
                for v in term.dep_vars:
                    need_dx.add(v)
            if term.d_dy_resfun:
                for v in term.dep_vars:
                    need_dy.add(v)

        for v in need_dx:
            key = f'd_dx_{v}'
            if key not in qf:
                qf[key] = self.quad_mgr.get_deriv_dx(v)
        for v in need_dy:
            key = f'd_dy_{v}'
            if key not in qf:
                qf[key] = self.quad_mgr.get_deriv_dy(v)

        return qf

    def get_M(self, qf: dict = None) -> NDArray:
        """Assemble Jacobian COO values."""
        if qf is None:
            qf = self._build_all_quad_fields()
        return self.assembly.assemble_matrix(qf, self.terms)

    def get_M_dense(self) -> NDArray:
        """Assemble Jacobian as a dense (res_size, res_size) matrix in block ordering."""
        coo = self.get_M()
        n_nnz = len(coo)
        block_rows = np.empty(n_nnz, dtype=np.int64)
        block_cols = np.empty(n_nnz, dtype=np.int64)

        for (res, var), block in self.assembly.block_order.items():
            s = block['nnz_idx_start']
            n = block['nb_nnz']
            row_off = self.assembly._res_slices[res].start
            col_off = self.assembly._sol_slices[var].start
            block_rows[s:s + n] = row_off + self.assembly.nnz_local_to[s:s + n]
            block_cols[s:s + n] = col_off + self.assembly.nnz_local_from[s:s + n]

        M = np.zeros((self.res_size, self.res_size), dtype=np.float64)
        np.add.at(M, (block_rows, block_cols), coo)
        return M

    def get_R_(self, qf: dict = None) -> NDArray:
        """Assemble residual vector."""
        if qf is None:
            qf = self._build_all_quad_fields()
        return self.assembly.assemble_rhs(qf, self.terms)

    def exchange_ghosts(self) -> None:
        """Exchange ghost cells (public wrapper for tests)."""
        self._exchange_ghosts()

    # =========================================================================
    # Solver step
    # =========================================================================

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        self.set_q_nodal(q_guess)
        self._exchange_ghosts()
        self.update_quad()
        qf = self._build_all_quad_fields()
        M = self.get_M(qf)
        R = self.get_R_(qf).copy()
        if self._debug_active:
            self._last_R_per_term = self.assembly.assemble_rhs_per_term(qf, self.terms)
        return M, R

    def get_R(self, q_guess: NDArray) -> float:
        self.set_q_nodal(q_guess)
        self._exchange_ghosts()
        self.update_quad()
        qf = self._build_all_quad_fields()
        R = self.get_R_(qf).copy()
        return R

    def get_R_norm_global(self, R: NDArray) -> float:
        p = self.problem
        comm = p.decomp._mpi_comm
        R_norm_local_sq = float(np.linalg.norm(R) ** 2)
        R_norm_global_sq = comm.allreduce(R_norm_local_sq, op=MPI.SUM)
        R_norm_global = float(np.sqrt(R_norm_global_sq))
        return R_norm_global

    # =========================================================================
    # Output helpers
    # =========================================================================

    def update_output_fields(self) -> None:
        p = self.problem
        self.quad_mgr.sync_to_problem_q()
        p.wall_stress_xz.update()
        p.wall_stress_yz.update()
        if hasattr(p, 'bulk_stress'):
            p.bulk_stress.update()

    # =========================================================================
    # Conservative density smoothing
    # =========================================================================

    def smooth_rho(self, q: NDArray, beta: float, delta: float) -> NDArray:
        """Apply one conservative flux-based smoothing sweep to the rho component.

        Smoothing is localised around rho_l using a Gaussian weight:
          w(rho) = exp(-((rho - rho_l) / delta)^2)
        so at rho = rho_l +/- 3*delta the weight is ~0.0001 (negligible).
        Set delta = eps / 3 so the smoothing support matches the EoS blend window.

        For each pair of neighboring nodes, the flux is weighted by the
        maximum of the two neighbors' weights, so smoothing activates if
        *either* node is near the cavitation front. Mass is exactly conserved.

        Parameters
        ----------
        q : NDArray
            Flat Newton solution vector (modified in-place and returned).
        beta : float
            Smoothing strength in (0, 0.25) for stability.
        delta : float
            Gaussian half-width in density units. 3*delta ~ effective support.
        """
        rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))
        rho_slice = self._sol_slice('rho')
        Nx, Ny = self.problem.decomp.nb_subdomain_grid_pts
        rho = q[rho_slice].reshape((Nx, Ny), order='F')

        w = np.exp(-((rho - rho_l) / delta) ** 2)

        # x-direction fluxes (between i and i+1)
        w_x = np.maximum(w[:-1, :], w[1:, :])
        flux_x = beta * w_x * (rho[:-1, :] - rho[1:, :])
        rho[:-1, :] -= flux_x
        rho[1:, :] += flux_x

        # y-direction fluxes (between j and j+1)
        w_y = np.maximum(w[:, :-1], w[:, 1:])
        flux_y = beta * w_y * (rho[:, :-1] - rho[:, 1:])
        rho[:, :-1] -= flux_y
        rho[:, 1:] += flux_y

        q[rho_slice] = rho.flatten(order='F')
        return q

    # =========================================================================
    # Cavitation guard
    # =========================================================================

    def _detect_rho_oscillations(self, rho: np.ndarray) -> np.ndarray:
        """Detect spurious density oscillations (checkerboard / wiggles).

        A node is flagged if the density gradient changes sign between its
        left and right neighbors (local extremum) in either x or y direction.
        Only flags nodes near rho_l (within 5 * grid spacing worth of density
        variation) to avoid flagging physical features far from cavitation.

        Returns a boolean mask of shape (Nx, Ny) where True = oscillating.
        """
        rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))
        Nx, Ny = rho.shape

        osc = np.zeros_like(rho, dtype=bool)

        # x-direction: sign change in diff between consecutive pairs
        if Nx >= 3:
            dx = np.diff(rho, axis=0)  # (Nx-1, Ny)
            sign_change_x = dx[:-1, :] * dx[1:, :] < 0  # (Nx-2, Ny)
            osc[1:-1, :] |= sign_change_x

        # y-direction: same logic
        if Ny >= 3:
            dy = np.diff(rho, axis=1)  # (Nx, Ny-1)
            sign_change_y = dy[:, :-1] * dy[:, 1:] < 0  # (Nx, Ny-2)
            osc[:, 1:-1] |= sign_change_y

        # Restrict to vicinity of rho_l
        near_cav = np.abs(rho - rho_l) < 5.0
        osc &= near_cav

        n_osc = int(np.sum(osc))
        if n_osc > 0:
            osc_rho = rho[osc]
            ix, iy = np.where(osc)
            print(f"  Cavitation guard: {n_osc} oscillating nodes detected "
                  f"(rho range [{osc_rho.min():.4f}, {osc_rho.max():.4f}], "
                  f"x=[{ix.min()},{ix.max()}], y=[{iy.min()},{iy.max()}])")
        return osc

    def _detect_rho_crossings(self, rho_old: np.ndarray,
                               rho_new: np.ndarray) -> np.ndarray:
        """Detect nodes where rho crossed rho_l during the Newton update.

        A crossing means the node switched between liquid (rho > rho_l)
        and mixture (rho < rho_l) in a single Newton step, which can cause
        the solver to oscillate between branches of the piecewise EoS.

        Returns a boolean mask of shape (Nx, Ny) where True = crossed.
        """
        rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))

        crossed = (rho_old - rho_l) * (rho_new - rho_l) < 0

        n_cross = int(np.sum(crossed))
        if n_cross > 0:
            # Which direction: liquid->mixture or mixture->liquid
            to_mix = crossed & (rho_old > rho_l)
            to_liq = crossed & (rho_old < rho_l)
            delta = np.abs(rho_new[crossed] - rho_old[crossed])
            ix, iy = np.where(crossed)
            print(f"  Cavitation guard: {n_cross} rho_l crossings "
                  f"({int(np.sum(to_mix))} liq->mix, "
                  f"{int(np.sum(to_liq))} mix->liq, "
                  f"max |drho|={delta.max():.4f}, "
                  f"x=[{ix.min()},{ix.max()}], y=[{iy.min()},{iy.max()}])")
        return crossed

    def _pressure_liquid(self, rho: np.ndarray) -> np.ndarray:
        """Fast inline pressure for liquid branch: p = Pcav + (rho - rho_l) * c_l²."""
        prop = self.problem.prop
        rho_l = prop.get('rho_l', prop.get('rho0'))
        c_l = prop.get('c_l', 1.0)
        if not hasattr(self, '_Pcav'):
            from .models.pressure import eos_pressure
            self._Pcav = float(eos_pressure(np.array(rho_l), prop))
        return self._Pcav + (rho - rho_l) * c_l**2

    def _limit_pressure_change(self, rho_old: np.ndarray, rho_new: np.ndarray,
                               p_old: np.ndarray, p_new: np.ndarray,
                               max_rel_dp: float = 0.5) -> float:
        """Compute step limiting factor from pressure change on liquid-side nodes.

        For nodes with rho > rho_l where |dp/p_old| exceeds max_rel_dp,
        compute the density that gives exactly the 50% capped pressure,
        then derive the limiting factor from drho_limited / drho.

        On the liquid branch: p = Pcav + (rho - rho_l) * c_l², so
        rho(p) = rho_l + (p - Pcav) / c_l².

        Returns the global minimum limiting factor (1.0 if no limiting needed).
        """
        rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))
        c_l = self.problem.prop.get('c_l', 1.0)
        if not hasattr(self, '_Pcav'):
            from .models.pressure import eos_pressure
            self._Pcav = float(eos_pressure(np.array(rho_l), self.problem.prop))
        p_cav = self._Pcav

        # Only consider liquid-side nodes (rho_old > rho_l)
        liquid = rho_old > rho_l
        if not np.any(liquid):
            return 1.0

        p_o = p_old[liquid]
        p_n = p_new[liquid]
        rel_dp = np.abs(p_n - p_o) / (np.abs(p_o) + 1e-10)

        exceeds = rel_dp > max_rel_dp
        if not np.any(exceeds):
            return 1.0

        # For each exceeding node, compute limited pressure and corresponding rho
        drho_full = rho_new[liquid] - rho_old[liquid]
        sign_dp = np.sign(p_n - p_o)
        p_limited = p_o + sign_dp * max_rel_dp * np.abs(p_o)
        # Invert liquid branch: rho = rho_l + (p - Pcav) / c_l²
        rho_limited = rho_l + (p_limited - p_cav) / c_l**2
        drho_limited = rho_limited - rho_old[liquid]

        # Per-node factor: drho_limited / drho (only for exceeding nodes)
        # Avoid div-by-zero for nodes with drho ~ 0
        factors = np.where(
            exceeds & (np.abs(drho_full) > 1e-15),
            np.abs(drho_limited) / (np.abs(drho_full) + 1e-30),
            1.0,
        )
        f_min = float(np.min(factors))

        rank = self.problem.decomp.rank
        if rank == 0:
            n_exc = int(np.sum(exceeds))
            ix_all, iy_all = np.where(liquid)
            iworst = np.argmax(rel_dp)
            print(f"  Pressure guard: {n_exc} nodes exceed {max_rel_dp:.0%} dp, "
                  f"max |dp/p|={rel_dp[iworst]:.2e} "
                  f"at ({ix_all[iworst]},{iy_all[iworst]}), "
                  f"f_min={f_min:.4f}")
        return f_min

    def cavitation_guard(self, q_old: NDArray, q_new: NDArray) -> NDArray:
        """Check for cavitation-related issues and limit the step if needed.

        Detects:
          1. Spurious rho oscillations (checkerboard pattern near rho_l)
          2. Nodes where rho crossed rho_l during the update

        If crossings are detected, the step is reduced so that the most
        critical node barely crosses rho_l (plus eps), with a 0.5 safety
        factor. This limits the full dq vector, not just rho.

        Parameters
        ----------
        q_old : NDArray
            Flat solution vector before the Newton update.
        q_new : NDArray
            Flat solution vector after the Newton update (= q_old + alpha*dq).

        Returns
        -------
        NDArray
            Possibly reduced solution vector.
        """
        rho_slice = self._sol_slice('rho')
        Nx, Ny = self.problem.decomp.nb_subdomain_grid_pts

        rho_old = q_old[rho_slice].reshape((Nx, Ny), order='F')
        rho_new = q_new[rho_slice].reshape((Nx, Ny), order='F')

        # Early exit: if no node is near rho_l, skip all checks
        rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))
        near_threshold = 5.0
        if not (np.any(np.abs(rho_old - rho_l) < near_threshold)
                or np.any(np.abs(rho_new - rho_l) < near_threshold)):
            return q_new

        osc_mask = self._detect_rho_oscillations(rho_new)
        cross_mask = self._detect_rho_crossings(rho_old, rho_new)

        # Liquid-branch pressure (fast inline, no EoS dispatcher)
        p_old = self._pressure_liquid(rho_old)
        p_new = self._pressure_liquid(rho_new)

        # Pressure change limiter
        max_rel_dp = self.problem.fem_solver.get('cavitation_guard_max_rel_dp', 0.1)
        f_pressure = self._limit_pressure_change(
            rho_old, rho_new, p_old, p_new, max_rel_dp)

        # Crossing limiter
        f_crossing = 1.0
        if np.any(cross_mask):
            rho_l = self.problem.prop.get('rho_l', self.problem.prop.get('rho0'))
            eps = self.problem.fem_solver.get('cavitation_guard_eps', 1e-3)
            safety = self.problem.fem_solver.get('cavitation_guard_safety', 0.1)

            drho = rho_new[cross_mask] - rho_old[cross_mask]
            target = rho_l + np.sign(drho) * eps
            f = (target - rho_old[cross_mask]) / drho

            f_min = float(np.min(f))
            f_crossing = f_min * safety

            rank = self.problem.decomp.rank
            if rank == 0:
                print(f"  Cavitation guard: crossing limiter f={f_crossing:.4f} "
                      f"(f_min={f_min:.4f}, {int(np.sum(cross_mask))} crossings)")

        # Apply the most restrictive factor
        f_total = min(f_crossing, f_pressure)
        if f_total < 1.0:
            dq_full = q_new - q_old
            return q_old + f_total * dq_full

        return q_new

    # =========================================================================
    # Time step
    # =========================================================================

    def update_dynamic(self) -> None:
        p = self.problem
        fem_solver = p.fem_solver

        self.update_prev_quad()

        tic = time.time()
        q = self.get_q_nodal().copy()

        max_iter = 1 if self._debug_active else fem_solver['max_iter']
        tol = fem_solver['R_norm_tol']
        alpha = fem_solver.get('newton_relax', 1.0)
        alpha_init = alpha
        dt_init = p.numerics['dt']
        md_alpha_init = fem_solver.get('mass_diffusion_alpha', 1e-3)
        md_alpha_max = fem_solver.get('mass_diffusion_alpha_max', md_alpha_init * 1e5)
        comm = p.decomp._mpi_comm
        rank = p.decomp.rank

        if rank == 0:
            self.R_norm_history.append([])

        any_guard_fired = False
        for it in range(max_iter):
            M, R = self.solver_step_fun(q)
            R_norm = self.get_R_norm_global(R)

            if rank == 0:
                self.R_norm_history[-1].append(R_norm)
                print(R_norm)

            if R_norm < tol and it > 0:
                break

            if fem_solver.get('scaling', False):
                M_scaled, R_scaled = self.scaling.scale_system(M, R)
                self.linear_solver.assemble(M_scaled, R_scaled)
                dq_scaled = self.linear_solver.solve()
                dq = self.scaling.unscale_solution(dq_scaled)
            else:
                M_scaled = M
                self.linear_solver.assemble(M, R)
                dq = self.linear_solver.solve()

            if self._debug_active:
                self.debugger.step(
                    timestep=p.step, it=it, R=R, dq=dq, q=q,
                    R_per_term=self._last_R_per_term, M_scaled=M_scaled)
                self._debug_steps_done += 1

            q_before = q.copy()
            q = q + alpha * dq

            # Linearization guard: limit step so dp/drho stays within tolerance
            q, fired = linearization_guard(q_before, q - q_before, self)
            any_guard_fired |= fired

            if fem_solver.get('line_search', False):
                dq_guarded = q - q_before
                R_new = self.get_R(q)
                R_new_norm = self.get_R_norm_global(R_new)
                if rank == 0:
                    print(f"  [LineSearch] R: {R_norm:.6e} -> {R_new_norm:.6e}"
                          f" ({'OK' if R_new_norm < R_norm else 'INCREASED'})")
                if R_new_norm >= R_norm:
                    ls_alpha = 0.5
                    ls_min = fem_solver.get('line_search_alpha_min', 1e-8)
                    accepted = False
                    while ls_alpha >= ls_min:
                        q_trial = q_before + ls_alpha * dq_guarded
                        R_trial = self.get_R(q_trial)
                        R_trial_norm = self.get_R_norm_global(R_trial)
                        if R_trial_norm < R_norm:
                            q = q_trial
                            accepted = True
                            if rank == 0:
                                print(f"  [LineSearch] accepted ls_alpha={ls_alpha:.2e},"
                                      f" R {R_norm:.4e} -> {R_trial_norm:.4e}")
                            break
                        ls_alpha *= 0.5
                    if not accepted:
                        q = q_before
                        if rank == 0:
                            print(f"  [LineSearch] exhausted (ls_min={ls_min:.2e}),"
                                  f" reverting step, stopping simulation")
                        p._stop = True
                        break

            if fem_solver.get('mass_diffusion_adaptive', False):
                R_new = self.get_R(q)
                R_new_norm = self.get_R_norm_global(R_new)
                print(f"new R norm: {R_new_norm:.4e}, old R norm: {R_norm:.4e}")
                md_alpha = fem_solver['mass_diffusion_alpha']
                if R_new_norm > R_norm:
                    q = q - alpha * dq
                    md_alpha_new = min(md_alpha * 2.0, md_alpha_max)
                    fem_solver['mass_diffusion_alpha'] = md_alpha_new
                    if rank == 0:
                        print(f"Mass diffusion: R increased, "
                              f"alpha {md_alpha:.2e} -> {md_alpha_new:.2e}")
                    md_alpha = md_alpha_new
                else:
                    if md_alpha > md_alpha_init:
                        md_alpha_new = max(md_alpha / 1.5, md_alpha_init)
                        fem_solver['mass_diffusion_alpha'] = md_alpha_new
                        if rank == 0:
                            print(f"Mass diffusion: R decreased, "
                                  f"alpha {md_alpha:.2e} -> {md_alpha_new:.2e}")
                        md_alpha = md_alpha_new


            if fem_solver.get('rho_smoothing', False):
                R_new = self.get_R(q)
                R_new_norm = self.get_R_norm_global(R_new)
                if R_new_norm > R_norm:
                    q = q - alpha * dq
                    beta = fem_solver.get('rho_smoothing_beta', 1e-03)
                    delta = fem_solver.get('rho_smoothing_delta', 0.01/3)
                    q = self.smooth_rho(q, beta, delta)
                    if rank == 0:
                        print(f"Rho smoothing: R increased "
                              f"({R_norm:.2e} -> {R_new_norm:.2e}), "
                              f"applied beta={beta:.2e}")

            if fem_solver.get('line_search_then_smooth', False):
                alpha_min = fem_solver.get('line_search_alpha_min', 1e-8)
                R_new = self.get_R(q)
                R_new_norm = self.get_R_norm_global(R_new)
                if R_new_norm > R_norm:
                    # Phase 1: backtrack alpha until R decreases or alpha_min reached
                    q = q - alpha * dq  # revert
                    trial_alpha = alpha * 0.5
                    while trial_alpha >= alpha_min:
                        q_trial = q + trial_alpha * dq
                        R_trial = self.get_R(q_trial)
                        R_trial_norm = self.get_R_norm_global(R_trial)
                        if R_trial_norm < R_norm:
                            q = q_trial
                            if rank == 0:
                                print(f"LS+smooth: accepted alpha={trial_alpha:.2e}")
                            alpha = trial_alpha
                            break
                        trial_alpha *= 0.5
                    else:
                        # Phase 2: line search exhausted, apply smoothing
                        beta = fem_solver.get('rho_smoothing_beta', 1e-3)
                        delta = fem_solver.get('rho_smoothing_delta', 0.01/3)
                        q = self.smooth_rho(q, beta, delta)
                        if rank == 0:
                            print(f"LS+smooth: line search exhausted "
                                  f"(alpha_min={alpha_min:.2e}), "
                                  f"applied rho smoothing beta={beta:.2e}")

            # if fem_solver.get('cavitation_guard', True):
            #     q = self.cavitation_guard(q_before, q)

            # q, fired = apply_guards(q_before, q - q_before, self)
            # any_guard_fired |= fired

            self.set_q_nodal(q)
            self._exchange_ghosts()

        p.numerics['dt'] = dt_init
        fem_solver['mass_diffusion_alpha'] = md_alpha_init

        toc = time.time()
        self.time_inner = toc - tic
        self.inner_iterations = it + 1

        # Signal the PID to hold during the next outer timestep if the
        # solution guard had to intervene at any point in this timestep.
        if any_guard_fired:
            p.topo._fb_hold_next = True

        if self.debugger is not None and self._debug_steps_done >= 5:
            p._stop = True

        self.update_output_fields()
        p._post_update()

    def update(self) -> None:
        self.update_dynamic()

    # =========================================================================
    # Pre-run setup
    # =========================================================================

    def pre_run(self, **kwargs) -> None:
        self._init_accessors()
        self._init_quad_fields()
        self._get_active_terms()
        self._build_assembly()
        self._build_jit_functions()
        self._build_terms()
        self.assembly.build_assembly_templates(self.terms)
        self._init_linear_solver()

        # Populate fine-grid corners from problem.q initial state
        self.quad_mgr.sync_from_problem_q()
        self._exchange_ghosts()

        self.update_quad()
        self.update_prev_quad()
        self.update_output_fields()

        self.time_inner = 0.0
        self.inner_iterations = 0

    # =========================================================================
    # Status / diagnostics
    # =========================================================================

    def print_status_header(self) -> None:
        p = self.problem
        if p.options.get('print_progress') and p.decomp.rank == 0:
            print(75 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} "
                  f"{'Iter':<6s} {'Conv. Time':<12s} {'Residual':<12s}")
            print(75 * '-')
        if p.options.get('save_output'):
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        p = self.problem
        if scalars and p.options.get('print_progress') and p.decomp.rank == 0:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} "
                  f"{self.inner_iterations:<6d} "
                  f"{self.time_inner:<12.4e} {p.residual:<12.4e}")
