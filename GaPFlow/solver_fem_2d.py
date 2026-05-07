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
from .fem_2d.scaling import build_scaling, build_scaling_from_blocks
from .fem_2d.scipy_system import ScipySystem
from .fem_2d.solution_guards import (linearization_guard, linearization_guard_p,  # noqa: F401
                                      report_jacobian_block_changes)
from .models.pressure import eos_pressure, eos_rho

if TYPE_CHECKING:
    from .problem import Problem
    from .parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]

# Variable-to-grid mapping: which grid each variable lives on
_VAR_TO_GRID = {'jx': 'v', 'jy': 'v', 'p': 'p', 'E': 'p', 'theta': 'p'}
# Residual-to-grid mapping: which grid each residual equation lives on
_RES_TO_GRID = {
    'momentum_x': 'v',
    'momentum_y': 'v',
    'mass':       'p',
    'energy':     'p',
    'fb':         'p',
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
        self.R_scaled_norm_history: List[List[float]] = []

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_accessors(self) -> None:
        p = self.problem
        self.energy = p.fem_solver['equations'].get('energy', False)
        self.cavitation = p.fem_solver['equations'].get('cavitation', False)

        self.dx = p.grid['dx']
        self.dy = p.grid['dy']

        self.variables = ['jx', 'jy', 'p']
        self.residuals = ['momentum_x', 'momentum_y', 'mass']
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
        if self.cavitation:
            self.variables.append('theta')
            self.residuals.append('fb')

        self._sync_rho = p.fem_solver.get('sync_rho_before_exchange', True)

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
            cavitation=self.cavitation,
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
            ctx['lap_p_alpha'] = lambda: p.fem_solver.get(
                'lap_pressure_alpha', 0.0)
            ctx['lap_theta_alpha'] = lambda: p.fem_solver.get(
                'lap_theta_alpha', 0.0)
            ctx['theta_stab_alpha'] = lambda: p.fem_solver.get(
                'theta_stab_alpha', 0.0)
            ctx['p_cav'] = lambda: p.prop['p_cav']
            ctx['pen_eps'] = lambda: p.fem_solver.get('pen_eps', 0.0)
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
            self.problem, self.energy, self.variables, self.assembly,
            cavitation=self.cavitation)

        if self.problem.decomp.rank == 0:
            p_ref = self.scaling.char_scales['p']
            print(f"[FEMSolver2d] Using reference pressure p_ref = "
                  f"{p_ref:.3e} Pa for scaling")

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
        """Exchange ghost cells for all grids after Newton update.

        Pressure-based ghost exchange (Approach B):
          - MPI halo exchange on p, rho (P1), jx, jy (P2). p and rho are
            both exchanged so rank-boundary ghosts of both are consistent
            (neighbour set rho = eos_rho(p) before sending).
          - User BC callback fills rho at physical boundaries (unchanged API).
          - Post-step: fill p at physical-boundary ghosts from the freshly
            BC-applied rho via EoS forward.
        """
        p = self.problem
        rho = self.quad_mgr.nodal_fields['rho']
        p_field = self.quad_mgr.nodal_fields['p']
        jx  = self.quad_mgr.nodal_fields['jx']
        jy  = self.quad_mgr.nodal_fields['jy']

        exchange_specs = [(rho, 'P1'), (p_field, 'P1'), (jx, 'P2'), (jy, 'P2')]
        if self.cavitation:
            exchange_specs.append((self.quad_mgr.nodal_fields['theta'], 'P1'))

        p.decomp.update_ghosts(
            exchange_specs=exchange_specs,
            bc_specs=[
                (rho.pg[0], 'rho', 'P1_nodal'),
                (jx.pg[0],  'jx',  'P2_nodal'),
                (jy.pg[0],  'jy',  'P2_nodal'),
            ],
            problem=p,
        )
        self._fill_p_at_physical_boundaries()
        if self.cavitation:
            self._fill_theta_at_physical_boundaries()

    def _fill_p_at_physical_boundaries(self) -> None:
        """Fill p at physical-boundary ghost strips from the BC-applied rho.

        Only touches ghosts at physical (non-periodic) boundaries; rank-
        boundary ghosts already hold correct p from the MPI exchange.
        Also pushes the updated p to problem.pressure for downstream readers.
        """
        p = self.problem
        decomp = p.decomp
        rho = self.quad_mgr.nodal_fields['rho'].pg[0]
        p_ng = self.quad_mgr.nodal_fields['p'].pg[0]
        if decomp.bc_at_W:
            p_ng[0, :] = eos_pressure(rho[0, :], p.prop)
        if decomp.bc_at_E:
            p_ng[-1, :] = eos_pressure(rho[-1, :], p.prop)
        if decomp.bc_at_S:
            p_ng[:, 0] = eos_pressure(rho[:, 0], p.prop)
        if decomp.bc_at_N:
            p_ng[:, -1] = eos_pressure(rho[:, -1], p.prop)
        self.quad_mgr._push_p_to_pressure_field()

    def _fill_theta_at_physical_boundaries(self) -> None:
        """Set theta=0 at physical-boundary ghost strips (Dirichlet, no user callback)."""
        decomp = self.problem.decomp
        theta = self.quad_mgr.nodal_fields['theta'].pg[0]
        if decomp.bc_at_W:
            theta[0, :] = 0.0
        if decomp.bc_at_E:
            theta[-1, :] = 0.0
        if decomp.bc_at_S:
            theta[:, 0] = 0.0
        if decomp.bc_at_N:
            theta[:, -1] = 0.0

    def _log_bc_pressures(self) -> None:
        """Print pressure at each active Dirichlet boundary ghost strip.

        Called once from pre_run so the user can verify that xW_D/xE_D map to
        the intended pressures via the EoS.  Warns if the pressure at a
        boundary differs from prop['p_cav'] by more than 1 Pa.
        """
        p = self.problem
        decomp = p.decomp
        p_ng = self.quad_mgr.nodal_fields['p'].pg[0]
        grid = p.grid
        rank = decomp._mpi_comm.Get_rank()

        bnd_info = [
            ('W', decomp.bc_at_W, grid.get('bc_xW'), p_ng[0,  :]),
            ('E', decomp.bc_at_E, grid.get('bc_xE'), p_ng[-1, :]),
            ('S', decomp.bc_at_S, grid.get('bc_yS'), p_ng[:,  0]),
            ('N', decomp.bc_at_N, grid.get('bc_yN'), p_ng[:, -1]),
        ]

        for label, owns, bc_types, strip in bnd_info:
            if not owns:
                continue
            if bc_types is None or not any(b == 'D' for b in bc_types):
                continue
            p_mean = float(np.mean(strip))
            p_min  = float(np.min(strip))
            p_max  = float(np.max(strip))
            if rank == 0:
                print(f"  BC pressure {label}: mean={p_mean:.6g} Pa  "
                      f"min={p_min:.6g}  max={p_max:.6g}")

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

    def _sync_rho_before_exchange(self) -> None:
        """Re-derive rho from current p before ghost exchange.

        In the pressure-based solver, rho is not a Newton DOF — it is derived
        from p in update_physics (called after the exchange). If a Neumann rho
        BC is active, _apply_field_bcs sets rho_ghost = rho_interior. When
        set_q_nodal has just written a new p but rho hasn't been updated yet,
        rho_interior is stale (from the previous Newton step), and the Neumann
        BC propagates that stale value into the p ghost strips via
        _fill_p_at_physical_boundaries. Updating rho from the current inner p
        here ensures the BC reads the p-consistent rho.
        Controlled by fem_solver.sync_rho_before_exchange (default True).
        """
        qm = self.quad_mgr
        qm.nodal_fields['rho'].pg[0] = eos_rho(
            qm.nodal_fields['p'].pg[0], self.problem.prop)
        qm.sync_to_problem_q()

    # =========================================================================
    # Solver step
    # =========================================================================

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        self.set_q_nodal(q_guess)
        if self._sync_rho:
            self._sync_rho_before_exchange()
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
        if self._sync_rho:
            self._sync_rho_before_exchange()
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

    def log_jacobian_block_norms(self, M_coo: NDArray = None,
                                  scaled: bool = True) -> None:
        """Print Frobenius norm of each (residual, variable) block of the Jacobian.

        Assembles the Jacobian at the current state if M_coo is not supplied.
        Prints two tables: raw COO norms, then scaled norms (if scaling exists).
        Useful for diagnosing whether characteristic scales in scaling.py are
        appropriate for the current physical regime.

        Parameters
        ----------
        M_coo : array, optional
            COO Jacobian values from get_M(). Assembled fresh if None.
        scaled : bool
            If True and self.scaling exists, also print the scaled table.
        """
        if M_coo is None:
            qf = self._build_all_quad_fields()
            M_coo = self.get_M(qf)

        asm = self.assembly
        res_names = list(asm._res_slices.keys())
        var_names = list(asm._sol_slices.keys())
        res_idx = {r: i for i, r in enumerate(res_names)}
        var_idx = {v: i for i, v in enumerate(var_names)}

        # Map each COO entry to its (res_block, var_block) indices
        r_blk = np.empty(len(M_coo), dtype=np.int32)
        v_blk = np.empty(len(M_coo), dtype=np.int32)
        for (res, var), block in asm.block_order.items():
            s = block['nnz_idx_start']
            n = block['nb_nnz']
            r_blk[s:s + n] = res_idx[res]
            v_blk[s:s + n] = var_idx[var]

        def _print_table(vals, title):
            col_w = 11
            hdr = f"  {'res \\ var':<14s}" + ''.join(f"{v:>{col_w}s}" for v in var_names)
            print(f"\n{title}")
            print(hdr)
            print('  ' + '-' * (len(hdr) - 2))
            for ri, res in enumerate(res_names):
                row = f"  {res:<14s}"
                for vi, var in enumerate(var_names):
                    mask = (r_blk == ri) & (v_blk == vi)
                    norm = np.linalg.norm(vals[mask]) if mask.any() else 0.0
                    row += f"{norm:>{col_w}.2e}"
                print(row)

        _print_table(M_coo, 'Jacobian block norms (unscaled)')

        if scaled and hasattr(self, 'scaling') and self.scaling is not None:
            _print_table(M_coo * self.scaling.display_scale, 'Jacobian block norms (scaled)')
            cs = self.scaling.char_scales
            print(f"\n  Characteristic scales: "
                  + '  '.join(f"{k}={v:.2e}" for k, v in cs.items()))

    def _limit_cavitation_step(self, q: NDArray, dq: NDArray) -> NDArray:
        """Limit the Newton step so no p-DOF changes by more than a relative fraction.

        Computes a single global scalar factor
            f = min(1, min_i( max_rel_dp * max(|p_i|, p_floor) / |dq_p_i| ))
        and returns f * dq.  The floor prevents division-by-zero when p_cav = 0.

        Activated by fem_solver.transition_damping: true.
        Fraction: fem_solver.transition_damping_max_rel_dp  (default 0.05)
        Floor:    fem_solver.transition_damping_p_floor     (default 1e3 Pa)
        """
        p_sl = self._sol_slice('p')
        max_rel_dp = float(self.problem.fem_solver.get('transition_damping_max_rel_dp', 0.5))
        p_floor = float(self.problem.fem_solver.get('transition_damping_p_floor', 1e4))

        p_cur = q[p_sl]
        dp = dq[p_sl]

        allowed = max_rel_dp * np.maximum(np.abs(p_cur), p_floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = np.where(np.abs(dp) > 0.0, allowed / np.abs(dp), 1.0)
        f = float(np.min(factors))
        if f < 1.0:
            if self.problem.decomp.rank == 0:
                worst = int(np.argmin(factors))
                print(f"  [transition_damping] f={f:.4e}  "
                      f"worst node={worst}  |dp|={abs(dp[worst]):.3e}  "
                      f"allowed={allowed[worst]:.3e}")
            dq = dq * f
        return dq

    def _clamp_cavitation(self, q: NDArray) -> NDArray:
        """Clamp p >= p_cav and theta >= 0, with singularity guard at (a,theta)=(0,0)."""
        p_sl = self._sol_slice('p')
        theta_sl = self._sol_slice('theta')
        p_cav = float(self.problem.prop.get('p_cav', 0.0))
        theta_min = self.problem.fem_solver.get('theta_min', np.finfo(float).eps)
        q[p_sl] = np.maximum(q[p_sl], p_cav)
        th = np.maximum(q[theta_sl], 0.0)
        a = q[p_sl] - p_cav
        q[theta_sl] = np.where(
            (np.abs(a) < theta_min) & (th < theta_min),
            theta_min, th)
        return q

    def _print_nodal_diagnostics(self, label: str = '') -> None:
        """Print min/max of inner nodal fields and quad arrays for diagnostics."""
        qm = self.quad_mgr
        prefix = f'  [nodal {label}]' if label else '  [nodal]'
        nodal_names = ['p', 'rho'] + (['theta'] if self.cavitation else [])
        for name in nodal_names:
            inner = qm.nodal_fields[name].p[0]
            print(f'{prefix}  {name}: [{inner.min():.4e}, {inner.max():.4e}]')
        prefix_q = f'  [quad  {label}]' if label else '  [quad]'
        quad_names = ['p', 'rho'] + (['theta'] if self.cavitation else [])
        for name in quad_names:
            if name in qm.quad_fields:
                qm.interpolate_nodal_to_quad(name)
                q_arr = qm.get_quad(name)
                print(f'{prefix_q}  {name}: [{q_arr.min():.4e}, {q_arr.max():.4e}]')

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
            self.R_scaled_norm_history.append([])

        any_guard_fired = False
        _M_prev = None
        for it in range(max_iter):
            M, R = self.solver_step_fun(q)
            R_norm = self.get_R_norm_global(R)

            if rank == 0:
                self.R_norm_history[-1].append(R_norm)
                print(f'{R_norm}')

            if R_norm < tol and it > 0:
                break

            if fem_solver.get('log_jacobian_block_norms', False) and it == 0 and p.step == 0 and rank == 0:
                self.log_jacobian_block_norms(M_coo=M, scaled=fem_solver.get('scaling', False))

            if fem_solver.get('linearization_guard', False):
                if _M_prev is not None:
                    report_jacobian_block_changes(
                        _M_prev, M, self.assembly.block_order,
                        p.decomp._mpi_comm)
                _M_prev = M.copy()

            if fem_solver.get('scaling', False):
                scale_interval = fem_solver.get('scaling_update_interval', 100)
                if it == 0 and (p.step == 0 or p.step % scale_interval == 0):
                    self.scaling = build_scaling_from_blocks(
                        M, self.variables, self.residuals, self.assembly,
                        n_iter=fem_solver.get('scaling_ruiz_iter', 10))
                    if rank == 0:
                        self.log_jacobian_block_norms(M_coo=M, scaled=True)
                M_scaled, R_scaled = self.scaling.scale_system(M, R)
                R_scaled_norm = self.get_R_norm_global(R_scaled)
                if rank == 0:
                    print(f'  R_scaled={R_scaled_norm:.6e}')
                    self.R_scaled_norm_history[-1].append(R_scaled_norm)
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

            if self.cavitation and fem_solver.get('transition_damping', False):
                dq = self._limit_cavitation_step(q, dq)

            q_before = q.copy()
            q = q + alpha * dq

            if fem_solver.get('linearization_guard', False):
                q, fired = linearization_guard_p(q_before, q - q_before, self)
                any_guard_fired |= fired
            else:
                fired = False

            if fem_solver.get('line_search', False):
                if self.cavitation:
                    q = self._clamp_cavitation(q)
                dq_guarded = q - q_before
                # Use scaled norm when available, unscaled otherwise
                use_scaled = fem_solver.get('scaling', False)
                R_ref = R_scaled_norm if use_scaled else R_norm

                def _ls_norm(R_raw):
                    if use_scaled:
                        return self.get_R_norm_global(R_raw / self.scaling.rhs_scale)
                    return self.get_R_norm_global(R_raw)

                R_new = self.get_R(q)
                R_new_norm = _ls_norm(R_new)
                if rank == 0:
                    label = 'R_scaled' if use_scaled else 'R'
                    print(f"  [LineSearch] {label}: {R_ref:.6e} -> {R_new_norm:.6e}"
                          f" ({'OK' if R_new_norm < R_ref else 'INCREASED'})")
                if R_new_norm >= R_ref:
                    ls_alpha = 0.5
                    ls_min = fem_solver.get('line_search_alpha_min', 1e-12)
                    accepted = False
                    while ls_alpha >= ls_min:
                        q_trial = q_before + ls_alpha * dq_guarded
                        if self.cavitation:
                            q_trial = self._clamp_cavitation(q_trial)
                        R_trial = self.get_R(q_trial)
                        R_trial_norm = _ls_norm(R_trial)
                        if R_trial_norm < R_ref:
                            q = q_trial
                            accepted = True
                            if rank == 0:
                                print(f"  [LineSearch] accepted ls_alpha={ls_alpha:.2e},"
                                      f" {label} {R_ref:.4e} -> {R_trial_norm:.4e}")
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

            if self.cavitation:
                q = self._clamp_cavitation(q)

            self.set_q_nodal(q)
            self._exchange_ghosts()

            if fem_solver.get('nodal_diagnostics', False) and rank == 0:
                self._print_nodal_diagnostics(f'it={it}')

        # Final clamp + push: ensures the nodal state is physically valid even
        # when the loop exited early via the convergence break (which fires after
        # solver_step_fun has already pushed an unclamped q into the nodal fields).
        if self.cavitation:
            q = self._clamp_cavitation(q)
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

        # Optionally shift initial pressure uniformly (useful when p_init > p_cav
        # is needed to start in a stable full-film regime before Newton iterates).
        p_init = self.fem_spec.get('p_init', None)
        if p_init is not None:
            p0 = float(p_init)
            rho_init = eos_rho(np.full_like(
                self.quad_mgr.nodal_fields['p'].pg[0], p0), self.problem.prop)
            self.quad_mgr.nodal_fields['p'].pg[0] = p0
            self.quad_mgr.nodal_fields['rho'].pg[0] = rho_init
            self.problem.q[0] = rho_init
            self.quad_mgr.sync_to_problem_q()

        self._exchange_ghosts()
        self.update_quad()
        self._log_bc_pressures()
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
