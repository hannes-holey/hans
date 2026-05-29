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
from .fem_2d.assembly import Assembly, FieldSpec
from .fem_2d.terms import NonLinearTerm, get_active_terms
from .fem_2d.scaling import build_scaling
from .fem_2d.scipy_system import ScipySystem
from .fem_2d.solution_guards import (linearization_guard, linearization_guard_p,  # noqa: F401
                                      report_jacobian_block_changes,
                                      solve_linear_system, line_search)
from .models.pressure import eos_pressure, eos_rho

if TYPE_CHECKING:
    from .problem import Problem
    from .parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]

VAR_GRID = {'jx': 'v', 'jy': 'v', 'p': 'p', 'E': 'p', 'theta': 'p', 'xi': 'p'}
RES_GRID = {
    'momentum_x': 'v',
    'momentum_y': 'v',
    'mass':       'p',
    'energy':     'p',
    'fb':         'p',
    'R_oss':      'p',
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
        self.energy = p.fem_solver['equations']['energy']
        self.cavitation = p.fem_solver['equations']['cavitation']

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
        if self.cavitation and p.fem_solver['physics']['oss_theta']:
            self.variables.append('xi')
            self.residuals.append('R_oss')

        self._sync_rho = p.fem_solver.get('sync_rho_before_exchange', True)

        self.var_specs = [FieldSpec(name=v, grid=VAR_GRID[v], idx=i)
                          for i, v in enumerate(self.variables)]
        self.res_specs = [FieldSpec(name=r, grid=RES_GRID[r], idx=i)
                          for i, r in enumerate(self.residuals)]
        self.var_to_grid = {s.name: s.grid for s in self.var_specs}
        self.res_to_grid = {s.name: s.grid for s in self.res_specs}

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
            var_specs=self.var_specs,
            res_specs=self.res_specs,
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
            ctx['theta_stab_alpha'] = lambda: p.fem_solver['theta_stab_alpha']
            ctx['p_cav'] = lambda: p.prop['p_cav']
            ctx['fb_p_ref'] = lambda: p.prop['P0']
            ctx['dx'] = lambda: p.grid['dx']
            ctx['dy'] = lambda: p.grid['dy']
            ctx['oss_correction_alpha'] = lambda: p.fem_solver['oss_correction_alpha']
            if self.energy:
                ctx['k'] = lambda: p.energy.k
            term.build(ctx)

    def _init_linear_solver(self) -> None:
        petsc_info = self.assembly.get_petsc_info(res_size=self.res_size)
        solver_type = self.problem.fem_solver['linear_solver']

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

        debug_from = self.problem.fem_solver['newton_debug']
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
                quad_mgr=self.quad_mgr,
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
        if 'xi' in self.variables:
            exchange_specs.append((self.quad_mgr.nodal_fields['xi'], 'P1'))

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
        if 'xi' in self.variables:
            self._fill_xi_at_physical_boundaries()

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

    def _fill_xi_at_physical_boundaries(self) -> None:
        """Set xi=0 at physical-boundary ghost strips (no mathematical BC needed)."""
        decomp = self.problem.decomp
        xi = self.quad_mgr.nodal_fields['xi'].pg[0]
        if decomp.bc_at_W:
            xi[0, :] = 0.0
        if decomp.bc_at_E:
            xi[-1, :] = 0.0
        if decomp.bc_at_S:
            xi[:, 0] = 0.0
        if decomp.bc_at_N:
            xi[:, -1] = 0.0

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
        ('d_dx_<var>', 'd_dy_<var>') for any variable that needs a spatial
        derivative (trial_deriv set on that variable's slot).
        """
        qf: dict = {}
        need_dx: set = set()
        need_dy: set = set()

        for term in self.terms:
            for v in term.dep_vars:
                if v not in qf:
                    qf[v] = self.quad_mgr.get_quad(v)
                d = term.depvar_deriv_for(v)
                if d == 'x':
                    need_dx.add(v)
                elif d == 'y':
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

    def _clamp_cavitation(self, q: NDArray) -> NDArray:
        """Clamp p >= p_cav and theta >= 0, with singularity guard at (a,theta)=(0,0)."""
        p_sl = self._sol_slice('p')
        theta_sl = self._sol_slice('theta')
        p_cav = float(self.problem.prop['p_cav'])
        theta_min = self.problem.fem_solver['theta_min']
        q[p_sl] = np.maximum(q[p_sl], p_cav)
        th = np.maximum(q[theta_sl], 0.0)
        a = q[p_sl] - p_cav
        q[theta_sl] = np.where(
            (np.abs(a) < theta_min) & (th < theta_min),
            theta_min, th)
        return q

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
    # Time step
    # =========================================================================

    def update_dynamic(self) -> None:
        
        tic = time.time()
        
        p = self.problem
        fem_solver = p.fem_solver
        self.update_prev_quad()
        q = self.get_q_nodal().copy()

        max_iter = 1 if self._debug_active else fem_solver['max_iter']
        tol = fem_solver['R_norm_tol']
        alpha = fem_solver['newton_relax']
        dt_init = p.numerics['dt']
        rank = p.decomp.rank

        if rank == 0:
            self.R_norm_history.append([])
            self.R_scaled_norm_history.append([])

        n_iter = 0
        for it in range(max_iter):
            self._current_it = it
            M, R = self.solver_step_fun(q)
            R_norm = self.get_R_norm_global(R)

            if rank == 0:
                self.R_norm_history[-1].append(R_norm)
                print(f'{R_norm}')

            if R_norm < tol and it > 0:
                break

            dq, M_scaled = solve_linear_system(M, R, self)

            if self._debug_active:
                self.debugger.step(
                    timestep=p.step, it=it, R=R, dq=dq, q=q,
                    R_per_term=self._last_R_per_term, M_scaled=M_scaled)
                self._debug_steps_done += 1

            if fem_solver['line_search']:
                q = line_search(q, alpha * dq, R_norm, self)
            else:
                q = q + alpha * dq
                if self.cavitation:
                    q = self._clamp_cavitation(q)

            self.set_q_nodal(q)
            self._exchange_ghosts()
            n_iter += 1

        if self.cavitation:
            q = self._clamp_cavitation(q)
            self.set_q_nodal(q)
            self._exchange_ghosts()

        p.numerics['dt'] = dt_init

        toc = time.time()
        self.time_inner = toc - tic
        self.inner_iterations = n_iter

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
