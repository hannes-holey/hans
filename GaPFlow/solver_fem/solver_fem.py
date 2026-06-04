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
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .quad_fields import QuadFieldManager
from .assembly import Assembly
from .fieldspec import FieldSpec, VAR_GRID, RES_GRID
from .scipy_system import ScipySystem

from ..bc import GhostUpdater, BoundarySpec, sample_bc_spec, translate_bc_rho_to_p
from .solution_guards import solve_linear_system, line_search
from .bayada_stabilization import bayada_linearization_guard
from .terms import get_active_terms
from .scaling import build_scaling

if TYPE_CHECKING:
    from ..problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver:
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
        self.rank = problem.decomp.rank

        self._build_variable_and_residual_lists()

        self.elements = TaylorHoodP2P1(problem.grid['dx'], problem.grid['dy'])
        self.grid_idx = GridIndexManager(problem.decomp, self)

        self._get_active_terms()

    def _build_variable_and_residual_lists(self) -> None:
        """Check active equations and build lists."""

        p = self.problem
        self.energy = p.fem_solver['equations']['energy']
        self.cavitation = p.fem_solver['equations']['cavitation']
        self.oss = p.fem_solver['physics']['oss_theta']

        self.variables = ['jx', 'jy', 'p']
        self.residuals = ['momentum_x', 'momentum_y', 'mass']
        self.add_fields = []

        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
            self.add_fields.append('e')
        if self.cavitation:
            self.variables.append('theta')
            self.residuals.append('fb')
            self.add_fields.append('theta')
        if self.cavitation and self.oss:
            self.variables.append('xi')
            self.residuals.append('R_oss')
            self.add_fields.append('xi')

        self.var_specs = [FieldSpec(name=v, grid=VAR_GRID[v], idx=i)
                          for i, v in enumerate(self.variables)]
        self.res_specs = [FieldSpec(name=r, grid=RES_GRID[r], idx=i)
                          for i, r in enumerate(self.residuals)]

        nb_inner = {
            'P1': np.prod(p.decomp.nb_subdomain_grid_pts),
            'P2': np.prod(p.decomp.nb_subdomain_grid_pts_P2),
        }
        self.res_size = sum(nb_inner[spec.grid] for spec in self.res_specs)

    # =========================================================================
    # Boundary conditions
    # =========================================================================

    def build_boundary_conditions(self):
        """Build the list of BoundarySpec objects for BC application.
        Note: bc_spec needs to be sampled in the same order as variables."""

        self._init_quad_fields()

        specs = []
        no_fun = [None]*4

        field = self.quad_mgr.nodal_fields['jx']
        jx_bc_type, jx_bc_vals = sample_bc_spec(self.problem.grid, 1)
        specs.append(BoundarySpec(field, 'P2', jx_bc_type,
                                  jx_bc_vals, no_fun, self.problem.decomp))
        
        field = self.quad_mgr.nodal_fields['jy']
        jy_bc_type, jy_bc_vals = sample_bc_spec(self.problem.grid, 2)
        specs.append(BoundarySpec(field, 'P2', jy_bc_type,
                                  jy_bc_vals, no_fun, self.problem.decomp))

        field = self.quad_mgr.nodal_fields['p']
        rho_bc_type, rho_bc_vals = sample_bc_spec(self.problem.grid, 0)
        p_bc_vals = translate_bc_rho_to_p(rho_bc_type, rho_bc_vals, self.problem)
        specs.append(BoundarySpec(field, 'P1', rho_bc_type,
                                  p_bc_vals, no_fun, self.problem.decomp))

        if self.cavitation:
            field = self.quad_mgr.nodal_fields['theta']
            specs.append(BoundarySpec(field, 'P1', rho_bc_type,
                                      [0, 0, 0, 0], no_fun, self.problem.decomp))

        if self.cavitation and self.oss:
            field = self.quad_mgr.nodal_fields['xi']
            specs.append(BoundarySpec(field, 'P1', ['D', 'D', 'D', 'D'],
                                      [0, 0, 0, 0], no_fun, self.problem.decomp))

        if self.energy:
            pass

        self.ghost_updater = GhostUpdater(
            decomp=self.problem.decomp,
            problem=self.problem,
            specs=specs
        )

        self.bc_specs = specs

    # =========================================================================
    # Initialisation
    # =========================================================================

    @property
    def _res_slices(self):
        return self.assembly._res_slices

    @property
    def _sol_slices(self):
        return self.assembly._sol_slices

    @property
    def nb_sol(self):
        return 3 + len(self.add_fields)

    def _get_active_terms(self) -> None:
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_fields(self) -> None:
        self.quad_mgr = QuadFieldManager(
            problem=self.problem,
            energy=self.energy,
            cavitation=self.cavitation,
            variables=self.variables,
            add_fields=self.add_fields,
            elements=self.elements,
            decomp=self.problem.decomp,
            terms=self.terms,
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
            return lambda: self.quad_mgr.get_quad_sq(name)

        _, quad_fields, _ = self.quad_mgr._needed_fields()
        for term in self.terms:

            # fields
            ctx = {name: make_getter(name) for name in quad_fields}

            # scalars
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
            from .. import HAS_PETSC
        except ImportError:
            HAS_PETSC = False

        if HAS_PETSC:
            from .petsc_system import PETScSystem
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
            print(f"[FEMSolver] Using reference pressure p_ref = "
                  f"{p_ref:.3e} Pa for scaling")

        debug_from = self.problem.fem_solver['newton_debug']
        if debug_from is not None:
            from .newton_debug import NewtonDebugger
            self.debugger = NewtonDebugger(
                output_dir=self.problem.options['output'],
                variables=self.variables,
                residuals=self.residuals,
                res_slices=self.assembly._res_slices,
                sol_slices=self.assembly._sol_slices,
                Nx_p=self.grid_idx.Nx_P1_inner,
                Ny_p=self.grid_idx.Ny_P1_inner,
                Nx_P2=self.grid_idx.Nx_P2_inner,
                Ny_P2=self.grid_idx.Ny_P2_inner,
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

    def update_quad(self) -> dict:
        self.quad_mgr.update_physics()
        self.quad_mgr.update_quad_fields()
        return self.collect_all_quad_fields()

    def update_prev_quad(self) -> None:
        self.quad_mgr.store_prev_values()

    def collect_all_quad_fields(self) -> dict:
        """Collect all needed quadrature fields once.
        Derivative fields are built on the fly here."""
        qf: dict = {}
        need_dx: set = set()
        need_dy: set = set()

        for term in self.terms:
            for v in term.dep_vars:
                if v not in qf:
                    qf[v] = self.quad_mgr.get_quad_sq(v)
                d = term.depvar_deriv_for(v)
                if d == 'x':
                    need_dx.add(v)
                elif d == 'y':
                    need_dy.add(v)

        for v in need_dx:
            key = f'd_dx_{v}'
            if key not in qf:
                qf[key] = self.quad_mgr.get_quad_dx_sq(v)
        for v in need_dy:
            key = f'd_dy_{v}'
            if key not in qf:
                qf[key] = self.quad_mgr.get_quad_dy_sq(v)

        return qf

    # =========================================================================
    # Newton scatter / gather
    # =========================================================================

    def get_q_nodal(self) -> NDArray:
        """Gather inner nodal values into a flat solution vector."""
        q = np.zeros(self.res_size)
        for var in self.variables:
            q[self._sol_slices[var]] = self.quad_mgr.get_nodal_sol_val(var)
        return q

    def update_q_nodal(self, q: NDArray) -> None:
        """Scatter flat solution vector back to nodal fields and update ghosts."""
        for var in self.variables:
            self.quad_mgr.set_nodal_sol_val(var, q[self._sol_slices[var]])
        self.ghost_updater.update()

    # =========================================================================
    # Assembly
    # =========================================================================

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

    # =========================================================================
    # Solver step
    # =========================================================================

    def update_quad_and_assemble(self) -> Tuple[NDArray, NDArray]:
        """Update models, quadrature fields, and assemble M and R."""

        qf = self.update_quad()

        M = self.get_M(qf)
        R = self.get_R_(qf).copy()

        if self._debug_active:
            self._last_R_per_term = self.assembly.assemble_rhs_per_term(qf, self.terms)
        return M, R

    def get_R(self, q_guess: NDArray) -> float:
        self.update_q_nodal(q_guess)
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
        p_sl = self._sol_slices['p']
        theta_sl = self._sol_slices['theta']
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

    def check_residual(self, R: NDArray, it: int) -> float:
        """Check residual norm and return True if converged."""

        R_norm = self.get_R_norm_global(R)
        if self.rank == 0:
            self.R_norm_history[-1].append(R_norm)
            print(f'{R_norm}')
        if R_norm < self.tol and it > 0:
            return True
        return False

    def post_solve(self, q: NDArray, dq: NDArray, R: NDArray, it: int, M_scaled: NDArray) -> None:
        """Post-process obtained solution update: debug output, line search, and cavitation clamping."""
        
        if self._debug_active:
            self.debugger.step(
                timestep=self.problem.step, it=it, R=R, dq=dq, q=q,
                R_per_term=self._last_R_per_term, M_scaled=M_scaled)
            self._debug_steps_done += 1

        if self.problem.fem_solver['line_search']:
            q = line_search(q, self.alpha * dq, self.R_norm, self)
        else:
            if self.problem.prop.get('EOS') == 'Bayada':
                q, _ = bayada_linearization_guard(q, self.alpha * dq, self)
            else:
                q = q + self.alpha * dq

        if self.cavitation:
            q = self._clamp_cavitation(q)
        
        return q

    def wrap_up_timestep(self, it: int, tic: float) -> None:
        """Wrap up the finished timstep."""

        toc = time.time()
        self.time_inner = toc - tic
        self.inner_iterations = it + 1

        if self.debugger is not None and self._debug_steps_done >= 5:
            self.problem._stop = True
        
        self.print_status()

    def update(self) -> None:
        """Perform one time step with Newton iteration."""
        
        p = self.problem
        tic = time.time()
        max_iter = 1 if self._debug_active else self.max_iter

        self.update_prev_quad()
        q = self.get_q_nodal().copy()

        if self.rank == 0:
            self.R_norm_history.append([])

        # Inner Newton loop
        for it in range(max_iter):

            M, R = self.update_quad_and_assemble()
            if self.check_residual(R, it): break
            dq, M_scaled = solve_linear_system(M, R, self, it)
            q = self.post_solve(q, dq, R, it, M_scaled)
            self.update_q_nodal(q)

        self.wrap_up_timestep(it, tic)
        self.update_output_fields()
        p._post_update()

    # =========================================================================
    # Pre-run setup
    # =========================================================================

    def pre_run(self, **kwargs) -> None:
        """Initialize assembly, jit functions, and solver. Load initial state and
        update quadrature fields. Fetch solver parameters."""

        self._build_assembly()
        self._build_jit_functions()
        self._build_terms()
        self.assembly.build_assembly_templates(self.terms)
        self._init_linear_solver()

        self.quad_mgr.sync_from_problem_q()

        self.update_quad()
        self.update_prev_quad()
        self.update_output_fields()

        self.time_inner = 0.0
        self.inner_iterations = 0

        p = self.problem
        self.tol = p.fem_solver['R_norm_tol']
        self.alpha = p.fem_solver['newton_relax']
        self.max_iter = p.fem_solver['max_iter']

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
