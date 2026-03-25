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
Newton iteration loop.  Replaces solver_fem_2d.FEMSolver2D for the P2P1
discretisation.

Key differences from the old P1P1 solver
-----------------------------------------
* Two inner sizes: nb_inner_v (jx, jy — P2 fine grid) and nb_inner_p (rho, [E]
  — P1 coarse grid).  Residual / solution vectors have variable-length blocks.
* Newton scatter/gather goes through QuadFieldManager, which owns the fine-grid
  jx/jy nodal fields.
* Ghost exchange after each Newton step calls update_ghosts with P1/P2
  exchange_specs and bc_specs for rho (P1_nodal), jx, jy (P2_nodal).
* sync_to_problem_q() is called once per time step (not inside Newton loop)
  before output / callbacks.
* No PSPG/GLS stabilization terms — Taylor-Hood is inf-sup stable.
"""

import time
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .quad_fields import QuadFieldManager
from .assembly import Assembly
from .terms import NonLinearTerm, get_active_terms
from .scaling import build_scaling
from .scipy_system import ScipySystem

if TYPE_CHECKING:
    from ..problem import Problem
    from ..parallel import DomainDecomposition

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

        # Precompute per-residual / per-variable slices in the flat vectors
        self._build_slices()

    def _build_slices(self) -> None:
        """Precompute contiguous slices for residual and solution vectors."""
        # Residual slices
        self._res_slices: Dict[str, slice] = {}
        offset = 0
        for res in self.residuals:
            n = (self.nb_inner_v if self.res_to_grid[res] == 'v'
                 else self.nb_inner_p)
            self._res_slices[res] = slice(offset, offset + n)
            offset += n

        # Solution slices (same ordering as residuals, by variable)
        self._sol_slices: Dict[str, slice] = {}
        offset = 0
        for var in self.variables:
            n = (self.nb_inner_v if self.var_to_grid[var] == 'v'
                 else self.nb_inner_p)
            self._sol_slices[var] = slice(offset, offset + n)
            offset += n

    def _res_slice(self, res_name: str) -> slice:
        return self._res_slices[res_name]

    def _sol_slice(self, var_name: str) -> slice:
        return self._sol_slices[var_name]

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
        # cols_v = number of mass-flux inner columns (x-direction, F-order)
        cols_v = self.grid_idx.Nx_v_inner
        M = self.grid_idx.Nx_p_inner  # pressure inner columns (x-direction)

        self.assembly = Assembly(
            grid_idx=self.grid_idx,
            variables=self.variables,
            residuals=self.residuals,
            res_to_grid=self.res_to_grid,
            var_to_grid=self.var_to_grid,
            cols_v=cols_v,
            M=M,
            energy=self.energy,
        )
        self.assembly.build_assembly_templates(
            grid_idx=self.grid_idx,
            variables=self.variables,
            residuals=self.residuals,
            res_to_grid=self.res_to_grid,
            var_to_grid=self.var_to_grid,
            elements=self.elements,
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
            return lambda: self.quad_mgr.get(name)

        all_field_names = self.quad_mgr._needed_fields()
        for term in self.terms:
            ctx = {name: make_getter(name) for name in all_field_names}
            ctx['dt'] = p.numerics['dt']
            if self.energy:
                ctx['k'] = lambda: p.energy.k
            term.build(ctx)

    def _init_linear_solver(self) -> None:
        petsc_info = self.assembly.get_petsc_info(res_size=self.res_size)
        solver_type = self.problem.fem_solver.get('linear_solver', 'direct')

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
            self.problem, self.energy, self.variables, self.assembly)

    # =========================================================================
    # Quadrature field update
    # =========================================================================

    def update_quad(self) -> None:
        self.quad_mgr.update_nodal_fields()
        self.quad_mgr.update_quad_nodal()
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
            q[self._sol_slice(var)] = self.quad_mgr.get_nodal_val(var)
        return q

    def set_q_nodal(self, q: NDArray) -> None:
        """Scatter flat solution vector back to nodal fields."""
        for var in self.variables:
            nx, ny = self._inner_shape(var)
            self.quad_mgr.set_nodal_val(var, q[self._sol_slice(var)], nx, ny)

    def _inner_shape(self, var: str) -> Tuple[int, int]:
        if self.var_to_grid[var] == 'v':
            return self.grid_idx.Nx_v_inner, self.grid_idx.Ny_v_inner
        return self.grid_idx.Nx_p_inner, self.grid_idx.Ny_p_inner

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
                    qf[v] = self.quad_mgr.get(v)
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

    def get_M(self) -> NDArray:
        """Assemble Jacobian COO values."""
        coo = np.zeros(self.assembly.nnz, dtype=np.float64)
        qf = self._build_all_quad_fields()
        self.assembly.assemble_matrix(
            nnz_values=coo,
            quad_fields=qf,
            terms=self.terms,
        )
        return coo

    def get_R(self) -> NDArray:
        """Assemble residual vector."""
        R = np.zeros(self.res_size, dtype=np.float64)
        qf = self._build_all_quad_fields()
        self.assembly.assemble_rhs(
            rhs_values=R,
            quad_fields=qf,
            terms=self.terms,
        )
        return R

    # =========================================================================
    # Solver step
    # =========================================================================

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        self.set_q_nodal(q_guess)
        self.update_quad()
        M = self.get_M()
        R = self.get_R()
        return M, R

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
        p = self.problem
        fem_solver = p.fem_solver

        self.update_prev_quad()

        tic = time.time()
        q = self.get_q_nodal().copy()

        max_iter  = fem_solver['max_iter']
        tol       = fem_solver['R_norm_tol']
        alpha     = fem_solver.get('newton_relax', 1.0)

        if p.decomp.rank == 0:
            self.R_norm_history.append([])

        for it in range(max_iter):
            M, R = self.solver_step_fun(q)

            R_norm_local_sq = float(np.linalg.norm(R) ** 2)
            R_norm_global_sq = p.decomp._mpi_comm.allreduce(
                R_norm_local_sq, op=MPI.SUM)
            R_norm = float(np.sqrt(R_norm_global_sq))

            if p.decomp.rank == 0:
                self.R_norm_history[-1].append(R_norm)

            if R_norm < tol and it > 0:
                break

            M_scaled, R_scaled = self.scaling.scale_system(M, R)

            self.linear_solver.assemble(M_scaled, R_scaled)
            dq_scaled = self.linear_solver.solve()
            dq = self.scaling.unscale_solution(dq_scaled)

            q = q + alpha * dq

            self.set_q_nodal(q)
            self._exchange_ghosts()

        toc = time.time()
        self.time_inner = toc - tic
        self.inner_iterations = it + 1

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
