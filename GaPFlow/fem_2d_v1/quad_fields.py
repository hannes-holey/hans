"""Quadrature field management for Taylor-Hood P2P1 FEM assembly.

Two-grid layout:
  - Pressure / density / topography / viscosity:  coarse grid  (ghost depth 1)
  - Mass flux jx, jy:                             fine grid    (ghost depth 2)

The authoritative fine-grid jx/jy nodal arrays are owned here.  problem.q[1/2]
holds only the even-even (coarse-grid) subset of jx/jy and is used exclusively
for output, saving, and external BC callbacks.  Call sync_to_problem_q() before
any of those operations.

Field shapes
------------
Nodal coarse :  (1, Nx_p_padded, Ny_p_padded)   — muGrid 'pixel' sub-pts
Nodal fine   :  (1, Nx_v_padded, Ny_v_padded)
Quad output  :  (6, Nx_sq, Ny_sq)               — 2 triangles × 3 quad pts
"""
from typing import Dict, List, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from muGrid import Field

from .elements import TaylorHoodP2P1

if TYPE_CHECKING:
    from ..problem import Problem
    from ..parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Field name sets (stabilisation fields removed — Taylor-Hood is inf-sup stable)
# ---------------------------------------------------------------------------

BASE_FIELDS = {
    'rho', 'jx', 'jy',
    'p', 'h', 'dh_dx', 'dh_dy', 'eta',
    'U_bot', 'V_bot', 'U_top', 'V_top', 'Ls',
    'dp_drho',
    'rho_prev', 'jx_prev', 'jy_prev',
    'force_x', 'force_y',
}

STRESS_XZ_FIELDS = {
    'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
    'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx',
}

STRESS_YZ_FIELDS = {
    'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
    'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy',
}

ENERGY_FIELDS = {
    'E', 'Tb_top', 'Tb_bot',
    'T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE',
    'S', 'dS_drho', 'dS_djx', 'dS_djy', 'dS_dE',
    'E_prev',
}

# Fields whose nodal values live on the fine (mass-flux) grid
_FINE_GRID_FIELDS = {'jx', 'jy'}


class QuadFieldManager:
    """Manages nodal and quadrature fields for Taylor-Hood P2P1 assembly.

    Parameters
    ----------
    problem : Problem
        Problem instance (coarse-grid state, physics models).
    energy : bool
        Whether the energy equation is active.
    variables : list of str
        Newton variables in order, e.g. ['jx', 'jy', 'rho'] or with 'E'.
    elements : TaylorHoodP2P1
        Element instance carrying P1 and P2 operators.
    decomp : DomainDecomposition
        Decomposition holding both coarse (_p) and fine (_v) grid info.
    """

    def __init__(self, problem: "Problem", energy: bool,
                 variables: List[str],
                 elements: TaylorHoodP2P1,
                 decomp: "DomainDecomposition") -> None:
        self.problem = problem
        self.energy = energy
        self.variables = variables
        self.elements = elements
        self.decomp = decomp

        self.dx = problem.grid['dx']
        self.dy = problem.grid['dy']

        # Fine-grid ghost depth
        self._ghost_v = 2

        self.nodal_fields: Dict[str, Field] = {}
        self.quad_fields: Dict[str, Field] = {}

        self._init_fields()

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_fields(self) -> None:
        p = self.problem
        fc = p.fc

        # Quad sub-pts: 6 = 2 triangles × 3 quadrature points
        fc.set_nb_sub_pts('quad', 6)

        # --- Coarse-grid nodal fields (pressure/density grid) ---
        coarse_nodal = ['rho', 'h', 'dh_dx', 'dh_dy']
        if self.energy:
            coarse_nodal.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in coarse_nodal:
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # Existing single-component fields (shared with other modules)
        self.nodal_fields['p']   = Field(fc.get_real_field('pressure'))
        self.nodal_fields['eta'] = Field(fc.get_real_field('shear_viscosity'))

        # --- Fine-grid nodal fields (mass-flux grid) ---
        # These need a separate FieldCollection on the fine grid.
        # We use the fine-grid FieldCollection from the decomposition.
        fc_v = self.decomp.fc_v   # muGrid FieldCollection on fine grid
        for name in ('jx', 'jy'):
            self.nodal_fields[name] = fc_v.real_field(f'{name}_nodal', 1, 'pixel')

        # Placeholder for on-the-fly derivative computation (same sub-pts as quad)
        self._deriv_placeholder = fc.real_field('deriv_placeholder_p2p1', 1,
                                                 'quad')

        # --- Quad output fields (all on coarse grid quad sub-pts) ---
        for name in self._needed_fields():
            self.quad_fields[name] = fc.real_field(f'{name}_q_p2p1', 1, 'quad')

    def _needed_fields(self) -> Set[str]:
        needed = BASE_FIELDS | STRESS_XZ_FIELDS | STRESS_YZ_FIELDS
        if self.energy:
            needed |= ENERGY_FIELDS
        return needed

    # =========================================================================
    # Field access  (assembly calls these)
    # =========================================================================

    def get(self, name: str) -> NDArray:
        """Quadrature values for inner squares.

        Returns shape (6, sq_per_row, sq_per_col).
        """
        return self.quad_fields[name].pg[..., :-1, :-1]

    def get_deriv_dx(self, name: str) -> NDArray:
        """d(field)/dx at quad points, shape (2, sq_per_row, sq_per_col)."""
        op = (self.elements.P2.dx_operator if name in _FINE_GRID_FIELDS
              else self.elements.P1.dx_operator)
        op.apply(self.nodal_fields[name], self._deriv_placeholder)
        return self._deriv_placeholder.pg[..., :-1, :-1].copy()

    def get_deriv_dy(self, name: str) -> NDArray:
        """d(field)/dy at quad points, shape (2, sq_per_row, sq_per_col)."""
        op = (self.elements.P2.dy_operator if name in _FINE_GRID_FIELDS
              else self.elements.P1.dy_operator)
        op.apply(self.nodal_fields[name], self._deriv_placeholder)
        return self._deriv_placeholder.pg[..., :-1, :-1].copy()

    def interpolate_nodal_to_quad(self, name: str) -> None:
        """Interpolate a single nodal field to its quad output field."""
        op = (self.elements.P2.interpolation_operator
              if name in _FINE_GRID_FIELDS
              else self.elements.P1.interpolation_operator)
        op.apply(self.nodal_fields[name], self.quad_fields[name])

    # =========================================================================
    # Newton scatter / gather  (solver calls these)
    # =========================================================================

    def get_nodal_val(self, var: str) -> NDArray:
        """Inner nodal values for Newton gather, shape (nb_inner,) F-order.

        For mass-flux variables the inner region is [2:-2, 2:-2] (ghost depth 2).
        For density the inner region is [1:-1, 1:-1] (ghost depth 1).
        """
        if var in _FINE_GRID_FIELDS:
            return self.nodal_fields[var].pg[0, 2:-2, 2:-2].flatten(order='F')
        else:
            return self.nodal_fields[var].pg[0, 1:-1, 1:-1].flatten(order='F')

    def set_nodal_val(self, var: str, values: NDArray,
                      nx_inner: int, ny_inner: int) -> None:
        """Scatter Newton solution vector slice back to nodal field inner region.

        Parameters
        ----------
        var : str
            Variable name.
        values : NDArray
            Flat (F-order) inner nodal values, length nx_inner * ny_inner.
        nx_inner, ny_inner : int
            Inner grid dimensions for reshaping.
        """
        arr = values.reshape((nx_inner, ny_inner), order='F')
        if var in _FINE_GRID_FIELDS:
            self.nodal_fields[var].pg[0, 2:-2, 2:-2] = arr
        else:
            self.nodal_fields[var].pg[0, 1:-1, 1:-1] = arr

    # =========================================================================
    # Sync gate: fine-grid jx/jy → problem.q  (for output / BCs)
    # =========================================================================

    def sync_to_problem_q(self) -> None:
        """Copy even-even fine-grid values of jx/jy to problem.q[1/2].

        Even-even positions on the fine grid coincide with coarse-grid nodes.
        Fine inner region starts at index 2 (ghost depth 2); coarse at index 1.
        So fine[2::2, 2::2] maps to coarse[1:, 1:] inner + ghost boundary.

        Call before: saving results, plotting, BC callbacks, output fields.
        Do NOT call inside the Newton inner loop.
        """
        p = self.problem
        jx_fine = self.nodal_fields['jx'].pg[0]   # (Nx_v_padded, Ny_v_padded)
        jy_fine = self.nodal_fields['jy'].pg[0]

        # Even-even fine nodes align with coarse padded nodes starting at index 0
        # fine index 0 = coarse index 0 (SW ghost corner)
        p.q[1] = jx_fine[::2, ::2]
        p.q[2] = jy_fine[::2, ::2]

    def sync_from_problem_q(self) -> None:
        """Copy problem.q initial state to Newton-owned nodal fields.

        Used at initialisation and after external state changes (e.g. restart).
        Initialises rho (and E if active) from problem.q, and the even-even
        subset of jx/jy from problem.q[1/2].  Fine-grid mid-points and edge
        nodes of jx/jy are left at zero on first call.
        """
        p = self.problem
        self.nodal_fields['rho'].pg[0] = p.q[0]
        self.nodal_fields['jx'].pg[0, ::2, ::2] = p.q[1]
        self.nodal_fields['jy'].pg[0, ::2, ::2] = p.q[2]
        if self.energy:
            self.nodal_fields['E'].pg[0] = p.energy.energy

    # =========================================================================
    # Field updates  (called once per Newton step)
    # =========================================================================

    def update_nodal_fields(self) -> None:
        """Update coarse-grid nodal fields from problem state.

        Mirrors old QuadFieldManager.update_nodal_fields().
        Does NOT touch jx/jy — those are owned here and updated by the solver.
        """
        p = self.problem
        p.pressure.update()
        p.topo.update()

        dp_dx = np.gradient(p.pressure.pressure, self.dx, axis=0)
        dp_dy = np.gradient(p.pressure.pressure, self.dy, axis=1)
        p.viscosity.update(p.pressure.pressure, dp_dx, dp_dy,
                           p.topo.h,
                           p.geo['U_bot'], p.geo['V_bot'],
                           p.geo['U_top'], p.geo['V_top'])
        if self.energy:
            p.energy.update_temperature()

    def update_quad_nodal(self) -> None:
        """Copy coarse-grid problem state to nodal fields and interpolate all to quad.

        jx/jy are already up-to-date in their fine-grid nodal fields
        (written by the Newton solver via set_nodal_val), so they are
        interpolated directly without a copy step.
        """
        p = self.problem

        # Coarse-grid fields from problem state (rho/E are owned by Newton solver,
        # written by set_nodal_val — do NOT overwrite them here)
        self.nodal_fields['h'].pg[0]      = p.topo.h
        self.nodal_fields['dh_dx'].pg[0]  = p.topo.dh_dx
        self.nodal_fields['dh_dy'].pg[0]  = p.topo.dh_dy
        if self.energy:
            self.nodal_fields['Tb_top'].pg[0]  = p.energy.Tb_top
            self.nodal_fields['Tb_bot'].pg[0]  = p.energy.Tb_bot

        # Interpolate all nodal → quad (P2 for jx/jy, P1 for rest)
        coarse_interp = ['rho', 'p', 'h', 'dh_dx', 'dh_dy', 'eta']
        if self.energy:
            coarse_interp.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in coarse_interp:
            self.interpolate_nodal_to_quad(name)
        for name in ('jx', 'jy'):
            self.interpolate_nodal_to_quad(name)

        # Broadcast scalar constants
        p_obj = p
        self.quad_fields['U_bot'].pg[:] = p_obj.geo['U_bot']
        self.quad_fields['V_bot'].pg[:] = p_obj.geo['V_bot']
        self.quad_fields['U_top'].pg[:] = p_obj.geo['U_top']
        self.quad_fields['V_top'].pg[:] = p_obj.geo['V_top']
        self.quad_fields['Ls'].pg[:]    = p_obj.prop.get('slip_length', 0.0)

    def _apply_2d_vmap(self, func, *args):
        shape = args[0].shape
        args_2d = [a.reshape(shape[0], -1) for a in args]
        return func(*args_2d).reshape(shape)

    def update_quad_computed(self) -> None:
        """Compute derived quantities at quad points (wall stress, dp_drho, etc.)."""
        p = self.problem
        s = np.s_[..., :-1, :-1]
        q = lambda name: self.quad_fields[name].pg[s]
        apply = self._apply_2d_vmap

        self.quad_fields['dp_drho'].pg[s] = apply(p.pressure.dp_drho, q('rho'))

        args_xz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dx'),
                   q('U_bot'), q('V_bot'), q('U_top'), q('V_top'), q('Ls'))
        for name in ['tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
                     'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx']:
            self.quad_fields[name].pg[s] = apply(
                getattr(p.wall_stress_xz, name), *args_xz)

        args_yz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dy'),
                   q('U_bot'), q('V_bot'), q('U_top'), q('V_top'), q('Ls'))
        for name in ['tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
                     'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy']:
            self.quad_fields[name].pg[s] = apply(
                getattr(p.wall_stress_yz, name), *args_yz)

        self.quad_fields['force_x'].pg[s] = np.full_like(
            q('rho'), p.prop.get('force_x', 0.0))
        self.quad_fields['force_y'].pg[s] = np.full_like(
            q('rho'), p.prop.get('force_y', 0.0))

        if self.energy:
            args_T = (q('rho'), q('jx'), q('jy'), q('E'))
            for name, func in [('T', 'T_func'), ('dT_drho', 'T_grad_rho'),
                                ('dT_djx', 'T_grad_jx'), ('dT_djy', 'T_grad_jy'),
                                ('dT_dE', 'T_grad_E')]:
                self.quad_fields[name].pg[s] = getattr(p.energy, func)(*args_T)

            args_S = (q('h'), q('eta'), q('rho'), q('E'), q('jx'), q('jy'),
                      q('U_bot'), q('V_bot'), q('Tb_top'), q('Tb_bot'))
            for name, func in [('S', 'q_wall_sum'), ('dS_drho', 'q_wall_grad_rho'),
                                ('dS_djx', 'q_wall_grad_jx'),
                                ('dS_djy', 'q_wall_grad_jy'),
                                ('dS_dE', 'q_wall_grad_E')]:
                self.quad_fields[name].pg[s] = apply(
                    getattr(p.energy, func), *args_S)

    def store_prev_values(self) -> None:
        """Store current quad values for time derivatives."""
        for var in self.variables:
            prev_key = f'{var}_prev'
            if var in self.quad_fields and prev_key in self.quad_fields:
                self.quad_fields[prev_key].pg[:] = self.quad_fields[var].pg.copy()
