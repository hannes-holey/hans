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

from typing import Dict, List, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from muGrid import Field
from scipy.ndimage import zoom

from .elements import TaylorHoodP2P1
from ..models.pressure import eos_pressure, eos_rho, eos_drho_dp

if TYPE_CHECKING:
    from ..problem import Problem
    from ..parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Field name sets
# ---------------------------------------------------------------------------

BASE_FIELDS = {
    'rho', 'jx', 'jy',
    'p', 'h', 'dh_dx', 'dh_dy', 'eta',
    'U_bot', 'V_bot', 'U_top', 'V_top', 'Ls',
    'dp_drho', 'drho_dp', 'd2p_drho2',
    'd_dx_jx', 'd_dy_jy',
    'p_prev', 'jx_prev', 'jy_prev',
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

    def __init__(self, problem: "Problem",
                 energy: bool,
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
        """Register all nodal and quadrature fields in the muGrid FieldCollections.

        set_nb_sub_pts('quad', nb_quad_sq) must be called before any 'quad' field is
        registered; it has no effect on already-registered fields or other tags.
        'pixel' is a muGrid built-in (always 1 sub-point per cell, pre-registered).

        - create new nodal fields for coarse-grid variables rho and h (to have individual fields)
        - get reference to p, eta fields from problem
        - create nodal fields for fine-grid variables jx, jy
        - create placeholder field for on-the-fly derivative computation
        - create quadrature fields for all needed variables
        """
        fc = self.decomp.fc

        nb_quad_sq = self.elements.n_tri * self.elements.Quadrature.nb_points
        fc.set_nb_sub_pts('quad', nb_quad_sq)

        # Create new single-component nodal fields (including 'p' — the P1
        # DOF for the pressure-based solver). We intentionally do NOT wrap
        # the pre-existing 'pressure' muGrid field here because it has a
        # different `.pg` shape (2D, no component axis); a fresh field keeps
        # all nodal_fields uniform as (1, Nx_pad, Ny_pad). The 'pressure'
        # field is kept in sync via `_push_p_to_pressure` on each update.
        for name in ['rho', 'p', 'h', 'dh_dx', 'dh_dy']:
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # Reference existing fields from fc
        self.nodal_fields['eta'] = Field(fc.get_real_field('shear_viscosity'))
        if self.energy:
            self.nodal_fields['E']      = Field(fc.get_real_field('total_energy'))
            self.nodal_fields['Tb_top'] = Field(fc.get_real_field('Tb_top'))
            self.nodal_fields['Tb_bot'] = Field(fc.get_real_field('Tb_bot'))

        # Create fine-grid nodal fields
        fc_v = self.decomp.fc_v
        for name in ('jx', 'jy'):
            self.nodal_fields[name] = fc_v.real_field(f'{name}_nodal', 1, 'pixel')

        # Placeholder for on-the-fly derivative computation
        self._deriv_placeholder = fc.real_field('deriv_placeholder_p2p1', 1, 'quad')

        # Create all quadrature fields
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

    def get_quad(self, name: str) -> NDArray:
        """Quadrature values for all squares.
        Returns shape (nb_sq, nb_quad_sq): squares x-major, quad innermost.
        """
        sq = self.quad_fields[name].pg[:, :-1, :-1]  # (nb_quad_sq, sq_per_row, sq_per_col)
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def get_deriv_dx(self, name: str) -> NDArray:
        """d(field)/dx at quad points, shape (nb_sq, nb_quad_sq)."""
        op = (self.elements.P2.dx_operator if name in _FINE_GRID_FIELDS
              else self.elements.P1.dx_operator)
        op.apply(self.nodal_fields[name], self._deriv_placeholder)
        sq = self._deriv_placeholder.pg[:, :-1, :-1]
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0]) / self.dx

    def get_deriv_dy(self, name: str) -> NDArray:
        """d(field)/dy at quad points, shape (nb_sq, nb_quad_sq)."""
        op = (self.elements.P2.dy_operator if name in _FINE_GRID_FIELDS
              else self.elements.P1.dy_operator)
        op.apply(self.nodal_fields[name], self._deriv_placeholder)
        sq = self._deriv_placeholder.pg[:, :-1, :-1]
        result = sq.transpose(2, 1, 0).reshape(-1, sq.shape[0]) / self.dy
        return result

    def interpolate_nodal_to_quad(self, name: str) -> None:
        """Interpolate a single nodal field to its quad output field.
        No return, but updates self.quad_fields[name] in-place."""
        op = (self.elements.P2.interpolation_operator
              if name in _FINE_GRID_FIELDS
              else self.elements.P1.interpolation_operator)
        op.apply(self.nodal_fields[name], self.quad_fields[name])

    # =========================================================================
    # Newton scatter / gather  (solver calls these)
    # =========================================================================

    def get_nodal_sol_val(self, var: str) -> NDArray:
        """Extract inner nodal values as a flat vector for the Newton solve.
        """
        return self.nodal_fields[var].p[0].flatten(order='F')

    def set_nodal_sol_val(self, var: str, values: NDArray) -> None:
        """Scatter Newton solution vector back to the nodal field inner region.
        """
        self.nodal_fields[var].p[0] = values.reshape(
            self.nodal_fields[var].p[0].shape, order='F')

    # =========================================================================
    # Sync gate: fine-grid jx/jy → problem.q  (for output / BCs)
    # =========================================================================

    def sync_to_problem_q(self) -> None:
        """Sync Newton-owned nodal fields back to problem.q.
        - problem.q[0] stays ρ-based (canonical external representation).
        - nodal rho is already current (derived from p in update_physics).
        - E: direct reference to p.energy's field — always in sync, no copy needed.
        """
        p = self.problem
        p.q[0] = self.nodal_fields['rho'].pg[0]
        p.q[1] = self.nodal_fields['jx'].pg[0, ::2, ::2]
        p.q[2] = self.nodal_fields['jy'].pg[0, ::2, ::2]

    def sync_from_problem_q(self) -> None:
        """Copy problem.q initial state to Newton-owned nodal fields.
        q[0] is ρ (user-specified IC); derive p via forward EoS.
        Used at initialisation and after external state changes.
        """
        p = self.problem
        Nx_p, Ny_p = self.decomp.local_shape_padded
        Nx_v, Ny_v = self.decomp.local_shape_padded_v
        zoom_factors = (Nx_v / Nx_p, Ny_v / Ny_p)

        self.nodal_fields['rho'].pg[0] = p.q[0]
        self.nodal_fields['p'].pg[0]   = eos_pressure(p.q[0], p.prop)
        self._push_p_to_pressure_field()
        self.nodal_fields['jx'].pg[0] = zoom(p.q[1], zoom_factors, order=1)
        self.nodal_fields['jy'].pg[0] = zoom(p.q[2], zoom_factors, order=1)
        # E is a direct reference to p.energy's field — no copy needed

    def _push_p_to_pressure_field(self) -> None:
        """Copy current nodal p into problem.pressure's underlying field so
        downstream readers (topography, IO, etc.) stay current.
        """
        pressure_arr = np.asarray(self.problem.pressure.pressure)
        src = self.nodal_fields['p'].pg[0]
        pressure_arr[...] = src.reshape(pressure_arr.shape)

    # =========================================================================
    # Field updates  (called once per Newton step)
    # =========================================================================

    def update_physics(self) -> None:
        """Update physics nodal fields after solution field update.

        In the pressure-based solver, `p` is the authoritative nodal DOF
        (just set by Newton). Derive `rho` nodally via the inverse EoS,
        then sync to problem.q so other backends (IO, BCs) see consistent
        ρ. `p.pressure.update()` is NOT called — the pressure field
        (`nodal_fields['p']` alias of `fc.get_real_field('pressure')`) is
        already the DOF.
        """
        p = self.problem
        p_nodal = self.nodal_fields['p'].pg[0]

        # Derive rho nodally from the authoritative p (inverse EoS).
        self.nodal_fields['rho'].pg[0] = eos_rho(p_nodal, p.prop)

        # Keep problem.pressure.pressure in sync for downstream readers.
        self._push_p_to_pressure_field()

        # Expose ρ to problem.q (user BCs, IO, other backends) and jx/jy.
        self.sync_to_problem_q()

        p.topo.update()

        dp_dx = np.gradient(p_nodal, self.dx, axis=0)
        dp_dy = np.gradient(p_nodal, self.dy, axis=1)
        p.viscosity.update(p_nodal, dp_dx, dp_dy,
                           p.topo.h,
                           p.geo['U_bot'], p.geo['V_bot'],
                           p.geo['U_top'], p.geo['V_top'])
        if self.energy:
            p.energy.update_temperature()

    def update_nodal_to_quad(self) -> None:
        """Interpolate nodal fields to quad fields.
        """
        p = self.problem

        # rho, jx, jy, p, eta, E, Tb_top, Tb_bot are always in sync
        self.nodal_fields['h'].pg[0]      = p.topo.h
        self.nodal_fields['dh_dx'].pg[0]  = p.topo.dh_dx
        self.nodal_fields['dh_dy'].pg[0]  = p.topo.dh_dy

        # Interpolate all nodal fields to quad fields.
        # rho is excluded: in the pressure-based solver it is derived from p_q
        # in update_quad_computed(), ensuring residual/Jacobian consistency.
        coarse_interp = ['p', 'h', 'dh_dx', 'dh_dy', 'eta']
        if self.energy:
            coarse_interp.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in coarse_interp:
            self.interpolate_nodal_to_quad(name)
        for name in ('jx', 'jy'):
            self.interpolate_nodal_to_quad(name)

        # Broadcast scalar constants
        self.quad_fields['U_bot'].pg[:] = p.geo['U_bot']
        self.quad_fields['V_bot'].pg[:] = p.geo['V_bot']
        self.quad_fields['U_top'].pg[:] = p.geo['U_top']
        self.quad_fields['V_top'].pg[:] = p.geo['V_top']
        self.quad_fields['Ls'].pg[:]    = p.prop.get('slip_length', 0.0)

    def _apply_2d_vmap(self, func, *args):
        shape = args[0].shape
        args_2d = [a.reshape(shape[0], -1) for a in args]
        return func(*args_2d).reshape(shape)

    def update_quad_computed(self) -> None:
        """Compute derived quantities at quad points (wall stress, drho_dp, etc.)."""
        p = self.problem
        s = np.s_[..., :-1, :-1]
        q = lambda name: self.quad_fields[name].pg[s]  # only on valid squares
        apply = self._apply_2d_vmap

        # drho_dp and dp_drho both evaluated directly from p_q (one-step path).
        # dp_drho = 1/drho_dp avoids the lossy two-step path p_q → rho_q →
        # dp_drho(rho_q), where Δrho can fall below 1 ULP near the mixture
        # region (rho ≈ 850, Δrho ~ 1e-13 < 1 ULP). The identity
        # d(1/drho_dp)/dp_q = d2p_drho2 * drho_dp ensures R11x_corr remains exact.
        drho_dp_q = eos_drho_dp(q('p'), p.prop)
        self.quad_fields['drho_dp'].pg[s] = drho_dp_q
        self.quad_fields['dp_drho'].pg[s] = 1.0 / drho_dp_q

        rho_q = eos_rho(q('p'), p.prop)
        self.quad_fields['rho'].pg[s] = rho_q
        self.quad_fields['d2p_drho2'].pg[s] = apply(p.pressure.d2p_drho2, rho_q)

        self.elements.P2.dx_operator.apply(self.nodal_fields['jx'], self._deriv_placeholder)
        self.quad_fields['d_dx_jx'].pg[s] = self._deriv_placeholder.pg[s] / self.dx

        self.elements.P2.dy_operator.apply(self.nodal_fields['jy'], self._deriv_placeholder)
        self.quad_fields['d_dy_jy'].pg[s] = self._deriv_placeholder.pg[s] / self.dy

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
