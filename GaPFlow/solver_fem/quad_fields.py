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
from .fieldspec import (
    NODAL_P1, NODAL_P2, BASE_FIELDS, STRESS_FIELDS,
    ENERGY_FIELDS, CAVITATION_FIELDS, OSS_FIELDS,
)

from ..models.pressure import eos_pressure, eos_rho, eos_drho_dp

if TYPE_CHECKING:
    from ..problem import Problem
    from ..parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]


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
        Decomposition holding both coarse (_p) and fine (_P2) grid info.
    """

    def __init__(self, problem: "Problem",
                 energy: bool,
                 cavitation: bool,
                 variables: List[str],
                 add_fields: List[str],
                 elements: TaylorHoodP2P1,
                 decomp: "DomainDecomposition") -> None:

        self.problem = problem
        self.energy = energy
        self.cavitation = cavitation
        self.variables = variables
        self.add_fields = add_fields
        self.elements = elements
        self.decomp = decomp

        self.dx = problem.grid['dx']
        self.dy = problem.grid['dy']

        self.nodal_fields: Dict[str, Field] = {}
        self.quad_fields: Dict[str, Field] = {}

        self.nf = lambda name: self.nodal_fields[name].pg[0]
        self.qf = lambda name: self.quad_fields[name].pg

        self._init_fields()

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_fields(self) -> None:
        """Register all nodal and quadrature fields in the muGrid FieldCollections.

        - create new nodal fields for coarse-grid variables rho and h (to have individual fields)
        - get reference to p, eta fields from problem
        - create nodal fields for fine-grid variables jx, jy
        - create placeholder field for on-the-fly derivative computation
        - create quadrature fields for all nodal and intermediate quantities
        """
        fc = self.decomp.fc

        # ---- P1 fields ----
        p1_nodal = NODAL_P1 + self.add_fields
        for name in p1_nodal:
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # plug in existing fields that we access directly
        self.nodal_fields['eta'] = Field(fc.get_real_field('shear_viscosity'))
        if self.energy:
            self.nodal_fields['E'] = Field(fc.get_real_field('total_energy'))
            self.nodal_fields['Tb_top'] = Field(fc.get_real_field('Tb_top'))
            self.nodal_fields['Tb_bot'] = Field(fc.get_real_field('Tb_bot'))

        # ---- P2 fields ----
        fc_P2 = self.decomp.fc_P2
        for name in NODAL_P2:
            self.nodal_fields[name] = fc_P2.real_field(f'{name}_nodal', 1, 'pixel')

        # ---- Quadrature fields ----
        nb_quad_sq = self.elements.n_tri * self.elements.Quadrature.nb_points
        fc.set_nb_sub_pts('quad', nb_quad_sq)

        for name in self._needed_quad_fields():
            self.quad_fields[name] = fc.real_field(f'{name}_q', 1, 'quad')

        # placeholder for on-the-fly derivative computation
        self._deriv_placeholder = fc.real_field('deriv_placeholder', 1, 'quad')

    def _needed_quad_fields(self) -> Set[str]:
        needed = BASE_FIELDS | STRESS_FIELDS
        if self.energy:
            needed |= ENERGY_FIELDS
        if self.cavitation:
            needed |= CAVITATION_FIELDS
        if 'xi' in self.variables:
            needed |= OSS_FIELDS
        return needed

    # =========================================================================
    # Field access  (assembly calls these)
    # transpose from (nb_quad_sq, sq_per_row, sq_per_col) to (nb_sq, nb_quad_sq)
    # =========================================================================

    def _deriv_pg(self, name: str, axis: str) -> NDArray:
        """Derivative of nodal field `name` along `axis` ('x' or 'y') at quad points.
        Returns raw pg shape (nb_quad_sq, Nx-1, Ny-1), trimmed to inner squares."""
        el = self.elements.P2 if name in NODAL_P2 else self.elements.P1
        getattr(el, f'd{axis}_operator').apply(self.nodal_fields[name], self._deriv_placeholder)
        return self._deriv_placeholder.pg[..., :-1, :-1] / getattr(self, f'd{axis}')

    def get_quad_sq(self, name: str) -> NDArray:
        """Quadrature values for all squares.
        Returns shape (nb_sq, nb_quad_sq): squares x-major, quad innermost.
        """
        sq = self.quad_fields[name].pg[:, :-1, :-1]
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def get_quad_dx_sq(self, name: str) -> NDArray:
        """d(field)/dx at quad points, shape (nb_sq, nb_quad_sq)."""
        sq = self._deriv_pg(name, 'x')
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def get_quad_dy_sq(self, name: str) -> NDArray:
        """d(field)/dy at quad points, shape (nb_sq, nb_quad_sq)."""
        sq = self._deriv_pg(name, 'y')
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def interpolate_nodal_to_quad(self, name: str) -> None:
        """Interpolate a single nodal field to its quad output field.
        No return, but updates self.quad_fields[name] in-place."""
        el = self.elements.P2 if name in NODAL_P2 else self.elements.P1
        el.interpolation_operator.apply(self.nodal_fields[name], self.quad_fields[name])

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
        """Sync FEM-owned nodal fields back to problem.q."""
        p = self.problem
        p.q[0] = self.nf('rho')
        p.q[1] = self.nf('jx')[::2, ::2]
        p.q[2] = self.nf('jy')[::2, ::2]
        for i, name in enumerate(self.add_fields):
            p.q[3 + i] = self.nf(name)

    def sync_from_problem_q(self) -> None:
        """Initial state copy of problem.q to FEM-owned nodal fields."""
        p = self.problem

        # P1 fields
        self.nf('rho')[:] = p.q[0]
        self.nf('p')[:]   = eos_pressure(p.q[0], p.prop)
        self.problem.pressure.pressure[:] = self.nf('p')

        for i, name in enumerate(self.add_fields):
            self.nf(name)[:] = p.q[3 + i]

        # P2 fields
        Nx_p, Ny_p = self.decomp.local_shape_padded
        Nx_P2, Ny_P2 = self.decomp.local_shape_padded_P2
        zoom_factors = (Nx_P2 / Nx_p, Ny_P2 / Ny_p)
        self.nf('jx')[:] = zoom(p.q[1], zoom_factors, order=1)
        self.nf('jy')[:] = zoom(p.q[2], zoom_factors, order=1)

    # =========================================================================
    # Field updates  (called once per Newton step)
    # =========================================================================

    def update_physics(self) -> None:
        """Update physics model fields after solution field update."""
        p = self.problem

        # sync p to pressure module
        p_nodal = self.nf('p')
        self.problem.pressure.pressure[:] = p_nodal

        # rho update
        self.nf('rho')[:] = eos_rho(p_nodal, p.prop)
        self.sync_to_problem_q()

        # height update (needs updated p in pressure module)
        p.topo.update()

        # viscosity
        dp_dx = np.gradient(p_nodal, self.dx, axis=0)
        dp_dy = np.gradient(p_nodal, self.dy, axis=1)
        p.viscosity.update(p_nodal, dp_dx, dp_dy,
                           p.topo.h,
                           p.geo['U_bot'], p.geo['V_bot'],
                           p.geo['U_top'], p.geo['V_top'])

        # energy
        if self.energy:
            p.energy.update_temperature()

    def update_nodal_to_quad(self) -> None:
        """Interpolate nodal fields to quad fields."""
        p = self.problem

        # rho, jx, jy, p, eta, E, Tb_top, Tb_bot are always in sync
        self.nf('h')[:]      = p.topo.h
        self.nf('dh_dx')[:] = p.topo.dh_dx
        self.nf('dh_dy')[:] = p.topo.dh_dy

        for name in self.nodal_fields:
            self.interpolate_nodal_to_quad(name)

        # broadcast scalar constants
        self.qf('U_bot')[:] = p.geo['U_bot']
        self.qf('V_bot')[:] = p.geo['V_bot']
        self.qf('U_top')[:] = p.geo['U_top']
        self.qf('V_top')[:] = p.geo['V_top']
        self.qf('Ls')[:]    = p.prop.get('slip_length', 0.0)

    def _apply_2d_vmap(self, func, *args):
        shape = args[0].shape
        args_2d = [a.reshape(shape[0], -1) for a in args]
        return func(*args_2d).reshape(shape)

    def update_quad_computed(self) -> None:
        """Compute derived quantities at quad points (wall stress, drho_dp, etc.).
        Note: s to neglect periodic wrap-around strip in quad fields
        """
        p = self.problem
        s = np.s_[..., :-1, :-1]
        q = lambda name: self.quad_fields[name].pg[s]
        apply = self._apply_2d_vmap

        drho_dp_q = eos_drho_dp(q('p'), p.prop)
        q('drho_dp')[:] = drho_dp_q
        q('dp_drho')[:] = 1.0 / drho_dp_q

        rho_q = eos_rho(q('p'), p.prop)
        q('rho')[:] = rho_q
        q('d2p_drho2')[:] = apply(p.pressure.d2p_drho2, rho_q)

        # for R11 correction term
        q('d_dx_jx')[:] = self._deriv_pg('jx', 'x')
        q('d_dy_jy')[:] = self._deriv_pg('jy', 'y')

        if 'xi' in self.variables:
            self._update_oss_quad_fields(q)

        # wall stress preparation
        dp_dx_q = self._deriv_pg('p', 'x')
        dp_dy_q = self._deriv_pg('p', 'y')
        theta_q = q('theta') if self.cavitation else np.zeros_like(q('rho'))

        # wall stress xz
        args_xz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dx'),
                   q('U_bot'), q('V_bot'), q('U_top'), q('V_top'), q('Ls'), theta_q,
                   dp_dx_q, dp_dy_q)
        for name in ['tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
                     'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx']:
            q(name)[:] = apply(getattr(p.wall_stress_xz, name), *args_xz)
        if self.cavitation:
            for name in ['dtau_xz_dtheta', 'dtau_xz_bot_dtheta']:
                q(name)[:] = apply(getattr(p.wall_stress_xz, name), *args_xz)

        # wall stress yz
        args_yz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dy'),
                   q('U_bot'), q('V_bot'), q('U_top'), q('V_top'), q('Ls'), theta_q,
                   dp_dx_q, dp_dy_q)
        for name in ['tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
                     'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy']:
            q(name)[:] = apply(getattr(p.wall_stress_yz, name), *args_yz)
        if self.cavitation:
            for name in ['dtau_yz_dtheta', 'dtau_yz_bot_dtheta']:
                q(name)[:] = apply(getattr(p.wall_stress_yz, name), *args_yz)

        # energy
        if self.energy:
            args_T = (q('rho'), q('jx'), q('jy'), q('E'))
            for name, func in [('T', 'T_func'), ('dT_drho', 'T_grad_rho'),
                                ('dT_djx', 'T_grad_jx'), ('dT_djy', 'T_grad_jy'),
                                ('dT_dE', 'T_grad_E')]:
                q(name)[:] = getattr(p.energy, func)(*args_T)

            args_S = (q('h'), q('eta'), q('rho'), q('E'), q('jx'), q('jy'),
                      q('U_bot'), q('V_bot'), q('Tb_top'), q('Tb_bot'))
            for name, func in [('S', 'q_wall_sum'), ('dS_drho', 'q_wall_grad_rho'),
                                ('dS_djx', 'q_wall_grad_jx'),
                                ('dS_djy', 'q_wall_grad_jy'),
                                ('dS_dE', 'q_wall_grad_E')]:
                q(name)[:] = apply(getattr(p.energy, func), *args_S)

        # body force
        q('force_x')[:] = p.prop['force_x']
        q('force_y')[:] = p.prop['force_y']

    def _update_oss_quad_fields(self, q) -> None:
        """Compute OSS stabilisation fields (a_vec, tau, one_minus_theta) at quad points."""

        a_vec_x = q('dp_drho') * q('jx')
        a_vec_y = q('dp_drho') * q('jy')
        q('a_vec_x')[:] = a_vec_x
        q('a_vec_y')[:] = a_vec_y

        norm_a = np.sqrt(a_vec_x**2 + a_vec_y**2)
        h_elem = min(self.dx, self.dy)
        if self.problem.fem_solver.get('oss_tau_pointwise', True):
            tau = h_elem / (2.0 * np.maximum(norm_a, 0.1))
        else:
            tau = 1.0 / (2.0 / h_elem * max(float(norm_a.mean()), 0.1))

        alpha = self.problem.fem_solver.get('oss_theta_alpha', 0.0)
        q('tau_a_x')[:] = alpha * tau * a_vec_x
        q('tau_a_y')[:] = alpha * tau * a_vec_y
        q('one_minus_theta')[:] = 1.0 - q('theta')

    def store_prev_values(self) -> None:
        """Store current quad values for time derivatives."""
        for var in self.variables:
            prev_key = f'{var}_prev'
            if var in self.quad_fields and prev_key in self.quad_fields:
                self.qf(prev_key)[:] = self.qf(var).copy()
