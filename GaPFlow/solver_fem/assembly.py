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

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .global_matrix import field_to_global
from .terms import Term
from .fieldspec import FieldSpec

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]

@dataclass
class GlobalIndexPattern:
    """Global indices for nnz values -> global matrix entries."""
    local_size: int
    mat_global_rows: "IntArray"
    mat_global_cols: "IntArray"
    rhs_global_rows: "IntArray"

@dataclass
class AssemblyTemplate:
    """Precomputed injection template for one (res, [var,] deriv_key) combination."""
    w: NDArray
    entries_per_quad: int
    nnz: IntArray

# Residual, Variable
BLOCK = (('P1', 'P1'), ('P1', 'P2'), ('P2', 'P1'), ('P2', 'P2'))


class Assembly:
    """Precomputed sparsity structure for P2P1 assembly.

    Parameters
    ----------
    grid_idx : GridIndexManager
    element : TaylorHoodP2P1
    var_specs : list of FieldSpec
        Active variables in block order.
    res_specs : list of FieldSpec
        Active residuals in block order.
    """

    def __init__(self,
                 grid_idx: GridIndexManager,
                 element: TaylorHoodP2P1,
                 var_specs: List[FieldSpec],
                 res_specs: List[FieldSpec],
                 ) -> None:

        self.grid_idx = grid_idx
        self.element = element
        self.var_specs = var_specs
        self.res_specs = res_specs

        # Convenience views derived from specs — single source of truth
        self.variables = [s.name for s in var_specs]
        self.residuals = [s.name for s in res_specs]
        self.var_to_grid = {s.name: s.grid for s in var_specs}
        self.res_to_grid = {s.name: s.grid for s in res_specs}
        self.var_spec = {s.name: s for s in var_specs}
        self.res_spec = {s.name: s for s in res_specs}
        self.p_factor = sum(1 for s in var_specs if s.grid == 'P1')

        # Ordered list of all (res, var) block combinations
        self.block_order = {
            (rs.name, vs.name): {'res_grid': rs.grid, 'var_grid': vs.grid}
            for rs in res_specs
            for vs in var_specs
        }

        self.conn = self.build_connectivity()

        self._build_coo_pattern(self.conn)

        self.coo_lookups = self._build_coo_lookups()

        self._build_rhs_pattern()
        self._build_slices()
        self._build_scaling_indices()

        self.res_size = self._res_slices[self.residuals[-1]].stop
        # Add +1 as trash can for non-valid entries
        self._nnz_buf = np.zeros(len(self.nnz_global_rows) + 1, dtype=np.float64)
        self._rhs_buf = np.zeros(self.res_size + 1, dtype=np.float64)

    # ======================================================================
    # Block connectivity — stencil-based
    # ======================================================================

    @staticmethod
    def _compute_stencils(element: TaylorHoodP2P1):
        """Compute stencils for all four block types from the fine-grid stencils.
        """

        stencil = {}
        stencil[3] = {
            (0, 0): np.array(element.stencil_even_even, dtype=np.int32),
            (1, 1): np.array(element.stencil_odd_odd,   dtype=np.int32),
            (0, 1): np.array(element.stencil_even_odd,  dtype=np.int32),
            (1, 0): np.array(element.stencil_odd_even,  dtype=np.int32),
        }

        for idx, combination in enumerate(BLOCK[0:3]):

            stencil[idx] = {}
            res = combination[0]   # BLOCK = (res, var)
            var = combination[1]

            for origin in stencil[3].keys():

                if res == 'P1' and origin != (0, 0):
                    continue

                if var == 'P2':
                    stencil[idx][origin] = stencil[3][origin]
                    continue

                # var = 'P1'
                points = stencil[3][origin]
                stencil[idx][origin] = np.empty((0, 2), dtype=np.int32)
                for point in points:
                    x = origin[0] + point[0]
                    y = origin[1] + point[1]
                    if not (x % 2 == 0 and y % 2 == 0):
                        continue
                    stencil[idx][origin] = np.vstack([stencil[idx][origin], point])

        return stencil

    def build_connectivity(self) -> Dict[Tuple[str, str], Tuple[IntArray, IntArray, int]]:
        """Build connectivity for all four block types using fine-grid stencils.
        """
        connectivity = {}
        stencil = self._compute_stencils(self.element)

        for block_index, block_type in enumerate(BLOCK):
            inner_pts, contrib_pts = self.apply_stencil(stencil[block_index], block_type)
            connectivity[block_type] = (inner_pts, contrib_pts, len(inner_pts))
        
        return connectivity


    def apply_stencil(self, stencil, block_type):
        """Apply the given stencil to compute (inner_pts, contrib_pts) for one block type."""

        res_grid, var_grid = block_type
        grid_idx = self.grid_idx

        m_inner_P2  = grid_idx.index_mask_inner_local_P2
        m_padded_P2 = grid_idx.index_mask_padded_local_P2()
        m_padded_P1= grid_idx.index_mask_padded_local_P1()

        inner_pts_2d = np.argwhere(m_inner_P2 >= 0)          # (N, 2)
        inner_idx    = m_inner_P2[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]

        inner_list, contrib_list = [], []

        for origin, offsets in stencil.items():
            sel = ((inner_pts_2d[:, 0] % 2 == origin[0]) &
                   (inner_pts_2d[:, 1] % 2 == origin[1]))
            pts = inner_pts_2d[sel]
            idx = inner_idx[sel]

            for dx, dy in offsets:
                nx = pts[:, 0] + dx
                ny = pts[:, 1] + dy

                if var_grid == 'P2':
                    contrib = m_padded_P2[nx, ny]
                else:
                    contrib = m_padded_P1[nx // 2, ny // 2]

                valid = contrib >= 0
                if res_grid == 'P2':
                    inner_list.append(idx[valid])
                else:
                    inner_list.append(m_padded_P1[pts[valid, 0] // 2, pts[valid, 1] // 2])
                contrib_list.append(contrib[valid])

        inner_all = np.concatenate(inner_list).astype(np.int32)
        contrib_all = np.concatenate(contrib_list).astype(np.int32)

        # TODO: on small periodic grids, distinct fine-grid stencil offsets
        #  (e.g. (0,+2) and (0,-2)) can wrap to the same coarse node after
        #  // 2 + periodic mapping. Best solution: use another index_mask_padded_local
        # that excludes periodicity-wrapped nodes. For now, just remove duplicates here.
        pairs = np.column_stack([inner_all, contrib_all])
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        unique_idx.sort()
        return inner_all[unique_idx], contrib_all[unique_idx]

    # ======================================================================
    # COO pattern
    # ======================================================================

    def _build_coo_pattern(self, connectivity) -> None:
        """Build nnz list. Ordering determined by self._block_order
        """
        self.nnz_local_to = np.empty((0,), dtype=np.int32)
        self.nnz_local_from = np.empty((0,), dtype=np.int32)
        self.nnz_global_rows = np.empty((0,), dtype=np.int32)
        self.nnz_global_cols = np.empty((0,), dtype=np.int32)

        nnz_idx_start = 0

        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            inner_pts, contrib_pts, nb_nnz = connectivity[(rg, vg)]
            self.nnz_local_to = np.concatenate([self.nnz_local_to, inner_pts])
            self.nnz_local_from = np.concatenate([self.nnz_local_from, contrib_pts])

            inner_pts_global_idx = self.apply_l2g(rg, inner_pts)
            contrib_pts_global_idx = self.apply_l2g(vg, contrib_pts)

            global_rows = field_to_global(inner_pts_global_idx, self.res_spec[res], self.grid_idx, self.p_factor)
            global_cols = field_to_global(contrib_pts_global_idx, self.var_spec[var], self.grid_idx, self.p_factor)
            self.nnz_global_rows = np.concatenate([self.nnz_global_rows, global_rows])
            self.nnz_global_cols = np.concatenate([self.nnz_global_cols, global_cols])

            block['nnz_idx_start'] = nnz_idx_start
            block['nb_nnz'] = nb_nnz
            nnz_idx_start += nb_nnz

        decomp = self.grid_idx.decomp
        Nx_P2, Ny_P2 = decomp.nb_domain_grid_pts_P2
        Nx_P1, Ny_P1= decomp.nb_domain_grid_pts
        expected_global_size = 2 * Nx_P2 * Ny_P2 + self.p_factor * Nx_P1* Ny_P1
        local_max = int(self.nnz_global_rows.max()) if len(self.nnz_global_rows) else -1
        global_max = decomp._mpi_comm.allreduce(local_max, op=MPI.MAX)
        assert global_max + 1 == expected_global_size, (
            f"Global matrix size mismatch: computed {global_max + 1}, "
            f"expected {expected_global_size}"
        )

    def apply_l2g(self, grid_type: str, local_indices: IntArray) -> IntArray:
        """Apply local-to-global mapping for the given grid type."""
        if grid_type == 'P2':
            return self.grid_idx.l2g_list_P2[local_indices]
        else:
            return self.grid_idx.l2g_list_P1[local_indices]

    # ======================================================================
    # COO lookup
    # ======================================================================

    def _build_coo_lookups(self):
        """Build one dict per (res, var) block.

        Maps (res_local_idx, var_local_idx) -> absolute flat nnz index.
        Must be called after _build_coo_pattern (needs block['nnz_idx_start']).
        """
        lookups = {}
        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            inner_pts, contrib_pts, nb_nnz = self.conn[(rg, vg)]
            start = block['nnz_idx_start']
            d = {}
            for k in range(nb_nnz):
                pair = (int(inner_pts[k]), int(contrib_pts[k]))
                assert pair not in d, (
                    f"Duplicate pair {pair} in ({res}, {var})")
                d[pair] = start + k
            lookups[(res, var)] = d
        return lookups

    def lookup_nnz(self, res, res_local_idx, var, var_local_idx):
        """Return absolute flat nnz index for (res, res_local_idx, var, var_local_idx).

        Returns -1 if pair not in the sparsity pattern.
        """
        return self.coo_lookups[(res, var)].get(
            (int(res_local_idx), int(var_local_idx)), -1)

    # ======================================================================
    # RHS pattern
    # ======================================================================

    def _build_rhs_pattern(self) -> None:
        """Build global RHS row indices for all residuals.

        For each residual, maps local inner-node indices to global DOF indices
        using field_to_global. Result is stored as self.rhs_global_rows.

        Implicitly assumes that local residual vector is ordered in the
        self.residuals order, and for each residual, the local entries are 
        ordered by the inner node indices.
        """
        rows = []
        for spec in self.res_specs:
            nb_inner = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                        if spec.grid == 'P2'
                        else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)

            global_field_indices = self.apply_l2g(spec.grid, np.arange(nb_inner, dtype=np.int32))
            rows.append(field_to_global(global_field_indices, spec, self.grid_idx, self.p_factor))

        self.rhs_global_rows: IntArray = np.concatenate(rows)

    # ======================================================================
    # Slices
    # ======================================================================

    def _build_slices(self) -> None:
        """Precompute contiguous slices for residual and solution vectors."""
        self._res_slices: Dict[str, slice] = {}
        offset = 0
        for res in self.residuals:
            n = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                 if self.res_to_grid[res] == 'P2'
                 else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)
            self._res_slices[res] = slice(offset, offset + n)
            offset += n

        self._sol_slices: Dict[str, slice] = {}
        offset = 0
        for var in self.variables:
            n = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                 if self.var_to_grid[var] == 'P2'
                 else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)
            self._sol_slices[var] = slice(offset, offset + n)
            offset += n

    def _build_scaling_indices(self) -> None:
        """Build index arrays mapping each nnz / rhs entry to its variable/residual index.

        These are needed by scaling.build_scaling for per-variable/residual scaling.
        """
        var_idx_list = []
        res_idx_list = []
        for (res, var), block in self.block_order.items():
            nb_nnz = block['nb_nnz']
            var_idx_list.append(np.full(nb_nnz, self.variables.index(var), dtype=np.int32))
            res_idx_list.append(np.full(nb_nnz, self.residuals.index(res), dtype=np.int32))
        self.var_block_idx: IntArray = np.concatenate(var_idx_list)
        self.res_block_idx: IntArray = np.concatenate(res_idx_list)

        rhs_idx_list = []
        for i, res in enumerate(self.residuals):
            nb_inner = self._res_slices[res].stop - self._res_slices[res].start
            rhs_idx_list.append(np.full(nb_inner, i, dtype=np.int32))
        self.rhs_res_block_idx: IntArray = np.concatenate(rhs_idx_list)

    # ======================================================================
    # Linear solver info
    # ======================================================================

    def get_petsc_info(self, res_size: int) -> "GlobalIndexPattern":
        """Return assembly info for the linear solver.

        Parameters
        ----------
        res_size : int
            Total residual vector length (sum of nb_inner over all residuals).
        """
        return GlobalIndexPattern(
            local_size=res_size,
            mat_global_rows=self.nnz_global_rows,
            mat_global_cols=self.nnz_global_cols,
            rhs_global_rows=self.rhs_global_rows,
        )

    # ======================================================================
    # Neumann ghost-square zeroing
    # ======================================================================

    def zero_neumann_ghost_squares(self, quad_vals: NDArray,
                                   dep_vars: list) -> None:
        """Zero quad field values on ghost squares at Neumann boundaries.

        Ghost squares are outside the physical domain. For natural (homogeneous
        Neumann) BCs, the variational formulation requires no contribution from
        outside the domain — "do nothing" approach.

        Parameters
        ----------
        quad_vals : (nb_sq, nb_quad_sq)
            Quad field values to modify in-place.
        dep_vars : list of str
            Variable names of the term's dependent variables.
        """
        gi = self.grid_idx
        spr = gi.sq_per_row  # squares per row (x-direction)

        # Flat square index = iy * spr + ix  (ix varies fastest)
        for var in dep_vars:
            if gi.decomp.bc_at_W and gi.is_neumann('W', var):
                quad_vals[::spr, :] = 0.0                # ix=0
            if gi.decomp.bc_at_E and gi.is_neumann('E', var):
                quad_vals[spr - 1::spr, :] = 0.0         # ix=last
            if gi.decomp.bc_at_S and gi.is_neumann('S', var):
                quad_vals[:spr, :] = 0.0                  # iy=0
            if gi.decomp.bc_at_N and gi.is_neumann('N', var):
                quad_vals[-spr:, :] = 0.0                 # iy=last

    # ======================================================================
    # Assembly templates
    # ======================================================================

    def build_assembly_templates(self, terms) -> None:
        """Precompute injection templates for all (res, var, deriv_key) combinations.
        deriv_key = (depvar_deriv, test_deriv) from term.depvar_deriv_for(var).
        Shape weighting depends on deriv_key; nnz_index depends only on (res, var).
        """
        self.assembly_templates = {}

        # Collect all (res, var, dd, td) quadruples actually needed
        needed = set()
        for term in terms:
            td = term.test_deriv
            for var in term.dep_vars:
                dd = term.depvar_deriv_for(var)
                needed.add((term.res, var, dd, td))
            # residual-only key for assemble_rhs
            # rhs uses a single dd per term derived from whether any dep_var has a deriv
            # (handled separately below)

        for res, var, dd, td in needed:
            key = (res, var, dd, td)
            if key not in self.assembly_templates:
                w, entries_per_quad = self._build_weighting(res, var, dd, td)
                self.assembly_templates[key] = AssemblyTemplate(
                    w=w,
                    entries_per_quad=entries_per_quad,
                    nnz=self._build_nnz(res, var),
                )

        # Residual-only keys for assemble_rhs (one key per term: (res, dd, td))
        for term in terms:
            dd, td = term.deriv_key
            key = (term.res, dd, td)
            if key not in self.assembly_templates:
                w, entries_per_quad = self._build_res_weighting(term.res, dd, td)
                self.assembly_templates[key] = AssemblyTemplate(
                    w=w,
                    entries_per_quad=entries_per_quad,
                    nnz=self._build_nnz_res(term.res),
                )

    def _build_weighting(self, res: str, var: str,
                         depvar_deriv: str, test_deriv):
        """Build shape_weighting for one (res, var, depvar_deriv, test_deriv) combination.

        depvar_deriv : 'none', 'x', or 'y' — derivative acting on dep_var
        test_deriv   : None, 'x', or 'y'   — derivative acting on test function
        """
        res_grid, var_grid = self.res_to_grid[res], self.var_to_grid[var]

        res_element = self.element.P1 if res_grid == 'P1' else self.element.P2
        var_element = self.element.P1 if var_grid == 'P1' else self.element.P2

        nodes_tri_res = res_element.nodes_per_tri
        nodes_tri_var = var_element.nodes_per_tri

        # --- Trial function (var) shape functions ---
        deriv_scale = 1.0
        if depvar_deriv == 'none':
            var_N_tri = var_element.N
            var_N = np.tile(var_N_tri, (2, 1))
        else:
            var_N_tri = var_element.dN_dx if depvar_deriv == 'x' else var_element.dN_dy
            d = self.element.dx if depvar_deriv == 'x' else self.element.dy
            var_N = np.concatenate((var_N_tri, -var_N_tri))
            deriv_scale *= 1.0 / d

        # --- Test function (res) shape functions ---
        if not test_deriv:
            res_N_tri = res_element.N
            res_N = np.tile(res_N_tri, (2, 1))
        else:
            res_N_tri = res_element.dN_dx if test_deriv == 'x' else res_element.dN_dy
            d = self.element.dx if test_deriv == 'x' else self.element.dy
            res_N = np.concatenate((res_N_tri, -res_N_tri))
            deriv_scale *= -1.0 / d

        entries_per_quad = nodes_tri_res * nodes_tri_var
        assert entries_per_quad == np.shape(res_N)[1] * np.shape(var_N)[1]

        weights = self.element.Quadrature.weights  # shape (n_quad_tri,)
        area = self.element.sq_area
        weights_vec = np.tile(np.repeat(weights * area * deriv_scale, entries_per_quad), 2)

        var_N_vec = var_N.repeat(nodes_tri_res, axis=1).ravel()
        res_N_vec = np.tile(res_N, (1, nodes_tri_var)).ravel()

        return weights_vec * var_N_vec * res_N_vec, entries_per_quad

    def _build_nnz(self, res: str, var: str) -> IntArray:
        """Build nnz_index for one (res, var) block. Same for all derivative combos.

        Ordering: 0->0, 0->1, ... residual moves faster than variable
        """

        TO_P2 = self.grid_idx.sq_TO_inner_P2  # (n_sq, 9)
        TO_P1= self.grid_idx.sq_TO_inner_P1  # (n_sq, 4)
        FROM_P2 = self.grid_idx.sq_FROM_padded_P2(var)  # (n_sq, 9)
        FROM_P1= self.grid_idx.sq_FROM_padded_P1(var)  # (n_sq, 4)
        n_sq = self.grid_idx.nb_sq
        assert TO_P2.shape[0] == n_sq and TO_P1.shape[0] == n_sq

        res_sq_to_nodes = TO_P1 if self.res_to_grid[res] == 'P1' else TO_P2
        var_sq_to_nodes = FROM_P1 if self.var_to_grid[var] == 'P1' else FROM_P2

        res_element = self.element.P1 if self.res_to_grid[res] == 'P1' else self.element.P2
        var_element = self.element.P1 if self.var_to_grid[var] == 'P1' else self.element.P2

        nb_nnz = n_sq * 2 * res_element.nodes_per_tri * var_element.nodes_per_tri
        nnz = np.empty((nb_nnz), dtype=np.int32)
        nnz_idx = 0

        for sq_idx in range(n_sq):
            for tri_idx in (0, 1):
                res_nodes_on_sq = res_sq_to_nodes[sq_idx]
                var_nodes_on_sq = var_sq_to_nodes[sq_idx]

                res_nodes_on_tri = res_nodes_on_sq[res_element.idx_to_std[tri_idx]]
                var_nodes_on_tri = var_nodes_on_sq[var_element.idx_to_std[tri_idx]]

                for var_node in var_nodes_on_tri:
                    for res_node in res_nodes_on_tri:
                        # Find local nnz index for (res_node, var_node)
                        local_idx = self.lookup_nnz(res, res_node, var, var_node)
                        nnz[nnz_idx] = local_idx
                        nnz_idx += 1

        assert nnz_idx == nb_nnz
        return nnz

    def assemble_matrix(self,
                        quad_fields: Dict[str, NDArray],
                        terms: list,
                        ) -> NDArray:
        """Accumulate Jacobian term contributions and return a view on the result.

        Returns
        -------
        NDArray
            View of shape (n_nnz,) into the internal buffer. Valid until the
            next call to assemble_matrix.
        """
        self._nnz_buf[:] = 0.0

        for term in terms:

            td = term.test_deriv
            dep_vars = [quad_fields[v] for v in term.dep_vars]
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            for var in term.dep_vars:

                dd = term.depvar_deriv_for(var)
                key = (res, var, dd, td)
                tmpl = self.assembly_templates[key]
                sw = tmpl.w
                entries_per_quad = tmpl.entries_per_quad

                res_quad_field = term.evaluate_deriv(var, *dep_vars)  # shape (n_sq, n_quad_sq)
    
                # shape (n_sq * n_quad_sq * entries_per_quad,)
                quad_val_vec = np.repeat(res_quad_field.flatten(), entries_per_quad)
                sw_vec = np.tile(sw, nb_sq)

                q_vec = quad_val_vec * sw_vec
                assert len(q_vec) == nb_sq * quad_per_tri * 2 * entries_per_quad
                assert len(q_vec) % (quad_per_tri * entries_per_quad) == 0
                # size is reduced by factor: quad_per_tri
                ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

                np.add.at(self._nnz_buf, tmpl.nnz, ele_vec)

        return self._nnz_buf[:-1]


    def _build_res_weighting(self, res: str, depvar_deriv: str, test_deriv):
        """Residual weighting arrays (n_quad_sq * nodes_tri_res,)
        """
        res_grid = self.res_to_grid[res]

        res_element = self.element.P1 if res_grid == 'P1' else self.element.P2

        nodes_tri_res = res_element.nodes_per_tri

        if not test_deriv:
            res_N_tri = res_element.N
            res_N = np.tile(res_N_tri, (2, 1))
            factor = 1.0
        else:
            res_N_tri = res_element.dN_dx if test_deriv == 'x' else res_element.dN_dy
            res_N = np.concatenate((res_N_tri, -res_N_tri))
            factor = -1.0 / self.element.dx if test_deriv == 'x' else -1.0 / self.element.dy

        # shape (n_quad_sq * nodes_tri_res,)
        # (q0, N0), (q0, N1), (q0, N2), (q1, N0), ...
        res_N_vec = res_N.reshape(-1)

        # weights compensate for area on square
        weights = self.element.Quadrature.weights  # shape (n_quad_tri,)
        area = self.element.sq_area

        weights_vec = np.tile(np.repeat(weights * area * factor, nodes_tri_res), 2)
        assert len(weights_vec) == len(res_N_vec)

        res_weighting = weights_vec * res_N_vec

        return res_weighting, nodes_tri_res

    def _build_nnz_res(self, res: str) -> IntArray:
        """Build nnz_index for residual-only weighting. Same for all derivative combos.

        Ordering: 0->0, 0->1, ... residual moves faster than variable
        """

        TO_P2 = self.grid_idx.sq_TO_inner_P2  # (n_sq, 9)
        TO_P1= self.grid_idx.sq_TO_inner_P1  # (n_sq, 4)
        n_sq = self.grid_idx.nb_sq
        assert TO_P2.shape[0] == n_sq and TO_P1.shape[0] == n_sq

        res_sq_to_nodes = TO_P1 if self.res_to_grid[res] == 'P1' else TO_P2
        res_element = self.element.P1 if self.res_to_grid[res] == 'P1' else self.element.P2

        nb_nnz = n_sq * 2 * res_element.nodes_per_tri
        nnz = np.empty((nb_nnz), dtype=np.int32)
        nnz_idx = 0

        res_slice_start = self._res_slices[res].start

        for sq_idx in range(n_sq):
            for tri_idx in (0, 1):
                res_nodes_on_sq = res_sq_to_nodes[sq_idx]
                res_nodes_on_tri = res_nodes_on_sq[res_element.idx_to_std[tri_idx]]

                for res_node in res_nodes_on_tri:
                    local_idx = res_slice_start + res_node if res_node >= 0 else -1
                    nnz[nnz_idx] = local_idx
                    nnz_idx += 1

        assert nnz_idx == nb_nnz
        return nnz

    def assemble_rhs(self,
                     quad_fields: Dict[str, NDArray],
                     terms: List[Term],
                     ) -> NDArray:
        """Accumulate residual term contributions and return a view on the result.

        Returns
        -------
        NDArray
            View of shape (res_size,) into the internal buffer. Valid until the
            next call to assemble_rhs.
        """
        self._rhs_buf[:] = 0.0

        for term in terms:

            dd, td = term.deriv_key
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            key_res = (res, dd, td)

            tmpl = self.assembly_templates[key_res]
            nnz = tmpl.nnz
            entries_per_quad = tmpl.entries_per_quad
            sw = tmpl.w

            dep_vars_rhs = [
                quad_fields[v] if (d := term.depvar_deriv_for(v)) == 'none'
                else quad_fields[f'd_d{d}_{v}']
                for v in term.dep_vars
            ]
            quad_vals = term.evaluate(*dep_vars_rhs)

            quad_val_vec = np.repeat(quad_vals.flatten(), entries_per_quad)
            sw_vec = np.tile(sw, nb_sq)

            q_vec = quad_val_vec * sw_vec
            assert len(q_vec) == nb_sq * quad_per_tri * 2 * entries_per_quad

            ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

            np.add.at(self._rhs_buf, nnz, ele_vec)

        return self._rhs_buf[:-1]

    def assemble_rhs_per_term(self,
                              quad_fields: Dict[str, NDArray],
                              terms: List[Term],
                              ) -> Dict[str, NDArray]:
        """Assemble residual contribution of each term individually.

        Returns
        -------
        dict
            Mapping term.name -> NDArray of shape (res_size,) for that term's
            contribution to the residual vector.
        """
        result = {}
        for term in terms:
            self._rhs_buf[:] = 0.0

            dd, td = term.deriv_key
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            key_res = (res, dd, td)
            tmpl = self.assembly_templates[key_res]
            nnz = tmpl.nnz
            entries_per_quad = tmpl.entries_per_quad
            sw = tmpl.w

            dep_vars_rhs = [
                quad_fields[v] if (d := term.depvar_deriv_for(v)) == 'none'
                else quad_fields[f'd_d{d}_{v}']
                for v in term.dep_vars
            ]
            quad_vals = term.evaluate(*dep_vars_rhs)

            # self.zero_neumann_ghost_squares(quad_vals, term.dep_vars)
            quad_val_vec = np.repeat(quad_vals.flatten(), entries_per_quad)
            sw_vec = np.tile(sw, nb_sq)
            ele_vec = (quad_val_vec * sw_vec).reshape(
                -1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

            term_buf = np.zeros_like(self._rhs_buf)
            np.add.at(term_buf, nnz, ele_vec)
            result[term.name] = term_buf[:-1].copy()

        return result
