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
FEM Assembly — Taylor-Hood P2P1, sparsity structure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .global_matrix import field_to_global
from .terms import NonLinearTerm

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
    """Precomputed injection template for one (res, var, deriv_combo) combination.

    shape_weighting : (n_tri * n_quad * n_contr,)
        Element-uniform quadrature weights. Flat index: t*n_quad*n_contr + q*n_contr + contr
        where contr = i*n_j + j.
    nnz_index : (n_sq, n_tri, n_contr)
        Absolute flat nnz index for each square, triangle, node-node pair. -1 if invalid.
    """
    shape_weighting: NDArray
    nnz_index:       IntArray
    n_tri:           int
    n_quad:          int
    n_contr:         int
    n_sq:            int

# Residual, Variable
BLOCK = (('p', 'p'), ('p', 'v'), ('v', 'p'), ('v', 'v'))
DOF_GRID = {
    'jx': 'v', 'jy': 'v', 'rho': 'p', 'E': 'p',
    'momentum_x': 'v', 'momentum_y': 'v', 'mass': 'p', 'energy': 'p',
}
DOF_IDX = {
    'jx': 0, 'jy': 1, 'rho': 2, 'E': 3,
    'momentum_x': 0, 'momentum_y': 1, 'mass': 2, 'energy': 3,
}


class Assembly:
    """Precomputed sparsity structure for P2P1 assembly.

    Parameters
    ----------
    grid_idx : GridIndexManager
    variables : list of str
        All variable names in block order, e.g. ['jx', 'jy', 'rho'].
    residuals : list of str
        All residual names in block order, e.g. ['Rjx', 'Rjy', 'Rrho'].
    energy : bool
        Whether energy is active.
    """

    def __init__(self,
                 grid_idx: GridIndexManager,
                 element: TaylorHoodP2P1,
                 variables: List[str],
                 residuals: List[str],
                 energy: bool = False,
                 ) -> None:

        self.grid_idx = grid_idx
        self.variables = variables
        self.residuals = residuals
        self.energy = energy
        self.element = element

        self.res_to_grid = {r: DOF_GRID[r] for r in residuals}
        self.var_to_grid = {v: DOF_GRID[v] for v in variables}

        # Ordered list of all (res, var) block combinations
        self.block_order = {
            (res, var): {'res_grid': DOF_GRID[res], 'var_grid': DOF_GRID[var]}
            for res in residuals
            for var in variables
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

                if res == 'p' and origin != (0, 0):
                    continue

                if var == 'v':
                    stencil[idx][origin] = stencil[3][origin]
                    continue

                # var = 'p'
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

        m_inner_v  = grid_idx.index_mask_inner_local_v
        m_padded_v = grid_idx.index_mask_padded_local_v()
        m_padded_p = grid_idx.index_mask_padded_local_p()

        inner_pts_2d = np.argwhere(m_inner_v >= 0)          # (N, 2)
        inner_idx    = m_inner_v[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]

        inner_list, contrib_list = [], []

        for origin, offsets in stencil.items():
            sel = ((inner_pts_2d[:, 0] % 2 == origin[0]) &
                   (inner_pts_2d[:, 1] % 2 == origin[1]))
            pts = inner_pts_2d[sel]
            idx = inner_idx[sel]

            for dx, dy in offsets:
                nx = pts[:, 0] + dx
                ny = pts[:, 1] + dy

                if var_grid == 'v':
                    contrib = m_padded_v[nx, ny]
                else:
                    contrib = m_padded_p[nx // 2, ny // 2]

                valid = contrib >= 0
                if res_grid == 'v':
                    inner_list.append(idx[valid])
                else:
                    inner_list.append(m_padded_p[pts[valid, 0] // 2, pts[valid, 1] // 2])
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

            res_idx , var_idx = DOF_IDX[res], DOF_IDX[var]
            global_rows = field_to_global(inner_pts_global_idx, res_idx, self.grid_idx, self.energy)
            global_cols = field_to_global(contrib_pts_global_idx, var_idx, self.grid_idx, self.energy)
            self.nnz_global_rows = np.concatenate([self.nnz_global_rows, global_rows])
            self.nnz_global_cols = np.concatenate([self.nnz_global_cols, global_cols])

            block['nnz_idx_start'] = nnz_idx_start
            block['nb_nnz'] = nb_nnz
            nnz_idx_start += nb_nnz

        decomp = self.grid_idx._decomp
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v
        Nx_p, Ny_p = decomp.nb_domain_grid_pts
        p_factor = 2 if self.energy else 1
        expected_global_size = 2 * Nx_v * Ny_v + p_factor * Nx_p * Ny_p
        local_max = int(self.nnz_global_rows.max()) if len(self.nnz_global_rows) else -1
        global_max = decomp._mpi_comm.allreduce(local_max, op=MPI.MAX)
        assert global_max + 1 == expected_global_size, (
            f"Global matrix size mismatch: computed {global_max + 1}, "
            f"expected {expected_global_size}"
        )

    def apply_l2g(self, grid_type: str, local_indices: IntArray) -> IntArray:
        """Apply local-to-global mapping for the given grid type."""
        if grid_type == 'v':
            return self.grid_idx.l2g_list_v[local_indices]
        else:
            return self.grid_idx.l2g_list_p[local_indices]

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
        for res in self.residuals:

            res_type = DOF_IDX[res]

            if DOF_GRID[res] == 'v':
                nb_inner = self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
            else:
                nb_inner = self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner

            # get DomainDecomposition padded_global indices
            global_field_indices = self.apply_l2g(DOF_GRID[res], np.arange(nb_inner, dtype=np.int32))

            rows.append(field_to_global(global_field_indices, res_type,
                                        self.grid_idx, self.energy))

        self.rhs_global_rows: IntArray = np.concatenate(rows)

    # ======================================================================
    # Slices
    # ======================================================================

    def _build_slices(self) -> None:
        """Precompute contiguous slices for residual and solution vectors."""
        self._res_slices: Dict[str, slice] = {}
        offset = 0
        for res in self.residuals:
            n = (self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
                 if DOF_GRID[res] == 'v'
                 else self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner)
            self._res_slices[res] = slice(offset, offset + n)
            offset += n

        self._sol_slices: Dict[str, slice] = {}
        offset = 0
        for var in self.variables:
            n = (self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
                 if DOF_GRID[var] == 'v'
                 else self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner)
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
            if gi.bc_at_W and gi._bc_neumann['xW'].get(var, False):
                quad_vals[::spr, :] = 0.0                # ix=0
            if gi.bc_at_E and gi._bc_neumann['xE'].get(var, False):
                quad_vals[spr - 1::spr, :] = 0.0         # ix=last
            if gi.bc_at_S and gi._bc_neumann['yS'].get(var, False):
                quad_vals[:spr, :] = 0.0                  # iy=0
            if gi.bc_at_N and gi._bc_neumann['yN'].get(var, False):
                quad_vals[-spr:, :] = 0.0                 # iy=last

    # ======================================================================
    # Assembly templates
    # ======================================================================

    def build_assembly_templates(self, terms) -> None:
        """Precompute injection templates for all (res, var, deriv_key) combinations.
        deriv_key = (depvar_deriv, testfun_deriv) from term.deriv_key.
        Shape weighting depends on deriv_key; nnz_index depends only on (res, var).
        """
        self.assembly_templates = {}

        # Collect all (depvar_deriv, testfun_deriv) pairs from active terms
        deriv_keys = set(term.deriv_key for term in terms)

        for res in self.residuals:
            for dd, td in deriv_keys:
                for var in self.variables:

                    if not self._term_exists(terms, res, var, dd, td):
                        continue

                    key = (res, var, dd, td)
                    self.assembly_templates[key] = {}

                    shape_weighting, entries_per_quad = self._build_weighting(res, var, dd, td)
                    self.assembly_templates[key]['w'] = shape_weighting
                    self.assembly_templates[key]['entries_per_quad'] = entries_per_quad
                    self.assembly_templates[key]['nnz'] = self._build_nnz(res, var)

                key = (res, dd, td)
                if key not in self.assembly_templates:
                    res_weighting, entries_per_quad = self._build_res_weighting(res, dd, td)
                    self.assembly_templates[key] = {}
                    self.assembly_templates[key]['w'] = res_weighting
                    self.assembly_templates[key]['entries_per_quad'] = entries_per_quad
                    self.assembly_templates[key]['nnz'] = self._build_nnz_res(res)




    def _term_exists(self, terms, res: str, var: str,
                     depvar_deriv: str, testfun_deriv) -> bool:
        """Check if any term has the given (res, var, deriv_key) combination."""
        for term in terms:
            if term.res == res and var in term.dep_vars:
                if term.deriv_key == (depvar_deriv, testfun_deriv):
                    return True
        return False

    def _build_weighting(self, res: str, var: str,
                         depvar_deriv: str, der_testfun):
        """Build shape_weighting for one (res, var, depvar_deriv, der_testfun) combination.

        depvar_deriv : 'none', 'x', or 'y' — derivative acting on dep_var
        der_testfun  : False, 'x', or 'y'  — derivative acting on test function
        """
        res_grid, var_grid = DOF_GRID[res], DOF_GRID[var]

        res_element = self.element.P1 if res_grid == 'p' else self.element.P2
        var_element = self.element.P1 if var_grid == 'p' else self.element.P2

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
        if not der_testfun:
            res_N_tri = res_element.N
            res_N = np.tile(res_N_tri, (2, 1))
        else:
            res_N_tri = res_element.dN_dx if der_testfun == 'x' else res_element.dN_dy
            d = self.element.dx if der_testfun == 'x' else self.element.dy
            res_N = np.concatenate((res_N_tri, -res_N_tri))
            deriv_scale *= -1.0 / d

        entries_per_quad = nodes_tri_res * nodes_tri_var
        assert entries_per_quad == np.shape(res_N)[1] * np.shape(var_N)[1]

        nb_quad_sq = self.element.n_tri * self.element.Quadrature.nb_points
        shape_weighting = np.empty((nb_quad_sq * entries_per_quad,))

        weights = self.element.Quadrature.weights  # shape (n_quad_tri,)
        area = self.element.sq_area

        weights_vec = np.tile(np.repeat(weights * area * deriv_scale, entries_per_quad), 2)
        assert len(weights_vec) == len(shape_weighting)

        var_N_vec = np.empty((nb_quad_sq * entries_per_quad,))
        # var_N shape: (n_quad, nodes_per_tri)
        for quad_idx in range(nb_quad_sq):
            var_N_quad = var_N[quad_idx, :]  # shape (nodes_per_tri,)
            var_N_quad_tile = np.repeat(var_N_quad, nodes_tri_res)  # shape (entries_per_quad,)
            assert len(var_N_quad_tile) == entries_per_quad
            var_N_vec[quad_idx * entries_per_quad:(quad_idx + 1) * entries_per_quad] = var_N_quad_tile

        res_N_vec = np.empty((nb_quad_sq * entries_per_quad,))
        for quad_idx in range(nb_quad_sq):
            res_N_quad = res_N[quad_idx, :]  # shape (nodes_per_tri,)
            res_N_quad_repeat = np.tile(res_N_quad, nodes_tri_var)  # shape (entries_per_quad,)
            assert len(res_N_quad_repeat) == entries_per_quad
            res_N_vec[quad_idx * entries_per_quad:(quad_idx + 1) * entries_per_quad] = res_N_quad_repeat
        
        shape_weighting = weights_vec * var_N_vec * res_N_vec
        return shape_weighting, entries_per_quad

    def _build_nnz(self, res: str, var: str) -> IntArray:
        """Build nnz_index for one (res, var) block. Same for all derivative combos.

        Ordering: 0->0, 0->1, ... residual moves faster than variable
        """

        TO_v = self.grid_idx.sq_TO_inner_v  # (n_sq, 9)
        TO_p = self.grid_idx.sq_TO_inner_p  # (n_sq, 4)
        FROM_v = self.grid_idx.sq_FROM_padded_v(var)  # (n_sq, 9)
        FROM_p = self.grid_idx.sq_FROM_padded_p(var)  # (n_sq, 4)
        n_sq = self.grid_idx.nb_sq
        assert TO_v.shape[0] == n_sq and TO_p.shape[0] == n_sq

        res_sq_to_nodes = TO_p if DOF_GRID[res] == 'p' else TO_v
        var_sq_to_nodes = FROM_p if DOF_GRID[var] == 'p' else FROM_v

        res_element = self.element.P1 if DOF_GRID[res] == 'p' else self.element.P2
        var_element = self.element.P1 if DOF_GRID[var] == 'p' else self.element.P2

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

            dd, td = term.deriv_key
            dep_vars = [quad_fields[v] for v in term.dep_vars]
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            for var in term.dep_vars:

                key = (res, var, dd, td)
                sw = self.assembly_templates[key]['w']
                entries_per_quad = self.assembly_templates[key]['entries_per_quad']

                res_quad_field = term.evaluate_deriv(var, *dep_vars)  # shape (n_sq, n_quad_sq)
                # self.zero_neumann_ghost_squares(res_quad_field, term.dep_vars)

                # shape (n_sq * n_quad_sq * entries_per_quad,)
                quad_val_vec = np.repeat(res_quad_field.flatten(), entries_per_quad)
                sw_vec = np.tile(sw, nb_sq)

                q_vec = quad_val_vec * sw_vec
                assert len(q_vec) == nb_sq * quad_per_tri * 2 * entries_per_quad
                assert len(q_vec) % (quad_per_tri * entries_per_quad) == 0
                # size is reduced by factor: quad_per_tri
                ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

                np.add.at(self._nnz_buf, self.assembly_templates[key]['nnz'], ele_vec)

        return self._nnz_buf[:-1]


    def _build_res_weighting(self, res: str, depvar_deriv: str, der_testfun):
        """Residual weighting arrays (n_quad_sq * nodes_tri_res,)
        """
        res_grid = DOF_GRID[res]

        res_element = self.element.P1 if res_grid == 'p' else self.element.P2

        nodes_tri_res = res_element.nodes_per_tri

        if not der_testfun:
            res_N_tri = res_element.N
            res_N = np.tile(res_N_tri, (2, 1))
            factor = 1.0
        else:
            res_N_tri = res_element.dN_dx if der_testfun == 'x' else res_element.dN_dy
            res_N = np.concatenate((res_N_tri, -res_N_tri))
            factor = -1.0 / self.element.dx if der_testfun == 'x' else -1.0 / self.element.dy

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

        TO_v = self.grid_idx.sq_TO_inner_v  # (n_sq, 9)
        TO_p = self.grid_idx.sq_TO_inner_p  # (n_sq, 4)
        n_sq = self.grid_idx.nb_sq
        assert TO_v.shape[0] == n_sq and TO_p.shape[0] == n_sq

        res_sq_to_nodes = TO_p if DOF_GRID[res] == 'p' else TO_v
        res_element = self.element.P1 if DOF_GRID[res] == 'p' else self.element.P2

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
                     terms: List[NonLinearTerm],
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
            dep_vars = [quad_fields[v] for v in term.dep_vars]
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            key_res = (res, dd, td)

            nnz = self.assembly_templates[key_res]['nnz']
            entries_per_quad = self.assembly_templates[key_res]['entries_per_quad']
            sw = self.assembly_templates[key_res]['w']

            if dd == 'none':
                quad_vals = term.evaluate(*dep_vars)  # shape (n_sq, n_quad_sq)
            else:
                # Chain rule: sum df/dvar * dvar/d{dd} over all dep_vars
                quad_vals = sum(
                    term.evaluate_deriv(v, *dep_vars) * quad_fields[f'd_d{dd}_{v}']
                    for v in term.dep_vars
                )

            #self.zero_neumann_ghost_squares(quad_vals, term.dep_vars)
            quad_val_vec = np.repeat(quad_vals.flatten(), entries_per_quad)
            sw_vec = np.tile(sw, nb_sq)

            q_vec = quad_val_vec * sw_vec
            assert len(q_vec) == nb_sq * quad_per_tri * 2 * entries_per_quad

            ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

            np.add.at(self._rhs_buf, nnz, ele_vec)

        return self._rhs_buf[:-1]

    def assemble_rhs_per_term(self,
                              quad_fields: Dict[str, NDArray],
                              terms: List[NonLinearTerm],
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
            dep_vars = [quad_fields[v] for v in term.dep_vars]
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            key_res = (res, dd, td)
            nnz = self.assembly_templates[key_res]['nnz']
            entries_per_quad = self.assembly_templates[key_res]['entries_per_quad']
            sw = self.assembly_templates[key_res]['w']

            if dd == 'none':
                quad_vals = term.evaluate(*dep_vars)
            else:
                quad_vals = sum(
                    term.evaluate_deriv(v, *dep_vars) * quad_fields[f'd_d{dd}_{v}']
                    for v in term.dep_vars
                )

            self.zero_neumann_ghost_squares(quad_vals, term.dep_vars)
            quad_val_vec = np.repeat(quad_vals.flatten(), entries_per_quad)
            sw_vec = np.tile(sw, nb_sq)
            ele_vec = (quad_val_vec * sw_vec).reshape(
                -1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

            np.add.at(self._rhs_buf, nnz, ele_vec)
            result[term.name] = self._rhs_buf[:-1].copy()

        return result
