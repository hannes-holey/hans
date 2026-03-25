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

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .global_matrix import field_to_global

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
class CooLookup:
    """Binary-search lookup for one (res, var) block.

    Keys:   res_local_idx * max_col + var_local_idx  (int64)
    Values: absolute flat nnz index into the full nnz_values array (int32)
    """
    sorted_keys:   IntArray   # int64, shape (nb_nnz,)
    sorted_values: IntArray   # int32, shape (nb_nnz,)
    max_col:       int        # max var_local_idx + 1

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

DERIV_COMBOS = ('none', 'dx_var', 'dy_var', 'dx_test', 'dy_test')


def _deriv_combo(term) -> str:
    """Derive the deriv_combo string from a term's flags."""
    der_tf = getattr(term, 'der_testfun', False)
    if term.d_dx_resfun:
        return 'dx_var'
    if term.d_dy_resfun:
        return 'dy_var'
    if der_tf in ('x', True):
        return 'dx_test'
    if der_tf == 'y':
        return 'dy_test'
    return 'none'


def _build_shape_weighting(
        dc: str,
        n_tri: int, n_quad: int, weights: NDArray, area: float,
        N_i: NDArray, dNdx_i: NDArray, dNdy_i: NDArray, der_i: NDArray,
        N_j: NDArray, dNdx_j: NDArray, dNdy_j: NDArray, der_j: NDArray,
) -> NDArray:
    """Build shape_weighting array for one (block_type, deriv_combo).

    Returns flat array of shape (n_tri * n_quad * n_contr,) where
    n_contr = n_i * n_j and the flat index is t*n_quad*n_contr + q*n_contr + i*n_j + j.

    dNdx_i/dNdx_j are already divided by dx/dy by the caller.
    der_i/der_j are the per-triangle sign factors [+1, -1].
    """
    if dc == 'none':
        Phi_i, Phi_j = N_i, N_j
        sign_i = sign_j = np.ones(n_tri)
    elif dc == 'dx_var':
        Phi_i, Phi_j = N_i, dNdx_j
        sign_i, sign_j = np.ones(n_tri), der_j
    elif dc == 'dy_var':
        Phi_i, Phi_j = N_i, dNdy_j
        sign_i, sign_j = np.ones(n_tri), der_j
    elif dc == 'dx_test':
        Phi_i, Phi_j = dNdx_i, N_j
        sign_i, sign_j = der_i, np.ones(n_tri)
    else:  # dy_test
        Phi_i, Phi_j = dNdy_i, N_j
        sign_i, sign_j = der_i, np.ones(n_tri)

    n_i, n_j = Phi_i.shape[1], Phi_j.shape[1]
    sw = np.empty((n_tri, n_quad, n_i, n_j))
    for t in range(n_tri):
        for q in range(n_quad):
            sw[t, q] = (weights[q] * area * sign_i[t] * sign_j[t]
                        * np.outer(Phi_i[q], Phi_j[q]))
    return sw.reshape(-1)


def _build_nnz_index(
        res: str, var: str,
        n_sq: int, n_tri: int, n_i: int, n_j: int,
        std_i: IntArray, std_j: IntArray,
        TO: IntArray, FROM: IntArray,
        lookup_nnz,
) -> IntArray:
    """Build nnz_index array of shape (n_sq, n_tri, n_contr).

    Iteration order for contr index: i-major, j-minor (contr = i*n_j + j).
    Uses lookup_nnz(res, row_local, var, col_local) -> absolute flat nnz index.
    """
    n_contr = n_i * n_j
    nnz_index = np.full((n_sq, n_tri, n_contr), -1, dtype=np.int32)
    for i in range(n_i):
        for j in range(n_j):
            contr = i * n_j + j
            for t in range(n_tri):
                to_pts   = TO[:, std_i[t, i]]     # (n_sq,)
                from_pts = FROM[:, std_j[t, j]]   # (n_sq,)
                valid = (to_pts >= 0) & (from_pts >= 0)
                if valid.any():
                    nnz_index[valid, t, contr] = lookup_nnz(
                        res, to_pts[valid], var, from_pts[valid])
    return nnz_index


def _inject_term(nnz_values: NDArray,
                 tmpl: "AssemblyTemplate",
                 field: NDArray,
                 ) -> None:
    """Inject one Jacobian term into nnz_values (in-place).

    field : shape (n_tri * n_quad, sq_x, sq_y) — quadrature-point values.
    Accumulates: nnz_values[nnz_index[sq,t,contr]] += shape_weighting[t,q,contr] * field[t*nq+q, sq]
    """
    n_tri   = tmpl.n_tri
    n_quad  = tmpl.n_quad
    n_contr = tmpl.n_contr
    sw = tmpl.shape_weighting.reshape(n_tri, n_quad, n_contr)

    for t in range(n_tri):
        for q in range(n_quad):
            fv  = field[t * n_quad + q].ravel()    # (n_sq,)
            idx = tmpl.nnz_index[:, t, :]          # (n_sq, n_contr)
            valid = idx >= 0                        # (n_sq, n_contr)
            contrib = sw[t, q] * fv[:, np.newaxis] # (n_sq, n_contr)
            np.add.at(nnz_values, idx[valid], contrib[valid])


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
        self._build_scaling_indices()

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

        return (np.concatenate(inner_list).astype(np.int32),
                np.concatenate(contrib_list).astype(np.int32))

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

            res_idx , var_idx = DOF_IDX[res], DOF_IDX[var]
            global_rows = field_to_global(inner_pts, res_idx, self.grid_idx, self.energy)
            global_cols = field_to_global(contrib_pts, var_idx, self.grid_idx, self.energy)
            self.nnz_global_rows = np.concatenate([self.nnz_global_rows, global_rows])
            self.nnz_global_cols = np.concatenate([self.nnz_global_cols, global_cols])

            block['nnz_idx_start'] = nnz_idx_start
            block['nb_nnz'] = nb_nnz
            nnz_idx_start += nb_nnz

    # ======================================================================
    # COO lookup
    # ======================================================================

    def _build_coo_lookups(self) -> Dict[Tuple[str, str], CooLookup]:
        """Build one CooLookup per (res, var) block.

        Must be called after _build_coo_pattern (needs block['nnz_idx_start']).
        Keys: res_local_idx * max_col + var_local_idx.
        Values: absolute flat nnz index.
        """
        lookups = {}
        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            inner_pts, contrib_pts, nb_nnz = self.conn[(rg, vg)]
            start   = block['nnz_idx_start']
            max_col = int(contrib_pts.max()) + 1 if nb_nnz > 0 else 1
            keys    = inner_pts.astype(np.int64) * max_col + contrib_pts.astype(np.int64)
            values  = np.arange(start, start + nb_nnz, dtype=np.int32)
            order   = np.argsort(keys)
            lookups[(res, var)] = CooLookup(
                sorted_keys=keys[order],
                sorted_values=values[order],
                max_col=max_col,
            )
        return lookups

    def lookup_nnz(self,
                   res: str,
                   res_local_idx: IntArray,
                   var: str,
                   var_local_idx: IntArray,
                   ) -> IntArray:
        """Return absolute flat nnz indices for (res, res_local_idx, var, var_local_idx).

        Returns -1 for pairs not in the sparsity pattern.
        """
        lu    = self.coo_lookups[(res, var)]
        query = res_local_idx.astype(np.int64) * lu.max_col + var_local_idx
        pos   = np.searchsorted(lu.sorted_keys, query)
        pos   = np.minimum(pos, len(lu.sorted_keys) - 1)
        valid = lu.sorted_keys[pos] == query
        return np.where(valid, lu.sorted_values[pos], -1).astype(np.int32)

    # ======================================================================
    # RHS pattern
    # ======================================================================

    def _build_rhs_pattern(self) -> None:
        """Build global RHS row indices for all residuals.

        For each residual, maps local inner-node indices to global DOF indices
        using field_to_global. Result is stored as self.rhs_global_rows.
        """
        rows = []
        for res in self.residuals:
            res_type = DOF_IDX[res]
            if DOF_GRID[res] == 'v':
                l2g = self.grid_idx.l2g_list_v
                nb_inner = self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
            else:
                l2g = self.grid_idx.l2g_list_p
                nb_inner = self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner
            global_field_indices = l2g[:nb_inner]
            rows.append(field_to_global(global_field_indices, res_type,
                                        self.grid_idx, self.energy))

        self.rhs_global_rows: IntArray = np.concatenate(rows)

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
            if DOF_GRID[res] == 'v':
                nb_inner = self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
            else:
                nb_inner = self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner
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
    # Assembly templates
    # ======================================================================

    def build_assembly_templates(self, element: TaylorHoodP2P1) -> None:
        """Precompute AssemblyTemplates for all (res, var, deriv_combo) combinations.

        Must be called once after __init__ and before assemble_matrix / assemble_rhs.
        Templates are keyed by (res, var, deriv_combo) mirroring block_order.
        shape_weighting is shared (by reference) across all (res, var) pairs of the
        same block type — it depends only on grid type and deriv_combo, not on the
        specific variable.
        """
        dx = element.dx
        dy = element.dy
        area = dx * dy / 2.0
        n_tri = 2
        n_quad = element.Quadrature.nb_points    # 3
        weights = element.Quadrature.weights     # (3,)

        # Shape functions and derivative signs, keyed by grid type
        shape_data = {
            'v': (element.P2.N, element.P2.dN_dx / dx, element.P2.dN_dy / dy,
                  np.array(element.P2.der_factor, dtype=float),
                  np.array(element.P2.idx_to_std, dtype=np.int32)),
            'p': (element.P1.N, element.P1.dN_dx / dx, element.P1.dN_dy / dy,
                  np.array(element.P1.der_factor, dtype=float),
                  np.array(element.P1.idx_to_std, dtype=np.int32)),
        }

        # Precompute shape_weighting once per (res_grid, var_grid, deriv_combo)
        sw_cache: Dict[Tuple[str, str, str], NDArray] = {}
        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            for dc in ('none', 'dx_var', 'dy_var', 'dx_test', 'dy_test'):
                key_sw = (rg, vg, dc)
                if key_sw in sw_cache:
                    continue
                N_i, dNdx_i, dNdy_i, der_i, _ = shape_data[rg]
                N_j, dNdx_j, dNdy_j, der_j, _ = shape_data[vg]
                sw_cache[key_sw] = _build_shape_weighting(
                    dc, n_tri, n_quad, weights, area,
                    N_i, dNdx_i, dNdy_i, der_i,
                    N_j, dNdx_j, dNdy_j, der_j,
                )

        # Connectivity arrays from grid_idx
        TO_v = self.grid_idx.sq_TO_inner_v    # (n_sq, 9)
        TO_p = self.grid_idx.sq_TO_inner_p    # (n_sq, 4)
        n_sq = TO_v.shape[0]

        self.templates: Dict[Tuple[str, str, str], AssemblyTemplate] = {}
        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            TO   = TO_v if rg == 'v' else TO_p
            FROM = (self.grid_idx.sq_FROM_padded_v(var) if vg == 'v'
                    else self.grid_idx.sq_FROM_padded_p(var))
            _, _, _, _, std_i = shape_data[rg]
            _, _, _, _, std_j = shape_data[vg]
            n_i = std_i.shape[1]
            n_j = std_j.shape[1]
            n_contr = n_i * n_j

            nnz_index = _build_nnz_index(
                res, var, n_sq, n_tri, n_i, n_j, std_i, std_j,
                TO, FROM, self.lookup_nnz,
            )

            for dc in ('none', 'dx_var', 'dy_var', 'dx_test', 'dy_test'):
                self.templates[(res, var, dc)] = AssemblyTemplate(
                    shape_weighting=sw_cache[(rg, vg, dc)],
                    nnz_index=nnz_index,
                    n_tri=n_tri,
                    n_quad=n_quad,
                    n_contr=n_contr,
                    n_sq=n_sq,
                )

        # Cache element data needed for assemble_rhs
        self._dx = dx
        self._dy = dy
        self._area = area
        self._shape_data = shape_data

    # ======================================================================
    # Matrix and RHS assembly
    # ======================================================================

    def assemble_matrix(self,
                        nnz_values: NDArray,
                        quad_fields: Dict[str, NDArray],
                        terms: list,
                        ) -> None:
        """Accumulate Jacobian term contributions into nnz_values (in-place).

        nnz_values : NDArray, shape (total_nnz,) — pre-zeroed.
        quad_fields : dict mapping field name -> NDArray of shape (n_tri*n_quad, sq_x, sq_y).
        terms : list of NonLinearTerm objects.
        """
        for term in terms:
            dc = _deriv_combo(term)
            dep_vals = [quad_fields[v] for v in term.dep_vars]
            for var in term.dep_vars:
                tmpl = self.templates[(term.res, var, dc)]
                deriv_field = term.evaluate_deriv(var, *dep_vals)  # (n_tri*n_quad, sq_x, sq_y)
                _inject_term(nnz_values, tmpl, deriv_field)

    def assemble_rhs(self,
                     rhs_values: NDArray,
                     quad_fields: Dict[str, NDArray],
                     terms: list,
                     ) -> None:
        """Accumulate residual term contributions into rhs_values (in-place).

        rhs_values : NDArray, shape (total_nb_inner,) — pre-zeroed.
        quad_fields : dict mapping field name -> NDArray of shape (n_tri*n_quad, sq_x, sq_y).
        terms : list of NonLinearTerm objects.
        """
        if not hasattr(self, '_shape_data'):
            raise RuntimeError(
                "build_assembly_templates() must be called before assemble_rhs().")

        # Residual offsets in the flat rhs vector
        res_offsets: Dict[str, int] = {}
        off = 0
        for res in self.residuals:
            res_offsets[res] = off
            if self.res_to_grid[res] == 'v':
                off += self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
            else:
                off += self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner

        TO_v = self.grid_idx.sq_TO_inner_v
        TO_p = self.grid_idx.sq_TO_inner_p

        for term in terms:
            rg = self.res_to_grid[term.res]
            TO = TO_v if rg == 'v' else TO_p
            N_test, dNdx_test, dNdy_test, der_f, std = self._shape_data[rg]

            dc = _deriv_combo(term)
            dep_vals = [quad_fields[v] for v in term.dep_vars]

            if dc in ('dx_var', 'dy_var'):
                # Chain rule: ∂F/∂x = Σ_v (∂F/∂v) * (∂v/∂x)
                prefix = 'd_dx_' if dc == 'dx_var' else 'd_dy_'
                d_scale = self._dx if dc == 'dx_var' else self._dy
                fun_vals = sum(
                    term.evaluate_deriv(v, *dep_vals) * quad_fields[prefix + v] / d_scale
                    for v in term.dep_vars
                )
                Phi_test = N_test
                use_der_sign = False
            elif dc == 'dx_test':
                fun_vals = term.evaluate(*dep_vals)
                Phi_test = dNdx_test
                use_der_sign = True
            elif dc == 'dy_test':
                fun_vals = term.evaluate(*dep_vals)
                Phi_test = dNdy_test
                use_der_sign = True
            else:
                fun_vals = term.evaluate(*dep_vals)
                Phi_test = N_test
                use_der_sign = False

            n_quad = Phi_test.shape[0]
            R_local = np.zeros(self.grid_idx.Nx_v_inner * self.grid_idx.Ny_v_inner
                               if rg == 'v'
                               else self.grid_idx.Nx_p_inner * self.grid_idx.Ny_p_inner)

            for t in range(2):
                sign_t = der_f[t] if use_der_sign else 1.0
                TO_tri = TO[:, std[t]]           # (n_sq, n_nodes)
                for q in range(n_quad):
                    fv = fun_vals[t * n_quad + q].ravel()   # (n_sq,)
                    n_nodes = TO_tri.shape[1]
                    for i in range(n_nodes):
                        inner_idx = TO_tri[:, i]
                        valid = inner_idx >= 0
                        if valid.any():
                            np.add.at(R_local, inner_idx[valid],
                                      sign_t * self._area * Phi_test[q, i] * fv[valid])

            res_off = res_offsets[term.res]
            rhs_values[res_off:res_off + len(R_local)] += R_local
