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

"""
FEM Assembly — Taylor-Hood P2P1, sparsity structure.

Two-grid architecture: P2 for mass flux (jx, jy), P1 for pressure (rho, e).
There are four block types depending on which grid the residual and variable
live on:

    v→Rv  : TO=sq_TO_inner_v,  FROM=sq_FROM_padded_v   (6×6 nodes per tri)
    p→Rv  : TO=sq_TO_inner_v,  FROM=sq_FROM_padded_p   (6×3 nodes per tri)
    v→Rp  : TO=sq_TO_inner_p,  FROM=sq_FROM_padded_v   (3×6 nodes per tri)
    p→Rp  : TO=sq_TO_inner_p,  FROM=sq_FROM_padded_p   (3×3 nodes per tri)

Each block type has its own nnz count and its own COO lookup structure.
The global ordering uses field_to_global() from global_matrix.py.

Part 1 — sparsity structure:
    - _build_block_connectivity_vv/vp/pv/pp
    - _build_coo_pattern        (local + global COO arrays)
    - _build_coo_lookup         (per block type, for template construction)
    - _build_rhs_pattern        (global RHS row indices)

Part 2 — assembly templates and value injection:
    - _build_assembly_templates    (shape-weighting + nnz-index per block×deriv)
    - assemble_matrix              (O(nnz) tangent matrix assembly)
    - assemble_rhs                 (residual vector assembly)

Derivative convention
---------------------
dN_dx / dN_dy from elements.py are reference-coordinate derivatives
(∂N/∂ξ, ∂N/∂η).  Physical gradient = reference derivative / physical size:
    ∂u/∂x = (dN_dx @ u_nodes) / dx
    ∂u/∂y = (dN_dy @ u_nodes) / dy

The shape-weighting for derivative terms already bakes in the 1/dx or 1/dy
factor so that assemble_matrix only needs to multiply by the quad field values.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .elements import TaylorHoodP2P1
from .grid_index import GridIndexManager
from .global_matrix import field_to_global

if TYPE_CHECKING:
    from ..parallel import DomainDecomposition

NDArray  = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]

# Triangle-to-square-node index mapping from TaylorHoodP2P1:
#   tri0 → square node indices [0,1,2,4,5,6]   (6 P2 nodes)
#   tri1 → square node indices [3,2,1,8,7,6]   (6 P2 nodes)
_IDX_TO_STD_P2 = TaylorHoodP2P1.P2.idx_to_std   # shape (2, 6)
# P1 triangle → square node indices [0,1,2] and [3,2,1]
_IDX_TO_STD_P1 = TaylorHoodP2P1.P1.idx_to_std   # shape (2, 3)

# Convenience: number of nodes per triangle for each element type
_NB_NODES_P2 = TaylorHoodP2P1.P2.nb_nodes   # 6
_NB_NODES_P1 = TaylorHoodP2P1.P1.nb_nodes   # 3


@dataclass
class P2P1AssemblyInfo:
    """Assembly info for the linear solver (duck-type compatible with old PETScAssemblyInfo).

    Attributes
    ----------
    local_size : int
        Total residual vector length on this process.
    mat_global_rows, mat_global_cols : IntArray
        Global (row, col) indices for COO Jacobian values.
    rhs_global_rows : IntArray
        Global row indices for residual vector entries.
    """
    local_size:      int
    mat_global_rows: "IntArray"
    mat_global_cols: "IntArray"
    rhs_global_rows: "IntArray"


# ---------------------------------------------------------------------------
# Block-type identifier
# ---------------------------------------------------------------------------

# Block types as (residual_grid, variable_grid): 'v'=P2, 'p'=P1
BLOCK_VV = ('v', 'v')   # velocity residual, velocity variable
BLOCK_PV = ('p', 'v')   # pressure residual, velocity variable
BLOCK_VP = ('v', 'p')   # velocity residual, pressure variable
BLOCK_PP = ('p', 'p')   # pressure residual, pressure variable


@dataclass
class CooLookup:
    """Sorted-key binary-search structure for one block type's COO pattern.

    Attributes
    ----------
    sorted_keys : IntArray
        Sorted (row*max_col + col) keys for block (res=0, var=0).
    sorted_indices : IntArray
        Positions in the COO array corresponding to sorted_keys.
    max_col : int
        Column stride used to compute keys.
    nnz_per_block : int
        Number of non-zeros in one (residual, variable) block.
    """
    sorted_keys:    IntArray
    sorted_indices: IntArray
    max_col:        int
    nnz_per_block:  int


# Derivative combo type for AssemblyTemplate:
#   'none'    — no derivative on either function
#   'dx_var'  — ∂/∂x on the trial (variable) function, N_i plain
#   'dy_var'  — ∂/∂y on the trial function
#   'dx_test' — ∂/∂x on the test (residual) function, N_j plain
#   'dy_test' — ∂/∂y on the test function
DerivCombo = str


@dataclass
class AssemblyTemplate:
    """Precomputed injection template for one (block_type, deriv_combo) combination.

    Enables O(nnz) assembly for a given term type:

        nnz_values[nnz_index[k]] += quad_values[quad_flat_idx[k]] * shape_weighting[k]

    where the loop over k is implicit (vectorised with np.add.at or equivalent).

    The structure splits the computation into:
    - shape_weighting: pure FEM weights, same for all squares. Length = nb_tri * nb_quad * n_i * n_j.
    - nnz_index:       geometry-specific COO positions. Length = nb_sq * nb_tri * n_i * n_j.
                       -1 means skip (Dirichlet or out-of-domain node).
    - quad_flat_idx:   which quad value each (sq, tri, i, j) entry pulls from.
                       Length = nb_sq * nb_tri * n_i * n_j.

    The shape_weighting and quad_flat_idx are aligned with one another in a
    tri-major, quad-major, i-major, j-major order so that the assembly
    inner product can be written as:

        for each square s (vectorised over s):
            for each tri t:
                for each quad pt q:
                    for each i, j:
                        contrib = w_tj * f_q[s] * SW[t,q,i,j]
        folded = sum over q per (t,i,j)

    Attributes
    ----------
    shape_weighting : NDArray, shape (nb_tri * nb_quad * n_i * n_j,)
        Quadrature weights baked in, same for every square.
    nnz_index : IntArray, shape (nb_tri * n_i * n_j * nb_sq,)
        COO positions for block (res=0, var=0).  -1 = skip.
        Order: (tri, i, j, sq) — contr-major, sq-fast.
    quad_flat_idx : IntArray, shape (nb_sq * nb_tri * nb_quad * n_i * n_j,)
        Flat index into quad field array (shape 6*nb_sq → flattened).
    nb_tri : int
    nb_quad : int
    n_i : int  (TO nodes per tri)
    n_j : int  (FROM nodes per tri)
    nb_sq : int
    nnz_per_block : int
    """
    shape_weighting: NDArray
    nnz_index:       IntArray
    quad_flat_idx:   IntArray
    nb_tri:          int
    nb_quad:         int
    n_i:             int
    n_j:             int
    nb_sq:           int
    nnz_per_block:   int

    @property
    def n_contr(self) -> int:
        return self.n_i * self.n_j


class Assembly:
    """Precomputed sparsity structure for P2P1 assembly.

    Parameters
    ----------
    grid_idx : GridIndexManager
    variables : list of str
        All variable names in block order, e.g. ['jx', 'jy', 'rho'].
    residuals : list of str
        All residual names in block order, e.g. ['Rjx', 'Rjy', 'Rrho'].
    res_to_grid : dict
        Maps each residual name to 'v' or 'p'.
    var_to_grid : dict
        Maps each variable name to 'v' or 'p'.
    cols_v : int
        Number of columns in the fine (velocity) grid — used by field_to_global.
    M : int
        Number of pressure columns — used by field_to_global.
    energy : bool
        Whether energy is active — used by field_to_global.
    """

    def __init__(self,
                 grid_idx: GridIndexManager,
                 variables: List[str],
                 residuals: List[str],
                 res_to_grid: Dict[str, str],
                 var_to_grid: Dict[str, str],
                 cols_v: int,
                 M: int,
                 energy: bool = False,
                 ) -> None:
        self._grid_idx   = grid_idx
        self._variables  = variables
        self._residuals  = residuals
        self._res_to_grid = res_to_grid
        self._var_to_grid = var_to_grid
        self._cols_v     = cols_v
        self._M          = M
        self._energy     = energy

        nb_res  = len(residuals)
        nb_vars = len(variables)

        # ------------------------------------------------------------------
        # 1. Build (inner_pts, contrib_pts) for each block type
        # ------------------------------------------------------------------
        conn = self._build_connectivity_stencil(grid_idx, variables, var_to_grid)
        # conn[(res_grid, var_grid)] = (inner_pts, contrib_pts, nnz_per_block)

        # ------------------------------------------------------------------
        # 2. Build the full COO pattern (local + global)
        # ------------------------------------------------------------------
        self._build_coo_pattern(grid_idx, variables, residuals,
                                res_to_grid, var_to_grid, conn,
                                cols_v, M, energy, nb_res, nb_vars)

        # ------------------------------------------------------------------
        # 3. Build per-block-type COO lookup (for template construction)
        # ------------------------------------------------------------------
        self.coo_lookups: Dict[Tuple[str, str], CooLookup] = {}
        for block_type, (inner_pts, contrib_pts, nnz) in conn.items():
            self.coo_lookups[block_type] = self._build_coo_lookup(
                inner_pts, contrib_pts, nnz)

        # ------------------------------------------------------------------
        # 4. Build RHS pattern
        # ------------------------------------------------------------------
        self._build_rhs_pattern(grid_idx, residuals, res_to_grid,
                                cols_v, M, energy)

    # ======================================================================
    # Block connectivity — stencil-based
    # ======================================================================

    @staticmethod
    def _build_connectivity_stencil(
            grid_idx: GridIndexManager,
            variables: List[str],
            var_to_grid: Dict[str, str],
    ) -> Dict[Tuple[str, str], Tuple[IntArray, IntArray, int]]:
        """Build connectivity for all four block types using stencils.

        For each inner node on the residual grid, apply the appropriate stencil
        offsets (from TaylorHoodP2P1) to the contributor mask to find all valid
        (inner, contrib) pairs.  Stencil selection:

          vv : origin parity determines stencil (even-even / odd-odd /
               even-odd / odd-even on the fine grid).
          vp : same origin-parity stencils, but offsets filtered to even-even
               only (pressure lives only on even-even fine-grid nodes).
          pv : origins are always even-even; use stencil_even_even unfiltered
               (all v-node offsets reachable from an even-even origin).
          pp : origins are even-even; filter stencil_even_even to even-even
               offsets only.
        """
        from .elements import TaylorHoodP2P1

        stencils = {
            (0, 0): np.array(TaylorHoodP2P1.stencil_even_even, dtype=np.int32),
            (1, 1): np.array(TaylorHoodP2P1.stencil_odd_odd,   dtype=np.int32),
            (0, 1): np.array(TaylorHoodP2P1.stencil_even_odd,  dtype=np.int32),
            (1, 0): np.array(TaylorHoodP2P1.stencil_odd_even,  dtype=np.int32),
        }

        def _p_reachable(s: IntArray, px: int, py: int) -> IntArray:
            """Keep offsets where origin+(dx,dy) is even-even (a p-node).

            Destination is even-even iff (origin_x+dx)%2==0 and (origin_y+dy)%2==0,
            i.e. dx%2==px and dy%2==py.
            """
            mask = (s[:, 0] % 2 == px) & (s[:, 1] % 2 == py)
            return s[mask]

        # VP stencils per origin parity: keep only p-reachable offsets
        stencils_vp = {k: _p_reachable(v, k[0], k[1])
                       for k, v in stencils.items()}

        stencil_pv = stencils[(0, 0)]                # pv: even-even origin, all v neighbours
        stencil_pp = _p_reachable(stencils[(0, 0)], 0, 0)  # pp: even-even → even-even only

        # Representative variables for each grid
        rep_v = next(v for v in variables if var_to_grid[v] == 'v')
        rep_p = next(v for v in variables if var_to_grid[v] == 'p')

        m_inner_v  = grid_idx.index_mask_inner_local_v
        m_padded_v = grid_idx.index_mask_padded_local_v(rep_v)
        m_inner_p  = grid_idx.index_mask_inner_local_p
        m_padded_p = grid_idx.index_mask_padded_local_p(rep_p)

        Nx_v_pad = grid_idx.Nx_v_padded
        Ny_v_pad = grid_idx.Ny_v_padded

        result: Dict[Tuple[str, str], Tuple[IntArray, IntArray, int]] = {}

        # ------------------------------------------------------------------
        # vv and vp: iterate over all inner v-nodes
        # ------------------------------------------------------------------
        inner_v_pos = np.argwhere(m_inner_v >= 0)  # shape (nb_inner_v, 2)
        ix_v = inner_v_pos[:, 0]
        iy_v = inner_v_pos[:, 1]
        inner_v_idx = m_inner_v[ix_v, iy_v]

        # Parity of each inner v-node in fine-grid coordinates
        # (coordinates inside the padded array; ghost depth = 2)
        px = ix_v % 2   # 0 = even, 1 = odd
        py = iy_v % 2

        vv_inner_list:  List[IntArray] = []
        vv_contrib_list: List[IntArray] = []
        vp_inner_list:  List[IntArray] = []
        vp_contrib_list: List[IntArray] = []

        for parity, stencil_vv, stencil_vp in [
            ((0, 0), stencils[(0, 0)], stencils_vp[(0, 0)]),
            ((1, 1), stencils[(1, 1)], stencils_vp[(1, 1)]),
            ((0, 1), stencils[(0, 1)], stencils_vp[(0, 1)]),
            ((1, 0), stencils[(1, 0)], stencils_vp[(1, 0)]),
        ]:
            sel = (px == parity[0]) & (py == parity[1])
            if not sel.any():
                continue
            ix_sel = ix_v[sel]
            iy_sel = iy_v[sel]
            idx_sel = inner_v_idx[sel]

            for dx, dy in stencil_vv:
                nx = ix_sel + dx
                ny = iy_sel + dy
                in_bounds = (nx >= 0) & (nx < Nx_v_pad) & \
                            (ny >= 0) & (ny < Ny_v_pad)
                nx = nx[in_bounds]; ny = ny[in_bounds]
                contrib = m_padded_v[nx, ny]
                valid = contrib >= 0
                if valid.any():
                    vv_inner_list.append(idx_sel[in_bounds][valid])
                    vv_contrib_list.append(contrib[valid])

            for dx, dy in stencil_vp:
                nx = ix_sel + dx
                ny = iy_sel + dy
                in_bounds = (nx >= 0) & (nx < Nx_v_pad) & \
                            (ny >= 0) & (ny < Ny_v_pad)
                nx = nx[in_bounds]; ny = ny[in_bounds]
                contrib = m_padded_p[nx // 2, ny // 2]
                valid = contrib >= 0
                if valid.any():
                    vp_inner_list.append(idx_sel[in_bounds][valid])
                    vp_contrib_list.append(contrib[valid])

        for block, inner_list, contrib_list in [
            (BLOCK_VV, vv_inner_list, vv_contrib_list),
            (BLOCK_VP, vp_inner_list, vp_contrib_list),
        ]:
            if inner_list:
                inner  = np.concatenate(inner_list).astype(np.int32)
                contrib = np.concatenate(contrib_list).astype(np.int32)
                pairs  = np.unique(np.column_stack([inner, contrib]), axis=0)
                result[block] = (pairs[:, 0], pairs[:, 1], len(pairs))
            else:
                result[block] = (np.empty(0, np.int32),
                                 np.empty(0, np.int32), 0)

        # ------------------------------------------------------------------
        # pv and pp: iterate over all inner p-nodes
        # ------------------------------------------------------------------
        inner_p_pos = np.argwhere(m_inner_p >= 0)
        ix_p = inner_p_pos[:, 0]
        iy_p = inner_p_pos[:, 1]
        inner_p_idx = m_inner_p[ix_p, iy_p]

        # Corresponding fine-grid coordinates (p-node i,j → fine 2i, 2j,
        # offset by ghost depth difference: p ghost=1 → fine coords = 2*(ix_p-1)+2)
        ix_fine = (ix_p - 1) * 2 + 2   # map padded-p coord to padded-v coord
        iy_fine = (iy_p - 1) * 2 + 2

        Nx_p_pad = grid_idx.Nx_p_padded
        Ny_p_pad = grid_idx.Ny_p_padded

        pv_inner_list:  List[IntArray] = []
        pv_contrib_list: List[IntArray] = []
        pp_inner_list:  List[IntArray] = []
        pp_contrib_list: List[IntArray] = []

        for dx, dy in stencil_pv:   # all v-offsets from even-even
            nx_fine = ix_fine + dx
            ny_fine = iy_fine + dy
            in_bounds = (nx_fine >= 0) & (nx_fine < Nx_v_pad) & \
                        (ny_fine >= 0) & (ny_fine < Ny_v_pad)
            nxf = nx_fine[in_bounds]; nyf = ny_fine[in_bounds]
            contrib = m_padded_v[nxf, nyf]
            valid = contrib >= 0
            if valid.any():
                pv_inner_list.append(inner_p_idx[in_bounds][valid])
                pv_contrib_list.append(contrib[valid])

        for dx, dy in stencil_pp:   # even-even offsets only → p neighbours
            nx_fine = ix_fine + dx
            ny_fine = iy_fine + dy
            in_bounds = (nx_fine >= 0) & (nx_fine < Nx_v_pad) & \
                        (ny_fine >= 0) & (ny_fine < Ny_v_pad)
            nxf = nx_fine[in_bounds]; nyf = ny_fine[in_bounds]
            contrib = m_padded_p[nxf // 2, nyf // 2]
            valid = contrib >= 0
            if valid.any():
                pp_inner_list.append(inner_p_idx[in_bounds][valid])
                pp_contrib_list.append(contrib[valid])

        for block, inner_list, contrib_list in [
            (BLOCK_PV, pv_inner_list, pv_contrib_list),
            (BLOCK_PP, pp_inner_list, pp_contrib_list),
        ]:
            if inner_list:
                inner  = np.concatenate(inner_list).astype(np.int32)
                contrib = np.concatenate(contrib_list).astype(np.int32)
                pairs  = np.unique(np.column_stack([inner, contrib]), axis=0)
                result[block] = (pairs[:, 0], pairs[:, 1], len(pairs))
            else:
                result[block] = (np.empty(0, np.int32),
                                 np.empty(0, np.int32), 0)

        return result

    # ======================================================================
    # Block connectivity — element-based (reference implementation)
    # ======================================================================

    @staticmethod
    def _connect_one_block(TO: IntArray,
                           FROM: IntArray,
                           idx_to_std_TO: IntArray,
                           idx_to_std_FROM: IntArray,
                           ) -> Tuple[IntArray, IntArray]:
        """Build (inner_pts, contrib_pts) for one block type.

        Iterates over all squares and all triangle pairs, emitting one
        (inner, contrib) pair for every (TO_node_i, FROM_node_j) combination
        where both indices are >= 0 (i.e. neither is a Dirichlet ghost or
        out-of-domain node).

        Parameters
        ----------
        TO : IntArray, shape (nb_sq, nb_TO_nodes)
            Residual (inner) node indices per square.
        FROM : IntArray, shape (nb_sq, nb_FROM_nodes)
            Contributor node indices per square.
        idx_to_std_TO : IntArray, shape (2, nb_nodes_per_tri_TO)
            Triangle-to-square-node mapping for the TO (residual) element.
        idx_to_std_FROM : IntArray, shape (2, nb_nodes_per_tri_FROM)
            Triangle-to-square-node mapping for the FROM (variable) element.
        """
        inner_list  = []
        contrib_list = []

        for t in range(2):
            TO_tri   = TO[:, idx_to_std_TO[t]]     # (nb_sq, nb_TO_nodes_tri)
            FROM_tri = FROM[:, idx_to_std_FROM[t]]  # (nb_sq, nb_FROM_nodes_tri)

            nb_i = TO_tri.shape[1]
            nb_j = FROM_tri.shape[1]

            for i in range(nb_i):
                for j in range(nb_j):
                    valid = (TO_tri[:, i] >= 0) & (FROM_tri[:, j] >= 0)
                    if valid.any():
                        inner_list.append(TO_tri[valid, i])
                        contrib_list.append(FROM_tri[valid, j])

        if inner_list:
            return np.concatenate(inner_list), np.concatenate(contrib_list)
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    @staticmethod
    def _build_all_connectivity(
            grid_idx: GridIndexManager,
            variables: List[str],
            var_to_grid: Dict[str, str],
    ) -> Dict[Tuple[str, str], Tuple[IntArray, IntArray, int]]:
        """Build connectivity for all four block types.

        Returns a dict mapping (res_grid, var_grid) -> (inner_pts, contrib_pts, nnz).
        Uses a representative variable for each grid type (connectivity is the
        same for all variables on the same grid, ignoring Neumann/Dirichlet
        differences which only affect whether specific entries are -1).
        """
        # Representative variable for each grid (any variable on that grid)
        rep_v = next(v for v in variables if var_to_grid[v] == 'v')
        rep_p = next(v for v in variables if var_to_grid[v] == 'p')

        TO_v   = grid_idx.sq_TO_inner_v
        FROM_v = grid_idx.sq_FROM_padded_v(rep_v)
        TO_p   = grid_idx.sq_TO_inner_p
        FROM_p = grid_idx.sq_FROM_padded_p(rep_p)

        result = {}
        for (res_grid, var_grid), TO, FROM, std_TO, std_FROM in [
            (BLOCK_VV, TO_v, FROM_v, _IDX_TO_STD_P2, _IDX_TO_STD_P2),
            (BLOCK_VP, TO_v, FROM_p, _IDX_TO_STD_P2, _IDX_TO_STD_P1),
            (BLOCK_PV, TO_p, FROM_v, _IDX_TO_STD_P1, _IDX_TO_STD_P2),
            (BLOCK_PP, TO_p, FROM_p, _IDX_TO_STD_P1, _IDX_TO_STD_P1),
        ]:
            inner, contrib = Assembly._connect_one_block(TO, FROM, std_TO, std_FROM)
            # Deduplicate: multiple triangles can produce the same (i,j) pair
            pairs = np.unique(np.column_stack([inner, contrib]), axis=0)
            inner_dedup  = pairs[:, 0].astype(np.int32)
            contrib_dedup = pairs[:, 1].astype(np.int32)
            result[(res_grid, var_grid)] = (inner_dedup, contrib_dedup,
                                            len(inner_dedup))

        return result

    # ======================================================================
    # COO pattern
    # ======================================================================

    def _build_coo_pattern(self,
                           grid_idx: GridIndexManager,
                           variables: List[str],
                           residuals: List[str],
                           res_to_grid: Dict[str, str],
                           var_to_grid: Dict[str, str],
                           conn: Dict,
                           cols_v: int,
                           M: int,
                           energy: bool,
                           nb_res: int,
                           nb_vars: int,
                           ) -> None:
        """Build full COO arrays covering all (residual, variable) blocks."""

        # Count total nnz across all blocks
        total_nnz = 0
        block_nnz: Dict[Tuple[int, int], int] = {}
        for ri, res in enumerate(residuals):
            for vi, var in enumerate(variables):
                bt = (res_to_grid[res], var_to_grid[var])
                nnz = conn[bt][2]
                block_nnz[(ri, vi)] = nnz
                total_nnz += nnz

        local_rows  = np.empty(total_nnz, dtype=np.int32)
        local_cols  = np.empty(total_nnz, dtype=np.int32)
        global_rows = np.empty(total_nnz, dtype=np.int32)
        global_cols = np.empty(total_nnz, dtype=np.int32)

        # Residual type codes for field_to_global
        # 0=jx, 1=jy, 2=rho, 3=e  — caller must ensure naming convention
        res_type_map = self._make_res_type_map(residuals)
        var_type_map = self._make_res_type_map(variables)

        l2g_v = grid_idx.l2g_list_v
        l2g_p = grid_idx.l2g_list_p

        nb_inner_v = grid_idx.Nx_v_inner * grid_idx.Ny_v_inner
        nb_inner_p = grid_idx.Nx_p_inner * grid_idx.Ny_p_inner
        nb_contrib_v = grid_idx.nb_contributors_v
        nb_contrib_p = grid_idx.nb_contributors_p

        res_block_idx = np.empty(total_nnz, dtype=np.int8)
        var_block_idx = np.empty(total_nnz, dtype=np.int8)

        # Precompute cumulative residual / variable offsets in the flat vector
        res_offset = {}
        off = 0
        for res in residuals:
            res_offset[res] = off
            off += nb_inner_v if res_to_grid[res] == 'v' else nb_inner_p

        var_offset = {}
        off = 0
        for var in variables:
            var_offset[var] = off
            off += nb_contrib_v if var_to_grid[var] == 'v' else nb_contrib_p

        idx = 0
        for ri, res in enumerate(residuals):
            for vi, var in enumerate(variables):
                bt = (res_to_grid[res], var_to_grid[var])
                inner_pts, contrib_pts, nnz = conn[bt]
                end = idx + nnz

                # Local ordering: flat block-major with per-block sizes
                res_grid = res_to_grid[res]
                var_grid = var_to_grid[var]
                nb_inner  = nb_inner_v  if res_grid == 'v' else nb_inner_p
                nb_contrib = nb_contrib_v if var_grid == 'v' else nb_contrib_p
                local_rows[idx:end] = inner_pts  + res_offset[res]
                local_cols[idx:end] = contrib_pts + var_offset[var]

                res_block_idx[idx:end] = ri
                var_block_idx[idx:end] = vi

                # Global ordering via field_to_global
                l2g_res = l2g_v if res_grid == 'v' else l2g_p
                l2g_var = l2g_v if var_grid == 'v' else l2g_p
                g_inner  = l2g_res[inner_pts]
                g_contrib = l2g_var[contrib_pts]

                res_type = res_type_map[res]
                var_type = var_type_map[var]

                global_rows[idx:end] = np.array([
                    field_to_global(int(k), res_type, cols_v, M, energy)
                    for k in g_inner], dtype=np.int32)
                global_cols[idx:end] = np.array([
                    field_to_global(int(k), var_type, cols_v, M, energy)
                    for k in g_contrib], dtype=np.int32)

                idx = end

        self.nnz:           int      = total_nnz
        self.local_rows:    IntArray = local_rows
        self.local_cols:    IntArray = local_cols
        self.global_rows:   IntArray = global_rows
        self.global_cols:   IntArray = global_cols
        self.res_block_idx: IntArray = res_block_idx
        self.var_block_idx: IntArray = var_block_idx
        self._block_nnz:    Dict     = block_nnz
        self._conn:         Dict     = conn

    @staticmethod
    def _make_res_type_map(names: List[str]) -> Dict[str, int]:
        """Map residual/variable names to field_to_global res_type codes.

        Convention: jx->0, jy->1, rho->2, e->3.
        """
        code = {'jx': 0, 'jy': 1, 'rho': 2, 'e': 3}
        # Also handle residual names like 'Rjx', 'Rjy', 'Rrho', 'Re'
        # and equation-style names 'momentum_x', 'momentum_y', 'mass', 'energy'
        eq_alias = {
            'momentum_x': 'jx', 'momentum_y': 'jy',
            'mass': 'rho', 'energy': 'e',
        }
        result = {}
        for name in names:
            if name in eq_alias:
                result[name] = code[eq_alias[name]]
                continue
            stripped = name.lstrip('R')
            if stripped in code:
                result[name] = code[stripped]
            elif name in code:
                result[name] = code[name]
            else:
                raise ValueError(f"Unknown field name: {name!r}. "
                                 f"Expected one of {list(code)} or R-prefixed.")
        return result

    # ======================================================================
    # COO lookup
    # ======================================================================

    @staticmethod
    def _build_coo_lookup(inner_pts: IntArray,
                          contrib_pts: IntArray,
                          nnz_per_block: int,
                          ) -> CooLookup:
        """Build sorted-key binary-search lookup for one block type.

        Only indexes block (res=0, var=0) since callers add block offsets.
        """
        if nnz_per_block == 0:
            return CooLookup(
                sorted_keys=np.empty(0, dtype=np.int64),
                sorted_indices=np.empty(0, dtype=np.int32),
                max_col=1,
                nnz_per_block=0,
            )
        max_col = int(contrib_pts.max()) + 1
        keys = inner_pts.astype(np.int64) * max_col + contrib_pts
        order = np.argsort(keys)
        return CooLookup(
            sorted_keys=keys[order],
            sorted_indices=order.astype(np.int32),
            max_col=max_col,
            nnz_per_block=nnz_per_block,
        )

    def coo_lookup(self,
                   block_type: Tuple[str, str],
                   row_local: IntArray,
                   col_local: IntArray,
                   ) -> IntArray:
        """Vectorized COO index lookup for a given block type.

        Returns the position in the block-(0,0) COO slice for each
        (row, col) pair, or -1 if the pair is not in the pattern.
        """
        lu = self.coo_lookups[block_type]
        if lu.nnz_per_block == 0:
            return np.full(len(row_local), -1, dtype=np.int32)
        query = row_local.astype(np.int64) * lu.max_col + col_local
        pos = np.searchsorted(lu.sorted_keys, query)
        pos = np.minimum(pos, len(lu.sorted_keys) - 1)
        valid = lu.sorted_keys[pos] == query
        return np.where(valid, lu.sorted_indices[pos], -1).astype(np.int32)

    # ======================================================================
    # RHS pattern
    # ======================================================================

    def _build_rhs_pattern(self,
                           grid_idx: GridIndexManager,
                           residuals: List[str],
                           res_to_grid: Dict[str, str],
                           cols_v: int,
                           M: int,
                           energy: bool,
                           ) -> None:
        """Build global RHS row indices for all residuals."""
        res_type_map = self._make_res_type_map(residuals)
        rows = []
        local_rows_list = []
        rhs_res_block = []
        offset = 0
        for ri, res in enumerate(residuals):
            res_grid = res_to_grid[res]
            res_type = res_type_map[res]
            if res_grid == 'v':
                l2g = grid_idx.l2g_list_v
                nb_inner = grid_idx.Nx_v_inner * grid_idx.Ny_v_inner
            else:
                l2g = grid_idx.l2g_list_p
                nb_inner = grid_idx.Nx_p_inner * grid_idx.Ny_p_inner
            g_pts = l2g[:nb_inner]
            rows.append(np.array([
                field_to_global(int(k), res_type, cols_v, M, energy)
                for k in g_pts], dtype=np.int32))
            local_rows_list.append(np.arange(offset, offset + nb_inner, dtype=np.int32))
            rhs_res_block.append(np.full(nb_inner, ri, dtype=np.int8))
            offset += nb_inner

        self.rhs_global_rows:    IntArray = np.concatenate(rows)
        self.rhs_local_rows:     IntArray = np.concatenate(local_rows_list)
        self.rhs_res_block_idx:  IntArray = np.concatenate(rhs_res_block)

    # ======================================================================
    # Linear solver info
    # ======================================================================

    def get_petsc_info(self, res_size: int) -> "P2P1AssemblyInfo":
        """Return assembly info for the linear solver.

        Parameters
        ----------
        res_size : int
            Total residual vector length (sum of nb_inner over all residuals).
        """
        return P2P1AssemblyInfo(
            local_size=res_size,
            mat_global_rows=self.global_rows,
            mat_global_cols=self.global_cols,
            rhs_global_rows=self.rhs_global_rows,
        )

    # ======================================================================
    # Assembly templates — Part 2
    # ======================================================================

    def build_assembly_templates(self,
                                  grid_idx: GridIndexManager,
                                  variables: List[str],
                                  residuals: List[str],
                                  res_to_grid: Dict[str, str],
                                  var_to_grid: Dict[str, str],
                                  elements: "TaylorHoodP2P1",
                                  ) -> None:
        """Precompute AssemblyTemplates for all (block_type, deriv_combo, var) combinations.

        One template is built per (block_type, deriv_combo, variable) triple so
        that each variable's Neumann/Dirichlet BC is correctly reflected in the
        FROM index array.  The shape_weighting is variable-independent (pure FEM
        weights); only the nnz_index and quad_flat_idx differ between variables
        on the same grid when their BCs differ.

        Must be called once before assemble_matrix / assemble_rhs.

        Parameters
        ----------
        grid_idx : GridIndexManager
        variables, residuals : list of str
        res_to_grid, var_to_grid : dict
        elements : TaylorHoodP2P1
            Instance carrying P1.N, P1.dN_dx, etc. and dx, dy.
        """
        self.assembly_templates: Dict[Tuple[Tuple[str,str], DerivCombo, str],
                                      AssemblyTemplate] = {}
        dx = elements.dx
        dy = elements.dy

        # Quadrature data
        from .elements import Quadrature3Points
        q3 = Quadrature3Points
        nb_quad = q3.nb_points           # 3
        weights = q3.weights             # (3,)

        # Shape function matrices (nb_quad, nb_nodes)
        N_p2    = elements.P2.N          # (3, 6)
        dNdx_p2 = elements.P2.dN_dx     # (3, 6) — reference-coord derivative
        dNdy_p2 = elements.P2.dN_dy     # (3, 6)
        N_p1    = elements.P1.N          # (3, 3)
        dNdx_p1 = elements.P1.dN_dx     # (3, 3)
        dNdy_p1 = elements.P1.dN_dy     # (3, 3)
        der_factor_p2 = np.array(elements.P2.der_factor, dtype=float)  # [1, -1]
        der_factor_p1 = np.array(elements.P1.der_factor, dtype=float)  # [1, -1]

        # Triangle area |J| = dx*dy/2
        area = dx * dy / 2.0
        nb_tri = 2

        # Map block type → (N_i, dNdx_i, dNdy_i, der_i, N_j, dNdx_j, dNdy_j, der_j)
        # Shape functions depend only on the grid type (v/p), not the variable.
        block_shapes = {
            BLOCK_VV: (N_p2, dNdx_p2, dNdy_p2, der_factor_p2,
                       N_p2, dNdx_p2, dNdy_p2, der_factor_p2),
            BLOCK_VP: (N_p2, dNdx_p2, dNdy_p2, der_factor_p2,
                       N_p1, dNdx_p1, dNdy_p1, der_factor_p1),
            BLOCK_PV: (N_p1, dNdx_p1, dNdy_p1, der_factor_p1,
                       N_p2, dNdx_p2, dNdy_p2, der_factor_p2),
            BLOCK_PP: (N_p1, dNdx_p1, dNdy_p1, der_factor_p1,
                       N_p1, dNdx_p1, dNdy_p1, der_factor_p1),
        }

        TO_v = grid_idx.sq_TO_inner_v   # (nb_sq, 9)  — same for all v-residuals
        TO_p = grid_idx.sq_TO_inner_p   # (nb_sq, 4)  — same for all p-residuals

        nb_sq = grid_idx.nb_sq_p

        for var in variables:
            vg = var_to_grid[var]
            # FROM array is variable-specific: Neumann forwarding differs per var.
            FROM_var = (grid_idx.sq_FROM_padded_v(var) if vg == 'v'
                        else grid_idx.sq_FROM_padded_p(var))

            for block_type in [BLOCK_VV, BLOCK_VP, BLOCK_PV, BLOCK_PP]:
                _, bt_vg = block_type          # variable-grid side of this block
                if bt_vg != vg:
                    continue                   # this var doesn't belong to this block

                (N_i, dNdx_i, dNdy_i, der_i,
                 N_j, dNdx_j, dNdy_j, der_j) = block_shapes[block_type]
                TO    = TO_v if block_type[0] == 'v' else TO_p
                std_i = _IDX_TO_STD_P2 if block_type[0] == 'v' else _IDX_TO_STD_P1
                std_j = _IDX_TO_STD_P2 if vg == 'v' else _IDX_TO_STD_P1
                n_i   = std_i.shape[1]
                n_j   = std_j.shape[1]
                lu    = self.coo_lookups[block_type]

                for deriv_combo in ['none', 'dx_var', 'dy_var', 'dx_test', 'dy_test']:
                    key = (block_type, deriv_combo, var)
                    tmpl = self._build_one_template(
                        deriv_combo, nb_tri, nb_quad, weights, area, dx, dy,
                        N_i, dNdx_i, dNdy_i, der_i,
                        N_j, dNdx_j, dNdy_j, der_j,
                        n_i, n_j, nb_sq, nb_sq,
                        TO, FROM_var, std_i, std_j, lu,
                        grid_idx.sq_per_col_v,
                    )
                    self.assembly_templates[key] = tmpl

        self._store_element_data(elements)

    @staticmethod
    def _build_one_template(
            deriv_combo: DerivCombo,
            nb_tri: int, nb_quad: int, weights: NDArray,
            area: float, dx: float, dy: float,
            N_i: NDArray, dNdx_i: NDArray, dNdy_i: NDArray, der_i: NDArray,
            N_j: NDArray, dNdx_j: NDArray, dNdy_j: NDArray, der_j: NDArray,
            n_i: int, n_j: int,
            nb_sq: int, nb_sq_v: int,
            TO: IntArray, FROM: IntArray,
            std_i: IntArray, std_j: IntArray,
            lu: CooLookup,
            sq_per_col: int,
    ) -> AssemblyTemplate:
        """Build one AssemblyTemplate for a (block_type, deriv_combo).

        deriv_combo:
          'none'    — N_i(q) * N_j(q)
          'dx_var'  — N_i(q) * dN_j/dξ(q) / dx   (derivative on trial fn)
          'dy_var'  — N_i(q) * dN_j/dη(q) / dy
          'dx_test' — dN_i/dξ(q) / dx * N_j(q)   (derivative on test fn)
          'dy_test' — dN_i/dη(q) / dy * N_j(q)

        Shape-weighting array [t, q, i, j] = w_q * Ni_q * Nj_q * area * sign_correction
        where sign_correction = der_factor[t] when derivative is on test or trial fn.
        """
        n_contr = n_i * n_j

        # Select shape function arrays for i-side (test) and j-side (trial)
        if deriv_combo == 'none':
            Phi_i = N_i          # (nb_quad, n_i)
            Phi_j = N_j          # (nb_quad, n_j)
            sign_i = np.ones(nb_tri)
            sign_j = np.ones(nb_tri)
        elif deriv_combo == 'dx_var':
            Phi_i = N_i
            Phi_j = dNdx_j / dx
            sign_i = np.ones(nb_tri)
            sign_j = der_j          # [1, -1]
        elif deriv_combo == 'dy_var':
            Phi_i = N_i
            Phi_j = dNdy_j / dy
            sign_i = np.ones(nb_tri)
            sign_j = der_j
        elif deriv_combo == 'dx_test':
            Phi_i = dNdx_i / dx
            Phi_j = N_j
            sign_i = der_i          # [1, -1]
            sign_j = np.ones(nb_tri)
        else:  # 'dy_test'
            Phi_i = dNdy_i / dy
            Phi_j = N_j
            sign_i = der_i
            sign_j = np.ones(nb_tri)

        # shape_weighting[t, q, i, j] = w_q * Phi_i[q,i] * Phi_j[q,j] * area * sign_i[t] * sign_j[t]
        # shape (nb_tri, nb_quad, n_i, n_j)
        sw = np.empty((nb_tri, nb_quad, n_i, n_j))
        for t in range(nb_tri):
            for q in range(nb_quad):
                sw[t, q] = (weights[q] * area * sign_i[t] * sign_j[t]
                            * np.outer(Phi_i[q], Phi_j[q]))
        shape_weighting = sw.reshape(-1)   # (nb_tri * nb_quad * n_contr,)

        # nnz_index and quad_flat_idx arrays: size (nb_sq * nb_tri * n_contr,)
        nnz_idx_list   = []
        qflat_idx_list = []

        for t in range(nb_tri):
            TO_tri   = TO[:, std_i[t]]    # (nb_sq, n_i)
            FROM_tri = FROM[:, std_j[t]]  # (nb_sq, n_j)

            for ii in range(n_i):
                for jj in range(n_j):
                    to_pts   = TO_tri[:, ii]    # (nb_sq,)
                    from_pts = FROM_tri[:, jj]  # (nb_sq,)

                    valid = (to_pts >= 0) & (from_pts >= 0)
                    # COO lookup for valid pairs only; -1 for invalid
                    coo_pos = np.full(nb_sq, -1, dtype=np.int32)
                    if valid.any():
                        coo_pos[valid] = lu.sorted_indices[
                            np.searchsorted(lu.sorted_keys,
                                            (to_pts[valid].astype(np.int64) * lu.max_col
                                             + from_pts[valid]))
                        ]
                        # verify no false positives
                        found_keys = lu.sorted_keys[coo_pos[valid]]
                        expected_keys = (to_pts[valid].astype(np.int64) * lu.max_col
                                         + from_pts[valid])
                        coo_pos[valid] = np.where(found_keys == expected_keys,
                                                  coo_pos[valid], -1)

                    nnz_idx_list.append(coo_pos)

                    # quad field flat index:  quad_idx * nb_sq + flat_sq
                    # quad_idx = t * nb_quad + q  (for q = 0..nb_quad-1)
                    #
                    # Squares are numbered x-fast: sq_x = sq % sq_per_row,
                    # sq_y = sq // sq_per_row, where sq_per_row = nb_sq // sq_per_col.
                    # The quad field has shape (6, sq_per_row, sq_per_col) and is
                    # flattened C-order: flat_sq = sq_x * sq_per_col + sq_y  (y-fast).
                    quad_base  = t * nb_quad  # first quad pt index for this tri
                    sq_indices = np.arange(nb_sq, dtype=np.int32)
                    sq_per_row = nb_sq // sq_per_col
                    flat_sq = ((sq_indices % sq_per_row) * sq_per_col
                               + sq_indices // sq_per_row)
                    qflat = np.empty((nb_quad, nb_sq), dtype=np.int32)
                    for q in range(nb_quad):
                        qflat[q] = (quad_base + q) * nb_sq + flat_sq
                    qflat_idx_list.append(qflat.reshape(nb_quad, nb_sq))

        # nnz_index: shape (nb_tri * n_contr, nb_sq), loop order was (t, i, j)
        # Flatten contr-major, sq-fast for consistency with _inject_term.
        nnz_index = np.stack(nnz_idx_list, axis=0)  # (nb_tri * n_contr, nb_sq)
        nnz_index = nnz_index.reshape(-1)            # flattened contr-major

        # quad_flat_idx: (nb_tri * n_contr, nb_quad, nb_sq) → align with shape_weighting
        # shape_weighting order: [t, q, i, j] = [t*nb_quad*n_contr + q*n_contr + i*n_j + j]
        # We stored qflat as (nb_quad, nb_sq) per (t,i,j)
        qflat_arr = np.stack(qflat_idx_list, axis=0)  # (nb_tri*n_contr, nb_quad, nb_sq)
        # Rearrange to (nb_tri*nb_quad*n_contr, nb_sq):
        #   current: [t*n_contr + contr_idx, q, sq]
        #   want:    [t*nb_quad*n_contr + q*n_contr + contr_idx, sq]
        ntc = nb_tri * n_contr
        qflat_arr = qflat_arr.reshape(nb_tri, n_contr, nb_quad, nb_sq)
        qflat_arr = qflat_arr.transpose(0, 2, 1, 3)   # (nb_tri, nb_quad, n_contr, nb_sq)
        qflat_arr = qflat_arr.reshape(nb_tri * nb_quad * n_contr, nb_sq)
        quad_flat_idx = qflat_arr.reshape(-1)  # (nb_tri * nb_quad * n_contr * nb_sq,)

        return AssemblyTemplate(
            shape_weighting=shape_weighting,
            nnz_index=nnz_index,
            quad_flat_idx=quad_flat_idx,
            nb_tri=nb_tri,
            nb_quad=nb_quad,
            n_i=n_i,
            n_j=n_j,
            nb_sq=nb_sq,
            nnz_per_block=lu.nnz_per_block,
        )

    # ======================================================================
    # Matrix and RHS assembly
    # ======================================================================

    def assemble_matrix(self,
                        nnz_values: NDArray,
                        quad_fields: Dict[str, NDArray],
                        terms: list,
                        residuals: Optional[List[str]] = None,
                        variables: Optional[List[str]] = None,
                        res_to_grid: Optional[Dict[str, str]] = None,
                        var_to_grid: Optional[Dict[str, str]] = None,
                        ) -> None:
        """Accumulate all term contributions into nnz_values (in-place).

        Parameters
        ----------
        nnz_values : NDArray, shape (self.nnz,)
            Pre-zeroed COO values array.
        quad_fields : dict
            Maps field name -> NDArray of shape (6, sq_per_row, sq_per_col)
            containing values at quadrature points.  quad_fields[name].ravel()
            is indexed by quad_flat_idx in the templates.
        terms : list of NonLinearTerm
            Active PDE terms.
        residuals, variables, res_to_grid, var_to_grid : optional
            If None, use the values stored in self (from construction).
        """
        residuals    = residuals    or self._residuals
        variables    = variables    or self._variables
        res_to_grid  = res_to_grid  or self._res_to_grid
        var_to_grid  = var_to_grid  or self._var_to_grid

        nb_vars = len(variables)
        nb_res  = len(residuals)

        for term in terms:
            for dep_var in term.dep_vars:
                # Identify block type for (term.res, dep_var)
                rg = res_to_grid[term.res]
                vg = var_to_grid[dep_var]
                block_type = (rg, vg)

                # Determine deriv_combo from term flags
                der_tf = getattr(term, 'der_testfun', False)
                if (term.d_dx_resfun or term.d_dy_resfun) and der_tf:
                    raise ValueError(
                        f"Term {term.name!r}: simultaneous derivative on both "
                        f"trial (d_dx/dy_resfun) and test (der_testfun) function "
                        f"is not supported.")
                if term.d_dx_resfun:
                    dc = 'dx_var'
                elif term.d_dy_resfun:
                    dc = 'dy_var'
                elif der_tf in ('x', True):
                    dc = 'dx_test'
                elif der_tf == 'y':
                    dc = 'dy_test'
                else:
                    dc = 'none'

                tmpl = self.assembly_templates[(block_type, dc, dep_var)]
                if tmpl.nnz_per_block == 0:
                    continue

                # Block offset in the flat nnz_values array
                ri = residuals.index(term.res)
                vi = variables.index(dep_var)
                block_offset = self._get_block_offset(ri, vi, nb_vars, block_type)

                # Evaluate derivative of residual function w.r.t. dep_var
                dep_vals = [quad_fields[v] for v in term.dep_vars]
                deriv_vals = term.evaluate_deriv(dep_var, *dep_vals)  # (6, Ny_sq, Nx_sq)

                self._inject_term(nnz_values, tmpl, deriv_vals, block_offset)

    def _get_block_offset(self, ri: int, vi: int, nb_vars: int,
                          block_type: Tuple[str, str]) -> int:
        """Compute byte-offset of block (ri, vi) into the flat nnz array."""
        # Count nnz before this block by summing all earlier (res, var) blocks
        # that share the same block_type
        offset = 0
        for r in range(len(self._residuals)):
            for v in range(len(self._variables)):
                bt = (self._res_to_grid[self._residuals[r]],
                      self._var_to_grid[self._variables[v]])
                bnnz = self._conn[bt][2]
                if r == ri and v == vi:
                    return offset
                offset += bnnz
        raise ValueError(f"Block ({ri},{vi}) not found")

    @staticmethod
    def _inject_term(nnz_values: NDArray,
                     tmpl: AssemblyTemplate,
                     deriv_vals: NDArray,
                     block_offset: int) -> None:
        """Inject one term's contribution into nnz_values.

        deriv_vals : shape (6, sq_per_row, sq_per_col) — quad-point field.
        """
        nb_tri  = tmpl.nb_tri
        nb_quad = tmpl.nb_quad
        n_contr = tmpl.n_contr
        nb_sq   = tmpl.nb_sq

        # Flatten quad field: shape (6*nb_sq,)
        quad_flat = deriv_vals.reshape(-1)

        # Gather field values at all (tri, quad_pt, i, j, sq) positions:
        # quad_flat_idx shape: (nb_tri * nb_quad * n_contr * nb_sq,)
        # shape_weighting shape: (nb_tri * nb_quad * n_contr,)
        field_vals = quad_flat[tmpl.quad_flat_idx]  # (nb_tri*nb_quad*n_contr*nb_sq,)

        # Reshape for quad-pt summation:
        # (nb_tri * n_contr, nb_quad, nb_sq)
        field_vals_r = field_vals.reshape(nb_tri * nb_quad * n_contr, nb_sq)
        field_vals_r = field_vals_r.reshape(nb_tri, nb_quad, n_contr, nb_sq)
        sw_r = tmpl.shape_weighting.reshape(nb_tri, nb_quad, n_contr)

        # Weighted sum over quad pts: (nb_tri, n_contr, nb_sq)
        contrib = (field_vals_r * sw_r[:, :, :, np.newaxis]).sum(axis=1)

        # Scatter into nnz.
        # nnz_index stored as (nb_tri * n_contr, nb_sq), contr-major
        # contrib shape: (nb_tri, n_contr, nb_sq) → reshape to (nb_tri*n_contr, nb_sq)
        nnz_index_2d = tmpl.nnz_index.reshape(nb_tri * n_contr, nb_sq)
        contrib_2d   = contrib.reshape(nb_tri * n_contr, nb_sq)

        valid = nnz_index_2d >= 0
        actual_idx = np.where(valid, nnz_index_2d + block_offset, 0)
        np.add.at(nnz_values, actual_idx[valid], contrib_2d[valid])

    def assemble_rhs(self,
                     rhs_values: NDArray,
                     quad_fields: Dict[str, NDArray],
                     terms: list,
                     residuals: Optional[List[str]] = None,
                     res_to_grid: Optional[Dict[str, str]] = None,
                     ) -> None:
        """Accumulate all term contributions into rhs_values (in-place).

        Parameters
        ----------
        rhs_values : NDArray, shape (sum of nb_inner for each residual,)
            Pre-zeroed local residual vector.
        quad_fields : dict
            Maps field name -> NDArray of shape (6, sq_per_row, sq_per_col).
        terms : list of NonLinearTerm
        """
        if not hasattr(self, '_area'):
            raise RuntimeError(
                "build_assembly_templates() must be called before assemble_rhs().")

        residuals   = residuals   or self._residuals
        res_to_grid = res_to_grid or self._res_to_grid

        nb_inner_v = self._grid_idx.Nx_v_inner * self._grid_idx.Ny_v_inner
        nb_inner_p = self._grid_idx.Nx_p_inner * self._grid_idx.Ny_p_inner

        # Compute RHS offset for each residual
        res_offsets: Dict[str, int] = {}
        off = 0
        for res in residuals:
            res_offsets[res] = off
            nb = nb_inner_v if res_to_grid[res] == 'v' else nb_inner_p
            off += nb

        TO_v = self._grid_idx.sq_TO_inner_v   # (nb_sq, 9)
        TO_p = self._grid_idx.sq_TO_inner_p   # (nb_sq, 4)
        nb_sq = TO_v.shape[0]
        nb_quad = 3

        from .elements import Quadrature3Points
        weights = Quadrature3Points.weights    # (3,)

        for term in terms:
            rg = res_to_grid[term.res]
            if rg == 'v':
                TO      = TO_v
                std     = _IDX_TO_STD_P2
                n_nodes = _NB_NODES_P2
                der_f   = self._der_factor_p2
                N_test  = self._N_p2
                dNdx_test = self._dNdx_p2
                dNdy_test = self._dNdy_p2
                d_scale_x = self._dx
                d_scale_y = self._dy
            else:
                TO      = TO_p
                std     = _IDX_TO_STD_P1
                n_nodes = _NB_NODES_P1
                der_f   = self._der_factor_p1
                N_test  = self._N_p1
                dNdx_test = self._dNdx_p1
                dNdy_test = self._dNdy_p1
                d_scale_x = self._dx
                d_scale_y = self._dy

            res_off  = res_offsets[term.res]
            nb_inner = nb_inner_v if rg == 'v' else nb_inner_p

            # Build quad-point values for the integrand.
            #
            # For d_dx_resfun / d_dy_resfun terms the integrand is
            #   ∂F/∂x = Σ_vars (∂F/∂var)(u_q) * (∂var/∂x)(q)
            # using the chain rule, consistent with the old solver's
            # _residual_vector_term_deriv.  The caller must supply gradient
            # quad fields under the key 'd_dx_<varname>' or 'd_dy_<varname>'.
            # The test function remains plain N_i (no der_factor needed).
            der_tf = getattr(term, 'der_testfun', False)
            if (term.d_dx_resfun or term.d_dy_resfun) and der_tf:
                raise ValueError(
                    f"Term {term.name!r}: simultaneous derivative on both "
                    f"trial (d_dx/dy_resfun) and test (der_testfun) function "
                    f"is not supported.")
            dep_var_vals = [quad_fields[v] for v in term.dep_vars]

            if term.d_dx_resfun or term.d_dy_resfun:
                # R_i = ∫ N_i * ∂F/∂x dΩ  (derivative on the variable/integrand)
                # ∂F/∂x computed via chain rule; caller supplies 'd_dx_<v>' fields.
                # dx_operator output = Σ_k dNdxi[q,k]*u_k*der_factor[t], so
                # physical ∂v/∂x = operator_output / dx.
                prefix  = 'd_dx_' if term.d_dx_resfun else 'd_dy_'
                d_scale = d_scale_x if term.d_dx_resfun else d_scale_y
                fun_vals = np.zeros_like(dep_var_vals[0])
                for v, uq in zip(term.dep_vars, dep_var_vals):
                    dF_dv    = term.evaluate_deriv(v, *dep_var_vals)
                    dv_dx    = quad_fields[prefix + v] / d_scale
                    fun_vals = fun_vals + dF_dv * dv_dx
                Phi_test     = N_test
                use_der_sign = False
            elif der_tf in ('x', True):
                # R_i = ∫ (∂N_i/∂x) * f(u) dΩ  (derivative on test function)
                fun_vals     = term.evaluate(*dep_var_vals)
                Phi_test     = dNdx_test / d_scale_x
                use_der_sign = True
            elif der_tf == 'y':
                fun_vals     = term.evaluate(*dep_var_vals)
                Phi_test     = dNdy_test / d_scale_y
                use_der_sign = True
            else:
                fun_vals     = term.evaluate(*dep_var_vals)
                Phi_test     = N_test
                use_der_sign = False

            # Build sq→(sq_x, sq_y) mapping once so fv[sq] = fun_vals[level, sq_x, sq_y].
            # fun_vals shape: (6, sq_per_row, sq_per_col); squares are x-fast:
            # sq_x = sq % sq_per_row, sq_y = sq // sq_per_row.
            _sq_per_col = fun_vals.shape[-1]
            _sq_per_row = fun_vals.shape[-2]
            _sq_all = np.arange(nb_sq)
            _sq_xi  = _sq_all % _sq_per_row
            _sq_yi  = _sq_all // _sq_per_row

            R_local = np.zeros(nb_inner)
            for t in range(2):
                TO_tri = TO[:, std[t]]              # (nb_sq, n_nodes)
                sign_t = der_f[t] if use_der_sign else 1.0
                for q in range(nb_quad):
                    w   = weights[q] * self._area * sign_t
                    fv  = fun_vals[t * nb_quad + q][_sq_xi, _sq_yi]  # (nb_sq,)
                    for i in range(n_nodes):
                        phi_i      = Phi_test[q, i]
                        inner_mask = TO_tri[:, i]
                        valid      = inner_mask >= 0
                        if valid.any():
                            np.add.at(R_local,
                                      inner_mask[valid],
                                      w * phi_i * fv[valid])

            rhs_values[res_off:res_off + nb_inner] += R_local

    def _store_element_data(self, elements: "TaylorHoodP2P1") -> None:
        """Cache element data for RHS assembly after build_assembly_templates."""
        from .elements import Quadrature3Points
        self._dx    = elements.dx
        self._dy    = elements.dy
        self._area  = elements.dx * elements.dy / 2.0
        self._N_p2  = elements.P2.N          # (3, 6)
        self._N_p1  = elements.P1.N          # (3, 3)
        self._dNdx_p2 = elements.P2.dN_dx
        self._dNdy_p2 = elements.P2.dN_dy
        self._dNdx_p1 = elements.P1.dN_dx
        self._dNdy_p1 = elements.P1.dN_dy
        self._der_factor_p2 = np.array(elements.P2.der_factor, dtype=float)
        self._der_factor_p1 = np.array(elements.P1.der_factor, dtype=float)
