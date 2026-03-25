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
"""Grid index management for Taylor-Hood P2P1 FEM assembly.

Two grids are managed by a single GridIndexManager:
  - Pressure (P1): coarse grid, Nx x Ny inner nodes, ghost depth 1.
  - Mass flux (P2): fine grid, Nx_v x Ny_v inner nodes, ghost depth 2.

Index mask convention (identical to fem_2d_old):
  - index_mask_inner_local:  inner nodes get sequential indices 0..N-1
    (column-major); ghost nodes are -1.
  - index_mask_padded_local: equals inner mask on inner nodes; ghost nodes
    that contribute to residuals are assigned new sequential indices
    (inter-subdomain) or forwarded (Neumann) or left -1 (Dirichlet).
  - index_mask_padded_global: unique global indices from DomainDecomposition.
  - l2g_list: maps local contributor index -> global index.

For the mass flux grid, Neumann forwarding is applied to both ghost layers
(depth 1 and depth 2); Dirichlet ghosts stay -1 on both layers.
"""
from functools import cached_property, lru_cache
from typing import List, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]


class GridIndexManager:
    """Index masks and square connectivity for Taylor-Hood P2P1 assembly.

    Manages two grids simultaneously:
      - Pressure (P1) on the coarse grid via decomp._decomp (ghost depth 1).
      - Mass flux (P2) on the fine grid via decomp._decomp_v (ghost depth 2).

    Parameters
    ----------
    decomp : DomainDecomposition
        Must have been initialised with solver='fem'.
    variables : list of str
        Ordered variable names, e.g. ['rho', 'jx', 'jy']. Used to look up
        per-variable BC types.
    """

    def __init__(self,
                 decomp: "DomainDecomposition",
                 variables: List[str]):
        assert decomp._is_fem, "GridIndexManager requires solver='fem'"
        self._decomp = decomp
        self._variables = variables
        grid = decomp.grid

        # ------------------------------------------------------------------
        # Pressure (P1) grid dimensions
        # ------------------------------------------------------------------
        Nx_p, Ny_p = decomp.nb_subdomain_grid_pts
        self.Nx_p_inner  = Nx_p
        self.Ny_p_inner  = Ny_p
        self.Nx_p_padded = Nx_p + 2
        self.Ny_p_padded = Ny_p + 2

        # ------------------------------------------------------------------
        # Mass flux (P2) grid dimensions
        # ------------------------------------------------------------------
        Nx_v, Ny_v = decomp.nb_subdomain_grid_pts_v
        self.Nx_v_inner  = Nx_v
        self.Ny_v_inner  = Ny_v
        self.Nx_v_padded = Nx_v + 4   # ghost depth 2 on each side
        self.Ny_v_padded = Ny_v + 4

        # ------------------------------------------------------------------
        # Domain boundary flags (physical, non-periodic boundaries only)
        # ------------------------------------------------------------------
        self.bc_at_W = decomp.bc_at_W
        self.bc_at_E = decomp.bc_at_E
        self.bc_at_S = decomp.bc_at_S
        self.bc_at_N = decomp.bc_at_N

        # ------------------------------------------------------------------
        # Per-variable Neumann flags from grid config
        # The grid BC lists are always ordered [rho, jx, jy] (indices 0,1,2).
        # Build a lookup keyed by variable name so callers don't need to
        # know the BC-list index order.
        # ------------------------------------------------------------------
        _BC_LIST_ORDER = ['rho', 'jx', 'jy']   # fixed convention in io.py

        def _neumann_for_var(bc_list, var_name):
            idx = _BC_LIST_ORDER.index(var_name) if var_name in _BC_LIST_ORDER else -1
            return (bc_list[idx] == 'N') if idx >= 0 and idx < len(bc_list) else False

        self._bc_neumann = {
            side: {v: _neumann_for_var(grid[f'bc_{side}'], v)
                   for v in variables}
            for side in ('xW', 'xE', 'yS', 'yN')
        }

        # ------------------------------------------------------------------
        # Square counts
        # P1: (Nx_p_inner + 1) x (Ny_p_inner + 1) squares (include ghost squares)
        # P2: one P2 square = one 2x2 block on the fine grid; the number of
        #     P2 squares equals the number of P1 squares since each P1 square
        #     maps to exactly one P2 square (fine grid has 2x resolution).
        #     A P2 square spans indices [2i, 2i+2] x [2j, 2j+2] on fine grid.
        # ------------------------------------------------------------------
        self.sq_per_row_p = self.Nx_p_inner + 1
        self.sq_per_col_p = self.Ny_p_inner + 1
        self.nb_sq_p = self.sq_per_row_p * self.sq_per_col_p

        # P2 squares: same count as P1 squares (one-to-one mapping)
        self.sq_per_row_v = self.sq_per_row_p
        self.sq_per_col_v = self.sq_per_col_p
        self.nb_sq_v = self.nb_sq_p

        # Square origin arrays (x-major, shape (nb_sq,))
        sq_idx = np.arange(self.nb_sq_p)
        # P1 square origins: corners in padded P1 grid
        self.sq_x_arr_p = sq_idx % self.sq_per_row_p
        self.sq_y_arr_p = sq_idx // self.sq_per_row_p
        # P2 square origins: step by 2 in padded P2 grid (ghost offset = 2)
        self.sq_x_arr_v = self.sq_x_arr_p * 2
        self.sq_y_arr_v = self.sq_y_arr_p * 2

    # ======================================================================
    # Pressure (P1) index masks
    # ======================================================================

    @cached_property
    def index_mask_inner_local_p(self) -> IntArray:
        """Local indices for inner pressure nodes, shape (Nx_p_padded, Ny_p_padded).

        Inner nodes: sequential 0..Nx_p*Ny_p-1 (column-major).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_p_padded, self.Ny_p_padded), -1, dtype=np.int32)
        nb = self.Nx_p_inner * self.Ny_p_inner
        mask[1:-1, 1:-1] = np.arange(nb).reshape(
            (self.Nx_p_inner, self.Ny_p_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_p(self, var: str = '') -> IntArray:
        """Local contributor indices for pressure grid, shape (Nx_p_padded, Ny_p_padded).

        Ghost nodes receive new sequential indices (inter-subdomain), are
        forwarded to their inner neighbour (Neumann), or stay -1 (Dirichlet).

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_p.copy()
        decomp = self._decomp

        # Periodic wrapping (serial only; MPI handled by ghost exchange)
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_p_padded - 2, :]
            mask[self.Nx_p_padded - 1, :] = mask[1, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_p_padded - 2]
            mask[:, self.Ny_p_padded - 1] = mask[:, 1]

        # Assign new indices to remaining valid (non-Dirichlet) ghost nodes
        cur_val = self.Nx_p_inner * self.Ny_p_inner
        for x in range(self.Nx_p_padded):
            for y in range(self.Ny_p_padded):
                if mask[x, y] == -1 and not self._is_dirichlet_p(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1

        # Neumann forwarding: ghost -> adjacent inner node
        if var:
            if self.bc_at_W and self._bc_neumann['xW'].get(var, False):
                mask[0, :] = mask[1, :]
            if self.bc_at_E and self._bc_neumann['xE'].get(var, False):
                mask[self.Nx_p_padded - 1, :] = mask[self.Nx_p_padded - 2, :]
            if self.bc_at_S and self._bc_neumann['yS'].get(var, False):
                mask[:, 0] = mask[:, 1]
            if self.bc_at_N and self._bc_neumann['yN'].get(var, False):
                mask[:, self.Ny_p_padded - 1] = mask[:, self.Ny_p_padded - 2]

        return mask

    def _is_dirichlet_p(self, x: int, y: int) -> bool:
        """True if (x, y) is a physical boundary ghost on the pressure grid.

        All physical boundary ghosts are excluded from index assignment here.
        Neumann forwarding later overrides relevant ghosts with inner indices.
        Dirichlet ghosts remain -1.
        """
        if x == 0 and self.bc_at_W:
            return True
        if x == self.Nx_p_padded - 1 and self.bc_at_E:
            return True
        if y == 0 and self.bc_at_S:
            return True
        if y == self.Ny_p_padded - 1 and self.bc_at_N:
            return True
        return False

    @cached_property
    def nb_contributors_p(self) -> int:
        """Number of unique contributor indices on the pressure grid."""
        mask = self.index_mask_padded_local_p('')
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1 if len(valid) > 0 else 0

    @cached_property
    def l2g_list_p(self) -> IntArray:
        """Local-to-global mapping for pressure contributors, shape (nb_contributors_p,)."""
        mask_local  = self.index_mask_padded_local_p('')
        mask_global = self._decomp.index_mask_padded_global
        l2g = np.zeros(self.nb_contributors_p, dtype=np.int32)
        valid = mask_local >= 0
        l2g[mask_local[valid]] = mask_global[valid]
        return l2g

    # ======================================================================
    # Mass flux (P2) index masks
    # ======================================================================

    @cached_property
    def index_mask_inner_local_v(self) -> IntArray:
        """Local indices for inner mass flux nodes, shape (Nx_v_padded, Ny_v_padded).

        Inner nodes: sequential 0..Nx_v*Ny_v-1 (column-major).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_v_padded, self.Ny_v_padded), -1, dtype=np.int32)
        nb = self.Nx_v_inner * self.Ny_v_inner
        mask[2:-2, 2:-2] = np.arange(nb).reshape(
            (self.Nx_v_inner, self.Ny_v_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_v(self, var: str = '') -> IntArray:
        """Local contributor indices for mass flux grid, shape (Nx_v_padded, Ny_v_padded).

        Inter-subdomain ghost nodes (both layers) receive new sequential
        indices. Neumann ghost nodes (both layers) are forwarded to the
        adjacent inner node. Dirichlet ghost nodes stay -1.

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_v.copy()
        decomp = self._decomp

        # Periodic wrapping (serial only)
        if decomp.periodic_x and decomp.has_full_x:
            # depth-2 ghost layers at W and E
            mask[0, :] = mask[self.Nx_v_padded - 4, :]
            mask[1, :] = mask[self.Nx_v_padded - 3, :]
            mask[self.Nx_v_padded - 2, :] = mask[2, :]
            mask[self.Nx_v_padded - 1, :] = mask[3, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_v_padded - 4]
            mask[:, 1] = mask[:, self.Ny_v_padded - 3]
            mask[:, self.Ny_v_padded - 2] = mask[:, 2]
            mask[:, self.Ny_v_padded - 1] = mask[:, 3]

        # Assign new indices to remaining valid ghost nodes
        cur_val = self.Nx_v_inner * self.Ny_v_inner
        for x in range(self.Nx_v_padded):
            for y in range(self.Ny_v_padded):
                if mask[x, y] == -1 and not self._is_dirichlet_v(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1

        # Neumann forwarding: both ghost layers forward to adjacent inner node
        if var:
            if self.bc_at_W and self._bc_neumann['xW'].get(var, False):
                # ghost1 = layer 1, ghost2 = layer 0; inner = layer 2
                mask[1, :] = mask[2, :]
                mask[0, :] = mask[2, :]
            if self.bc_at_E and self._bc_neumann['xE'].get(var, False):
                mask[self.Nx_v_padded - 2, :] = mask[self.Nx_v_padded - 3, :]
                mask[self.Nx_v_padded - 1, :] = mask[self.Nx_v_padded - 3, :]
            if self.bc_at_S and self._bc_neumann['yS'].get(var, False):
                mask[:, 1] = mask[:, 2]
                mask[:, 0] = mask[:, 2]
            if self.bc_at_N and self._bc_neumann['yN'].get(var, False):
                mask[:, self.Ny_v_padded - 2] = mask[:, self.Ny_v_padded - 3]
                mask[:, self.Ny_v_padded - 1] = mask[:, self.Ny_v_padded - 3]

        return mask

    def _is_dirichlet_v(self, x: int, y: int) -> bool:
        """True if (x, y) is a Dirichlet ghost node on the mass flux grid.

        Both ghost layers (depth 1 and 2) are Dirichlet when at a Dirichlet
        boundary.
        """
        if (x in (0, 1)) and self.bc_at_W:
            return True
        if (x in (self.Nx_v_padded - 1, self.Nx_v_padded - 2)) and self.bc_at_E:
            return True
        if (y in (0, 1)) and self.bc_at_S:
            return True
        if (y in (self.Ny_v_padded - 1, self.Ny_v_padded - 2)) and self.bc_at_N:
            return True
        return False

    @cached_property
    def nb_contributors_v(self) -> int:
        """Number of unique contributor indices on the mass flux grid."""
        mask = self.index_mask_padded_local_v('')
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1 if len(valid) > 0 else 0

    @cached_property
    def l2g_list_v(self) -> IntArray:
        """Local-to-global mapping for mass flux contributors, shape (nb_contributors_v,)."""
        mask_local  = self.index_mask_padded_local_v('')
        mask_global = self._decomp.index_mask_padded_global_v
        l2g = np.zeros(self.nb_contributors_v, dtype=np.int32)
        valid = mask_local >= 0
        l2g[mask_local[valid]] = mask_global[valid]
        return l2g

    # ======================================================================
    # Square corner connectivity
    # ======================================================================

    @cached_property
    def sq_TO_inner_p(self) -> IntArray:
        """Residual (inner) indices for P1 square corners, shape (nb_sq, 4).

        Corner order: [bl, br, tl, tr] = [(sx,sy), (sx+1,sy), (sx,sy+1), (sx+1,sy+1)].
        -1 for corners outside the inner domain.
        """
        m  = self.index_mask_inner_local_p
        sx = self.sq_x_arr_p
        sy = self.sq_y_arr_p
        return np.column_stack([
            m[sx,     sy    ],   # bl
            m[sx + 1, sy    ],   # br
            m[sx,     sy + 1],   # tl
            m[sx + 1, sy + 1],   # tr
        ])

    @lru_cache(maxsize=None)
    def sq_FROM_padded_p(self, var: str) -> IntArray:
        """Contributor indices for P1 square corners, shape (nb_sq, 4).

        Parameters
        ----------
        var : str
            Variable name for Neumann BC handling.
        """
        m  = self.index_mask_padded_local_p(var)
        sx = self.sq_x_arr_p
        sy = self.sq_y_arr_p
        return np.column_stack([
            m[sx,     sy    ],
            m[sx + 1, sy    ],
            m[sx,     sy + 1],
            m[sx + 1, sy + 1],
        ])

    @cached_property
    def sq_TO_inner_v(self) -> IntArray:
        """Residual (inner) indices for P2 square nodes, shape (nb_sq, 9).

        Node order matches P2.square_node_offsets (in fine-grid steps):
        [bl, br, tl, tr, ml, bm, mm, tm, mr]
        = offsets (0,0),(2,0),(0,2),(2,2),(0,1),(1,0),(1,1),(1,2),(2,1).
        Origin is at (sx_v, sy_v) on the padded fine grid.
        -1 for nodes outside the inner domain.
        """
        m  = self.index_mask_inner_local_v
        sx = self.sq_x_arr_v
        sy = self.sq_y_arr_v
        return np.column_stack([
            m[sx,     sy    ],   # bl  (0,0)
            m[sx + 2, sy    ],   # br  (2,0)
            m[sx,     sy + 2],   # tl  (0,2)
            m[sx + 2, sy + 2],   # tr  (2,2)
            m[sx,     sy + 1],   # ml  (0,1)
            m[sx + 1, sy    ],   # bm  (1,0)
            m[sx + 1, sy + 1],   # mm  (1,1)
            m[sx + 1, sy + 2],   # tm  (1,2)
            m[sx + 2, sy + 1],   # mr  (2,1)
        ])

    @lru_cache(maxsize=None)
    def sq_FROM_padded_v(self, var: str) -> IntArray:
        """Contributor indices for P2 square nodes, shape (nb_sq, 9).

        Parameters
        ----------
        var : str
            Variable name for Neumann BC handling.
        """
        m  = self.index_mask_padded_local_v(var)
        sx = self.sq_x_arr_v
        sy = self.sq_y_arr_v
        return np.column_stack([
            m[sx,     sy    ],
            m[sx + 2, sy    ],
            m[sx,     sy + 2],
            m[sx + 2, sy + 2],
            m[sx,     sy + 1],
            m[sx + 1, sy    ],
            m[sx + 1, sy + 1],
            m[sx + 1, sy + 2],
            m[sx + 2, sy + 1],
        ])
