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
"""Grid index management for Taylor-Hood P2P1 FEM assembly.

Two grids are managed by a single GridIndexManager:
  - Density (P1): coarse grid, Nx_p x Ny_p inner nodes, ghost depth 1.
  - Mass flux (P2): fine grid, Nx_v x Ny_v inner nodes, ghost depth 2.

Index mask convention:
  - index_mask_inner_local: inner nodes get sequential indices 0..(Nx*Ny)-1
    (row-major: x varies fastest); ghost nodes are -1.
  - index_mask_padded_local: equals inner mask on inner nodes; ghost nodes
    that represent DOF are assigned new sequential indices
      - inter-subdomain ghost nodes (DOFs that live on another process)
      - Neumann ghost nodes (reflecting DOF on adjacent inner node)
      - Dirichlet ghost nodes (no DOF, stay -1)
      - serial periodic ghost nodes (reflecting DOF on opposite side of domain)
  - index_mask_padded_global: unique global indices from DomainDecomposition.
  - l2g_list: maps local contributor index -> global index.
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
      - Density (P1) on the coarse grid via decomp._decomp (ghost depth 1).
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

        self._decomp = decomp
        self._variables = variables

        # ------------------------------------------------------------------
        # Density (P1) grid dimensions
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
        self.Nx_v_padded = Nx_v + 4
        self.Ny_v_padded = Ny_v + 4

        # ------------------------------------------------------------------
        # Domain boundary flags (physical, non-periodic boundaries only)
        # ------------------------------------------------------------------
        self.bc_at_W = decomp.bc_at_W
        self.bc_at_E = decomp.bc_at_E
        self.bc_at_S = decomp.bc_at_S
        self.bc_at_N = decomp.bc_at_N

        # ------------------------------------------------------------------
        # _bc_neumann['xW']['jx'] == True means jx has Neumann BC on West side.
        # User-facing BC list order in problem config stays ρ-based; the
        # FEM2D pressure-based internals may query with var='p', which
        # resolves to the 'rho' BC entry via the alias below.
        # ------------------------------------------------------------------
        self._bc_neumann = {}
        _BC_LIST_ORDER = ['rho', 'jx', 'jy']
        _BC_VAR_ALIAS = {'p': 'rho', 'theta': 'rho', 'fb': 'rho'}  # internal DOF name → user-facing BC name

        def _is_neumann(side, var):
            bc_var = _BC_VAR_ALIAS.get(var, var)
            return decomp.grid[f'bc_{side}'][_BC_LIST_ORDER.index(bc_var)] == 'N'

        for side in ('xW', 'xE', 'yS', 'yN'):
            self._bc_neumann[side] = {}
            for var in variables:
                self._bc_neumann[side][var] = _is_neumann(side, var)

        # ------------------------------------------------------------------
        # Square counts
        # ------------------------------------------------------------------
        self.sq_per_row = self.Nx_p_padded -1
        self.sq_per_col = self.Ny_p_padded - 1
        self.nb_sq = self.sq_per_row * self.sq_per_col

        # Square origin arrays (x-major, shape (nb_sq,))
        sq_idx = np.arange(self.nb_sq)

        self.sq_x_arr_p = sq_idx % self.sq_per_row
        self.sq_y_arr_p = sq_idx // self.sq_per_row

        self.sq_x_arr_v = self.sq_x_arr_p * 2
        self.sq_y_arr_v = self.sq_y_arr_p * 2

    # ======================================================================
    # Density (P1) index masks
    # ======================================================================

    @cached_property
    def index_mask_inner_local_p(self) -> IntArray:
        """Local indices for inner density nodes, shape (Nx_p_padded, Ny_p_padded).

        Inner nodes: sequential 0..Nx_p*Ny_p-1 (row-major, x varies fastest).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_p_padded, self.Ny_p_padded), -1, dtype=np.int32)
        nb_inner = self.Nx_p_inner * self.Ny_p_inner
        mask[1:-1, 1:-1] = np.arange(nb_inner).reshape(
            (self.Nx_p_inner, self.Ny_p_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_p(self, var: str = '') -> IntArray:
        """Local contributor indices for density grid, shape (Nx_p_padded, Ny_p_padded).

        - serial + periodic: wrap around
        - inter-subdomain ghost nodes: new sequential indices
        - Neumann ghost nodes: forward to adjacent inner node
        - Dirichlet ghost nodes: stay -1

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_p.copy()
        decomp = self._decomp

        # Periodic AND serial -> wrap around
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_p_padded - 2, :]
            mask[self.Nx_p_padded - 1, :] = mask[1, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_p_padded - 2]
            mask[:, self.Ny_p_padded - 1] = mask[:, 1]

        # New indices for inter-subdomain (non-boundary) ghost nodes
        cur_val = self.Nx_p_inner * self.Ny_p_inner
        for x in range(self.Nx_p_padded):
            for y in range(self.Ny_p_padded):
                if mask[x, y] == -1 and not self._is_boundary_p(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1

        # Neumann forwarding removed: natural BC via "do nothing" approach.
        # Ghost nodes at Neumann boundaries stay -1 (no DOF), same as Dirichlet.
        # The zero-flux condition is enforced naturally by the variational formulation.
        if True:
            if var:
                if self.bc_at_W and self._bc_neumann['xW'][var]:
                    mask[0, :] = mask[1, :]
                if self.bc_at_E and self._bc_neumann['xE'][var]:
                    mask[self.Nx_p_padded - 1, :] = mask[self.Nx_p_padded - 2, :]
                if self.bc_at_S and self._bc_neumann['yS'][var]:
                    mask[:, 0] = mask[:, 1]
                if self.bc_at_N and self._bc_neumann['yN'][var]:
                    mask[:, self.Ny_p_padded - 1] = mask[:, self.Ny_p_padded - 2]

        return mask

    def _is_boundary_p(self, x: int, y: int) -> bool:
        """True if (x, y) represents a boundary condition on the density grid.
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
        """Number of unique contributor indices on the density grid."""
        mask = self.index_mask_padded_local_p()
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1 if len(valid) > 0 else 0

    @cached_property
    def l2g_list_p(self) -> IntArray:
        """Local-to-global mapping for density contributors, shape (nb_contributors_p,)."""
        mask_local = self.index_mask_padded_local_p()
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

        Inner nodes: sequential 0..Nx_v*Ny_v-1 (row-major, x varies fastest).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_v_padded, self.Ny_v_padded), -1, dtype=np.int32)
        nb_inner = self.Nx_v_inner * self.Ny_v_inner
        mask[2:-2, 2:-2] = np.arange(nb_inner).reshape(
            (self.Nx_v_inner, self.Ny_v_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_v(self, var: str = '') -> IntArray:
        """Local contributor indices for mass flux grid, shape (Nx_v_padded, Ny_v_padded).

        - serial + periodic: wrap around (both ghost layers)
        - inter-subdomain ghost nodes: new sequential indices
        - Neumann ghost nodes: forward both layers to adjacent inner node
        - Dirichlet ghost nodes: stay -1

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_v.copy()
        decomp = self._decomp

        # Periodic AND serial -> wrap around (both ghost layers)
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_v_padded - 4, :]
            mask[1, :] = mask[self.Nx_v_padded - 3, :]
            mask[self.Nx_v_padded - 2, :] = mask[2, :]
            mask[self.Nx_v_padded - 1, :] = mask[3, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_v_padded - 4]
            mask[:, 1] = mask[:, self.Ny_v_padded - 3]
            mask[:, self.Ny_v_padded - 2] = mask[:, 2]
            mask[:, self.Ny_v_padded - 1] = mask[:, 3]

        # New indices for inter-subdomain (non-boundary) ghost nodes
        cur_val = self.Nx_v_inner * self.Ny_v_inner
        for x in range(self.Nx_v_padded):
            for y in range(self.Ny_v_padded):
                if mask[x, y] == -1 and not self._is_boundary_v(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1

        # Neumann forwarding removed: natural BC via "do nothing" approach.
        # Ghost nodes at Neumann boundaries stay -1 (no DOF), same as Dirichlet.
        if True:
            if var:
                if self.bc_at_W and self._bc_neumann['xW'][var]:
                    mask[1, :] = mask[2, :]
                    mask[0, :] = mask[2, :]
                if self.bc_at_E and self._bc_neumann['xE'][var]:
                    mask[self.Nx_v_padded - 2, :] = mask[self.Nx_v_padded - 3, :]
                    mask[self.Nx_v_padded - 1, :] = mask[self.Nx_v_padded - 3, :]
                if self.bc_at_S and self._bc_neumann['yS'][var]:
                    mask[:, 1] = mask[:, 2]
                    mask[:, 0] = mask[:, 2]
                if self.bc_at_N and self._bc_neumann['yN'][var]:
                    mask[:, self.Ny_v_padded - 2] = mask[:, self.Ny_v_padded - 3]
                    mask[:, self.Ny_v_padded - 1] = mask[:, self.Ny_v_padded - 3]

        return mask

    def _is_boundary_v(self, x: int, y: int) -> bool:
        """True if (x, y) is a boundary ghost node on the mass flux grid (depth 1 or 2)."""
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
        mask = self.index_mask_padded_local_v()
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1 if len(valid) > 0 else 0

    @cached_property
    def l2g_list_v(self) -> IntArray:
        """Local-to-global mapping for mass flux contributors, shape (nb_contributors_v,)."""
        mask_local  = self.index_mask_padded_local_v()
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

        Corner order: [bl, br, tl, tr]
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

        Corner order: [bl, br, tl, tr, ml, bm, mm, tm, mr]
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
