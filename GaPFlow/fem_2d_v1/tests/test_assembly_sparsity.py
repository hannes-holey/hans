"""Phase 5a tests: sparsity structure of the P2P1 assembly.

Tests the Assembly class without any value injection — only the COO index
arrays, RHS pattern, and COO lookup correctness.

Grid setup: serial, non-periodic, Dirichlet on all sides (so boundary ghosts
are -1 everywhere, which exercises the most constrained case).

Checks:
  - No duplicate (global_row, global_col) entries.
  - No out-of-range global indices.
  - Global rows/cols are non-negative everywhere.
  - Symmetry of sparsity pattern for symmetric blocks (vv and pp).
  - RHS global rows are unique and in range.
  - COO lookup round-trip: every (inner, contrib) pair in connectivity can be
    found, and a pair outside the pattern returns -1.
  - Block nnz counts match the product of triangle-node counts.
"""
import pytest
import numpy as np

from GaPFlow.fem_2d.assembly import Assembly, BLOCK_VV, BLOCK_VP, BLOCK_PV, BLOCK_PP
from GaPFlow.fem_2d.grid_index import GridIndexManager
from GaPFlow.fem_2d.global_matrix import field_to_global


# ---------------------------------------------------------------------------
# Minimal DomainDecomposition stub (serial, non-periodic, all-Dirichlet)
# ---------------------------------------------------------------------------

class _StubDecomp:
    """Minimal stub satisfying GridIndexManager's interface."""

    def __init__(self, Nx_p: int, Ny_p: int):
        """
        Nx_p, Ny_p : inner pressure node counts.
        Velocity grid: (2*Nx_p-1) x (2*Ny_p-1) global, same locally (serial).
        """
        self.Nx_p = Nx_p
        self.Ny_p = Ny_p
        Nx_v = 2 * Nx_p - 1
        Ny_v = 2 * Ny_p - 1

        # DomainDecomposition interface
        self._is_fem       = True
        self.periodic_x    = False
        self.periodic_y    = False
        self.has_full_x    = True
        self.has_full_y    = True
        self.is_at_xW      = True
        self.is_at_xE      = True
        self.is_at_yS      = True
        self.is_at_yN      = True

        # All-Dirichlet boundary conditions
        nb_vars = 3  # jx, jy, rho
        self.grid = {
            'bc_xW': ['D'] * nb_vars,
            'bc_xE': ['D'] * nb_vars,
            'bc_yS': ['D'] * nb_vars,
            'bc_yN': ['D'] * nb_vars,
        }

        # Pressure grid (ghost depth 1)
        self.nb_subdomain_grid_pts = (Nx_p, Ny_p)
        Nx_p_pad = Nx_p + 2
        Ny_p_pad = Ny_p + 2
        nb_p = Nx_p * Ny_p
        self.index_mask_padded_global = np.full(
            (Nx_p_pad, Ny_p_pad), -1, dtype=np.int32)
        self.index_mask_padded_global[1:-1, 1:-1] = np.arange(nb_p).reshape(
            Nx_p, Ny_p, order='F')

        # Velocity grid (ghost depth 2)
        self.nb_subdomain_grid_pts_v = (Nx_v, Ny_v)
        Nx_v_pad = Nx_v + 4
        Ny_v_pad = Ny_v + 4
        nb_v = Nx_v * Ny_v
        self.index_mask_padded_global_v = np.full(
            (Nx_v_pad, Ny_v_pad), -1, dtype=np.int32)
        self.index_mask_padded_global_v[2:-2, 2:-2] = np.arange(nb_v).reshape(
            Nx_v, Ny_v, order='F')


def _make_assembly(Nx_p: int = 4, Ny_p: int = 4):
    """Build a small Assembly instance for testing."""
    decomp = _StubDecomp(Nx_p, Ny_p)
    variables = ['jx', 'jy', 'rho']
    residuals = ['jx', 'jy', 'rho']   # residual names share variable names here

    grid_idx = GridIndexManager(decomp, variables)

    Nx_v = 2 * Nx_p - 1
    Ny_v = 2 * Ny_p - 1
    cols_v = Ny_v   # column = y direction (F-order storage)
    M      = Ny_p

    res_to_grid = {'jx': 'v', 'jy': 'v', 'rho': 'p'}
    var_to_grid = {'jx': 'v', 'jy': 'v', 'rho': 'p'}

    asm = Assembly(
        grid_idx   = grid_idx,
        variables  = variables,
        residuals  = residuals,
        res_to_grid = res_to_grid,
        var_to_grid = var_to_grid,
        cols_v     = cols_v,
        M          = M,
        energy     = False,
    )
    return asm, grid_idx, decomp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSparsityNoDuplicates:
    """Global (row, col) pairs must be unique."""

    def test_no_duplicate_global_entries(self):
        asm, _, _ = _make_assembly()
        pairs = np.column_stack([asm.global_rows, asm.global_cols])
        unique_pairs = np.unique(pairs, axis=0)
        assert len(unique_pairs) == len(pairs), (
            f"Duplicate (global_row, global_col) entries: "
            f"{len(pairs) - len(unique_pairs)} duplicates found")


class TestSparsityRange:
    """All global indices must be non-negative and within the global matrix."""

    def test_global_rows_non_negative(self):
        asm, _, _ = _make_assembly()
        assert np.all(asm.global_rows >= 0)

    def test_global_cols_non_negative(self):
        asm, _, _ = _make_assembly()
        assert np.all(asm.global_cols >= 0)

    def test_global_rows_in_range(self):
        Nx_p, Ny_p = 4, 4
        asm, grid_idx, _ = _make_assembly(Nx_p, Ny_p)
        # Global size: nb_v_inner * 2 (jx,jy) + nb_p_inner * 1 (rho)
        # Upper bound from field_to_global on the last rho node
        Nx_v = 2 * Nx_p - 1
        Ny_v = 2 * Ny_p - 1
        cols_v = Ny_v
        M      = Ny_p
        nb_v = Nx_v * Ny_v
        nb_p = Nx_p * Ny_p
        # max global index = field_to_global(nb_p-1, res_type=2, ...)
        max_global = field_to_global(nb_p - 1, 2, cols_v, M, False)
        assert np.all(asm.global_rows <= max_global)

    def test_global_cols_in_range(self):
        Nx_p, Ny_p = 4, 4
        asm, grid_idx, _ = _make_assembly(Nx_p, Ny_p)
        Nx_v = 2 * Nx_p - 1
        Ny_v = 2 * Ny_p - 1
        cols_v = Ny_v
        M      = Ny_p
        max_global = field_to_global(Ny_p * Nx_p - 1, 2, cols_v, M, False)
        assert np.all(asm.global_cols <= max_global)


class TestSparsitySymmetry:
    """vv and pp blocks should produce symmetric sparsity patterns."""

    def _extract_block_pairs(self, asm, ri_name, vi_name):
        residuals = asm._residuals
        variables = asm._variables
        ri = residuals.index(ri_name)
        vi = variables.index(vi_name)
        bt = (asm._res_to_grid[ri_name], asm._var_to_grid[vi_name])
        nnz = asm._conn[bt][2]

        grid_idx = asm._grid_idx
        res_grid = asm._res_to_grid[ri_name]
        var_grid = asm._var_to_grid[vi_name]

        nb_inner_v  = grid_idx.Nx_v_inner * grid_idx.Ny_v_inner
        nb_inner_p  = grid_idx.Nx_p_inner * grid_idx.Ny_p_inner
        nb_contrib_v = grid_idx.nb_contributors_v
        nb_contrib_p = grid_idx.nb_contributors_p

        nb_inner  = nb_inner_v  if res_grid == 'v' else nb_inner_p
        nb_contrib = nb_contrib_v if var_grid == 'v' else nb_contrib_p

        # Find slice offset in COO arrays
        offset = 0
        for r, res in enumerate(residuals):
            for v, var in enumerate(variables):
                if r == ri and v == vi:
                    break
                btt = (asm._res_to_grid[res], asm._var_to_grid[var])
                offset += asm._conn[btt][2]
            else:
                continue
            break

        rows = asm.global_rows[offset:offset + nnz]
        cols = asm.global_cols[offset:offset + nnz]
        return set(zip(rows.tolist(), cols.tolist()))

    def test_vv_jx_jx_symmetric(self):
        asm, _, _ = _make_assembly()
        pairs_jxjx = self._extract_block_pairs(asm, 'jx', 'jx')
        # Symmetry: if (r,c) is in jx→jx block, (c,r) should be too
        for r, c in pairs_jxjx:
            assert (c, r) in pairs_jxjx, f"Missing transpose ({c},{r}) of ({r},{c})"

    def test_pp_rho_rho_symmetric(self):
        asm, _, _ = _make_assembly()
        pairs_pp = self._extract_block_pairs(asm, 'rho', 'rho')
        for r, c in pairs_pp:
            assert (c, r) in pairs_pp, f"Missing transpose ({c},{r}) of ({r},{c})"


class TestRhsPattern:
    """RHS global rows must be unique and in range."""

    def test_rhs_unique(self):
        asm, _, _ = _make_assembly()
        assert len(np.unique(asm.rhs_global_rows)) == len(asm.rhs_global_rows)

    def test_rhs_non_negative(self):
        asm, _, _ = _make_assembly()
        assert np.all(asm.rhs_global_rows >= 0)

    def test_rhs_count(self):
        Nx_p, Ny_p = 4, 4
        asm, grid_idx, _ = _make_assembly(Nx_p, Ny_p)
        Nx_v = 2 * Nx_p - 1
        Ny_v = 2 * Ny_p - 1
        nb_v = Nx_v * Ny_v
        nb_p = Nx_p * Ny_p
        # jx + jy + rho inner nodes
        expected = 2 * nb_v + nb_p
        assert len(asm.rhs_global_rows) == expected


class TestCooLookup:
    """COO lookup must find every pair in the connectivity and reject unknown pairs."""

    def test_lookup_finds_all_pairs_vv(self):
        asm, _, _ = _make_assembly()
        inner, contrib, nnz = asm._conn[BLOCK_VV]
        found = asm.coo_lookup(BLOCK_VV, inner, contrib)
        assert np.all(found >= 0), f"{(found < 0).sum()} pairs not found in vv lookup"

    def test_lookup_finds_all_pairs_pp(self):
        asm, _, _ = _make_assembly()
        inner, contrib, nnz = asm._conn[BLOCK_PP]
        found = asm.coo_lookup(BLOCK_PP, inner, contrib)
        assert np.all(found >= 0), f"{(found < 0).sum()} pairs not found in pp lookup"

    def test_lookup_finds_all_pairs_vp(self):
        asm, _, _ = _make_assembly()
        inner, contrib, nnz = asm._conn[BLOCK_VP]
        found = asm.coo_lookup(BLOCK_VP, inner, contrib)
        assert np.all(found >= 0)

    def test_lookup_finds_all_pairs_pv(self):
        asm, _, _ = _make_assembly()
        inner, contrib, nnz = asm._conn[BLOCK_PV]
        found = asm.coo_lookup(BLOCK_PV, inner, contrib)
        assert np.all(found >= 0)

    def test_lookup_rejects_unknown_pair(self):
        asm, grid_idx, _ = _make_assembly()
        # A pair (0, nb_contributors_v - 1) is very unlikely to be in the
        # connectivity since node 0 and the last contributor are far apart.
        nb_c = grid_idx.nb_contributors_v
        row = np.array([0], dtype=np.int32)
        col = np.array([nb_c - 1], dtype=np.int32)
        result = asm.coo_lookup(BLOCK_VV, row, col)
        # Either -1 (not found) or a valid index — just check no crash.
        # The more useful check: lookup is consistent with known pairs.
        inner, contrib, _ = asm._conn[BLOCK_VV]
        in_pattern = set(zip(inner.tolist(), contrib.tolist()))
        if (0, nb_c - 1) not in in_pattern:
            assert result[0] == -1

    def test_lookup_indices_are_unique(self):
        """Each (inner, contrib) pair maps to a unique COO position."""
        asm, _, _ = _make_assembly()
        for bt in [BLOCK_VV, BLOCK_VP, BLOCK_PV, BLOCK_PP]:
            inner, contrib, nnz = asm._conn[bt]
            if nnz == 0:
                continue
            found = asm.coo_lookup(bt, inner, contrib)
            assert len(np.unique(found)) == len(found), \
                f"Non-unique COO indices in block {bt}"


class TestConnectivityEquivalence:
    """Stencil-based and element-based connectivity must produce identical pairs."""

    def _get_both_conn(self, Nx_p=4, Ny_p=4):
        decomp = _StubDecomp(Nx_p, Ny_p)
        variables = ['jx', 'jy', 'rho']
        var_to_grid = {'jx': 'v', 'jy': 'v', 'rho': 'p'}
        grid_idx = GridIndexManager(decomp, variables)

        conn_elem = Assembly._build_all_connectivity(
            grid_idx, variables, var_to_grid)
        conn_sten = Assembly._build_connectivity_stencil(
            grid_idx, variables, var_to_grid)
        return conn_elem, conn_sten

    def _pairs(self, conn, block):
        inner, contrib, _ = conn[block]
        return set(zip(inner.tolist(), contrib.tolist()))

    @pytest.mark.parametrize("block", [BLOCK_VV, BLOCK_VP, BLOCK_PV, BLOCK_PP])
    def test_nnz_match(self, block):
        conn_elem, conn_sten = self._get_both_conn()
        assert conn_elem[block][2] == conn_sten[block][2], (
            f"Block {block}: element nnz={conn_elem[block][2]}, "
            f"stencil nnz={conn_sten[block][2]}")

    @pytest.mark.parametrize("block", [BLOCK_VV, BLOCK_VP, BLOCK_PV, BLOCK_PP])
    def test_pairs_match(self, block):
        conn_elem, conn_sten = self._get_both_conn()
        pairs_e = self._pairs(conn_elem, block)
        pairs_s = self._pairs(conn_sten, block)
        missing = pairs_e - pairs_s
        extra   = pairs_s - pairs_e
        assert not missing and not extra, (
            f"Block {block}: {len(missing)} pairs in element-only, "
            f"{len(extra)} pairs in stencil-only")


class TestBlockNnzCounts:
    """nnz per block type should be consistent with node counts per triangle."""

    def test_vv_nnz_bounded(self):
        # Each square contributes at most 2 triangles × 6×6 = 72 pairs,
        # but many are shared between squares, so actual nnz << nb_sq * 72.
        asm, grid_idx, _ = _make_assembly()
        _, _, nnz = asm._conn[BLOCK_VV]
        nb_sq = grid_idx.nb_sq_v
        assert nnz <= nb_sq * 2 * 6 * 6
        assert nnz > 0

    def test_pp_nnz_bounded(self):
        asm, grid_idx, _ = _make_assembly()
        _, _, nnz = asm._conn[BLOCK_PP]
        nb_sq = grid_idx.nb_sq_p
        assert nnz <= nb_sq * 2 * 3 * 3
        assert nnz > 0

    def test_vp_nnz_bounded(self):
        asm, grid_idx, _ = _make_assembly()
        _, _, nnz = asm._conn[BLOCK_VP]
        nb_sq = grid_idx.nb_sq_v
        assert nnz <= nb_sq * 2 * 6 * 3
        assert nnz > 0

    def test_pv_nnz_bounded(self):
        asm, grid_idx, _ = _make_assembly()
        _, _, nnz = asm._conn[BLOCK_PV]
        nb_sq = grid_idx.nb_sq_p
        assert nnz <= nb_sq * 2 * 3 * 6
        assert nnz > 0

    def test_total_nnz_consistent(self):
        asm, _, _ = _make_assembly()
        assert asm.nnz == len(asm.global_rows) == len(asm.global_cols)
