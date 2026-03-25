"""Tests for GridIndexManager (Phase 2).

Run serial:
    pytest GaPFlow/fem_2d/tests/test_grid_index.py

Run multi-process:
    mpirun -n 4 pytest GaPFlow/fem_2d/tests/test_grid_index.py
"""

import math
import numpy as np
import pytest
from mpi4py import MPI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from GaPFlow.parallel import DomainDecomposition
from GaPFlow.fem_2d.grid_index import GridIndexManager

FEM_NUMERICS = {'solver': 'fem'}
VARIABLES = ['rho', 'jx', 'jy']

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def skip_if_not_divisible(Nx, Ny):
    nx_splits = int(math.floor(math.sqrt(size)))
    ny_splits = size // nx_splits
    if Nx % nx_splits != 0 or Ny % ny_splits != 0:
        pytest.skip(f"({Nx},{Ny}) not divisible by splits ({nx_splits},{ny_splits}) "
                    f"for size={size}")


def make_grid(Nx, Ny, bc_type='N'):
    """All-Neumann or all-Dirichlet grid."""
    bc = [bc_type] * 3
    g = {
        'Nx': Nx, 'Ny': Ny,
        'Lx': float(Nx), 'Ly': float(Ny),
        'dx': 1.0, 'dy': 1.0,
        'bc_xW': bc, 'bc_xE': bc,
        'bc_yS': bc, 'bc_yN': bc,
    }
    if bc_type == 'D':
        g['bc_xW_D_val'] = [0.0, 0.0, 0.0]
        g['bc_xE_D_val'] = [0.0, 0.0, 0.0]
        g['bc_yS_D_val'] = [0.0, 0.0, 0.0]
        g['bc_yN_D_val'] = [0.0, 0.0, 0.0]
    return g


def make_gim(Nx, Ny, bc_type='N'):
    decomp = DomainDecomposition(make_grid(Nx, Ny, bc_type), FEM_NUMERICS)
    return GridIndexManager(decomp, VARIABLES)


# Marker applied to all classes that assume serial (full grid on one rank)
serial_only = pytest.mark.skipif(size > 1, reason="Serial-only test")


# ===========================================================================
# Pressure grid
# ===========================================================================

@serial_only
class TestPressureInnerMask:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6), (10, 8)])
    def test_inner_pts_sequential(self, Nx, Ny):
        """Inner nodes get 0..N-1, ghost nodes get -1."""
        gim = make_gim(Nx, Ny)
        mask = gim.index_mask_inner_local_p
        inner = mask[1:-1, 1:-1]
        assert inner.min() == 0
        assert inner.max() == Nx * Ny - 1
        assert set(inner.flatten()) == set(range(Nx * Ny))
        # ghost border is all -1
        assert np.all(mask[0, :] == -1)
        assert np.all(mask[-1, :] == -1)
        assert np.all(mask[:, 0] == -1)
        assert np.all(mask[:, -1] == -1)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_inner_mask_column_major(self, Nx, Ny):
        """Inner indexing is Fortran order: index increases along x (axis-0) first."""
        gim = make_gim(Nx, Ny)
        mask = gim.index_mask_inner_local_p
        # F-order: mask[1,1]=0, mask[2,1]=1, mask[3,1]=2
        assert mask[1, 1] == 0
        assert mask[2, 1] == 1
        assert mask[3, 1] == 2


@serial_only
class TestPressurePaddedMask:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_padded_equals_inner_on_inner_nodes(self, Nx, Ny):
        """Padded mask must equal inner mask on all inner nodes."""
        gim = make_gim(Nx, Ny)
        inner = gim.index_mask_inner_local_p
        padded = gim.index_mask_padded_local_p('rho')
        np.testing.assert_array_equal(padded[1:-1, 1:-1], inner[1:-1, 1:-1])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_neumann_ghost_forwards_to_inner(self, Nx, Ny):
        """Neumann ghost nodes must get the same local index as adjacent inner node."""
        gim = make_gim(Nx, Ny, bc_type='N')  # all Neumann
        mask = gim.index_mask_padded_local_p('jx')  # jx is index 1, Neumann
        # W ghost row should equal first inner row
        np.testing.assert_array_equal(mask[0, :], mask[1, :])
        # E ghost row should equal last inner row
        np.testing.assert_array_equal(mask[-1, :], mask[-2, :])
        # S ghost col
        np.testing.assert_array_equal(mask[:, 0], mask[:, 1])
        # N ghost col
        np.testing.assert_array_equal(mask[:, -1], mask[:, -2])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_dirichlet_ghost_is_minus1(self, Nx, Ny):
        """Dirichlet ghost nodes must remain -1."""
        gim = make_gim(Nx, Ny, bc_type='D')  # all Dirichlet
        mask = gim.index_mask_padded_local_p('jx')
        assert np.all(mask[0, :] == -1)
        assert np.all(mask[-1, :] == -1)
        assert np.all(mask[:, 0] == -1)
        assert np.all(mask[:, -1] == -1)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_no_neumann_on_rho_dirichlet_boundary(self, Nx, Ny):
        """rho has Dirichlet BC: ghost must be -1 even when jx/jy have Neumann."""
        grid = make_grid(Nx, Ny)
        # mixed: rho=D, jx=N, jy=N
        grid['bc_xW'] = ['D', 'N', 'N']
        grid['bc_xE'] = ['D', 'N', 'N']
        grid['bc_yS'] = ['D', 'N', 'N']
        grid['bc_yN'] = ['D', 'N', 'N']
        grid['bc_xW_D_val'] = [0.0, 0.0, 0.0]
        grid['bc_xE_D_val'] = [0.0, 0.0, 0.0]
        grid['bc_yS_D_val'] = [0.0, 0.0, 0.0]
        grid['bc_yN_D_val'] = [0.0, 0.0, 0.0]
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)
        mask_rho = gim.index_mask_padded_local_p('rho')
        mask_jx  = gim.index_mask_padded_local_p('jx')
        # rho ghost: Dirichlet → -1
        assert np.all(mask_rho[0, :] == -1)
        # jx ghost: Neumann → forwarded
        np.testing.assert_array_equal(mask_jx[0, :], mask_jx[1, :])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_neumann_ghost_has_index_after_forwarding(self, Nx, Ny):
        """After Neumann forwarding, no ghost node at a physical boundary is -1."""
        gim = make_gim(Nx, Ny, bc_type='N')
        # With var='jx' (Neumann), all ghost nodes should be forwarded to inner
        mask = gim.index_mask_padded_local_p('jx')
        assert np.all(mask >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_l2g_roundtrip_p(self, Nx, Ny):
        """l2g_list_p must map inner local indices to correct global indices."""
        gim = make_gim(Nx, Ny)
        mask_local  = gim.index_mask_padded_local_p('')
        mask_global = gim._decomp.index_mask_padded_global
        l2g = gim.l2g_list_p
        valid = mask_local >= 0
        # For each valid node: l2g[local_idx] == global_idx
        np.testing.assert_array_equal(
            l2g[mask_local[valid]], mask_global[valid])


# ===========================================================================
# Mass flux (P2) grid
# ===========================================================================

@serial_only
class TestMassFluxInnerMask:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_inner_pts_sequential(self, Nx, Ny):
        """Inner nodes get 0..Nx_v*Ny_v-1, ghost layers are -1."""
        gim = make_gim(Nx, Ny)
        mask = gim.index_mask_inner_local_v
        # In serial: Nx_v = 2*Nx-1, Ny_v = 2*Ny-1
        Nx_v, Ny_v = gim.Nx_v_inner, gim.Ny_v_inner
        inner = mask[2:-2, 2:-2]
        assert inner.min() == 0
        assert inner.max() == Nx_v * Ny_v - 1
        assert set(inner.flatten()) == set(range(Nx_v * Ny_v))
        # Both ghost layers on each side must be -1
        assert np.all(mask[:2, :] == -1)
        assert np.all(mask[-2:, :] == -1)
        assert np.all(mask[:, :2] == -1)
        assert np.all(mask[:, -2:] == -1)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_inner_mask_column_major(self, Nx, Ny):
        """Inner indexing is Fortran order: index increases along x (axis-0) first."""
        gim = make_gim(Nx, Ny)
        mask = gim.index_mask_inner_local_v
        # F-order: mask[2,2]=0, mask[3,2]=1, mask[4,2]=2
        assert mask[2, 2] == 0
        assert mask[3, 2] == 1
        assert mask[4, 2] == 2


@serial_only
class TestMassFluxPaddedMask:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_padded_equals_inner_on_inner_nodes(self, Nx, Ny):
        gim = make_gim(Nx, Ny)
        inner  = gim.index_mask_inner_local_v
        padded = gim.index_mask_padded_local_v('jx')
        np.testing.assert_array_equal(padded[2:-2, 2:-2], inner[2:-2, 2:-2])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_neumann_both_ghost_layers_forward_to_inner(self, Nx, Ny):
        """Both ghost layers at a Neumann boundary must map to the adjacent inner node."""
        gim = make_gim(Nx, Ny, bc_type='N')
        mask = gim.index_mask_padded_local_v('jx')
        # W: ghost layers 0 and 1 should equal inner layer 2
        np.testing.assert_array_equal(mask[0, :], mask[2, :])
        np.testing.assert_array_equal(mask[1, :], mask[2, :])
        # E: ghost layers -1 and -2 should equal inner layer -3
        np.testing.assert_array_equal(mask[-1, :], mask[-3, :])
        np.testing.assert_array_equal(mask[-2, :], mask[-3, :])
        # S
        np.testing.assert_array_equal(mask[:, 0], mask[:, 2])
        np.testing.assert_array_equal(mask[:, 1], mask[:, 2])
        # N
        np.testing.assert_array_equal(mask[:, -1], mask[:, -3])
        np.testing.assert_array_equal(mask[:, -2], mask[:, -3])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_dirichlet_both_ghost_layers_minus1(self, Nx, Ny):
        """Both ghost layers at a Dirichlet boundary must be -1."""
        gim = make_gim(Nx, Ny, bc_type='D')
        mask = gim.index_mask_padded_local_v('jx')
        assert np.all(mask[:2, :] == -1)
        assert np.all(mask[-2:, :] == -1)
        assert np.all(mask[:, :2] == -1)
        assert np.all(mask[:, -2:] == -1)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_l2g_roundtrip_v(self, Nx, Ny):
        """l2g_list_v must map inner local indices to correct global indices."""
        gim = make_gim(Nx, Ny)
        mask_local  = gim.index_mask_padded_local_v('')
        mask_global = gim._decomp.index_mask_padded_global_v
        l2g = gim.l2g_list_v
        valid = mask_local >= 0
        np.testing.assert_array_equal(
            l2g[mask_local[valid]], mask_global[valid])


# ===========================================================================
# Square connectivity
# ===========================================================================

@serial_only
class TestSquareConnectivity:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_counts(self, Nx, Ny):
        """Number of squares must be (Nx_inner+1) * (Ny_inner+1)."""
        gim = make_gim(Nx, Ny)
        assert gim.nb_sq_p == (Nx + 1) * (Ny + 1)
        assert gim.nb_sq_v == gim.nb_sq_p

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_TO_inner_p_shape(self, Nx, Ny):
        gim = make_gim(Nx, Ny)
        assert gim.sq_TO_inner_p.shape == (gim.nb_sq_p, 4)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_TO_inner_v_shape(self, Nx, Ny):
        gim = make_gim(Nx, Ny)
        assert gim.sq_TO_inner_v.shape == (gim.nb_sq_v, 9)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_FROM_padded_p_shape(self, Nx, Ny):
        gim = make_gim(Nx, Ny)
        assert gim.sq_FROM_padded_p('rho').shape == (gim.nb_sq_p, 4)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_FROM_padded_v_shape(self, Nx, Ny):
        gim = make_gim(Nx, Ny)
        assert gim.sq_FROM_padded_v('jx').shape == (gim.nb_sq_v, 9)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_TO_inner_p_inner_squares_positive(self, Nx, Ny):
        """Interior squares (not at boundary) must have all 4 corners >= 0."""
        gim = make_gim(Nx, Ny)
        m = gim.sq_TO_inner_p
        sq_x = gim.sq_x_arr_p
        sq_y = gim.sq_y_arr_p
        # Interior squares: sx in [1, Nx_inner-1], sy in [1, Ny_inner-1]
        interior = ((sq_x >= 1) & (sq_x < Nx) & (sq_y >= 1) & (sq_y < Ny))
        assert np.all(m[interior] >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_TO_inner_v_interior_squares_positive(self, Nx, Ny):
        """Interior P2 squares must have all 9 nodes >= 0."""
        gim = make_gim(Nx, Ny)
        m = gim.sq_TO_inner_v
        sq_x = gim.sq_x_arr_p   # same square grid as P1
        sq_y = gim.sq_y_arr_p
        interior = ((sq_x >= 1) & (sq_x < Nx) & (sq_y >= 1) & (sq_y < Ny))
        assert np.all(m[interior] >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_v_origin_at_2x_p_origin(self, Nx, Ny):
        """P2 square origins must be exactly 2 * P1 square origins on padded grid."""
        gim = make_gim(Nx, Ny)
        np.testing.assert_array_equal(gim.sq_x_arr_v, 2 * gim.sq_x_arr_p)
        np.testing.assert_array_equal(gim.sq_y_arr_v, 2 * gim.sq_y_arr_p)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_inner_p_indices_in_range(self, Nx, Ny):
        """All non-(-1) entries in sq_TO_inner_p must be in [0, Nx*Ny)."""
        gim = make_gim(Nx, Ny)
        m = gim.sq_TO_inner_p
        valid = m[m >= 0]
        assert valid.max() < Nx * Ny

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_inner_v_indices_in_range(self, Nx, Ny):
        """All non-(-1) entries in sq_TO_inner_v must be in [0, Nx_v*Ny_v)."""
        gim = make_gim(Nx, Ny)
        Nx_v, Ny_v = gim.Nx_v_inner, gim.Ny_v_inner
        m = gim.sq_TO_inner_v
        valid = m[m >= 0]
        assert valid.max() < Nx_v * Ny_v

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_sq_TO_inner_v_boundary_squares_have_minus1(self, Nx, Ny):
        """Boundary squares must have -1 for nodes that fall outside inner domain."""
        gim = make_gim(Nx, Ny)
        m = gim.sq_TO_inner_v
        sq_x = gim.sq_x_arr_p
        sq_y = gim.sq_y_arr_p
        # Squares on the W edge (sq_x == 0): bl, tl, ml nodes are in ghost zone
        # Node offsets (0,0),(0,2),(0,1) → x index in padded = sq_x_v + 0 = 0,1 → ghost
        w_edge = sq_x == 0
        # columns 0,2,4 of sq_TO_inner_v = bl(0,0), tl(0,2), ml(0,1) — all outside inner
        assert np.all(m[w_edge][:, [0, 2, 4]] == -1)
        # Squares on the S edge (sq_y == 0): bl, br, bm nodes are in ghost zone
        s_edge = sq_y == 0
        assert np.all(m[s_edge][:, [0, 1, 5]] == -1)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_FROM_padded_v_neumann_no_minus1(self, Nx, Ny):
        """sq_FROM_padded_v with Neumann var must have no -1 (all nodes contribute)."""
        gim = make_gim(Nx, Ny, bc_type='N')
        m = gim.sq_FROM_padded_v('jx')
        assert np.all(m >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_sq_FROM_padded_v_dirichlet_boundary_minus1(self, Nx, Ny):
        """sq_FROM_padded_v with Dirichlet var: boundary squares must have -1 for ghost nodes."""
        gim = make_gim(Nx, Ny, bc_type='D')
        m = gim.sq_FROM_padded_v('jx')
        sq_x = gim.sq_x_arr_p
        sq_y = gim.sq_y_arr_p
        # W edge squares: ghost nodes (offset x=0,1 in padded) must be -1
        w_edge = sq_x == 0
        assert np.all(m[w_edge][:, [0, 2, 4]] == -1)
        s_edge = sq_y == 0
        assert np.all(m[s_edge][:, [0, 1, 5]] == -1)


# ===========================================================================
# nb_contributors
# ===========================================================================

@serial_only
class TestNbContributors:

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_nb_contributors_p_neumann(self, Nx, Ny):
        """Neumann P1: nb_contributors_p equals inner nodes only.

        Physical boundary ghosts are excluded from index assignment (they are
        forwarded to inner nodes via Neumann handling, not assigned new indices).
        """
        gim = make_gim(Nx, Ny, bc_type='N')
        assert gim.nb_contributors_p == gim.Nx_p_inner * gim.Ny_p_inner

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_nb_contributors_p_dirichlet(self, Nx, Ny):
        """Dirichlet P1: nb_contributors_p equals inner nodes only (ghosts are -1)."""
        gim = make_gim(Nx, Ny, bc_type='D')
        assert gim.nb_contributors_p == gim.Nx_p_inner * gim.Ny_p_inner

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_nb_contributors_v_neumann(self, Nx, Ny):
        """Neumann P2: nb_contributors_v equals inner nodes only."""
        gim = make_gim(Nx, Ny, bc_type='N')
        assert gim.nb_contributors_v == gim.Nx_v_inner * gim.Ny_v_inner

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4), (4, 6)])
    def test_nb_contributors_v_dirichlet(self, Nx, Ny):
        """Dirichlet P2: nb_contributors_v equals inner nodes only."""
        gim = make_gim(Nx, Ny, bc_type='D')
        assert gim.nb_contributors_v == gim.Nx_v_inner * gim.Ny_v_inner


# ===========================================================================
# Periodic wrapping (serial)
# ===========================================================================

@serial_only
class TestPeriodicWrapping:

    def _make_periodic_gim(self, Nx, Ny):
        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['P', 'P', 'P'], 'bc_xE': ['P', 'P', 'P'],
            'bc_yS': ['P', 'P', 'P'], 'bc_yN': ['P', 'P', 'P'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        return GridIndexManager(decomp, VARIABLES)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_p_ghost_wraps_to_opposite_inner(self, Nx, Ny):
        """Fully periodic P1: W ghost wraps to last inner x, E ghost to first inner x."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask = gim.index_mask_padded_local_p('')
        # W ghost (x=0) must equal last inner row (x = Nx_p_padded-2)
        np.testing.assert_array_equal(mask[0, :], mask[gim.Nx_p_padded - 2, :])
        # E ghost (x=-1) must equal first inner row (x=1)
        np.testing.assert_array_equal(mask[-1, :], mask[1, :])
        # S ghost (y=0) must equal last inner col (y = Ny_p_padded-2)
        np.testing.assert_array_equal(mask[:, 0], mask[:, gim.Ny_p_padded - 2])
        # N ghost (y=-1) must equal first inner col (y=1)
        np.testing.assert_array_equal(mask[:, -1], mask[:, 1])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_p_no_minus1(self, Nx, Ny):
        """Fully periodic P1: no ghost node should be -1."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask = gim.index_mask_padded_local_p('')
        assert np.all(mask >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_v_ghost_wraps_to_opposite_inner(self, Nx, Ny):
        """Fully periodic P2: both W ghost layers wrap to last two inner x rows."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask = gim.index_mask_padded_local_v('')
        Nxp = gim.Nx_v_padded
        Nyp = gim.Ny_v_padded
        # W ghost layer 0 → last inner x (Nxp-4), layer 1 → second-to-last (Nxp-3)
        np.testing.assert_array_equal(mask[0, :], mask[Nxp - 4, :])
        np.testing.assert_array_equal(mask[1, :], mask[Nxp - 3, :])
        # E ghost layer -1 → first inner x (2), layer -2 → second inner x (3)
        np.testing.assert_array_equal(mask[-1, :], mask[3, :])
        np.testing.assert_array_equal(mask[-2, :], mask[2, :])
        # S
        np.testing.assert_array_equal(mask[:, 0], mask[:, Nyp - 4])
        np.testing.assert_array_equal(mask[:, 1], mask[:, Nyp - 3])
        # N
        np.testing.assert_array_equal(mask[:, -1], mask[:, 3])
        np.testing.assert_array_equal(mask[:, -2], mask[:, 2])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_v_no_minus1(self, Nx, Ny):
        """Fully periodic P2: no ghost node should be -1."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask = gim.index_mask_padded_local_v('')
        assert np.all(mask >= 0)

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_p_l2g_roundtrip(self, Nx, Ny):
        """Fully periodic P1 l2g round-trip must hold."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask_local  = gim.index_mask_padded_local_p('')
        mask_global = gim._decomp.index_mask_padded_global
        l2g = gim.l2g_list_p
        valid = mask_local >= 0
        np.testing.assert_array_equal(l2g[mask_local[valid]], mask_global[valid])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_periodic_v_l2g_roundtrip(self, Nx, Ny):
        """Fully periodic P2 l2g round-trip must hold."""
        gim = self._make_periodic_gim(Nx, Ny)
        mask_local  = gim.index_mask_padded_local_v('')
        mask_global = gim._decomp.index_mask_padded_global_v
        l2g = gim.l2g_list_v
        valid = mask_local >= 0
        np.testing.assert_array_equal(l2g[mask_local[valid]], mask_global[valid])

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (6, 4)])
    def test_mixed_periodic_x_only(self, Nx, Ny):
        """Semi-periodic (x periodic, y Neumann): x ghosts wrap, y ghosts forward."""
        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['P', 'P', 'P'], 'bc_xE': ['P', 'P', 'P'],
            'bc_yS': ['N', 'N', 'N'], 'bc_yN': ['N', 'N', 'N'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)
        mask = gim.index_mask_padded_local_p('jx')
        # x: periodic wrap
        np.testing.assert_array_equal(mask[0, :], mask[gim.Nx_p_padded - 2, :])
        np.testing.assert_array_equal(mask[-1, :], mask[1, :])
        # y: Neumann forward
        np.testing.assert_array_equal(mask[:, 0], mask[:, 1])
        np.testing.assert_array_equal(mask[:, -1], mask[:, -2])


# ===========================================================================
# MPI: inter-subdomain ghost index assignment
# ===========================================================================

class TestMPIContributors:

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 8)])
    def test_nb_contributors_p_larger_on_interior_rank(self, Nx, Ny):
        """On a rank with no physical boundary, nb_contributors_p must exceed
        Nx_p_inner * Ny_p_inner due to inter-subdomain ghost nodes."""
        if size < 2:
            pytest.skip("Only meaningful for multi-process run")
        skip_if_not_divisible(Nx, Ny)

        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['N', 'N', 'N'], 'bc_xE': ['N', 'N', 'N'],
            'bc_yS': ['N', 'N', 'N'], 'bc_yN': ['N', 'N', 'N'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)

        nb_inner = gim.Nx_p_inner * gim.Ny_p_inner
        has_physical_boundary = (decomp.is_at_xW or decomp.is_at_xE or
                                  decomp.is_at_yS or decomp.is_at_yN)

        if not has_physical_boundary:
            assert gim.nb_contributors_p > nb_inner, (
                f"rank {rank}: expected nb_contributors_p > {nb_inner}, "
                f"got {gim.nb_contributors_p}")
        else:
            # Physical boundary ranks: ghosts excluded from cur_val loop,
            # but may still have inter-subdomain ghosts on non-boundary sides
            assert gim.nb_contributors_p >= nb_inner

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 8)])
    def test_nb_contributors_v_larger_on_interior_rank(self, Nx, Ny):
        """Same check for the P2 mass flux grid."""
        if size < 2:
            pytest.skip("Only meaningful for multi-process run")
        skip_if_not_divisible(Nx, Ny)

        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['N', 'N', 'N'], 'bc_xE': ['N', 'N', 'N'],
            'bc_yS': ['N', 'N', 'N'], 'bc_yN': ['N', 'N', 'N'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)

        nb_inner = gim.Nx_v_inner * gim.Ny_v_inner
        has_physical_boundary = (decomp.is_at_xW or decomp.is_at_xE or
                                  decomp.is_at_yS or decomp.is_at_yN)

        if not has_physical_boundary:
            assert gim.nb_contributors_v > nb_inner, (
                f"rank {rank}: expected nb_contributors_v > {nb_inner}, "
                f"got {gim.nb_contributors_v}")
        else:
            assert gim.nb_contributors_v >= nb_inner

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 8)])
    def test_l2g_roundtrip_p_mpi(self, Nx, Ny):
        """l2g_list_p round-trip must hold on all ranks in MPI runs."""
        if size < 2:
            pytest.skip("Only meaningful for multi-process run")
        skip_if_not_divisible(Nx, Ny)

        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['N', 'N', 'N'], 'bc_xE': ['N', 'N', 'N'],
            'bc_yS': ['N', 'N', 'N'], 'bc_yN': ['N', 'N', 'N'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)
        mask_local  = gim.index_mask_padded_local_p('')
        mask_global = decomp.index_mask_padded_global
        l2g = gim.l2g_list_p
        valid = mask_local >= 0
        np.testing.assert_array_equal(l2g[mask_local[valid]], mask_global[valid])

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 8)])
    def test_l2g_roundtrip_v_mpi(self, Nx, Ny):
        """l2g_list_v round-trip must hold on all ranks in MPI runs."""
        if size < 2:
            pytest.skip("Only meaningful for multi-process run")
        skip_if_not_divisible(Nx, Ny)

        grid = {
            'Nx': Nx, 'Ny': Ny,
            'Lx': float(Nx), 'Ly': float(Ny),
            'dx': 1.0, 'dy': 1.0,
            'bc_xW': ['N', 'N', 'N'], 'bc_xE': ['N', 'N', 'N'],
            'bc_yS': ['N', 'N', 'N'], 'bc_yN': ['N', 'N', 'N'],
        }
        decomp = DomainDecomposition(grid, FEM_NUMERICS)
        gim = GridIndexManager(decomp, VARIABLES)
        mask_local  = gim.index_mask_padded_local_v('')
        mask_global = decomp.index_mask_padded_global_v
        l2g = gim.l2g_list_v
        valid = mask_local >= 0
        np.testing.assert_array_equal(l2g[mask_local[valid]], mask_global[valid])
