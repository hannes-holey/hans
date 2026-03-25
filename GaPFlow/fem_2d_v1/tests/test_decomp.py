"""Tests for DomainDecomposition Phase 1: block decomposition + velocity grid.

Run serial:
    pytest GaPFlow/fem_2d/tests/test_decomp.py

Run multi-process:
    mpirun -n 3 pytest GaPFlow/fem_2d/tests/test_decomp.py
    mpirun -n 4 pytest GaPFlow/fem_2d/tests/test_decomp.py
"""

import numpy as np
import pytest
from mpi4py import MPI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from GaPFlow.parallel import DomainDecomposition


def skip_if_not_divisible(Nx, Ny):
    """Skip test if grid dimensions are not divisible by the MPI splits."""
    import math
    nx_splits = int(math.floor(math.sqrt(size)))
    ny_splits = size // nx_splits
    if Nx % nx_splits != 0 or Ny % ny_splits != 0:
        pytest.skip(f"({Nx},{Ny}) not divisible by splits ({nx_splits},{ny_splits}) "
                    f"for size={size}")


FEM_NUMERICS = {'solver': 'fem'}


def make_grid(Nx, Ny):
    return {
        'Nx': Nx, 'Ny': Ny,
        'Lx': float(Nx), 'Ly': float(Ny),
        'dx': 1.0, 'dy': 1.0,
        'bc_xW': ['N', 'N', 'N'],
        'bc_xE': ['N', 'N', 'N'],
        'bc_yS': ['N', 'N', 'N'],
        'bc_yN': ['N', 'N', 'N'],
    }


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Grid sizes to test: Nx/Ny must be divisible by nx_splits/ny_splits.
# For size=1: any grid works. For size=4 (2x2 splits): divisible by 2.
# For size=3 (1x3 splits): Ny divisible by 3.
GRID_SIZES_4 = [(10, 10), (12, 8), (6, 6), (20, 12), (8, 4)]
GRID_SIZES_3 = [(10, 9), (6, 12), (8, 6)]
GRID_SIZES = GRID_SIZES_4  # used for serial (size=1, all pass) and 4-proc


class TestBlockDecomposition:

    def test_subdivisions_cover_all_ranks(self):
        """nx_splits * ny_splits must equal comm.size."""
        # Use a grid divisible by any reasonable process count (LCM-friendly)
        decomp = DomainDecomposition(make_grid(12, 12), FEM_NUMERICS)
        nx, ny = decomp._nb_subdivisions
        assert nx * ny == size

    def test_subdivisions_are_2d_for_4_ranks(self):
        """For 4 ranks, expect 2x2 block decomposition."""
        if size != 4:
            pytest.skip("Only meaningful for 4-process run")
        decomp = DomainDecomposition(make_grid(10, 10), FEM_NUMERICS)
        assert decomp._nb_subdivisions == (2, 2)

    def test_subdivisions_are_1x3_for_3_ranks(self):
        """For 3 ranks, floor(sqrt(3))=1, so expect 1x3 (stripe fallback)."""
        if size != 3:
            pytest.skip("Only meaningful for 3-process run")
        decomp = DomainDecomposition(make_grid(10, 9), FEM_NUMERICS)
        assert decomp._nb_subdivisions == (1, 3)

    def test_divisibility_check_raises(self):
        """Grid dimensions not divisible by splits must raise ValueError."""
        if size == 1:
            pytest.skip("Divisibility only enforced for multi-process run")
        with pytest.raises(ValueError, match="divisible"):
            DomainDecomposition(make_grid(7, 11), FEM_NUMERICS)

    def test_velocity_loc_equals_2x_pressure_loc(self):
        """v_loc must equal 2 * p_loc for all ranks."""
        decomp = DomainDecomposition(make_grid(12, 12), FEM_NUMERICS)
        p_loc = decomp.subdomain_locations
        v_loc = decomp.subdomain_locations_v
        assert v_loc == (2 * p_loc[0], 2 * p_loc[1])

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_pressure_subdomain_sizes_cover_global(self, Nx, Ny):
        """All pressure subdomain inner sizes must tile the global domain exactly."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        all_sizes = comm.allgather(decomp.nb_subdomain_grid_pts)
        all_locs = comm.allgather(decomp.subdomain_locations)

        covered = np.zeros((Nx, Ny), dtype=int)
        for (sx, sy), (lx, ly) in zip(all_sizes, all_locs):
            covered[lx:lx+sx, ly:ly+sy] += 1

        assert np.all(covered == 1), \
            f"Pressure domain {Nx}x{Ny} not covered exactly once\n{covered}"

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_velocity_subdomain_sizes_cover_global(self, Nx, Ny):
        """All velocity subdomain sizes must tile the velocity global domain."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v

        all_sizes = comm.allgather(decomp.nb_subdomain_grid_pts_v)
        all_locs = comm.allgather(decomp.subdomain_locations_v)

        covered = np.zeros((Nx_v, Ny_v), dtype=int)
        for (sx, sy), (lx, ly) in zip(all_sizes, all_locs):
            covered[lx:lx+sx, ly:ly+sy] += 1

        assert np.all(covered == 1), \
            f"Velocity domain {Nx_v}x{Ny_v} not covered exactly once\n{covered}"


class TestVelocityGrid:

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_velocity_grid_global_size(self, Nx, Ny):
        """Velocity grid must be (2*Nx-1) x (2*Ny-1)."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        assert decomp.nb_domain_grid_pts_v == (2*Nx - 1, 2*Ny - 1)

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_velocity_padded_shape_has_depth_2_ghosts(self, Nx, Ny):
        """Velocity padded shape must be inner + 4 in each dimension."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        inner = decomp.nb_subdomain_grid_pts_v
        padded = decomp.local_shape_padded_v
        assert padded == (inner[0] + 4, inner[1] + 4)

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_pressure_padded_shape_has_depth_1_ghosts(self, Nx, Ny):
        """Pressure padded shape must be inner + 2 in each dimension."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        inner = decomp.nb_subdomain_grid_pts
        padded = decomp.local_shape_padded
        assert padded == (inner[0] + 2, inner[1] + 2)

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_velocity_icoordsg_shape(self, Nx, Ny):
        """icoordsg_v must have shape (2, Nx_v_padded, Ny_v_padded)."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        expected = (2,) + decomp.local_shape_padded_v
        assert decomp.icoordsg_v.shape == expected

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_velocity_index_mask_global_range(self, Nx, Ny):
        """index_mask_padded_global_v values must be in [0, Nx_v*Ny_v)."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v
        mask = decomp.index_mask_padded_global_v
        assert mask.min() >= 0
        assert mask.max() < Nx_v * Ny_v

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    def test_pressure_index_mask_global_range(self, Nx, Ny):
        """index_mask_padded_global values must be in [0, Nx*Ny)."""
        skip_if_not_divisible(Nx, Ny)
        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        mask = decomp.index_mask_padded_global
        assert mask.min() >= 0
        assert mask.max() < Nx * Ny

    def test_velocity_subdomain_sizes_4_process(self):
        """For 10x10 domain, 4 processes (2x2): verify non-uniform velocity sizes.

        W/S subdomains own interface nodes per plan.md convention:
          SW=(10,10), SE=(9,10), NW=(10,9), NE=(9,9)
        """
        if size != 4:
            pytest.skip("Only meaningful for 4-process run")

        decomp = DomainDecomposition(make_grid(10, 10), FEM_NUMERICS)
        all_sizes_v = comm.allgather(decomp.nb_subdomain_grid_pts_v)
        all_locs_p = comm.allgather(decomp.subdomain_locations)

        for loc_p, sv in zip(all_locs_p, all_sizes_v):
            is_W = (loc_p[0] == 0)
            is_S = (loc_p[1] == 0)
            expected_x = 10 if is_W else 9
            expected_y = 10 if is_S else 9
            assert sv == (expected_x, expected_y), \
                f"rank at p-loc {loc_p}: velocity size {sv} != ({expected_x},{expected_y})"

    @pytest.mark.parametrize("Nx,Ny", [(8, 12), (6, 4), (20, 10)])
    def test_velocity_subdomain_sizes_4_process_various_grids(self, Nx, Ny):
        """For various grids with 4 processes, velocity subdomains must tile
        the full velocity global domain exactly once."""
        if size != 4:
            pytest.skip("Only meaningful for 4-process run")

        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v

        all_sizes_v = comm.allgather(decomp.nb_subdomain_grid_pts_v)
        all_locs_v = comm.allgather(decomp.subdomain_locations_v)

        covered = np.zeros((Nx_v, Ny_v), dtype=int)
        for (sx, sy), (lx, ly) in zip(all_sizes_v, all_locs_v):
            covered[lx:lx+sx, ly:ly+sy] += 1

        assert np.all(covered == 1), \
            f"Velocity domain {Nx_v}x{Ny_v} not covered exactly once\n{covered}"


class TestGhostExchange:

    @pytest.mark.parametrize("Nx,Ny", [(10, 10), (12, 8), (6, 6)])
    def test_ghost_exchange_velocity_depth_2(self, Nx, Ny):
        """Ghost exchange on velocity grid: inner values must appear in neighbour ghosts."""
        skip_if_not_divisible(Nx, Ny)
        if size < 2:
            pytest.skip("Ghost exchange only meaningful for multi-process run")

        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v

        fc_v = decomp._decomp_v.collection
        field = fc_v.real_field(f'test_v_{Nx}_{Ny}', 1)

        inner = decomp.nb_subdomain_grid_pts_v
        loc = decomp.subdomain_locations_v
        p = field.p[0]
        for i in range(inner[0]):
            for j in range(inner[1]):
                p[i, j] = float((loc[0] + i) + (loc[1] + j) * Nx_v)

        decomp._exchange_ghosts_v(field)

        pg = field.pg[0]
        mask = decomp.index_mask_padded_global_v

        # Both ghost depth layers on each inter-subdomain side must equal global flat index
        for d in [1, 0]:
            if not decomp.is_at_xW:
                np.testing.assert_allclose(pg[d, 2:-2], mask[d, 2:-2],
                    err_msg=f"W depth-{2-d} ghost mismatch on {Nx}x{Ny}")
            if not decomp.is_at_xE:
                np.testing.assert_allclose(pg[-(d+1), 2:-2], mask[-(d+1), 2:-2],
                    err_msg=f"E depth-{2-d} ghost mismatch on {Nx}x{Ny}")
            if not decomp.is_at_yS:
                np.testing.assert_allclose(pg[2:-2, d], mask[2:-2, d],
                    err_msg=f"S depth-{2-d} ghost mismatch on {Nx}x{Ny}")
            if not decomp.is_at_yN:
                np.testing.assert_allclose(pg[2:-2, -(d+1)], mask[2:-2, -(d+1)],
                    err_msg=f"N depth-{2-d} ghost mismatch on {Nx}x{Ny}")

    @pytest.mark.parametrize("Nx,Ny", [(10, 10), (12, 8), (6, 6)])
    def test_ghost_exchange_pressure_depth_1(self, Nx, Ny):
        """Ghost exchange on pressure grid still works correctly after changes."""
        skip_if_not_divisible(Nx, Ny)
        if size < 2:
            pytest.skip("Ghost exchange only meaningful for multi-process run")

        decomp = DomainDecomposition(make_grid(Nx, Ny), FEM_NUMERICS)

        fc = decomp._decomp.collection
        field = fc.real_field(f'test_p_{Nx}_{Ny}', 1)

        inner = decomp.nb_subdomain_grid_pts
        loc = decomp.subdomain_locations
        p = field.p[0]
        for i in range(inner[0]):
            for j in range(inner[1]):
                p[i, j] = float((loc[0] + i) + (loc[1] + j) * Nx)

        decomp._exchange_ghosts(field)

        pg = field.pg[0]
        mask = decomp.index_mask_padded_global

        if not decomp.is_at_xW:
            np.testing.assert_allclose(
                pg[0, 1:-1], mask[0, 1:-1],
                err_msg=f"W ghost mismatch on pressure grid {Nx}x{Ny}")

        if not decomp.is_at_xE:
            np.testing.assert_allclose(
                pg[-1, 1:-1], mask[-1, 1:-1],
                err_msg=f"E ghost mismatch on pressure grid {Nx}x{Ny}")

        if not decomp.is_at_yS:
            np.testing.assert_allclose(
                pg[1:-1, 0], mask[1:-1, 0],
                err_msg=f"S ghost mismatch on pressure grid {Nx}x{Ny}")

        if not decomp.is_at_yN:
            np.testing.assert_allclose(
                pg[1:-1, -1], mask[1:-1, -1],
                err_msg=f"N ghost mismatch on pressure grid {Nx}x{Ny}")


class TestFemGating:

    def test_no_numerics_skips_decomp_v(self):
        """Without numerics, _decomp_v must be None and _v properties must assert."""
        decomp = DomainDecomposition(make_grid(10, 10))
        assert decomp._decomp_v is None
        assert not decomp._is_fem
        with pytest.raises(AssertionError):
            _ = decomp.nb_subdomain_grid_pts_v
        with pytest.raises(AssertionError):
            _ = decomp.subdomain_locations_v
        with pytest.raises(AssertionError):
            _ = decomp.nb_domain_grid_pts_v
        with pytest.raises(AssertionError):
            _ = decomp.index_mask_padded_global_v

    def test_non_fem_solver_skips_decomp_v(self):
        """With solver != 'fem', _decomp_v must be None."""
        decomp = DomainDecomposition(make_grid(10, 10), {'solver': 'explicit'})
        assert decomp._decomp_v is None
        assert not decomp._is_fem

    def test_fem_solver_initialises_decomp_v(self):
        """With solver='fem' and divisible grid, _decomp_v must be initialised."""
        decomp = DomainDecomposition(make_grid(12, 12), FEM_NUMERICS)
        assert decomp._decomp_v is not None
        assert decomp._is_fem


class TestMassFluxBCs:
    """Tests for _apply_mass_flux_bcs via update_ghosts_v."""

    def _make_problem_stub(self, grid, jx_val=0.0, jy_val=0.0):
        """Minimal problem stub with no callbacks and static D values."""
        class Stub:
            pass
        p = Stub()
        p._bc_callbacks = {}
        return p

    @pytest.mark.parametrize("Nx,Ny", [(10, 10), (12, 8)])
    def test_neumann_bc_both_ghost_layers(self, Nx, Ny):
        """Neumann BC: both ghost layers must equal the adjacent inner node."""
        skip_if_not_divisible(Nx, Ny)
        grid = make_grid(Nx, Ny)  # all-Neumann by default
        decomp = DomainDecomposition(grid, FEM_NUMERICS)

        fc_v = decomp._decomp_v.collection
        jx_field = fc_v.real_field('jx_test_N', 1)
        jy_field = fc_v.real_field('jy_test_N', 1)

        # Fill inner with a known pattern
        inner = decomp.nb_subdomain_grid_pts_v
        loc = decomp.subdomain_locations_v
        Nx_v, Ny_v = decomp.nb_domain_grid_pts_v
        for field in [jx_field, jy_field]:
            p = field.p[0]
            for i in range(inner[0]):
                for j in range(inner[1]):
                    p[i, j] = float((loc[0] + i) + (loc[1] + j) * Nx_v) + 1.0

        problem = self._make_problem_stub(grid)
        decomp.update_ghosts(
            exchange_specs=[(jx_field, 'P2'), (jy_field, 'P2')],
            bc_specs=[(jx_field.pg[0], 'jx', 'P2_nodal'), (jy_field.pg[0], 'jy', 'P2_nodal')],
            problem=problem,
        )

        for field in [jx_field, jy_field]:
            pg = field.pg[0]
            # ghost1 is index 1 (or -2), ghost2 is index 0 (or -1), inner is index 2 (or -3)
            if decomp.is_at_xW:
                np.testing.assert_array_equal(pg[1, 2:-2], pg[2, 2:-2],
                    err_msg="W ghost1 != inner for Neumann")
                np.testing.assert_array_equal(pg[0, 2:-2], pg[2, 2:-2],
                    err_msg="W ghost2 != inner for Neumann")
            if decomp.is_at_xE:
                np.testing.assert_array_equal(pg[-2, 2:-2], pg[-3, 2:-2],
                    err_msg="E ghost1 != inner for Neumann")
                np.testing.assert_array_equal(pg[-1, 2:-2], pg[-3, 2:-2],
                    err_msg="E ghost2 != inner for Neumann")
            if decomp.is_at_yS:
                np.testing.assert_array_equal(pg[2:-2, 1], pg[2:-2, 2],
                    err_msg="S ghost1 != inner for Neumann")
                np.testing.assert_array_equal(pg[2:-2, 0], pg[2:-2, 2],
                    err_msg="S ghost2 != inner for Neumann")
            if decomp.is_at_yN:
                np.testing.assert_array_equal(pg[2:-2, -2], pg[2:-2, -3],
                    err_msg="N ghost1 != inner for Neumann")
                np.testing.assert_array_equal(pg[2:-2, -1], pg[2:-2, -3],
                    err_msg="N ghost2 != inner for Neumann")

    @pytest.mark.parametrize("Nx,Ny", [(10, 10), (12, 8)])
    def test_dirichlet_bc_both_ghost_layers(self, Nx, Ny):
        """Dirichlet BC: both ghost layers must equal the prescribed value."""
        skip_if_not_divisible(Nx, Ny)
        grid = make_grid(Nx, Ny)
        grid['bc_xW'] = ['N', 'D', 'D']
        grid['bc_xE'] = ['N', 'D', 'D']
        grid['bc_yS'] = ['N', 'D', 'D']
        grid['bc_yN'] = ['N', 'D', 'D']
        grid['bc_xW_D_val'] = [0.0, 2.5, 3.5]
        grid['bc_xE_D_val'] = [0.0, 2.5, 3.5]
        grid['bc_yS_D_val'] = [0.0, 2.5, 3.5]
        grid['bc_yN_D_val'] = [0.0, 2.5, 3.5]
        decomp = DomainDecomposition(grid, FEM_NUMERICS)

        fc_v = decomp._decomp_v.collection
        jx_field = fc_v.real_field('jx_test_D', 1)
        jy_field = fc_v.real_field('jy_test_D', 1)
        # Fill with non-zero so a zero Dirichlet would be distinguishable
        jx_field.pg[0][:] = 99.0
        jy_field.pg[0][:] = 99.0

        problem = self._make_problem_stub(grid)
        decomp.update_ghosts(
            exchange_specs=[(jx_field, 'P2'), (jy_field, 'P2')],
            bc_specs=[(jx_field.pg[0], 'jx', 'P2_nodal'), (jy_field.pg[0], 'jy', 'P2_nodal')],
            problem=problem,
        )

        # jx (var_idx=1): D val = 2.5; jy (var_idx=2): D val = 3.5
        for field, expected in [(jx_field, 2.5), (jy_field, 3.5)]:
            pg = field.pg[0]
            if decomp.is_at_xW:
                np.testing.assert_array_equal(pg[1, :], expected,
                    err_msg=f"W ghost1 != {expected} for Dirichlet")
                np.testing.assert_array_equal(pg[0, :], expected,
                    err_msg=f"W ghost2 != {expected} for Dirichlet")
            if decomp.is_at_xE:
                np.testing.assert_array_equal(pg[-2, :], expected,
                    err_msg=f"E ghost1 != {expected} for Dirichlet")
                np.testing.assert_array_equal(pg[-1, :], expected,
                    err_msg=f"E ghost2 != {expected} for Dirichlet")
            if decomp.is_at_yS:
                np.testing.assert_array_equal(pg[:, 1], expected,
                    err_msg=f"S ghost1 != {expected} for Dirichlet")
                np.testing.assert_array_equal(pg[:, 0], expected,
                    err_msg=f"S ghost2 != {expected} for Dirichlet")
            if decomp.is_at_yN:
                np.testing.assert_array_equal(pg[:, -2], expected,
                    err_msg=f"N ghost1 != {expected} for Dirichlet")
                np.testing.assert_array_equal(pg[:, -1], expected,
                    err_msg=f"N ghost2 != {expected} for Dirichlet")
