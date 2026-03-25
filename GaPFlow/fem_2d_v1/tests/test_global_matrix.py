"""Tests for fem_2d/global_matrix.py — Phase 3."""
import pytest
from GaPFlow.fem_2d.global_matrix import field_to_global, global_to_field, _n_density_before


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_dofs(cols_v: int, M: int, energy: bool) -> int:
    """Total number of global DOFs for a grid with cols_v mass-flux columns
    and M density columns, with rows_v = 2*rows_p - 1 fine rows."""
    rows_p = M  # square grid assumption: rows_p == M
    rows_v = 2 * rows_p - 1
    n_rho = 2 if energy else 1
    # mass-flux DOFs
    total = 2 * cols_v * rows_v
    # density DOFs (at even-even positions)
    total += n_rho * M * rows_p
    return total


def _all_global_indices(cols_v: int, M: int, energy: bool):
    """Enumerate all (field_idx, res_type) in global-row order."""
    rows_p = M
    rows_v = 2 * rows_p - 1
    result = []
    for r in range(rows_v):
        for c in range(cols_v):
            k = c + r * cols_v
            result.append((k, 0))  # jx
            result.append((k, 1))  # jy
            if r % 2 == 0 and c % 2 == 0:
                rho_idx = (c // 2) + (r // 2) * M
                result.append((rho_idx, 2))
                if energy:
                    result.append((rho_idx, 3))
    return result


# ---------------------------------------------------------------------------
# Unit tests for _n_density_before
# ---------------------------------------------------------------------------

class TestNDensityBefore:
    """Test the count formula directly."""

    def test_k0_has_zero_density_before(self):
        # k=0: r=0 (even), c=0 → (c+1)//2 = 0 → n=0 density before
        M = 3
        cols_v = 2 * M - 1   # = 5
        assert _n_density_before(0, cols_v, M) == 0

    def test_k1_has_one_density(self):
        # k=1: r=0 (even), c=1 → (c+1)//2 = 1 → n=1 (density at col=0 emitted)
        M = 3
        cols_v = 5
        assert _n_density_before(1, cols_v, M) == 1

    def test_k2_has_one_density(self):
        # k=2: r=0 (even), c=2 → (c+1)//2 = 1 → n=1
        M = 3
        cols_v = 5
        assert _n_density_before(2, cols_v, M) == 1

    def test_k3_has_two_density(self):
        # k=3: r=0 (even), c=3 → (c+1)//2 = 2 → n=2
        M = 3
        cols_v = 5
        assert _n_density_before(3, cols_v, M) == 2

    def test_k4_has_two_density(self):
        # k=4: r=0 (even), c=4 → (c+1)//2 = 2 → n=2
        M = 3
        cols_v = 5
        assert _n_density_before(4, cols_v, M) == 2

    def test_first_node_of_odd_row(self):
        # k=5: r=1 (odd), c=0 → ((1+1)//2)*M = 1*3 = 3 (row 0 was even → M density nodes)
        M = 3
        cols_v = 5
        assert _n_density_before(5, cols_v, M) == 3

    def test_first_node_of_second_even_row(self):
        # k=10: r=2, c=0 → ((2+1)//2)*3 + (1//2) = 1*3+0 = 3
        M = 3
        cols_v = 5
        assert _n_density_before(10, cols_v, M) == 3


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """field_to_global(global_to_field(g)) == g for all g."""

    @pytest.mark.parametrize("M,energy", [
        (2, False),
        (3, False),
        (4, False),
        (2, True),
        (3, True),
        (5, False),
    ])
    def test_roundtrip_global_to_field_to_global(self, M, energy):
        cols_v = 2 * M - 1
        N = _total_dofs(cols_v, M, energy)
        for g in range(N):
            fi, rt = global_to_field(g, cols_v, M, energy)
            g2 = field_to_global(fi, rt, cols_v, M, energy)
            assert g2 == g, (
                f"M={M}, energy={energy}: g={g} → (fi={fi}, rt={rt}) → g2={g2}"
            )

    @pytest.mark.parametrize("M,energy", [
        (2, False),
        (3, False),
        (4, False),
        (2, True),
        (3, True),
        (5, False),
    ])
    def test_roundtrip_field_to_global_to_field(self, M, energy):
        cols_v = 2 * M - 1
        pairs = _all_global_indices(cols_v, M, energy)
        seen = set()
        for fi, rt in pairs:
            g = field_to_global(fi, rt, cols_v, M, energy)
            assert g not in seen, (
                f"M={M}, energy={energy}: duplicate global row {g} for (fi={fi}, rt={rt})"
            )
            seen.add(g)
            fi2, rt2 = global_to_field(g, cols_v, M, energy)
            assert fi2 == fi and rt2 == rt, (
                f"M={M}, energy={energy}: (fi={fi}, rt={rt}) → g={g} → (fi2={fi2}, rt2={rt2})"
            )


# ---------------------------------------------------------------------------
# Spot-check: 4×4 velocity grid (M=3, cols_v=5)
# ---------------------------------------------------------------------------

class TestSpotCheck:
    """
    Fine grid: 5 columns × 5 rows mass-flux, 3 columns × 3 rows density.
    No energy.

    Row 0 (even), col order:
      k=0 (even-even): jx(k=0)=row0, jy=1, rho(0,0)=2  → global [0,1,2]
      k=1 (even-odd):  jx=3, jy=4
      k=2 (even-even): jx=5, jy=6, rho(0,1)=7          → global [5,6,7]
      k=3 (even-odd):  jx=8, jy=9
      k=4 (even-even): jx=10, jy=11, rho(0,2)=12        → global [10,11,12]
    Row 1 (odd), no density:
      k=5: jx=13, jy=14
      k=6: jx=15, jy=16
      k=7: jx=17, jy=18
      k=8: jx=19, jy=20
      k=9: jx=21, jy=22
    Row 2 (even):
      k=10 (even-even): jx=23, jy=24, rho(1,0)=25
      k=11:             jx=26, jy=27
      k=12 (even-even): jx=28, jy=29, rho(1,1)=30
      ...
    """

    M = 3
    cols_v = 5
    energy = False

    def test_k0_jx(self):
        assert field_to_global(0, 0, self.cols_v, self.M) == 0

    def test_k0_jy(self):
        assert field_to_global(0, 1, self.cols_v, self.M) == 1

    def test_k0_rho(self):
        # density index 0 → k=0 (even-even) → rho at offset 2
        assert field_to_global(0, 2, self.cols_v, self.M) == 2

    def test_k1_jx(self):
        assert field_to_global(1, 0, self.cols_v, self.M) == 3

    def test_k2_jx(self):
        assert field_to_global(2, 0, self.cols_v, self.M) == 5

    def test_k2_rho(self):
        # density index for (col=1, row=0): idx = 1 + 0*3 = 1
        assert field_to_global(1, 2, self.cols_v, self.M) == 7

    def test_k4_rho(self):
        # density index for (col=2, row=0): idx = 2
        assert field_to_global(2, 2, self.cols_v, self.M) == 12

    def test_k5_jx(self):
        # first node of odd row 1
        assert field_to_global(5, 0, self.cols_v, self.M) == 13

    def test_k10_jx(self):
        assert field_to_global(10, 0, self.cols_v, self.M) == 23

    def test_k10_rho(self):
        # density index for (col=0, row=1): idx = 0 + 1*3 = 3
        assert field_to_global(3, 2, self.cols_v, self.M) == 25


# ---------------------------------------------------------------------------
# Spot-check with energy
# ---------------------------------------------------------------------------

class TestSpotCheckEnergy:
    """
    M=2, cols_v=3, energy=True.

    Row 0:
      k=0 (even-even): jx=0, jy=1, rho=2, e=3
      k=1 (even-odd):  jx=4, jy=5
      k=2 (even-even): jx=6, jy=7, rho=8, e=9
    Row 1 (odd):
      k=3: jx=10, jy=11
      k=4: jx=12, jy=13
      k=5: jx=14, jy=15
    Row 2 (even):
      k=6 (even-even): jx=16, jy=17, rho=18, e=19
      ...
    """

    M = 2
    cols_v = 3
    energy = True

    def test_k0_jx(self):
        assert field_to_global(0, 0, self.cols_v, self.M, self.energy) == 0

    def test_k0_rho(self):
        assert field_to_global(0, 2, self.cols_v, self.M, self.energy) == 2

    def test_k0_e(self):
        assert field_to_global(0, 3, self.cols_v, self.M, self.energy) == 3

    def test_k1_jx(self):
        assert field_to_global(1, 0, self.cols_v, self.M, self.energy) == 4

    def test_k2_jx(self):
        assert field_to_global(2, 0, self.cols_v, self.M, self.energy) == 6

    def test_k2_rho(self):
        # density index for (col=1, row=0): 1
        assert field_to_global(1, 2, self.cols_v, self.M, self.energy) == 8

    def test_k3_jx(self):
        # first odd row node
        assert field_to_global(3, 0, self.cols_v, self.M, self.energy) == 10

    def test_k6_jx(self):
        assert field_to_global(6, 0, self.cols_v, self.M, self.energy) == 16

    def test_k6_rho(self):
        # density index for (col=0, row=1): 0 + 1*2 = 2
        assert field_to_global(2, 2, self.cols_v, self.M, self.energy) == 18


# ---------------------------------------------------------------------------
# Uniqueness and contiguity
# ---------------------------------------------------------------------------

class TestUniquenessAndContiguity:
    """field_to_global must produce a bijection onto {0, 1, ..., N-1}.

    This is the ground-truth correctness test: it does not rely on
    global_to_field at all.  For every valid (field_idx, res_type) pair we
    collect the global index, then assert:
      - no duplicates  (injective)
      - the full range {0, …, N-1} is covered  (surjective / contiguous)
    """

    @pytest.mark.parametrize("M,energy", [
        (2, False),
        (3, False),
        (4, False),
        (2, True),
        (3, True),
        (5, False),
    ])
    def test_unique_and_contiguous(self, M, energy):
        cols_v = 2 * M - 1
        rows_p = M
        rows_v = 2 * rows_p - 1
        N = _total_dofs(cols_v, M, energy)

        seen = {}  # global_idx → (field_idx, res_type) for duplicate diagnosis

        # mass-flux nodes: field_idx runs over the full fine grid
        nb_v = cols_v * rows_v
        for k in range(nb_v):
            for rt in (0, 1):
                g = field_to_global(k, rt, cols_v, M, energy)
                assert g not in seen, (
                    f"M={M}, energy={energy}: duplicate g={g} "
                    f"from (k={k}, rt={rt}) and {seen[g]}"
                )
                seen[g] = (k, rt)

        # density nodes: field_idx runs over the coarse grid
        nb_p = M * rows_p
        for di in range(nb_p):
            for rt in ([2, 3] if energy else [2]):
                g = field_to_global(di, rt, cols_v, M, energy)
                assert g not in seen, (
                    f"M={M}, energy={energy}: duplicate g={g} "
                    f"from (di={di}, rt={rt}) and {seen[g]}"
                )
                seen[g] = (di, rt)

        assert len(seen) == N, (
            f"M={M}, energy={energy}: got {len(seen)} distinct indices, expected {N}"
        )
        assert set(seen.keys()) == set(range(N)), (
            f"M={M}, energy={energy}: gaps or out-of-range indices in "
            f"{sorted(set(range(N)) - set(seen.keys()))}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_res_type(self):
        with pytest.raises(ValueError):
            field_to_global(0, 4, 3, 2)

    def test_energy_type_without_energy(self):
        with pytest.raises(ValueError):
            field_to_global(0, 3, 3, 2, energy=False)
