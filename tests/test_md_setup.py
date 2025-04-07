import pytest

from hans.multiscale.md_setup import _get_MPI_grid


@pytest.mark.parametrize("Na,size,max_cpu", [(1000, 2, 4),
                                             (30_000, 3, 30),
                                             (30_000, 3, 3),
                                             (100_000, 4, 12),
                                             (500_000, 4, 24),
                                             (1_000_000, 5, 30)
                                             ])
def test_processors_grid(Na, size, max_cpu):

    nx, ny, nz = _get_MPI_grid(Na, size, max_cpu)

    assert nx * ny * nz <= max_cpu
