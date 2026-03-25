"""Tests for shape functions in fem_2d_new/elements.py."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from GaPFlow.fem_2d_new.elements import TaylorHoodP2P1


# Reference triangle nodes (xi, eta) and their node indices
P1_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
P2_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                     [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])

# Extra test points inside the reference triangle
TEST_POINTS = np.array([[1/6, 1/6],
                        [2/3, 1/6],
                        [1/6, 2/3],
                        [1/3, 1/3],
                        [0.5, 0.1],
                        [0.1, 0.5]])


def eval_p1(x, y):
    """Evaluate all 3 P1 shape functions at (x, y)."""
    N = [None] * 3
    N[0] = 1 - x - y
    N[1] = x
    N[2] = y
    return np.array(N)


def eval_p2(x, y):
    """Evaluate all 6 P2 shape functions at (x, y)."""
    N = [None] * 6
    N[0] = (1 - x - y) * (1 - 2*x - 2*y)
    N[1] = x * (2*x - 1)
    N[2] = y * (2*y - 1)
    N[3] = 4 * y * (1 - x - y)
    N[4] = 4 * x * (1 - x - y)
    N[5] = 4 * x * y
    return np.array(N)


# ─────────────────────────────────────────────
# P1 shape function tests
# ─────────────────────────────────────────────

class TestP1ShapeFunctions:

    @pytest.mark.parametrize("pt", TEST_POINTS)
    def test_partition_of_unity(self, pt):
        """Sum of all P1 shape functions must equal 1."""
        x, y = pt
        assert np.sum(eval_p1(x, y)) == pytest.approx(1.0)

    @pytest.mark.parametrize("i, pt", enumerate(P1_NODES))
    def test_kronecker_delta(self, i, pt):
        """N_i evaluated at node j must equal delta_ij."""
        x, y = pt
        N = eval_p1(x, y)
        expected = np.eye(3)[i]
        np.testing.assert_allclose(N, expected, atol=1e-14)

    @pytest.mark.parametrize("pt", TEST_POINTS)
    def test_non_negative_inside(self, pt):
        """All P1 shape functions are >= 0 inside the reference triangle."""
        x, y = pt
        assert np.all(eval_p1(x, y) >= -1e-14)


# ─────────────────────────────────────────────
# P2 shape function tests
# ─────────────────────────────────────────────

class TestP2ShapeFunctions:

    @pytest.mark.parametrize("pt", TEST_POINTS)
    def test_partition_of_unity(self, pt):
        """Sum of all P2 shape functions must equal 1."""
        x, y = pt
        assert np.sum(eval_p2(x, y)) == pytest.approx(1.0)

    @pytest.mark.parametrize("i, pt", enumerate(P2_NODES))
    def test_kronecker_delta(self, i, pt):
        """N_i evaluated at node j must equal delta_ij."""
        x, y = pt
        N = eval_p2(x, y)
        expected = np.eye(6)[i]
        np.testing.assert_allclose(N, expected, atol=1e-14)


# ─────────────────────────────────────────────
# TaylorHoodP2P1 class tests (quadrature values)
# ─────────────────────────────────────────────

class TestTaylorHoodElement:

    @pytest.fixture
    def element(self):
        return TaylorHoodP2P1(dx=1.0, dy=1.0)

    def test_quadrature_points_inside_triangle(self, element):
        """All quadrature points must lie inside the reference triangle."""
        coords = element.Quadrature.coordinates
        assert np.all(coords >= 0)
        assert np.all(coords[:, 0] + coords[:, 1] <= 1.0 + 1e-14)

    def test_quadrature_weights_sum(self, element):
        """Quadrature weights must sum to area of reference triangle (0.5)."""
        assert np.sum(element.Quadrature.weights) == pytest.approx(0.5)

    def test_p1_quadrature_shape(self, element):
        """P1.N at quadrature points has shape (nb_quad, 3)."""
        N = element.P1.N
        assert N.shape == (element.Quadrature.nb_points, 3)

    def test_p2_quadrature_shape(self, element):
        """P2.N at quadrature points has shape (nb_quad, 6)."""
        N = element.P2.N
        assert N.shape == (element.Quadrature.nb_points, 6)

    def test_p1_partition_of_unity_at_quadrature(self, element):
        """P1 shape functions sum to 1 at each quadrature point."""
        N = element.P1.N
        np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-14)

    def test_p2_partition_of_unity_at_quadrature(self, element):
        """P2 shape functions sum to 1 at each quadrature point."""
        N = element.P2.N
        np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-14)
