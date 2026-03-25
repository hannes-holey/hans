"""Phase 4 tests: nodal→quadrature interpolation via P1 and P2 operators.

Tests the operators in TaylorHoodP2P1 directly, without needing Problem or
muGrid FieldCollections.  The operators expose a numpy_fn that works on plain
arrays wrapped in a minimal stub.

Axis convention (matches the operator implementation):
  - Nodal arrays: shape (Ny, Nx) — first axis is row (y), second is col (x).
  - Output arrays: shape (n_tri*n_quad, Ny_sq, Nx_sq).
  - meshgrid with indexing='xy' produces (Ny, Nx) shaped X, Y.

Exactness expectations:
  - P2 is exact for polynomials up to degree 2 (linear and quadratic fields)
  - P1 is exact for linear fields
  - Derivative operators: exact for the corresponding polynomial degree
"""
import pytest
import numpy as np

from GaPFlow.fem_2d.elements import TaylorHoodP2P1


# ---------------------------------------------------------------------------
# Minimal field stub so QuadOperator.apply() works without muGrid
# ---------------------------------------------------------------------------

class FieldStub:
    """Minimal stub exposing .pg attribute (matching the muGrid field interface)."""
    def __init__(self, arr: np.ndarray):
        self.pg = arr


# ---------------------------------------------------------------------------
# Grid helpers — indexing='xy' gives (Ny, Nx) shaped arrays
# ---------------------------------------------------------------------------

def make_coarse_grid(Nx: int, Ny: int, dx: float = 1.0, dy: float = 1.0):
    """Return (X, Y) for a coarse P1 grid, both shape (Ny, Nx)."""
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    return np.meshgrid(x, y, indexing='xy')   # both (Ny, Nx)


def make_fine_grid(Nx: int, Ny: int, dx: float = 1.0, dy: float = 1.0):
    """Return (X, Y) for a fine P2 grid, both shape (2*Ny-1, 2*Nx-1)."""
    Nx_v = 2 * Nx - 1
    Ny_v = 2 * Ny - 1
    x = np.arange(Nx_v) * (dx / 2)
    y = np.arange(Ny_v) * (dy / 2)
    return np.meshgrid(x, y, indexing='xy')   # both (Ny_v, Nx_v)


def apply_operator(op, nodal_arr: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Apply a QuadOperator to a nodal array, return output array.

    nodal_arr : shape (Ny_nodal, Nx_nodal)
    out_shape : shape (n_tri*n_quad, Ny_sq, Nx_sq)

    The operator writes into out_pg[:, :Ny_sq, :Nx_sq], so the padded buffer
    needs one extra row and column.
    """
    nb_sub, ny_sq, nx_sq = out_shape
    out_pg = np.zeros((nb_sub, ny_sq + 1, nx_sq + 1))
    inp = FieldStub(nodal_arr)        # 2-D .pg — operator reads it directly
    oup = FieldStub(out_pg)
    op.apply(inp, oup)
    return out_pg[:, :ny_sq, :nx_sq]


# ---------------------------------------------------------------------------
# Quadrature point physical coordinates
# ---------------------------------------------------------------------------

def quad_coords_p1(Nx: int, Ny: int, dx: float, dy: float):
    """Physical (x, y) at each quad point for P1 elements.

    P1 squares: (Nx-1) × (Ny-1).  Each square has 2 triangles × 3 quad pts.
    Quad points in reference coords: (1/6,1/6), (2/3,1/6), (1/6,2/3).
    Triangle 0 (lower-left):  v0=SW=(0,0), v1=SE=(dx,0), v2=NW=(0,dy)
    Triangle 1 (upper-right): v0=NE=(dx,dy), v1=NW=(0,dy), v2=SE=(dx,0)
      x = (1-xi-eta)*dx + xi*0 + eta*dx = dx*(1-xi)
      y = (1-xi-eta)*dy + xi*dy + eta*0 = dy*(1-eta)
    Returns xq, yq each shape (6, Ny-1, Nx-1).
    """
    from GaPFlow.fem_2d.elements import Quadrature3Points
    qpts = Quadrature3Points.coordinates   # (3, 2)

    Nx_sq = Nx - 1
    Ny_sq = Ny - 1
    # SW corner coords — meshgrid with xy gives (Ny_sq, Nx_sq)
    sw_x = np.arange(Nx_sq) * dx
    sw_y = np.arange(Ny_sq) * dy
    SW_X, SW_Y = np.meshgrid(sw_x, sw_y, indexing='xy')  # (Ny_sq, Nx_sq)

    xq = np.empty((6, Ny_sq, Nx_sq))
    yq = np.empty((6, Ny_sq, Nx_sq))

    for k, (xi, eta) in enumerate(qpts):
        xq[k] = SW_X + xi * dx
        yq[k] = SW_Y + eta * dy

    for k, (xi, eta) in enumerate(qpts):
        xq[3 + k] = SW_X + dx * (1 - xi)
        yq[3 + k] = SW_Y + dy * (1 - eta)

    return xq, yq


def quad_coords_p2(Nx: int, Ny: int, dx: float, dy: float):
    """Physical (x,y) at each quad point for P2 elements.

    Same physical triangle geometry as P1 — only the basis functions differ.
    Returns xq, yq each shape (6, Ny-1, Nx-1).
    """
    return quad_coords_p1(Nx, Ny, dx, dy)


# ---------------------------------------------------------------------------
# P1 exactness tests
# ---------------------------------------------------------------------------

class TestP1Interpolation:
    """P1 interpolation is exact for linear fields."""

    Nx, Ny = 5, 6
    dx, dy = 0.4, 0.3

    @pytest.fixture(autouse=True)
    def setup(self):
        self.elem = TaylorHoodP2P1(self.dx, self.dy)
        self.X, self.Y = make_coarse_grid(self.Nx, self.Ny, self.dx, self.dy)
        # Output shape: (6, Ny-1, Nx-1)
        self.out_shape = (6, self.Ny - 1, self.Nx - 1)
        self.xq, self.yq = quad_coords_p1(self.Nx, self.Ny, self.dx, self.dy)

    def test_constant_field(self):
        nodal = np.ones((self.Ny, self.Nx)) * 3.14
        out = apply_operator(self.elem.P1.interpolation_operator,
                             nodal, self.out_shape)
        assert np.allclose(out, 3.14, atol=1e-12)

    def test_linear_x(self):
        a = 2.5
        nodal = a * self.X
        out = apply_operator(self.elem.P1.interpolation_operator,
                             nodal, self.out_shape)
        expected = a * self.xq
        assert np.allclose(out, expected, atol=1e-12)

    def test_linear_y(self):
        b = -1.7
        nodal = b * self.Y
        out = apply_operator(self.elem.P1.interpolation_operator,
                             nodal, self.out_shape)
        expected = b * self.yq
        assert np.allclose(out, expected, atol=1e-12)

    def test_linear_xy(self):
        a, b, c = 1.3, -0.7, 2.0
        nodal = a * self.X + b * self.Y + c
        out = apply_operator(self.elem.P1.interpolation_operator,
                             nodal, self.out_shape)
        expected = a * self.xq + b * self.yq + c
        assert np.allclose(out, expected, atol=1e-12)

    def test_dx_operator_linear(self):
        a = 1.5
        nodal = a * self.X + 0.3 * self.Y
        out_shape_d = (6, self.Ny - 1, self.Nx - 1)
        out = apply_operator(self.elem.P1.dx_operator, nodal, out_shape_d)
        # der_factor corrects tri1 orientation: all 6 entries equal a*dx.
        # Physical du/dx = out / dx = a everywhere.
        assert np.allclose(out, a * self.dx, atol=1e-12)

    def test_dy_operator_linear(self):
        b = 0.8
        nodal = 0.2 * self.X + b * self.Y
        out_shape_d = (6, self.Ny - 1, self.Nx - 1)
        out = apply_operator(self.elem.P1.dy_operator, nodal, out_shape_d)
        assert np.allclose(out, b * self.dy, atol=1e-12)


# ---------------------------------------------------------------------------
# P2 exactness tests
# ---------------------------------------------------------------------------

class TestP2Interpolation:
    """P2 interpolation is exact for polynomials up to degree 2."""

    Nx, Ny = 4, 5   # coarse grid; fine grid = (2*Nx-1) × (2*Ny-1)
    dx, dy = 0.5, 0.4

    @pytest.fixture(autouse=True)
    def setup(self):
        self.elem = TaylorHoodP2P1(self.dx, self.dy)
        self.X, self.Y = make_fine_grid(self.Nx, self.Ny, self.dx, self.dy)
        # Output shape: (6, Ny-1, Nx-1)
        self.out_shape = (6, self.Ny - 1, self.Nx - 1)
        self.xq, self.yq = quad_coords_p2(self.Nx, self.Ny, self.dx, self.dy)

    def test_constant_field(self):
        nodal = np.ones((2 * self.Ny - 1, 2 * self.Nx - 1)) * 2.71
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        assert np.allclose(out, 2.71, atol=1e-12)

    def test_linear_x(self):
        a = 1.8
        nodal = a * self.X
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = a * self.xq
        assert np.allclose(out, expected, atol=1e-11)

    def test_linear_y(self):
        b = -2.1
        nodal = b * self.Y
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = b * self.yq
        assert np.allclose(out, expected, atol=1e-11)

    def test_linear_xy(self):
        a, b, c = 0.9, -1.2, 3.0
        nodal = a * self.X + b * self.Y + c
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = a * self.xq + b * self.yq + c
        assert np.allclose(out, expected, atol=1e-11)

    def test_quadratic_x2(self):
        nodal = self.X ** 2
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = self.xq ** 2
        assert np.allclose(out, expected, atol=1e-11)

    def test_quadratic_y2(self):
        nodal = self.Y ** 2
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = self.yq ** 2
        assert np.allclose(out, expected, atol=1e-11)

    def test_quadratic_xy(self):
        nodal = self.X * self.Y
        out = apply_operator(self.elem.P2.interpolation_operator,
                             nodal, self.out_shape)
        expected = self.xq * self.yq
        assert np.allclose(out, expected, atol=1e-11)

    def test_dx_operator_linear(self):
        a = 1.5
        nodal = a * self.X + 0.3 * self.Y
        out_shape_d = (6, self.Ny - 1, self.Nx - 1)
        out = apply_operator(self.elem.P2.dx_operator, nodal, out_shape_d)
        # der_factor corrects tri1 orientation: all 6 entries equal a*dx.
        assert np.allclose(out, a * self.dx, atol=1e-11)

    def test_dy_operator_linear(self):
        b = 0.7
        nodal = 0.1 * self.X + b * self.Y
        out_shape_d = (6, self.Ny - 1, self.Nx - 1)
        out = apply_operator(self.elem.P2.dy_operator, nodal, out_shape_d)
        assert np.allclose(out, b * self.dy, atol=1e-11)

    def test_dx_operator_quadratic(self):
        # u = x^2, du/dx = 2*x.  P2 is exact for quadratic fields.
        # Both tris give the same physical gradient after der_factor correction.
        # Physical du/dx = out[k] / dx at quad pt k.
        from GaPFlow.fem_2d.elements import Quadrature3Points
        nodal = self.X ** 2
        out_shape_d = (6, self.Ny - 1, self.Nx - 1)
        out = apply_operator(self.elem.P2.dx_operator, nodal, out_shape_d)

        qpts = Quadrature3Points.coordinates
        sw_x = np.arange(self.Nx - 1) * self.dx
        sw_y = np.arange(self.Ny - 1) * self.dy
        SW_X, SW_Y = np.meshgrid(sw_x, sw_y, indexing='xy')

        for k, (xi, eta) in enumerate(qpts):
            xq0 = SW_X + xi * self.dx          # tri0 physical x
            xq1 = SW_X + self.dx * (1 - xi)   # tri1 physical x
            assert np.allclose(out[k] / self.dx, 2 * xq0, atol=1e-10)
            assert np.allclose(out[3 + k] / self.dx, 2 * xq1, atol=1e-10)
