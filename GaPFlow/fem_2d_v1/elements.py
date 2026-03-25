from functools import cached_property
import numpy as np
from muGrid import GenericLinearOperator
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class TaylorHoodP2P1:

    """
    =================================
    P1
    =================================

    square to node index mapping:
    2 ----- 3
    |       |
    |       |
    0 ----- 1

    internal triangleindexing order:
    2
    |
    |
    0 ----- 1

    =================================
    P2
    =================================

    square to node index mapping:
    2 --- 7 --- 3
    |           |
    4     6     8
    |           |
    0 --- 5 --- 1

    internal triangle indexing order:
    2
    |
    3     5
    |
    0 --- 4 --- 1

    """

    left_idx = 0
    right_idx = 1
    n_tri = 2

    # ===============================================================
    # Stencils
    # ===============================================================

    # All u-u combinations. p-* combinations found by checking even points.
    stencil_even_even = [
        (0, 0),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (0, -2), (0, -1), (0, 1), (0, 2),
        (-1, -1), (1, 1),
        (-2, 2), (-1, 1), (1, -1), (2, -2),
        (-2, 1), (-1, 2), (1, -2), (2, -1)
    ]
    stencil_odd_odd = [
        (0, 0),
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (1, 1),
        (-1, 1), (1, -1)
    ]
    stencil_even_odd = [
        (0, 0),
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, 1), (1, -1),
        (-2, 1), (2, -1)
    ]
    stencil_odd_even = [
        (0, 0),
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, 1), (1, -1),
        (-1, 2), (1, -2)
    ]

    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

        self.Quadrature = Quadrature3Points()
        self.P1 = self.P1(self.Quadrature)
        self.P2 = self.P2(self.Quadrature)


    class P1:
        
        # ================================================================
        # Sources of truth
        # ================================================================

        nb_nodes = 3

        idx_to_std = np.array([[0, 1, 2],
                               [3, 2, 1]])
        
        square_node_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        der_factor = [1, -1]

        _N_funcs = [None] * 3
        _N_funcs[0] = lambda x, y: 1 - x - y
        _N_funcs[1] = lambda x, y: x
        _N_funcs[2] = lambda x, y: y

        _N_x_funcs = [None] * 3
        _N_x_funcs[0] = lambda x, y: -1
        _N_x_funcs[1] = lambda x, y: 1
        _N_x_funcs[2] = lambda x, y: 0

        _N_y_funcs = [None] * 3
        _N_y_funcs[0] = lambda x, y: -1
        _N_y_funcs[1] = lambda x, y: 0
        _N_y_funcs[2] = lambda x, y: 1

        # ================================================================
        # Quadrature
        # ================================================================

        def __init__(self, quadrature):
            self.quadrature = quadrature

        @cached_property
        def N(self):
            """Returns shape (nb_quad, nb_nodes).
            """
            N = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                N[i] = [N_func(x, y) for N_func in self._N_funcs]

            return N
        
        @cached_property
        def dN_dx(self) -> NDArray:
            """Returns shape (nb_quad, nb_nodes)."""
            dN_dx = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                dN_dx[i] = [N_x_func(x, y) for N_x_func in self._N_x_funcs]

            return dN_dx

        @cached_property
        def dN_dy(self) -> NDArray:
            """Returns shape (nb_quad, nb_nodes)."""
            dN_dy = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                dN_dy[i] = [N_y_func(x, y) for N_y_func in self._N_y_funcs]

            return dN_dy

        def _make_operator(self, dN: NDArray, apply_der_factor: bool = False) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN (nb_q, nb_nodes).
            Input shape:  (ny+1, nx+1)
            Output shape: (nb_tri * nb_q, ny, nx)

            apply_der_factor: if True, multiply each triangle's contribution by
            der_factor[t] to correct for the orientation flip in tri1.
            """
            offsets    = self.square_node_offsets
            nb_nodes   = self.nb_nodes
            nb_tri     = 2
            der_factor = np.array(self.der_factor, dtype=float)  # shape (nb_tri,)

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg   # (Ny_padded, Nx_padded)
                out_pg = output_field.pg                  # (nb_sub, Ny_padded, Nx_padded)
                _, ny_pad, nx_pad = out_pg.shape
                ny = ny_pad - 1
                nx = nx_pad - 1

                node_vals = np.empty((nb_tri, nb_nodes, ny, nx))
                for t in range(nb_tri):
                    for k in range(nb_nodes):
                        ox, oy = offsets[self.idx_to_std[t, k]]
                        node_vals[t, k] = nodal[oy:ny+oy, ox:nx+ox]

                result = np.einsum('qk, tkrc -> tqrc', dN, node_vals)
                if apply_der_factor:
                    result *= der_factor[:, np.newaxis, np.newaxis, np.newaxis]
                out_pg[:, :ny, :nx] = result.reshape(len(dN) * nb_tri, ny, nx)

            return QuadOperator(None, numpy_fn=numpy_fn, backend='numpy')

        @property
        def weights(self) -> NDArray:
            return self.quadrature.weights

        @cached_property
        def interpolation_operator(self) -> "QuadOperator":
            return self._make_operator(self.N)

        @cached_property
        def dx_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dx, apply_der_factor=True)

        @cached_property
        def dy_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dy, apply_der_factor=True)


    class P2:

        # ================================================================
        # Sources of truth
        # ================================================================

        nb_nodes = 6

        idx_to_std = np.array([[0, 1, 2, 4, 5, 6],
                               [3, 2, 1, 8, 7, 6]])
        
        square_node_offsets = np.array([[0, 0], [2, 0], [0, 2], [2, 2],
                                        [0, 1], [1, 0], [1, 1],
                                        [1, 2], [2, 1]])

        der_factor = [1, -1]

        _N_funcs = [None] * 6
        _N_funcs[0] = lambda x, y: (1 - x - y) * (1 - 2*x - 2*y)
        _N_funcs[1] = lambda x, y: x * (2*x - 1)
        _N_funcs[2] = lambda x, y: y * (2*y - 1)
        _N_funcs[3] = lambda x, y: 4 * y * (1 - x - y)
        _N_funcs[4] = lambda x, y: 4 * x * (1 - x - y)
        _N_funcs[5] = lambda x, y: 4 * x * y

        _N_x_funcs = [None] * 6
        _N_x_funcs[0] = lambda x, y: -3 + 4*x + 4*y
        _N_x_funcs[1] = lambda x, y: 4*x - 1
        _N_x_funcs[2] = lambda x, y: 0
        _N_x_funcs[3] = lambda x, y: -4 * y
        _N_x_funcs[4] = lambda x, y: 4 - 8*x - 4*y
        _N_x_funcs[5] = lambda x, y: 4 * y

        _N_y_funcs = [None] * 6
        _N_y_funcs[0] = lambda x, y: -3 + 4*x + 4*y
        _N_y_funcs[1] = lambda x, y: 0
        _N_y_funcs[2] = lambda x, y: 4*y - 1
        _N_y_funcs[3] = lambda x, y: 4 - 4*x - 8*y
        _N_y_funcs[4] = lambda x, y: -4 * x
        _N_y_funcs[5] = lambda x, y: 4 * x

        # ================================================================
        # Quadrature
        # ================================================================

        def __init__(self, quadrature):
            self.quadrature = quadrature

        @cached_property
        def N(self) -> NDArray:
            """Returns shape (nb_quad, nb_nodes)."""
            N = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                N[i] = [N_func(x, y) for N_func in self._N_funcs]

            return N

        @cached_property
        def dN_dx(self) -> NDArray:
            """Returns shape (nb_quad, nb_nodes)."""
            dN_dx = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                dN_dx[i] = [N_x_func(x, y) for N_x_func in self._N_x_funcs]

            return dN_dx

        @cached_property
        def dN_dy(self) -> NDArray:
            """Returns shape (nb_quad, nb_nodes)."""
            dN_dy = np.empty((self.quadrature.nb_points, self.nb_nodes))
            for i in range(self.quadrature.nb_points):
                x, y = self.quadrature.coordinates[i]
                dN_dy[i] = [N_y_func(x, y) for N_y_func in self._N_y_funcs]

            return dN_dy

        def _make_operator(self, dN: NDArray, apply_der_factor: bool = False) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN (nb_q, nb_nodes).
            Input shape:  (ny_fine, nx_fine)  where ny_fine=2*ny+1, nx_fine=2*nx+1
            Output shape: (nb_tri * nb_q, ny, nx)

            Square mapping input -> output:
            3x3 -> 1x1; 5x5 -> 2x2; 7x7 -> 3x3, etc.

            apply_der_factor: if True, multiply each triangle's contribution by
            der_factor[t] to correct for the orientation flip in tri1.
            """
            offsets    = self.square_node_offsets
            nb_nodes   = self.nb_nodes
            nb_tri     = 2
            der_factor = np.array(self.der_factor, dtype=float)  # shape (nb_tri,)

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg   # (Ny_v_padded, Nx_v_padded)
                out_pg = output_field.pg                  # (nb_sub, Ny_p_padded, Nx_p_padded)
                _, ny_pad, nx_pad = out_pg.shape
                ny = ny_pad - 1
                nx = nx_pad - 1

                node_vals = np.empty((nb_tri, nb_nodes, ny, nx))
                for t in range(nb_tri):
                    for k in range(nb_nodes):
                        ox, oy = offsets[self.idx_to_std[t, k]]
                        node_vals[t, k] = nodal[oy:2*ny+oy:2, ox:2*nx+ox:2]

                result = np.einsum('qk, tkrc -> tqrc', dN, node_vals)
                if apply_der_factor:
                    result *= der_factor[:, np.newaxis, np.newaxis, np.newaxis]
                out_pg[:, :ny, :nx] = result.reshape(len(dN) * nb_tri, ny, nx)

            return QuadOperator(None, numpy_fn=numpy_fn, backend='numpy')

        @property
        def weights(self) -> NDArray:
            return self.quadrature.weights

        @cached_property
        def interpolation_operator(self) -> "QuadOperator":
            return self._make_operator(self.N)

        @cached_property
        def dx_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dx, apply_der_factor=True)

        @cached_property
        def dy_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dy, apply_der_factor=True)


class Quadrature3Points:

    nb_points = 3
    coordinates = np.array([[1/6, 1/6],
                            [2/3, 1/6],
                            [1/6, 2/3]])
    weights = np.array([1/6, 1/6, 1/6])


class QuadOperator:
    """Wraps GenericLinearOperator with an optional numpy backend.

    Exposes the same .apply(input_field, output_field) interface so it is a
    drop-in replacement.  Set backend='numpy' to use the numpy function instead.
    """

    def __init__(self, mugrid_op: GenericLinearOperator, numpy_fn=None,
                 backend: str = 'mugrid'):
        self._mugrid_op = mugrid_op
        self._numpy_fn  = numpy_fn
        self.backend    = backend

    def apply(self, input_field, output_field) -> None:
        if self.backend == 'numpy' and self._numpy_fn is not None:
            self._numpy_fn(input_field, output_field)
        else:
            self._mugrid_op.apply(input_field, output_field)
