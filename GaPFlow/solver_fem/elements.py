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

    internal triangle indexing order:
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

    # All u-u combinations. p-* combinations are derived by checking even points.
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
        self.sq_area = dx * dy
        self.Quadrature = Quadrature7Points()
        self.P1 = self.P1(self.Quadrature)
        self.P2 = self.P2(self.Quadrature)


    class P1:
        
        # ================================================================
        # Sources of truth
        # ================================================================

        nodes_per_tri = 3

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

        def _eval_funcs(self, funcs) -> NDArray:
            return np.array([[f(x, y) for f in funcs]
                             for x, y in self.quadrature.coordinates])

        @cached_property
        def N(self) -> NDArray:
            """Returns shape (nb_quad_tri, nodes_per_tri)."""
            return self._eval_funcs(self._N_funcs)

        @cached_property
        def dN_dx(self) -> NDArray:
            """Returns shape (nb_quad_tri, nodes_per_tri)."""
            return self._eval_funcs(self._N_x_funcs)

        @cached_property
        def dN_dy(self) -> NDArray:
            """Returns shape (nb_quad_tri, nodes_per_tri)."""
            return self._eval_funcs(self._N_y_funcs)

        def _make_operator(self, dN: NDArray, apply_der_factor: bool = False) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN (nb_q, nodes_per_tri).
            Input shape:  (ny+1, nx+1)
            Output shape: (nb_tri * nb_q, ny, nx)

            apply_der_factor: if True, multiply each triangle's contribution by
            der_factor[t] to correct for the orientation flip in tri1.
            """
            offsets = self.square_node_offsets
            nodes_per_tri = self.nodes_per_tri
            nb_tri = 2
            der_factor = np.array(self.der_factor, dtype=float)  # shape (nb_tri,)

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg
                out_pg = output_field.pg
                _, nx_pad, ny_pad = out_pg.shape
                nx = nx_pad - 1
                ny = ny_pad - 1

                node_vals = np.empty((nb_tri, nodes_per_tri, nx, ny))
                for t in range(nb_tri):
                    for k in range(nodes_per_tri):
                        ox, oy = offsets[self.idx_to_std[t, k]]
                        node_vals[t, k] = nodal[ox:nx+ox, oy:ny+oy]

                result = np.einsum('qk, tkrc -> tqrc', dN, node_vals)
                if apply_der_factor:
                    result *= der_factor[:, np.newaxis, np.newaxis, np.newaxis]
                out_pg[:, :nx, :ny] = result.reshape(len(dN) * nb_tri, nx, ny)

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

        nodes_per_tri = 6

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

        def _eval_funcs(self, funcs) -> NDArray:
            return np.array([[f(x, y) for f in funcs]
                             for x, y in self.quadrature.coordinates])

        @cached_property
        def N(self) -> NDArray:
            """Returns shape (nb_quad, nodes_per_tri)."""
            return self._eval_funcs(self._N_funcs)

        @cached_property
        def dN_dx(self) -> NDArray:
            """Returns shape (nb_quad, nodes_per_tri)."""
            return self._eval_funcs(self._N_x_funcs)

        @cached_property
        def dN_dy(self) -> NDArray:
            """Returns shape (nb_quad, nodes_per_tri)."""
            return self._eval_funcs(self._N_y_funcs)

        def _make_operator(self, dN: NDArray, apply_der_factor: bool = False) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN (nb_q, nodes_per_tri).
            Input shape:  (ny_fine, nx_fine)  where ny_fine=2*ny+1, nx_fine=2*nx+1
            Output shape: (nb_tri * nb_q, ny, nx)

            Square mapping input -> output:
            3x3 -> 1x1; 5x5 -> 2x2; 7x7 -> 3x3, etc.

            apply_der_factor: if True, multiply each triangle's contribution by
            der_factor[t] to correct for the orientation flip in tri1.
            """
            offsets = self.square_node_offsets
            nodes_per_tri = self.nodes_per_tri
            nb_tri = 2
            der_factor = np.array(self.der_factor, dtype=float)  # shape (nb_tri,)

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg
                out_pg = output_field.pg
                _, nx_pad, ny_pad = out_pg.shape
                nx = nx_pad - 1
                ny = ny_pad - 1

                node_vals = np.empty((nb_tri, nodes_per_tri, nx, ny))
                for t in range(nb_tri):
                    for k in range(nodes_per_tri):
                        ox, oy = offsets[self.idx_to_std[t, k]]
                        node_vals[t, k] = nodal[ox:2*nx+ox:2, oy:2*ny+oy:2]

                result = np.einsum('qk, tkrc -> tqrc', dN, node_vals)
                if apply_der_factor:
                    result *= der_factor[:, np.newaxis, np.newaxis, np.newaxis]
                out_pg[:, :nx, :ny] = result.reshape(len(dN) * nb_tri, nx, ny)

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


class Quadrature4Points:

    nb_points = 4
    coordinates = np.array([[1/3, 1/3],
                            [1/5, 1/5],
                            [3/5, 1/5],
                            [1/5, 3/5]])
    weights = np.array([-27/96, 25/96, 25/96, 25/96])


class Quadrature6Points:

    nb_points = 6
    _a1 = 0.091576213509771
    _b1 = 0.816847572980459
    _a2 = 0.445948490915965
    _b2 = 0.108103018168070
    _w1 = 0.054975871827661
    _w2 = 0.111690794839006
    coordinates = np.array([[_a1, _a1],
                            [_b1, _a1],
                            [_a1, _b1],
                            [_a2, _a2],
                            [_b2, _a2],
                            [_a2, _b2]])
    weights = np.array([_w1, _w1, _w1,
                        _w2, _w2, _w2])


class Quadrature7Points:

    nb_points = 7
    _a1 = 0.059715871789770
    _b1 = 0.470142064105115
    _a2 = 0.797426985353087
    _b2 = 0.101286507323456
    _w0 = 9.0/80.0
    _w1 = 0.066197076394253
    _w2 = 0.062969590272414
    coordinates = np.array([[1/3, 1/3],
                            [_a1, _b1],
                            [_b1, _a1],
                            [_b1, _b1],
                            [_a2, _b2],
                            [_b2, _a2],
                            [_b2, _b2]])
    weights = np.array([_w0,
                        _w1, _w1, _w1,
                        _w2, _w2, _w2])


class QuadOperator:
    """Wraps GenericLinearOperator with an optional numpy backend.
    Right now, numpy is faster than muGrid.
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
