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

import numpy as np

from .grid_index import GridIndexManager


def field_to_global(field_idx,
                    spec,
                    grid_index: GridIndexManager,
                    p_factor: int = 1,
                    ):
    """Computes the unique global node index."""

    decomp = grid_index._decomp
    Nx_global_p = decomp.nb_domain_grid_pts[0]
    Nx_global_v = decomp.nb_domain_grid_pts_v[0]

    if spec.grid == 'p':
        y_p, x_p = np.divmod(field_idx, Nx_global_p)
        y_v, x_v = 2*y_p, 2*x_p
    else:
        y_v, x_v = np.divmod(field_idx, Nx_global_v)

    nb_nodes_cur_row = _get_nb_nodes_cur_row(x_v, y_v, p_factor)
    nb_nodes_block_below = _get_nb_nodes_block_below(y_v, Nx_global_v, Nx_global_p, p_factor)

    return nb_nodes_block_below + nb_nodes_cur_row + spec.idx

def _get_nb_nodes_cur_row(x_v, y_v, p_factor):
    """Number of nodes in the current row BEFORE x_v.
    If y_v is even: x_v//2 additional nodes in the current row
    """
    nb_p = np.where(y_v % 2 == 0, (x_v + 1) // 2, 0)
    nb_v = x_v
    return 2*nb_v + p_factor*nb_p

def _get_nb_nodes_block_below(y_v, Nx_global_v, Nx_global_p, p_factor):
    """Number of nodes in all rows BELOW y_v.
    Each row has cols_v nodes, and each even row has cols_p//2 additional nodes.
     - if r is odd: no additional nodes in the current row
     - if r is even: c//2 additional nodes in the current row
    """
    nb_v = y_v * Nx_global_v
    rows_p_below = (y_v + 1) // 2
    nb_p = rows_p_below * Nx_global_p
    return 2*nb_v + p_factor*nb_p