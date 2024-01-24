#
# Copyright 2024 Hannes Holey
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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


def TVD_MC_correction(q, cfl):
    """
    Compute the correction for the TVD-MacCormack scheme (Davis, 1984).

    Parameters
    ----------
    q : np.array
        Copy of the solution field from the previous step
    cfl : float
        Current CFL number

    Returns
    ----------
    np.array
        TVD correction term
    """

    q_diffx_pos = np.roll(q, -1, axis=1) - q
    denom_x_pos = np.sum(q_diffx_pos**2, axis=0)
    q_diffx_neg = q - np.roll(q, 1, axis=1)
    denom_x_neg = np.sum(q_diffx_neg**2, axis=0)

    nonzero_x_pos = denom_x_pos != 0
    nonzero_x_neg = denom_x_neg != 0

    rx_pos = np.ones_like(denom_x_pos)
    rx_pos[nonzero_x_pos] = np.sum(q_diffx_neg * q_diffx_pos, axis=0)[nonzero_x_pos] / denom_x_pos[nonzero_x_pos]

    rx_neg = np.ones_like(denom_x_neg)
    rx_neg[nonzero_x_neg] = np.sum(q_diffx_neg * q_diffx_pos, axis=0)[nonzero_x_neg] / denom_x_neg[nonzero_x_neg]

    q_diffy_pos = np.roll(q, -1, axis=2) - q
    denom_y_pos = np.sum(q_diffx_pos**2, axis=0)
    q_diffy_neg = q - np.roll(q, 1, axis=2)
    denom_y_neg = np.sum(q_diffx_pos**2, axis=0)

    nonzero_y_pos = denom_y_pos != 0
    nonzero_y_neg = denom_y_neg != 0

    ry_pos = np.ones_like(denom_y_pos)
    ry_neg = np.ones_like(denom_y_pos)

    ry_pos[nonzero_y_pos] = np.sum(q_diffy_neg * q_diffy_pos, axis=0)[nonzero_y_pos] / denom_y_pos[nonzero_y_pos]
    ry_neg[nonzero_y_neg] = np.sum(q_diffy_neg * q_diffy_pos, axis=0)[nonzero_y_neg] / denom_y_neg[nonzero_y_neg]

    G_rx_pos = flux_limiter_function(rx_pos, cfl)
    G_rx_pos_W = flux_limiter_function(np.roll(rx_pos, 1, axis=0), cfl)

    G_rx_neg = flux_limiter_function(rx_neg, cfl)
    G_rx_neg_E = flux_limiter_function(np.roll(rx_neg, -1, axis=0), cfl)

    G_ry_pos = flux_limiter_function(ry_pos, cfl)
    G_ry_pos_S = flux_limiter_function(np.roll(ry_pos, 1, axis=1), cfl)

    G_ry_neg = flux_limiter_function(rx_neg, cfl)
    G_ry_neg_N = flux_limiter_function(np.roll(ry_neg, -1, axis=1), cfl)

    return (G_rx_pos + G_rx_neg_E) * q_diffx_pos \
        - (G_rx_pos_W + G_rx_neg) * q_diffx_neg \
        + (G_ry_pos + G_ry_neg_N) * q_diffy_pos \
        - (G_ry_pos_S + G_ry_neg) * q_diffy_neg


def flux_limiter_function(r, cfl):
    """Flux limiter function of the TCD-MacCormack scheme (Davis, 1984)

    Parameters
    ----------
    r : np.array
        Scalar field, quantifying the relative cell difference of neighboring cells
    cfl : float
        Current CFL number

    Returns
    ----------
    np.array
        Values of the flux limiter function
    """

    if cfl <= 0.5:
        C = cfl * (1 - cfl)
    else:
        C = 0.25

    phi = np.maximum(np.zeros_like(r), np.minimum(2*r, 1))

    return 0.5 * C * (1 - phi)
