#
# Copyright 2025-2026 Hannes Holey
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

# flake8: noqa: W503


"""Viscous stress tensor components.

This module contains functions that calculate the (generalized) Newtonian stress tensor
components at the walls of the bottom and top surface, and averaged across the gap.
"""

import numpy as np


def stress_bottom(q, h, U, V, eta, zeta, Lsb=0.0, Lst=0.0):
    """Viscous stress tensor at the bottom wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Lsb : float, optional
        Slip length bottom (default is 0. --> no slip)
    Lst : float, optional
        Slip length top (default is 0. --> no slip)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    # Voigt ordering: xx, yy, zz, yz, xz, xy
    tau = np.zeros((6, *q.shape[1:]))

    if np.all(np.isclose(Lsb, 0.0)):
        if np.all(np.isclose(Lst, 0.0)):
            denom = q[0] * h[0]
            tau[3] = 2 * eta * (-2 * V * q[0] + 3 * q[2]) / denom
            tau[4] = 2 * eta * (-2 * U * q[0] + 3 * q[1]) / denom
            return tau
        denom = q[0] * (4 * Lst * h[0] + h[0] ** 2)
        tau[3] = (
            2 * eta * (-6 * Lst * V * q[0] + 6 * Lst * q[2] - 2 * V * h[0] * q[0] + 3 * h[0] * q[2]) / denom
        )
        tau[4] = (
            2 * eta * (-6 * Lst * U * q[0] + 6 * Lst * q[1] - 2 * U * h[0] * q[0] + 3 * h[0] * q[1]) / denom
        )
        return tau

    if np.all(np.isclose(Lst, 0.0)):
        denom = q[0] ** 2 * (16 * Lsb**2 + 8 * Lsb * h[0] + h[0] ** 2)
        denom34 = q[0] * (4 * Lsb + h[0])
        tau[0] = (
            2
            * Lsb
            * (
                + 2 * U * h[1] * q[0] ** 2 * v1
                + 2 * V * h[2] * q[0] ** 2 * v2
                - 3 * h[1] * q[0] * q[1] * v1
                - 3 * h[2] * q[0] * q[2] * v2
            )
            / denom
        )
        tau[1] = (
            2
            * Lsb
            * (
                + 2 * U * h[1] * q[0] ** 2 * v2
                + 2 * V * h[2] * q[0] ** 2 * v1
                - 3 * h[1] * q[0] * q[1] * v2
                - 3 * h[2] * q[0] * q[2] * v1
            )
            / denom
        )
        tau[2] = (
            2
            * Lsb
            * v2
            * (
                + 2 * U * h[1] * q[0] ** 2
                + 2 * V * h[2] * q[0] ** 2
                - 3 * h[1] * q[0] * q[1]
                - 3 * h[2] * q[0] * q[2]
            )
            / denom
        )
        tau[3] = 2 * eta * (-2 * V * q[0] + 3 * q[2]) / denom34
        tau[4] = 2 * eta * (-2 * U * q[0] + 3 * q[1]) / denom34
        tau[5] = (
            2
            * Lsb
            * eta
            * (
                + 2 * U * h[2] * q[0] ** 2
                + 2 * V * h[1] * q[0] ** 2
                - 3 * h[1] * q[0] * q[2]
                - 3 * h[2] * q[0] * q[1]
            )
            / denom
        )
        return tau

    denom = q[0] ** 2 * (
        144 * Lsb**2 * Lst**2
        + 96 * Lsb**2 * Lst * h[0]
        + 16 * Lsb**2 * h[0] ** 2
        + 96 * Lsb * Lst**2 * h[0]
        + 56 * Lsb * Lst * h[0] ** 2
        + 8 * Lsb * h[0] ** 3
        + 16 * Lst**2 * h[0] ** 2
        + 8 * Lst * h[0] ** 3
        + h[0] ** 4
    )

    tau[0] = (
        2
        * Lsb
        * (
            + 12 * Lsb * Lst * h[1] * q[0] * q[1] * v1
            + 12 * Lsb * Lst * h[2] * q[0] * q[2] * v2
            + 24 * Lst**2 * U * h[1] * q[0] ** 2 * v1
            + 24 * Lst**2 * V * h[2] * q[0] ** 2 * v2
            - 24 * Lst**2 * h[1] * q[0] * q[1] * v1
            - 24 * Lst**2 * h[2] * q[0] * q[2] * v2
            + 12 * Lst * U * h[0] * h[1] * q[0] ** 2 * v1
            + 12 * Lst * V * h[0] * h[2] * q[0] ** 2 * v2
            - 12 * Lst * h[0] * h[1] * q[0] * q[1] * v1
            - 12 * Lst * h[0] * h[2] * q[0] * q[2] * v2
            + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
            + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
            - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
            - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
        )
        / denom
    )

    tau[1] = (
        2
        * Lsb
        * (
            + 12 * Lsb * Lst * h[1] * q[0] * q[1] * v2
            + 12 * Lsb * Lst * h[2] * q[0] * q[2] * v1
            + 24 * Lst**2 * U * h[1] * q[0] ** 2 * v2
            + 24 * Lst**2 * V * h[2] * q[0] ** 2 * v1
            - 24 * Lst**2 * h[1] * q[0] * q[1] * v2
            - 24 * Lst**2 * h[2] * q[0] * q[2] * v1
            + 12 * Lst * U * h[0] * h[1] * q[0] ** 2 * v2
            + 12 * Lst * V * h[0] * h[2] * q[0] ** 2 * v1
            - 12 * Lst * h[0] * h[1] * q[0] * q[1] * v2
            - 12 * Lst * h[0] * h[2] * q[0] * q[2] * v1
            + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
            + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
            - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
            - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
        )
        / denom
    )

    tau[2] = (
        2
        * Lsb
        * v2
        * (
            + 12 * Lsb * Lst * h[1] * q[0] * q[1]
            + 12 * Lsb * Lst * h[2] * q[0] * q[2]
            + 24 * Lst**2 * U * h[1] * q[0] ** 2
            + 24 * Lst**2 * V * h[2] * q[0] ** 2
            - 24 * Lst**2 * h[1] * q[0] * q[1]
            - 24 * Lst**2 * h[2] * q[0] * q[2]
            + 12 * Lst * U * h[0] * h[1] * q[0] ** 2
            + 12 * Lst * V * h[0] * h[2] * q[0] ** 2
            - 12 * Lst * h[0] * h[1] * q[0] * q[1]
            - 12 * Lst * h[0] * h[2] * q[0] * q[2]
            + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2
            + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2
            - 3 * h[0] ** 2 * h[1] * q[0] * q[1]
            - 3 * h[0] ** 2 * h[2] * q[0] * q[2]
        )
        / denom
    )

    denom34 = q[0] * (12 * Lsb * Lst + 4 * Lsb * h[0] + 4 * Lst * h[0] + h[0] ** 2)

    tau[3] = (
        2 * eta * (-6 * Lst * V * q[0] + 6 * Lst * q[2] - 2 * V * h[0] * q[0] + 3 * h[0] * q[2]) / denom34
    )

    tau[4] = (
        2 * eta * (-6 * Lst * U * q[0] + 6 * Lst * q[1] - 2 * U * h[0] * q[0] + 3 * h[0] * q[1]) / denom34
    )

    tau[5] = (
        2
        * Lsb
        * eta
        * (
            + 12 * Lsb * Lst * h[1] * q[0] * q[2]
            + 12 * Lsb * Lst * h[2] * q[0] * q[1]
            + 24 * Lst**2 * U * h[2] * q[0] ** 2
            + 24 * Lst**2 * V * h[1] * q[0] ** 2
            - 24 * Lst**2 * h[1] * q[0] * q[2]
            - 24 * Lst**2 * h[2] * q[0] * q[1]
            + 12 * Lst * U * h[0] * h[2] * q[0] ** 2
            + 12 * Lst * V * h[0] * h[1] * q[0] ** 2
            - 12 * Lst * h[0] * h[1] * q[0] * q[2]
            - 12 * Lst * h[0] * h[2] * q[0] * q[1]
            + 2 * U * h[0] ** 2 * h[2] * q[0] ** 2
            + 2 * V * h[0] ** 2 * h[1] * q[0] ** 2
            - 3 * h[0] ** 2 * h[1] * q[0] * q[2]
            - 3 * h[0] ** 2 * h[2] * q[0] * q[1]
        )
        / denom
    )

    return tau


def stress_top(q, h, U, V, eta, zeta, Lsb=0.0, Lst=0.0):
    """Viscous stress tensor at the top wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Lsb : float, optional
        Slip length bottom (default is 0. --> no slip)
    Lst : float, optional
        Slip length top (default is 0. --> no slip)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    # Voigt ordering: xx, yy, zz, yz, xz, xy
    tau = np.zeros((6, *q.shape[1:]))

    if np.all(np.isclose(Lsb, 0.0)):
        if np.all(np.isclose(Lst, 0.0)):
            denom = q[0] ** 2 * h[0]
            tau[0] = (
                2
                * (
                    -U * h[1] * q[0] ** 2 * v1
                    - V * h[2] * q[0] ** 2 * v2
                    + 3 * h[1] * q[0] * q[1] * v1
                    + 3 * h[2] * q[0] * q[2] * v2
                )
                / denom
            )
            tau[1] = (
                2
                * (
                    -U * h[1] * q[0] ** 2 * v2
                    - V * h[2] * q[0] ** 2 * v1
                    + 3 * h[1] * q[0] * q[1] * v2
                    + 3 * h[2] * q[0] * q[2] * v1
                )
                / denom
            )
            tau[2] = (
                2
                * v2
                * (
                    -U * h[1] * q[0] ** 2
                    - V * h[2] * q[0] ** 2
                    + 3 * h[1] * q[0] * q[1]
                    + 3 * h[2] * q[0] * q[2]
                )
                / denom
            )
            denom34 = q[0] * h[0]
            tau[3] = 2 * eta * (V * q[0] - 3 * q[2]) / denom34
            tau[4] = 2 * eta * (U * q[0] - 3 * q[1]) / denom34
            tau[5] = (
                2
                * eta
                * (
                    -U * h[2] * q[0] ** 2
                    - V * h[1] * q[0] ** 2
                    + 3 * h[1] * q[0] * q[2]
                    + 3 * h[2] * q[0] * q[1]
                )
                / denom
            )
            return tau
        denom = q[0] ** 2 * (16 * Lst ** 2 + 8 * Lst * h[0] + h[0] ** 2)
        tau[0] = (
            2
            * (
                - 3 * Lst * U * h[1] * q[0] ** 2 * v1
                - 3 * Lst * V * h[2] * q[0] ** 2 * v2
                + 9 * Lst * h[1] * q[0] * q[1] * v1
                + 9 * Lst * h[2] * q[0] * q[2] * v2
                - U * h[0] * h[1] * q[0] ** 2 * v1
                - V * h[0] * h[2] * q[0] ** 2 * v2
                + 3 * h[0] * h[1] * q[0] * q[1] * v1
                + 3 * h[0] * h[2] * q[0] * q[2] * v2
            )
            / denom
        )
        tau[1] = (
            2
            * (
                - 3 * Lst * U * h[1] * q[0] ** 2 * v2
                - 3 * Lst * V * h[2] * q[0] ** 2 * v1
                + 9 * Lst * h[1] * q[0] * q[1] * v2
                + 9 * Lst * h[2] * q[0] * q[2] * v1
                - U * h[0] * h[1] * q[0] ** 2 * v2
                - V * h[0] * h[2] * q[0] ** 2 * v1
                + 3 * h[0] * h[1] * q[0] * q[1] * v2
                + 3 * h[0] * h[2] * q[0] * q[2] * v1
            )
            / denom
        )
        tau[2] = (
            2
            * v2
            * (
                - 3 * Lst * U * h[1] * q[0] ** 2
                - 3 * Lst * V * h[2] * q[0] ** 2
                + 9 * Lst * h[1] * q[0] * q[1]
                + 9 * Lst * h[2] * q[0] * q[2]
                - U * h[0] * h[1] * q[0] ** 2
                - V * h[0] * h[2] * q[0] ** 2
                + 3 * h[0] * h[1] * q[0] * q[1]
                + 3 * h[0] * h[2] * q[0] * q[2]
            )
            / denom
        )
        denom34 = q[0] * (4 * Lst + h[0])
        tau[3] = 2 * eta * (V * q[0] - 3 * q[2]) / denom34
        tau[4] = 2 * eta * (U * q[0] - 3 * q[1]) / denom34
        tau[5] = (
            2
            * eta
            * (
                - 3 * Lst * U * h[2] * q[0] ** 2
                - 3 * Lst * V * h[1] * q[0] ** 2
                + 9 * Lst * h[1] * q[0] * q[2]
                + 9 * Lst * h[2] * q[0] * q[1]
                - U * h[0] * h[2] * q[0] ** 2
                - V * h[0] * h[1] * q[0] ** 2
                + 3 * h[0] * h[1] * q[0] * q[2]
                + 3 * h[0] * h[2] * q[0] * q[1]
            )
            / denom
        )
        return tau

    if np.all(np.isclose(Lst, 0.0)):
        denom = q[0] ** 2 * h[0] * (16 * Lsb**2 + 8 * Lsb * h[0] + h[0] ** 2)
        denom34 = q[0] * h[0] * (4 * Lsb + h[0])
        tau[0] = (
            2
            * (
                + 24 * Lsb**2 * h[1] * q[0] * q[1] * v1
                + 24 * Lsb**2 * h[2] * q[0] * q[2] * v2
                - 4 * Lsb * U * h[0] * h[1] * q[0] ** 2 * v1
                - 4 * Lsb * V * h[0] * h[2] * q[0] ** 2 * v2
                + 18 * Lsb * h[0] * h[1] * q[0] * q[1] * v1
                + 18 * Lsb * h[0] * h[2] * q[0] * q[2] * v2
                - U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                - V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                + 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                + 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            )
            / denom
        )
        tau[1] = (
            2
            * (
                + 24 * Lsb**2 * h[1] * q[0] * q[1] * v2
                + 24 * Lsb**2 * h[2] * q[0] * q[2] * v1
                - 4 * Lsb * U * h[0] * h[1] * q[0] ** 2 * v2
                - 4 * Lsb * V * h[0] * h[2] * q[0] ** 2 * v1
                + 18 * Lsb * h[0] * h[1] * q[0] * q[1] * v2
                + 18 * Lsb * h[0] * h[2] * q[0] * q[2] * v1
                - U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                - V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                + 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                + 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            )
            / denom
        )
        tau[2] = (
            2
            * v2
            * (
                + 24 * Lsb**2 * h[1] * q[0] * q[1]
                + 24 * Lsb**2 * h[2] * q[0] * q[2]
                - 4 * Lsb * U * h[0] * h[1] * q[0] ** 2
                - 4 * Lsb * V * h[0] * h[2] * q[0] ** 2
                + 18 * Lsb * h[0] * h[1] * q[0] * q[1]
                + 18 * Lsb * h[0] * h[2] * q[0] * q[2]
                - U * h[0] ** 2 * h[1] * q[0] ** 2
                - V * h[0] ** 2 * h[2] * q[0] ** 2
                + 3 * h[0] ** 2 * h[1] * q[0] * q[1]
                + 3 * h[0] ** 2 * h[2] * q[0] * q[2]
            )
            / denom
        )
        tau[3] = 2 * eta * (-6 * Lsb * q[2] + V * h[0] * q[0] - 3 * h[0] * q[2]) / denom34
        tau[4] = 2 * eta * (-6 * Lsb * q[1] + U * h[0] * q[0] - 3 * h[0] * q[1]) / denom34
        tau[5] = (
            2
            * eta
            * (
                + 24 * Lsb**2 * h[1] * q[0] * q[2]
                + 24 * Lsb**2 * h[2] * q[0] * q[1]
                - 4 * Lsb * U * h[0] * h[2] * q[0] ** 2
                - 4 * Lsb * V * h[0] * h[1] * q[0] ** 2
                + 18 * Lsb * h[0] * h[1] * q[0] * q[2]
                + 18 * Lsb * h[0] * h[2] * q[0] * q[1]
                - U * h[0] ** 2 * h[2] * q[0] ** 2
                - V * h[0] ** 2 * h[1] * q[0] ** 2
                + 3 * h[0] ** 2 * h[1] * q[0] * q[2]
                + 3 * h[0] ** 2 * h[2] * q[0] * q[1]
            )
            / denom
        )
        return tau

    denom = q[0] ** 2 * (
        144 * Lsb**2 * Lst**2
        + 96 * Lsb**2 * Lst * h[0]
        + 16 * Lsb**2 * h[0] ** 2
        + 96 * Lsb * Lst**2 * h[0]
        + 56 * Lsb * Lst * h[0] ** 2
        + 8 * Lsb * h[0] ** 3
        + 16 * Lst**2 * h[0] ** 2
        + 8 * Lst * h[0] ** 3
        + h[0] ** 4
    )

    tau[0] = (
        2
        * (
            + 48 * Lsb**2 * Lst * h[1] * q[0] * q[1] * v1
            + 48 * Lsb**2 * Lst * h[2] * q[0] * q[2] * v2
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1] * v1
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2] * v2
            - 12 * Lsb * Lst**2 * U * h[1] * q[0] ** 2 * v1
            - 12 * Lsb * Lst**2 * V * h[2] * q[0] ** 2 * v2
            + 12 * Lsb * Lst**2 * h[1] * q[0] * q[1] * v1
            + 12 * Lsb * Lst**2 * h[2] * q[0] * q[2] * v2
            - 12 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2 * v1
            - 12 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2 * v2
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1] * v1
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2] * v2
            - 4 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
            - 4 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
            + 18 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1] * v1
            + 18 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            - 3 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
            - 3 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
            + 9 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v1
            + 9 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            - U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
            - V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
            + 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v1
            + 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v2
        )
        / denom
    )

    tau[1] = (
        2
        * (
            + 48 * Lsb**2 * Lst * h[1] * q[0] * q[1] * v2
            + 48 * Lsb**2 * Lst * h[2] * q[0] * q[2] * v1
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1] * v2
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2] * v1
            - 12 * Lsb * Lst**2 * U * h[1] * q[0] ** 2 * v2
            - 12 * Lsb * Lst**2 * V * h[2] * q[0] ** 2 * v1
            + 12 * Lsb * Lst**2 * h[1] * q[0] * q[1] * v2
            + 12 * Lsb * Lst**2 * h[2] * q[0] * q[2] * v1
            - 12 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2 * v2
            - 12 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2 * v1
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1] * v2
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2] * v1
            - 4 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
            - 4 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
            + 18 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1] * v2
            + 18 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            - 3 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
            - 3 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
            + 9 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v2
            + 9 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            - U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
            - V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
            + 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v2
            + 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v1
        )
        / denom
    )

    tau[2] = (
        2
        * v2
        * (
            + 48 * Lsb**2 * Lst * h[1] * q[0] * q[1]
            + 48 * Lsb**2 * Lst * h[2] * q[0] * q[2]
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1]
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2]
            - 12 * Lsb * Lst**2 * U * h[1] * q[0] ** 2
            - 12 * Lsb * Lst**2 * V * h[2] * q[0] ** 2
            + 12 * Lsb * Lst**2 * h[1] * q[0] * q[1]
            + 12 * Lsb * Lst**2 * h[2] * q[0] * q[2]
            - 12 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2
            - 12 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1]
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2]
            - 4 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2
            - 4 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2
            + 18 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1]
            + 18 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2]
            - 3 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2
            - 3 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2
            + 9 * Lst * h[0] ** 2 * h[1] * q[0] * q[1]
            + 9 * Lst * h[0] ** 2 * h[2] * q[0] * q[2]
            - U * h[0] ** 3 * h[1] * q[0] ** 2
            - V * h[0] ** 3 * h[2] * q[0] ** 2
            + 3 * h[0] ** 3 * h[1] * q[0] * q[1]
            + 3 * h[0] ** 3 * h[2] * q[0] * q[2]
        )
        / denom
    )

    denom34 = q[0] * (12 * Lsb * Lst + 4 * Lsb * h[0] + 4 * Lst * h[0] + h[0] ** 2)

    tau[3] = (
        2 * eta * (-6 * Lsb * q[2] + V * h[0] * q[0] - 3 * h[0] * q[2]) / denom34
    )

    tau[4] = (
        2 * eta * (-6 * Lsb * q[1] + U * h[0] * q[0] - 3 * h[0] * q[1]) / denom34
    )

    tau[5] = (
        2
        * eta
        * (
            + 48 * Lsb**2 * Lst * h[1] * q[0] * q[2]
            + 48 * Lsb**2 * Lst * h[2] * q[0] * q[1]
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[2]
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[1]
            - 12 * Lsb * Lst**2 * U * h[2] * q[0] ** 2
            - 12 * Lsb * Lst**2 * V * h[1] * q[0] ** 2
            + 12 * Lsb * Lst**2 * h[1] * q[0] * q[2]
            + 12 * Lsb * Lst**2 * h[2] * q[0] * q[1]
            - 12 * Lsb * Lst * U * h[0] * h[2] * q[0] ** 2
            - 12 * Lsb * Lst * V * h[0] * h[1] * q[0] ** 2
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[2]
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[1]
            - 4 * Lsb * U * h[0] ** 2 * h[2] * q[0] ** 2
            - 4 * Lsb * V * h[0] ** 2 * h[1] * q[0] ** 2
            + 18 * Lsb * h[0] ** 2 * h[1] * q[0] * q[2]
            + 18 * Lsb * h[0] ** 2 * h[2] * q[0] * q[1]
            - 3 * Lst * U * h[0] ** 2 * h[2] * q[0] ** 2
            - 3 * Lst * V * h[0] ** 2 * h[1] * q[0] ** 2
            + 9 * Lst * h[0] ** 2 * h[1] * q[0] * q[2]
            + 9 * Lst * h[0] ** 2 * h[2] * q[0] * q[1]
            - U * h[0] ** 3 * h[2] * q[0] ** 2
            - V * h[0] ** 3 * h[1] * q[0] ** 2
            + 3 * h[0] ** 3 * h[1] * q[0] * q[2]
            + 3 * h[0] ** 3 * h[2] * q[0] * q[1]
        )
        / denom
    )

    return tau


def stress_avg(q, h, U, V, eta, zeta, Lsb=0.0, Lst=0.0):
    """Gap-averaged viscous stress tensor (normal and in-plane shear components).

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Lsb : float, optional
        Slip length bottom (default is 0. --> no slip)
    Lst : float, optional
        Slip length top (default is 0. --> no slip)

    Returns
    -------
    numpy.ndarray
        Gap-averaged viscous stress tensor components
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    # Ordering: xx, yy, xy
    tau = np.zeros((3, *q.shape[1:]))

    if np.all(np.isclose(Lsb, 0.0)):
        if np.all(np.isclose(Lst, 0.0)):
            denom = q[0] ** 2 * h[0] ** 2
            tau[0] = (
                + h[0] * h[1] * q[0] * q[1] * v1
                + h[0] * h[2] * q[0] * q[2] * v2
            ) / denom
            tau[1] = (
                + h[0] * h[1] * q[0] * q[1] * v2
                + h[0] * h[2] * q[0] * q[2] * v1
            ) / denom
            tau[2] = (
                eta
                * (
                    + h[0] * h[1] * q[0] * q[2]
                    + h[0] * h[2] * q[0] * q[1]
                )
                / denom
            )
            return tau
        denom = q[0] ** 2 * (4 * Lst * h[0] + h[0] ** 2)
        tau[0] = (
            + 2 * Lst * U * h[1] * q[0] ** 2 * v1
            + 2 * Lst * V * h[2] * q[0] ** 2 * v2
            - 2 * Lst * h[1] * q[0] * q[1] * v1
            - 2 * Lst * h[2] * q[0] * q[2] * v2
            + h[0] * h[1] * q[0] * q[1] * v1
            + h[0] * h[2] * q[0] * q[2] * v2
        ) / denom
        tau[1] = (
            + 2 * Lst * U * h[1] * q[0] ** 2 * v2
            + 2 * Lst * V * h[2] * q[0] ** 2 * v1
            - 2 * Lst * h[1] * q[0] * q[1] * v2
            - 2 * Lst * h[2] * q[0] * q[2] * v1
            + h[0] * h[1] * q[0] * q[1] * v2
            + h[0] * h[2] * q[0] * q[2] * v1
        ) / denom
        tau[2] = (
            eta
            * (
                + 2 * Lst * U * h[2] * q[0] ** 2
                + 2 * Lst * V * h[1] * q[0] ** 2
                - 2 * Lst * h[1] * q[0] * q[2]
                - 2 * Lst * h[2] * q[0] * q[1]
                + h[0] * h[1] * q[0] * q[2]
                + h[0] * h[2] * q[0] * q[1]
            )
            / denom
        )
        return tau

    if np.all(np.isclose(Lst, 0.0)):
        denom = q[0] ** 2 * (4 * Lsb * h[0] + h[0] ** 2)
        tau[0] = (
            + 4 * Lsb * h[1] * q[0] * q[1] * v1
            + 4 * Lsb * h[2] * q[0] * q[2] * v2
            + h[0] * h[1] * q[0] * q[1] * v1
            + h[0] * h[2] * q[0] * q[2] * v2
        ) / denom
        tau[1] = (
            + 4 * Lsb * h[1] * q[0] * q[1] * v2
            + 4 * Lsb * h[2] * q[0] * q[2] * v1
            + h[0] * h[1] * q[0] * q[1] * v2
            + h[0] * h[2] * q[0] * q[2] * v1
        ) / denom
        tau[2] = (
            eta
            * (
                + 4 * Lsb * h[1] * q[0] * q[2]
                + 4 * Lsb * h[2] * q[0] * q[1]
                + h[0] * h[1] * q[0] * q[2]
                + h[0] * h[2] * q[0] * q[1]
            )
            / denom
        )
        return tau

    denom = q[0] ** 2 * (12 * Lsb * Lst + 4 * Lsb * h[0] + 4 * Lst * h[0] + h[0] ** 2)

    tau[0] = (
        + 4 * Lsb * h[1] * q[0] * q[1] * v1
        + 4 * Lsb * h[2] * q[0] * q[2] * v2
        + 2 * Lst * U * h[1] * q[0] ** 2 * v1
        + 2 * Lst * V * h[2] * q[0] ** 2 * v2
        - 2 * Lst * h[1] * q[0] * q[1] * v1
        - 2 * Lst * h[2] * q[0] * q[2] * v2
        + h[0] * h[1] * q[0] * q[1] * v1
        + h[0] * h[2] * q[0] * q[2] * v2
    ) / denom

    tau[1] = (
        + 4 * Lsb * h[1] * q[0] * q[1] * v2
        + 4 * Lsb * h[2] * q[0] * q[2] * v1
        + 2 * Lst * U * h[1] * q[0] ** 2 * v2
        + 2 * Lst * V * h[2] * q[0] ** 2 * v1
        - 2 * Lst * h[1] * q[0] * q[1] * v2
        - 2 * Lst * h[2] * q[0] * q[2] * v1
        + h[0] * h[1] * q[0] * q[1] * v2
        + h[0] * h[2] * q[0] * q[2] * v1
    ) / denom

    tau[2] = (
        eta
        * (
            + 4 * Lsb * h[1] * q[0] * q[2]
            + 4 * Lsb * h[2] * q[0] * q[1]
            + 2 * Lst * U * h[2] * q[0] ** 2
            + 2 * Lst * V * h[1] * q[0] ** 2
            - 2 * Lst * h[1] * q[0] * q[2]
            - 2 * Lst * h[2] * q[0] * q[1]
            + h[0] * h[1] * q[0] * q[2]
            + h[0] * h[2] * q[0] * q[1]
        )
        / denom
    )

    return tau
