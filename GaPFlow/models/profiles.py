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

"""Gap profiles.

Analytical expressions for velocity and stress profiles as a function of the gap coordinate.
"""

# flake8: noqa: W503


def get_velocity_profiles(z, q, Lsb=0.0, Lst=0.0, U=1.0, V=0.0):
    """Velocity profiles for a given flow rate and wall velocity

    Parameters
    ----------
    z : array-like
        Gap coordinate (z)
    q : array-like
        Height-averaged solution, (rho, jx, jy) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    Lsb : float, optional
        Slip length bottom (the default is 0.0, which means no-slip)
    Lst : float, optional
        Slip length top (the default is 0.0, which means no-slip)
    U : float, optional
        Lower wall velocity in x direction (the default is 1.0)
    V : float, optional
        Lower wall velocity in y direction (the default is 0.0)

    Returns
    -------
    array-like, array-like
        Discretized profiles u(z) and v(z)
    """

    h = z[-1]

    u = (
        12 * Lsb * Lst * h * q[1]
        + 6 * Lsb * h ** 2 * q[1]
        - 6 * Lsb * q[1] * z**2
        + 4 * Lst * U * h ** 2 * q[0]
        - 12 * Lst * U * h * q[0] * z
        + 6 * Lst * U * q[0] * z**2
        + 12 * Lst * h * q[1] * z
        - 6 * Lst * q[1] * z**2
        + U * h ** 3 * q[0]
        - 4 * U * h ** 2 * q[0] * z
        + 3 * U * h * q[0] * z**2
        + 6 * h ** 2 * q[1] * z
        - 6 * h * q[1] * z**2
    ) / (h * q[0] * (12 * Lsb * Lst + 4 * Lsb * h + 4 * Lst * h + h ** 2))

    v = (
        12 * Lsb * Lst * h * q[2]
        + 6 * Lsb * h ** 2 * q[2]
        - 6 * Lsb * q[2] * z**2
        + 4 * Lst * V * h ** 2 * q[0]
        - 12 * Lst * V * h * q[0] * z
        + 6 * Lst * V * q[0] * z**2
        + 12 * Lst * h * q[2] * z
        - 6 * Lst * q[2] * z**2
        + V * h ** 3 * q[0]
        - 4 * V * h ** 2 * q[0] * z
        + 3 * V * h * q[0] * z**2
        + 6 * h ** 2 * q[2] * z
        - 6 * h * q[2] * z**2
    ) / (h * q[0] * (12 * Lsb * Lst + 4 * Lsb * h + 4 * Lst * h + h ** 2))

    return u, v


def get_stress_profiles(z, h, q, dqx, dqy, U=1.0, V=0.0, eta=1.0, zeta=1.0, Lsb=0., Lst=0.):
    """Viscous shear stress profiles for a given flow rate and wall velcoty

    Parameters
    ----------
    z : array-like
        Gap coordinate (z)
    h : array-like
        Gap height and gradients, (h, ∂h/∂x, ∂h/∂y) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    q : array-like
        Gap-averaged solution, (ρ, jx, jy) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    dqx : array-like
        Gap-averaged solution gradients, (∂ρ/∂x, ∂jx/∂x, ∂jy/∂x) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    dqy : array-like
        Gap-averaged solution gradients, (∂ρ/∂y, ∂jx/∂y, ∂jy/∂y) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    U : float, optional
        Lower wall velocity in x direction (the default is 1.0)
    V : float, optional
        Lower wall velocity in y direction (the default is 1.0)
    Lsb : float, optional
        Slip length bottom (the default is 0.0, which means no-slip)
    Lst : float, optional
        Slip length top (the default is 0.0, which means no-slip)

    Returns
    -------
    Tuple[array-like, ...]
        Discretized shear stress profiles  τ_xx(z), τ_yy, τ_zz(z), τ_yz(z), τ_xz, τ_xy(z)
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    tau_xx = (
        2
        * (
            -72 * Lsb**2 * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v1
            + 72 * Lsb**2 * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v1
            - 72 * Lsb**2 * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v2
            + 72 * Lsb**2 * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v2
            - 60 * Lsb**2 * Lst * dqx[0] * h[0] ** 3 * q[1] * v1
            + 36 * Lsb**2 * Lst * dqx[0] * h[0] * q[1] * v1 * z**2
            + 60 * Lsb**2 * Lst * dqx[1] * h[0] ** 3 * q[0] * v1
            - 36 * Lsb**2 * Lst * dqx[1] * h[0] * q[0] * v1 * z**2
            - 60 * Lsb**2 * Lst * dqy[0] * h[0] ** 3 * q[2] * v2
            + 36 * Lsb**2 * Lst * dqy[0] * h[0] * q[2] * v2 * z**2
            + 60 * Lsb**2 * Lst * dqy[2] * h[0] ** 3 * q[0] * v2
            - 36 * Lsb**2 * Lst * dqy[2] * h[0] * q[0] * v2 * z**2
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v1
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            + 36 * Lsb**2 * Lst * h[1] * q[0] * q[1] * v1 * z**2
            + 36 * Lsb**2 * Lst * h[2] * q[0] * q[2] * v2 * z**2
            - 12 * Lsb**2 * dqx[0] * h[0] ** 4 * q[1] * v1
            + 12 * Lsb**2 * dqx[0] * h[0] ** 2 * q[1] * v1 * z**2
            + 12 * Lsb**2 * dqx[1] * h[0] ** 4 * q[0] * v1
            - 12 * Lsb**2 * dqx[1] * h[0] ** 2 * q[0] * v1 * z**2
            - 12 * Lsb**2 * dqy[0] * h[0] ** 4 * q[2] * v2
            + 12 * Lsb**2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z**2
            + 12 * Lsb**2 * dqy[2] * h[0] ** 4 * q[0] * v2
            - 12 * Lsb**2 * dqy[2] * h[0] ** 2 * q[0] * v2 * z**2
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1] * v1 * z**2
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2] * v2 * z**2
            + 24 * Lsb * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
            - 36 * Lsb * Lst**2 * U * h[1] * q[0] ** 2 * v1 * z**2
            + 24 * Lsb * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
            - 36 * Lsb * Lst**2 * V * h[2] * q[0] ** 2 * v2 * z**2
            - 24 * Lsb * Lst**2 * dqx[0] * h[0] ** 3 * q[1] * v1
            - 72 * Lsb * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v1 * z
            + 36 * Lsb * Lst**2 * dqx[0] * h[0] * q[1] * v1 * z**2
            + 24 * Lsb * Lst**2 * dqx[1] * h[0] ** 3 * q[0] * v1
            + 72 * Lsb * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v1 * z
            - 36 * Lsb * Lst**2 * dqx[1] * h[0] * q[0] * v1 * z**2
            - 24 * Lsb * Lst**2 * dqy[0] * h[0] ** 3 * q[2] * v2
            - 72 * Lsb * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z
            + 36 * Lsb * Lst**2 * dqy[0] * h[0] * q[2] * v2 * z**2
            + 24 * Lsb * Lst**2 * dqy[2] * h[0] ** 3 * q[0] * v2
            + 72 * Lsb * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v2 * z
            - 36 * Lsb * Lst**2 * dqy[2] * h[0] * q[0] * v2 * z**2
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            + 36 * Lsb * Lst**2 * h[1] * q[0] * q[1] * v1 * z**2
            + 36 * Lsb * Lst**2 * h[2] * q[0] * q[2] * v2 * z**2
            + 12 * Lsb * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
            - 24 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2 * v1 * z**2
            + 12 * Lsb * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
            - 24 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2 * v2 * z**2
            - 18 * Lsb * Lst * dqx[0] * h[0] ** 4 * q[1] * v1
            - 60 * Lsb * Lst * dqx[0] * h[0] ** 3 * q[1] * v1 * z
            + 60 * Lsb * Lst * dqx[0] * h[0] ** 2 * q[1] * v1 * z**2
            + 18 * Lsb * Lst * dqx[1] * h[0] ** 4 * q[0] * v1
            + 60 * Lsb * Lst * dqx[1] * h[0] ** 3 * q[0] * v1 * z
            - 60 * Lsb * Lst * dqx[1] * h[0] ** 2 * q[0] * v1 * z**2
            - 18 * Lsb * Lst * dqy[0] * h[0] ** 4 * q[2] * v2
            - 60 * Lsb * Lst * dqy[0] * h[0] ** 3 * q[2] * v2 * z
            + 60 * Lsb * Lst * dqy[0] * h[0] ** 2 * q[2] * v2 * z**2
            + 18 * Lsb * Lst * dqy[2] * h[0] ** 4 * q[0] * v2
            + 60 * Lsb * Lst * dqy[2] * h[0] ** 3 * q[0] * v2 * z
            - 60 * Lsb * Lst * dqy[2] * h[0] ** 2 * q[0] * v2 * z**2
            - 12 * Lsb * Lst * h[0] ** 3 * h[1] * q[0] * q[1] * v1
            - 12 * Lsb * Lst * h[0] ** 3 * h[2] * q[0] * q[2] * v2
            + 12 * Lsb * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z
            + 12 * Lsb * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1] * v1 * z**2
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2] * v2 * z**2
            + 2 * Lsb * U * h[0] ** 4 * h[1] * q[0] ** 2 * v1
            - 6 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z**2
            + 2 * Lsb * V * h[0] ** 4 * h[2] * q[0] ** 2 * v2
            - 6 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z**2
            - 3 * Lsb * dqx[0] * h[0] ** 5 * q[1] * v1
            - 12 * Lsb * dqx[0] * h[0] ** 4 * q[1] * v1 * z
            + 15 * Lsb * dqx[0] * h[0] ** 3 * q[1] * v1 * z**2
            + 3 * Lsb * dqx[1] * h[0] ** 5 * q[0] * v1
            + 12 * Lsb * dqx[1] * h[0] ** 4 * q[0] * v1 * z
            - 15 * Lsb * dqx[1] * h[0] ** 3 * q[0] * v1 * z**2
            - 3 * Lsb * dqy[0] * h[0] ** 5 * q[2] * v2
            - 12 * Lsb * dqy[0] * h[0] ** 4 * q[2] * v2 * z
            + 15 * Lsb * dqy[0] * h[0] ** 3 * q[2] * v2 * z**2
            + 3 * Lsb * dqy[2] * h[0] ** 5 * q[0] * v2
            + 12 * Lsb * dqy[2] * h[0] ** 4 * q[0] * v2 * z
            - 15 * Lsb * dqy[2] * h[0] ** 3 * q[0] * v2 * z**2
            - 3 * Lsb * h[0] ** 4 * h[1] * q[0] * q[1] * v1
            - 3 * Lsb * h[0] ** 4 * h[2] * q[0] * q[2] * v2
            + 21 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z**2
            + 21 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z**2
            + 24 * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z
            - 24 * Lst**2 * U * h[0] * h[1] * q[0] ** 2 * v1 * z**2
            + 24 * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z
            - 24 * Lst**2 * V * h[0] * h[2] * q[0] ** 2 * v2 * z**2
            - 24 * Lst**2 * dqx[0] * h[0] ** 3 * q[1] * v1 * z
            + 12 * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v1 * z**2
            + 24 * Lst**2 * dqx[1] * h[0] ** 3 * q[0] * v1 * z
            - 12 * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v1 * z**2
            - 24 * Lst**2 * dqy[0] * h[0] ** 3 * q[2] * v2 * z
            + 12 * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z**2
            + 24 * Lst**2 * dqy[2] * h[0] ** 3 * q[0] * v2 * z
            - 12 * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v2 * z**2
            - 24 * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z
            - 24 * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z
            + 24 * Lst**2 * h[0] * h[1] * q[0] * q[1] * v1 * z**2
            + 24 * Lst**2 * h[0] * h[2] * q[0] * q[2] * v2 * z**2
            + 12 * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1 * z
            - 15 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z**2
            + 12 * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2 * z
            - 15 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z**2
            - 18 * Lst * dqx[0] * h[0] ** 4 * q[1] * v1 * z
            + 15 * Lst * dqx[0] * h[0] ** 3 * q[1] * v1 * z**2
            + 18 * Lst * dqx[1] * h[0] ** 4 * q[0] * v1 * z
            - 15 * Lst * dqx[1] * h[0] ** 3 * q[0] * v1 * z**2
            - 18 * Lst * dqy[0] * h[0] ** 4 * q[2] * v2 * z
            + 15 * Lst * dqy[0] * h[0] ** 3 * q[2] * v2 * z**2
            + 18 * Lst * dqy[2] * h[0] ** 4 * q[0] * v2 * z
            - 15 * Lst * dqy[2] * h[0] ** 3 * q[0] * v2 * z**2
            - 12 * Lst * h[0] ** 3 * h[1] * q[0] * q[1] * v1 * z
            - 12 * Lst * h[0] ** 3 * h[2] * q[0] * q[2] * v2 * z
            + 21 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z**2
            + 21 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z**2
            + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * v1 * z
            - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1 * z**2
            + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * v2 * z
            - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2 * z**2
            - 3 * dqx[0] * h[0] ** 5 * q[1] * v1 * z
            + 3 * dqx[0] * h[0] ** 4 * q[1] * v1 * z**2
            + 3 * dqx[1] * h[0] ** 5 * q[0] * v1 * z
            - 3 * dqx[1] * h[0] ** 4 * q[0] * v1 * z**2
            - 3 * dqy[0] * h[0] ** 5 * q[2] * v2 * z
            + 3 * dqy[0] * h[0] ** 4 * q[2] * v2 * z**2
            + 3 * dqy[2] * h[0] ** 5 * q[0] * v2 * z
            - 3 * dqy[2] * h[0] ** 4 * q[0] * v2 * z**2
            - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * v1 * z
            - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * v2 * z
            + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * v1 * z**2
            + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * v2 * z**2
        )
        / (
            h[0] ** 2
            * q[0] ** 2
            * (
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
        )
    )

    tau_yy = (
        2
        * (
            -72 * Lsb**2 * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v2
            + 72 * Lsb**2 * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v2
            - 72 * Lsb**2 * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v1
            + 72 * Lsb**2 * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v1
            - 60 * Lsb**2 * Lst * dqx[0] * h[0] ** 3 * q[1] * v2
            + 36 * Lsb**2 * Lst * dqx[0] * h[0] * q[1] * v2 * z**2
            + 60 * Lsb**2 * Lst * dqx[1] * h[0] ** 3 * q[0] * v2
            - 36 * Lsb**2 * Lst * dqx[1] * h[0] * q[0] * v2 * z**2
            - 60 * Lsb**2 * Lst * dqy[0] * h[0] ** 3 * q[2] * v1
            + 36 * Lsb**2 * Lst * dqy[0] * h[0] * q[2] * v1 * z**2
            + 60 * Lsb**2 * Lst * dqy[2] * h[0] ** 3 * q[0] * v1
            - 36 * Lsb**2 * Lst * dqy[2] * h[0] * q[0] * v1 * z**2
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v2
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            + 36 * Lsb**2 * Lst * h[1] * q[0] * q[1] * v2 * z**2
            + 36 * Lsb**2 * Lst * h[2] * q[0] * q[2] * v1 * z**2
            - 12 * Lsb**2 * dqx[0] * h[0] ** 4 * q[1] * v2
            + 12 * Lsb**2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z**2
            + 12 * Lsb**2 * dqx[1] * h[0] ** 4 * q[0] * v2
            - 12 * Lsb**2 * dqx[1] * h[0] ** 2 * q[0] * v2 * z**2
            - 12 * Lsb**2 * dqy[0] * h[0] ** 4 * q[2] * v1
            + 12 * Lsb**2 * dqy[0] * h[0] ** 2 * q[2] * v1 * z**2
            + 12 * Lsb**2 * dqy[2] * h[0] ** 4 * q[0] * v1
            - 12 * Lsb**2 * dqy[2] * h[0] ** 2 * q[0] * v1 * z**2
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1] * v2 * z**2
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2] * v1 * z**2
            + 24 * Lsb * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
            - 36 * Lsb * Lst**2 * U * h[1] * q[0] ** 2 * v2 * z**2
            + 24 * Lsb * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
            - 36 * Lsb * Lst**2 * V * h[2] * q[0] ** 2 * v1 * z**2
            - 24 * Lsb * Lst**2 * dqx[0] * h[0] ** 3 * q[1] * v2
            - 72 * Lsb * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z
            + 36 * Lsb * Lst**2 * dqx[0] * h[0] * q[1] * v2 * z**2
            + 24 * Lsb * Lst**2 * dqx[1] * h[0] ** 3 * q[0] * v2
            + 72 * Lsb * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v2 * z
            - 36 * Lsb * Lst**2 * dqx[1] * h[0] * q[0] * v2 * z**2
            - 24 * Lsb * Lst**2 * dqy[0] * h[0] ** 3 * q[2] * v1
            - 72 * Lsb * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v1 * z
            + 36 * Lsb * Lst**2 * dqy[0] * h[0] * q[2] * v1 * z**2
            + 24 * Lsb * Lst**2 * dqy[2] * h[0] ** 3 * q[0] * v1
            + 72 * Lsb * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v1 * z
            - 36 * Lsb * Lst**2 * dqy[2] * h[0] * q[0] * v1 * z**2
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            + 36 * Lsb * Lst**2 * h[1] * q[0] * q[1] * v2 * z**2
            + 36 * Lsb * Lst**2 * h[2] * q[0] * q[2] * v1 * z**2
            + 12 * Lsb * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
            - 24 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2 * v2 * z**2
            + 12 * Lsb * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
            - 24 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2 * v1 * z**2
            - 18 * Lsb * Lst * dqx[0] * h[0] ** 4 * q[1] * v2
            - 60 * Lsb * Lst * dqx[0] * h[0] ** 3 * q[1] * v2 * z
            + 60 * Lsb * Lst * dqx[0] * h[0] ** 2 * q[1] * v2 * z**2
            + 18 * Lsb * Lst * dqx[1] * h[0] ** 4 * q[0] * v2
            + 60 * Lsb * Lst * dqx[1] * h[0] ** 3 * q[0] * v2 * z
            - 60 * Lsb * Lst * dqx[1] * h[0] ** 2 * q[0] * v2 * z**2
            - 18 * Lsb * Lst * dqy[0] * h[0] ** 4 * q[2] * v1
            - 60 * Lsb * Lst * dqy[0] * h[0] ** 3 * q[2] * v1 * z
            + 60 * Lsb * Lst * dqy[0] * h[0] ** 2 * q[2] * v1 * z**2
            + 18 * Lsb * Lst * dqy[2] * h[0] ** 4 * q[0] * v1
            + 60 * Lsb * Lst * dqy[2] * h[0] ** 3 * q[0] * v1 * z
            - 60 * Lsb * Lst * dqy[2] * h[0] ** 2 * q[0] * v1 * z**2
            - 12 * Lsb * Lst * h[0] ** 3 * h[1] * q[0] * q[1] * v2
            - 12 * Lsb * Lst * h[0] ** 3 * h[2] * q[0] * q[2] * v1
            + 12 * Lsb * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z
            + 12 * Lsb * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1] * v2 * z**2
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2] * v1 * z**2
            + 2 * Lsb * U * h[0] ** 4 * h[1] * q[0] ** 2 * v2
            - 6 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z**2
            + 2 * Lsb * V * h[0] ** 4 * h[2] * q[0] ** 2 * v1
            - 6 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z**2
            - 3 * Lsb * dqx[0] * h[0] ** 5 * q[1] * v2
            - 12 * Lsb * dqx[0] * h[0] ** 4 * q[1] * v2 * z
            + 15 * Lsb * dqx[0] * h[0] ** 3 * q[1] * v2 * z**2
            + 3 * Lsb * dqx[1] * h[0] ** 5 * q[0] * v2
            + 12 * Lsb * dqx[1] * h[0] ** 4 * q[0] * v2 * z
            - 15 * Lsb * dqx[1] * h[0] ** 3 * q[0] * v2 * z**2
            - 3 * Lsb * dqy[0] * h[0] ** 5 * q[2] * v1
            - 12 * Lsb * dqy[0] * h[0] ** 4 * q[2] * v1 * z
            + 15 * Lsb * dqy[0] * h[0] ** 3 * q[2] * v1 * z**2
            + 3 * Lsb * dqy[2] * h[0] ** 5 * q[0] * v1
            + 12 * Lsb * dqy[2] * h[0] ** 4 * q[0] * v1 * z
            - 15 * Lsb * dqy[2] * h[0] ** 3 * q[0] * v1 * z**2
            - 3 * Lsb * h[0] ** 4 * h[1] * q[0] * q[1] * v2
            - 3 * Lsb * h[0] ** 4 * h[2] * q[0] * q[2] * v1
            + 21 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z**2
            + 21 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z**2
            + 24 * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z
            - 24 * Lst**2 * U * h[0] * h[1] * q[0] ** 2 * v2 * z**2
            + 24 * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z
            - 24 * Lst**2 * V * h[0] * h[2] * q[0] ** 2 * v1 * z**2
            - 24 * Lst**2 * dqx[0] * h[0] ** 3 * q[1] * v2 * z
            + 12 * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z**2
            + 24 * Lst**2 * dqx[1] * h[0] ** 3 * q[0] * v2 * z
            - 12 * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * v2 * z**2
            - 24 * Lst**2 * dqy[0] * h[0] ** 3 * q[2] * v1 * z
            + 12 * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * v1 * z**2
            + 24 * Lst**2 * dqy[2] * h[0] ** 3 * q[0] * v1 * z
            - 12 * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * v1 * z**2
            - 24 * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z
            - 24 * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z
            + 24 * Lst**2 * h[0] * h[1] * q[0] * q[1] * v2 * z**2
            + 24 * Lst**2 * h[0] * h[2] * q[0] * q[2] * v1 * z**2
            + 12 * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2 * z
            - 15 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z**2
            + 12 * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1 * z
            - 15 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z**2
            - 18 * Lst * dqx[0] * h[0] ** 4 * q[1] * v2 * z
            + 15 * Lst * dqx[0] * h[0] ** 3 * q[1] * v2 * z**2
            + 18 * Lst * dqx[1] * h[0] ** 4 * q[0] * v2 * z
            - 15 * Lst * dqx[1] * h[0] ** 3 * q[0] * v2 * z**2
            - 18 * Lst * dqy[0] * h[0] ** 4 * q[2] * v1 * z
            + 15 * Lst * dqy[0] * h[0] ** 3 * q[2] * v1 * z**2
            + 18 * Lst * dqy[2] * h[0] ** 4 * q[0] * v1 * z
            - 15 * Lst * dqy[2] * h[0] ** 3 * q[0] * v1 * z**2
            - 12 * Lst * h[0] ** 3 * h[1] * q[0] * q[1] * v2 * z
            - 12 * Lst * h[0] ** 3 * h[2] * q[0] * q[2] * v1 * z
            + 21 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z**2
            + 21 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z**2
            + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * v2 * z
            - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2 * z**2
            + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * v1 * z
            - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1 * z**2
            - 3 * dqx[0] * h[0] ** 5 * q[1] * v2 * z
            + 3 * dqx[0] * h[0] ** 4 * q[1] * v2 * z**2
            + 3 * dqx[1] * h[0] ** 5 * q[0] * v2 * z
            - 3 * dqx[1] * h[0] ** 4 * q[0] * v2 * z**2
            - 3 * dqy[0] * h[0] ** 5 * q[2] * v1 * z
            + 3 * dqy[0] * h[0] ** 4 * q[2] * v1 * z**2
            + 3 * dqy[2] * h[0] ** 5 * q[0] * v1 * z
            - 3 * dqy[2] * h[0] ** 4 * q[0] * v1 * z**2
            - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * v2 * z
            - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * v1 * z
            + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * v2 * z**2
            + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * v1 * z**2
        )
        / (
            h[0] ** 2
            * q[0] ** 2
            * (
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
        )
    )
    tau_zz = (
        2
        * v2
        * (
            -72 * Lsb**2 * Lst**2 * dqx[0] * h[0] ** 2 * q[1]
            + 72 * Lsb**2 * Lst**2 * dqx[1] * h[0] ** 2 * q[0]
            - 72 * Lsb**2 * Lst**2 * dqy[0] * h[0] ** 2 * q[2]
            + 72 * Lsb**2 * Lst**2 * dqy[2] * h[0] ** 2 * q[0]
            - 60 * Lsb**2 * Lst * dqx[0] * h[0] ** 3 * q[1]
            + 36 * Lsb**2 * Lst * dqx[0] * h[0] * q[1] * z**2
            + 60 * Lsb**2 * Lst * dqx[1] * h[0] ** 3 * q[0]
            - 36 * Lsb**2 * Lst * dqx[1] * h[0] * q[0] * z**2
            - 60 * Lsb**2 * Lst * dqy[0] * h[0] ** 3 * q[2]
            + 36 * Lsb**2 * Lst * dqy[0] * h[0] * q[2] * z**2
            + 60 * Lsb**2 * Lst * dqy[2] * h[0] ** 3 * q[0]
            - 36 * Lsb**2 * Lst * dqy[2] * h[0] * q[0] * z**2
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[1] * q[0] * q[1]
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[2] * q[0] * q[2]
            + 36 * Lsb**2 * Lst * h[1] * q[0] * q[1] * z**2
            + 36 * Lsb**2 * Lst * h[2] * q[0] * q[2] * z**2
            - 12 * Lsb**2 * dqx[0] * h[0] ** 4 * q[1]
            + 12 * Lsb**2 * dqx[0] * h[0] ** 2 * q[1] * z**2
            + 12 * Lsb**2 * dqx[1] * h[0] ** 4 * q[0]
            - 12 * Lsb**2 * dqx[1] * h[0] ** 2 * q[0] * z**2
            - 12 * Lsb**2 * dqy[0] * h[0] ** 4 * q[2]
            + 12 * Lsb**2 * dqy[0] * h[0] ** 2 * q[2] * z**2
            + 12 * Lsb**2 * dqy[2] * h[0] ** 4 * q[0]
            - 12 * Lsb**2 * dqy[2] * h[0] ** 2 * q[0] * z**2
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[1] * z**2
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[2] * z**2
            + 24 * Lsb * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2
            - 36 * Lsb * Lst**2 * U * h[1] * q[0] ** 2 * z**2
            + 24 * Lsb * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2
            - 36 * Lsb * Lst**2 * V * h[2] * q[0] ** 2 * z**2
            - 24 * Lsb * Lst**2 * dqx[0] * h[0] ** 3 * q[1]
            - 72 * Lsb * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * z
            + 36 * Lsb * Lst**2 * dqx[0] * h[0] * q[1] * z**2
            + 24 * Lsb * Lst**2 * dqx[1] * h[0] ** 3 * q[0]
            + 72 * Lsb * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * z
            - 36 * Lsb * Lst**2 * dqx[1] * h[0] * q[0] * z**2
            - 24 * Lsb * Lst**2 * dqy[0] * h[0] ** 3 * q[2]
            - 72 * Lsb * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * z
            + 36 * Lsb * Lst**2 * dqy[0] * h[0] * q[2] * z**2
            + 24 * Lsb * Lst**2 * dqy[2] * h[0] ** 3 * q[0]
            + 72 * Lsb * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * z
            - 36 * Lsb * Lst**2 * dqy[2] * h[0] * q[0] * z**2
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1]
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2]
            + 36 * Lsb * Lst**2 * h[1] * q[0] * q[1] * z**2
            + 36 * Lsb * Lst**2 * h[2] * q[0] * q[2] * z**2
            + 12 * Lsb * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2
            - 24 * Lsb * Lst * U * h[0] * h[1] * q[0] ** 2 * z**2
            + 12 * Lsb * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2
            - 24 * Lsb * Lst * V * h[0] * h[2] * q[0] ** 2 * z**2
            - 18 * Lsb * Lst * dqx[0] * h[0] ** 4 * q[1]
            - 60 * Lsb * Lst * dqx[0] * h[0] ** 3 * q[1] * z
            + 60 * Lsb * Lst * dqx[0] * h[0] ** 2 * q[1] * z**2
            + 18 * Lsb * Lst * dqx[1] * h[0] ** 4 * q[0]
            + 60 * Lsb * Lst * dqx[1] * h[0] ** 3 * q[0] * z
            - 60 * Lsb * Lst * dqx[1] * h[0] ** 2 * q[0] * z**2
            - 18 * Lsb * Lst * dqy[0] * h[0] ** 4 * q[2]
            - 60 * Lsb * Lst * dqy[0] * h[0] ** 3 * q[2] * z
            + 60 * Lsb * Lst * dqy[0] * h[0] ** 2 * q[2] * z**2
            + 18 * Lsb * Lst * dqy[2] * h[0] ** 4 * q[0]
            + 60 * Lsb * Lst * dqy[2] * h[0] ** 3 * q[0] * z
            - 60 * Lsb * Lst * dqy[2] * h[0] ** 2 * q[0] * z**2
            - 12 * Lsb * Lst * h[0] ** 3 * h[1] * q[0] * q[1]
            - 12 * Lsb * Lst * h[0] ** 3 * h[2] * q[0] * q[2]
            + 12 * Lsb * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * z
            + 12 * Lsb * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * z
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[1] * z**2
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[2] * z**2
            + 2 * Lsb * U * h[0] ** 4 * h[1] * q[0] ** 2
            - 6 * Lsb * U * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
            + 2 * Lsb * V * h[0] ** 4 * h[2] * q[0] ** 2
            - 6 * Lsb * V * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
            - 3 * Lsb * dqx[0] * h[0] ** 5 * q[1]
            - 12 * Lsb * dqx[0] * h[0] ** 4 * q[1] * z
            + 15 * Lsb * dqx[0] * h[0] ** 3 * q[1] * z**2
            + 3 * Lsb * dqx[1] * h[0] ** 5 * q[0]
            + 12 * Lsb * dqx[1] * h[0] ** 4 * q[0] * z
            - 15 * Lsb * dqx[1] * h[0] ** 3 * q[0] * z**2
            - 3 * Lsb * dqy[0] * h[0] ** 5 * q[2]
            - 12 * Lsb * dqy[0] * h[0] ** 4 * q[2] * z
            + 15 * Lsb * dqy[0] * h[0] ** 3 * q[2] * z**2
            + 3 * Lsb * dqy[2] * h[0] ** 5 * q[0]
            + 12 * Lsb * dqy[2] * h[0] ** 4 * q[0] * z
            - 15 * Lsb * dqy[2] * h[0] ** 3 * q[0] * z**2
            - 3 * Lsb * h[0] ** 4 * h[1] * q[0] * q[1]
            - 3 * Lsb * h[0] ** 4 * h[2] * q[0] * q[2]
            + 21 * Lsb * h[0] ** 2 * h[1] * q[0] * q[1] * z**2
            + 21 * Lsb * h[0] ** 2 * h[2] * q[0] * q[2] * z**2
            + 24 * Lst**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * z
            - 24 * Lst**2 * U * h[0] * h[1] * q[0] ** 2 * z**2
            + 24 * Lst**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * z
            - 24 * Lst**2 * V * h[0] * h[2] * q[0] ** 2 * z**2
            - 24 * Lst**2 * dqx[0] * h[0] ** 3 * q[1] * z
            + 12 * Lst**2 * dqx[0] * h[0] ** 2 * q[1] * z**2
            + 24 * Lst**2 * dqx[1] * h[0] ** 3 * q[0] * z
            - 12 * Lst**2 * dqx[1] * h[0] ** 2 * q[0] * z**2
            - 24 * Lst**2 * dqy[0] * h[0] ** 3 * q[2] * z
            + 12 * Lst**2 * dqy[0] * h[0] ** 2 * q[2] * z**2
            + 24 * Lst**2 * dqy[2] * h[0] ** 3 * q[0] * z
            - 12 * Lst**2 * dqy[2] * h[0] ** 2 * q[0] * z**2
            - 24 * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[1] * z
            - 24 * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[2] * z
            + 24 * Lst**2 * h[0] * h[1] * q[0] * q[1] * z**2
            + 24 * Lst**2 * h[0] * h[2] * q[0] * q[2] * z**2
            + 12 * Lst * U * h[0] ** 3 * h[1] * q[0] ** 2 * z
            - 15 * Lst * U * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
            + 12 * Lst * V * h[0] ** 3 * h[2] * q[0] ** 2 * z
            - 15 * Lst * V * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
            - 18 * Lst * dqx[0] * h[0] ** 4 * q[1] * z
            + 15 * Lst * dqx[0] * h[0] ** 3 * q[1] * z**2
            + 18 * Lst * dqx[1] * h[0] ** 4 * q[0] * z
            - 15 * Lst * dqx[1] * h[0] ** 3 * q[0] * z**2
            - 18 * Lst * dqy[0] * h[0] ** 4 * q[2] * z
            + 15 * Lst * dqy[0] * h[0] ** 3 * q[2] * z**2
            + 18 * Lst * dqy[2] * h[0] ** 4 * q[0] * z
            - 15 * Lst * dqy[2] * h[0] ** 3 * q[0] * z**2
            - 12 * Lst * h[0] ** 3 * h[1] * q[0] * q[1] * z
            - 12 * Lst * h[0] ** 3 * h[2] * q[0] * q[2] * z
            + 21 * Lst * h[0] ** 2 * h[1] * q[0] * q[1] * z**2
            + 21 * Lst * h[0] ** 2 * h[2] * q[0] * q[2] * z**2
            + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * z
            - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * z**2
            + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * z
            - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * z**2
            - 3 * dqx[0] * h[0] ** 5 * q[1] * z
            + 3 * dqx[0] * h[0] ** 4 * q[1] * z**2
            + 3 * dqx[1] * h[0] ** 5 * q[0] * z
            - 3 * dqx[1] * h[0] ** 4 * q[0] * z**2
            - 3 * dqy[0] * h[0] ** 5 * q[2] * z
            + 3 * dqy[0] * h[0] ** 4 * q[2] * z**2
            + 3 * dqy[2] * h[0] ** 5 * q[0] * z
            - 3 * dqy[2] * h[0] ** 4 * q[0] * z**2
            - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * z
            - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * z
            + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * z**2
            + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * z**2
        )
        / (
            h[0] ** 2
            * q[0] ** 2
            * (
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
        )
    )

    tau_xy = (
        2
        * eta
        * (
            -72 * Lsb**2 * Lst**2 * dqx[0] * h[0] ** 2 * q[2]
            + 72 * Lsb**2 * Lst**2 * dqx[2] * h[0] ** 2 * q[0]
            - 72 * Lsb**2 * Lst**2 * dqy[0] * h[0] ** 2 * q[1]
            + 72 * Lsb**2 * Lst**2 * dqy[1] * h[0] ** 2 * q[0]
            - 60 * Lsb**2 * Lst * dqx[0] * h[0] ** 3 * q[2]
            + 36 * Lsb**2 * Lst * dqx[0] * h[0] * q[2] * z**2
            + 60 * Lsb**2 * Lst * dqx[2] * h[0] ** 3 * q[0]
            - 36 * Lsb**2 * Lst * dqx[2] * h[0] * q[0] * z**2
            - 60 * Lsb**2 * Lst * dqy[0] * h[0] ** 3 * q[1]
            + 36 * Lsb**2 * Lst * dqy[0] * h[0] * q[1] * z**2
            + 60 * Lsb**2 * Lst * dqy[1] * h[0] ** 3 * q[0]
            - 36 * Lsb**2 * Lst * dqy[1] * h[0] * q[0] * z**2
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[1] * q[0] * q[2]
            + 12 * Lsb**2 * Lst * h[0] ** 2 * h[2] * q[0] * q[1]
            + 36 * Lsb**2 * Lst * h[1] * q[0] * q[2] * z**2
            + 36 * Lsb**2 * Lst * h[2] * q[0] * q[1] * z**2
            - 12 * Lsb**2 * dqx[0] * h[0] ** 4 * q[2]
            + 12 * Lsb**2 * dqx[0] * h[0] ** 2 * q[2] * z**2
            + 12 * Lsb**2 * dqx[2] * h[0] ** 4 * q[0]
            - 12 * Lsb**2 * dqx[2] * h[0] ** 2 * q[0] * z**2
            - 12 * Lsb**2 * dqy[0] * h[0] ** 4 * q[1]
            + 12 * Lsb**2 * dqy[0] * h[0] ** 2 * q[1] * z**2
            + 12 * Lsb**2 * dqy[1] * h[0] ** 4 * q[0]
            - 12 * Lsb**2 * dqy[1] * h[0] ** 2 * q[0] * z**2
            + 24 * Lsb**2 * h[0] * h[1] * q[0] * q[2] * z**2
            + 24 * Lsb**2 * h[0] * h[2] * q[0] * q[1] * z**2
            + 24 * Lsb * Lst**2 * U * h[0] ** 2 * h[2] * q[0] ** 2
            - 36 * Lsb * Lst**2 * U * h[2] * q[0] ** 2 * z**2
            + 24 * Lsb * Lst**2 * V * h[0] ** 2 * h[1] * q[0] ** 2
            - 36 * Lsb * Lst**2 * V * h[1] * q[0] ** 2 * z**2
            - 24 * Lsb * Lst**2 * dqx[0] * h[0] ** 3 * q[2]
            - 72 * Lsb * Lst**2 * dqx[0] * h[0] ** 2 * q[2] * z
            + 36 * Lsb * Lst**2 * dqx[0] * h[0] * q[2] * z**2
            + 24 * Lsb * Lst**2 * dqx[2] * h[0] ** 3 * q[0]
            + 72 * Lsb * Lst**2 * dqx[2] * h[0] ** 2 * q[0] * z
            - 36 * Lsb * Lst**2 * dqx[2] * h[0] * q[0] * z**2
            - 24 * Lsb * Lst**2 * dqy[0] * h[0] ** 3 * q[1]
            - 72 * Lsb * Lst**2 * dqy[0] * h[0] ** 2 * q[1] * z
            + 36 * Lsb * Lst**2 * dqy[0] * h[0] * q[1] * z**2
            + 24 * Lsb * Lst**2 * dqy[1] * h[0] ** 3 * q[0]
            + 72 * Lsb * Lst**2 * dqy[1] * h[0] ** 2 * q[0] * z
            - 36 * Lsb * Lst**2 * dqy[1] * h[0] * q[0] * z**2
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[2]
            - 24 * Lsb * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[1]
            + 36 * Lsb * Lst**2 * h[1] * q[0] * q[2] * z**2
            + 36 * Lsb * Lst**2 * h[2] * q[0] * q[1] * z**2
            + 12 * Lsb * Lst * U * h[0] ** 3 * h[2] * q[0] ** 2
            - 24 * Lsb * Lst * U * h[0] * h[2] * q[0] ** 2 * z**2
            + 12 * Lsb * Lst * V * h[0] ** 3 * h[1] * q[0] ** 2
            - 24 * Lsb * Lst * V * h[0] * h[1] * q[0] ** 2 * z**2
            - 18 * Lsb * Lst * dqx[0] * h[0] ** 4 * q[2]
            - 60 * Lsb * Lst * dqx[0] * h[0] ** 3 * q[2] * z
            + 60 * Lsb * Lst * dqx[0] * h[0] ** 2 * q[2] * z**2
            + 18 * Lsb * Lst * dqx[2] * h[0] ** 4 * q[0]
            + 60 * Lsb * Lst * dqx[2] * h[0] ** 3 * q[0] * z
            - 60 * Lsb * Lst * dqx[2] * h[0] ** 2 * q[0] * z**2
            - 18 * Lsb * Lst * dqy[0] * h[0] ** 4 * q[1]
            - 60 * Lsb * Lst * dqy[0] * h[0] ** 3 * q[1] * z
            + 60 * Lsb * Lst * dqy[0] * h[0] ** 2 * q[1] * z**2
            + 18 * Lsb * Lst * dqy[1] * h[0] ** 4 * q[0]
            + 60 * Lsb * Lst * dqy[1] * h[0] ** 3 * q[0] * z
            - 60 * Lsb * Lst * dqy[1] * h[0] ** 2 * q[0] * z**2
            - 12 * Lsb * Lst * h[0] ** 3 * h[1] * q[0] * q[2]
            - 12 * Lsb * Lst * h[0] ** 3 * h[2] * q[0] * q[1]
            + 12 * Lsb * Lst * h[0] ** 2 * h[1] * q[0] * q[2] * z
            + 12 * Lsb * Lst * h[0] ** 2 * h[2] * q[0] * q[1] * z
            + 48 * Lsb * Lst * h[0] * h[1] * q[0] * q[2] * z**2
            + 48 * Lsb * Lst * h[0] * h[2] * q[0] * q[1] * z**2
            + 2 * Lsb * U * h[0] ** 4 * h[2] * q[0] ** 2
            - 6 * Lsb * U * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
            + 2 * Lsb * V * h[0] ** 4 * h[1] * q[0] ** 2
            - 6 * Lsb * V * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
            - 3 * Lsb * dqx[0] * h[0] ** 5 * q[2]
            - 12 * Lsb * dqx[0] * h[0] ** 4 * q[2] * z
            + 15 * Lsb * dqx[0] * h[0] ** 3 * q[2] * z**2
            + 3 * Lsb * dqx[2] * h[0] ** 5 * q[0]
            + 12 * Lsb * dqx[2] * h[0] ** 4 * q[0] * z
            - 15 * Lsb * dqx[2] * h[0] ** 3 * q[0] * z**2
            - 3 * Lsb * dqy[0] * h[0] ** 5 * q[1]
            - 12 * Lsb * dqy[0] * h[0] ** 4 * q[1] * z
            + 15 * Lsb * dqy[0] * h[0] ** 3 * q[1] * z**2
            + 3 * Lsb * dqy[1] * h[0] ** 5 * q[0]
            + 12 * Lsb * dqy[1] * h[0] ** 4 * q[0] * z
            - 15 * Lsb * dqy[1] * h[0] ** 3 * q[0] * z**2
            - 3 * Lsb * h[0] ** 4 * h[1] * q[0] * q[2]
            - 3 * Lsb * h[0] ** 4 * h[2] * q[0] * q[1]
            + 21 * Lsb * h[0] ** 2 * h[1] * q[0] * q[2] * z**2
            + 21 * Lsb * h[0] ** 2 * h[2] * q[0] * q[1] * z**2
            + 24 * Lst**2 * U * h[0] ** 2 * h[2] * q[0] ** 2 * z
            - 24 * Lst**2 * U * h[0] * h[2] * q[0] ** 2 * z**2
            + 24 * Lst**2 * V * h[0] ** 2 * h[1] * q[0] ** 2 * z
            - 24 * Lst**2 * V * h[0] * h[1] * q[0] ** 2 * z**2
            - 24 * Lst**2 * dqx[0] * h[0] ** 3 * q[2] * z
            + 12 * Lst**2 * dqx[0] * h[0] ** 2 * q[2] * z**2
            + 24 * Lst**2 * dqx[2] * h[0] ** 3 * q[0] * z
            - 12 * Lst**2 * dqx[2] * h[0] ** 2 * q[0] * z**2
            - 24 * Lst**2 * dqy[0] * h[0] ** 3 * q[1] * z
            + 12 * Lst**2 * dqy[0] * h[0] ** 2 * q[1] * z**2
            + 24 * Lst**2 * dqy[1] * h[0] ** 3 * q[0] * z
            - 12 * Lst**2 * dqy[1] * h[0] ** 2 * q[0] * z**2
            - 24 * Lst**2 * h[0] ** 2 * h[1] * q[0] * q[2] * z
            - 24 * Lst**2 * h[0] ** 2 * h[2] * q[0] * q[1] * z
            + 24 * Lst**2 * h[0] * h[1] * q[0] * q[2] * z**2
            + 24 * Lst**2 * h[0] * h[2] * q[0] * q[1] * z**2
            + 12 * Lst * U * h[0] ** 3 * h[2] * q[0] ** 2 * z
            - 15 * Lst * U * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
            + 12 * Lst * V * h[0] ** 3 * h[1] * q[0] ** 2 * z
            - 15 * Lst * V * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
            - 18 * Lst * dqx[0] * h[0] ** 4 * q[2] * z
            + 15 * Lst * dqx[0] * h[0] ** 3 * q[2] * z**2
            + 18 * Lst * dqx[2] * h[0] ** 4 * q[0] * z
            - 15 * Lst * dqx[2] * h[0] ** 3 * q[0] * z**2
            - 18 * Lst * dqy[0] * h[0] ** 4 * q[1] * z
            + 15 * Lst * dqy[0] * h[0] ** 3 * q[1] * z**2
            + 18 * Lst * dqy[1] * h[0] ** 4 * q[0] * z
            - 15 * Lst * dqy[1] * h[0] ** 3 * q[0] * z**2
            - 12 * Lst * h[0] ** 3 * h[1] * q[0] * q[2] * z
            - 12 * Lst * h[0] ** 3 * h[2] * q[0] * q[1] * z
            + 21 * Lst * h[0] ** 2 * h[1] * q[0] * q[2] * z**2
            + 21 * Lst * h[0] ** 2 * h[2] * q[0] * q[1] * z**2
            + 2 * U * h[0] ** 4 * h[2] * q[0] ** 2 * z
            - 3 * U * h[0] ** 3 * h[2] * q[0] ** 2 * z**2
            + 2 * V * h[0] ** 4 * h[1] * q[0] ** 2 * z
            - 3 * V * h[0] ** 3 * h[1] * q[0] ** 2 * z**2
            - 3 * dqx[0] * h[0] ** 5 * q[2] * z
            + 3 * dqx[0] * h[0] ** 4 * q[2] * z**2
            + 3 * dqx[2] * h[0] ** 5 * q[0] * z
            - 3 * dqx[2] * h[0] ** 4 * q[0] * z**2
            - 3 * dqy[0] * h[0] ** 5 * q[1] * z
            + 3 * dqy[0] * h[0] ** 4 * q[1] * z**2
            + 3 * dqy[1] * h[0] ** 5 * q[0] * z
            - 3 * dqy[1] * h[0] ** 4 * q[0] * z**2
            - 3 * h[0] ** 4 * h[1] * q[0] * q[2] * z
            - 3 * h[0] ** 4 * h[2] * q[0] * q[1] * z
            + 6 * h[0] ** 3 * h[1] * q[0] * q[2] * z**2
            + 6 * h[0] ** 3 * h[2] * q[0] * q[1] * z**2
        )
        / (
            h[0] ** 2
            * q[0] ** 2
            * (
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
        )
    )

    tau_xz = (
        2
        * eta
        * (
            -6 * Lsb * q[1] * z
            - 6 * Lst * U * h[0] * q[0]
            + 6 * Lst * U * q[0] * z
            + 6 * Lst * h[0] * q[1]
            - 6 * Lst * q[1] * z
            - 2 * U * h[0] ** 2 * q[0]
            + 3 * U * h[0] * q[0] * z
            + 3 * h[0] ** 2 * q[1]
            - 6 * h[0] * q[1] * z
        )
        / (h[0] * q[0] * (12 * Lsb * Lst + 4 * Lsb * h[0] + 4 * Lst * h[0] + h[0] ** 2))
    )

    tau_yz = (
        2
        * eta
        * (
            -6 * Lsb * q[2] * z
            - 6 * Lst * V * h[0] * q[0]
            + 6 * Lst * V * q[0] * z
            + 6 * Lst * h[0] * q[2]
            - 6 * Lst * q[2] * z
            - 2 * V * h[0] ** 2 * q[0]
            + 3 * V * h[0] * q[0] * z
            + 3 * h[0] ** 2 * q[2]
            - 6 * h[0] * q[2] * z
        )
        / (h[0] * q[0] * (12 * Lsb * Lst + 4 * Lsb * h[0] + 4 * Lst * h[0] + h[0] ** 2))
    )

    return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy
