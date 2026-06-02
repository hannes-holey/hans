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
"""Linearization guard for the Bayada-Chupin EoS.

The Bayada-Chupin pressure law has a kink at p(rho_l): dp/drho jumps from
c_v²*rho_v/rho_l (mixture side) to c_l² (liquid side). A Newton step that
crosses this boundary sees a Jacobian built on the wrong side and can
overshoot badly.

The guard works in pressure space (p is the Newton DOF). Before applying
the update dq, it checks how much dp/drho changes along the step. If the
relative change exceeds `max_rel_change`, bisection finds the largest safe
scaling factor f so the kink is approached but not crossed in one step.
"""

import warnings

import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
from typing import TYPE_CHECKING

from ..models.pressure import eos_drho_dp

if TYPE_CHECKING:
    from ..solver_fem import FEMSolver


# Default tolerance: allow at most 50% relative change in dp/drho per step.
DEFAULT_MAX_REL_CHANGE = 0.8
# Maximum bisection iterations before giving up and taking the best effort step.
MAX_BISECT_ITER = 25
# Floor for the denominator to avoid division by zero near p=0.
_ABS_FLOOR = 1e-10


def _dpdrho_at_p(p_flat: np.ndarray, Nx: int, Ny: int, prop: dict) -> np.ndarray:
    """Evaluate dp/drho = 1/eos_drho_dp(p) on the flat (Fortran-order) P1 grid."""
    p_2d = jnp.asarray(p_flat.reshape((Nx, Ny), order='F'))
    drho_dp = np.asarray(eos_drho_dp(p_2d, prop))
    return (1.0 / drho_dp).ravel(order='F')


def bayada_linearization_guard(
        q: np.ndarray,
        dq: np.ndarray,
        solver: "FEMSolver",
        max_rel_change: float = DEFAULT_MAX_REL_CHANGE,
) -> tuple:
    """Limit the Newton step so dp/drho does not change by more than `max_rel_change`.

    Uses bisection to find the largest f in (0, 1] such that

        max_nodes |dp/drho(p + f*dp) - dp/drho(p)| / |dp/drho(p)| <= max_rel_change

    Parameters
    ----------
    q : ndarray
        Current solution vector (inner DOFs, pressure-based layout).
    dq : ndarray
        Proposed Newton update (already multiplied by alpha).
    solver : FEMSolver
        Active solver instance.
    max_rel_change : float
        Allowed relative change in dp/drho per step (default 0.5).

    Returns
    -------
    q_new : ndarray
        Updated solution with the guard-limited step applied.
    fired : bool
        True if the step was reduced.
    """
    comm = solver.problem.decomp._mpi_comm
    p_sl = solver._sol_slices['p']
    Nx, Ny = solver.problem.decomp.nb_subdomain_grid_pts
    prop = solver.problem.prop

    p_old = q[p_sl]
    dp = dq[p_sl]

    dpdrho_old = _dpdrho_at_p(p_old, Nx, Ny, prop)

    # Fast path: full step is safe.
    dpdrho_new = _dpdrho_at_p(p_old + dp, Nx, Ny, prop)
    rel_full = np.abs(dpdrho_new - dpdrho_old) / np.maximum(np.abs(dpdrho_old), _ABS_FLOOR)
    worst_glob = comm.allreduce(float(np.max(rel_full)), op=MPI.MAX)
    if worst_glob <= max_rel_change:
        return q + dq, False

    # Bisect for the largest safe f.
    f_lo, f_hi = 0.0, 1.0
    for _ in range(MAX_BISECT_ITER):
        f_mid = 0.5 * (f_lo + f_hi)
        dpdrho_trial = _dpdrho_at_p(p_old + f_mid * dp, Nx, Ny, prop)
        rel_trial = np.abs(dpdrho_trial - dpdrho_old) / np.maximum(np.abs(dpdrho_old), _ABS_FLOOR)
        if comm.allreduce(float(np.max(rel_trial)), op=MPI.MAX) <= max_rel_change:
            f_lo = f_mid
        else:
            f_hi = f_mid
        if f_lo > 0.0 and (f_hi - f_lo) < 0.1 * f_lo:
            break

    f = f_lo if f_lo > 0.0 else f_hi
    satisfied = f_lo > 0.0

    if comm.Get_rank() == 0:
        dpdrho_acc = _dpdrho_at_p(p_old + f * dp, Nx, Ny, prop)
        rel_acc = np.abs(dpdrho_acc - dpdrho_old) / np.maximum(np.abs(dpdrho_old), _ABS_FLOOR)
        rel_2d = rel_acc.reshape((Nx, Ny), order='F')
        imax = np.unravel_index(np.argmax(rel_2d), rel_2d.shape)
        dpdrho_old_2d = dpdrho_old.reshape((Nx, Ny), order='F')
        dpdrho_acc_2d = dpdrho_acc.reshape((Nx, Ny), order='F')
        status = "OK" if satisfied else "best-effort"
        ix, iy = int(imax[0]), int(imax[1])
        print(f"  [BayadaGuard] f={f:.3f} ({status})  node ({ix},{iy}):"
              f" dp/drho {dpdrho_old_2d[ix, iy]:.4e} -> {dpdrho_acc_2d[ix, iy]:.4e}"
              f"  rel={rel_2d[ix, iy]:.3f}/{max_rel_change}")
        if not satisfied:
            warnings.warn(
                f"BayadaLinearizationGuard: bisection did not converge "
                f"({MAX_BISECT_ITER} iters). Best f={f:.3f}, "
                f"rel change={float(np.max(rel_2d)):.3f} (target {max_rel_change}). "
                f"Consider reducing newton_relax or using a smoothed EoS.",
                stacklevel=2)

    return q + f * dq, True
