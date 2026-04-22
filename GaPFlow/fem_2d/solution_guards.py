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
import warnings

import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2d


# Maximum allowed relative change per Newton step
MAX_REL_CHANGE = 0.001
# Minimum allowed density (hard floor after update)
RHO_MIN = 1e-10
# Minimum absolute value for relative change denominator (avoids div-by-zero for jx/jy ~ 0)
ABS_FLOOR = 1e-10

# Linearization guard: max allowed relative change in dp/drho
DPDRHO_MAX_REL_CHANGE = 0.3
# Maximum bisection iterations
DPDRHO_BISECT_MAX_ITER = 25


def _eval_dpdrho(rho_flat: np.ndarray, Nx: int, Ny: int,
                 dp_drho_fn) -> np.ndarray:
    """Evaluate dp/drho via the solver's JAX-autodiffed pressure derivative.

    Parameters
    ----------
    rho_flat : ndarray, shape (Nx*Ny,)
        Density values in Fortran-order flat layout.
    Nx, Ny : int
        P1 inner grid dimensions.
    dp_drho_fn : callable
        JIT-compiled vmap(vmap(grad(eos_pressure))), expects (Nx, Ny).

    Returns
    -------
    ndarray, shape (Nx*Ny,)
        dp/drho at each node, same flat layout as input.
    """
    rho_2d = rho_flat.reshape((Nx, Ny), order='F')
    dpdrho_2d = np.asarray(dp_drho_fn(rho_2d))
    return dpdrho_2d.ravel(order='F')


def linearization_guard(q: np.ndarray, dq: np.ndarray,
                        solver: "FEMSolver2d",
                        max_rel_change: float = DPDRHO_MAX_REL_CHANGE,
                        ) -> tuple:
    """Limit the Newton update so that dp/drho does not change too much.

    The Jacobian is built from dp/drho at the current state. If a Newton step
    moves rho into a region where dp/drho is significantly different, the
    linearization is no longer valid. This guard uses bisection to find the
    largest scaling factor f in (0, 1] such that

        max_nodes |dp/drho(rho + f*drho) - dp/drho(rho)| / |dp/drho(rho)|
            <= max_rel_change

    dp/drho is evaluated via the solver's JAX-autodiffed EOS derivative
    (problem.pressure.dp_drho), so it is consistent with any EOS variant
    (standard, smoothed, user-supplied).

    Parameters
    ----------
    q : ndarray
        Current solution vector (inner DOFs).
    dq : ndarray
        Proposed Newton update (already multiplied by alpha).
    solver : FEMSolver2d
        Solver instance (needs _sol_slices, problem.pressure.dp_drho,
        problem.decomp).
    max_rel_change : float
        Maximum allowed relative change in dp/drho (default 0.3 = 30%).

    Returns
    -------
    q_new : ndarray
        Updated solution with the guard-limited step applied.
    guard_fired : bool
        True if the step had to be reduced.
    """
    comm = solver.problem.decomp._mpi_comm
    rho_sl = solver._sol_slices['rho']
    Nx, Ny = solver.problem.decomp.nb_subdomain_grid_pts
    dp_drho_fn = solver.problem.pressure.dp_drho

    rho_old = q[rho_sl]
    drho = dq[rho_sl]
    dpdrho_old = _eval_dpdrho(rho_old, Nx, Ny, dp_drho_fn)

    # --- Check full step (f = 1) first ---
    dpdrho_new = _eval_dpdrho(rho_old + drho, Nx, Ny, dp_drho_fn)
    rel_full = np.abs(dpdrho_new - dpdrho_old) / np.maximum(np.abs(dpdrho_old), ABS_FLOOR)
    worst_rel_loc = float(np.max(rel_full))
    worst_rel_glob = comm.allreduce(worst_rel_loc, op=MPI.MAX)

    if worst_rel_glob <= max_rel_change:
        q_new = q + dq
        q_new[rho_sl] = np.maximum(q_new[rho_sl], RHO_MIN)
        return q_new, False

    # Identify the worst node from the full-step evaluation (for reporting)
    worst_idx = np.argmax(rel_full)

    # --- Bisect to find the largest safe scaling factor ---
    f_lo = 0.0
    f_hi = 1.0
    bisect_rtol = 0.1  # stop when interval is within 10% of f_lo

    for _ in range(DPDRHO_BISECT_MAX_ITER):
        f_mid = 0.5 * (f_lo + f_hi)
        dpdrho_trial = _eval_dpdrho(rho_old + f_mid * drho, Nx, Ny, dp_drho_fn)
        rel_trial = np.abs(dpdrho_trial - dpdrho_old) / np.maximum(
            np.abs(dpdrho_old), ABS_FLOOR)
        trial_rel_glob = comm.allreduce(float(np.max(rel_trial)), op=MPI.MAX)

        if trial_rel_glob <= max_rel_change:
            f_lo = f_mid
        else:
            f_hi = f_mid

        if f_lo > 0.0 and (f_hi - f_lo) < bisect_rtol * f_lo:
            break

    # Use f_lo if it moved (found a safe point), otherwise fall back to
    # f_hi (smallest tested value) so we always take some step.
    if f_lo > 0.0:
        f = f_lo
    else:
        f = f_hi

    if comm.Get_rank() == 0:
        # Evaluate dp/drho at the accepted step to report actual change.
        dpdrho_accepted = _eval_dpdrho(rho_old + f * drho, Nx, Ny, dp_drho_fn)
        rel_accepted = np.abs(dpdrho_accepted - dpdrho_old) / np.maximum(
            np.abs(dpdrho_old), ABS_FLOOR)
        rel_accepted_2d = rel_accepted.reshape((Nx, Ny), order='F')
        imax = np.unravel_index(np.argmax(rel_accepted_2d), rel_accepted_2d.shape)
        dpdrho_old_2d = dpdrho_old.reshape((Nx, Ny), order='F')
        dpdrho_accepted_2d = dpdrho_accepted.reshape((Nx, Ny), order='F')
        achieved_rel = float(rel_accepted_2d[imax])
        satisfied = f_lo > 0.0
        status = "achieved" if satisfied else "best effort"
        print(f"  [LinGuard] f={f:.4e}, node ({imax[0]},{imax[1]}):"
              f" dp/drho {dpdrho_old_2d[imax]:.4e} -> {dpdrho_accepted_2d[imax]:.4e}"
              f" ({achieved_rel:.3f}/{max_rel_change})")
        if not satisfied:
            warnings.warn(
                f"LinearizationGuard: bisection did not converge after "
                f"{DPDRHO_BISECT_MAX_ITER} iterations at f={f:.6e}. "
                f"Achieved rel. change {achieved_rel:.3f} "
                f"(target {max_rel_change}). "
                f"This likely indicates a near-discontinuity in dp/drho "
                f"(e.g. EOS phase boundary). Consider using a smoothed EOS.",
                stacklevel=2)

    q_new = q + f * dq
    q_new[rho_sl] = np.maximum(q_new[rho_sl], RHO_MIN)

    return q_new, True


def apply_guards(q: np.ndarray, dq: np.ndarray, solver: "FEMSolver2d") -> tuple:
    """Apply solution update with physical safeguards.

    Scales dq uniformly if any variable update exceeds MAX_REL_CHANGE
    relative to the current value. Checks rho, jx, and jy independently
    and applies the most restrictive scaling. Consistent across MPI ranks
    via allreduce. After the update, densities are clamped to RHO_MIN.

    Parameters
    ----------
    q : ndarray
        Current solution vector (inner DOFs).
    dq : ndarray
        Proposed Newton update (already multiplied by alpha).
    solver : FEMSolver2d
        Solver instance providing sol_slices and MPI communicator.

    Returns
    -------
    q_new : ndarray
        Updated solution vector with guards applied.
    guard_fired : bool
        True if any change guard was triggered.
    """
    comm = solver.problem.decomp._mpi_comm
    max_rel_glob = 0.0
    worst_var = None

    for var in ['rho']:
        sl = solver._sol_slices[var]
        denom = np.maximum(np.abs(q[sl]), ABS_FLOOR)
        max_rel_loc = float(np.max(np.abs(dq[sl]) / denom))
        max_rel_var = comm.allreduce(max_rel_loc, op=MPI.MAX)
        if max_rel_var > max_rel_glob:
            max_rel_glob = max_rel_var
            worst_var = var

    guard_fired = max_rel_glob > MAX_REL_CHANGE
    if guard_fired:
        scale = MAX_REL_CHANGE / max_rel_glob
        if comm.Get_rank() == 0:
            rho_sl = solver._sol_slices['rho']
            Nx, Ny = solver.problem.decomp.nb_subdomain_grid_pts
            rho_old = q[rho_sl].reshape((Nx, Ny), order='F')
            drho = dq[rho_sl].reshape((Nx, Ny), order='F')
            rel = np.abs(drho) / np.maximum(np.abs(rho_old), ABS_FLOOR)
            imax = np.unravel_index(np.argmax(rel), rel.shape)
            print(f"  [SolutionGuard] {worst_var} change {max_rel_glob:.3f} > {MAX_REL_CHANGE},"
                  f" scaled dq by {scale:.3f},"
                  f" worst node ({imax[0]},{imax[1]}):"
                  f" rho {rho_old[imax]:.4f} -> {rho_old[imax] + scale * drho[imax]:.4f}"
                  f" (drho={drho[imax]:.4f})")
        dq = dq * scale

    q_new = q + dq
    rho_sl = solver._sol_slices['rho']
    q_new[rho_sl] = np.maximum(q_new[rho_sl], RHO_MIN)

    return q_new, guard_fired
