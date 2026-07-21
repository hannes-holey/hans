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

# flake8: noqa: W503
import warnings

import numpy as np
import jax.numpy as jnp
from typing import TYPE_CHECKING

from ..parallel import MPI
from ..models.pressure import eos_drho_dp
from ..models.viscosity import piezoviscosity
from .scaling import build_scaling_from_blocks

if TYPE_CHECKING:
    from ..solver_fem import FEMSolver


# Maximum allowed relative change per Newton step
MAX_REL_CHANGE = 0.001
# Minimum allowed density (hard floor after update)
RHO_MIN = 1e-10
# Minimum absolute value for relative change denominator (avoids div-by-zero for jx/jy ~ 0)
ABS_FLOOR = 1e-10

# Linearization guard: max allowed relative change in dp/drho
DPDRHO_MAX_REL_CHANGE = 0.5
# Maximum bisection iterations
DPDRHO_BISECT_MAX_ITER = 25

# Viscosity growth guard: max allowed relative change in piezoviscosity per Newton step
VISCOSITY_MAX_REL_CHANGE = 0.05
# Maximum bisection iterations
VISCOSITY_BISECT_MAX_ITER = 25


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
                        solver: "FEMSolver",
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
    solver : FEMSolver
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


def report_jacobian_block_changes(M_old: np.ndarray, M_new: np.ndarray,
                                  block_order: dict, comm) -> None:
    """Print a table of per-block Frobenius relative changes in the Jacobian.

    For each (res, var) block in the assembled COO Jacobian, computes

        rel = ||M_new_block - M_old_block||_F / ||M_old_block||_F

    and prints rows sorted by descending relative change. Useful for
    identifying which physical coupling drives Jacobian variation between
    Newton iterations.

    Parameters
    ----------
    M_old, M_new : ndarray, shape (n_nnz,)
        COO value arrays from two consecutive assemble_matrix calls.
    block_order : dict
        assembly.block_order — maps (res, var) -> {'nnz_idx_start', 'nb_nnz'}.
    comm : MPI communicator
    """
    results = []
    for (res, var), block in block_order.items():
        s = block['nnz_idx_start']
        n = block['nb_nnz']
        diff = M_new[s:s + n] - M_old[s:s + n]
        diff_sq = comm.allreduce(float(np.sum(diff**2)), op=MPI.SUM)
        old_sq = comm.allreduce(float(np.sum(M_old[s:s + n]**2)), op=MPI.SUM)
        rel = np.sqrt(diff_sq) / max(np.sqrt(old_sq), ABS_FLOOR)
        results.append(((res, var), rel))

    if comm.Get_rank() == 0:
        results.sort(key=lambda x: x[1], reverse=True)
        print("  [JacBlock] Frobenius relative changes:")
        for (res, var), rel in results:
            print(f"    ({res:12s}, {var:3s})  rel={rel:.4e}")


def _eval_dpdrho_from_p(p_flat: np.ndarray, Nx: int, Ny: int,
                        prop: dict) -> np.ndarray:
    """Evaluate dp/drho = 1 / (drho/dp) at given pressure values.

    Parameters
    ----------
    p_flat : ndarray, shape (Nx*Ny,)
        Pressure values in Fortran-order flat layout.
    Nx, Ny : int
        P1 inner grid dimensions.
    prop : dict
        Material properties passed to eos_drho_dp.

    Returns
    -------
    ndarray, shape (Nx*Ny,)
        dp/drho at each node, same flat layout as input.
    """
    p_2d = jnp.asarray(p_flat.reshape((Nx, Ny), order='F'))
    drho_dp_2d = np.asarray(eos_drho_dp(p_2d, prop))
    return (1.0 / drho_dp_2d).ravel(order='F')


def linearization_guard_p(q: np.ndarray, dq: np.ndarray,
                          solver: "FEMSolver",
                          max_rel_change: float = DPDRHO_MAX_REL_CHANGE,
                          ) -> tuple:
    """Limit the Newton update so that dp/drho does not change too much.

    Pressure-based analogue of linearization_guard. The Jacobian is built
    from dp/drho evaluated at the current pressure state. If the Newton step
    moves p into a region where dp/drho differs significantly, the
    linearization is no longer valid. This guard uses bisection to find the
    largest scaling factor f in (0, 1] such that

        max_nodes |dp/drho(p + f*dp) - dp/drho(p)| / |dp/drho(p)|
            <= max_rel_change

    dp/drho is evaluated as 1 / eos_drho_dp(p, prop), consistent with the
    EOS used in the solver.

    Parameters
    ----------
    q : ndarray
        Current solution vector (inner DOFs).
    dq : ndarray
        Proposed Newton update (already multiplied by alpha).
    solver : FEMSolver
        Solver instance (needs _sol_slices, problem.prop, problem.decomp).
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
    p_sl = solver._sol_slices['p']
    Nx, Ny = solver.problem.decomp.nb_subdomain_grid_pts
    prop = solver.problem.prop

    p_old = q[p_sl]
    dp = dq[p_sl]
    dpdrho_old = _eval_dpdrho_from_p(p_old, Nx, Ny, prop)

    # --- Check full step (f = 1) first ---
    dpdrho_new = _eval_dpdrho_from_p(p_old + dp, Nx, Ny, prop)
    rel_full = np.abs(dpdrho_new - dpdrho_old) / np.maximum(np.abs(dpdrho_old), ABS_FLOOR)
    worst_rel_glob = comm.allreduce(float(np.max(rel_full)), op=MPI.MAX)

    if worst_rel_glob <= max_rel_change:
        return q + dq, False

    # --- Bisect to find the largest safe scaling factor ---
    f_lo = 0.0
    f_hi = 1.0
    bisect_rtol = 0.1

    for _ in range(DPDRHO_BISECT_MAX_ITER):
        f_mid = 0.5 * (f_lo + f_hi)
        dpdrho_trial = _eval_dpdrho_from_p(p_old + f_mid * dp, Nx, Ny, prop)
        rel_trial = np.abs(dpdrho_trial - dpdrho_old) / np.maximum(
            np.abs(dpdrho_old), ABS_FLOOR)
        trial_rel_glob = comm.allreduce(float(np.max(rel_trial)), op=MPI.MAX)

        if trial_rel_glob <= max_rel_change:
            f_lo = f_mid
        else:
            f_hi = f_mid

        if f_lo > 0.0 and (f_hi - f_lo) < bisect_rtol * f_lo:
            break

    f = f_lo if f_lo > 0.0 else f_hi

    if comm.Get_rank() == 0:
        dpdrho_accepted = _eval_dpdrho_from_p(p_old + f * dp, Nx, Ny, prop)
        rel_accepted = np.abs(dpdrho_accepted - dpdrho_old) / np.maximum(
            np.abs(dpdrho_old), ABS_FLOOR)
        rel_accepted_2d = rel_accepted.reshape((Nx, Ny), order='F')
        imax = np.unravel_index(np.argmax(rel_accepted_2d), rel_accepted_2d.shape)
        dpdrho_old_2d = dpdrho_old.reshape((Nx, Ny), order='F')
        dpdrho_accepted_2d = dpdrho_accepted.reshape((Nx, Ny), order='F')
        achieved_rel = float(rel_accepted_2d[imax])
        satisfied = f_lo > 0.0
        status = "achieved" if satisfied else "best effort"
        print(f"  [LinGuard-p] f={f:.4e}, node ({imax[0]},{imax[1]}):"
              f" dp/drho {dpdrho_old_2d[imax]:.4e} -> {dpdrho_accepted_2d[imax]:.4e}"
              f" ({achieved_rel:.3f}/{max_rel_change}) [{status}]")
        if not satisfied:
            warnings.warn(
                f"LinearizationGuardP: bisection did not converge after "
                f"{DPDRHO_BISECT_MAX_ITER} iterations at f={f:.6e}. "
                f"Achieved rel. change {achieved_rel:.3f} "
                f"(target {max_rel_change}). "
                f"This likely indicates a near-discontinuity in dp/drho "
                f"(e.g. EOS phase boundary). Consider using a smoothed EOS.",
                stacklevel=2)

    return q + f * dq, True


def viscosity_growth_guard_scale(q: np.ndarray, dq: np.ndarray,
                                 solver: "FEMSolver", eta_prev: np.ndarray,
                                 max_rel_change: float = VISCOSITY_MAX_REL_CHANGE,
                                 ) -> tuple:
    """Find the largest safe scaling factor for a proposed Newton step so that
    piezoviscosity does not grow/shrink too fast.

    Piezoviscosity mu(p) (Barus/Roelands) is exponential in pressure, so an
    overshooting Newton step in p can spike the viscosity field by orders of
    magnitude, poisoning the next Jacobian assembly. This guard uses
    bisection to find the largest scaling factor f in (0, 1] such that

        max_nodes |mu(p + f*dp) - eta_prev| / |eta_prev| <= max_rel_change

    where eta_prev is the piezoviscosity from the previous Newton iteration.
    A no-op (f=1) if no piezoviscosity model is configured. The caller is
    responsible for applying f to dq before the Newton step.

    Parameters
    ----------
    q : ndarray
        Current solution vector (inner DOFs), before this Newton step.
    dq : ndarray
        Proposed Newton update (already multiplied by alpha).
    solver : FEMSolver
        Solver instance (needs _sol_slices, problem.prop, problem.decomp).
    eta_prev : ndarray
        Piezoviscosity field from the previous Newton iteration, nodal
        (Nx, Ny) layout.
    max_rel_change : float
        Maximum allowed relative change in piezoviscosity (default 0.2).

    Returns
    -------
    f : float
        Safe scaling factor in (0, 1] to apply to dq.
    guard_fired : bool
        True if the step had to be reduced (f < 1).
    """
    prop = solver.problem.prop
    piezo_dict = prop.get('piezo')
    if piezo_dict is None or piezo_dict.get('name') not in ('Barus', 'Roelands'):
        return 1.0, False

    comm = solver.problem.decomp._mpi_comm
    p_sl = solver._sol_slices['p']
    Nx, Ny = solver.problem.decomp.nb_subdomain_grid_pts

    def _eta_at_p(p_flat):
        p_2d = p_flat.reshape((Nx, Ny), order='F')
        eta_2d = np.asarray(piezoviscosity(p_2d, prop['shear'], piezo_dict))
        return eta_2d.ravel(order='F')

    p_old = q[p_sl]
    dp = dq[p_sl]
    eta_prev_flat = eta_prev.ravel(order='F')

    # --- Check full step (f = 1) first ---
    eta_new = _eta_at_p(p_old + dp)
    rel_full = np.abs(eta_new - eta_prev_flat) / np.maximum(np.abs(eta_prev_flat), ABS_FLOOR)
    worst_glob = comm.allreduce(float(np.max(rel_full)), op=MPI.MAX)

    if worst_glob <= max_rel_change:
        return 1.0, False

    # --- Bisect to find the largest safe scaling factor ---
    f_lo, f_hi = 0.0, 1.0
    bisect_rtol = 0.1

    for _ in range(VISCOSITY_BISECT_MAX_ITER):
        f_mid = 0.5 * (f_lo + f_hi)
        eta_trial = _eta_at_p(p_old + f_mid * dp)
        rel_trial = np.abs(eta_trial - eta_prev_flat) / np.maximum(
            np.abs(eta_prev_flat), ABS_FLOOR)
        trial_rel_glob = comm.allreduce(float(np.max(rel_trial)), op=MPI.MAX)

        if trial_rel_glob <= max_rel_change:
            f_lo = f_mid
        else:
            f_hi = f_mid

        if f_lo > 0.0 and (f_hi - f_lo) < bisect_rtol * f_lo:
            break

    f = f_lo if f_lo > 0.0 else f_hi
    satisfied = f_lo > 0.0

    if comm.Get_rank() == 0:
        eta_acc = _eta_at_p(p_old + f * dp)
        rel_acc = np.abs(eta_acc - eta_prev_flat) / np.maximum(np.abs(eta_prev_flat), ABS_FLOOR)
        rel_2d = rel_acc.reshape((Nx, Ny), order='F')
        imax = np.unravel_index(np.argmax(rel_2d), rel_2d.shape)
        eta_prev_2d = eta_prev_flat.reshape((Nx, Ny), order='F')
        eta_acc_2d = eta_acc.reshape((Nx, Ny), order='F')
        status = "OK" if satisfied else "best-effort"
        ix, iy = int(imax[0]), int(imax[1])
        #print(f"  [ViscGuard] f={f:.3f} ({status})  node ({ix},{iy}):"
        #      f" eta {eta_prev_2d[ix, iy]:.4e} -> {eta_acc_2d[ix, iy]:.4e}"
        #      f"  rel={rel_2d[ix, iy]:.3f}/{max_rel_change}")
        if not satisfied:
            warnings.warn(
                f"ViscosityGrowthGuard: bisection did not converge "
                f"({VISCOSITY_BISECT_MAX_ITER} iters). Best f={f:.3f}, "
                f"rel change={float(np.max(rel_2d)):.3f} (target {max_rel_change}). "
                f"Consider reducing newton_relax.",
                stacklevel=2)

    return f, True


def solve_linear_system(M: np.ndarray, R: np.ndarray,
                        solver: "FEMSolver", it: int = 0) -> tuple:
    """Scale the system, solve for dq, and unscale.
    Returns M_scaled for debugging purposes.

    Returns
    -------
    dq : ndarray
        Unscaled Newton update.
    M_scaled : ndarray
        The matrix passed to the linear solver (M itself if scaling is off).
    """
    fem_solver = solver.problem.fem_solver
    if fem_solver['scaling']:
        scale_interval = fem_solver['scaling_update_interval']
        step = solver.problem.step
        if (it == 0
                and (step == 0 or step % scale_interval == 0)):
            solver.scaling = build_scaling_from_blocks(
                M, solver.variables, solver.residuals, solver.assembly,
                n_iter=fem_solver['scaling_ruiz_iter'])
        M_scaled, R_scaled = solver.scaling.scale_system(M, R)
        solver.linear_solver.assemble(M_scaled, R_scaled)
        dq = solver.scaling.unscale_solution(solver.linear_solver.solve())
    else:
        M_scaled = M
        solver.linear_solver.assemble(M, R)
        dq = solver.linear_solver.solve()

    return dq, M_scaled


def line_search(q: np.ndarray, dq: np.ndarray, R_norm: float,
                solver: "FEMSolver") -> np.ndarray:
    """Backtracking line search on the proposed step q + dq.

    Clamps cavitation before evaluating residual norms if cavitation is active.
    Falls back to a fixed small step if backtracking is exhausted.

    Parameters
    ----------
    q : ndarray
        Solution vector before the step.
    dq : ndarray
        Proposed full step (already multiplied by alpha).
    R_norm : float
        Residual norm before the step (reference for acceptance).

    Returns
    -------
    q_new : ndarray
        Accepted solution vector.
    """
    fem_solver = solver.problem.fem_solver
    rank = solver.problem.decomp.rank

    q_new = q + dq
    if solver.cavitation:
        q_new = solver._clamp_cavitation(q_new)

    R_new_norm = solver.get_R_norm_global(solver.get_R(q_new))
    if rank == 0:
        print(f"  [LineSearch] R: {R_norm:.6e} -> {R_new_norm:.6e}"
              f" ({'OK' if R_new_norm < R_norm else 'INCREASED'})")

    if R_new_norm < R_norm:
        return q_new

    ls_alpha = 0.5
    ls_min = fem_solver['line_search_alpha_min']
    while ls_alpha >= ls_min:
        q_trial = q + ls_alpha * dq
        if solver.cavitation:
            q_trial = solver._clamp_cavitation(q_trial)
        R_trial_norm = solver.get_R_norm_global(solver.get_R(q_trial))
        if R_trial_norm < R_norm:
            if rank == 0:
                print(f"  [LineSearch] accepted ls_alpha={ls_alpha:.2e},"
                      f" R {R_norm:.4e} -> {R_trial_norm:.4e}")
            return q_trial
        ls_alpha *= 0.5

    fallback_alpha = 1e-1
    q_fallback = q + fallback_alpha * dq
    if solver.cavitation:
        q_fallback = solver._clamp_cavitation(q_fallback)
    if rank == 0:
        print(f"  [LineSearch] exhausted (ls_min={ls_min:.2e}),"
              f" falling back to alpha={fallback_alpha:.2e} and continuing")
    return q_fallback


def apply_guards(q: np.ndarray, dq: np.ndarray, solver: "FEMSolver") -> tuple:
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
    solver : FEMSolver
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
