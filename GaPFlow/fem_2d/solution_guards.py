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
from mpi4py import MPI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2d


# Maximum allowed relative density change per Newton step
MAX_DENSITY_CHANGE = 0.05
# Minimum allowed density (hard floor after update)
RHO_MIN = 1e-10


def apply_guards(q: np.ndarray, dq: np.ndarray, solver: "FEMSolver2d") -> tuple:
    """Apply solution update with physical safeguards.

    Scales dq uniformly if any density update exceeds MAX_DENSITY_CHANGE
    relative to the current density. The scaling is consistent across MPI ranks
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
        True if the density change guard was triggered.
    """
    rho_sl = solver._sol_slices['rho']
    comm = solver.problem.decomp._mpi_comm

    rho = q[rho_sl]
    d_rho = dq[rho_sl]
    max_rel_loc = float(np.max(np.abs(d_rho) / np.abs(rho)))
    max_rel_glob = comm.allreduce(max_rel_loc, op=MPI.MAX)

    guard_fired = max_rel_glob > MAX_DENSITY_CHANGE
    if guard_fired:
        scale = MAX_DENSITY_CHANGE / max_rel_glob
        dq = dq * scale
        if comm.Get_rank() == 0:
            print(f"  [SolutionGuard] density change {max_rel_glob:.3f} > {MAX_DENSITY_CHANGE},"
                  f" scaled dq by {scale:.3f}")

    q_new = q + dq
    q_new[rho_sl] = np.maximum(q_new[rho_sl], RHO_MIN)

    return q_new, guard_fired
