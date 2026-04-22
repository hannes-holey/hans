#
# Copyright 2025 Christoph Huber
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

"""Linear system scaling for Taylor-Hood P2P1 FEM solver conditioning."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from GaPFlow import Problem
    from .assembly import Assembly

NDArray = npt.NDArray[np.floating]


@dataclass
class ScalingInfo:
    """Precomputed scaling factors for linear system conditioning.

    Transforms J·dq = -R into J*·dq* = -R* where all entries are O(1).

    Transformation:
        J*[i,j] = J[i,j] · D_q[j] / D_R[i]
        R*[i]   = R[i] / D_R[i]
        dq[j]   = dq*[j] · D_q[j]
    """
    coo_scale:   NDArray
    rhs_scale:   NDArray
    sol_scale:   NDArray
    char_scales: Dict[str, float]

    def scale_system(self, M_coo: NDArray,
                     R: NDArray) -> Tuple[NDArray, NDArray]:
        return M_coo * self.coo_scale, R / self.rhs_scale

    def unscale_solution(self, dq_scaled: NDArray) -> NDArray:
        return dq_scaled * self.sol_scale


def build_scaling(
    problem: "Problem",
    energy: bool,
    variables: List[str],
    assembly: "Assembly",
) -> ScalingInfo:
    """Build scaling factors for linear system conditioning.

    Works for Taylor-Hood P2P1 where different variables can have different
    numbers of inner points (nb_inner_v for jx/jy, nb_inner_p for rho/e).
    The assembly.var_block_idx and res_block_idx / rhs_res_block_idx arrays
    (added in Phase 7) provide the per-entry variable/residual indices.

    Parameters
    ----------
    problem : Problem
    energy : bool
    variables : list of str
        Variable names in block order, e.g. ['jx', 'jy', 'rho'].
    assembly : Assembly
        Must have .var_block_idx, .res_block_idx, .rhs_res_block_idx.
    """
    char_scales = compute_characteristic_scales(problem, energy)

    q_scales = np.array([char_scales[v] for v in variables], dtype=np.float64)
    r_scales = q_scales  # residual scales match corresponding variable

    coo_scale = q_scales[assembly.var_block_idx] / r_scales[assembly.res_block_idx]
    rhs_scale = r_scales[assembly.rhs_res_block_idx]
    sol_scale = q_scales[assembly.rhs_res_block_idx]

    return ScalingInfo(
        coo_scale=coo_scale,
        rhs_scale=rhs_scale,
        sol_scale=sol_scale,
        char_scales=char_scales,
    )


def compute_characteristic_scales(problem: "Problem",
                                   energy: bool) -> Dict[str, float]:
    """Derive characteristic scales from problem specification."""
    rho_ref = problem.prop['rho0']
    U_ref = _get_characteristic_velocity(problem)
    c_ref = problem.prop.get('c_l', np.sqrt(problem.prop.get('P0', 1.0) / rho_ref))
    j_ref = rho_ref * U_ref * c_ref

    scales = {'rho': rho_ref, 'jx': j_ref, 'jy': j_ref}

    if energy:
        cv    = problem.energy_spec['cv']
        T_ref = problem.energy_spec['T_wall']
        E_ref = rho_ref * cv * T_ref
        scales['E'] = E_ref

    return scales


def _get_characteristic_velocity(problem: "Problem") -> float:
    u_wall_max = max(abs(problem.geo['U_bot']), abs(problem.geo['V_bot']),
                     abs(problem.geo['U_top']), abs(problem.geo['V_top']))
    if u_wall_max > 1e-10:
        return u_wall_max

    # Body-force driven flow: estimate U ~ f * Ly^2 / (8*mu)  (Poiseuille)
    force_x = problem.geo.get('force_x', 0.0)
    force_y = problem.geo.get('force_y', 0.0)
    force_mag = max(abs(force_x), abs(force_y))
    if force_mag > 0.0:
        mu = problem.prop.get('shear', problem.prop.get('mu', 1.0))
        Ly = problem.grid['Ly']
        return force_mag * Ly**2 / (8.0 * mu)

    return 1.0
