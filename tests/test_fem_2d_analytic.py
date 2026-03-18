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
"""Analytical validation tests for 2D FEM solver."""

import os
import pytest
import numpy as np

from GaPFlow import Problem
from GaPFlow.models.pressure import eos_pressure

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')


def compute_l2_error(numerical, analytical):
    """Compute L2 relative error."""
    return np.sqrt(((numerical - analytical)**2).mean()) / analytical.max()


# =============================================================================
# Test: 2D Poiseuille Flow (body force, periodic x, no-slip y walls)
# =============================================================================

@pytest.fixture
def poiseuille_problem():
    """Create and run Poiseuille flow problem."""
    config_path = os.path.join(CONFIG_DIR, 'poiseuille_2d_body_force.yaml')
    problem = Problem.from_yaml(config_path)
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_poiseuille_2d(poiseuille_problem):
    """2D Poiseuille flow with body force.

    Tests in-plane viscous diffusion (R23xy, R23yx) driven by body force.
    BCs applied at ghost cell centers: y_lo=-dy/2, y_hi=Ly+dy/2.
    """
    problem = poiseuille_problem

    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    mu = problem.prop['shear']
    f_x = problem.prop['force_x']
    rho_m = rho.mean()

    # Cell centers and analytical solution
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    y_lo, y_hi = -dy / 2, Ly + dy / 2
    u_ana = (rho_m * f_x) / (2 * mu) * (y - y_lo) * (y_hi - y)

    l2_err = compute_l2_error(vx, u_ana)

    assert l2_err < 0.01, f"L2 error = {l2_err:.2e} exceeds tolerance 0.01"


# =============================================================================
# Test: 2D Couette Flow (wall velocity, periodic x, no-slip y walls)
# =============================================================================

@pytest.fixture
def couette_problem():
    """Create and run Couette flow problem."""
    config_path = os.path.join(CONFIG_DIR, 'couette_2d_wall_velocity.yaml')
    problem = Problem.from_yaml(config_path)
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_couette_2d(couette_problem):
    """2D Couette flow with wall velocity.

    Tests in-plane viscous diffusion (R23xy, R23yx) driven by wall motion.
    BCs applied at ghost cell centers: y_lo=-dy/2, y_hi=Ly+dy/2.
    """
    problem = couette_problem

    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    rho_m = rho.mean()

    # Wall velocity from BC (jx_wall / rho = U)
    U_wall = problem.grid['bc_yN_D_val'][1] / rho_m

    # Cell centers and analytical solution (linear profile)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    y_lo, y_hi = -dy / 2, Ly + dy / 2
    u_ana = U_wall * (y - y_lo) / (y_hi - y_lo)

    l2_err = compute_l2_error(vx, u_ana)

    assert l2_err < 0.01, f"L2 error = {l2_err:.2e} exceeds tolerance 0.01"


# =============================================================================
# Test: 2D Temperature Diffusion
# =============================================================================

def test_temp_diffusion_2d():
    """2D temperature diffusion with Dirichlet T=0 boundaries.

    Tests thermal diffusion terms (R35x, R35y) against analytical solution.
    Compares at each timestep since both profiles decay to zero at steady state.
    """
    config_path = os.path.join(CONFIG_DIR, 'temp_diffusion_2d.yaml')
    problem = Problem.from_yaml(config_path)

    # Parameters for analytical solution
    Lx, Nx = problem.grid['Lx'], problem.grid['Nx']
    dx = problem.grid['dx']
    cv, k = problem.energy_spec['cv'], problem.energy_spec['k']
    rho = problem.prop['rho0']
    T_max = problem.energy_spec['T0'][2]
    L_eff = Lx + dx  # effective length (BCs at ghost cell centers)

    # Precompute spatial part of analytical solution
    x = np.arange(Nx + 2) * dx - dx / 2
    x_shifted = x + dx / 2
    sin_profile = np.sin(np.pi * x_shifted / L_eff)

    # Decay rate
    alpha = k / (cv * rho)
    lam_sq = alpha * (np.pi / L_eff)**2

    # Track max error across all timesteps
    max_l2_err = 0.0
    j_mid = problem.grid['Ny'] // 2 + 1

    def check_consistency():
        nonlocal max_l2_err
        t = problem.simtime
        T_ana = T_max * np.exp(-lam_sq * t) * sin_profile
        T_num = problem.energy.temperature[:, j_mid]
        l2_err = compute_l2_error(T_num, T_ana)
        max_l2_err = max(max_l2_err, l2_err)

    problem.add_callback(check_consistency)
    problem.run()

    assert max_l2_err < 0.02, f"Max L2 error = {max_l2_err:.2e} exceeds tolerance 0.02"
