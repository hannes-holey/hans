"""
Twin parabolic slider bearing — Bayada cavitation test case.

Run generate_topography.py first to create the height field file:
    python generate_topography.py

Then run this script:
    python run.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('twin_parabolic_slider.yaml')

# --- Tracking callback ---
rho_l = problem.prop['rho_l']
dx = problem.grid['dx']
Nx = problem.grid['Nx']

# x-coordinates of cell centres (inner grid, no ghost)
x_cells = (np.arange(Nx) + 0.5) * dx

R_norm_log = []       # list of lists (per timestep, per Newton iteration)
cav_boundary_log = []  # x-position of first cavitation onset per timestep

def track_cavitation():
    # R_norm history: solver already stores it (skip initial callback before first step)
    if not problem.solver.R_norm_history:
        return
    R_norm_log.append(list(problem.solver.R_norm_history[-1]))

    # Cavitation boundary: find leftmost x where rho < rho_l (inner grid)
    rho_inner = problem.q[0][1:-1, 1:-1]  # (Nx, Ny)
    rho_x = rho_inner.mean(axis=1)         # average over y
    cav_mask = rho_x < rho_l
    if np.any(cav_mask):
        cav_boundary_log.append(float(x_cells[cav_mask][0]))
    else:
        cav_boundary_log.append(np.nan)

problem.add_callback(track_cavitation)
problem.run()

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

# Subplot 1: R_norm per Newton iteration (all timesteps concatenated)
all_R = []
boundaries = [0]
for step_R in R_norm_log:
    all_R.extend(step_R)
    boundaries.append(len(all_R))
ax1.semilogy(all_R, 'k-', lw=0.8)
for b in boundaries[1:-1]:
    ax1.axvline(b, color='grey', lw=0.3, ls='--')
ax1.set_ylabel('R_norm')
ax1.set_xlabel('Newton iteration (cumulative)')
ax1.set_title('Residual norm')

# Subplot 2: cavitation boundary position vs timestep
ax2.plot(cav_boundary_log, 'b.-', lw=0.8, ms=3)
ax2.set_ylabel('x_cav [m]')
ax2.set_xlabel('Timestep')
ax2.set_title('Cavitation boundary (leftmost x where rho < rho_l)')

fig.tight_layout()
fig.savefig('convergence_tracking.png', dpi=150)
plt.show()
print('Saved convergence_tracking.png')
