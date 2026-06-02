"""Parabolic slider 2D: check for mass flux oscillations.

Runs the parabolic_slider_2d use case for a few time steps and
inspects whether the P2 mass flux field shows spurious oscillations
(especially at mid-edge nodes vs corner nodes).

Usage:
    python GaPFlow/solver_fem/tests/test_parabolic_slider_oscillations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from GaPFlow.problem import Problem

# =============================================================================
# Config — smaller grid for quick diagnostics
# =============================================================================

CONFIG = """
options:
    output: /tmp/slider_osc_test
    write_freq: 1
    save_output: False

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {Nx}
    Ny: {Ny}
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['D', 'N', 'N']
    yN: ['D', 'N', 'N']
    xE_D: 1.1853
    xW_D: 1.1853
    yS_D: 1.1853
    yN_D: 1.1853

geometry:
    type: parabolic_2d
    hmax: 6.6e-5
    hmin: 1.0e-5
    U: 50.
    V: 0.

numerics:
    solver: fem
    dt: {dt}
    tol: 0.
    max_it: {max_it}

properties:
    EOS: PL
    shear: 1.846e-5
    bulk: 0.
    P0: 101325
    rho0: 1.1853
    alpha: 0.

fem_solver:
    type: newton_alpha
    linear_solver: direct
    dynamic: True
    R_norm_tol: 1e-11
    max_iter: {newton_iter}
    newton_relax: {alpha}
    physics:
        pspg: false
        gls: false
"""

# =============================================================================
# Run
# =============================================================================

Nx, Ny = 16, 16
dt = 1.0
max_it = 5
newton_iter = 50
alpha = 0.05

config = CONFIG.format(Nx=Nx, Ny=Ny, dt=dt, max_it=max_it,
                       newton_iter=newton_iter, alpha=alpha)

print(f'Parabolic slider 2D: {Nx}x{Ny}, dt={dt}, max_it={max_it}')
print(f'Newton: max_iter={newton_iter}, alpha={alpha}')

problem = Problem.from_string(config)
solver = problem.solver

# Track Newton convergence per timestep
R_norms = []

def track_norms():
    if hasattr(solver, 'R_norm_history') and solver.R_norm_history:
        R_norms.append(solver.R_norm_history[-1].copy())

problem.add_callback(track_norms)
problem.run()

# =============================================================================
# Extract P2 mass flux field
# =============================================================================

jx_nodal = solver.quad_mgr.nodal_fields['jx'].pg[0]  # padded P2
jy_nodal = solver.quad_mgr.nodal_fields['jy'].pg[0]

gi = solver.grid_idx
g = 2  # ghost depth
jx_inner = jx_nodal[g:-g, g:-g]
jy_inner = jy_nodal[g:-g, g:-g]

Nx_P2 = gi.Nx_v_inner
Ny_P2 = gi.Ny_v_inner

# P1 fields from problem.q
rho_p1 = problem.q[0][1:-1, 1:-1]
jx_p1 = problem.q[1][1:-1, 1:-1]

# =============================================================================
# Oscillation analysis
# =============================================================================

print(f'\n{"="*60}')
print('Oscillation analysis')
print(f'{"="*60}')
print(f'P2 jx inner shape: {jx_inner.shape}')
print(f'P1 rho inner shape: {rho_p1.shape}')
print(f'jx range: [{jx_inner.min():.6e}, {jx_inner.max():.6e}]')
print(f'jy range: [{jy_inner.min():.6e}, {jy_inner.max():.6e}]')
print(f'rho range: [{rho_p1.min():.6e}, {rho_p1.max():.6e}]')

# Check P2 corner nodes (even,even) vs mid-edge nodes (odd,*)
jx_corners = jx_inner[::2, ::2]
jx_mid_x = jx_inner[1::2, ::2]    # mid-edge in x (odd,even)
jx_mid_y = jx_inner[::2, 1::2]    # mid-edge in y (even,odd)
jx_mid_diag = jx_inner[1::2, 1::2]  # mid-diagonal (odd,odd)

print(f'\nP2 jx node types:')
print(f'  corners  (even,even): mean={jx_corners.mean():.6e}, std={jx_corners.std():.6e}')
print(f'  mid-x    (odd, even): mean={jx_mid_x.mean():.6e}, std={jx_mid_x.std():.6e}')
print(f'  mid-y    (even, odd): mean={jx_mid_y.mean():.6e}, std={jx_mid_y.std():.6e}')
print(f'  mid-diag (odd,  odd): mean={jx_mid_diag.mean():.6e}, std={jx_mid_diag.std():.6e}')

# Compare P2 corners to P1 values (should be close)
jx_p2_at_p1 = jx_inner[::2, ::2]
if jx_p2_at_p1.shape == jx_p1.shape:
    p2_p1_diff = jx_p2_at_p1 - jx_p1
    print(f'\nP2 corners vs P1 jx: max_diff={np.max(np.abs(p2_p1_diff)):.6e}')

# Oscillation metric: difference between adjacent nodes along x
# at mid-y slice
mid_y_idx = Ny_P2 // 2
jx_line = jx_inner[:, mid_y_idx]
jx_diff = np.diff(jx_line)
sign_changes = np.sum(np.diff(np.sign(jx_diff)) != 0)
print(f'\njx along x at mid-y (iy={mid_y_idx}):')
print(f'  sign changes in diff: {sign_changes} out of {len(jx_diff)-1}')
print(f'  max abs diff: {np.max(np.abs(jx_diff)):.6e}')

# Newton convergence
print(f'\nNewton convergence per timestep:')
for i, norms in enumerate(R_norms):
    print(f'  step {i+1}: {len(norms)} iters, '
          f'R0={norms[0]:.2e}, Rfinal={norms[-1]:.2e}')

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

# jx P2 field (full)
ax = axes[0, 0]
im = ax.imshow(jx_inner.T, origin='lower', aspect='equal', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label='jx')
ax.set_title('jx (P2, full)')
ax.set_xlabel('ix (P2)')
ax.set_ylabel('iy (P2)')

# jx P2 corners only
ax = axes[0, 1]
im = ax.imshow(jx_corners.T, origin='lower', aspect='equal', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label='jx')
ax.set_title('jx (P2 corners = P1 nodes)')
ax.set_xlabel('ix (P1)')
ax.set_ylabel('iy (P1)')

# jx P2 mid-edge x only
ax = axes[0, 2]
im = ax.imshow(jx_mid_x.T, origin='lower', aspect='equal', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label='jx')
ax.set_title('jx (P2 mid-edge x)')
ax.set_xlabel('ix (mid)')
ax.set_ylabel('iy (P1)')

# jx line along x at mid-y
ax = axes[1, 0]
ix_coords = np.arange(Nx_P2)
ax.plot(ix_coords, jx_line, 'b.-', markersize=3)
ax.plot(ix_coords[::2], jx_line[::2], 'ro', markersize=5, label='corners')
ax.plot(ix_coords[1::2], jx_line[1::2], 'gs', markersize=4, label='mid-edge')
ax.set_xlabel('ix (P2)')
ax.set_ylabel('jx')
ax.set_title(f'jx along x at iy={mid_y_idx}')
ax.legend()

# jy field
ax = axes[1, 1]
im = ax.imshow(jy_inner.T, origin='lower', aspect='equal', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label='jy')
ax.set_title('jy (P2, full)')
ax.set_xlabel('ix (P2)')
ax.set_ylabel('iy (P2)')

# rho P1 field
ax = axes[1, 2]
im = ax.imshow(rho_p1.T, origin='lower', aspect='equal', cmap='viridis')
fig.colorbar(im, ax=ax, label='rho')
ax.set_title('rho (P1)')
ax.set_xlabel('ix (P1)')
ax.set_ylabel('iy (P1)')

fig.suptitle(f'Parabolic Slider 2D: {Nx}x{Ny}, dt={dt}, {max_it} steps', fontsize=13)
plt.savefig('test_parabolic_slider_oscillations.png', dpi=150, bbox_inches='tight')
print(f'\nSaved: test_parabolic_slider_oscillations.png')
plt.show()
