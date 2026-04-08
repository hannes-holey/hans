"""Lid-driven cavity test for the Taylor-Hood P2P1 FEM solver.

Runs the lid-driven cavity problem: unit square, uniform gap, moving lid (North wall)
at U_WALL in x, all other walls stationary and impermeable.

Config is loaded from doc/tutorials/examples/lid_driven_cavity.yaml.

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/fem_2d/tests/test_lid_cavity.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from GaPFlow.problem import Problem

U_WALL = 0.01  # Lid velocity [m/s]

YAML_PATH = (
    Path(__file__).parent.parent.parent.parent
    / 'doc' / 'tutorials' / 'examples' / 'lid_driven_cavity.yaml'
)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print(f'Lid-driven cavity: config={YAML_PATH}')

    problem = Problem.from_yaml(str(YAML_PATH))

    # Taper width: fraction of Lx over which the lid velocity ramps down at each corner
    TAPER_W = 0.1

    def jx_lid(ctx):
        rho_ghost = ctx.problem.q[0][ctx.ghost_slice]
        _Lx = ctx.problem.grid['Lx']
        x = np.linspace(0, _Lx, rho_ghost.shape[0])
        # Raised-cosine ramp: 0 at wall, 1 after TAPER_W, symmetric at both ends
        t_left = np.clip(x / (TAPER_W * _Lx), 0, 1)
        t_right = np.clip((_Lx - x) / (TAPER_W * _Lx), 0, 1)
        taper = 0.5 * (1 - np.cos(np.pi * t_left)) * 0.5 * (1 - np.cos(np.pi * t_right))
        # Rescale so the flat region equals 1 (not 0.25)
        taper /= taper.max()
        return rho_ghost * U_WALL * taper[:, np.newaxis]

    problem.set_bc_function('jx', 'N', jx_lid)
    problem.run()

    Nx = problem.grid['Nx']
    Ny = problem.grid['Ny']
    dx = problem.grid['dx']
    dy = problem.grid['dy']
    Lx = problem.grid['Lx']
    Ly = problem.grid['Ly']

    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    jy = problem.q[2][1:-1, 1:-1]

    vx = jx / rho
    vy = jy / rho

    v_mag = np.sqrt(vx**2 + vy**2)

    P0 = problem.prop['P0']
    rho0 = problem.prop['rho0']
    alpha = problem.prop['alpha']
    pressure = P0 + (rho - rho0) / alpha

    print(f'rho range : [{rho.min():.6f}, {rho.max():.6f}]')
    print(f'vx  range : [{vx.min():.6f}, {vx.max():.6f}]')
    print(f'vy  range : [{vy.min():.6f}, {vy.max():.6f}]')

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # -------------------------------------------------------------------------
    # Plot: 2x3 layout matching the notebook
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8), facecolor='white', constrained_layout=True)

    # 1. Pressure + streamlines
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.pcolormesh(X, Y, pressure, cmap='RdBu_r', shading='auto')
    n_start = 5
    start_pts = np.array([[sx, sy]
                           for sx in np.linspace(0.2, 0.8, n_start)
                           for sy in np.linspace(0.2, 0.8, n_start)])
    ax1.streamplot(x, y, vx.T, vy.T, color='k', linewidth=0.6, density=0.6,
                   arrowsize=0.8, start_points=start_pts)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pressure + Streamlines')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='P', shrink=0.7)

    # 2. Velocity magnitude + normalised quiver
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.pcolormesh(X, Y, v_mag, cmap='viridis', shading='auto')
    skip = 3
    v_mag_sub = v_mag[::skip, ::skip]
    vx_norm = np.where(v_mag_sub > 0, vx[::skip, ::skip] / v_mag_sub, 0)
    vy_norm = np.where(v_mag_sub > 0, vy[::skip, ::skip] / v_mag_sub, 0)
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], vx_norm, vy_norm,
               color='white', alpha=0.9, scale=25, width=0.003)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity Magnitude + Arrows')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='|v|', shrink=0.7)

    # 3. Density
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.pcolormesh(X, Y, rho, cmap='plasma', shading='auto')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Density')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label=r'$\rho$', shrink=0.7)

    # 4. x-momentum jx
    ax4 = fig.add_subplot(2, 3, 4)
    vmax4 = np.abs(jx).max()
    im4 = ax4.pcolormesh(X, Y, jx, cmap='RdBu_r', shading='auto',
                         vmin=-vmax4, vmax=vmax4)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(r'x-Momentum $j_x$')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label=r'$j_x$', shrink=0.7)

    # 5. y-momentum jy
    ax5 = fig.add_subplot(2, 3, 5)
    vmax5 = np.abs(jy).max()
    im5 = ax5.pcolormesh(X, Y, jy, cmap='RdBu_r', shading='auto',
                         vmin=-vmax5, vmax=vmax5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title(r'y-Momentum $j_y$')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5, label=r'$j_y$', shrink=0.7)

    # 6. Vorticity
    ax6 = fig.add_subplot(2, 3, 6)
    dvx_dy = np.gradient(vx, dy, axis=1)
    dvy_dx = np.gradient(vy, dx, axis=0)
    vorticity = dvx_dy - dvy_dx
    vmax6 = np.abs(vorticity).max()
    im6 = ax6.pcolormesh(X, Y, vorticity, cmap='RdBu_r', shading='auto',
                         vmin=-vmax6, vmax=vmax6)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title(r'Vorticity $\omega$')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6, label=r'$\omega$', shrink=0.7)

    fig.suptitle(f'Lid-Driven Cavity Flow  Nx={Nx} Ny={Ny}  U_wall={U_WALL} m/s',
                 fontsize=14, fontweight='bold')

    out = Path(__file__).parent / 'test_lid_cavity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
    plt.show()
