"""Quasi-1D parabolic slider bearing test for the Taylor-Hood P2P1 FEM solver.

Runs the parabolic slider problem (Dirichlet rho in x, periodic in y)
with the fem_2d solver and plots the steady-state density and mass flux.

Config: configs/parabolic_slider.yaml

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/fem_2d/tests/test_parabolic_slider.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from GaPFlow.problem import Problem

CONFIG_DIR = Path(__file__).parent / 'configs'

# =============================================================================
# Config — edit these
# =============================================================================

Nx = 200
Ny = 2
max_it = 10
dt = 0.1

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    template = (CONFIG_DIR / 'parabolic_slider.yaml').read_text()
    config = template.format(
        Nx=Nx, Ny=Ny, dt=dt, max_it=max_it,
        solver='fem', adaptive=0, CFL=0.5,
        rho_bc=850.0,
    )

    print(f'Parabolic slider: Nx={Nx}, Ny={Ny}, dt={dt}, max_it={max_it}')
    problem = Problem.from_string(config)
    problem.run()

    Lx = problem.grid['Lx']
    Ly = problem.grid['Ly']
    dx = problem.grid['dx']
    dy = problem.grid['dy']
    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)

    rho = problem.q[0][1:-1, 1:-1].copy()
    jx = problem.q[1][1:-1, 1:-1].copy()

    print(f'rho range: [{rho.min():.4f}, {rho.max():.4f}]')
    print(f'jx  range: [{jx.min():.6f}, {jx.max():.6f}]')

    X, Y = np.meshgrid(x, y, indexing='ij')

    fig = plt.figure(figsize=(12, 5))

    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.plot_surface(X, Y, rho, cmap='viridis')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('rho')
    ax0.set_title('Density')

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(X, Y, jx, cmap='plasma')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('jx')
    ax1.set_title('Mass flux x')

    plt.suptitle(f'Parabolic Slider (fem_2d): Nx={Nx}, Ny={Ny}')
    plt.savefig('test_parabolic_slider.png', dpi=150, bbox_inches='tight')
    print('Saved: test_parabolic_slider.png')
    plt.show()
