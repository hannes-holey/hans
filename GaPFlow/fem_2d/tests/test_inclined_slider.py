"""Quasi-1D inclined slider bearing test for the Taylor-Hood P2P1 FEM solver.

Runs the inclined slider problem (Dirichlet rho in x, periodic in y)
with the fem_2d solver and plots the steady-state density and mass flux.

Config: configs/inclined_slider.yaml

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/fem_2d/tests/test_inclined_slider.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from GaPFlow.problem import Problem

CONFIG_DIR = Path(__file__).parent / 'configs'

# =============================================================================
# Config — edit these
# =============================================================================

Nx = 100
Ny = 3
max_it = 30
dt = 0.1

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    template = (CONFIG_DIR / 'inclined_slider.yaml').read_text()
    config = template.format(
        Nx=Nx, Ny=Ny, dt=dt, max_it=max_it,
        solver='fem', adaptive=0, CFL=0.5,
    )

    print(f'Inclined slider: Nx={Nx}, Ny={Ny}, dt={dt}, max_it={max_it}')
    problem = Problem.from_string(config)
    problem.run()

    Lx = problem.grid['Lx']
    dx = problem.grid['dx']
    x = np.linspace(dx / 2, Lx - dx / 2, Nx)

    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()

    print(f'rho range: [{rho.min():.4f}, {rho.max():.4f}]')
    print(f'jx  range: [{jx.min():.6f}, {jx.max():.6f}]')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, rho, 'b-')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('rho')
    axes[0].set_title('Density')

    axes[1].plot(x, jx, 'b-')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('jx')
    axes[1].set_title('Mass flux x')

    plt.suptitle(f'Inclined Slider (fem_2d): Nx={Nx}, Ny={Ny}')
    plt.savefig('test_inclined_slider.png', dpi=150, bbox_inches='tight')
    print('Saved: test_inclined_slider.png')
    plt.show()
