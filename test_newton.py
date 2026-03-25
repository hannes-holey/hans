"""Quick Newton convergence test for Poiseuille P2P1."""
import sys
sys.path.insert(0, '/home/qd5728/fem_taylor_hood/GaPFlow')

from GaPFlow.problem import Problem
from GaPFlow.fem_2d.solver import TaylorHoodFEMSolver

config = '/home/qd5728/fem_taylor_hood/GaPFlow/GaPFlow/fem_2d/tests/configs/poiseuille_p2p1.yaml'

print("Loading problem...")
problem = Problem.from_yaml(config)

print("Setting up problem...")
problem._pre_run()
solver = problem.solver

print("Running one Newton solve...")
solver.update_dynamic()

history = solver.R_norm_history[-1]
print(f"\nNewton residual history ({len(history)} iterations):")
for i, r in enumerate(history):
    print(f"  it {i:2d}: {r:.3e}")

if history[-1] < 1e-8:
    print("\nNewton CONVERGED!")
else:
    print(f"\nNewton did NOT converge (final R={history[-1]:.3e})")
