"""
2D Parabolic slider bearing — FB cavitation (DH EOS).

Run:
    python run.py
"""

import os
import sys
import GaPFlow
from plot_newton_residual import plot_newton_residual

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('parabolic_2D_FB.yaml')
problem.run()

plot_newton_residual(problem.solver, problem.outdir)
