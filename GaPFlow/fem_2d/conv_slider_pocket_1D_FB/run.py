"""
1D Convergent slider with pocket — FB cavitation (DH EOS).

Run generate_topography.py first:
    python generate_topography.py

Then run:
    python run.py
"""

import os
import GaPFlow
from plot_newton_residual import plot_newton_residual

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('conv_slider_pocket_1D_FB.yaml')
problem.run()

plot_newton_residual(problem.solver, problem.outdir)
