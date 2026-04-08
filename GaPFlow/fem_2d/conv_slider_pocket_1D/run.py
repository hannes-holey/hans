"""
1D Convergent slider with pocket — Bertocchi et al. (2013) benchmark.

Run generate_topography.py first:
    python generate_topography.py

Then run the simulation:
    python run.py
"""

import os
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('conv_slider_pocket_1D.yaml')
problem.run()
