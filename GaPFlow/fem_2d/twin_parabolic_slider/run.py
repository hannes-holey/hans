"""
Twin parabolic slider bearing — Bayada cavitation test case.

Run generate_topography.py first to create the height field file:
    python generate_topography.py

Then run this script:
    python run.py
"""

import os
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('twin_parabolic_slider.yaml')
problem.run()
