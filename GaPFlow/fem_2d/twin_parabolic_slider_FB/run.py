"""
Twin parabolic slider — FB cavitation (DH EOS).

Run generate_topography.py first:
    python generate_topography.py

Then run:
    python run.py
"""

import os
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('twin_parabolic_slider_FB.yaml')
problem.run()
