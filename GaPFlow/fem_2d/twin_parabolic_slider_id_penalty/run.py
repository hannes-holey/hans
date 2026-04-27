"""
Twin identical parabolic slider — penalty cavitation with Dowson-Higginson EOS.

Run generate_topography.py first to create the height field file:
    python generate_topography.py

Then run this script:
    python run.py
"""

import os
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('twin_parabolic_slider_id_penalty.yaml')
problem.run()
