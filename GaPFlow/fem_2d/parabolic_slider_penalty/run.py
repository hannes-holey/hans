"""
Parabolic slider bearing — penalty cavitation with Dowson-Higginson EOS.

Run this script from its directory:
    python run.py
"""

import os
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('parabolic_slider_penalty.yaml')
problem.run()
