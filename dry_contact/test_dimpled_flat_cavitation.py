import os
import subprocess
import sys
import GaPFlow
import matplotlib.pyplot as plt
import numpy as np

# Generate the topography file if it doesn't exist yet
topo_file = os.path.join(os.path.dirname(__file__), 'topography', 'dimpled_flat.npy')
if not os.path.exists(topo_file):
    script = os.path.join(os.path.dirname(__file__), 'generate_dimpled_topography.py')
    subprocess.run([sys.executable, script], check=True)

problem = GaPFlow.Problem.from_yaml('dimpled_flat_cavitation.yaml')

problem.run()

