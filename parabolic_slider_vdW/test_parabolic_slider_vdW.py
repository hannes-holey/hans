import os
import GaPFlow
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(__file__))

problem = GaPFlow.Problem.from_yaml('parabolic_slider_vdW.yaml')
problem.run()
