import os
import GaPFlow
import matplotlib.pyplot as plt
from plot_newton_residual import plot_newton_residual

os.chdir(os.path.dirname(os.path.abspath(__file__)))

problem = GaPFlow.Problem.from_yaml('circular_FB.yaml')
problem.run()

plot_newton_residual(problem.solver, problem.outdir)

rhv_history = problem.topo.rhv_history
plt.plot(rhv_history)
plt.xlabel('timestep')
plt.ylabel('h0 [m]')
plt.title('Rigid height variation history')
plt.savefig(os.path.join(problem.outdir, 'rhv_history.png'), facecolor='white')
plt.show()
