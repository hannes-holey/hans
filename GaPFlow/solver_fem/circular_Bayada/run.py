import GaPFlow
import matplotlib.pyplot as plt

problem = GaPFlow.Problem.from_yaml('circular_Bayada.yaml')

problem.run()

rhv_history = problem.topo.rhv_history
plt.plot(rhv_history)
plt.xlabel('timestep')
plt.ylabel('h0 [m]')
plt.title('Rigid height variation history')
plt.show()
