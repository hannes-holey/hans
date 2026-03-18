import GaPFlow
import matplotlib.pyplot as plt


problem = GaPFlow.Problem.from_yaml('circular.yaml')

# problem.plot_topo()
# plt.show()


problem.run()

rhv_history = problem.topo.rhv_history
plt.plot(rhv_history)
plt.show()
