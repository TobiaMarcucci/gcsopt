import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Data.
side = 3
radius = .3

# Vertices.
def add_vertices(graph):
    for i in range(side):
        for j in range(side):
            v = graph.add_vertex((i, j))
            x = v.add_variable(2)
            c = np.array([i, j])
            v.add_constraint(cp.norm2(x - c) <= radius)

# Edges.
def add_edges(graph):
    for i in range(side):
        for j in range(side):
            for k in range(i, side):
                for l in range(j, side):
                    dx = abs(i - k)
                    dy = abs(j - l)
                    if graph.directed:
                        reachable = dx + dy <= 1
                    else:
                        reachable = dx <= 1 and dy <= 1
                    if reachable and dx + dy != 0:
                        tail = graph.get_vertex((i, j))
                        head = graph.get_vertex((k, l))
                        edge = graph.add_edge(tail, head)
                        edge.add_cost(cp.norm2(head.variables[0] - tail.variables[0]))

# Solve shortest path.
graph = GraphOfConvexSets(directed=True)
add_vertices(graph)
add_edges(graph)
source = graph.vertices[0]
target = graph.vertices[-1]
graph.solve_shortest_path(source, target)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# Plot optimal solution.
plt.figure(figsize=(3,3))
plt.axis("off")
graph.plot_2d()
graph.plot_2d_solution()
plt.xlim([-.5, side-.5])
plt.ylim([-.5, side-.5])
plt.savefig("demo_spp.pdf", bbox_inches="tight")
plt.show()

# Solve traveling salesman.
graph = GraphOfConvexSets(directed=False)
add_vertices(graph)
add_edges(graph)
graph.solve_traveling_salesman()
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# Plot optimal solution.
plt.figure(figsize=(3,3))
plt.axis("off")
graph.plot_2d()
graph.plot_2d_solution()
plt.xlim([-.5, side-.5])
plt.ylim([-.5, side-.5])
plt.savefig("demo_tsp.pdf", bbox_inches="tight")
plt.show()

# Solve traveling salesman.
graph.solve_minimum_spanning_tree()
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# Plot optimal solution.
plt.figure(figsize=(3,3))
plt.axis("off")
graph.plot_2d()
graph.plot_2d_solution()
plt.xlim([-.5, side-.5])
plt.ylim([-.5, side-.5])
plt.savefig("demo_mstp.pdf", bbox_inches="tight")
plt.show()