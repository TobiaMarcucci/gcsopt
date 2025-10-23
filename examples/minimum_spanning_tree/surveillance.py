import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets

# This file runs only with gurobipy installed. The deafault MSTP cannot solve this.
from gcsopt.gurobipy.utils import has_gurobi
assert has_gurobi()
from gcsopt.gurobipy.graph_problems.minimum_spanning_tree import minimum_spanning_tree

# Generate random rooms.
np.random.seed(0)
sides = np.array([60, 15])
L = np.zeros((*sides, 2))
U = np.zeros((*sides, 2))
low = 2 / 3
high = 1
for i in range(sides[0]):
    for j in range(sides[1]):
        box_sides = [2, 1] if (i + j) % 2 == 0 else [1, 2]
        d = np.multiply(np.random.uniform(low, high, 2), box_sides) / 2
        c = (i, j)
        L[i, j] = c - d
        U[i, j] = c + d
L = np.vstack(L)
U = np.vstack(U)

# Rooms.
main_room = 0

# Initialize empty graph.
graph = GraphOfConvexSets() # Directed by default.

# Add vertices.
for i, (l, u) in enumerate(zip(L, U)):
    v = graph.add_vertex(i)
    x = v.add_variable(2)
    v.add_constraints([x >= l, x <= u])
    D = np.diag(.2 / (u - l))
    c = (l + u) / 2
    v.add_cost(cp.norm_inf(D @ (x - c)))

# Add edges.
def connect(i, j):
    return i != j and np.all(L[i] <= U[j]) and np.all(L[j] <= U[i])
for i, tail in enumerate(graph.vertices):
    li, ui = L[i], U[i]
    for j, head in enumerate(graph.vertices):
        if connect(i, j) and j != main_room:
            edge = graph.add_edge(tail, head)
            x_head = head.variables[0]
            edge.add_constraints([x_head >= li, x_head <= ui])

# Solve problem with gurobipy (way too big for deafault MSTP method).
root = graph.vertices[main_room]
params = {"OutputFlag": 0}
save_bounds = False
minimum_spanning_tree(graph, root, gurobi_parameters=params, save_bounds=save_bounds)
if save_bounds:
    np.save("surveillance_bounds.npy", graph.solver_stats.callback_bounds)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print("Solver time:", graph.solver_stats.solve_time)

# Plot rooms and optimal spanning tree.
plt.figure(figsize=sides/2)
plt.axis("off")
for l, u in zip(L, U):
    rect = patches.Rectangle(l, *(u - l),
        edgecolor="k", facecolor="mintcream", alpha=.5)
    plt.gca().add_patch(rect)
graph.plot_2d_solution()
plt.xlim([-1, sides[0]])
plt.ylim([-1, sides[1]])
plt.savefig("surveillance.pdf", bbox_inches="tight")
plt.show()