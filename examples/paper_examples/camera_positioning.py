import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets

# This example runs only if gurobipy installed. Since cvxpy does not allow lazy
# constraints, and this problem is too large for the exponential formulation of
# the MSTP.
from gcsopt.gurobipy.utils import has_gurobi
assert has_gurobi()
from gcsopt.gurobipy.graph_problems.minimum_spanning_tree import minimum_spanning_tree

# Problem data.
grid_width = 60
grid_height = 15
min_room_side = 2 / 3
max_room_side = 1
main_room = 0

# Flags.
binary = True # True for solving MICP and False for convex relaxation.
plot_bounds = False # Plot branch and bound progress (only if gurobipy is available and MICP is solved).

# Generate random rooms positioned on a grid.
np.random.seed(0)
L = [] # Room lower-left corners.
U = [] # Room lpper-right corners.
for i in range(grid_width):
    for j in range(grid_height):
        nom_room_sides = [2, 1] if (i + j) % 2 == 0 else [1, 2]
        rand_room_sides = np.random.uniform(min_room_side, max_room_side, 2)
        diag = np.multiply(rand_room_sides, nom_room_sides) / 2
        center = np.array([i, j])
        L.append(center - diag)
        U.append(center + diag)

# Initialize empty graph.
graph = GraphOfConvexSets()

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

# Solve problem with gurobipy.
root = graph.vertices[main_room]
parameters = {"OutputFlag": 0}
minimum_spanning_tree(graph, root, binary=binary, gurobi_parameters=parameters,
                      save_bounds=plot_bounds)

# Print optimal solution stats.
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print("Solver time:", graph.solver_stats.solve_time)

# Plot result only if MICP is solved optimally.
if graph.status == "optimal" and binary:

    # Plot rooms and optimal spanning tree.
    plt.figure(figsize=(grid_width/2, grid_height/2))
    plt.axis("off")
    for l, u in zip(L, U):
        rect = patches.Rectangle(l, *(u - l),
            edgecolor="k", facecolor="mintcream", alpha=.5)
        plt.gca().add_patch(rect)
    graph.plot_2d_solution()
    plt.xlim([-1, grid_width])
    plt.ylim([-1, grid_height])
    plt.show()

    # Plot branch and bound progress.
    if has_gurobi() and plot_bounds:
        from gcsopt.plot_utils import plot_bb_progress
        plot_bb_progress(graph.solver_stats.callback_bounds)