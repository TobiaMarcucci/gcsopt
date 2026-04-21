import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Problem data.
np.random.seed(0)
n_kids = 18
max_walk = 3
school_position = np.array([45, 7])
kid_positions = np.random.randint([30, 1], [60, 12], (n_kids, 2))

# Flags.
binary = True # True for solving MICP and False for convex relaxation.
plot_bounds = False # Plot branch and bound progress (only if gurobipy is available and MICP is solved).

# Initialize empty graph.
graph = GraphOfConvexSets(directed=False)

# Vertex for every kid.
for i, position in enumerate(kid_positions):
    kid = graph.add_vertex(i + 1)
    x = kid.add_variable(2)
    d = kid.add_variable(2) # Slack variables for L1 norm.
    kid.add_constraints([ # Implementing this using cp.norm1 is less efficient.
        d >= x - position,
        d >= position - x,
        sum(d) <= max_walk])
    kid.add_cost(sum(d))

# Vertex for the school.
school = graph.add_vertex(0)
x = school.add_variable(2)
school.add_constraint(x == school_position)

# Edge between every pair of distinct positions.
for i, tail in enumerate(graph.vertices):
    for head in graph.vertices[i + 1:]:
        edge = graph.add_edge(tail, head)
        x_tail = tail.variables[0]
        x_head = head.variables[0]
        edge.add_cost(cp.norm1(x_tail - x_head)) # L1 distance traveled by bus.

# Solve problem using gurobipy if possible, as it allows lazy constraints and is
# much faster. Otherwise use exponential formulation and default cvxpy solver.
from gcsopt.gurobipy.utils import has_gurobi
if has_gurobi():
    parameters = {"OutputFlag": 0}
    from gcsopt.gurobipy.graph_problems.traveling_salesman import traveling_salesman
    traveling_salesman(graph, binary=binary, gurobi_parameters=parameters,
                       save_bounds=plot_bounds)
    
# Solve problem using cvxpy default solver.
else:
    graph.solve_traveling_salesman(binary=binary)

# Print optimal solution stats.
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print("Solver time:", graph.solver_stats.solve_time)

# Plot result only if MICP is solved optimally.
if graph.status == "optimal" and binary:

    # Function that draws an L1 path between two points.
    def l1_path(tail, head, color, ls):
        options = dict(color=color, ls=ls, zorder=2)
        if not np.isclose(tail[0], head[0]):
            plt.plot([tail[0], head[0]], [tail[1], tail[1]], **options)
        if not np.isclose(tail[1], head[1]):
            plt.plot([head[0], head[0]], [tail[1], head[1]], **options)

    # Plot solution.
    plt.figure(figsize=(8, 3))
    plt.grid()

    # Bus path.
    for edge in graph.edges:
        if np.isclose(edge.binary_variable.value, 1):
            tail = edge.tail.variables[0].value
            head = edge.head.variables[0].value
            l1_path(tail, head, "red", "--")

    # Kid walks.
    for vertex, position in zip(graph.vertices, kid_positions):
        xv = vertex.variables[0].value
        l1_path(position, xv, "blue", "-")
        plt.scatter(*xv, c="blue", marker="x", zorder=3)
        plt.scatter(*position, fc="white", ec="blue", zorder=3)
        
    # School position.
    plt.scatter(*school_position, marker="*", fc="yellow", ec="black", zorder=3,
                s=200, label="school")

    # Empty plots for clean legend.
    nans = np.full((2, 2), np.nan)
    plt.scatter(*nans[0], fc="white", ec="blue", label="home")
    plt.scatter(*nans[0], c="blue", marker="x", label="pick-up point")
    plt.plot(*nans, c="blue", ls="-", label="kid's walk")
    plt.plot(*nans, c="red", ls="--", label="bus tour")

    # Bounding box for all positions.
    positions = np.vstack((school_position, kid_positions))
    l = np.min(positions, axis=0)
    u = np.max(positions, axis=0)

    # Additional plot settings.
    plt.xticks(range(l[0] - 1, u[0] + 2))
    plt.yticks(range(l[1] - 1, u[1] + 2))
    plt.xlim([l[0] - 1, u[0] + 1])
    plt.ylim([l[1] - 1, u[1] + 1])
    plt.xlabel('street')
    plt.ylabel('avenue')
    plt.legend()
    plt.show()

    # Plot branch and bound progress.
    if has_gurobi() and plot_bounds:
        from gcsopt.plot_utils import plot_bb_progress
        plot_bb_progress(graph.solver_stats.callback_bounds)