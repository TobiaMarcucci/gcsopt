import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Problem parameters.
num_circles = 5 # Maximum number of circles.
mesh = np.array([
    [[1, 0], [0, 2], [0, 1]],
    [[1, 0], [0, 2], [1, 3]],
    [[1, 0], [2, 0], [1, 3]],
    [[3, 1], [2, 0], [1, 3]],
    [[3, 1], [3, 3], [1, 3]],
    [[3, 1], [3, 3], [5, 3]],
    [[3, 1], [5, 3], [5, 1]],
    [[7, 1], [5, 3], [5, 1]],
    [[7, 3], [5, 3], [7, 1]],
    [[7, 3], [9, 3], [7, 1]],
    [[9, 3], [9, 1], [7, 1]],
    [[13, 1], [9, 3], [9, 1]],
    [[13, 1], [9, 3], [14, 3]],
    [[10, 5], [9, 3], [14, 3]],
    [[10, 5], [14, 5], [14, 3]],
    [[10, 5], [14, 5], [11, 6]],
    [[13, 6], [14, 5], [11, 6]],
])

# Flags.
binary = True # True for solving MICP and False for convex relaxation.
plot_bounds = True # Plot branch and bound progress (only if gurobipy is available and MICP is solved).

# Initialize empty graph.
graph = GraphOfConvexSets()

# Compute bounding box for entire mesh.
l = np.min(np.vstack(mesh), axis=0)
u = np.max(np.vstack(mesh), axis=0)

# Compute minimum radii.
min_radius = np.inf
radius = cp.Variable()
center = cp.Variable(2)
for points in mesh:
    constraints = [cp.norm2(point - center) <= radius for point in points]
    prob = cp.Problem(cp.Minimize(radius), constraints)
    prob.solve()
    min_radius = min(min_radius, radius.value)

# Add all circles (facilities).
circles = []
for i in range(num_circles):
    circle = graph.add_vertex(f"c{i}")
    center = circle.add_variable(2)
    radius = circle.add_variable(1)
    circle.add_constraints([center >= l, center <= u, radius >= min_radius])
    circle.add_cost(np.pi * radius ** 2)
    circles.append(circle)

# Add all triangles (clients).
triangles = []
for i, points in enumerate(mesh):
    triangle = graph.add_vertex(f"t{i}")
    dummy = triangle.add_variable(1)
    triangle.add_constraint(dummy == 0)
    triangles.append(triangle)

# Add edge from every circle to every triangle.
for circle in circles:
    center, radius = circle.variables
    for triangle, points in zip(triangles, mesh):
        edge = graph.add_edge(circle, triangle)
        for point in points:
            edge.add_constraint(cp.norm2(point - center) <= radius)

# Solve problem using gurobipy if available.
from gcsopt.gurobipy.utils import has_gurobi
if has_gurobi():
    parameters = {"OutputFlag": 0}
    from gcsopt.gurobipy.graph_problems.facility_location import facility_location
    facility_location(graph, binary=binary, gurobi_parameters=parameters,
                      save_bounds=plot_bounds)
    
# Solve problem using cvxpy default solver.
else:
    graph.solve_facility_location(binary=binary)

# Print optimal solution stats.
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print("Solver time:", graph.solver_stats.solve_time)

# Plot result only if MICP is solved optimally.
if graph.status == "optimal" and binary:

    # Plot solution.
    plt.figure()
    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.axis("square")
    plt.xlim([l[0] - 1, u[0] + 1])
    plt.ylim([l[1] - 1, u[1] + 1])
    plt.xticks(range(l[0] - 1, u[0] + 2))
    plt.yticks(range(l[1] - 1, u[1] + 2))

    # Plot mesh.
    for triangle in mesh:
        patch = plt.Polygon(triangle[:3], fc="mintcream", ec="k")
        plt.gca().add_patch(patch)

    # Plot circle cover.
    for circle in circles:
        if np.isclose(circle.binary_variable.value, 1):
            center, radius = circle.variables
            patch = plt.Circle(center.value, radius.value, fc="None", ec="b")
            plt.gca().add_patch(patch)
    plt.show()

    # Plot branch and bound progress.
    if has_gurobi() and plot_bounds:
        from gcsopt.plot_utils import plot_bb_progress
        plot_bb_progress(graph.solver_stats.callback_bounds)