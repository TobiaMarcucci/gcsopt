import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Triangular mesh for 2d robot link.
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

# Problem parameters.
circle_cost = 0 # Fixed cost of using a circle.
num_circles = 5 # Maximum number of circles.

# Initialize empty graph.
graph = GraphOfConvexSets()

# Compute bounding box for entire mesh.
l = np.min(np.vstack(mesh), axis=0)
u = np.max(np.vstack(mesh), axis=0)

# Compute minimum radius.
min_radius = np.inf
r = cp.Variable()
c = cp.Variable(2)
for points in mesh:
    constraints = [cp.norm2(p - c) <= r for p in points]
    prob = cp.Problem(cp.Minimize(r), constraints)
    prob.solve()
    min_radius = min(min_radius, r.value)

# Compute maximum radius.
constraints = []
for points in mesh:
    constraints += [cp.norm2(p - c) <= r for p in points]
    prob = cp.Problem(cp.Minimize(r), constraints)
    prob.solve()
    max_radius = r.value

# Add all circles (facilities).
circles = []
for i in range(num_circles):
    circle = graph.add_vertex(f"c{i}")
    center = circle.add_variable(2)
    radius = circle.add_variable(1)
    circle.add_constraints([center >= l, center <= u])
    circle.add_constraints([radius >= min_radius, radius <= max_radius])
    circle.add_cost(circle_cost + np.pi * radius ** 2)
    circles.append(circle)

# Add all triangles (clients).
triangles = []
for i in range(len(mesh)):
    triangle = graph.add_vertex(f"t{i}")
    slack = triangle.add_variable(1) # Necessary to add at least one variable.
    triangle.add_constraint(slack == 0)
    triangles.append(triangle)

# Add edge from every circle to every triangle.
for circle in circles:
    center, radius = circle.variables
    for points, triangle in zip(mesh, triangles):
        edge = graph.add_edge(circle, triangle)
        for point in points:
            edge.add_constraint(cp.norm2(point - center) <= radius)

# Solve problem.
plot_bounds = False
if plot_bounds:
    import importlib.util
    assert importlib.util.find_spec("gurobipy")
    from gcsopt.gurobipy.graph_problems.facility_location import facility_location
    from gcsopt.gurobipy.plot_utils import plot_optimal_value_bounds
    params = {"OutputFlag": 1, "QCPDual": 1}
    plot_bounds = True
    facility_location(graph, gurobi_parameters=params, save_bounds=plot_bounds)
    plot_optimal_value_bounds(graph.solver_stats.callback_bounds, "cover_bounds")
else:
    graph.solve_facility_location(verbose=True, solver="GUROBI")
print('Problem status:', graph.status)
print('Optimal value:', graph.value)

# Plot solution.
plt.figure()
plt.grid()
plt.gca().set_axisbelow(True) 
plt.axis("square")
limits = np.array([l - 1, u + 1])
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
plt.savefig("cover.pdf", bbox_inches="tight")
plt.show()