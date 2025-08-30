import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Problem data.
num_islands = 100
min_radius = .02
max_radius = .1
l = np.array([0, 0])
u = np.array([5, 1])
speed = 1
discharge_rate = 5
charge_rate = 1
max_range = speed / discharge_rate

# Generate random islands that do not intersect.
np.random.seed(0)
centers = np.full((num_islands, 2), np.inf) # inf ensures no intersection with sampled islands.
radii = np.zeros(num_islands)
sampled_islands = 0
while sampled_islands < num_islands:
    center = np.random.uniform(l, u)
    radius = np.random.uniform(min_radius, max_radius)

    # Discard island if it intersects with a previous ones.
    if all(np.linalg.norm(center - centers, axis=1) > radius + radii):
        centers[sampled_islands] = center
        radii[sampled_islands] = radius
        sampled_islands += 1

# Select start and goal islands.
start = np.argmin(np.linalg.norm(centers - l, axis=1))
goal = np.argmin(np.linalg.norm(centers - u, axis=1))

# Initialize empty graph.
graph = GraphOfConvexSets()

# One vertex for every island, including start and goal.
for i, (center, radius) in enumerate(zip(centers, radii)):
    vertex = graph.add_vertex(i)
    q = vertex.add_variable(2) # Helicopted landing position on the island.
    z = vertex.add_variable(2) # Batter level at landing and take off.
    t = vertex.add_variable(1) # Recharge time.
    vertex.add_cost(t)
    vertex.add_constraints([
        cp.norm2(q - center) <= radius,
        z >= 0, z <= 1,
        t >= 0, t <= 1 / charge_rate,
        z[1] == z[0] + charge_rate * t])
    
    # Battery is fully charged at the beginning.
    if i == start:
        vertex.add_constraint(z[0] == 1)

# Edges between pairs of islands that are close enough.
for i, (center_i, radius_i) in enumerate(zip(centers, radii)):
    vertex_i = graph.get_vertex(i)
    qi, zi = vertex_i.variables[:2]
    for j, (center_j, radius_j) in enumerate(zip(centers, radii)):
        if i != j:
            center_dist = np.linalg.norm(center_i - center_j)
            island_dist = center_dist - radius_i - radius_j
            if island_dist < max_range: # Necessary condition for flight feasibility.
                vertex_j = graph.get_vertex(j)
                qj, zj = vertex_j.variables[:2]
                edge = graph.add_edge(vertex_i, vertex_j)
                t = edge.add_variable(1) # Flight time between islands i and j.
                edge.add_cost(t)
                edge.add_constraints([
                    t >= cp.norm2(qi - qj) / speed,
                    zi[1] == zj[0] + discharge_rate * t])
                
# Solve shortest path problem from start to goal points.
source = graph.vertices[start]
target = graph.vertices[goal]
plot_bounds = True
if plot_bounds:
    import importlib.util
    assert importlib.util.find_spec("gurobipy")
    from gcsopt.gurobipy.graph_problems.shortest_path import shortest_path
    from gcsopt.gurobipy.plot_utils import plot_optimal_value_bounds
    params = {"OutputFlag": 0}
    plot_bounds = True
    shortest_path(graph, source, target, binary=False, gurobi_parameters=params, save_bounds=plot_bounds)
    plot_optimal_value_bounds(graph.solver_stats.callback_bounds, "flight_bounds")
else:
    graph.solve_shortest_path(source, target, verbose=True, solver="GUROBI")
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print("Solver time", graph.solver_stats.solve_time)

# Plot optimal flight trajectory.
plt.figure(figsize=(10, 2))
graph.plot_2d_solution()

# Plot ocean.
l_plot = l - max_radius
d_plot = (u - l) + 2 * max_radius
u_plot = l_plot + d_plot
ocean = plt.Rectangle(l_plot, *d_plot, fc="azure")
plt.gca().add_patch(ocean)

# Plot islands.
for i in range(num_islands):
    island = plt.Circle(centers[i], radii[i], ec="k", fc="lightgreen")
    plt.gca().add_patch(island)

# Plot settings.
plt.gca().set_aspect("equal")
limits = np.array([l_plot, u_plot])
plt.xlim(limits[:, 0])
plt.ylim(limits[:, 1])
plt.savefig("flight.pdf", bbox_inches="tight")
plt.show()

# Reconstruct battery level as a function of time.
battery_levels = []
times = [0]
vertex = source
while vertex != target:
    z, t = vertex.variables[1:]
    battery_levels.extend(z.value)
    times.extend(times[-1] + t.value)
    for edge in graph.outgoing_edges(vertex):
        if np.isclose(edge.binary_variable.value, 1):
            t = edge.variables[0]
            times.extend(times[-1] + t.value)
            vertex = edge.head
            break
battery_levels.append(target.variables[1].value[0])

# Plot battery level.
plt.figure(figsize=(10, 1.5))
end_times = (times[0], times[-1])
plt.plot(end_times, (0, 0), "r--") # Minimum level.
plt.plot(end_times, (1, 1), "g--") # Maximum level.
plt.plot(times, battery_levels)
plt.xlabel("Time")
plt.ylabel("Battery level")
plt.xlim(end_times)
plt.grid()
plt.savefig("battery.pdf", bbox_inches="tight")
plt.show()
