"""
This script runs the comparison between the proposed MICP, the MINCP, and the
McCormick formulation for the helicopter-flight problem. It saves the results in
.npy files that can then be used to reproduce the plot in the paper.
WARNING: Running this file takes a few hours. To reproduce a subset of the
results more quickly, consider reducing the problem time_limit.
"""

# This comparison runs only if gurobipy installed.
from gcsopt.gurobipy.utils import has_gurobi
assert has_gurobi()

import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets
from gcsopt.gurobipy.graph_problems.shortest_path import shortest_path
from shortest_path_spatial_bb import shortest_path_spatial_bb
from shortest_path_mccormick import shortest_path_mccormick

# Comparison parameters.
time_limit = 1000
min_islands = 30
max_islands = 300
assert max_islands % min_islands == 0

# Problem data.
min_radius = .02
max_radius = .1
speed = 1
discharge_rate = 5
charge_rate = 1

# Function that creates a random GCS.
def create_graph(num_islands, u):
    np.random.seed(0)

    # Generate random islands that do not intersect.
    centers = np.full((num_islands, 2), np.inf) # inf ensures no intersection with sampled islands.
    radii = np.zeros(num_islands)
    l = np.array([0, 0])
    i = 0
    while i < num_islands:
        center = np.random.uniform(l, u)
        radius = np.random.uniform(min_radius, max_radius)

        # Discard island if it intersects with a previous ones.
        if all(np.linalg.norm(center - centers, axis=1) > radius + radii):
            centers[i] = center
            radii[i] = radius
            i += 1

    # Select start and goal islands.
    start = np.argmin(np.linalg.norm(centers - l, axis=1))
    goal = np.argmin(np.linalg.norm(centers - u, axis=1))

    # Initialize empty graph.
    graph = GraphOfConvexSets()

    # One vertex for every island, including start and goal.
    L = np.zeros((num_islands, 4))
    U = np.ones((num_islands, 4))
    for i, (center, radius) in enumerate(zip(centers, radii)):
        vertex = graph.add_vertex(i)
        q = vertex.add_variable(2) # Helicopter landing position on the island.
        z = vertex.add_variable(2) # Batter level at landing and take off.
        t = (z[1] - z[0]) / charge_rate # Recharge time.
        vertex.add_cost(t)
        vertex.add_constraints([
            cp.norm2(q - center) <= radius,
            z >= 0, z <= 1, t >= 0])
        # The following are hand tuned.
        L[i, 2:] = center - radius
        U[i, 2:] = center + radius
        
        # Battery is fully charged at the beginning.
        if i == start:
            vertex.add_constraint(z[0] == 1)

    # Helper function that check if two islands should be connected.
    max_range = speed / discharge_rate
    def connect(i, j):
        center_dist = np.linalg.norm(centers[i] - centers[j])
        island_dist = center_dist - radii[i] - radii[j]
        return i != j and island_dist < max_range

    # Edges between pairs of islands that are close enough.
    for i, vertex_i in enumerate(graph.vertices):
        qi, zi = vertex_i.variables
        for j, vertex_j in enumerate(graph.vertices):
            if connect(i, j):
                qj, zj = vertex_j.variables
                tij = (zi[1] - zj[0]) / discharge_rate # Flight time.
                edge = graph.add_edge(vertex_i, vertex_j)
                edge.add_cost(tij)
                edge.add_constraint(tij >= cp.norm2(qi - qj) / speed)
                    
    # Source and target vertices.
    source = graph.vertices[start]
    target = graph.vertices[goal]

    return graph, source, target, L, U

# Generate and solve random instances.
n_islands = np.arange(min_islands, max_islands + 1, min_islands)
u_max = [1/60, 2]
times = np.full((3, len(n_islands)), np.nan)
values = np.full((3, len(n_islands)), np.nan)
parameters = {"OutputFlag": 0, "TimeLimit": time_limit}
micp_parameters = parameters | {"PreMIQCPForm": 1}

# Iterate over problem size.
for i, n in enumerate(n_islands):
    print("Num. islands:", n)

    # Generate graph.
    u = [u_max[0] * n, u_max[1]]
    graph, source, target, L, U = create_graph(n, u)

    # MICP.
    shortest_path(graph, source, target, gurobi_parameters=micp_parameters)
    times[0, i] = graph.solver_stats.solve_time
    values[0, i] = graph.value
    print("MICP:", times[0, i], values[0, i])

    # MINCP.
    shortest_path_spatial_bb(graph, source, target, gurobi_parameters=parameters)
    times[1, i] = graph.solver_stats.solve_time
    values[1, i] = graph.value
    print("MINCP:", times[1, i], values[1, i])

    # McCormick.
    shortest_path_mccormick(graph, source, target, L, U, gurobi_parameters=parameters)
    times[2, i] = graph.solver_stats.solve_time
    values[2, i] = graph.value
    print("McCormick:", times[2, i], values[2, i])

np.save("islands.npy", n_islands)
np.save("times.npy", times)
np.save("values.npy", values)