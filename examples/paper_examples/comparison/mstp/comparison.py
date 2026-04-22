"""
This script runs the comparison between the proposed MICP, the MINCP, and the
McCormick formulation for the camera-positioning problem. It saves the results
in .npy files that can then be used to reproduce the plot in the paper.
WARNING: Running this file takes a few hours. To reproduce a subset of the
results more quickly, consider reducing the problem time_limit.
"""

# This comparison runs only if gurobipy installed.
from gcsopt.gurobipy.utils import has_gurobi
assert has_gurobi()

import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets
from gcsopt.gurobipy.graph_problems.minimum_spanning_tree import minimum_spanning_tree
from minimum_spanning_tree_spatial_bb import minimum_spanning_tree_spatial_bb
from minimum_spanning_tree_mccormick import minimum_spanning_tree_mccormick

# Comparison parameters.
time_limit = 1000
min_width = 5
max_width = 60
assert max_width % min_width == 0

# Problem data.
grid_height = 15
min_room_side = 2 / 3
max_room_side = 1
main_room = 0

# Function that creates a random GCS.
def create_graph(grid_width):
    np.random.seed(0)

    # Generate random rooms.
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
    main_room = 0
    def connect(i, j):
        return i != j and np.all(L[i] <= U[j]) and np.all(L[j] <= U[i])
    for i, tail in enumerate(graph.vertices):
        li, ui = L[i], U[i]
        for j, head in enumerate(graph.vertices):
            if connect(i, j) and j != main_room:
                edge = graph.add_edge(tail, head)
                x_head = head.variables[0]
                edge.add_constraints([x_head >= li, x_head <= ui])

    return graph, L, U

# Generate and solve random instances.
widths = np.arange(min_width, max_width + 1, min_width)
times = np.zeros((3, len(widths)))
values = np.zeros((3, len(widths)))
parameters = {"OutputFlag": 0, "TimeLimit": time_limit}

# Iterate over problem size.
for i, width in enumerate(widths):
    print("Width:", width)

    # Generate graph.
    graph, L, U = create_graph(width)
    root = graph.vertices[0]

    # MICP.
    minimum_spanning_tree(graph, root, gurobi_parameters=parameters)
    times[0, i] = graph.solver_stats.solve_time
    values[0, i] = graph.value
    print("MICP:", times[0, i], values[0, i])

    # MINCP.
    minimum_spanning_tree_spatial_bb(graph, root, gurobi_parameters=parameters)
    times[1, i] = graph.solver_stats.solve_time
    values[1, i] = graph.value
    print("MINCP:", times[1, i], values[1, i])

    # McCormick.
    minimum_spanning_tree_mccormick(graph, root, L, U, gurobi_parameters=parameters)
    times[2, i] = graph.solver_stats.solve_time
    values[2, i] = graph.value
    print("McCormick:", times[2, i], values[2, i])

np.save("rooms.npy", widths * 15)
np.save("times.npy", times)
np.save("values.npy", values)
