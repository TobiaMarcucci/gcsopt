"""
This script runs the comparison between the proposed MICP, the MINCP, and the
McCormick formulation for the school-bus problem. It saves the results in .npy
files that can then be used to reproduce the plot in the paper.
WARNING: Running this file takes a few hours. To reproduce a subset of the
results more quickly, consider reducing the problem time_limit.
"""

import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets
from gcsopt.gurobipy.graph_problems.traveling_salesman import traveling_salesman
from traveling_salesman_spatial_bb import traveling_salesman_spatial_bb
from traveling_salesman_mccormick import traveling_salesman_mccormick

# Comparison parameters.
time_limit = 1000
min_kids = 2
max_kids = 18
assert max_kids % min_kids == 0

# Problem data.
max_walk = 3
school_position = np.array([45, 7])

# Helper function that creates a random GCS.
def create_graph(n_kids):
    np.random.seed(0)

    # Sample kid positions.
    kid_positions = np.random.randint([30, 1], [60, 12], (n_kids, 2))
    L = np.zeros((n_kids + 1, 2))
    U = np.zeros((n_kids + 1, 2))
    L[:-1] = kid_positions - max_walk
    U[:-1] = kid_positions + max_walk

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
    L[-1] = school_position
    U[-1] = school_position

    # Edge between every pair of distinct positions.
    for i, tail in enumerate(graph.vertices):
        for head in graph.vertices[i + 1:]:
            edge = graph.add_edge(tail, head)
            x_tail = tail.variables[0]
            x_head = head.variables[0]
            edge.add_cost(cp.norm1(x_tail - x_head)) # L1 distance traveled by bus.

    return graph, L, U

# Generate and solve random instances.
n_kids = np.arange(min_kids, max_kids + 1, min_kids)
times = np.zeros((3, len(n_kids)))
values = np.zeros((3, len(n_kids)))
parameters = {"OutputFlag": 0, "TimeLimit": time_limit}

# Iterate over problem size.
for i, n in enumerate(n_kids):
    print("Num. kids:", n)

    # Generate graph.
    graph, L, U = create_graph(n)

    # MICP.
    traveling_salesman(graph, gurobi_parameters=parameters)
    times[0, i] = graph.solver_stats.solve_time
    values[0, i] = graph.value
    print("MICP:", times[0, i], values[0, i])

    # MINCP.
    traveling_salesman_spatial_bb(graph, gurobi_parameters=parameters)
    times[1, i] = graph.solver_stats.solve_time
    values[1, i] = graph.value
    print("MINCP:", times[1, i], values[1, i])

    # McCormick.
    traveling_salesman_mccormick(graph, L, U, gurobi_parameters=parameters)
    times[2, i] = graph.solver_stats.solve_time
    values[2, i] = graph.value
    print("McCormick:", times[2, i], values[2, i])

np.save("kids.npy", n_kids)
np.save("times.npy", times)
np.save("values.npy", values)
