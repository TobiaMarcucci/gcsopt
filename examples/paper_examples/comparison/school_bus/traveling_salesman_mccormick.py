import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, edge_cost_homogenization, edge_constraint_homogenization,
    constraint_homogenization, set_solution, SubtourEliminationCallback)

def traveling_salesman_mccormick_conic(conic_graph, L, U, tol, gurobi_parameters=None):
    assert not conic_graph.directed

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph)

    # Edge costs and constraints.
    cost = 0
    for k, edge in enumerate(conic_graph.edges):
        cost += edge_cost_homogenization(edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        edge_constraint_homogenization(model, edge, ze_tail[k], ze_head[k], ze[k], ye[k])

    # Vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)
        constraint_homogenization(model, vertex, zv[i], 1)

        # McCormick envelopes.
        incident = conic_graph.incident_edge_indices(vertex)
        model.addConstr(sum(ye[incident]) == 2)
        for k in incident:
            tail = conic_graph.edges[k].tail
            zek = ze_tail[k] if tail == vertex else ze_head[k]
            idx = range(2) if vertex.name == 0 else range(2, 4)
            model.addConstr(zek[idx] >= ye[k] * L[i])
            model.addConstr(zek[idx] <= ye[k] * U[i])
            model.addConstr(zv[i][idx] - zek[idx] >= (1 - ye[k]) * L[i])
            model.addConstr(zv[i][idx] - zek[idx] <= (1 - ye[k]) * U[i])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with lazy constraints.
    model.Params.LazyConstraints = 1
    callback = SubtourEliminationCallback(conic_graph, ye)
    model.optimize(callback)
    set_solution(model, conic_graph, yv, zv, ye, ze, tol, callback)

def traveling_salesman_mccormick(convex_graph, L, U, tol=1e-4, gurobi_parameters=None):
    conic_graph = convex_graph.to_conic()
    traveling_salesman_mccormick_conic(conic_graph, L, U, tol, gurobi_parameters)
    convex_graph._set_solution(conic_graph)