import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, edge_cost_homogenization, 
    edge_constraint_homogenization, constraint_homogenization,
    set_solution, CutsetCallback)

def minimum_spanning_tree_mccormick_conic(conic_graph, conic_root, L, U, tol, gurobi_parameters=None):

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

    # Vertex costs.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)
        constraint_homogenization(model, vertex, zv[i], 1)

        # McCormick envelopes.
        incident = conic_graph.incident_edge_indices(vertex)
        for k in incident:
            tail = conic_graph.edges[k].tail
            zek = ze_tail[k] if tail == vertex else ze_head[k]
            model.addConstr(zek[1:] >= ye[k] * L[i])
            model.addConstr(zek[1:] <= ye[k] * U[i])
            model.addConstr(zv[i][1:] - zek[1:] >= (1 - ye[k]) * L[i])
            model.addConstr(zv[i][1:] - zek[1:] <= (1 - ye[k]) * U[i])

        # Constraints on incoming edges.
        inc = conic_graph.incoming_edge_indices(vertex)
        if vertex == conic_root:
            model.addConstrs((ye[k] == 0 for k in inc))
        else:
            model.addConstr(sum(ye[inc]) == 1)

        # Constraints on outgoing edges.
        for k in conic_graph.outgoing_edge_indices(vertex):
            constraint_homogenization(model, vertex, zv[i] - ze_tail[k], 1 - ye[k])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with lazy constraints.
    model.Params.LazyConstraints = 1
    callback = CutsetCallback(conic_graph, ye)
    model.optimize(callback)
    set_solution(model, conic_graph, yv, zv, ye, ze, tol, callback)

def minimum_spanning_tree_mccormick(convex_graph, root, L, U, tol=1e-4, gurobi_parameters=None):
    """
    Parameter root is ignored for undirected graphs.
    """
    conic_graph = convex_graph.to_conic()
    conic_root = conic_graph.get_vertex(root.name) if root else None
    minimum_spanning_tree_mccormick_conic(conic_graph, conic_root, L, U, tol, gurobi_parameters)
    convex_graph._set_solution(conic_graph)