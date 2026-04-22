import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, edge_cost_homogenization, edge_constraint_homogenization,
    constraint_homogenization, set_solution)

def shortest_path_spatial_bb_conic(conic_graph, source, target, tol, gurobi_parameters=None):

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, add_yv=True)

    # Edge costs and constraints.
    cost = 0
    for k, edge in enumerate(conic_graph.edges):
        cost += edge_cost_homogenization(edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        edge_constraint_homogenization(model, edge, ze_tail[k], ze_head[k], ze[k], ye[k])

    # Vertex costs.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], yv[i])
        constraint_homogenization(model, vertex, zv[i], yv[i])

        # Enforce vertex costs and constraints.
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # Source vertex.
        if vertex.name == source.name:
            model.addConstr(yv[i] == 1)
            model.addConstr(sum(ye[out]) == 1)
            for k in inc:
                model.addConstr(ye[k] == 0)

        # Target vertex.
        elif vertex.name == target.name:
            model.addConstr(yv[i] == 1)
            model.addConstr(sum(ye[inc]) == 1)
            for k in out:
                model.addConstr(ye[k] == 0)

        # All other vertices.
        else:
            model.addConstr(yv[i] <= 1)
            model.addConstr(sum(ye[inc]) == yv[i])
            model.addConstr(sum(ye[out]) == yv[i])

        # Bilinear constraints.
        for k in inc:
            model.addConstr(ze_head[k] == ye[k] * zv[i])
        for k in out:
            model.addConstr(ze_tail[k] == ye[k] * zv[i])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with or without callback.
    model.optimize()
    set_solution(model, conic_graph, yv, zv, ye, ze, tol)

def shortest_path_spatial_bb(convex_graph, source, target, tol=1e-4, gurobi_parameters=None):
        conic_graph = convex_graph.to_conic()
        conic_source = conic_graph.get_vertex(source.name)
        conic_target = conic_graph.get_vertex(target.name)
        shortest_path_spatial_bb_conic(conic_graph, conic_source, conic_target, tol, gurobi_parameters)
        convex_graph._set_solution(conic_graph)