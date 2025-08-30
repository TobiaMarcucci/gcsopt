import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, enforce_edge_programs, constraint_homogenization,
    set_solution, BaseCallback)

def shortest_path_conic(conic_graph, source, target, binary, tol, gurobi_parameters=None, save_bounds=False):

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary, add_yv=True)

    # Edge costs and constraints.
    cost = enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head)

    # Enforce vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # Source vertex.
        if vertex.name == source.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            model.addConstr(yv[i] == 1)
            model.addConstr(sum(ye[out]) == 1)
            model.addConstr(sum(ze_tail[out]) == zv[i])
            for k in inc:
                model.addConstr(ye[k] == 0)
                model.addConstr(ze_head[k] == 0)

        # Target vertex.
        elif vertex.name == target.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            model.addConstr(yv[i] == 1)
            model.addConstr(sum(ye[inc]) == 1)
            model.addConstr(sum(ze_head[inc]) == zv[i])
            for k in out:
                model.addConstr(ye[k] == 0)
                model.addConstr(ze_tail[k] == 0)

        # All other vertices.
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            model.addConstr(yv[i] <= 1)
            model.addConstr(sum(ye[inc]) == yv[i])
            model.addConstr(sum(ye[out]) == yv[i])
            model.addConstr(sum(ze_head[inc]) == zv[i])
            model.addConstr(sum(ze_tail[out]) == zv[i])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with or without callback.
    if save_bounds:
        callback = BaseCallback(conic_graph, ye, save_bounds)
        model.optimize(callback)
        set_solution(model, conic_graph, yv, zv, ye, ze, tol, callback)
    else:
        model.optimize()
        set_solution(model, conic_graph, yv, zv, ye, ze, tol)

def shortest_path(convex_graph, source, target, binary=True, tol=1e-4, gurobi_parameters=None, save_bounds=False):
        conic_graph = convex_graph.to_conic()
        conic_source = conic_graph.get_vertex(source.name)
        conic_target = conic_graph.get_vertex(target.name)
        shortest_path_conic(conic_graph, conic_source, conic_target, binary, tol, gurobi_parameters, save_bounds)
        convex_graph._set_solution(conic_graph)