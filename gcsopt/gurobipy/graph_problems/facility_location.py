import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, enforce_edge_programs, constraint_homogenization,
    set_solution, BaseCallback)

def facility_location_conic(conic_graph, binary, tol, gurobi_parameters=None, save_bounds=False):

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

        # Check that graph topology is correct.
        if len(inc) > 0 and len(out) > 0:
            raise ValueError("Graph is not bipartite.")

        # User vertices.
        if len(inc) > 0:
            cost += vertex.cost_homogenization(zv[i], 1)
            model.addConstr(yv[i] == 1)
            model.addConstr(sum(ye[inc]) == 1)
            model.addConstr(sum(ze_head[inc]) == zv[i])
        
        # Facility vertices.
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            model.addConstr(yv[i] <= 1)

    # Edge constraints.
    for k, edge in enumerate(conic_graph.edges):
        i = conic_graph.vertex_index(edge.tail)
        constraint_homogenization(model, edge.tail, zv[i] - ze_tail[k], yv[i] - ye[k])
        
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

def facility_location(convex_graph, binary=True, tol=1e-4, gurobi_parameters=None, save_bounds=False):
        conic_graph = convex_graph.to_conic()
        facility_location_conic(conic_graph, binary, tol, gurobi_parameters, save_bounds)
        convex_graph._set_solution(conic_graph)