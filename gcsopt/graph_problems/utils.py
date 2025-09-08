import cvxpy as cp
import numpy as np
from itertools import combinations
from gcsopt.safe_variable import safe_variable

def define_variables(conic_graph, binary):

    # Binary variables.
    yv = cp.Variable(conic_graph.num_vertices(), boolean=binary)
    ye = cp.Variable(conic_graph.num_edges(), boolean=binary)
    
    # Auxiliary continuous varibales.
    zv = np.array([safe_variable(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([safe_variable(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([safe_variable(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([safe_variable(edge.head.size) for edge in conic_graph.edges])

    return yv, zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head):

    # Edge costs and constraints.
    cost = 0
    constraints = []
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.cost_homogenization(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.constraint_homogenization(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.tail.constraint_homogenization(ze_tail[k], ye[k])
        constraints += edge.head.constraint_homogenization(ze_head[k], ye[k])

    return cost, constraints

def subtour_elimination_constraints(conic_graph, ye):
    """
    Subtour elimination constraints for all subsets of vertices with
    cardinality between 2 and num_vertices - 1.
    """
    constraints = []
    start = 2 if conic_graph.directed else 3
    for n_vertices in range(start, conic_graph.num_vertices() - 1):
        for vertices in combinations(conic_graph.vertices, n_vertices):
            ind = conic_graph.induced_edge_indices(vertices)
            constraints.append(sum(ye[ind]) <= n_vertices - 1)
    return constraints

def set_solution(conic_graph, prob, ye, ze, yv, zv, tol):

    # Set problem value and stats.
    conic_graph.value = prob.value
    conic_graph.status = prob.status
    conic_graph.solver_stats = prob.solver_stats

    # Set vertex variable values.
    for vertex, y, z in zip(conic_graph.vertices, yv, zv):
        vertex.binary_variable.value = y.value
        if y.value is not None and y.value > tol:
            vertex.x.value = z.value / y.value
        else:
            vertex.x.value = None

    # Set edge variable values.
    for edge, y, z in zip(conic_graph.edges, ye, ze):
        edge.binary_variable.value = y.value
        z_value = z.value if z.size > 0 else np.array([])
        if y.value is not None and y.value > tol:
            edge.x.value = np.concatenate((
                edge.tail.x.value,
                edge.head.x.value,
                z_value / y.value))