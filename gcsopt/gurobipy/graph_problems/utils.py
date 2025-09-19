import numpy as np
import cvxpy as cp
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

def create_environment(gurobi_parameters=None):
    env = gp.Env()
    gurobi_parameters = dict(gurobi_parameters or {})
    if not "OutputFlag" in gurobi_parameters:
        gurobi_parameters.setdefault("OutputFlag", 0)
    for key, value in gurobi_parameters.items():
        env.setParam(key, value)
    return env

def constrain_in_cone(model, z, K):

    # Linear constraints.
    if K == cp.Zero:
        model.addConstr(z == 0)
    elif K == cp.NonNeg:
        model.addConstr(z >= 0)
    elif K == cp.NonPos:
        model.addConstr(z <= 0)

    # Second order cone constraint.
    elif K == cp.SOC:
        z0 = model.addVar() # Nonnegative slack variable.
        z1 = model.addMVar(z.size - 1, lb=-np.inf)
        model.addConstr(z[0] == z0)
        model.addConstr(z[1:] == z1)
        quad_expr = gp.quicksum(zi.item() * zi.item() for zi in z1)
        model.addQConstr(quad_expr <= z0 * z0) # Convex for gurobi.

    # There are no other constraints we can support for MICPs.
    else:
        raise NotImplementedError

def constraint_homogenization(model, prog, x, y):
    z = prog.A @ x + prog.b * y
    start = 0
    for cone_type, cone_size in prog.K:
        stop = start + cone_size
        constrain_in_cone(model, z[start:stop], cone_type)
        start = stop

def edge_cost_homogenization(edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    return edge.c @ x + edge.d * y
    
def edge_constraint_homogenization(model, edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    constraint_homogenization(model, edge, x, y)

def define_variables(model, conic_graph, binary=True, add_yv=False):

    # Binary variables.
    vtype = GRB.BINARY if binary else GRB.CONTINUOUS
    ye = model.addMVar(conic_graph.num_edges(), vtype=vtype)

    # Function that allows adding variables of zero size.
    add_var = lambda size: model.addMVar(size, lb=-np.inf) if size > 0 else np.array([])
    
    # Auxiliary continuous varibales.
    zv = np.array([add_var(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([add_var(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([add_var(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([add_var(edge.head.size) for edge in conic_graph.edges])

    if add_yv:
        yv = model.addMVar(conic_graph.num_vertices(), vtype=vtype)
    else:
        yv = None
    return yv, zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head):

    # Edge costs and constraints.
    cost = 0
    for k, edge in enumerate(conic_graph.edges):
        cost += edge_cost_homogenization(edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        edge_constraint_homogenization(model, edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        constraint_homogenization(model, edge.tail, ze_tail[k], ye[k])
        constraint_homogenization(model, edge.head, ze_head[k], ye[k])

    return cost

def set_solution(model, conic_graph, yv, zv, ye, ze, tol, callback=None):

    # Set problem value and stats.
    if model.status == 2:
        conic_graph.status = "optimal"
        conic_graph.value = model.ObjVal
    elif model.status == 3:
        conic_graph.status = "infeasible"
        conic_graph.value = np.inf
    elif model.status == 4:
        conic_graph.status = "infeasible_or_unbounded"
        conic_graph.value = None
    elif model.status == 5:
        conic_graph.status = "unbounded"
        conic_graph.value = - np.inf
    elif model.status == 12:
        conic_graph.status = "numeric_error"
        conic_graph.value = model.ObjVal
    else:
        conic_graph.status = model.status
        conic_graph.value = None
    conic_graph.solver_stats = cp.problems.problem.SolverStats(
        solver_name = 'GUROBI',
        solve_time = model.Runtime)
    
    # Set bounds from callback.
    if callback is not None and callback.save_bounds:
        callback.callback_times.append(model.Runtime)
        callback.lower_bounds.append(model.ObjVal)
        callback.upper_bounds.append(model.ObjVal)
        callback.upper_bounds = [v if v != 1e100 else np.nan for v in callback.upper_bounds]
        conic_graph.solver_stats.callback_bounds = np.vstack([
            callback.callback_times,
            callback.lower_bounds,
            callback.upper_bounds])

    # Set vertex variable values.
    for i, vertex in enumerate(conic_graph.vertices):
        if model.status in [2, 12, 13]: # Optimal, numeric error, suboptimal.
            y = 1 if yv is None else yv[i].X
            vertex.binary_variable.value = y
            if y > tol:
                vertex.x.value = zv[i].X / y
            else:
                vertex.x.value = None
        else:
            vertex.binary_variable.value = None
            vertex.x.value = None

    # Set edge variable values.
    for edge, y, z in zip(conic_graph.edges, ye, ze):
        if model.status in [2, 12, 13]: # Optimal, numeric error, suboptimal.
            edge.binary_variable.value = y.X
            z_value = z.X if z.size > 0 else np.array([])
            if y.X > tol:
            # if y.X is not None and y.X > tol:
                edge.x.value = np.concatenate((
                    edge.tail.x.value,
                    edge.head.x.value,
                    z_value / y.X))
            else:
                edge.x.value = None
        else:
            edge.binary_variable.value = None
            edge.x.value = None

def subtour_elimination_constraints(model, conic_graph, ye):
    """
    Subtour elimination constraints for all subsets of vertices.
    """
    start = 2 if conic_graph.directed else 3
    for n_vertices in range(start, conic_graph.num_vertices() - 1):
        for vertices in combinations(conic_graph.vertices, n_vertices):
            ind = conic_graph.induced_edge_indices(vertices)
            model.addConstr(sum(ye[ind]) <= n_vertices - 1)
            
class BaseCallback:

    def __init__(self, conic_graph, ye, save_bounds=False):
        self.conic_graph = conic_graph
        self.ye = ye
        self.save_bounds = save_bounds
        if self.save_bounds:
            self.callback_times = []
            self.lower_bounds = []
            self.upper_bounds = []

    def __call__(self, model, where):
        if self.save_bounds and where == GRB.Callback.MIP:
            self.callback_times.append(model.cbGet(GRB.Callback.RUNTIME))
            self.lower_bounds.append(model.cbGet(GRB.Callback.MIP_OBJBND))
            self.upper_bounds.append(model.cbGet(GRB.Callback.MIP_OBJBST))

class SubtourEliminationCallback(BaseCallback):

    def __call__(self, model, where):
        super().__call__(model, where)
        if where == GRB.Callback.MIPSOL:
            ye = model.cbGetSolution(self.ye)
            edges = [self.conic_graph.edges[k] for k, y in enumerate(ye) if y > 0.5]
            G = nx.DiGraph() if self.conic_graph.directed else nx.Graph()
            G.add_edges_from([(e.tail, e.head) for e in edges])
            length_bound = None if self.conic_graph.directed else nx.girth(G)
            tours = nx.simple_cycles(G, length_bound)
            tour = min(tours, key=len, default=None)
            if tour is not None and len(tour) < self.conic_graph.num_vertices():
                ind = self.conic_graph.induced_edge_indices(tour)
                model.cbLazy(sum(self.ye[ind]) <= len(tour) - 1)

    # def shortest_subtour(self, edges):
    #     """
    #     The edges here are only the ones that have binary equal to one. It is
    #     assumed there is exactly one incoming edge and one outgoing edge for
    #     every vertex represented in the edge list.
    #     """

    #     # Create a mapping from each vertex to its neighbors. Do not use the
    #     # neighbors method provided by the graph since it would also add
    #     # neighbors connected by edges with binary equal to zero.
    #     vertex_neighbors = {}
    #     for edge in edges:
    #         vertex_neighbors.setdefault(edge.tail, []).append(edge.head)
    #         if not self.conic_graph.directed:
    #             vertex_neighbors.setdefault(edge.head, []).append(edge.tail)

    #     # Follow edges to find cycles. Each time a new cycle is found, keep track
    #     # of the shortest cycle found so far and restart from an unvisited vertex.
    #     unvisited = set(vertex_neighbors)
    #     shortest = None
    #     while unvisited:
    #         cycle = []
    #         neighbors = list(unvisited)
    #         while neighbors:
    #             current = neighbors.pop()
    #             cycle.append(current)
    #             unvisited.remove(current)
    #             neighbors = [vertex for vertex in vertex_neighbors[current] if vertex in unvisited]
    #         if shortest is None or len(cycle) < len(shortest):
    #             shortest = cycle

    #     return shortest

class CutsetCallback(BaseCallback):

    def __init__(self, conic_graph, ye, save_bounds=False):
        if not conic_graph.directed:
            raise ValueError("Graph must be directed for cutset callback.")
        super().__init__(conic_graph, ye, save_bounds)

    def __call__(self, model, where):
        """
        Note that a cyle cannot contain the root here, since the root has zero
        incoming edges. Also, we could use the connected components instead, but
        the cycles seem to be more effective.
        """
        super().__call__(model, where)
        if where == GRB.Callback.MIPSOL:
            ye = model.cbGetSolution(self.ye)
            edges = [self.conic_graph.edges[k] for k, y in enumerate(ye) if y > 0.5]
            G = nx.DiGraph()
            G.add_edges_from([(e.tail, e.head) for e in edges])
            cycles = list(nx.simple_cycles(G))
            min_length = min((len(cycle) for cycle in cycles), default=np.inf)
            for cycle in cycles:
                if len(cycle) == min_length:
                    cut = self.conic_graph.incoming_edge_indices(cycle)
                    model.cbLazy(sum(self.ye[cut]) >= 1)

# class CutsetCallback(BaseCallback):

#     def __init__(self, conic_graph, ye, root, save_bounds=False):
#         if not conic_graph.directed:
#             raise ValueError("Graph must be directed for cutset callback.")
#         super().__init__(conic_graph, ye, save_bounds)
#         self.root = root

#     def __call__(self, model, where):
#         super().__call__(model, where)
#         if where == GRB.Callback.MIPSOL:
#             ye = model.cbGetSolution(self.ye)
#             edges = [self.conic_graph.edges[k] for k, y in enumerate(ye) if y > 0.5]
#             cut = self.get_cut(edges)
#             if cut:
#                 model.cbLazy(sum(self.ye[cut]) >= 1)

#     def get_cut(self, edges):
#         G = nx.DiGraph()
#         G.add_edges_from([(e.tail, e.head) for e in edges])
#         if not G.has_node(self.root):
#             return self.conic_graph.outgoing_edge_indices(self.root)
#         reachable = nx.descendants(G, self.root) | {self.root}
#         unreachable = set(self.conic_graph.vertices) - reachable
#         return self.conic_graph.incoming_edge_indices(unreachable)
