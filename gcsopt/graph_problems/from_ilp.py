import cvxpy as cp
import numpy as np
from gcsopt.programs import ConvexProgram
from gcsopt.graph_problems.utils import define_variables, enforce_edge_programs, set_solution

def from_ilp(conic_graph, ilp_constraints, binary, tol, **kwargs):

    # Put given constraints in conic form. Next lines are not nice but I cannot
    # use convex_ilp.add_variables.
    convex_ilp = ConvexProgram()
    vertex_binaries = conic_graph.vertex_binaries()
    edge_binaries = conic_graph.edge_binaries()
    convex_ilp.variables.extend(vertex_binaries)
    convex_ilp.variables.extend(edge_binaries)
    convex_ilp.add_constraints(ilp_constraints)
    conic_ilp = convex_ilp.to_conic()

    # Indices of vertex and edge binaries in conic program. Uses the fact that
    # binaries are scalars.
    vertex_indices = [conic_ilp.id_to_range[y.id].start for y in vertex_binaries]
    edge_indices = [conic_ilp.id_to_range[y.id].start for y in edge_binaries]
    Av = conic_ilp.A[:, vertex_indices]
    Ae = conic_ilp.A[:, edge_indices]

    # Variables of MICP. Note that xv and xe can always be omitted.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Vertex costs and constraints.
    constraints.append(yv <= 1)
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], yv[i])

    # Enforce constraints implied by subgraph constraint yv >= ye. Letting the
    # user decide when to enforce these constraints would be error prone.
    for k, edge in enumerate(conic_graph.edges):
        i = conic_graph.vertex_index(edge.tail)
        j = conic_graph.vertex_index(edge.head)
        constraints += edge.tail.constraint_homogenization(zv[i] - ze_tail[k], yv[i] - ye[k])
        constraints += edge.head.constraint_homogenization(zv[j] - ze_head[k], yv[j] - ye[k])

    # Check each line of each conic constraint.
    start = 0
    for K, size in conic_ilp.K:
        stop = start + size
        for j in range(start, stop):

            # Evaluate affine constraint using the problem binaries.
            av = Av[j]
            ae = Ae[j]
            bj = conic_ilp.b[j]

            # If there are no shared vertices, just enforce scalar constraint.
            shared_vertices = find_shared_vertices(conic_graph, av, ae)
            if not shared_vertices:
                constraints.append(K(av @ yv + ae @ ye + bj))

            # If there are shared vertices, apply constraint-generation lemma to
            # each affine constraint.
            for vertex in shared_vertices:

                # Assemble implied constraint.
                i = conic_graph.vertex_index(vertex)
                inc = conic_graph._incoming_edge_indices(vertex)
                out = conic_graph._outgoing_edge_indices(vertex)
                lhs = (bj + av[i]) * yv[i] + ae @ ye
                vector_lhs = (bj + av[i]) * zv[i]
                vector_lhs += sum(ae[inc] * ze_head[inc])
                vector_lhs += sum(ae[out] * ze_tail[out])

                # Enforce implied constraints.
                if K == cp.Zero:
                    constraints += [lhs == 0, vector_lhs == 0]
                    if not np.isclose(bj, 0):
                        constraints.append(yv[i] == 1)
                elif K == cp.NonNeg:
                    constraints += vertex.constraint_homogenization(vector_lhs, lhs)
                    if bj < 0:
                        constraints.append(yv[i] == 1)
                elif K == cp.NonPos:
                    constraints += vertex.constraint_homogenization(-vector_lhs, -lhs)
                    if bj > 0:
                        constraints.append(yv[i] == 1)
                else:
                    raise ValueError(
                        "All the constraints of ILP must be affine. Got cone "
                        f"of type {type(K)}.")
                
        # Shift row indices.
        start = stop
                
    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)
    
def find_shared_vertices(conic_graph, av, ae):
    """
    Checks if the linear function
    av^T y_v + ae^T y_e
    is amenable to the lemma that is used to generate spatial constraints
    (see Lemma 5.1 from thesis). To this end, it should be possible to rewrite
    the linear function above as
    b y_v_i + sum_{k in edges_incident_i} c_k y_e_k
    where i is the index of a vertex and k is the index of an edge incident with
    the ith vertex. This function returns all the values of i that allow such
    a decomposition.
    """

    # Extract nonzero vertices and edges.
    nonzero_vertices = [conic_graph.vertices[i] for i in np.nonzero(av)[0]]
    nonzero_edges = [conic_graph.edges[k] for k in np.nonzero(ae)[0]]

    # Compute shared vertices among all edges.
    tails_and_heads = [{edge.tail, edge.head} for edge in nonzero_edges]
    shared_vertices = set.intersection(*tails_and_heads) if tails_and_heads else set()

    # If av is zero, return all the vertices shared by the edges.
    if not nonzero_vertices:
        return list(shared_vertices)
    
    # If av has one nonzero entry, there is only one candidate vertex, and
    # no other shared vertex can be different from it.
    elif len(nonzero_vertices) == 1 and shared_vertices <= set(nonzero_vertices):
        return nonzero_vertices

    # In all other cases, the decomposition is not possible.
    else:
        return []
