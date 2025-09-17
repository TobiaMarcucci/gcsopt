import numpy as np
import cvxpy as cp
from collections.abc import Iterable
from gcsopt.vertices import ConicVertex, ConvexVertex
from gcsopt.edges import ConicEdge, ConvexEdge
from gcsopt.graph_problems.shortest_path import shortest_path
from gcsopt.graph_problems.traveling_salesman import traveling_salesman
from gcsopt.graph_problems.facility_location import facility_location
from gcsopt.graph_problems.minimum_spanning_tree import minimum_spanning_tree
from gcsopt.graph_problems.from_ilp import from_ilp

class Graph:
    """
    Base class that contains the method that are common to GraphOfConicSets
    and GraphOfConvexSets.
    """

    def __init__(self, directed=True):
        self.vertices = []
        self.edges = []
        self.directed = directed
        self.value = None
        self.status = None
        self.solver_stats = None

    def has_vertex(self, name):
        return name in [vertex.name for vertex in self.vertices]
    
    def has_edge(self, name):
        if self.directed:
            return name in [edge.name for edge in self.edges]
        else:
            return set(name) in [set(edge.name) for edge in self.edges]

    def ensure_vertex_name_available(self, name):
        if self.has_vertex(name):
            raise ValueError(f"Vertex with name {name} is aleady defined.")
        
    def ensure_edge_name_available(self, tail_name, head_name):
        if not self.has_vertex(tail_name):
            raise ValueError(f"Vertex with name {tail_name} is not defined.")
        if not self.has_vertex(head_name):
            raise ValueError(f"Vertex with name {head_name} is not defined.")
        name = (tail_name, head_name)
        if self.has_edge(name):
            raise ValueError(f"Edge with name {name} is aleady defined.")
    
    def add_vertex(self, name, *args, **kwargs):
        self.ensure_vertex_name_available(name)
        return self._add_vertex(name, *args, **kwargs)

    def add_edge(self, tail, head, *args, **kwargs):
        self.ensure_edge_name_available(tail.name, head.name)
        return self._add_edge(tail, head, *args, **kwargs)

    def _add_vertex(self, name, *args, **kwargs):
        """
        This method must be overwritte by the derived class.
        """
        raise NotImplementedError

    def _add_edge(self, tail, head, *args, **kwargs):
        """
        This method must be overwritte by the derived class.
        """
        raise NotImplementedError
    
    def append_vertex(self, vertex):
        self.ensure_vertex_name_available(vertex.name)
        return self.vertices.append(vertex)

    def append_edge(self, edge):
        self.ensure_edge_name_available(edge.tail.name, edge.head.name)
        return self.edges.append(edge)

    def get_vertex(self, name):
        for vertex in self.vertices:
            if vertex.name == name:
                return vertex
        raise ValueError(f"There is no vertex with name {name}.")

    def get_edge(self, tail_name, head_name):
        name = (tail_name, head_name)
        if not self.directed:
            name_set = set(name)
        for edge in self.edges:
            if self.directed:
                if edge.name == name:
                    return edge
            else:
                if set(edge.name) == name_set:
                    return edge
        raise ValueError(f"No edge found with name {name}.")

    def vertex_index(self, vertex):
        return self.vertices.index(vertex)

    def edge_index(self, edge):
        return self.edges.index(edge)
    
    def _incoming_edge_indices(self, vertices):
        if not isinstance(vertices, Iterable):
            vertices = [vertices]
        def is_incoming(edge, vertices):
            return edge.tail not in vertices and edge.head in vertices
        return [k for k, edge in enumerate(self.edges) if is_incoming(edge, vertices)]
    
    def incoming_edge_indices(self, vertices):
        """
        Return indices of edges that are incoming to `vertices` (i.e., edges
        whose head is in `vertices` but tail is not).
        """
        if not self.directed:
            raise ValueError("Incoming edge indices cannot be computed for undirected graphs.")
        return self._incoming_edge_indices(vertices)
    
    def _outgoing_edge_indices(self, vertices):
        if not isinstance(vertices, Iterable):
            vertices = [vertices]
        def is_outgoing(edge, vertices):
            return edge.tail in vertices and edge.head not in vertices
        return [k for k, edge in enumerate(self.edges) if is_outgoing(edge, vertices)]

    def outgoing_edge_indices(self, vertices):
        """
        Return indices of edges that are outgoing from `vertices` (i.e., edges
        whose tail is in `vertices` but head is not).
        """
        if not self.directed:
            raise ValueError("Outgoing edge indices cannot be computed for undirected graphs.")
        return self._outgoing_edge_indices(vertices)

    def incident_edge_indices(self, vertices):
        """
        Return indices of edges that are incident with `vertices` (either
        incoming or outgoing).
        """
        incoming = self._incoming_edge_indices(vertices)
        outgoing = self._outgoing_edge_indices(vertices)
        return incoming + outgoing
    
    def induced_edge_indices(self, vertices):
        """
        Return indices of edges that have both ends in `vertices`.
        """
        def is_induced(edge, vertices):
            return (edge.tail in vertices) and (edge.head in vertices)
        return [k for k, edge in enumerate(self.edges) if is_induced(edge, vertices)]
    
    def _incoming_edges(self, vertices):
        return [self.edges[k] for k in self._incoming_edge_indices(vertices)]

    def incoming_edges(self, vertices):
        """
        Return edges that are incoming to `vertices` (i.e., edges whose head is
        in `vertices` but tail is not).
        """
        if not self.directed:
            raise ValueError("Incoming edges cannot be computed for undirected graphs.")
        return self._incoming_edges(vertices)

    def _outgoing_edges(self, vertices):
        return [self.edges[k] for k in self._outgoing_edge_indices(vertices)]
    
    def outgoing_edges(self, vertices):
        """
        Return edges that are outgoing from `vertices` (i.e., edges whose tail
        is in `vertices` but head is not).
        """
        if not self.directed:
            raise ValueError("Outgoing edges cannot be computed for undirected graphs.")
        return self._outgoing_edges(vertices)
        
    def incident_edges(self, vertices):
        """
        Return edges that are incident with `vertices` (either incoming or
        outgoing).
        """
        return self._incoming_edges(vertices) + self._outgoing_edges(vertices)
    
    def induced_edges(self, vertices):
        """
        Return edges that have both ends in `vertices`.
        """
        return [self.edges[k] for k in self.induced_edge_indices(vertices)]

    def num_vertices(self):
        """
        Return number of vertices in the graph.
        """
        return len(self.vertices)

    def num_edges(self):
        """
        Return number of edges in the graph.
        """
        return len(self.edges)
    
    def vertex_binaries(self):
        return np.array([vertex.binary_variable for vertex in self.vertices])

    def edge_binaries(self):
        return np.array([edge.binary_variable for edge in self.edges])
    
    def vertex_binary_values(self):
        return np.array([vertex.binary_variable.value for vertex in self.vertices])

    def edge_binary_values(self):
        return np.array([edge.binary_variable.value for edge in self.edges])
    
    def add_disjoint_subgraph(self, graph):
        for vertex in graph.vertices:
            self.append_vertex(vertex)
        for edge in graph.edges:
            self.append_edge(edge)
    
    def graphviz(self):
        from gcsopt.plot_utils import graphviz_graph
        return graphviz_graph(self)

class GraphOfConicSets(Graph):

    def __init__(self, directed=True):
        super().__init__(directed)

    def _add_vertex(self, name, size, id_to_range=None):
        vertex = ConicVertex(name, size, id_to_range)
        self.vertices.append(vertex)
        return vertex

    def _add_edge(self, tail, head, size, id_to_range=None):
        edge = ConicEdge(tail, head, size, id_to_range)
        self.edges.append(edge)
        return edge
    
    def solve_shortest_path(self, source, target, binary=True, tol=1e-4, **kwargs):
        if self.directed:
            shortest_path(self, source, target, binary, tol, **kwargs)
        else:
            raise NotImplementedError

    def solve_traveling_salesman(self, subtour_elimination=True, binary=True, tol=1e-4, **kwargs):
        return traveling_salesman(self, subtour_elimination, binary, tol, **kwargs)
    
    def solve_facility_location(self, binary=True, tol=1e-4, **kwargs):
        if self.directed:
            facility_location(self, binary, tol, **kwargs)
        else:
            raise ValueError("Graph must be directed for facility location problem.")
    
    def solve_minimum_spanning_tree(self, root=None, subtour_elimination=True, binary=True, tol=1e-4, **kwargs):
        """
        Parameter root is ignored for undirected graphs.
        """
        return minimum_spanning_tree(self, root, subtour_elimination, binary, tol, **kwargs)
    
    def solve_from_ilp(self, ilp_constraints, binary=True, tol=1e-4, **kwargs):
        from_ilp(self, ilp_constraints, binary, tol, **kwargs)

class GraphOfConvexSets(Graph):

    def __init__(self, directed=True):
        super().__init__(directed)
        
    def _add_vertex(self, name):
        vertex = ConvexVertex(name)
        self.vertices.append(vertex)
        return vertex

    def _add_edge(self, tail, head):
        edge = ConvexEdge(tail, head)
        self.edges.append(edge)
        return edge

    def to_conic(self):

        # Initialize empty conic graph.
        conic_graph = GraphOfConicSets(directed=self.directed)

        # Add one vertex at the time.
        for convex_vertex in self.vertices:
            conic_vertex = convex_vertex.to_conic()
            conic_graph.append_vertex(conic_vertex)

        # Add one edge at the time.
        for convex_edge in self.edges:
            conic_tail = conic_graph.get_vertex(convex_edge.tail.name)
            conic_head = conic_graph.get_vertex(convex_edge.head.name)
            conic_edge = convex_edge.to_conic(conic_tail, conic_head)
            conic_graph.append_edge(conic_edge)

        return conic_graph

    def _set_solution(self, conic_graph):

        # Set problem value and stats.
        self.value = conic_graph.value
        self.status = conic_graph.status
        self.solver_stats = conic_graph.solver_stats

        # Set value of vertex variables for convex program.
        for convex_vertex, conic_vertex in zip(self.vertices, conic_graph.vertices):
            if conic_vertex.x.value is None:
                for variable in convex_vertex.variables:
                    variable.value = None
            else:
                for variable in convex_vertex.variables:
                    variable.value = conic_vertex.get_convex_variable_value(variable, conic_vertex.x.value)

        # Set value of edge variables for convex program.
        for convex_edge, conic_edge in zip(self.edges, conic_graph.edges):
            if conic_edge.x.value is None:
                for variable in convex_edge.variables:
                    variable.value = None
            else:
                for variable in convex_edge.variables:
                    variable.value = conic_edge.get_convex_variable_value(variable, conic_edge.x.value)
    
    def solve_shortest_path(self, source, target, binary=True, tol=1e-4, **kwargs):
        conic_graph = self.to_conic()
        conic_source = conic_graph.get_vertex(source.name)
        conic_target = conic_graph.get_vertex(target.name)
        conic_graph.solve_shortest_path(conic_source, conic_target, binary, tol, **kwargs)
        self._set_solution(conic_graph)

    def solve_traveling_salesman(self, subtour_elimination=True, binary=True, tol=1e-4, **kwargs):
        conic_graph = self.to_conic()
        conic_graph.solve_traveling_salesman(subtour_elimination, binary, tol, **kwargs)
        self._set_solution(conic_graph)
    
    def solve_facility_location(self, binary=True, tol=1e-4, **kwargs):
        conic_graph = self.to_conic()
        conic_graph.solve_facility_location(binary, tol, **kwargs)
        self._set_solution(conic_graph)
    
    def solve_minimum_spanning_tree(self, root=None, subtour_elimination=True, binary=True, tol=1e-4, **kwargs):
        """
        Parameter root is ignored for undirected graphs.
        """
        conic_graph = self.to_conic()
        conic_root = conic_graph.get_vertex(root.name) if root else None
        conic_graph.solve_minimum_spanning_tree(conic_root, subtour_elimination, binary, tol, **kwargs)
        self._set_solution(conic_graph)
    
    def solve_from_ilp(self, ilp_constraints, binary=True, tol=1e-4, **kwargs):
        conic_graph = self.to_conic()
        conic_graph.solve_from_ilp(ilp_constraints, binary, tol, **kwargs)
        self._set_solution(conic_graph)

    # TODO: try to reuse the following method for all the graph problems.
    # TODO: add the following method also to the conic graph.
    def solve_shortest_path_with_rounding(self, source, target, rounding_fn, tol=1e-4, **kwargs):
        binary = False
        self.solve_shortest_path(source, target, binary, tol, **kwargs)
        restriction = rounding_fn(self, source, target)
        return restriction

    def solve_convex_restriction(self, vertices, edges, **kwargs):
        """
        Solves convex program obtained by discarding all the vertices and edges
        that are not in the given lists.
        """

        # Check that given vertices and edges form a valid subgraph.
        for vertex in vertices:
            if vertex not in self.vertices:
                raise ValueError("Vertices are not a subset of the graph vertices.")
        for edge in edges:
            if edge.tail not in vertices or edge.head not in vertices:
                raise ValueError("Given vertices and edges do not form a subgraph.")

        # Assemble convex program.
        cost = 0
        constraints = []
        for vertex in vertices:
            cost += vertex.cost
            constraints.extend(vertex.constraints)
        for edge in edges:
            cost += edge.cost
            constraints.extend(edge.constraints)

        # Solve convex program.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(**kwargs)

        # Set problem value and stats.
        self.value = prob.value
        self.status = prob.status
        self.solver_stats = prob.solver_stats

        # Set vertex variable values.
        for vertex in self.vertices:
            if vertex in vertices:
                vertex.binary_variable.value = 1
            else:
                vertex.binary_variable.value = 0
                for variable in vertex.variables:
                    variable.value = None

        # Set edge variable values.
        for edge in self.edges:
            if edge in edges:
                edge.binary_variable.value = 1
            else:
                edge.binary_variable.value = 0
                for variable in edge.variables:
                    variable.value = None

    def plot_2d(self, **kwargs):
        from gcsopt.plot_utils import plot_2d_graph
        return plot_2d_graph(self, **kwargs)

    def plot_2d_solution(self):
        from gcsopt.plot_utils import plot_2d_solution
        return plot_2d_solution(self)
