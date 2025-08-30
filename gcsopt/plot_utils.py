import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import graphviz as gv

def discretize_2d_vertex(vertex, n=50):
    values = get_values(vertex)
    variable = vertex.variables[0]
    cost = cp.Parameter(2)
    prob = cp.Problem(cp.Maximize(cost @ variable), vertex.constraints)
    vertices = np.zeros((n, 2))
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n)):
        cost.value = np.array([np.cos(angle), np.sin(angle)])            
        prob.solve(warm_start=True)
        vertices[i] = variable.value
    set_value(vertex, values)
    return vertices

def get_values(vertex):
    return [variable.value for variable in vertex.variables]

def set_value(vertex, values):
    for variable, value in zip(vertex.variables, values):
        variable.value = value

def plot_2d_vertex(vertex, n=50, tol=1e-4, **kwargs):
    vertices = discretize_2d_vertex(vertex, n)
    options = {'fc': 'mintcream', 'ec': 'black'}
    options.update(kwargs)
    vertex_min = np.min(vertices, axis=0)
    vertex_max = np.max(vertices, axis=0)
    vertex_dist = np.linalg.norm(vertex_max - vertex_min)
    if vertex_dist <= tol:
        plt.scatter(*vertices[0], fc='k', ec='k')
    else:
        plt.fill(*vertices.T, **options, zorder=0)
    
def plot_2d_edge(edge, endpoints=None, directed=True, **kwargs):
    for variables in [edge.tail.variables, edge.head.variables]:
        if variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
    if directed:
        arrowstyle = "->, head_width=2, head_length=5"
    else:
        arrowstyle = "-"
    options = dict(zorder=2, arrowstyle=arrowstyle)
    options.update(kwargs)
    if endpoints is None:
        endpoints = closest_points(edge.tail, edge.head)
    arrow = patches.FancyArrowPatch(*endpoints, **options)
    plt.gca().add_patch(arrow)

def closest_points(vertex1, vertex2):
    values1 = get_values(vertex1)
    values2 = get_values(vertex2)
    variable1 = vertex1.variables[0]
    variable2 = vertex2.variables[0]
    cost = cp.sum_squares(variable2 - variable1)
    constraints = vertex1.constraints + vertex2.constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    points = [variable1.value, variable2.value]
    set_value(vertex1, values1)
    set_value(vertex2, values2)
    return points

def plot_2d_graph(graph, n=50):
    for vertex in graph.vertices:
        if vertex.variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
        plot_2d_vertex(vertex, n)
    for edge in graph.edges:
        plot_2d_edge(edge, directed=graph.directed, color='grey')

def plot_2d_solution(graph, tol=1e-4):
    for vertex in graph.vertices:
        if vertex.binary_variable.value is not None and vertex.binary_variable.value > tol:
            variable = vertex.variables[0]
            plt.scatter(*variable.value, fc='w', ec='k', zorder=3)
    for edge in graph.edges:
        if edge.binary_variable.value is not None and edge.binary_variable.value > tol:
            tail = edge.tail.variables[0].value
            head = edge.head.variables[0].value
            endpoints = (tail, head)
            plot_2d_edge(edge, endpoints, directed=graph.directed, color='blue')

def graphviz_graph(graph, vertex_labels=None, edge_labels=None):
    if vertex_labels is None:
        vertex_labels = [vertex.name for vertex in graph.vertices]
    if edge_labels is None:
        edge_labels = [''] * graph.num_edges()
    if graph.directed:
        dot = gv.Digraph()
    else:
        dot = gv.Graph()
    for label in vertex_labels:
        dot.node(str(label))
    for edge, label in zip(graph.edges, edge_labels):
        tail = vertex_labels[graph.vertices.index(edge.tail)]
        head = vertex_labels[graph.vertices.index(edge.head)]
        dot.edge(str(tail), str(head), str(label))
    return dot
