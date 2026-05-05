import numpy as np

def single_dfs(graph, source, target):
    """
    Perform a randomized depth-first search (DFS) from source to target in a
    graph.
    """

    # initialize path and set of visited vertices
    path = [source]
    visited = []

    # repeat until target is reached
    while path:
        if path[-1] == target:
            return path

        # collect neighbors and probabilities
        neighbors = []
        probabilities = []
        for edge in graph.outgoing_edges(path[-1]):
            neighbor = edge.head
            probability = edge.binary_variable.value
            if neighbor not in path + visited and probability > 0:
                neighbors.append(neighbor)
                probabilities.append(probability)

        # explore random neighbor
        if neighbors:
            probabilities = np.array(probabilities) / sum(probabilities)
            neighbor = np.random.choice(neighbors, p=probabilities)
            path.append(neighbor)
            
        # backtrack and prevent revisit of same vertex
        else:
            visited.append(path.pop())

    # path not found
    return None

def solve_path_convex_restriction(graph, path):
    """
    Solve a convex restriction of the original problem over a fixed path.
    """

    # retrieve edges along the given path
    path_edges = []
    for tail, head in zip(path[:-1], path[1:]):
        edge = graph.get_edge(tail.name, head.name)
        path_edges.append(edge)

    # solve convex restriction for given subgraph
    graph.solve_convex_restriction(path, path_edges)

def randomized_dfs(graph, source, target, num_paths=5, max_trials=100):
    """
    Run randomized DFS multiple times to find diverse paths, and solve a convex
    restriction for each path to determine the best one.
    """

    # try to find num_paths distinct paths within max_trials trials
    paths = []
    for _ in range(max_trials):
        if len(paths) == num_paths:
            break
        path = single_dfs(graph, source, target)
        if path is not None and path not in paths:
            paths.append(path)

    # solve convex restriction for each different path and keep best solution
    best_value = np.inf
    for path in paths:
        solve_path_convex_restriction(graph, path)
        if graph.value is not None and graph.value < best_value:
            best_path = path
            best_value = graph.value

    # solve problem another time if optimal solution has been overwritten
    if num_paths > 1 and not np.isinf(best_value) and best_value != graph.value:
        solve_path_convex_restriction(graph, best_path)