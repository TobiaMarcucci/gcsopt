import cvxpy as cp
from gcsopt import GraphOfConvexSets

# Initialize directed graph.
directed = True
G = GraphOfConvexSets(directed)

# Add vertices to graph.
l = 3 # Side of vertex grid.
r = .3 # Radius of vertex circles.
for i in range(l):
    for j in range(l):
        c = (i, j) # Center of the circle.
        v = G.add_vertex(c) # Vertex named after its center.
        x = v.add_variable(2) # Continuous variable.
        v.add_constraint(cp.norm2(x - c) <= r) # Constrain in circle.

# Add edges to graph.
for i in range(l):
    for j in range(l):
        cv = (i, j) # Name of vertex v.
        v = G.get_vertex(cv) # Retrieve vertex from its name.
        xv = v.variables[0] # Get first and only variable added to vertex.

        # Add right and upward neighbors if not at grid boundary.
        neighbors = [(i + 1, j), (i, j + 1)] # Names of candidate neighbors.
        for cw in neighbors:
            if G.has_vertex(cw): # False if at grid boundary.
                w = G.get_vertex(cw) # Get neighbor vertex from its name.
                xw = w.variables[0] # Get neighbor variable.
                e = G.add_edge(v, w) # Connect vertices with edge.
                e.add_cost(cp.norm2(xw - xv)) # Add L2 cost to edge.

# Solve shortest-path problem.
s = G.get_vertex((0, 0)) # Source vertex.
t = G.get_vertex((l - 1, l - 1)) # Target vertex.
G.solve_shortest_path(s, t) # Populates graph with result of optimization problem.

# Print solution statistics.
print("Problem status:", G.status)
print("Problem optimal value:", G.value)
for v in G.vertices:
    x = v.variables[0]
    print(f"Variable {v.name} optimal value:", x.value)

# Plot optimal solution.
import matplotlib.pyplot as plt
plt.figure() # Initializes empty figure.
G.plot_2d() # Plot graph of convex sets.
G.plot_2d_solution() # Plot optimal subgraph.
plt.show() # Show figure.