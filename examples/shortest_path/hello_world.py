import cvxpy
import gcsopt

# Initialize directed graph.
directed = True
G = gcsopt.GraphOfConvexSets(directed)

# Add vertices to graph.
l = 3 # Side length of vertex grid.
r = .3 # Radius of vertex circles.
for i in range(l):
    for j in range(l):
        c = (i, j) # Center of the circle.
        v = G.add_vertex(c) # Vertex named after its center.
        x = v.add_variable(2) # Continuous variable.
        v.add_constraint(cvxpy.norm2(x - c) <= r) # Constrain in circle.

# Add edges to graph.
for i in range(l):
    for j in range(l):
        cv = (i, j) # Name of vertex v.
        v = G.get_vertex(cv) # Retrieve vertex using its name.
        xv = v.variables[0] # Get first and only variable paired with vertex.

        # Add right and upward neighbors if not at grid boundary.
        neighbors = [(i + 1, j), (i, j + 1)] # Names of candidate neighbors.
        for cw in neighbors:
            if G.has_vertex(cw): # False if at grid boundary.
                w = G.get_vertex(cw) # Get neighbor vertex from its name.
                xw = w.variables[0] # Get neighbor variable.
                e = G.add_edge(v, w) # Connect vertices with edge.
                e.add_cost(cvxpy.norm2(xw - xv)) # Add L2 cost to edge.

# Solve shortest-path problem.
s = G.get_vertex((0, 0)) # Source vertex.
t = G.get_vertex((l - 1, l - 1)) # Target vertex.
G.solve_shortest_path(s, t) # Solve problem and populate graph with result.

# Print solution statistics.
print("Problem status:", G.status)
print("Problem optimal value:", G.value)
for v in G.vertices:
    x = v.variables[0]
    print(f"Variable {v.name} optimal value:", x.value)

# Plot optimal solution.
import matplotlib.pyplot as plt
plt.figure() # Initialize empty figure.
G.plot_2d() # Plot graph of convex sets.
G.plot_2d_solution() # Plot optimal subgraph.
plt.show() # Show figure.

# Solve problem from list of ILP constraints.

# Edge constraints.
ilp = [] 
for e in G.edges:
    ye = e.binary_variable
    ilp.append(ye >= 0)

# Vertex constraints.
for v in G.vertices:
    yv = v.binary_variable
    if v in [s, t]:
        ilp.append(yv == 1)
    else:
        ilp.append(yv <= 1)
    if v != s:
        ye_inc = sum (e.binary_variable for e in G.incoming_edges(v))
        ilp.append(yv == ye_inc)
    if v != t:
        ye_out = sum (e.binary_variable for e in G.outgoing_edges(v))
        ilp.append(yv == ye_out)
    
# Solve probelm from ILP constraints. Check that optimal value is equal to the
# one above.
G.solve_from_ilp(ilp)
print("Problem status from ILP:", G.status)
print("Optimal value from ILP:", G.value)