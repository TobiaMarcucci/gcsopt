# GCSOPT

Python library to solve optimization problems in Graphs of Convex Sets (GCS).
For a detailed description of the algorithms implemented implemented in this library see the PhD thesis [Graphs of Convex Sets with Applications to Optimal Control and Motion Planning](https://dspace.mit.edu/handle/1721.1/156598?show=full).
(Please note that the library recently changed name, and in the thesis it is called `gcspy`.)

## Main features

- Uses the syntax of [CVXPY](https://www.cvxpy.org) for describing convex sets and convex functions.
- Provides a simple interface for assembling your graphs.
- Interface with state-of-the-art solvers via [CVXPY](https://www.cvxpy.org/).

## Installation

You can install the latest release from [PyPI](https://pypi.org/project/gcsopt/):
```bash
pip install gcsopt
```

To install from source:
```bash
git clone https://github.com/TobiaMarcucci/gcsopt.git
cd gcsopt
pip install .
```


## Example
Here is a minimal example of how to use gcsopt for solving a shortest-path problem in GCS:
```python
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
```

The otput of this script is:
```bash
Problem status: optimal
Optimal value: 2.4561622509772887
Variable (0, 0) optimal value: [0.28768714 0.08506533]
Variable (0, 1) optimal value: None
Variable (0, 2) optimal value: None
Variable (1, 0) optimal value: [0.82565028 0.24413557]
Variable (1, 1) optimal value: [1.21213203 0.78786797]
Variable (1, 2) optimal value: None
Variable (2, 0) optimal value: None
Variable (2, 1) optimal value: [1.75586443 1.17434971]
Variable (2, 2) optimal value: [1.91493467 1.71231286]
```

## License
This project is licensed under the MIT License.

## Author
Developed and maintained by Tobia Marcucci.
