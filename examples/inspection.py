import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets

# All rooms in the layer plant. Each room is described by a triplet with room
# index, lower corner, and upper corner.
rooms = [
    (0, [0, 5], [2, 8]), 
    (1, [0, 3], [4, 5]),
    (2, [0, 0], [2, 3]),
    (3, [2, 5], [4, 8]),
    (4, [2, 0], [4, 3]),
    (5, [4, 0], [6, 8]),
    (6, [6, 7], [8, 8]),
    (7, [6, 1], [8, 7]),
    (8, [6, 0], [8, 1]),
    (9, [8, 0], [10, 10]),
    (10, [10, 5], [12, 10]),
    (11, [10, 4], [12, 5]),
    (12, [10, 0], [12, 4]),
    (13, [12, 0], [16, 2]),
    (14, [12, 2], [14, 10]),
    (15, [14, 7], [16, 10]),
    (17, [0, 8], [4, 10]),
    (18, [4, 8], [8, 10]),
    (19, [14, 2], [16, 7]),
]

# All doors in the layer plant. Each door is described by first room index,
# second room index, door lower corner, and door upper corner.
doors = [
    (0, 3, [2, 7], [2, 8]),
    (1, 2, [1, 3], [2, 3]),
    (1, 5, [4, 4], [4, 5]),
    (2, 4, [2, 2], [2, 3]),
    (3, 5, [4, 5], [4, 8]),
    (3, 17, [3, 8], [4, 8]),
    (4, 5, [4, 0], [4, 3]),
    (5, 6, [6, 7], [6, 8]),
    (5, 7, [6, 1], [6, 2]),
    (5, 8, [6, 0], [6, 1]),
    (6, 9, [8, 7], [8, 8]),
    (7, 9, [8, 3], [8, 4]),
    (8, 9, [8, 0], [8, 1]),
    (9, 10, [10, 7], [10, 8]),
    (9, 11, [10, 4], [10, 5]),
    (9, 12, [10, 3], [10, 4]),
    (9, 18, [8, 9], [8, 10]),
    (10, 14, [12, 5], [12, 6]),
    (11, 14, [12, 4], [12, 5]),
    (12, 13, [12, 0], [12, 2]),
    (14, 15, [14, 7], [14, 8]),
    (14, 19, [14, 6], [14, 7]),
]

# Rooms that must be visited by the trajectory.
visit_rooms = [0, 2, 7, 10, 13, 15, 18]

# Helper class that allows us to construct one layer of the graph.
class Layer(GraphOfConvexSets):

    def __init__(self, rooms, doors, name):
        super().__init__()
        self.rooms = rooms
        self.doors = doors
        self.name = name
        for room in rooms:
            self.add_room(*room)
        for door in doors:
            self.add_door(*door)
        
    def add_room(self, n, l, u):
        v = self.add_vertex(f"{self.name}_{n}")
        x1 = v.add_variable(2)
        x2 = v.add_variable(2)
        v.add_constraints([x1 >= l, x1 <= u])
        v.add_constraints([x2 >= l, x2 <= u])
        v.add_cost(cp.norm2(x2 - x1))
        return v
    
    def add_one_way_door(self, n, m, l, u):
        tail = self.get_vertex(f"{self.name}_{n}")
        head = self.get_vertex(f"{self.name}_{m}")
        e = self.add_edge(tail, head)
        e.add_constraint(tail.variables[1] == head.variables[0])
        e.add_constraint(tail.variables[1] >= l)
        e.add_constraint(tail.variables[1] <= u)
        return e
    
    def add_door(self, n, m, l, u):
        e1 = self.add_one_way_door(n, m, l, u)
        e2 = self.add_one_way_door(m, n, l, u)
        return e1, e2
    
# Helper function that connects layers at given room.
def connect_layers(layer1, layer2, room):
    tail = graph.get_vertex(f"{layer1}_{room}")
    head = graph.get_vertex(f"{layer2}_{room}")
    edge = graph.add_edge(tail, head)
    edge.add_constraint(tail.variables[1] == head.variables[0])

# Helper function that returns the edge binary variable that connects two layers
# through a given room.
def get_binary_variable(layer1, layer2, room):
    tail_name = f"{layer1}_{room}"
    head_name = f"{layer2}_{room}"
    edge = graph.get_edge(tail_name, head_name)
    return ye[graph.edge_index(edge)]

# Initialize empty graph.
graph = GraphOfConvexSets()

# Add one layer for each room that we must visit.
num_layers = len(visit_rooms)
for i in range(num_layers):
    layer = Layer(rooms, doors, i)
    graph.add_disjoint_subgraph(layer)

# Connect top layer to bottom layer in correspondence of first room.
first_room = visit_rooms[0]
first_layer = 0
last_layer = num_layers - 1
connect_layers(last_layer, first_layer, first_room)

# Connect each layer to the next at visit room.
for layer in range(last_layer):
    for room in visit_rooms[1:]:
        connect_layers(layer, layer + 1, room)

# Initialize constraints of the integer linear program.
yv = graph.vertex_binaries()
ye = graph.edge_binaries()
ilp_constraints = []

# Relate edge and vertex binaries.
for i, vertex in enumerate(graph.vertices):
    inc_edges = graph.incoming_edge_indices(vertex)
    out_edges = graph.outgoing_edge_indices(vertex)
    ilp_constraints.append(yv[i] == sum(ye[inc_edges]))
    ilp_constraints.append(yv[i] == sum(ye[out_edges]))

# Force the trajectory to move between layers.
ilp_constraints.append(get_binary_variable(last_layer, first_layer, first_room) == 1)
for room in visit_rooms[1:]:
    flow = sum(get_binary_variable(layer, layer + 1, room) for layer in range(last_layer))
    ilp_constraints.append(flow == 1)

# Solve problem.
graph.solve_from_ilp(ilp_constraints, verbose=True, solver="GUROBI")
print('Problem status:', graph.status)
print('Optimal value:', graph.value)

# Plot solution.
plt.figure(figsize=(6, 3.75))
plt.axis("off")

# Helper function that plots one room.
def plot_room(n, l, u):
    l = np.array(l)
    u = np.array(u)
    d = u - l
    fc = "mistyrose" if n in visit_rooms else "mintcream"
    rect = patches.Rectangle(l, *d, fc=fc, ec="k")
    plt.gca().add_patch(rect)
        
# Helper function that plots one door.
def plot_door(l, u):
    endpoints =  np.array([l, u]).T
    plt.plot(*endpoints, color="mintcream", solid_capstyle="butt")
    plt.plot(*endpoints, color="grey", linestyle=":")
        
# Plot all rooms and doors.
for room in rooms:
    plot_room(*room)
for door in doors:
    plot_door(door[2], door[3])

# Plot optimal trajectory.
for vertex in graph.vertices:
    if np.isclose(vertex.binary_variable.value, 1):
        x1, x2 = vertex.variables
        values = np.array([x1.value, x2.value]).T
        plt.plot(*values, c="b", linestyle="--")
plt.savefig("inspection.pdf", bbox_inches="tight")
plt.show()