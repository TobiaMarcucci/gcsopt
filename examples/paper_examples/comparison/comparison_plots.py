"""
This script generates the plot used in the paper to compare the proposed MICP,
the MINCP, and the McCormick formulation. It loads data from the .npy files
stored in the helicopter_flight, school_bus, and camera_positioning
subdirectories. If these files are not available, run the comparison.py script
in each subdirectory to generate them.
"""

import numpy as np
import matplotlib.pyplot as plt

# Time limit used in the tests.
time_limit = 1000

# Initialize helicopter_flight plot.
figsize = (8, 2)
plt.figure(figsize=figsize)

# Load helicopter flight files.
islands = np.load("helicopter_flight/islands.npy")
times = np.load("helicopter_flight/times.npy")
times[times >= time_limit] = np.nan

# Plot helicopter flight runtimes.
plt.plot(islands, times[0], label="MICP", marker="o")
plt.plot(islands, times[1], label="MINCP", marker="^")
plt.plot(islands, times[2], label="McCormick", marker="s")

# Polish helicopter flight plot.
plt.xlabel(r"number of islands")
plt.ylabel("solver time (s)")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower left",
           ncols=3, mode="expand", borderaxespad=0.)
plt.yscale("log")
plt.xlim(islands[0], islands[-1])
plt.ylim(1e-1, 1e3)
plt.xticks(islands)
plt.grid()
plt.show()

# Initialize school bus plot.
plt.figure(figsize=figsize)

# Load school_bus files.
kids = np.load("school_bus/kids.npy")
times = np.load("school_bus/times.npy")
times[times >= time_limit] = np.nan

# Plot school bus runtimes.
plt.plot(kids, times[0], marker="o")
plt.plot(kids, times[1], marker="^")
plt.plot(kids, times[2], marker="s")

# Polish school bus plot.
plt.xlabel(r"number of kids")
plt.ylabel("solver time (s)")
plt.yscale("log")
plt.xlim(kids[0], kids[-1])
plt.ylim(1e-3, 1e3)
plt.xticks(kids)
plt.grid()
plt.show()

# Initialize camera positioning plot.
plt.figure(figsize=figsize)

# Load camera positioning files.
rooms = np.load("camera_positioning/rooms.npy")
times = np.load("camera_positioning/times.npy")
times[times >= time_limit] = np.nan

# Plot camera positioning runtimes.
plt.plot(rooms, times[0], marker="o")
plt.plot(rooms, times[1], marker="^")
plt.plot(rooms, times[2], marker="s")

# Polish camera positioning plot.
plt.xlabel(r"number of rooms")
plt.ylabel("solver time (s)")
plt.yscale("log")
plt.xlim(rooms[0], rooms[-1])
plt.ylim(1e-1, 1e3)
plt.xticks(rooms)
plt.grid()
plt.show()
