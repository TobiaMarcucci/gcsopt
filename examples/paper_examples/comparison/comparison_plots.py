"""
This script generates the plot used in the paper to compare the proposed MICP,
the MINCP, and the McCormick formulation. It loads data from the .npy files
stored in the spp, tsp, and mstp subdirectories. If these files are not
available, run the comparison.py script in each subdirectory to generate them.
"""

import numpy as np
import matplotlib.pyplot as plt

# Time limit used in the tests.
time_limit = 1000

# Initialize SPP plot.
figsize = (8, 2)
plt.figure(figsize=figsize)

# Load SPP files.
islands_spp = np.load("spp/islands.npy")
times_spp = np.load("spp/times.npy")
times_spp[times_spp >= time_limit] = np.nan

# Plot SPP runtimes.
plt.plot(islands_spp, times_spp[0], label="MICP", marker="o")
plt.plot(islands_spp, times_spp[1], label="MINCP", marker="^")
plt.plot(islands_spp, times_spp[2], label="McCormick", marker="s")

# Polish SPP plot.
plt.xlabel(r"number of islands")
plt.ylabel("solver time (s)")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower left",
           ncols=3, mode="expand", borderaxespad=0.)
plt.yscale("log")
plt.xlim(islands_spp[0], islands_spp[-1])
plt.ylim(1e-1, 1e3)
plt.xticks(islands_spp)
plt.grid()
plt.show()

# Initialize TSP plot.
plt.figure(figsize=figsize)

# Load TSP files.
kids_tsp = np.load("tsp/kids.npy")
times_tsp = np.load("tsp/times.npy")
times_tsp[times_tsp >= time_limit] = np.nan

# Plot TSP runtimes.
plt.plot(kids_tsp, times_tsp[0], marker="o")
plt.plot(kids_tsp, times_tsp[1], marker="^")
plt.plot(kids_tsp, times_tsp[2], marker="s")

# Polish TSP plot.
plt.xlabel(r"number of kids")
plt.ylabel("solver time (s)")
plt.yscale("log")
plt.xlim(kids_tsp[0], kids_tsp[-1])
plt.ylim(1e-3, 1e3)
plt.xticks(kids_tsp)
plt.grid()
plt.show()

# Initialize MSTP plot.
plt.figure(figsize=figsize)

# Load MSTP files.
rooms_mstp = np.load("mstp/rooms.npy")
times_mstp = np.load("mstp/times.npy")
times_mstp[times_mstp >= time_limit] = np.nan

# Plot MSTP runtimes.
plt.plot(rooms_mstp, times_mstp[0], marker="o")
plt.plot(rooms_mstp, times_mstp[1], marker="^")
plt.plot(rooms_mstp, times_mstp[2], marker="s")

# Polish MSTP plot.
plt.xlabel(r"number of rooms")
plt.ylabel("solver time (s)")
plt.yscale("log")
plt.xlim(rooms_mstp[0], rooms_mstp[-1])
plt.ylim(1e-1, 1e3)
plt.xticks(rooms_mstp)
plt.grid()
plt.show()