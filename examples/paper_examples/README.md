This folder collects the scripts necessary to reproduce the results presented in the paper *A Unified and Scalable Method for Optimization over Graphs of Convex Sets* by Tobia Marcucci.

The results in the paper are obtained using `gurobi 12.0.3` and `cvxpy 1.7.4`.

## Available Examples

The following scripts reproduce the examples from Section 8 of the paper:

- `helicopter_flight.py` reproduces the SPP in GCS in Section 8.1,  
- `school_bus.py` reproduces the TSP in GCS in Section 8.2,  
- `camera_positioning.py` reproduces the MSTP in GCS in Section 8.3,  
- `circle_cover.py` reproduces the FLP in GCS in Section 8.4.  

## Comparison in Section 8.5

The scripts for Section 8.5 are located in the `comparison` subfolder.

### Generating the data

First run the `comparison.py` script in each of the following directories:

- `comparison/helicopter_flight`  
- `comparison/school_bus`  
- `comparison/camera_positioning`  

These scripts generate the comparison data and save them as `.npy` files.  

The run the `comparison_plots.py` script to reproduce the comparison plots.

**Note:** Running the comparison scripts may take several hours.
Precomputed `.npy` files are already included in each folder.
