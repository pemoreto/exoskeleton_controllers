import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from create_splines import create_splines
from safeBO_unified import safeBO_unified
from run_simulation import run_simulation
import config
from sconetools import sconepy
import time

'''
Main script for SafeBO.

This script performs the following steps:
1. Loads gait cycle data and reference knee angles from config.py.
2. Generates spline curves and visualizes key extrema points (min/max) for both knees.
3. Initializes SCONE and loads the simulation model.
4. Runs SafeBO to tune 7 control parameters (3 PID, 4 shifts).
5. Executes a SCONE simulation with the optimized parameters.
6. Prints the total runtime for both optimization and simulation.

Notes: 
- Sometimes sconepy appears as not imported (red underlined). Just ignore it. It is indeed correctly imported.'''


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(1234)

# Extract the script name for versioning
simu_name = os.path.splitext(os.path.basename(__file__))[0]

# Load the gait cycle data from an Excel file
data = config.data
time_cycle = config.time_cycle
knee_r_ref = config.knee_r_ref
knee_l_ref = config.knee_l_ref

# Define simulation time range
time_range = config.time

# Generate the spline interpolation for the entire time array
spline_ref_r, spline_ref_l = create_splines(config.cp_idx_r, config.cp_ang_r, config.cp_idx_l, config.cp_ang_l)

# Categorize extrema
min_1_r = config.min_idx_r[0]
max_1_r = config.max_idx_r[0]
min_2_r = config.min_idx_r[1]
max_2_r = config.max_idx_r[1]
min_2_l = config.min_idx_l[0]
max_2_l = config.max_idx_l[0]
min_1_l = config.min_idx_l[1]
max_1_l = config.max_idx_l[1]

# Plot categorized extrema points for right and left knee angles
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Right Knee Plot
axs[0].plot(time_cycle, spline_ref_r, label='Real Curve')
axs[0].scatter(time_cycle[max_1_r], spline_ref_r[max_1_r], color='r', label='Max 1', zorder=5)
axs[0].scatter(time_cycle[min_1_r], spline_ref_r[min_1_r], color='b', label='Min 1', zorder=5)
axs[0].scatter(time_cycle[max_2_r], spline_ref_r[max_2_r], color='g', label='Max 2', zorder=5)
axs[0].scatter(time_cycle[min_2_r], spline_ref_r[min_2_r], color='m', label='Min 2', zorder=5)
axs[0].set_title('Right Knee Angle with Categorized Extrema')
axs[0].grid(True)
axs[0].legend()

# Left Knee Plot
axs[1].plot(time_cycle, spline_ref_l, label='Real Curve')
axs[1].scatter(time_cycle[max_1_l], spline_ref_l[max_1_l], color='r', label='Max 1', zorder=5)
axs[1].scatter(time_cycle[min_1_l], spline_ref_l[min_1_l], color='b', label='Min 1', zorder=5)
axs[1].scatter(time_cycle[max_2_l], spline_ref_l[max_2_l], color='g', label='Max 2', zorder=5)
axs[1].scatter(time_cycle[min_2_l], spline_ref_l[min_2_l], color='m', label='Min 2', zorder=5)
axs[1].set_title('Left Knee Angle with Categorized Extrema')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.close()
#plt.show()

# SCONE Simulation Initialization
sconepy.set_log_level(3)
print('SCONE Version', sconepy.version())
sconepy.set_array_dtype_float32()

# Load SCONE simulation model
model = sconepy.load_model('Simulation_H0918RS2_actuated/Simulation_H0918RS2_actuated.scone')

# Define search space bounds for Bayesian Optimization
bounds = torch.tensor([[0.1, 0, 0, 0, -2, 0, -2],  # Lower bounds
                       [3, 1.5, 1.5, 2, 0, 2, 0]],  # Upper bounds
                      dtype=torch.double)

# Add init_points to improve the performance
init_points = [
    [0, 0, 0, 0, 0, 0, 0],
    [0,	0, 0, 1.04398478, -0.555836298, 1.599756304, -1.241550102],
    [0.123221057, 0.177307192, 0.034104545, 1.706721544, -0.348581314, 1.769079804, -0.172509313]
]

'''
other good points:
    [0,	0,	0, 1.04398478, -0.555836298, 1.599756304,	-1.241550102],
    [0.123221057, 0.177307192, 0.034104545, 1.706721544, -0.348581314, 1.769079804, -0.172509313]
'''

# Start Bayesian Optimization (BO) for PID tuning
print('\n------------ Starting Bayesian Optimization ------------')
start_time_BO = time.time()
best_pid_param, best_shifts, version = safeBO_unified(init_points, 30, bounds, 1, 0.1, True)
end_time_BO = time.time()
print(f'\nBO ended. Time taken: {end_time_BO - start_time_BO:.2f} seconds')

# Run the SCONE simulation with optimized PID parameters
print('\n ------------ Starting Simulation ------------')
start_time_sim = time.time()
run_simulation(model, best_pid_param, best_shifts, True, time_range, version)
end_time_sim = time.time()
print(f'\nSimulation ended. Time taken: {end_time_sim - start_time_sim:.2f} seconds')

print(f'\n ****************** THE END ******************')
