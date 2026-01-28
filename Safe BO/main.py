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
from scipy.interpolate import CubicHermiteSpline

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
hip_r_ref = config.hip_r_ref
hip_l_ref = config.hip_l_ref
knee_r_ref = config.knee_r_ref
knee_l_ref = config.knee_l_ref
ankle_r_ref = config.ankle_r_ref
ankle_l_ref = config.ankle_l_ref

# Define simulation time range
time_range = config.time

# Generate the spline interpolation for the entire time array
spline_ref_hip_r, spline_ref_hip_l = create_splines(config.cp_hip_idx_r, config.cp_hip_ang_r, config.cp_hip_idx_l, config.cp_hip_ang_l)
spline_ref_knee_r, spline_ref_knee_l = create_splines(config.cp_knee_idx_r, config.cp_knee_ang_r, config.cp_knee_idx_l, config.cp_knee_ang_l)
spline_ref_ankle_r, spline_ref_ankle_l = create_splines(config.cp_ankle_idx_r, config.cp_ankle_ang_r, config.cp_ankle_idx_l, config.cp_ankle_ang_l)

# Categorize extrema

# Plot categorized extrema points for right and left knee angles
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
# Right Hip Plot
axs[0].plot(time_cycle, spline_ref_hip_r, label='Spline')
axs[0].plot(time_cycle, hip_r_ref, label='Real Curve')
for i in range(len(config.max_hip_idx_r)):
    axs[0].scatter(time_cycle[config.max_hip_idx_r[i]], spline_ref_hip_r[config.max_hip_idx_r[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_hip_idx_r)):
    axs[0].scatter(time_cycle[config.min_hip_idx_r[i]], spline_ref_hip_r[config.min_hip_idx_r[i]], label=f'Min {i}', zorder=5)
axs[0].set_title('Right Hip Angle with Categorized Extrema')
axs[0].grid(True)
axs[0].legend(loc="upper right")

# Left Hip Plot
axs[1].plot(time_cycle, spline_ref_hip_l, label='Spline')
axs[1].plot(time_cycle, hip_l_ref, label='Real Curve')
for i in range(len(config.max_hip_idx_l)):
    axs[1].scatter(time_cycle[config.max_hip_idx_l[i]], spline_ref_hip_l[config.max_hip_idx_l[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_hip_idx_l)):
    axs[1].scatter(time_cycle[config.min_hip_idx_l[i]], spline_ref_hip_l[config.min_hip_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1].set_title('Left Hip Angle with Categorized Extrema')
axs[1].grid(True)
axs[1].legend(loc="upper right")

fig, axs = plt.subplots(2, 1, figsize=(12, 12))
# Right Knee Plot
axs[0].plot(time_cycle, spline_ref_knee_r, label='Spline')
axs[0].plot(time_cycle, knee_r_ref, label='Real Curve')
for i in range(len(config.max_knee_idx_r)):
    axs[0].scatter(time_cycle[config.max_knee_idx_r[i]], spline_ref_knee_r[config.max_knee_idx_r[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_knee_idx_r)):
    axs[0].scatter(time_cycle[config.min_knee_idx_r[i]], spline_ref_knee_r[config.min_knee_idx_r[i]], label=f'Min {i}', zorder=5)
axs[0].set_title('Right Knee Angle with Categorized Extrema')
axs[0].grid(True)
axs[0].legend(loc="upper right")

# Left Knee Plot
axs[1].plot(time_cycle, spline_ref_knee_l, label='Spline')
axs[1].plot(time_cycle, knee_l_ref, label='Real Curve')
for i in range(len(config.max_knee_idx_l)):
    axs[1].scatter(time_cycle[config.max_knee_idx_l[i]], spline_ref_knee_l[config.max_knee_idx_l[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_knee_idx_l)):
    axs[1].scatter(time_cycle[config.min_knee_idx_l[i]], spline_ref_knee_l[config.min_knee_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1].set_title('Left Knee Angle with Categorized Extrema')
axs[1].grid(True)
axs[1].legend(loc="upper right")

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Right Ankle Plot
axs[0].plot(time_cycle, spline_ref_ankle_r, label='Spline')
axs[0].plot(time_cycle, ankle_r_ref, label='Real Curve')
for i in range(len(config.max_ankle_idx_r)):
    axs[0].scatter(time_cycle[config.max_ankle_idx_r[i]], spline_ref_ankle_r[config.max_ankle_idx_r[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_ankle_idx_r)):
    axs[0].scatter(time_cycle[config.min_ankle_idx_r[i]], spline_ref_ankle_r[config.min_ankle_idx_r[i]], label=f'Min {i}', zorder=5)
axs[0].set_title('Right Ankle Angle with Categorized Extrema')
axs[0].grid(True)
axs[0].legend(loc="upper right")

# Left Ankle Plot
axs[1].plot(time_cycle, spline_ref_ankle_l, label='Spline')
axs[1].plot(time_cycle, ankle_l_ref, label='Real Curve')
for i in range(len(config.max_ankle_idx_l)):
    axs[1].scatter(time_cycle[config.max_ankle_idx_l[i]], spline_ref_ankle_l[config.max_ankle_idx_l[i]], label=f'Max {i}', zorder=5)
for i in range(len(config.min_ankle_idx_l)):
    axs[1].scatter(time_cycle[config.min_ankle_idx_l[i]], spline_ref_ankle_l[config.min_ankle_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1].set_title('Left Ankle Angle with Categorized Extrema')
axs[1].grid(True)
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.close()
plt.close()
plt.close()
#plt.show()

# SCONE Simulation Initialization
sconepy.set_log_level(3)
print('SCONE Version', sconepy.version())
sconepy.set_array_dtype_float32()

# Load SCONE simulation model
model_type = 'osim'

par_file_hfd = '0774_0.895_0.880.par'

if model_type == 'hfd':
    model = sconepy.load_model(
        f'Simulation_H0918RS2_actuated/Simulation_H0918RS2_actuated_hfd.scone', par_file_hfd)
elif model_type == 'osim':
    model = sconepy.load_model(
        f'Simulation_H0918RS2_actuated/Simulation_H0918RS2_actuated.scone')
else:
    raise ValueError("model_type must be either 'hfd' or 'osim'.")

# Define search space bounds for Bayesian Optimization

shift_lbounds = [0, -2, -2, 0, -2, 0, -2, 0, -2, 0, -2]
shift_ubounds = [2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0]
#shift_lbounds = [0, -2, -2, 0, -2, 0, -2, 0, -2, 0, -2]
#shift_ubounds = [2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0]

pid_lbounds = [0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0]
pid_ubounds = [3, 1.5, 1.5, 3, 1.5, 1.5, 3, 1.5, 1.5]
#pid_lbounds = [0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0]
#pid_ubounds = [3, 1.5, 1.5, 3, 1.5, 1.5, 3, 1.5, 1.5]

bounds = torch.tensor([pid_lbounds + shift_lbounds,  # Lower bounds
                       pid_ubounds + shift_ubounds],  # Upper bounds
                      dtype=torch.double)

# Add init_points to improve the performance
init_points = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0,	0, 0, 0, 0, 0, 0, 0, 0, 1.04, -0.55, -1.24, 1.04, -0.55, 1.59, -1.24, 1.04, -0.55, 1.59, -1.24],
    [0.12, 0.17, 0.03, 0.12, 0.17, 0.03, 0.12, 0.17, 0.03, 1.70, -0.34, -0.17, 1.70, -0.34, 1.76, -0.17, 1.70, -0.34, 1.76, -0.17]
]

'''
other good points:
    [0,	0,	0, 1.04398478, -0.555836298, 1.599756304,	-1.241550102],
    [0.123221057, 0.177307192, 0.034104545, 1.706721544, -0.348581314, 1.769079804, -0.172509313]

'''

bo = True

if bo:
    # Start Bayesian Optimization (BO) for PID tuning
    print('\n------------ Starting Bayesian Optimization ------------')
    start_time_BO = time.time()
    best_pid_param, best_shifts, version = safeBO_unified(model_type, init_points, 50, bounds, 1, 0.1, True)
    end_time_BO = time.time()
    print(f'\nBO ended. Time taken: {end_time_BO - start_time_BO:.2f} seconds')

    pid = best_pid_param
    shifts = best_shifts
else:
    pid_param_hip = np.array([0, 0, 0])
    pid_param_knee = np.array([1.69953865, 0.07541722, 0.03242994])
    #pid_param_knee = np.array([0, 0, 0])
    pid_param_ankle = np.array([0, 0, 0])
    shifts_hip = np.array([0, 0, 0])
    shifts_knee = np.array([0.7666024,  -0.16908622,  0.82656652, -1.35391402])
    #shifts_knee = np.array([0, 0, 0, 0])
    shifts_ankle = np.array([0, 0, 0, 0])
    version = 'manual'

    pid = np.concatenate((pid_param_hip, pid_param_knee, pid_param_ankle))
    shifts = np.concatenate((shifts_hip, shifts_knee, shifts_ankle))


# Run the SCONE simulation with optimized PID parameters
print('\n ------------ Starting Simulation ------------')
start_time_sim = time.time()
run_simulation(model, pid, shifts, True, time_range, version)
end_time_sim = time.time()
print(f'\nSimulation ended. Time taken: {end_time_sim - start_time_sim:.2f} seconds')

print(f'\n ****************** THE END ******************')
