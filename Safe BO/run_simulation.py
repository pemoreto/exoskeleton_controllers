import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PID import PID
from gait_cycle_estimation import SimpleGaitCycleEstimator
from create_splines import create_splines
import config
from track_angles import analyze_gait

"""
This script defines the `run_simulation` function, which runs a gait simulation
using PID controllers, gait cycle estimation, and the SCONE model (used to run the final simulation
with optimized control parameters).

Main features:
1. Creates time-varying knee angle references using splines.
2. Applies PID control to both knees with updated setpoints during simulation.
3. Estimates the gait cycle using ground reaction forces.
4. Runs the SCONE simulation step by step and records joint states and forces.
5. Plots reference and actual knee trajectories, control inputs, GRFs, etc.
6. Analyzes the gait using Poincaré maps and phase portraits.
7. Saves plots and data to a results folder for further analysis.

Parameters:
- `model`: SCONE model object.
- `param`: Tuple of PID gains (Kp, Ki, Kd).
- `shifts`: Vertical shifts applied to reference extrema.
- `store_data`: If True, results will be saved as Excel and .sto files.
- `time_span`: Time vector for simulation.
- `version`: Identifier for result files and folders.

Note:
- This function is called from `main.py` after the Bayesian optimization.
- All plots are saved in the `results_<version>` folder.
"""


# Extract simulation name from script filename
simu_name = os.path.splitext(os.path.basename(__file__))[0]


def run_simulation(model, param, shifts, store_data, time_span, version):
    """
    Runs a gait simulation with PID control and gait cycle estimation.

    Parameters:
        model (object): Simulation model.
        param (tuple): PID controller parameters (Kp, Ki, Kd).
        shifts (list): Vertical shift adjustments for extrema points.
        store_data (bool): Whether to store results.
        time_span (array): Time vector for simulation.
        version (str): Simulation version identifier.

    Returns:
        None
    """

    # Enable data storage in the model
    measure = model.measure()
    model.set_store_data(True)
    results_dir = f"results_{version}"
    os.makedirs(results_dir, exist_ok=True)

    time_cycle = config.time_cycle
    time = config.time
    cp_idx_r = config.cp_idx_r
    cp_ang_r = config.cp_ang_r
    cp_idx_l = config.cp_idx_l
    cp_ang_l = config.cp_ang_l

    # Initialize arrays for joint angles, velocities, and muscle activations
    pelvis_tilt, pelvis_tx, pelvis_ty, hip_flexion_r, knee_r, ankle_angle_r, hip_flexion_l, knee_l, ankle_angle_l = (
        np.zeros_like(time_span) for _ in range(9)
    )
    pelvis_tilt_u, pelvis_tx_u, pelvis_ty_u, hip_flexion_r_u, knee_r_u, ankle_angle_r_u, hip_flexion_l_u, knee_l_u, ankle_angle_l_u = (
        np.zeros_like(time_span) for _ in range(9)
    )
    hamstrings_r, bifemsh_r, glut_max_r, iliopsoas_r, rect_fem_r, vasti_r, gastroc_r, soleus_r, tib_ant_r, \
        hamstrings_l, bifemsh_l, glut_max_l, iliopsoas_l, rect_fem_l, vasti_l, gastroc_l, soleus_l, tib_ant_l = (
        np.zeros_like(time_span) for _ in range(18)
    )

    # Extract PID gains
    Kp, Ki, Kd = param

    # Extract vertical shift parameters
    v_shift_max_1, v_shift_min_1, v_shift_max_2, v_shift_min_2 = shifts[:4]

    # Apply vertical shifts to extrema
    cp_ang_r_shifted = np.copy(config.cp_ang_r)
    cp_ang_r_shifted[0] += v_shift_min_1
    cp_ang_r_shifted[1] += v_shift_max_1
    cp_ang_r_shifted[2] += v_shift_min_2
    cp_ang_r_shifted[3] += v_shift_max_2

    cp_ang_l_shifted = np.copy(config.cp_ang_l)
    cp_ang_l_shifted[0] += v_shift_min_2
    cp_ang_l_shifted[1] += v_shift_max_2
    cp_ang_l_shifted[2] += v_shift_min_1
    cp_ang_l_shifted[3] += v_shift_max_1

    spline_ref_r, spline_ref_l = create_splines(cp_idx_r, cp_ang_r, cp_idx_l, cp_ang_l)

    # Create splines for the knee reference trajectories
    knee_r_ref_shift, knee_l_ref_shift = create_splines(cp_idx_r, cp_ang_r_shifted, cp_idx_l, cp_ang_l_shifted)

    # Interpolation functions for reference knee trajectories
    linear_interp_knee_r = interp1d(time_cycle, knee_r_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_knee_l = interp1d(time_cycle, knee_l_ref_shift, kind='linear', fill_value='extrapolate')

    # Initialize PID controllers
    pid_r = PID(Kp, Ki, Kd, setpoint=linear_interp_knee_r(0))
    pid_l = PID(Kp, Ki, Kd, setpoint=linear_interp_knee_l(0))

    # Initialize storage arrays
    knee_ref_r, knee_ref_l, input_r, input_l, estimator_r, estimator_l, effort = (np.zeros_like(time) for _ in range(7))
    grf_r, grf_l = (np.array([]) for _ in range(2))
    input_array = np.zeros(20)

    # Initialize gait cycle estimators
    gait_cycle_estimator_r = SimpleGaitCycleEstimator(delay=0)
    gait_cycle_estimator_l = SimpleGaitCycleEstimator(delay=0)

    for i, t in enumerate(time):
        knee_r[i] = model.dofs()[4].pos()
        knee_l[i] = model.dofs()[7].pos()

        knee_r_u[i] = model.dofs()[4].vel()
        knee_l_u[i] = model.dofs()[7].vel()

        grf_r = np.append(grf_r, model.bodies()[4].contact_force().array()[1])
        grf_l = np.append(grf_l, model.bodies()[7].contact_force().array()[1])

        effort[i] = measure.current_result(model)

        # Estimate gait cycle based on ground reaction force
        if len(grf_r) > gait_cycle_estimator_r.horizon:
            estimated_gait_cycle_r = gait_cycle_estimator_r(t, grf_r[-gait_cycle_estimator_r.horizon: -1])
            estimated_gait_cycle_l = gait_cycle_estimator_l(t, grf_r[-gait_cycle_estimator_l.horizon: -1])
        else:
            estimated_gait_cycle_r = 0  # Initial assumption if insufficient data
            estimated_gait_cycle_l = 0

        # Store the estimated gait cycle
        estimator_r[i] = estimated_gait_cycle_r
        estimator_l[i] = estimated_gait_cycle_l

        knee_ref_r[i] = linear_interp_knee_r(estimated_gait_cycle_r)
        knee_ref_l[i] = linear_interp_knee_l(estimated_gait_cycle_l)

        if t > 2.5:
            # Set PID setpoints based on linear interpolation of the reference trajectories at the estimated gait cycle
            pid_r.setpoint = knee_ref_r[i]
            pid_l.setpoint = knee_ref_l[i]

            # Update PID controllers with the current knee positions and time step (0.01 as example)
            input_array[-2] = pid_r.update(knee_r[i], 0.01)
            input_array[-1] = pid_l.update(knee_l[i], 0.01)

            # Store the control inputs
            input_r[i] = input_array[-2]
            input_l[i] = input_array[-1]

        # Advance the simulation to the next time step
        model.set_actuator_inputs(input_array)
        model.advance_simulation_to(t)

    # Calculate error between simulated and reference knee trajectories
    tot_error = np.sum(np.abs(knee_r - knee_ref_r)) + np.sum(np.abs(knee_l - knee_ref_l))

    print(f"\nPID Parameters: {param}")
    print(f"Vertical Shifts: {shifts[:4]}")
    print(f"Mean current results: {np.mean(effort[250:])}")
    print(f"Tot Error (simulation-ref): {tot_error}\n")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Right knee
    axs[0].plot(time_cycle, spline_ref_r, '--', linewidth=1, color='red')
    axs[0].plot(time_cycle, knee_r_ref_shift, color='blue', linewidth=1)
    axs[0].scatter(time_cycle[cp_idx_r], spline_ref_r[cp_idx_r], color='black', zorder=5, s=30, edgecolor='black')
    axs[0].scatter(time_cycle[cp_idx_r], knee_r_ref_shift[cp_idx_r], color='black', zorder=5, s=30, edgecolor='black')
    axs[0].set_ylabel('Knee Angle (Right) (rad)', fontsize=12)
    axs[0].grid(True)

    # Left knee
    axs[1].plot(time_cycle, spline_ref_l, '--', linewidth=1, color='red')
    axs[1].plot(time_cycle, knee_l_ref_shift, color='blue', linewidth=1)
    axs[1].scatter(time_cycle[cp_idx_l], spline_ref_l[cp_idx_l], color='black', zorder=5, s=30, edgecolor='black')
    axs[1].scatter(time_cycle[cp_idx_l], knee_l_ref_shift[cp_idx_l], color='black', zorder=5, s=30, edgecolor='black')
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Knee Angle (Left) (rad)', fontsize=12)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"spline_ref_cycle_{version}.png"), dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Right knee angle trajectory
    axs[0].plot(time, knee_r, linewidth=1, color='red')
    axs[0].plot(time, knee_ref_r, 'b--', linewidth=1)
    axs[0].set_ylabel('Knee Angle (Right) (rad)', fontsize=12)
    axs[0].grid(True)

    # Left knee angle trajectory
    axs[1].plot(time, knee_l, linewidth=1, color='red')
    axs[1].plot(time, knee_ref_l, 'b--', linewidth=1)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Knee Angle (Left) (rad)', fontsize=12)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"knee_angle_trajectory_{version}.svg"), dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Right knee input
    axs[0].plot(time, input_r, linewidth=1, color='blue')
    axs[0].set_ylabel('Input (Right)', fontsize=12)
    axs[0].grid(True)

    # Left knee input
    axs[1].plot(time, input_l, linewidth=1, color='blue')
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Input (Left)', fontsize=12)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"input_trajectory_{version}.svg"), dpi=300)
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Right knee angle trajectory
    axs[0].plot(time, knee_ref_r, linewidth=1, color='blue')
    axs[0].set_ylabel('Knee Angle (Right)', fontsize=12)
    axs[0].grid(True)

    # Left knee angle trajectory
    axs[1].plot(time, knee_ref_l, linewidth=1, color='blue')
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Knee Angle (Left)', fontsize=12)
    axs[1].grid(True)

    # Effort
    axs[2].plot(time, effort, '--', linewidth=1, color='blue')
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Effort', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"effort_{version}.svg"), dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Gait Cycle Estimator
    axs[0].plot(time, estimator_r, linewidth=1, color='blue')
    axs[0].plot(time, estimator_l, linewidth=1, color='red')
    axs[0].set_ylabel('Gait Cycle Estimator (%)', fontsize=12)
    axs[0].grid(True)

    # Ground Reaction Force (GRF)
    axs[1].plot(time, grf_r, linewidth=1, color='blue')
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Force (N)', fontsize=12)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"gait_estimator_{version}.svg"), dpi=300)
    plt.show()

    # Labels and plot settings
    angle_label = "Angle (rad)"
    velocity_label = "Angular Velocity (rad/s)"
    title = "Phase Portrait of Simulated Gait"
    color = "blue"

    # Perform gait analysis
    results = analyze_gait(knee_r, knee_r_u, time, angle_label, velocity_label, title, color,
                           show_plots=True)
    poincare_values = results["poincare_values"]
    poincare_times = results["poincare_times"]
    return_map_x = results["return_map_x"]
    return_map_y = results["return_map_y"]

    plt.figure()
    plt.plot(knee_r, knee_r_u, linewidth=1, color=color)
    plt.scatter(poincare_values, np.zeros_like(poincare_values), color='black', label='Poincaré Section')
    plt.xlabel(angle_label)
    plt.ylabel(velocity_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"phase_portrait_{version}.svg"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(poincare_times, poincare_values, 'ro-', label='Intersection Angles')
    plt.xlabel(f'Time')
    plt.ylabel(f'Intersection Angle')
    plt.title(f"Intersection Angles Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"Intersections_over_time_{version}.svg"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(return_map_x, return_map_y, 'bo-', label="Poincaré First-Return Map")
    plt.plot(return_map_x, return_map_x, 'r-', label="y = x (Reference Line)")
    plt.xlabel(f'Poincaré Value n [{angle_label}]')
    plt.ylabel(f'Poincaré Value n+1 [{angle_label}]')
    plt.title(f'Poincaré First-Return Map for {angle_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"poincare_map_{version}.svg"), dpi=300)
    plt.close()

    repeated_cycles_df = pd.DataFrame({
        'time': time,
        'pelvis_tilt': pelvis_tilt,
        'pelvis_tx': pelvis_tx,
        'pelvis_ty': pelvis_ty,
        'hip_flexion_r': hip_flexion_r,
        'knee_angle_r': knee_r,
        'ankle_angle_r': ankle_angle_r,
        'hip_flexion_l': hip_flexion_l,
        'knee_angle_l': knee_l,
        'ankle_angle_l': ankle_angle_l,

        'pelvis_tilt_u': pelvis_tilt_u,
        'pelvis_tx_u': pelvis_tx_u,
        'pelvis_ty_u': pelvis_ty_u,
        'hip_flexion_r_u': hip_flexion_r_u,
        'knee_angle_u_r': knee_r_u,
        'ankle_angle_u_r': ankle_angle_r_u,
        'hip_flexion_l_u': hip_flexion_l_u,
        'knee_angle_u_l': knee_l_u,
        'ankle_angle_u_l': ankle_angle_l_u,

        'grf_r': grf_r,
        'grf_l': grf_l,

        'hamstrings_r': hamstrings_r,
        'bifemsh_r': bifemsh_r,
        'glut_max_r': glut_max_r,
        'iliopsoas_r': iliopsoas_r,
        'rect_fem_r': rect_fem_r,
        'vasti_r': vasti_r,
        'gastroc_r': gastroc_r,
        'soleus_r': soleus_r,
        'tib_ant_r': tib_ant_r,
        'hamstrings_l': hamstrings_l,
        'bifemsh_l': bifemsh_l,
        'glut_max_l': glut_max_l,
        'iliopsoas_l': iliopsoas_l,
        'rect_fem_l': rect_fem_l,
        'vasti_l': vasti_l,
        'gastroc_l': gastroc_l,
        'soleus_l': soleus_l,
        'tib_ant_l': tib_ant_l
    })

    # Save DataFrame
    if store_data:
        # Save excel file
        filename = os.path.join(results_dir, f"Simu_results_Data_{version}.xlsx")
        repeated_cycles_df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")

        # Save Simulation results for SCONE
        dirname = f'{simu_name}_{version}_' + model.name()
        filename = model.name()
        model.write_results(dirname, filename)
        print(f'Results written to {dirname}/{filename}; please use SCONE Studio to replay the .sto file.',
              flush=True)

