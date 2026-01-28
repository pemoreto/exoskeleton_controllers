import numpy as np
from scipy.interpolate import interp1d
from sconetools import sconepy
from gait_cycle_estimation import SimpleGaitCycleEstimator
from create_splines import create_splines
from PID import PID
from track_angles import analyze_gait
import config
import os
import matplotlib.pyplot as plt

"""
Objective and constraint function for PID parameter optimization in SCONE-based gait simulation.

This function is used during optimization to:
1. Run a full gait simulation with a given set of parameters.
2. Calculate the objective (mean effort) and a custom constraint value (based on stability).
3. Save detailed plots and simulation data if required.

Inputs:
- PID gains: Kp, Ki, Kd
- Vertical shifts: v_shift_max_1, v_shift_min_1, v_shift_max_2, v_shift_min_2
- iter (int): Current iteration number (used for file naming).
- save_plots (bool): If True, stores plots and results of the simulation.

Outputs:
- mean_current_effort: Objective value (lower = better).
- constraint_value: Stability measure based on phase portrait analysis.
- com_height[-1]: Final height of center of mass (used to verify dynamic stability).
- max_distance_r: Largest distance in Poincaré return map (indicates instability).
"""

def obj_con_function(model_type, params, iter, save_plots):
    """
    Runs the simulation once and computes:
    - Objective: Metabolic cost (total penalty)
    - Constraint: Torque violation check

    Args:
        params (array-like): [Kp, Ki, Kd, v_shift_max_1, v_shift_min_1, v_shift_max_2, v_shift_min_2]

    Returns:
        tuple: (objective_value, constraint_value)
    """

    # Load SCONE model
    par_file_hfd = '0774_0.895_0.880.par'

    if model_type == 'hfd':
        model = sconepy.load_model(
            f'Simulation_H0918RS2_actuated/Simulation_H0918RS2_actuated_hfd.scone', par_file_hfd)
    elif model_type == 'osim':
        model = sconepy.load_model(
            f'Simulation_H0918RS2_actuated/Simulation_H0918RS2_actuated.scone')
    else:
        raise ValueError("model_type must be either 'hfd' or 'osim'.")
    measure = model.measure()
    model.set_store_data(True)

    # Extract PID parameters
    Kp_hip, Ki_hip, Kd_hip = params[:3]
    Kp_knee, Ki_knee, Kd_knee = params[3:6]
    Kp_ankle, Ki_ankle, Kd_ankle = params[6:9]

    time_cycle = config.time_cycle
    time = config.time
    cp_hip_idx_r = config.cp_hip_idx_r
    cp_hip_ang_r = config.cp_hip_ang_r
    cp_hip_idx_l = config.cp_hip_idx_l
    cp_hip_ang_l = config.cp_hip_ang_l
    cp_knee_idx_r = config.cp_knee_idx_r
    cp_knee_ang_r = config.cp_knee_ang_r
    cp_knee_idx_l = config.cp_knee_idx_l
    cp_knee_ang_l = config.cp_knee_ang_l
    cp_ankle_idx_r = config.cp_ankle_idx_r
    cp_ankle_ang_r = config.cp_ankle_ang_r
    cp_ankle_idx_l = config.cp_ankle_idx_l
    cp_ankle_ang_l = config.cp_ankle_ang_l

    # Extract and apply vertical shifts to hip
    v_shift_hip_max_1, v_shift_hip_min_1, v_shift_hip_min_2 = params[9:12]
    cp_hip_ang_r_shifted = cp_hip_ang_r + np.array([v_shift_hip_min_1, v_shift_hip_min_2, v_shift_hip_max_1])
    cp_hip_ang_l_shifted = cp_hip_ang_l + np.array([v_shift_hip_min_2, v_shift_hip_max_1, v_shift_hip_min_1])

    # Extract and apply vertical shifts to knee
    v_shift_knee_max_1, v_shift_knee_min_1, v_shift_knee_max_2, v_shift_knee_min_2 = params[12:16]
    cp_knee_ang_r_shifted = cp_knee_ang_r + np.array([v_shift_knee_max_1, v_shift_knee_min_1, v_shift_knee_max_2, v_shift_knee_min_2])
    cp_knee_ang_l_shifted = cp_knee_ang_l + np.array([v_shift_knee_min_2, v_shift_knee_max_2, v_shift_knee_min_1, v_shift_knee_max_1])

    # Extract and apply vertical shifts to ankle
    v_shift_ankle_max_1, v_shift_ankle_min_1, v_shift_ankle_max_2, v_shift_ankle_min_2 = params[16:]
    cp_ankle_ang_r_shifted = cp_ankle_ang_r + np.array([v_shift_ankle_max_1, v_shift_ankle_min_1, v_shift_ankle_max_2, v_shift_ankle_min_2])
    cp_ankle_ang_l_shifted = cp_ankle_ang_l + np.array([v_shift_ankle_min_2, v_shift_ankle_max_2, v_shift_ankle_min_1, v_shift_ankle_max_1])

    # Generate hip reference trajectories
    spline_ref_hip_r, spline_ref_hip_l = create_splines(cp_hip_idx_r, cp_hip_ang_r, cp_hip_idx_l, cp_hip_ang_l)
    hip_r_ref_shift, hip_l_ref_shift = create_splines(cp_hip_idx_r, cp_hip_ang_r_shifted, cp_hip_idx_l, cp_hip_ang_l_shifted)

    # Generate knee reference trajectories
    spline_ref_knee_r, spline_ref_knee_l = create_splines(cp_knee_idx_r, cp_knee_ang_r, cp_knee_idx_l, cp_knee_ang_l)
    knee_r_ref_shift, knee_l_ref_shift = create_splines(cp_knee_idx_r, cp_knee_ang_r_shifted, cp_knee_idx_l, cp_knee_ang_l_shifted)

    # Generate ankle reference trajectories
    spline_ref_ankle_r, spline_ref_ankle_l = create_splines(cp_ankle_idx_r, cp_ankle_ang_r, cp_ankle_idx_l, cp_ankle_ang_l)
    ankle_r_ref_shift, ankle_l_ref_shift = create_splines(cp_ankle_idx_r, cp_ankle_ang_r_shifted, cp_ankle_idx_l, cp_ankle_ang_l_shifted)

    # Create linear interpolation functions
    linear_interp_hip_r = interp1d(time_cycle, hip_r_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_hip_l = interp1d(time_cycle, hip_l_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_knee_r = interp1d(time_cycle, knee_r_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_knee_l = interp1d(time_cycle, knee_l_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_ankle_r = interp1d(time_cycle, ankle_r_ref_shift, kind='linear', fill_value='extrapolate')
    linear_interp_ankle_l = interp1d(time_cycle, ankle_l_ref_shift, kind='linear', fill_value='extrapolate')

    # Initialize PID controllers
    pid_hip_r = PID(Kp_hip, Ki_hip, Kd_hip, setpoint=linear_interp_hip_r(0))
    pid_hip_l = PID(Kp_hip, Ki_hip, Kd_hip, setpoint=linear_interp_hip_l(0))
    pid_knee_r = PID(Kp_knee, Ki_knee, Kd_knee, setpoint=linear_interp_knee_r(0))
    pid_knee_l = PID(Kp_knee, Ki_knee, Kd_knee, setpoint=linear_interp_knee_l(0))
    pid_ankle_r = PID(Kp_ankle, Ki_ankle, Kd_ankle, setpoint=linear_interp_ankle_r(0))
    pid_ankle_l = PID(Kp_ankle, Ki_ankle, Kd_ankle, setpoint=linear_interp_ankle_l(0))

    # Preallocate arrays
    hip_ref_r, hip_ref_l, hip_r, hip_l,  knee_ref_r, knee_ref_l, knee_r, knee_l, ankle_ref_r, ankle_ref_l, ankle_r, ankle_l = [np.zeros_like(time) for _ in range(12)]
    input_hip_r, input_hip_l, input_knee_r, input_knee_l, input_ankle_r, input_ankle_l, estimator_r, estimator_l, effort, com_height = [np.zeros_like(time) for _ in range(10)]
    hip_r_u, hip_l_u, knee_r_u, knee_l_u, ankle_r_u, ankle_l_u = [np.zeros_like(time) for _ in range(6)]
    grf_r = np.array([])
    input_array = np.zeros(24)

    # Initialize gait cycle estimators
    gait_cycle_estimator_r = SimpleGaitCycleEstimator(delay=0)
    gait_cycle_estimator_l = SimpleGaitCycleEstimator(delay=0)

    # Simulation loop
    for i, t in enumerate(time):
        # Update ground reaction force for right side
        grf_r = np.append(grf_r, model.bodies()[4].contact_force().array()[1])

        # Read current knee angles and center of mass height
        hip_r[i] = model.dofs()[3].pos()
        knee_r[i] = model.dofs()[4].pos()
        ankle_r[i] = model.dofs()[5].pos()
        hip_l[i] = model.dofs()[6].pos()
        knee_l[i] = model.dofs()[7].pos()
        ankle_l[i] = model.dofs()[8].pos()

        hip_r_u[i] = model.dofs()[3].vel()
        knee_r_u[i] = model.dofs()[4].vel()
        ankle_r_u[i] = model.dofs()[5].vel()
        hip_l_u[i] = model.dofs()[6].vel()
        knee_l_u[i] = model.dofs()[7].vel()
        ankle_l_u[i] = model.dofs()[8].vel()

        com_height[i] = model.com_pos().y

        # Compute current effort
        effort[i] = measure.current_result(model)

        # Estimate gait cycle
        if len(grf_r) > gait_cycle_estimator_r.horizon:
            estimated_gait_cycle_r = gait_cycle_estimator_r(t, grf_r[-gait_cycle_estimator_r.horizon: -1])
            estimated_gait_cycle_l = gait_cycle_estimator_l(t, grf_r[-gait_cycle_estimator_l.horizon: -1])
        else:
            estimated_gait_cycle_r = estimated_gait_cycle_l = 0  # Default if insufficient data

        # Store estimated gait cycle
        estimator_r[i] = estimated_gait_cycle_r
        estimator_l[i] = estimated_gait_cycle_l

        hip_ref_r[i] = linear_interp_hip_r(estimated_gait_cycle_r)
        hip_ref_l[i] = linear_interp_hip_l(estimated_gait_cycle_l)
        knee_ref_r[i] = linear_interp_knee_r(estimated_gait_cycle_r)
        knee_ref_l[i] = linear_interp_knee_l(estimated_gait_cycle_l)
        ankle_ref_r[i] = linear_interp_ankle_r(estimated_gait_cycle_r)
        ankle_ref_l[i] = linear_interp_ankle_l(estimated_gait_cycle_l)

        if t > 2.5:
            # Set PID setpoints using interpolated reference trajectories
            pid_hip_r.setpoint = hip_ref_r[i]
            pid_hip_l.setpoint = hip_ref_l[i]
            pid_knee_r.setpoint = knee_ref_r[i]
            pid_knee_l.setpoint = knee_ref_l[i]
            pid_ankle_r.setpoint = ankle_ref_r[i]
            pid_ankle_l.setpoint = ankle_ref_l[i]

            # Compute control torques
            input_array[-6] = pid_hip_r.update(hip_r[i], 0.01)
            input_array[-5] = pid_hip_l.update(hip_l[i], 0.01)
            input_array[-4] = pid_knee_r.update(knee_r[i], 0.01)
            input_array[-3] = pid_knee_l.update(knee_l[i], 0.01)
            input_array[-2] = pid_ankle_r.update(ankle_r[i], 0.01)
            input_array[-1] = pid_ankle_l.update(ankle_l[i], 0.01)

            input_hip_r[i] = input_array[-6]
            input_hip_l[i] = input_array[-5]
            input_knee_r[i] = input_array[-4]
            input_knee_l[i] = input_array[-3]
            input_ankle_r[i] = input_array[-2]
            input_ankle_l[i] = input_array[-1]

        # Apply actuator inputs and advance simulation
        model.set_actuator_inputs(input_array)
        model.advance_simulation_to(t)

    # Compute final effort and mean effort over last part of simulation
    #final_effort = measure.final_result(model)
    mean_current_effort = np.mean(effort[250:])

    analysis_result_r = analyze_gait(
        knee_r, knee_r_u, time,
        angle_label="Right Knee Angle", velocity_label="Right Knee Angular Velocity",
        title="Phase Portrait", color="red", show_plots=False
    )

    #analysis_result_l = analyze_gait(
    #    knee_l, knee_l_u, time,
    #    angle_label="Left Knee Angle", velocity_label="Left Knee Angular Velocity",
    #    title="Phase Portrait", color="blue", show_plots=False
    #)

    max_distance_r = analysis_result_r["max_distance"]
    is_unstable_r = analysis_result_r["is_unstable"]
    #max_distance_l = analysis_result_l["max_distance"]
    #is_unstable_l = analysis_result_l["is_unstable"]
    poincare_values = analysis_result_r["poincare_values"]
    poincare_times = analysis_result_r["poincare_times"]
    return_map_x = analysis_result_r["return_map_x"]
    return_map_y = analysis_result_r["return_map_y"]

    #print(f"Max Distance (left): {max_distance_l:.4f}")
    #print(f"Unstable Gait (left): {'Yes' if is_unstable_l else 'No'}")

    #constraint_value = -20/3 * (max_distance_r - 0.15) ** 2 + 0.15
    #constraint_value = -11 * max_distance_r + 3.3
    constraint_value = -5 * np.tanh( 13 * ( max_distance_r - 0.3) )

    # Log results
    print(f"\nEvaluating PID Parameters - Hip: {Kp_hip:.4f}, {Ki_hip:.4f}, {Kd_hip:.4f}")
    print(f"\nEvaluating PID Parameters - Knee: {Kp_knee:.4f}, {Ki_knee:.4f}, {Kd_knee:.4f}")
    print(f"\nEvaluating PID Parameters - Ankle: {Kp_ankle:.4f}, {Ki_ankle:.4f}, {Kd_ankle:.4f}")
    print(f"Evaluating Vertical Shifts - Hip: {v_shift_hip_max_1:.4f}, {v_shift_hip_min_1:.4f}, {v_shift_hip_min_2:.4f}")
    print(f"Evaluating Vertical Shifts - Knee: {v_shift_knee_max_1:.4f}, {v_shift_knee_min_1:.4f}, {v_shift_knee_max_2:.4f}, {v_shift_knee_min_2:.4f}")
    print(f"Evaluating Vertical Shifts - Ankle: {v_shift_ankle_max_1:.4f}, {v_shift_ankle_min_1:.4f}, {v_shift_ankle_max_2:.4f}, {v_shift_ankle_min_2:.4f}")
    print(f"Mean Effort: {mean_current_effort:.4f}, Constraint_value: {constraint_value:.4f}")
    print(f"Stability Constraint: {'Unstable Gait' if is_unstable_r else 'Stable Gait'}")
    print(f"Max Distance (right): {max_distance_r:.4f}")
    print(f"CoM last height: {com_height[-1]}\n")

    if save_plots:
        # Hauptordner für die Plots

        stability_true = 'ST' if com_height[-1] > 0 else 'UNST'
        stability_detected = 'ST' if max_distance_r < 0.3 else 'UNST'

        if stability_true != stability_detected:
            tag = f"ERROR_{stability_true}_{stability_detected}"
        else:
            tag = stability_true

        results_dir = "Optimization_Plots"
        iter = str(iter + 1)
        iter_results_dir = os.path.join(results_dir, f"iter_{iter}_{tag}")  # Unterordner für jede Iteration
        os.makedirs(iter_results_dir, exist_ok=True)  # Erstelle den Unterordner für die aktuelle Iteration

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Right knee
        axs[0].plot(time_cycle, spline_ref_knee_r, '--', linewidth=1, color='red')
        axs[0].plot(time_cycle, knee_r_ref_shift, color='blue', linewidth=1)
        axs[0].scatter(time_cycle[cp_knee_idx_r], spline_ref_knee_r[cp_knee_idx_r], color='black', zorder=5, s=30, edgecolor='black')
        axs[0].scatter(time_cycle[cp_knee_idx_r], knee_r_ref_shift[cp_knee_idx_r], color='black', zorder=5, s=30, edgecolor='black')
        axs[0].set_ylabel('Knee Angle (Right) (rad)', fontsize=12)
        axs[0].grid(True)

        # Left knee
        axs[1].plot(time_cycle, spline_ref_knee_l, '--', linewidth=1, color='red')
        axs[1].plot(time_cycle, knee_l_ref_shift, color='blue', linewidth=1)
        axs[1].scatter(time_cycle[cp_knee_idx_l], spline_ref_knee_l[cp_knee_idx_l], color='black', zorder=5, s=30, edgecolor='black')
        axs[1].scatter(time_cycle[cp_knee_idx_l], knee_l_ref_shift[cp_knee_idx_l], color='black', zorder=5, s=30, edgecolor='black')
        axs[1].set_xlabel('Time (s)', fontsize=12)
        axs[1].set_ylabel('Knee Angle (Left) (rad)', fontsize=12)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_spline_ref_cycle.svg"), dpi=300)
        plt.close()

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
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_knee_angle_trajectory.svg"), dpi=300)
        plt.close()

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Right knee input
        axs[0].plot(time, input_knee_r, linewidth=1, color='blue')
        axs[0].set_ylabel('Input (Right)', fontsize=12)
        axs[0].grid(True)

        # Left knee input
        axs[1].plot(time, input_knee_l, linewidth=1, color='blue')
        axs[1].set_xlabel('Time (s)', fontsize=12)
        axs[1].set_ylabel('Input (Left)', fontsize=12)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_input_trajectory.svg"), dpi=300)
        plt.close()

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
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_effort.svg"), dpi=300)
        plt.close()

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
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_gait_estimator.svg"), dpi=300)
        plt.close()

        # Labels and plot settings
        angle_label = "Angle (rad)"
        velocity_label = "Angular Velocity (rad/s)"
        title = "Phase Portrait of Simulated Gait"
        color = "blue"

        plt.figure()
        plt.plot(knee_r, knee_r_u, linewidth=1, color=color)
        plt.scatter(poincare_values, np.zeros_like(poincare_values), color='black', label='Poincaré Section')
        plt.xlabel(angle_label)
        plt.ylabel(velocity_label)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_phase_portrait.svg"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(poincare_times, poincare_values, 'ro-', label='Intersection Angles')
        plt.xlabel(f'Time')
        plt.ylabel(f'Intersection Angle')
        plt.title(f"Intersection Angles Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_Intersections_over_time.svg"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(return_map_x, return_map_y, 'bo-', label="Poincaré First-Return Map")
        plt.plot(return_map_x, return_map_x, 'r-', label="y = x (Reference Line)")
        plt.xlabel(f'Poincaré Value n [{angle_label}]')
        plt.ylabel(f'Poincaré Value n+1 [{angle_label}]')
        plt.title(f'Poincaré First-Return Map for {angle_label}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(iter_results_dir, f"{iter}_poincare_map.svg"), dpi=300)
        plt.close()

        dirname = f'{iter}_safeBO__' + model.name()
        filename = model.name()
        model.write_results(dirname, filename)
        print(f'Results written to {dirname}/{filename}; please use SCONE Studio to replay the .sto file.',
              flush=True)


    return mean_current_effort, constraint_value, com_height[-1], max_distance_r
