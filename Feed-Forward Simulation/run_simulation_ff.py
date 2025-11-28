import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sconetools import sconepy
import time

# Extract simulation name from script filename
simu_name = os.path.splitext(os.path.basename(__file__))[0]

def run_simulation_ff(model, start_actuation, input_hip, input_knee, input_ankle, store_data, duration, version):
    """
    Runs a gait simulation with predefined PID parameters.
    Records some kinetic and kinematic data over the simulation duration.
    Saves simulation results and generates performance summaries.

    Parameters:
        model (object): The SCONE-model.
        start_actuation (float): Time at which knee inputs start being applied.
        input_knee_r (float): Control input for the right knee.
        input_knee_l (float): Control input for the left knee.
        store_data (bool): Whether to save the simulation results.
        duration (float): Total simulation time in seconds.
        version (str): Identifier for the simulation version.

    Returns:
        None
    """

    print('Starting Simulation')
    start_time_sim = time.time()

    # Enable data storage in the model
    measure = model.measure()
    model.set_store_data(True)
    version = str(version)

    results_dir = f"results_{version}"
    os.makedirs(results_dir, exist_ok=True)

    time_span = np.arange(0, duration + 0.01, 0.01)

    # Initialize arrays for joint angles, velocities, and muscle activations
    pelvis_tilt, pelvis_tx, pelvis_ty, hip_flexion_r, knee_r, ankle_r, hip_flexion_l, knee_l, ankle_l = (
        np.zeros_like(time_span) for _ in range(9))
    pelvis_tilt_u, pelvis_tx_u, pelvis_ty_u, hip_flexion_r_u, knee_r_u, ankle_r_u, hip_flexion_l_u, knee_l_u, ankle_l_u = (
        np.zeros_like(time_span) for _ in range(9))
    hamstrings_r, bifemsh_r, glut_max_r, iliopsoas_r, rect_fem_r, vasti_r, gastroc_r, soleus_r, tib_ant_r, hamstrings_l, bifemsh_l, glut_max_l, iliopsoas_l, rect_fem_l, vasti_l, gastroc_l, soleus_l, tib_ant_l = (
        np.zeros_like(time_span) for _ in range(18))

    # Initialize storage arrays
    input_hip_r, input_knee_r, input_ankle_r, input_hip_l, input_knee_l, input_ankle_l, effort = (np.zeros_like(time_span) for _ in range(7))
    grf_r, grf_l = (np.array([]) for _ in range(2))
    input_array = np.zeros(len(model.actuators()))

    for i, t in enumerate(time_span):
        pelvis_tilt[i] = model.dofs()[0].pos()
        pelvis_tx[i] = model.dofs()[1].pos()
        pelvis_ty[i] = model.dofs()[2].pos()
        hip_flexion_r[i] = model.dofs()[3].pos()
        knee_r[i] = model.dofs()[4].pos()
        ankle_r[i] = model.dofs()[5].pos()
        hip_flexion_l[i] = model.dofs()[6].pos()
        knee_l[i] = model.dofs()[7].pos()
        ankle_l[i] = model.dofs()[8].pos()

        pelvis_tilt_u[i] = model.dofs()[0].vel()
        pelvis_tx_u[i] = model.dofs()[1].vel()
        pelvis_ty_u[i] = model.dofs()[2].vel()
        hip_flexion_r_u[i] = model.dofs()[3].vel()
        knee_r_u[i] = model.dofs()[4].vel()
        ankle_r_u[i] = model.dofs()[5].vel()
        hip_flexion_l_u[i] = model.dofs()[6].vel()
        knee_l_u[i] = model.dofs()[7].vel()
        ankle_l_u[i] = model.dofs()[8].vel()

        hamstrings_r[i] = model.muscles()[0].activation()
        bifemsh_r[i] = model.muscles()[1].activation()
        glut_max_r[i] = model.muscles()[2].activation()
        iliopsoas_r[i] = model.muscles()[3].activation()
        rect_fem_r[i] = model.muscles()[4].activation()
        vasti_r[i] = model.muscles()[5].activation()
        gastroc_r[i] = model.muscles()[6].activation()
        soleus_r[i] = model.muscles()[7].activation()
        tib_ant_r[i] = model.muscles()[8].activation()
        hamstrings_l[i] = model.muscles()[9].activation()
        bifemsh_l[i] = model.muscles()[10].activation()
        glut_max_l[i] = model.muscles()[11].activation()
        iliopsoas_l[i] = model.muscles()[12].activation()
        rect_fem_l[i] = model.muscles()[13].activation()
        vasti_l[i] = model.muscles()[14].activation()
        gastroc_l[i] = model.muscles()[15].activation()
        soleus_l[i] = model.muscles()[16].activation()
        tib_ant_l[i] = model.muscles()[17].activation()

        grf_r = np.append(grf_r, model.bodies()[4].contact_force().array()[1])
        grf_l = np.append(grf_l, model.bodies()[7].contact_force().array()[1])

        effort[i] = measure.current_result(model)

        if t == start_actuation:
            # Update PID controllers with the current knee positions and time step (0.01 as example)
            input_array[-6] = input_hip[0]
            input_array[-3] = input_hip[1]
            input_array[-5] = input_knee[0]
            input_array[-2] = input_knee[1]
            input_array[-4] = input_ankle[0]
            input_array[-1] = input_ankle[1]

            # Store the control inputs
            input_hip_r[i] = input_array[-6]
            input_hip_l[i] = input_array[-3]
            input_knee_r[i] = input_array[-5]
            input_knee_l[i] = input_array[-2]
            input_ankle_r[i] = input_array[-4]
            input_ankle_l[i] = input_array[-1]

        # Advance the simulation to the next time step
        model.set_actuator_inputs(input_array)
        model.advance_simulation_to(t)

    # Calculate the error
    final_effort = measure.final_result(model)
    mean_current_effort = np.mean(effort[(start_actuation)*100:])

    print(f"\nMean current results: {mean_current_effort}")
    print(f"Final Result: {final_effort}\n")

    end_time_sim = time.time()
    process_duration = end_time_sim - start_time_sim
    print(f'Simulation ended. Time taken: {process_duration:.2f} seconds')

    # Save results
    repeated_cycles_df = pd.DataFrame({
        'time': time_span,
        'pelvis_tilt': pelvis_tilt,
        'pelvis_tx': pelvis_tx,
        'pelvis_ty': pelvis_ty,
        'hip_flexion_r': hip_flexion_r,
        'knee_angle_r': knee_r,
        'ankle_angle_r': ankle_r,
        'hip_flexion_l': hip_flexion_l,
        'knee_angle_l': knee_l,
        'ankle_angle_l': ankle_l,

        'pelvis_tilt_u': pelvis_tilt_u,
        'pelvis_tx_u': pelvis_tx_u,
        'pelvis_ty_u': pelvis_ty_u,
        'hip_flexion_r_u': hip_flexion_r_u,
        'knee_angle_u_r': knee_r_u,
        'ankle_angle_u_r': ankle_r_u,
        'hip_flexion_l_u': hip_flexion_l_u,
        'knee_angle_u_l': knee_l_u,
        'ankle_angle_u_l': ankle_l_u,

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
        filename = os.path.join(results_dir, f"FF_Simu_results_Data_{version}.xlsx")
        repeated_cycles_df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")

        # Save Simulation results for SCONE
        dirname = f'{simu_name}_{version}_' + model.name()
        filename = model.name()
        model.write_results(dirname, filename)
        print(f'Results written to {dirname}/{filename}; please use SCONE Studio to replay the .sto file.',
              flush=True)



    #----------------------------------------------------------------------

    # Knee Angles Plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(time_span, hip_flexion_r, linewidth=1, color='blue', label='Hip')
    axs[0].plot(time_span, knee_r, linewidth=1, color='red', label='Knee')
    axs[0].plot(time_span, ankle_r, linewidth=1, color='green', label='Ankle')
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel('Angle (Right) (rad)', fontsize=12)
    axs[0].grid(True)
    axs[1].plot(time_span, hip_flexion_l, linewidth=1, color='blue', label='Hip')
    axs[1].plot(time_span, knee_l, linewidth=1, color='red', label='Knee')
    axs[1].plot(time_span, ankle_l, linewidth=1, color='green', label='Ankle')
    axs[1].legend(loc="upper right")
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Angle (Left) (rad)', fontsize=12)
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"Knee_Angle_Trajectory_{version}.svg"), dpi=300)
    plt.show()


    # Input Plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(time_span, input_hip_r, linewidth=1, color='blue', label='Hip')
    axs[0].plot(time_span, input_knee_r, linewidth=1, color='red', label='Knee')
    axs[0].plot(time_span, input_ankle_r, linewidth=1, color='green', label='Ankle')
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel('Input (Right)', fontsize=12)
    axs[0].grid(True)

    axs[1].plot(time_span, input_hip_l, linewidth=1, color='blue', label='Hip')
    axs[1].plot(time_span, input_knee_l, linewidth=1, color='red', label='Knee')
    axs[1].plot(time_span, input_ankle_l, linewidth=1, color='green', label='Ankle')
    axs[1].legend(loc="upper right")
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Input (Left)', fontsize=12)
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"Input_Trajectory_{version}.svg"), dpi=300)
    plt.show()

    txt_filename = os.path.join(results_dir, f"Simu_results_{version}.txt")

    with open(txt_filename, "w") as f:
        f.write(f"Mean current result: {mean_current_effort}\n")
        f.write(f"Model type: {version}\n")
        f.write(f"Duration of calculation: {process_duration}\n")

