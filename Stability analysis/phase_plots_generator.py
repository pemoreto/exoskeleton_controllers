import numpy as np
import matplotlib.pyplot as plt
import os

def compute_phase_portrait(angle_data, velocity_data, grf, time_data,
                           angle_label, velocity_label, title, color,
                           save_dir, version, show_plots=True, joint_index=0):
    """
    Computes the phase portrait and extracts multidimensional Poincaré section data at heel strike events.

    Parameters:
    - angle_data (array): Joint angle values over time, shape (T,) or (T, D).
    - velocity_data (array): Joint angular velocity values over time, shape (T,) or (T, D).
    - grf (array): Ground reaction force values for detecting heel strikes.
    - time_data (array): Time stamps.
    - angle_label (str): Label for the angle axis.
    - velocity_label (str): Label for the velocity axis.
    - title (str): Title for the plot.
    - color (str): Trajectory color.
    - save_dir (str): Path to save the figure.
    - version (str): Suffix for saving the figure.
    - show_plots (bool): Whether to display the plot.
    - joint_index (int): Index of the joint to plot (0=knee, 1=ankle, etc.)

    Returns:
    - tuple: (portrait_angles, portrait_velocities, portrait_times, poincare_points)
    """
    threshold = 0.04
    crossings = np.diff(np.where(grf > threshold, 1, 0))
    surpass_indices = np.where(crossings == 1)[0][2:]

    # Extract Poincaré points (entire state at each heel strike)
    portrait_angles = [angle_data[i, joint_index] for i in surpass_indices if i < len(angle_data)]
    portrait_velocities = [velocity_data[i, joint_index] for i in surpass_indices if i < len(velocity_data)]
    portrait_times = [time_data[i] for i in surpass_indices if i < len(time_data)]

    # Stack into (N, D) array for full state vector (angle + velocity for all joints)
    poincare_points = np.hstack((
        angle_data[surpass_indices, :],
        velocity_data[surpass_indices, :]
    ))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(portrait_angles, portrait_velocities, color='red', label='Poincaré Section')
    plt.plot(angle_data[:, joint_index], velocity_data[:, joint_index], linewidth=1, color=color, label="Trajectory")
    plt.xlabel(f"{angle_label} (Joint {joint_index})")
    plt.ylabel(f"{velocity_label} (Joint {joint_index})")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}_{version}.svg"), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return portrait_angles, portrait_velocities, portrait_times, poincare_points




def compute_poincare_map(poincare_values, axis, angle_label, title, save_dir, version, show_plots=True, dim_index=0):
    """
    Computes and plots the Poincaré first-return map for one dimension of a potentially multidimensional system.

    Parameters:
    - poincare_values (list or ndarray): List or array of N state vectors from Poincaré section (N x D).
    - axis (tuple): Axis limits for the 1D plot (x and y).
    - angle_label (str): Label for the plot (typically angle unit).
    - title (str): Title of the plot.
    - save_dir (str): Directory to save the plot.
    - version (str): Version suffix for saved filename.
    - show_plots (bool): Whether to display the plot.
    - dim_index (int): Which dimension (e.g., 0 for angle, 1 for angular velocity) to plot in 1D map.

    Returns:
    - tuple: (x_values, y_values) for the selected dimension, full_X (N-1 x D), full_Y (N-1 x D)
    """
    poincare_values = np.asarray(poincare_values)

    if poincare_values.ndim == 1:
        poincare_values = poincare_values.reshape(-1, 1)

    if poincare_values.shape[0] < 2:
        print("Not enough data for Poincaré map.")
        return None, None, None, None

    full_X = poincare_values[:-1]
    full_Y = poincare_values[1:]

    x_values = full_X[:, dim_index]
    y_values = full_Y[:, dim_index]

    plt.figure(figsize=(8, 6))
    plt.axline((0, 0), slope=1, color='r', linestyle='--', label='y = x')
    plt.plot(x_values, y_values, 'bo-', label="Return Map")
    plt.xlabel(f'Poincaré Value n [{angle_label}]')
    plt.ylabel(f'Poincaré Value n+1 [{angle_label}]')
    plt.title(title)
    plt.legend()
    plt.xlim(axis)
    plt.ylim(axis)
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}_{version}.svg"), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return x_values, y_values, full_X, full_Y



