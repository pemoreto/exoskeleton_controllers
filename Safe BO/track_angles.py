import numpy as np
import matplotlib.pyplot as plt

"""
Poincaré Analysis Tools for Gait Cycles

This module provides functions for analyzing cyclic movements (e.g., gait cycles)
using Poincaré section methods. It includes:

- Phase portrait plotting with Poincaré section detection
- Visualization of the Poincaré first-return map
- Plotting the evolution of intersection angles over time
- Computing the maximal fluctuation to assess stability
- An integrated gait analysis function for streamlined evaluation

These tools are useful for quantitatively assessing the stability and periodicity
of gait cycles.

Functions:
- plot_phase_portrait(...)
- poincare_first_return_map(...)
- plot_intersection_angles(...)
- compute_max_distance(...)
- analyze_gait(...)
"""

def poincare_first_return_map(poincare_values, angle_label, show_plots=True):
    """
    Plots the Poincaré first-return map, showing how intersection values evolve over time.

    Parameters:
    - poincare_values (list): Values of the Poincaré section.
    - angle_label (str): Label for the angle used in the plot.
    - show_plots (bool): Whether to display the plot.

    Returns:
    - tuple: (x_values, y_values) representing return map points.
    """
    if len(poincare_values) < 2:
        print("Not enough data points for the first-return map.")
        return None

    x_values = poincare_values[:-1]
    y_values = poincare_values[1:]

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, 'bo-', label="Poincaré First-Return Map")
        plt.plot(x_values, x_values, 'r-', label="y = x (Reference Line)")
        plt.xlabel(f'Poincaré Value n [{angle_label}]')
        plt.ylabel(f'Poincaré Value n+1 [{angle_label}]')
        plt.title(f'Poincaré First-Return Map for {angle_label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return x_values, y_values

def plot_intersection_angles(time_points, poincare_values, angle_label, show_plots=True):
    """
    Plots the intersection angles over time.

    Parameters:
    - time_points (list): Time instances of Poincaré intersections.
    - poincare_values (list): Corresponding intersection angles.
    - angle_label (str): Label for the angle used in the plot.
    - show_plots (bool): Whether to display the plot.

    Returns:
    - tuple: (time_points, poincare_values) for further processing.
    """
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(time_points, poincare_values, 'ro-', label='Intersection Angles')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Intersection Angle [{angle_label}]')
        plt.title('Evolution of Intersection Angles Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return time_points, poincare_values

def compute_max_distance(poincare_values, angle_label):
    """
    Calculates the maximal fluctuation of intersection angles.

    Parameters:
    - poincare_values (list): Values of the Poincaré section.
    - angle_label (str): Label for the angle used in the plot.

    Returns:
    - tuple: (max_distance, is_unstable), where:
        - max_distance (float): Maximum difference between intersection angles.
        - is_unstable (bool): True if the system is unstable.
    """
    if len(poincare_values) == 0:
        print(f"No Poincaré values found for {angle_label}.")
        return None, False

    max_distance = np.abs(np.max(poincare_values) - np.min(poincare_values))
    is_unstable = max_distance > 0.3

    #print(f'Maximum fluctuation of intersection angles for {angle_label}: {max_distance:.4f} degrees')

    #if is_unstable:
    #    print(f'Warning: The gait cycle is UNSTABLE for {angle_label}')

    return max_distance, is_unstable

def plot_phase_portrait(angle_data, velocity_data, time_data, angle_label, velocity_label, title, color, show_plots=True):
    """
    Plots the phase portrait and extracts Poincaré section data.

    Parameters:
    - angle_data (array): Array of angle values.
    - velocity_data (array): Array of corresponding angular velocities.
    - time_data (array): Time instances corresponding to angle data.
    - angle_label (str): Label for the angle axis.
    - velocity_label (str): Label for the velocity axis.
    - title (str): Title of the phase portrait.
    - color (str): Color for the trajectory.
    - show_plots (bool): Whether to display the plot.

    Returns:
    - tuple: (poincare_values, poincare_times) for further analysis.
    """
    zero_crossings = np.where((velocity_data[:-1] < 0) & (velocity_data[1:] >= 0))[0]
    poincare_values = []
    poincare_times = []

    for idx in zero_crossings:
        v1, v2 = velocity_data[idx], velocity_data[idx + 1]
        a1, a2 = angle_data[idx], angle_data[idx + 1]
        t1, t2 = time_data[idx], time_data[idx + 1]

        if v1 < 0 and v2 >= 0:
            interpolated_value = a1 + (a2 - a1) * (-v1 / (v2 - v1))
            interpolated_time = t1 + (t2 - t1) * (-v1 / (v2 - v1))
            if interpolated_value < -0.6:
                poincare_values.append(interpolated_value)
                poincare_times.append(interpolated_time)

    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(angle_data, velocity_data, linewidth=1, color=color)
        plt.scatter(poincare_values, np.zeros_like(poincare_values), color='black', label='Poincaré Section')
        plt.xlabel(angle_label)
        plt.ylabel(velocity_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return poincare_values, poincare_times

def analyze_gait(angle_data, velocity_data, time_data, angle_label, velocity_label, title, color, show_plots=True):
    """
    Runs a full gait analysis including:
    - Phase portrait generation
    - Poincaré section extraction
    - Poincaré first-return map
    - Intersection angle evolution
    - Maximum fluctuation computation

    Parameters:
    - angle_data (array): Angle values over time.
    - velocity_data (array): Angular velocity values.
    - time_data (array): Corresponding time values.
    - angle_label (str): Label for angle axis.
    - velocity_label (str): Label for velocity axis.
    - title (str): Title for phase portrait.
    - color (str): Color of the trajectory line.
    - show_plots (bool): Whether to display plots.

    Returns:
    - dict: Dictionary containing:
        - poincare_values (list): Poincaré section angles.
        - poincare_times (list): Corresponding times.
        - return_map_x (list): First-return map x-values.
        - return_map_y (list): First-return map y-values.
        - max_distance (float): Maximum fluctuation in intersection angles.
        - is_unstable (bool): Indicates gait stability.
    """
    poincare_values, poincare_times = plot_phase_portrait(angle_data, velocity_data, time_data, angle_label, velocity_label, title, color, show_plots)
    return_map_x, return_map_y = poincare_first_return_map(poincare_values, angle_label, show_plots)
    plot_intersection_angles(poincare_times, poincare_values, angle_label, show_plots)
    max_distance, is_unstable = compute_max_distance(poincare_values, angle_label)

    return {
        "poincare_values": poincare_values,
        "poincare_times": poincare_times,
        "return_map_x": return_map_x,
        "return_map_y": return_map_y,
        "max_distance": max_distance,
        "is_unstable": is_unstable
    }
