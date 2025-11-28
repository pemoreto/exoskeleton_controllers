import numpy as np
from scipy.interpolate import CubicHermiteSpline
import config

"""
Creates periodic reference trajectories for right and left knee angles
using cubic Hermite spline interpolation.

This function takes pre-extracted control point indices and angles (minima & maxima)
and builds a smooth periodic spline by connecting:
1. A middle segment (one full gait cycle),
2. Two edge/transition regions to ensure continuity across cycles.

Steps:
- Uses zero-derivative Hermite splines for smooth transitions.
- Builds two segments for each leg: middle + edges.
- Returns continuous and repeatable joint trajectories for simulation.

Inputs:
- cp_idx_r / cp_idx_l: Indices of control points for right/left knee.
- cp_ang_r / cp_ang_l: Corresponding knee angles at the control points.

Returns:
- y_spline_result_r: Full periodic spline for the right knee.
- y_spline_result_l: Full periodic spline for the left knee.
"""

def create_splines(cp_idx_r, cp_ang_r, cp_idx_l, cp_ang_l):
    # Load the time cycle from the configuration
    time_cycle = config.time_cycle

    # Define derivatives for the middle splines as zero (no slope at control points)
    dydx_middle = np.zeros(len(cp_idx_r))

    # Create Hermite spline for the middle values (right side)
    spline_middle_r = CubicHermiteSpline(time_cycle[cp_idx_r], cp_ang_r, dydx_middle)
    x_middle_r = np.arange(0, time_cycle[-1] + 0.01, 0.01)  # Generate discrete time va lues
    y_middle_r = spline_middle_r(x_middle_r)  # Apply spline function to time values
    # Limit the values to the relevant range
    x_middle_r = x_middle_r[cp_idx_r[0]:cp_idx_r[-1] + 1]
    y_middle_r = y_middle_r[cp_idx_r[0]:cp_idx_r[-1] + 1]

    # Same process for the left side
    dydx_middle = np.zeros(len(cp_idx_l))
    spline_middle_l = CubicHermiteSpline(time_cycle[cp_idx_l], cp_ang_l, dydx_middle)
    x_middle_l = np.arange(0, time_cycle[-1] + 0.01, 0.01)
    y_middle_l = spline_middle_l(x_middle_l)
    x_middle_l = x_middle_l[cp_idx_l[0]:cp_idx_l[-1] + 1]
    y_middle_l = y_middle_l[cp_idx_l[0]:cp_idx_l[-1] + 1]

    # Define derivatives for the edge splines (transition regions)
    dydx_edge = [0, 0]

    # Create the spline for the transition region (right side)
    edge_time_r = np.array([time_cycle[cp_idx_r[-1]], time_cycle[-1] + time_cycle[cp_idx_r[0]]])
    edge_ang_r = cp_ang_r[[-1, 0]]  # Values at endpoints
    spline_edge_r = CubicHermiteSpline(edge_time_r, edge_ang_r, dydx_edge)
    x_edge_r = np.linspace(edge_time_r[0], edge_time_r[1], round((edge_time_r[1]-edge_time_r[0])/ 0.01+1))
    y_edge_r = spline_edge_r(x_edge_r)

    # Same process for the left side
    edge_time_l = np.array([time_cycle[cp_idx_l[-1]], time_cycle[-1] + time_cycle[cp_idx_l[0]]])
    edge_ang_l = cp_ang_l[[-1, 0]]
    spline_edge_l = CubicHermiteSpline(edge_time_l, edge_ang_l, dydx_edge)
    x_edge_l = np.linspace(edge_time_l[0], edge_time_l[1], round((edge_time_l[1]-edge_time_l[0])/ 0.01+1))
    y_edge_l = spline_edge_l(x_edge_l)

    # Split the edge region into two sections (right side)
    split_idx_r = len(time_cycle)-cp_idx_r[-1]-1
    x_edge_1_r = x_edge_r[1:split_idx_r + 1]
    y_edge_1_r = y_edge_r[1:split_idx_r + 1]
    x_edge_2_r = x_edge_r[split_idx_r:-1] - time_cycle[-1]  # Reset time for continuity
    y_edge_2_r = y_edge_r[split_idx_r:-1]
    x_spline_result_r = np.concatenate((x_edge_2_r, x_middle_r, x_edge_1_r))
    y_spline_result_r = np.concatenate((y_edge_2_r, y_middle_r, y_edge_1_r))

    # Split the edge region into two sections (left side)
    split_idx_l = len(time_cycle)-cp_idx_l[-1]-1
    x_edge_1_l = x_edge_l[1:split_idx_l + 1]
    y_edge_1_l = y_edge_l[1:split_idx_l + 1]
    x_edge_2_l = x_edge_l[split_idx_l:-1] - time_cycle[-1]
    y_edge_2_l = y_edge_l[split_idx_l:-1]
    x_spline_result_l = np.concatenate((x_edge_2_l, x_middle_l, x_edge_1_l))
    y_spline_result_l = np.concatenate((y_edge_2_l, y_middle_l, y_edge_1_l))

    # Combine values for the final periodic spline (right side)
    y_spline_result_r = np.concatenate((y_edge_2_r, y_middle_r, y_edge_1_r))

    # Combine values for the final periodic spline (left side)
    y_spline_result_l = np.concatenate((y_edge_2_l, y_middle_l, y_edge_1_l))

    return y_spline_result_r, y_spline_result_l
