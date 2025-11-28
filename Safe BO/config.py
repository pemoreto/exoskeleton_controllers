import pandas as pd
import numpy as np
from scipy.signal import find_peaks

"""
This script extracts characteristic points (local minima and maxima) from one gait cycle
of knee joint angle data (right and left). The extracted indices and angles are used
to create spline-based reference trajectories in the main simulation. 

Inputs:
- 'extracted_cycle.xlsx': Excel file containing time and knee angle data:
    - time_cycle
    - knee_angle_r_cycle
    - knee_angle_l_cycle

Outputs (variables):
- cp_idx_r, cp_idx_l: Indices of local minima and maxima for right and left knee.
- cp_ang_r, cp_ang_l: Corresponding knee angle values at those indices.
- time: Full simulation time array (0 to 20 s, step = 0.01 s)

Note:
- This file is used for preprocessing and storing before optimization.
"""

# Load the data from the Excel file
data = pd.read_excel('extracted_cycle.xlsx')
time_cycle = data['time_cycle'].values
knee_r_ref = data['knee_angle_r_cycle'].values
knee_l_ref = data['knee_angle_l_cycle'].values
time = np.arange(0, 20.01, 0.01)

# Detect extrema for both right and left knee angles
min_idx_r, _ = find_peaks(-knee_r_ref, distance=40)
max_idx_r, _ = find_peaks(knee_r_ref, distance=40)
min_idx_l, _ = find_peaks(-knee_l_ref, distance=40)
max_idx_l, _ = find_peaks(knee_l_ref, distance=40)

# Combine minima and maxima into one array and sort them by their order in the time sequence
cp_idx_r = np.sort(np.concatenate((min_idx_r, max_idx_r)))
cp_idx_l = np.sort(np.concatenate((min_idx_l, max_idx_l)))

cp_ang_r = knee_r_ref[cp_idx_r]
cp_ang_l = knee_l_ref[cp_idx_l]
