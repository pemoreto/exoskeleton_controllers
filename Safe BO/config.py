import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
new = True
if new:
    data = pd.read_excel('extracted_cycle_new.xlsx')
    time_cycle = data['time'].values
    hip_r_ref = data['hip_flexion_r'].values
    hip_l_ref = data['hip_flexion_l'].values
    knee_r_ref = data['knee_angle_r'].values
    knee_l_ref = data['knee_angle_l'].values
    ankle_r_ref = data['ankle_angle_r'].values
    ankle_l_ref = data['ankle_angle_l'].values
else:
    data = pd.read_excel('extracted_cycle.xlsx')
    time_cycle = data['time_cycle'].values
    hip_r_ref = data['hip_flexion_r_cycle'].values
    hip_l_ref = data['hip_flexion_l_cycle'].values
    knee_r_ref = data['knee_angle_r_cycle'].values
    knee_l_ref = data['knee_angle_l_cycle'].values
    ankle_r_ref = data['ankle_angle_r_cycle'].values
    ankle_l_ref = data['ankle_angle_l_cycle'].values
time = np.arange(0, 20.01, 0.01)

# Detect extrema for right hip angles
temp_min, _ = find_peaks(np.concatenate((-hip_r_ref, -hip_r_ref, -hip_r_ref)), distance=1)
min_hip_idx_r = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((hip_r_ref, hip_r_ref, hip_r_ref)), distance=1)
max_hip_idx_r = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((hip_r_ref, hip_r_ref, hip_r_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((hip_r_ref, hip_r_ref, hip_r_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Right Hip Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, hip_r_ref)
for i in range(len(min_hip_idx_r)):
    axs[1, 0].scatter(time_cycle[min_hip_idx_r[i]], hip_r_ref[min_hip_idx_r[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Right Hip Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((hip_r_ref, hip_r_ref, hip_r_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((hip_r_ref, hip_r_ref, hip_r_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Right Hip Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, hip_r_ref)
for i in range(len(max_hip_idx_r)):
    axs[1, 1].scatter(time_cycle[max_hip_idx_r[i]], hip_r_ref[max_hip_idx_r[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Right Hip Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Detect extrema for left hip angles
temp_min, _ = find_peaks(np.concatenate((-hip_l_ref, -hip_l_ref, -hip_l_ref)), distance=1)
min_hip_idx_l = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((hip_l_ref, hip_l_ref, hip_l_ref)), distance=1)
max_hip_idx_l = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((hip_l_ref, hip_l_ref, hip_l_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((hip_l_ref, hip_l_ref, hip_l_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Left Hip Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, hip_l_ref)
for i in range(len(min_hip_idx_l)):
    axs[1, 0].scatter(time_cycle[min_hip_idx_l[i]], hip_l_ref[min_hip_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Left Hip Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((hip_l_ref, hip_l_ref, hip_l_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((hip_l_ref, hip_l_ref, hip_l_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Left Hip Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, hip_l_ref)
for i in range(len(max_hip_idx_l)):
    axs[1, 1].scatter(time_cycle[max_hip_idx_l[i]], hip_l_ref[max_hip_idx_l[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Left Hip Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Detect extrema for right knee angles
temp_min, _ = find_peaks(np.concatenate((-knee_r_ref, -knee_r_ref, -knee_r_ref)), distance=1)
min_knee_idx_r = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((knee_r_ref, knee_r_ref, knee_r_ref)), distance=1)
max_knee_idx_r = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((knee_r_ref, knee_r_ref, knee_r_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((knee_r_ref, knee_r_ref, knee_r_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Right Knee Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, knee_r_ref)
for i in range(len(min_knee_idx_r)):
    axs[1, 0].scatter(time_cycle[min_knee_idx_r[i]], knee_r_ref[min_knee_idx_r[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Right Knee Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((knee_r_ref, knee_r_ref, knee_r_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((knee_r_ref, knee_r_ref, knee_r_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Right Knee Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, knee_r_ref)
for i in range(len(max_knee_idx_r)):
    axs[1, 1].scatter(time_cycle[max_knee_idx_r[i]], knee_r_ref[max_knee_idx_r[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Right Knee Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Detect extrema for left knee angles
temp_min, _ = find_peaks(np.concatenate((-knee_l_ref, -knee_l_ref, -knee_l_ref)), distance=1)
min_knee_idx_l = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((knee_l_ref, knee_l_ref, knee_l_ref)), distance=1)
max_knee_idx_l = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((knee_l_ref, knee_l_ref, knee_l_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((knee_l_ref, knee_l_ref, knee_l_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Left Knee Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, knee_l_ref)
for i in range(len(min_knee_idx_l)):
    axs[1, 0].scatter(time_cycle[min_knee_idx_l[i]], knee_l_ref[min_knee_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Left Knee Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((knee_l_ref, knee_l_ref, knee_l_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((knee_l_ref, knee_l_ref, knee_l_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Left Knee Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, knee_l_ref)
for i in range(len(max_knee_idx_l)):
    axs[1, 1].scatter(time_cycle[max_knee_idx_l[i]], knee_l_ref[max_knee_idx_l[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Left Knee Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Detect extrema for right ankle angles
temp_min, _ = find_peaks(np.concatenate((-ankle_r_ref, -ankle_r_ref, -ankle_r_ref)), distance=1)
min_ankle_idx_r = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((ankle_r_ref, ankle_r_ref, ankle_r_ref)), distance=1)
max_ankle_idx_r = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((ankle_r_ref, ankle_r_ref, ankle_r_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((ankle_r_ref, ankle_r_ref, ankle_r_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Right Ankle Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, ankle_r_ref)
for i in range(len(min_ankle_idx_r)):
    axs[1, 0].scatter(time_cycle[min_ankle_idx_r[i]], ankle_r_ref[min_ankle_idx_r[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Right Ankle Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((ankle_r_ref, ankle_r_ref, ankle_r_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((ankle_r_ref, ankle_r_ref, ankle_r_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Right Ankle Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, ankle_r_ref)
for i in range(len(max_ankle_idx_r)):
    axs[1, 1].scatter(time_cycle[max_ankle_idx_r[i]], ankle_r_ref[max_ankle_idx_r[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Right Ankle Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Detect extrema for left ankle angles
temp_min, _ = find_peaks(np.concatenate((-ankle_l_ref, -ankle_l_ref, -ankle_l_ref)), distance=1)
min_ankle_idx_l = np.subtract(temp_min[(len(time_cycle)-1 < temp_min) & (temp_min < 2*len(time_cycle))], len(time_cycle))

temp_max, _ = find_peaks(np.concatenate((ankle_l_ref, ankle_l_ref, ankle_l_ref)), distance=1)
max_ankle_idx_l = np.subtract(temp_max[(len(time_cycle)-1 < temp_max) & (temp_max < 2*len(time_cycle))], len(time_cycle))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((ankle_l_ref, ankle_l_ref, ankle_l_ref)))
for i in range(len(temp_min)):
    axs[0, 0].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_min[i]], np.concatenate((ankle_l_ref, ankle_l_ref, ankle_l_ref))[temp_min[i]], label=f'Min {i}', zorder=5)
axs[0, 0].set_title('Extended Left Ankle Angle with Categorized Minima')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper right")

axs[1, 0].plot(time_cycle, ankle_l_ref)
for i in range(len(min_ankle_idx_l)):
    axs[1, 0].scatter(time_cycle[min_ankle_idx_l[i]], ankle_l_ref[min_ankle_idx_l[i]], label=f'Min {i}', zorder=5)
axs[1, 0].set_title('Left Ankle Angle with Categorized Minima')
axs[1, 0].grid(True)
axs[1, 0].legend(loc="upper right")

axs[0, 1].plot(np.arange(0, 3*len(time_cycle)*0.01, 0.01), np.concatenate((ankle_l_ref, ankle_l_ref, ankle_l_ref)))
for i in range(len(temp_max)):
    axs[0, 1].scatter(np.arange(0, 3*len(time_cycle)*0.01, 0.01)[temp_max[i]], np.concatenate((ankle_l_ref, ankle_l_ref, ankle_l_ref))[temp_max[i]], label=f'Max {i}', zorder=5)
axs[0, 1].set_title('Extended Left Ankle Angle with Categorized Maxima')
axs[0, 1].grid(True)
axs[0, 1].legend(loc="upper right")

axs[1, 1].plot(time_cycle, ankle_l_ref)
for i in range(len(max_ankle_idx_l)):
    axs[1, 1].scatter(time_cycle[max_ankle_idx_l[i]], ankle_l_ref[max_ankle_idx_l[i]], label=f'Max {i}', zorder=5)
axs[1, 1].set_title('Left Ankle Angle with Categorized Maxima')
axs[1, 1].grid(True)
axs[1, 1].legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.close()

# Combine minima and maxima into one array and sort them by their order in the time sequence
cp_hip_idx_r = np.sort(np.concatenate((min_hip_idx_r, max_hip_idx_r)))
cp_hip_idx_l = np.sort(np.concatenate((min_hip_idx_l, max_hip_idx_l)))
cp_knee_idx_r = np.sort(np.concatenate((min_knee_idx_r, max_knee_idx_r)))
cp_knee_idx_l = np.sort(np.concatenate((min_knee_idx_l, max_knee_idx_l)))
cp_ankle_idx_r = np.sort(np.concatenate((min_ankle_idx_r, max_ankle_idx_r)))
cp_ankle_idx_l = np.sort(np.concatenate((min_ankle_idx_l, max_ankle_idx_l)))

cp_hip_ang_r = hip_r_ref[cp_hip_idx_r]
cp_hip_ang_l = hip_l_ref[cp_hip_idx_l]
cp_knee_ang_r = knee_r_ref[cp_knee_idx_r]
cp_knee_ang_l = knee_l_ref[cp_knee_idx_l]
cp_ankle_ang_r = ankle_r_ref[cp_ankle_idx_r]
cp_ankle_ang_l = ankle_l_ref[cp_ankle_idx_l]