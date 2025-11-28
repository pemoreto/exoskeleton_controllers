import numpy as np
import os
import pandas as pd
from phase_plots_generator import compute_phase_portrait, compute_poincare_map
from FM_rechner import compute_jacobian
from MoS_rechner import compute_mos
from VPP_rechner import calculate_VPP

"""
This script analyzes .sto simulation files and extracts multiple stability and performance metrics.

For each .sto file found in the "Database" folder (and its subfolders), the following steps are performed:
- Extract joint angle and velocity data (hip, knee, ankle), GRF, and COM height.
- Generate a phase portrait and compute the Poincaré map based on right knee motion.
- Compute Floquet multipliers to assess orbital stability.
- Evaluate COM-based stability by checking if minimum COM height ≥ 1.0 m.
- Calculate Margin of Stability (MoS) and braking time (bt) from ground reaction force and kinematic data.
- Analyze the quality of the Virtual Pivot Point (VPP) by computing the mean R² value for both legs.

All results (including plots and text summaries) are saved to the `results_FM_MoS_VPP_R2` directory.
A final summary spreadsheet (`summary.xlsx`) compiles results across all processed files."""

# Main paths
base_dir = "Database"
version = "FM_MoS_VPP_R2"
results_dir = f"results_{version}"
results_plots_dir = os.path.join(results_dir, "results_plots_FM")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(results_plots_dir, exist_ok=True)

# Find .sto-files
sto_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".sto"):
            sto_files.append(os.path.join(root, file))

summary = []

for file_path in sto_files:
    folder_name = os.path.basename(os.path.dirname(file_path))
    filename = folder_name
    print(f"Processing {filename} (Original: {file_path})")

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip().lower() == 'endheader':
                header_end_idx = i
                break
        else:
            raise ValueError("Header end not found in .sto file")

        column_names = lines[header_end_idx + 1].split()

        def find_col(name):
            return next((col for col in column_names if name in col.lower()), None)

        # Spalten finden
        knee_column = find_col("knee_angle_r")
        knee_u_column = find_col("knee_angle_r_u")
        ankle_column = find_col("ankle_angle_r")
        ankle_u_column = find_col("ankle_angle_r_u")
        hip_column = find_col("hip_flexion_r")
        hip_u_column = find_col("hip_flexion_r_u")
        grf_r_column = find_col("leg1_r.grf_y")
        time_column = find_col("time")
        com_y_column = find_col("com_y")

        required_columns = {
            "knee_angle_r": knee_column,
            "knee_angle_r_u": knee_u_column,
            "ankle_angle_r": ankle_column,
            "ankle_angle_r_u": ankle_u_column,
            "hip_flexion_r": hip_column,
            "hip_flexion_r_u": hip_u_column,
            "leg1_r.grf_y": grf_r_column,
            "time": time_column,
            "com_y": com_y_column
        }

        missing = [key for key, val in required_columns.items() if val is None]
        if missing:
            print(f"Datei übersprungen wegen fehlende Spalten: {missing}")
            summary.append({
                "Datei": filename,
                "Stabil_EW": f"Fehlende Spalten: {', '.join(missing)}",
                "Stabil_COM": "n/a",
                "MaxAbsEW": "n/a",
                "MoS_min": "n/a",
                "MoS_max": "n/a",
                "MoS_mean": "n/a",
                "bt_min": "n/a",
                "bt_max": "n/a",
                "bt_mean": "n/a",
                "Mean R2 Right Leg": "n/a",
                "Mean R2 Left Leg": "n/a"
            })
            continue

        data = np.loadtxt(file_path, comments='#', skiprows=header_end_idx + 2)
        get_col = lambda col_name: data[:, column_names.index(col_name)]

        angle_data = np.column_stack((get_col(knee_column), get_col(ankle_column), get_col(hip_column)))
        velocity_data = np.column_stack((get_col(knee_u_column), get_col(ankle_u_column), get_col(hip_u_column)))
        grf_r_data = get_col(grf_r_column)
        time_data = get_col(time_column)
        com_y_data = get_col(com_y_column)

        # Phase Portrait & Poincare Map
        _, _, _, poincare_points = compute_phase_portrait(
            angle_data, velocity_data, grf_r_data, time_data,
            'Right Knee Angle [rad]', 'Right Knee Angular Velocity [rad/s]',
            f'{filename}_Phase_Portrait', 'blue', results_plots_dir, version, show_plots=False
        )

        _, _, full_x, full_y = compute_poincare_map(
            poincare_points, [-np.pi / 2, 0],
            'Right Knee Angle [rad]', f'{filename}_Poincare_Map',
            results_plots_dir, version, show_plots=False
        )

        if len(full_x) == 0 or len(full_y) == 0:
            raise ValueError("Keine gültigen Poincaré-Daten")

        J, multipliers = compute_jacobian(full_x, full_y, results_plots_dir, version, title=f'{filename}_FM',
                                          show_plot=False)
        abs_multipliers = np.abs(multipliers)

        com_stabil = np.min(com_y_data) >= 1.0
        eig_stabil = np.all(abs_multipliers < 1)

        # MoS berechnen
        mos_ap_min, mos_ap_max, mos_ap_mean, bt_ap_min, bt_ap_max, bt_ap_mean = compute_mos(lines)

        # VPP – nur gemittelter R² pro Bein
        results = calculate_VPP(file_path)

        mean_r2_right = np.mean(results['right']['R2']) if results['right']['R2'] else np.nan
        mean_r2_left = np.mean(results['left']['R2']) if results['left']['R2'] else np.nan

        # Ergebnis schreiben
        result_txt_path = os.path.join(results_dir, f"{filename}_result.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Datei: {filename}\n")
            f.write(f"Stabil nach Eigenwerten (<1): {'Ja' if eig_stabil else 'Nein'}\n")
            f.write(f"Stabil nach COM_y (≥1.0): {'Ja' if com_stabil else 'Nein'}\n")
            f.write("Beträge der Eigenwerte:\n")
            for ev in abs_multipliers:
                f.write(f"{ev:.4f}\n")
            f.write(f"\nMin. COM_y: {np.min(com_y_data):.3f} m\n")
            f.write(f"\nMoS AP min: {mos_ap_min:.4f}, max: {mos_ap_max:.4f}, mean: {mos_ap_mean:.4f}\n")
            f.write(f"bt  AP min: {bt_ap_min:.4f}, max: {bt_ap_max:.4f}, mean: {bt_ap_mean:.4f}\n")
            f.write(f"Mean VPP R² (Right Leg): {mean_r2_right:.4f}\n")
            f.write(f"Mean VPP R² (Left Leg): {mean_r2_left:.4f}\n")

        # Zusammenfassung für Excel
        summary.append({
            "Datei": filename,
            "Stabil_EW": "Ja" if eig_stabil else "Nein",
            "Stabil_COM": "Ja" if com_stabil else "Nein",
            "MaxAbsEW": round(np.max(abs_multipliers), 4),
            "MoS_min": round(mos_ap_min, 4),
            "MoS_max": round(mos_ap_max, 4),
            "MoS_mean": round(mos_ap_mean, 4),
            "bt_min": round(bt_ap_min, 4),
            "bt_max": round(bt_ap_max, 4),
            "bt_mean": round(bt_ap_mean, 4),
            "Mean R2 Right Leg": round(mean_r2_right, 4),
            "Mean R2 Left Leg": round(mean_r2_left, 4)
        })

    except Exception as e:
        print(f"Fehler bei {filename}: {e}")
        summary.append({
            "Datei": filename,
            "Stabil_EW": f"Fehler: {str(e)}",
            "Stabil_COM": "n/a",
            "MaxAbsEW": "n/a",
            "MoS_min": "n/a",
            "MoS_max": "n/a",
            "MoS_mean": "n/a",
            "bt_min": "n/a",
            "bt_max": "n/a",
            "bt_mean": "n/a",
            "Mean R2 Right Leg": "n/a",
            "Mean R2 Left Leg": "n/a"
        })

# Excel schreiben
summary_df = pd.DataFrame(summary, columns=[
    "Datei", "Stabil_EW", "Stabil_COM", "MaxAbsEW",
    "MoS_min", "MoS_max", "MoS_mean",
    "bt_min", "bt_max", "bt_mean",
    "Mean R2 Right Leg", "Mean R2 Left Leg"
])
excel_path = os.path.join(results_dir, "summary.xlsx")
summary_df.to_excel(excel_path, index=False)

print(f"\nAnalysis ended. Results saved in:\n{excel_path}")
