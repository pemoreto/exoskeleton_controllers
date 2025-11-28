import os
from sconetools import sconepy
from run_simulation_ff import run_simulation_ff

"""
Main script to configure and run a SCONE gait simulation.

- Initializes the SCONE environment.
- Selects and loads a biomechanical model (either 'hfd' or 'osim').
- Applies an optimized parameter file for 'hfd' models.
- Runs the simulation using a feedforward control function.
- Stores and processes the results.

Settings:
    model_type (str): Choose between 'hfd' or 'osim'.
    duration (float): Total simulation time in seconds.
    start_actuation (float): Time when control inputs begin."""

# Extract the script name for versioning
simu_name = os.path.splitext(os.path.basename(__file__))[0]

# SCONE Simulation Initialization
sconepy.set_log_level(3)
print('SCONE Version', sconepy.version())
sconepy.set_array_dtype_float32()

# 'hfd' or 'osim'
model_type = 'osim'

# Load SCONE simulation model with optimized par-file (if hfd). If osim use "H0918RS2v3_opt_det_param_osim.scone"
par_file_hfd = '0774_0.895_0.880.par'

if model_type == 'hfd':
    model = sconepy.load_model(
        f'Simulation_H0918RS2_{model_type}_actuated/Simulation_H0918RS2_actuated.scone',
        par_file_hfd)
elif model_type == 'osim':
    model = sconepy.load_model(
        f'Simulation_H0918RS2_{model_type}_actuated/Simulation_H0918RS2_actuated.scone')
else:
    raise ValueError("model_type must be either 'hfd' or 'osim'.")

duration = 20
start_actuation = 2

# Run the SCONE simulation
run_simulation_ff(model, start_actuation,0, 0,True, duration, model_type)





