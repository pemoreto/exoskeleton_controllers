import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from analytic_barrier_acqf import ConstrainedExpectedImprovementSafeUpdate
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from obj_con_function import obj_con_function

def safeBO_unified(init_points, num_iterations, bounds, beta, tau, save_plots):
    """
    Perform constrained Bayesian Optimization to tune PID controller parameters
    along with vertical shift parameters for a gait model.

    The function uses GP models to approximate the objective and constraint
    functions and optimizes a constrained acquisition function (Constrained Expected Improvement)
    to safely explore the parameter space.

    Parameters:
    - init_points (array-like): Initial sample points (each with 7 parameters: Kp, Ki, Kd, Shift1-4).
    - num_iterations (int): Number of Bayesian Optimization iterations to perform.
    - bounds (torch.Tensor): Tensor specifying the lower and upper bounds for each parameter (shape: 2 x 7).
    - beta (float): Exploration-exploitation tradeoff parameter in acquisition function.
    - tau (float): Safety threshold parameter for constraint handling.
    - save_plots (bool): Whether to save plots generated during optimization.

    Returns:
    - best_pid_param (numpy.ndarray): Best PID parameters found (Kp, Ki, Kd).
    - best_shifts (numpy.ndarray): Best vertical shift parameters found (Shift1, Shift2, Shift3, Shift4).
    - version (str): Version identifier string based on best objective value.
    """

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(1234)

    train_x = torch.tensor(init_points, dtype=torch.double)
    num_initial_points = len(init_points)

    # Evaluate the objective and constraint functions using the unified function
    results = [obj_con_function(x, 0, False) for x in train_x]
    train_y_obj = torch.tensor([-r[0] for r in results], dtype=torch.double).unsqueeze(-1)
    train_y_con = torch.tensor([-r[1] for r in results], dtype=torch.double).unsqueeze(-1)
    last_com_height = torch.tensor([r[2] for r in results], dtype=torch.double).unsqueeze(-1)
    max_distance = torch.tensor([r[3] for r in results], dtype=torch.double).unsqueeze(-1)

    # Initialize history with initial points (no loop)
    history = [{
        "iteration": (i + 1) * (-1),
        "objective_value": -train_y_obj[i].item(),
        "constraint_value": -train_y_con[i].item(),
        "acquisition_value": None,
        "Kp": train_x[i][0].item(),
        "Ki": train_x[i][1].item(),
        "Kd": train_x[i][2].item(),
        "Shift1": train_x[i][3].item(),
        "Shift2": train_x[i][4].item(),
        "Shift3": train_x[i][5].item(),
        "Shift4": train_x[i][6].item(),
        "Stable": True if -train_y_con[i].item() > 0 else False,
        "last_com_height": last_com_height[-1].item(),
        "max_distance": max_distance[i].item()
    } for i in range(train_x.size(0))]

    # Define Gaussian Process models for objective and constraint
    gp_obj = SingleTaskGP(train_x, train_y_obj, outcome_transform=Standardize(m=1), input_transform=Normalize(d=7))
    gp_con = SingleTaskGP(train_x, train_y_con, outcome_transform=Standardize(m=1), input_transform=Normalize(d=7))

    mll_obj = ExactMarginalLogLikelihood(gp_obj.likelihood, gp_obj)
    mll_con = ExactMarginalLogLikelihood(gp_con.likelihood, gp_con)

    fit_gpytorch_mll(mll_obj)
    fit_gpytorch_mll(mll_con)

    model = ModelListGP(gp_obj, gp_con)

    # Lists for tracking performance
    objective_values, acquisition_values = [], []

    # Optimization settings
    num_restarts, raw_samples = 10, 200

    best_entry = None

    # Bayesian Optimization loop
    for i in range(num_iterations):

        cei = ConstrainedExpectedImprovementSafeUpdate(
            model=model,
            best_f=train_y_obj.max().item(),
            objective_index=0,
            constraints={1: (None, None)},
            beta=beta,
            tau=tau,
        )

        initial_conditions = (
                (bounds[1] - bounds[0]) *
                torch.quasirandom.SobolEngine(dimension=7, scramble=True).draw(num_restarts)
                + bounds[0]
        ).unsqueeze(1)

        # Optimize acquisition function
        candidate_X, acq_value = optimize_acqf(acq_function=cei,
                                               bounds=bounds,
                                               q=1,
                                               num_restarts=num_restarts,
                                               raw_samples=raw_samples,
                                               batch_initial_conditions=initial_conditions
                                               )

        # Evaluate both objective and constraint at the new candidate
        candidate_Y, candidate_Y_con, last_com_height, max_distance = obj_con_function(candidate_X[0], i, save_plots)
        candidate_Y = torch.tensor([-candidate_Y], dtype=torch.double).unsqueeze(-1)
        candidate_Y_con = torch.tensor([-candidate_Y_con], dtype=torch.double).unsqueeze(-1)

        # Append new data points
        train_x = torch.cat([train_x, candidate_X])
        train_y_obj = torch.cat([train_y_obj, candidate_Y])
        train_y_con = torch.cat([train_y_con, candidate_Y_con])

        # Update Gaussian Process models
        gp_obj = SingleTaskGP(train_x, train_y_obj, outcome_transform=Standardize(m=1), input_transform=Normalize(d=7))
        gp_con = SingleTaskGP(train_x, train_y_con, outcome_transform=Standardize(m=1), input_transform=Normalize(d=7))

        mll_obj = ExactMarginalLogLikelihood(gp_obj.likelihood, gp_obj)
        mll_con = ExactMarginalLogLikelihood(gp_con.likelihood, gp_con)

        fit_gpytorch_mll(mll_obj)
        fit_gpytorch_mll(mll_con)

        # Store best objective values for plotting
        objective_values.append(-candidate_Y.item())
        acquisition_values.append(acq_value.item())

        # Extract PID and shift parameters
        pid_params = candidate_X[0][:3].numpy()
        shift_params = candidate_X[0][3:].numpy()

        # Store all relevant values in history
        history.append({
            "iteration": i + 1,
            "objective_value": -candidate_Y.item(),
            "constraint_value": -candidate_Y_con.item(),
            "acquisition_value": acq_value.item(),
            "Kp": pid_params[0],
            "Ki": pid_params[1],
            "Kd": pid_params[2],
            "Shift1": shift_params[0],
            "Shift2": shift_params[1],
            "Shift3": shift_params[2],
            "Shift4": shift_params[3],
            "Stable": True if -candidate_Y_con.item() > 0 else False,
            "last_com_height": last_com_height,
            "max_distance": max_distance
        })

        # Filter entries with positive constraint values
        positive_constraint_entries = [entry for entry in history if entry["constraint_value"] > 0]

        if positive_constraint_entries:
            # If there are positive constraint values, find the entry with the smallest objective_value among them
            best_entry = min(positive_constraint_entries, key=lambda x: x["objective_value"])
        else:
            # If no positive constraint values, find the entry with the smallest objective_value overall
            best_entry = min(history, key=lambda x: x["objective_value"])

        # Extract the best current cost and corresponding constraint value
        best_current_cost = best_entry["objective_value"]
        best_constraint_value = best_entry["constraint_value"]

        print(f"\n---------Iteration {i + 1}/{num_iterations}, Current best cost: {best_current_cost:.4f}, Corresponding Constraint Value: {best_constraint_value:.4f}---------")


    # Retrieve best parameters found
    best_idx = best_entry["iteration"] -1 + num_initial_points
    best_params = train_x[best_idx]
    best_cost = train_y_obj[best_idx].item()

    # Generate a version identifier
    version = f"v_{-best_cost:.4f}"

    results_dir = f"results_{version}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Best PID parameters: Kp = {best_params[0].item():.4f}, Ki = {best_params[1].item():.4f}, Kd = {best_params[2].item():.4f}")
    print(f"Best Shifts vertical: {best_params.numpy()[3:7]}")
    print(f"Best cost: {-best_cost:.4f}")

    # Plot the best objective values over iterations
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute the staircase of best values so far
    best_so_far = np.minimum.accumulate(objective_values)

    # Plot best values so far
    ax.step(range(len(objective_values)), best_so_far, where='post', color='r', linestyle='--',
            label='Best Value So Far', linewidth=2)

    # Formatting
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Function Value', fontsize=12)
    ax.set_ylim(min(objective_values) - 0.5, 705)
    ax.set_xlim(0, num_iterations)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=10)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"obj_func_over_iterations_{version}.png"), dpi=300)
    plt.show()

    # Save results to a text file
    best_pid_param = best_params.numpy()[:3]
    best_shifts = best_params.numpy()[3:]

    txt_filename = os.path.join(results_dir, f"optimization_results_{version}.txt")

    with open(txt_filename, "w") as f:
        f.write(f"Best value: {-best_cost}\n")
        f.write(f"Best PID Param: {best_pid_param}\n")
        f.write(f"Best Shifts: {best_shifts}\n")
        f.write(f"Bounds: {bounds.numpy()}\n")
        f.write(
            "Notizen: \n Actuated Simulation with actuated HFD Model. \n PID-Param and vertical Extremes-Shifts are optimized.\n With one stable points before BO. \n sigmoidal cost function \n beta=1 \n tau=0.1 \n"
            f"No improvement init_points: {init_points}\n"
        )

    df = pd.DataFrame(history)
    excel_filename = os.path.join(results_dir, f"optimization_history_{version}.xlsx")
    df.to_excel(excel_filename, index=False, engine='openpyxl')
    print(f"optimization History saved to '{excel_filename}'.")

    return best_pid_param, best_shifts, version
