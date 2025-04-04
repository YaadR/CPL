# --- main_cpl_visualized.py (Adapted for Repo problem.py/model.py) ---
import torch
from problem import get_problem # Uses the repo's problem.py
# Import standalone functions for plotting
from problem import ackley_func_orig, himmelblau_func_orig, rosenbrock_func_orig
from model import ParetoSetModel # Uses the adjusted model.py
import timeit
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Ensure the plots directory exists
os.makedirs("plots", exist_ok=True)

# --- Configuration ---
# Use float64 consistently as model output and problem functions expect it
torch.set_default_dtype(torch.float64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

n_run = 5 # Reduced for faster demonstration
test_ins = 'ackley' # Choose from ['ackley', 'himmelblau', 'rosenbrock']

# Problem-specific hyperparameters (from original user script)
if test_ins == 'ackley':
    n_steps = 225
    n_levels = 4
    n_local_search = 100
    init_point = torch.tensor([5.0, 5.0], device=device, dtype=torch.float64)
    plot_range = (-6, 6) # Plotting boundaries
    obj_func_for_plot = ackley_func_orig
elif test_ins == 'himmelblau':
    n_steps = 450
    n_levels = 4
    n_local_search = 200
    init_point = torch.tensor([-3.0, -2.0], device=device, dtype=torch.float64)
    plot_range = (-6, 6)
    obj_func_for_plot = himmelblau_func_orig
elif test_ins == 'rosenbrock':
    n_steps = 4500
    n_levels = 4
    n_local_search = 2000
    init_point = torch.tensor([-3.0, 2.0], device=device, dtype=torch.float64)
    plot_range = (-4, 4)
    obj_func_for_plot = rosenbrock_func_orig
else:
    raise ValueError("Unknown test instance")

# --- Load Problem ---
# The repo's get_problem returns an instance of the specific problem class
problem = get_problem(test_ins)
n_dim = problem.n_dim

# --- Plotting Function ---
# Uses the standalone original objective functions for contours
def plot_optimization_path(problem_name, obj_func_plot, path_history, final_x, run_iter, plot_range):
    """Plots the optimization path on the function's contour map."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create contour data
    x_grid = np.linspace(plot_range[0], plot_range[1], 100)
    y_grid = np.linspace(plot_range[0], plot_range[1], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Use torch.no_grad() for purely evaluative plotting calculations
    with torch.no_grad():
        # Ensure grid tensor is float64 and on the correct device for the obj_func
        XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float64).to(device)
        Z = obj_func_plot(XY).cpu().numpy().reshape(X.shape)

    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax, label=f'{problem_name} function value')

    # Plot path
    if path_history: # Check if path_history is not empty
        path = np.array(path_history)
        ax.plot(path[:, 0], path[:, 1], 'r-o', markersize=3, linewidth=1, label='Optimization Path (t=1)')
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start (Matched Init)')
    else:
        print("Warning: path_history is empty, cannot plot path.")

    ax.plot(final_x[0], final_x[1], 'y*', markersize=12, label='Final Solution')

    # Plot known minima for reference (optional, add if known)
    if problem_name == 'ackley':
        ax.plot(0, 0, 'kx', markersize=10, mew=2, label='Global Minimum (0,0)')
    elif problem_name == 'himmelblau':
        minima = [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
        min_labels = [f'Minimum ({mx:.2f},{my:.2f})' for mx, my in minima]
        # Plot first minimum with label
        ax.plot(minima[0][0], minima[0][1], 'kx', markersize=10, mew=2, label=min_labels[0])
        # Plot remaining minima without label to avoid clutter
        for i in range(1, len(minima)):
             ax.plot(minima[i][0], minima[i][1], 'kx', markersize=10, mew=2)
    elif problem_name == 'rosenbrock':
         ax.plot(1, 1, 'kx', markersize=10, mew=2, label='Global Minimum (1,1)')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'CPL Optimization Path for {problem_name} (Run {run_iter+1})')
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)
    # Consolidate legend to avoid duplicates if minima were plotted separately
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save plot
    plot_filename = os.path.join("plots", f"cpl_path_{problem_name}_run_{run_iter+1}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close(fig)


# --- Main Loop ---
value_list = []
all_run_histories = []

print(f"Starting CPL for {test_ins}...")

for run_iter in range(n_run):
    print(f"\n--- Run {run_iter+1}/{n_run} ---")
    path_history_run = [] # Store path for this run

    # Model initialization (using adjusted model)
    psmodel = ParetoSetModel(n_dim).to(device=device) # Already float64 by default now
    psmodel.train() # Set model to training mode

    # --- Initial Point Matching ---
    print("Matching initial point...")
    # Use a smaller LR for matching as in original script
    optimizer_init = torch.optim.Adam(psmodel.parameters(), lr=1e-4)
    t_one = torch.ones([1, 1], device=device, dtype=torch.float64)

    # Record initial state before matching
    with torch.no_grad():
         initial_x_before_match = psmodel(t_one).cpu().numpy().flatten()
         path_history_run.append(initial_x_before_match)

    for i in range(1000): # As in original script
        # Gradients needed here
        x = psmodel(t_one)
        loss = torch.mean((x - init_point)**2) # Use MSE loss for matching

        optimizer_init.zero_grad()
        loss.backward()
        optimizer_init.step()
        if i % 200 == 0 or i == 999: # Record progress and final state
            with torch.no_grad():
                current_x = psmodel(t_one).cpu().numpy().flatten()
            path_history_run.append(current_x)

    with torch.no_grad():
        matched_x = psmodel(t_one).cpu().numpy().flatten()
        print(f"Initial point matching complete. Target: {init_point.cpu().numpy()}, Achieved: {matched_x}")
        if len(path_history_run) <= 1:
             path_history_run.append(matched_x)


    # --- CPL Training ---
    print("Starting CPL training...")
    start_time = timeit.default_timer()
    # Use LR from original script for CPL training
    optimizer_cpl = torch.optim.Adam(psmodel.parameters(), lr=5e-3)

    for t_step in range(n_steps):
        psmodel.train() # Ensure model is in training mode
        # Sample n_levels homotopy levels
        t = torch.rand([n_levels, 1], device=device, dtype=torch.float64)
        t[0] = 1.0 # Ensure t=1 is always included

        # Get current corresponding solutions (needs grad for backward pass later)
        # Ensure model output is used directly (should be float64)
        x = psmodel(t)

        # Evaluate homotopy function and get gradient using repo's method
        # The repo's evaluate returns F (homotopy value) and grad
        # No compute_grad flag needed
        value_F, grad = problem.evaluate(x, t) # grad is needed here

        # Gradient-based continuation path model update
        optimizer_cpl.zero_grad()
        # Backward pass using the external gradient from problem.evaluate
        psmodel(t).backward(grad)
        optimizer_cpl.step()

        # Record position at t=1 for visualization (no grad needed)
        if t_step % 10 == 0: # Record every 10 steps
             with torch.no_grad():
                psmodel.eval() # Temporarily set to eval mode
                t_one_eval = torch.ones([1,1], device=device, dtype=torch.float64)
                x_at_t1 = psmodel(t_one_eval).cpu().numpy().flatten()
                path_history_run.append(x_at_t1)
                psmodel.train() # Set back to train mode

    # --- Local Search (Fine-tuning at t=1) ---
    print("Starting local search at t=1...")
    t_one = torch.ones([1, 1], device=device, dtype=torch.float64)
    # Use a smaller LR for fine-tuning (e.g., 1e-3 or as needed)
    optimizer_local = torch.optim.Adam(psmodel.parameters(), lr=1e-3)

    for ls_step in range(n_local_search):
        psmodel.train() # Ensure model is in training mode for backward pass
        x = psmodel(t_one)
        # Evaluate H(x, 1) and get gradient using repo's method
        value_F, grad = problem.evaluate(x, t_one) # grad is needed here

        optimizer_local.zero_grad()
        # Use gradient returned by evaluate (which corresponds to t=1)
        psmodel(t_one).backward(grad)
        optimizer_local.step()

        # Record position during local search (no grad needed)
        if ls_step % (max(1, n_local_search // 10)) == 0: # Record a few steps
            with torch.no_grad():
                psmodel.eval() # Temporarily set to eval mode
                x_at_t1 = psmodel(t_one).cpu().numpy().flatten()
                path_history_run.append(x_at_t1)
                psmodel.train() # Set back to train mode

    # --- Final Evaluation ---
    print("Performing final evaluation...")
    stop_time = timeit.default_timer()
    psmodel.eval() # Set model to evaluation mode
    final_x_np = None
    final_value_item = float('nan')
    with torch.no_grad(): # Disable gradients globally for final evaluation
        t_one_final = torch.ones([1, 1], device=device, dtype=torch.float64)
        final_x = psmodel(t_one_final)

        # Evaluate using repo's evaluate at t=1. It returns F, grad. We only need F.
        # Note: For Ackley, repo returns f(x). For others, it returns F(x, t=1).
        # This F(x, t=1) should correspond to f(x) if the homotopy is defined correctly.
        final_value_F, _ = problem.evaluate(final_x, t_one_final)

        final_x_np = final_x.cpu().numpy().flatten()
        # Use the returned F value as the final value.
        final_value_item = final_value_F.item() # F should be scalar here

    # Append final state to path history for plotting the endpoint accurately
    if final_x_np is not None:
        path_history_run.append(final_x_np)

    value_list.append(final_value_item)
    all_run_histories.append(path_history_run)

    print(f'Run {run_iter+1} Finished')
    print(f'  Solution (x at t=1): {final_x_np}')
    # The value printed is F(x, t=1) from the repo's evaluate method
    print(f'  Value (F(x, t=1)): {final_value_item:.6f}')
    print(f'  Time: {stop_time - start_time:.2f} seconds')
    print("************************************************************")

    # --- Plotting for this run ---
    if final_x_np is not None:
        # Pass the standalone original objective function for plotting contours
        plot_optimization_path(test_ins, obj_func_for_plot, path_history_run, final_x_np, run_iter, plot_range)
    else:
        print(f"Skipping plot for run {run_iter+1} due to missing final solution.")


# --- Overall Results ---
if value_list:
    valid_values = [v for v in value_list if not np.isnan(v)]
    if valid_values:
        avg_value = np.mean(valid_values)
        std_value = np.std(valid_values)
        print(f'\n--- Overall Results ({len(valid_values)} valid runs out of {n_run}) ---')
        print(f'Average Final Value (F(x, t=1)): {avg_value:.6f}') # Label clearly
        print(f'Standard Deviation: {std_value:.6f}')
    else:
        print("\nNo valid runs completed successfully.")
else:
    print("\nNo runs completed.")

print(f"Completed CPL for {test_ins}.")