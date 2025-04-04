# --- main_cpl_visualized.py (Adapted for Repo problem.py/model.py with Online Plotting) ---
import torch
# Import standalone functions for plotting contours
from problem import ackley_func_orig, himmelblau_func_orig, rosenbrock_func_orig, get_problem
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

n_run = 5 # Reduced for faster demonstration and visualization
test_ins = 'ackley' # Choose from ['ackley', 'himmelblau', 'rosenbrock']
plot_update_frequency = 50 # Update plot every N steps during training/search
online_plotting = True # Set to False to disable live plots and only save final

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

# --- Plotting Function (Modified for Online Updates) ---
def plot_optimization_path(
    fig, ax, # Pass figure and axes for updates
    problem_name,
    obj_func_plot,
    path_history,
    current_x, # Current position at t=1
    final_x, # Final solution (can be None if not final plot)
    run_iter,
    plot_range,
    title_suffix="", # e.g., " (Training Step X)"
    is_final_save=False, # Flag to control saving and closing
    contour_data=None # Precomputed contour data
):
    """Plots or updates the optimization path on the function's contour map."""
    ax.clear() # Clear previous plot elements for update

    # Plot contours (using precomputed data if available)
    if contour_data is None:
        # Create contour data only if not provided
        x_grid = np.linspace(plot_range[0], plot_range[1], 100)
        y_grid = np.linspace(plot_range[0], plot_range[1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        with torch.no_grad():
            XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float64).to(device)
            Z = obj_func_plot(XY).cpu().numpy().reshape(X.shape)
        contour_data = (X, Y, Z) # Store for reuse
    else:
        X, Y, Z = contour_data # Unpack precomputed data

    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    # Add colorbar only once or if it doesn't exist
    if not fig.axes or len(fig.axes) == 1: # Check if colorbar needs to be added
         # Clear existing colorbars before adding a new one
        for cax in fig.get_axes():
            if cax is not ax and cax.get_label() == f'<colorbar for {ax.get_label()}>':
                 fig.delaxes(cax)
        try: # Add colorbar safely
             fig.colorbar(contour, ax=ax, label=f'{problem_name} function value')
        except Exception as e:
             print(f"Colorbar error (ignoring): {e}") # Handle potential issues if axes change unexpectedly

    # Plot path history
    start_point = None
    if path_history: # Check if path_history is not empty
        path = np.array(path_history)
        if len(path) > 0:
            start_point = path[0]
            ax.plot(path[:, 0], path[:, 1], 'r-o', markersize=3, linewidth=1, label='Optimization Path (t=1)')
            ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start (Matched Init)')
    else:
        print("Warning: path_history is empty, cannot plot path.")

    # Plot current position at t=1
    if current_x is not None:
         ax.plot(current_x[0], current_x[1], 'bo', markersize=6, label='Current Pos (t=1)')


    # Plot final solution if available
    if final_x is not None:
        ax.plot(final_x[0], final_x[1], 'y*', markersize=12, label='Final Solution')

    # Plot known minima for reference
    minima_plotted = False
    if problem_name == 'ackley':
        ax.plot(0, 0, 'kx', markersize=10, mew=2, label='Global Minimum (0,0)')
        minima_plotted = True
    elif problem_name == 'himmelblau':
        minima = [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
        min_labels = [f'Minimum ({mx:.2f},{my:.2f})' for mx, my in minima]
        ax.plot(minima[0][0], minima[0][1], 'kx', markersize=10, mew=2, label=min_labels[0])
        for i in range(1, len(minima)):
             ax.plot(minima[i][0], minima[i][1], 'kx', markersize=10, mew=2)
        minima_plotted = True
    elif problem_name == 'rosenbrock':
         ax.plot(1, 1, 'kx', markersize=10, mew=2, label='Global Minimum (1,1)')
         minima_plotted = True

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'CPL Path for {problem_name} (Run {run_iter+1}){title_suffix}')
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)

    # Consolidate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize='small') # Smaller font for legend
    ax.grid(True, linestyle='--', alpha=0.5)

    if online_plotting and not is_final_save:
        # Draw and pause briefly to allow the GUI event loop to process updates
        fig.canvas.draw_idle()
        plt.pause(0.01) # Small pause is crucial for display update
    elif is_final_save:
        # Save final plot for the run
        plot_filename = os.path.join("plots", f"cpl_path_{problem_name}_run_{run_iter+1}_final.png")
        plt.savefig(plot_filename)
        print(f"Saved final plot to {plot_filename}")
        plt.close(fig) # Close the figure after saving the final version

    return contour_data # Return contour data for reuse

# --- Final Results Plotting Function ---
def plot_final_results(problem_name, value_list, n_run):
    """Plots a summary of the final values from all runs."""
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))

    valid_indices = [i + 1 for i, v in enumerate(value_list) if not np.isnan(v)]
    valid_values = [v for v in value_list if not np.isnan(v)]

    if not valid_values:
        print("\nNo valid runs completed successfully to plot summary.")
        plt.close(fig_summary)
        return

    avg_value = np.mean(valid_values)
    std_value = np.std(valid_values)

    bars = ax_summary.bar(valid_indices, valid_values, color='skyblue', label='Final F(x, t=1)')
    ax_summary.axhline(avg_value, color='r', linestyle='--', label=f'Average: {avg_value:.4f}')

    # Add text labels for each bar
    for bar in bars:
        yval = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') # Adjust position

    ax_summary.set_xlabel('Run Number')
    ax_summary.set_ylabel('Final Value (F(x, t=1))')
    ax_summary.set_title(f'Final Optimization Values for {problem_name} ({len(valid_values)}/{n_run} Valid Runs)')
    ax_summary.set_xticks(range(1, n_run + 1)) # Show ticks for all potential runs
    ax_summary.legend()
    ax_summary.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add Std Dev info
    ax_summary.text(0.01, 0.95, f'Std Dev: {std_value:.4f}', transform=ax_summary.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    summary_filename = os.path.join("plots", f"cpl_summary_{problem_name}.png")
    plt.tight_layout()
    plt.savefig(summary_filename)
    print(f"\nSaved final results summary plot to {summary_filename}")
    if online_plotting:
        plt.show() # Show the summary plot if online plotting was enabled
    else:
        plt.close(fig_summary)


# --- Main Loop ---
value_list = []
all_run_histories = []

if online_plotting:
    plt.ion() # Turn on interactive mode for live plotting

print(f"Starting CPL for {test_ins}...")

for run_iter in range(n_run):
    print(f"\n--- Run {run_iter+1}/{n_run} ---")
    path_history_run = [] # Store path for this run
    fig_run, ax_run = None, None
    contour_data_run = None # To store contours for the current run

    if online_plotting:
        # Create figure for the current run
        fig_run, ax_run = plt.subplots(figsize=(8, 8))

    # Model initialization
    psmodel = ParetoSetModel(n_dim).to(device=device)
    psmodel.train()

    # --- Initial Point Matching ---
    print("Matching initial point...")
    optimizer_init = torch.optim.Adam(psmodel.parameters(), lr=1e-4)
    t_one = torch.ones([1, 1], device=device, dtype=torch.float64)

    with torch.no_grad():
         initial_x_before_match = psmodel(t_one).cpu().numpy().flatten()
         path_history_run.append(initial_x_before_match)
         current_vis_x = initial_x_before_match # For initial plot

    # Initial plot before matching starts
    if online_plotting:
        contour_data_run = plot_optimization_path(
            fig_run, ax_run, test_ins, obj_func_for_plot, path_history_run,
            current_vis_x, None, run_iter, plot_range, title_suffix=" (Before Matching)",
            contour_data=contour_data_run
        )


    for i in range(1000):
        x = psmodel(t_one)
        loss = torch.mean((x - init_point)**2)
        optimizer_init.zero_grad()
        loss.backward()
        optimizer_init.step()

        if i % plot_update_frequency == 0 or i == 999: # Update plot periodically
            with torch.no_grad():
                current_x_np = psmodel(t_one).cpu().numpy().flatten()
                # Only add to history if significantly different or first/last update
                if not path_history_run or np.linalg.norm(current_x_np - path_history_run[-1]) > 1e-3 or i == 999:
                     path_history_run.append(current_x_np)
                current_vis_x = current_x_np # Update for visualization
            if online_plotting:
                 contour_data_run = plot_optimization_path(
                     fig_run, ax_run, test_ins, obj_func_for_plot, path_history_run,
                     current_vis_x, None, run_iter, plot_range, title_suffix=f" (Matching Step {i+1})",
                     contour_data=contour_data_run
                 )

    with torch.no_grad():
        matched_x = psmodel(t_one).cpu().numpy().flatten()
        print(f"Initial point matching complete. Target: {init_point.cpu().numpy()}, Achieved: {matched_x}")
        if not path_history_run or np.linalg.norm(matched_x - path_history_run[-1]) > 1e-3:
             path_history_run.append(matched_x)
        current_vis_x = matched_x


    # --- CPL Training ---
    print("Starting CPL training...")
    start_time = timeit.default_timer()
    optimizer_cpl = torch.optim.Adam(psmodel.parameters(), lr=5e-3)

    for t_step in range(n_steps):
        psmodel.train()
        t = torch.rand([n_levels, 1], device=device, dtype=torch.float64)
        t[0] = 1.0
        x = psmodel(t)
        value_F, grad = problem.evaluate(x, t)

        optimizer_cpl.zero_grad()
        psmodel(t).backward(grad)
        optimizer_cpl.step()

        # Update plot periodically
        if t_step % plot_update_frequency == 0 or t_step == n_steps - 1:
            with torch.no_grad():
                psmodel.eval()
                t_one_eval = torch.ones([1,1], device=device, dtype=torch.float64)
                x_at_t1 = psmodel(t_one_eval).cpu().numpy().flatten()
                if not path_history_run or np.linalg.norm(x_at_t1 - path_history_run[-1]) > 1e-3 or t_step == n_steps - 1:
                    path_history_run.append(x_at_t1)
                current_vis_x = x_at_t1
                psmodel.train()
            if online_plotting:
                 contour_data_run = plot_optimization_path(
                     fig_run, ax_run, test_ins, obj_func_for_plot, path_history_run,
                     current_vis_x, None, run_iter, plot_range, title_suffix=f" (Training Step {t_step+1})",
                     contour_data=contour_data_run
                 )

    # --- Local Search (Fine-tuning at t=1) ---
    print("Starting local search at t=1...")
    t_one = torch.ones([1, 1], device=device, dtype=torch.float64)
    optimizer_local = torch.optim.Adam(psmodel.parameters(), lr=1e-3)

    for ls_step in range(n_local_search):
        psmodel.train()
        x = psmodel(t_one)
        value_F, grad = problem.evaluate(x, t_one)

        optimizer_local.zero_grad()
        psmodel(t_one).backward(grad)
        optimizer_local.step()

        # Update plot periodically
        if ls_step % plot_update_frequency == 0 or ls_step == n_local_search - 1:
            with torch.no_grad():
                psmodel.eval()
                x_at_t1 = psmodel(t_one).cpu().numpy().flatten()
                if not path_history_run or np.linalg.norm(x_at_t1 - path_history_run[-1]) > 1e-3 or ls_step == n_local_search - 1:
                     path_history_run.append(x_at_t1)
                current_vis_x = x_at_t1
                psmodel.train()
            if online_plotting:
                 contour_data_run = plot_optimization_path(
                     fig_run, ax_run, test_ins, obj_func_for_plot, path_history_run,
                     current_vis_x, None, run_iter, plot_range, title_suffix=f" (Local Search Step {ls_step+1})",
                     contour_data=contour_data_run
                 )

    # --- Final Evaluation ---
    print("Performing final evaluation...")
    stop_time = timeit.default_timer()
    psmodel.eval()
    final_x_np = None
    final_value_item = float('nan')
    with torch.no_grad():
        t_one_final = torch.ones([1, 1], device=device, dtype=torch.float64)
        final_x = psmodel(t_one_final)
        final_value_F, _ = problem.evaluate(final_x, t_one_final)
        final_x_np = final_x.cpu().numpy().flatten()
        final_value_item = final_value_F.item()

    if final_x_np is not None:
         # Ensure final point is distinct before adding
         if not path_history_run or np.linalg.norm(final_x_np - path_history_run[-1]) > 1e-6:
              path_history_run.append(final_x_np)
         current_vis_x = final_x_np # Update final position for plot

    value_list.append(final_value_item)
    all_run_histories.append(path_history_run)

    # --- Final Plotting and Saving for this Run ---
    print(f'Run {run_iter+1} Finished')
    # Print results clearly
    print(f'  Solution (x at t=1): {final_x_np}')
    print(f'  Value (F(x, t=1)): {final_value_item:.6f}')
    print(f'  Time: {stop_time - start_time:.2f} seconds')
    print("************************************************************")

    if final_x_np is not None:
        # Generate and save the final plot for this run
        # Need a new figure instance if online plotting was off, or use existing if on
        if not online_plotting:
             fig_run_final, ax_run_final = plt.subplots(figsize=(8, 8))
        else:
             fig_run_final, ax_run_final = fig_run, ax_run # Use the existing figure/axes

        plot_optimization_path(
            fig_run_final, ax_run_final, test_ins, obj_func_for_plot, path_history_run,
            current_vis_x, # Show final point as current
            final_x_np, # Also mark as final solution
            run_iter, plot_range, title_suffix=" (Final)",
            is_final_save=True, # Trigger save and close
            contour_data=contour_data_run # Reuse contours
        )
    else:
        print(f"Skipping final plot for run {run_iter+1} due to missing final solution.")
        if online_plotting and fig_run:
            plt.close(fig_run) # Close the online plot window if run failed


# --- Overall Results Visualization ---
if online_plotting:
    plt.ioff() # Turn off interactive mode after all runs

plot_final_results(test_ins, value_list, n_run)

print(f"\nCompleted CPL for {test_ins}.")
