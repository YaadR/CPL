# --- problem.py (From Repository - As Provided by User) ---
import torch
import numpy as np

# device - Note: Using global device is less flexible than passing it.
# The main script will handle moving tensors to the correct device.
# Consider refactoring this later if needed.
device = 'cpu' # Or 'cuda' if available

def get_problem(name, *args, **kwargs):
    """Factory function to retrieve problem class instances."""
    name = name.lower()

    PROBLEM = {
        'ackley': ackley,
        'rosenbrock': rosenbrock,
        'himmelblau': himmelblau,
        # Add regression problems if needed by the main script
        # 'regression_f1': regression_f1,
        # 'regression_f2': regression_f2,
        # 'regression_f3': regression_f3,
        # 'regression_f4': regression_f4,
    }

    if name not in PROBLEM:
        raise Exception(f"Problem '{name}' not found.")

    return PROBLEM[name](*args, **kwargs)

class ackley():
    """Ackley function problem definition."""
    def __init__(self, n_dim = 2):
        if n_dim != 2:
            raise ValueError("Repository Ackley implementation currently supports n_dim=2")
        self.n_dim = n_dim

    def f_func(self, x_input):
        """The original Ackley function f(x)."""
        # Ensure input is float64 for consistency if needed, though internal ops might convert
        x_input = x_input.to(torch.float64)
        x = x_input[:,0]
        y = x_input[:,1]

        f = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) \
            - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))) \
            + np.e + 20

        return f

    def evaluate(self, x_input, t):
        """
        Evaluates the homotopy value and gradient estimator for Ackley.
        Args:
            x_input (Tensor): Solution tensor [batch_size, n_dim].
            t (Tensor): Homotopy parameter tensor [batch_size, 1].
        Returns:
            Tuple[Tensor, Tensor]: (Homotopy Value F, Gradient grad_hat)
        """
        # Ensure inputs are on the correct device and type
        x_input = x_input.to(device=t.device, dtype=torch.float64)
        t = t.to(device=x_input.device, dtype=torch.float64)

        # Homotopy parameter transformation (t -> 0 means more smoothing)
        t_smooth = 1 - t + 0.0001 # Add epsilon to avoid division by zero

        # Sample noise for gradient estimator
        u = torch.randn([t.shape[0], self.n_dim], device=x_input.device, dtype=torch.float64)

        # Calculate original function value at x_input
        f = self.f_func(x_input)

        # Calculate gradient estimator using finite difference on smoothed function
        # H(x, t_smooth) approx E[f(x + sqrt(t_smooth)*noise)]
        # grad_H approx E[ (f(x+t_smooth*u) - f(x))/t_smooth * u ] according to repo formula
        # Note: Repo uses t_smooth directly, not sqrt(t_smooth) for noise scaling in grad_hat
        f_perturbed = self.f_func(x_input + t_smooth.expand(t.shape[0], self.n_dim) * u)

        # Repeat (f_perturbed - f) to match shape [batch_size, n_dim] for element-wise mult
        delta_f_repeated = (f_perturbed - f).unsqueeze(1).repeat(1, self.n_dim)

        # Calculate gradient hat
        grad_hat = (1 / t_smooth.expand(t.shape[0], self.n_dim)) * delta_f_repeated * u

        # Return f(x) as the 'value' and the estimated gradient grad_hat
        # Note: The paper implies H(x,t) should be returned, but repo code returns f(x).
        # We follow the repo code here.
        return f, grad_hat


class rosenbrock():
    """Rosenbrock function problem definition with analytical homotopy."""
    def __init__(self, n_dim = 2):
        if n_dim != 2:
            raise ValueError("Repository Rosenbrock implementation currently supports n_dim=2")
        self.n_dim = n_dim

    def evaluate(self, x_input, t):
        """
        Evaluates the analytical homotopy function F(x,t) and its exact gradient.
        Args:
            x_input (Tensor): Solution tensor [batch_size, n_dim].
            t (Tensor): Homotopy parameter tensor [batch_size, 1].
        Returns:
            Tuple[Tensor, Tensor]: (Homotopy Value F, Gradient dF/dx)
        """
        # Ensure inputs are on the correct device and type
        x_input = x_input.to(device=t.device, dtype=torch.float64)
        t = t.to(device=x_input.device, dtype=torch.float64)

        x = x_input[:,0]
        y = x_input[:,1]
        # Homotopy parameter transformation (t_homotopy -> 0 means original problem)
        t_homotopy = 1.5 * (1 - t.squeeze(-1)) # Squeeze t to match dimensions

        # Analytical Homotopy Function F(x, y, t_homotopy) from repo
        F = 100 * x**4 + (-200 * y + 600 * t_homotopy**2 + 1) * x**2 - 2 * x \
            + 100 * y**2 - 200 * t_homotopy**2 * y + (300 * t_homotopy**4 + 101 * t_homotopy**2 + 1)

        # Analytical Gradients dF/dx, dF/dy from repo
        dFdx = 400 * x**3 + 2 * (-200 * y + 600 * t_homotopy**2 + 1) * x - 2
        dFdy = -200 * x**2 + 200 * y - 200 * t_homotopy**2

        # Stack gradients into [batch_size, n_dim] tensor
        grad = torch.stack([dFdx, dFdy], dim=-1)

        return F, grad

class himmelblau():
    """Himmelblau function problem definition with analytical homotopy."""
    def __init__(self, n_dim = 2):
        if n_dim != 2:
            raise ValueError("Repository Himmelblau implementation currently supports n_dim=2")
        self.n_dim = n_dim

    def evaluate(self, x_input, t):
        """
        Evaluates the analytical homotopy function F(x,t) and its exact gradient.
        Args:
            x_input (Tensor): Solution tensor [batch_size, n_dim].
            t (Tensor): Homotopy parameter tensor [batch_size, 1].
        Returns:
            Tuple[Tensor, Tensor]: (Homotopy Value F, Gradient dF/dx)
        """
         # Ensure inputs are on the correct device and type
        x_input = x_input.to(device=t.device, dtype=torch.float64)
        t = t.to(device=x_input.device, dtype=torch.float64)

        x = x_input[:,0]
        y = x_input[:,1]
         # Homotopy parameter transformation (t_homotopy -> 0 means original problem)
        t_homotopy = 2.0 * (1 - t.squeeze(-1)) # Squeeze t to match dimensions

        # Analytical Homotopy Function F(x, y, t_homotopy) from repo
        F = x**4 + (2*y + 6*t_homotopy**2 - 21) * x**2 + (2*y**2 + 2*t_homotopy**2 - 14) * x \
            + y**4 + (6*t_homotopy**2 - 13) * y**2 + (2*t_homotopy**2 - 22) * y \
            + (6*t_homotopy**4 - 34*t_homotopy**2 + 170)

        # Analytical Gradients dF/dx, dF/dy from repo
        dFdx = 4*x**3 + 2*(2*y + 6*t_homotopy**2 - 21)*x + (2*y**2 + 2*t_homotopy**2 - 14)
        dFdy = 2*x**2 + 4*x*y + 4*y**3 + 2*(6*t_homotopy**2 - 13)*y + (2*t_homotopy**2 - 22)

        # Stack gradients into [batch_size, n_dim] tensor
        grad = torch.stack([dFdx, dFdy], dim=-1)

        return F, grad

# --- Standalone Original Objective Functions (for Plotting) ---
# These are needed because the repo's evaluate methods return the homotopy value F,
# not necessarily the original f(x) when t=1, and f(x) is needed for contour plots.

def ackley_func_orig(x_input):
    """Original Ackley function f(x) for plotting."""
    x_input = x_input.to(torch.float64)
    x = x_input[..., 0] # Use ... to handle potential batch dim
    y = x_input[..., 1]
    f = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) \
        - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))) \
        + np.e + 20
    return f

def himmelblau_func_orig(x_input):
    """Original Himmelblau function f(x) for plotting."""
    x_input = x_input.to(torch.float64)
    x = x_input[..., 0]
    y = x_input[..., 1]
    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return f

def rosenbrock_func_orig(x_input):
    """Original Rosenbrock function f(x) for plotting."""
    x_input = x_input.to(torch.float64)
    x = x_input[..., 0]
    y = x_input[..., 1]
    a = 1.0
    b = 100.0
    f = (a - x)**2 + b * (y - x**2)**2
    return f