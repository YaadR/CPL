# --- model.py (Aligned with Repo Version) ---
"""
A simple FC Continuation Path model.
Aligned with the CPL repository version.
"""
import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    """
    Neural Network model mapping homotopy parameter t to solution x.
    Matches the architecture from the CPL repository.
    """
    def __init__(self, n_dim):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim

        # Define fully connected layers
        self.fc1 = nn.Linear(1, 256) # Input: t (scalar), Output: 256
        self.fc2 = nn.Linear(256, 256) # Hidden layer
        self.fc3 = nn.Linear(256, self.n_dim) # Output: x (n_dim vector)

    def forward(self, t):
        """Forward pass: t -> x"""
        # Ensure t is in the correct shape [batch_size, 1]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # Pass through network with ReLU activations
        x = torch.relu(self.fc1(t))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Cast output to float64 as done in the repo version
        return x.to(torch.float64)