"""Neural Network architecture for Deep Q Network (DQN).

Implements a standard feedforward neural network in PyTorch with configurable
hidden layer units to predict Q-values from state inputs.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """Deep Q-Network (QNetwork) using PyTorch."""

    def __init__(self, input_dim: int, output_dim: int, hidden_units: List[int]) -> None:
        """Initializes the QNetwork.

        Constructs a feedforward network with:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output Q Values

        Args:
            input_dim: Dimension of the input state vector.
            output_dim: Dimension of the output action vector.
            hidden_units: A list of integers representing the number of hidden units.
                          Must contain exactly 2 integers to match the required architecture.
        """
        super().__init__()
        if len(hidden_units) != 2:
            raise ValueError(
                f"hidden_units must contain exactly 2 elements to match "
                f"the Linear -> ReLU -> Linear -> ReLU -> Linear architecture. "
                f"Got: {hidden_units}"
            )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass to compute action Q-values.

        Args:
            x: Input tensor representing preprocessed batch of states.

        Returns:
            torch.Tensor: Predicted Q-values for each action.
        """
        return self.network(x)
