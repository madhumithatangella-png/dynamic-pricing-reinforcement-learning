"""Unit tests for the QNetwork architecture."""

import pytest
import torch
from src.dqn.network import QNetwork


def test_q_network_initialization() -> None:
    """Verifies that the network creates successfully with exactly two hidden layers configuration."""
    model = QNetwork(input_dim=5, output_dim=9, hidden_units=[64, 32])
    assert len(model.network) == 5  # 3 Linear layers + 2 ReLU layers = 5 elements in nn.Sequential
    
    # Test incorrect configuration: hidden units count must be exactly 2
    with pytest.raises(ValueError):
        QNetwork(input_dim=5, output_dim=9, hidden_units=[64])

    with pytest.raises(ValueError):
        QNetwork(input_dim=5, output_dim=9, hidden_units=[64, 64, 64])


def test_q_network_forward_shape() -> None:
    """Verifies output shape dimensions for different inputs batches."""
    input_dim = 5
    output_dim = 9
    hidden_units = [64, 64]
    
    model = QNetwork(input_dim, output_dim, hidden_units)
    
    # Single sample pass (batch size 1)
    x_single = torch.randn(1, input_dim)
    y_single = model(x_single)
    assert y_single.shape == (1, output_dim)
    
    # Batch pass (batch size 32)
    x_batch = torch.randn(32, input_dim)
    y_batch = model(x_batch)
    assert y_batch.shape == (32, output_dim)
