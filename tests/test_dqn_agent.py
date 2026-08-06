"""Unit tests for the DQNAgent class."""

import pytest
from pathlib import Path
import torch
import numpy as np
from src.dqn.dqn_agent import DQNAgent


@pytest.fixture
def agent_kwargs() -> dict:
    """Fixture providing standard DQN Agent keyword args."""
    return {
        "state_dim": 5,
        "action_dim": 3,
        "price_actions": [3000.0, 4000.0, 5000.0],
        "max_inventory": 5,
        "max_days": 10,
        "batch_size": 2,
        "memory_size": 10,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9,
        "min_epsilon": 0.1,
        "hidden_units": [8, 8],
        "device": "cpu",
        "random_seed": 42,
    }


def test_dqn_agent_init(agent_kwargs) -> None:
    """Verifies that DQNAgent initializes parameters, networks and buffer correctly."""
    agent = DQNAgent(**agent_kwargs)
    assert agent.state_dim == 5
    assert agent.action_dim == 3
    assert agent.epsilon == 1.0
    assert len(agent.memory) == 0
    assert agent.device.type == "cpu"


def test_dqn_agent_state_preprocessing(agent_kwargs) -> None:
    """Verifies state normalization mapping function."""
    agent = DQNAgent(**agent_kwargs)
    state = (5, 10, 0, 0, 0)
    tensor = agent._preprocess_state(state)
    assert tensor.shape == (5,)
    # inventory = 5/5 = 1.0, days = 10/10 = 1.0, others = 0
    assert torch.allclose(tensor, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


def test_dqn_choose_action(agent_kwargs) -> None:
    """Verifies exploration and exploitation decision branches."""
    # 1. Test pure exploration (epsilon = 1.0)
    agent = DQNAgent(**agent_kwargs)
    actions = [agent.choose_action((2, 2, 0, 0, 0), use_epsilon=True) for _ in range(100)]
    assert len(set(actions)) > 1  # Should explore all action index range

    # 2. Test pure exploitation (epsilon = 0.0)
    agent_kwargs["epsilon"] = 0.0
    agent_exploit = DQNAgent(**agent_kwargs)
    state = (2, 2, 0, 0, 0)
    action_idx = agent_exploit.choose_action(state, use_epsilon=False)
    assert action_idx in [0, 1, 2]


def test_dqn_remember_and_train_step(agent_kwargs) -> None:
    """Verifies pushing items to memory and executing gradient steps."""
    agent = DQNAgent(**agent_kwargs)

    state = (5, 10, 0, 0, 0)
    action = 1
    reward = 50.0
    next_state = (4, 9, 0, 0, 0)
    done = False

    # Check memory push
    agent.remember(state, action, reward, next_state, done)
    assert len(agent.memory) == 1

    # Batch size is 2, length is 1: train_step should return 0 loss and skip
    loss_empty = agent.train_step()
    assert loss_empty == 0.0

    # Add second element to trigger training step
    agent.remember(state, action, reward, next_state, done)
    assert len(agent.memory) == 2

    # Check that optimization returns a real loss value and runs without error
    loss_val = agent.train_step()
    assert isinstance(loss_val, float)
    assert loss_val > 0.0


def test_dqn_update_target_network(agent_kwargs) -> None:
    """Verifies target network syncing updates weights to match policy."""
    agent = DQNAgent(**agent_kwargs)
    # Manually modify policy network parameter to make them mismatch
    with torch.no_grad():
        list(agent.q_net.parameters())[0].fill_(99.0)

    # They should differ
    policy_params = list(agent.q_net.parameters())[0]
    target_params = list(agent.target_net.parameters())[0]
    assert not torch.allclose(policy_params, target_params)

    # Sync
    agent.update_target_network()
    target_params_updated = list(agent.target_net.parameters())[0]
    assert torch.allclose(policy_params, target_params_updated)


def test_dqn_save_load_checkpoint(agent_kwargs, tmp_path: Path) -> None:
    """Verifies state serialization load/save mechanisms."""
    agent = DQNAgent(**agent_kwargs)
    # Mark policy model weights
    with torch.no_grad():
        list(agent.q_net.parameters())[0].fill_(4.2)

    model_file = tmp_path / "model.pth"
    opt_file = tmp_path / "opt.pth"

    agent.save(model_file, opt_file)
    assert model_file.exists()
    assert opt_file.exists()

    # Load into another agent
    agent_new = DQNAgent(**agent_kwargs)
    assert not torch.allclose(
        list(agent_new.q_net.parameters())[0],
        list(agent.q_net.parameters())[0]
    )

    agent_new.load(model_file, opt_file)
    assert torch.allclose(
        list(agent_new.q_net.parameters())[0],
        list(agent.q_net.parameters())[0]
    )
