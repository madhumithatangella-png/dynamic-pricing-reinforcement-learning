"""Unit tests for Q-Learning Agent."""

import pytest
from pathlib import Path
import numpy as np
from src.rl.q_learning import QLearningAgent


def test_q_learning_agent_init() -> None:
    """Verifies agent Q-table shape and initialization values."""
    price_actions = [3000.0, 4000.0, 5000.0]
    agent = QLearningAgent(
        price_actions=price_actions,
        max_inventory=5,
        max_days=10,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
    )

    # Q-table shape: (inventory+1=6, days+1=11, seasons=4, months=12, hotels=2, actions=3)
    expected_shape = (6, 11, 4, 12, 2, 3)
    assert agent.q_table.shape == expected_shape
    assert np.all(agent.q_table == 0.0)


def test_choose_action() -> None:
    """Verifies action selection behavior under exploration and exploitation."""
    price_actions = [3000.0, 4000.0, 5000.0]
    agent = QLearningAgent(
        price_actions=price_actions,
        max_inventory=5,
        max_days=10,
        epsilon=0.0,  # Pure exploitation
    )

    state = (5, 10, 1, 6, 0)
    # Set Q-value for action index 1 to be high
    agent.q_table[state][1] = 10.0

    action = agent.choose_action(state, use_epsilon=False)
    assert action == 1

    # With high epsilon, it should sometimes select other actions (explore)
    agent.epsilon = 1.0
    actions = [agent.choose_action(state, use_epsilon=True) for _ in range(100)]
    assert len(set(actions)) > 1  # Should explore multiple actions


def test_update_q_table() -> None:
    """Verifies Bellman update of Q-values."""
    price_actions = [3000.0, 4000.0, 5000.0]
    agent = QLearningAgent(
        price_actions=price_actions,
        max_inventory=5,
        max_days=10,
        alpha=0.1,
        gamma=0.9,
    )

    state = (5, 10, 1, 6, 0)
    action = 1
    reward = 50.0
    next_state = (4, 9, 1, 6, 0)

    # Set Q(s', a') max values
    agent.q_table[next_state][0] = 20.0
    agent.q_table[next_state][2] = 100.0  # max next state Q is 100.0

    # 1. Update when not done:
    # Target = reward + gamma * max_next_q = 50.0 + 0.9 * 100.0 = 140.0
    # Q_new = Q_old + alpha * (Target - Q_old) = 0.0 + 0.1 * (140.0 - 0.0) = 14.0
    updated_val = agent.update_q_table(state, action, reward, next_state, done=False)
    assert updated_val == 14.0
    assert agent.q_table[state][action] == 14.0

    # 2. Update when done:
    # Target = reward = 50.0
    # Q_new = Q_old + alpha * (Target - Q_old) = 14.0 + 0.1 * (50.0 - 14.0) = 14.0 + 3.6 = 17.6
    updated_val_done = agent.update_q_table(state, action, reward, next_state, done=True)
    assert round(updated_val_done, 2) == 17.6


def test_decay_epsilon() -> None:
    """Verifies epsilon decay is bounded by min_epsilon."""
    agent = QLearningAgent(epsilon=0.1, epsilon_decay=0.9, min_epsilon=0.05)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.09)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.081)
    # Bounded decay
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)


def test_save_load_q_table(tmp_path: Path) -> None:
    """Verifies model save and load works seamlessly."""
    agent = QLearningAgent(max_inventory=5, max_days=5)
    state = (2, 2, 0, 0, 0)
    agent.q_table[state][2] = 99.5

    save_file = tmp_path / "q_table.npy"
    agent.save(save_file)
    assert save_file.exists()

    # Load into another agent
    agent_new = QLearningAgent(max_inventory=5, max_days=5)
    assert agent_new.q_table[state][2] == 0.0

    agent_new.load(save_file)
    assert agent_new.q_table[state][2] == 99.5
