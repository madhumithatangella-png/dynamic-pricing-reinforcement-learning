"""Unit tests for the DQN ReplayBuffer class."""

import pytest
from src.dqn.replay_buffer import ReplayBuffer


def test_replay_buffer_init() -> None:
    """Verifies that the replay buffer initializes empty and with correct size."""
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0


def test_replay_buffer_push_and_length() -> None:
    """Verifies that pushing transitions updates length correctly and capacity bounds it."""
    buffer = ReplayBuffer(capacity=3)

    state = (50, 100, 1, 6, 0)
    action = 2
    reward = 10.0
    next_state = (49, 99, 1, 6, 0)
    done = False

    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

    buffer.push(state, action, reward, next_state, done)
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 3

    # Exceed capacity: oldest should be dropped, size remains 3
    buffer.push(state, action, 99.0, next_state, done)
    assert len(buffer) == 3


def test_replay_buffer_sample() -> None:
    """Verifies that sampling retrieves correct batch dimensions and content."""
    buffer = ReplayBuffer(capacity=10)
    state = (50, 100, 1, 6, 0)
    action = 2
    reward = 10.0
    next_state = (49, 99, 1, 6, 0)
    done = False

    for i in range(5):
        buffer.push(state, action, reward + i, next_state, done)

    # Sample batch of 3
    batch = buffer.sample(3)
    assert len(batch) == 3
    for s, a, r, ns, d in batch:
        assert s == state
        assert a == action
        assert ns == next_state
        assert d == done
        assert r in [10.0, 11.0, 12.0, 13.0, 14.0]


def test_replay_buffer_clear() -> None:
    """Verifies that clearing removes all elements."""
    buffer = ReplayBuffer(capacity=10)
    buffer.push((1, 2, 3, 4, 0), 1, 10.0, (1, 2, 3, 4, 0), False)
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0
