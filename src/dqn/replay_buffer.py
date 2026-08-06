"""Replay Buffer implementation for DQN Agent.

Stores transitions (state, action, reward, next_state, done) and allows sampling
random batches of experiences to break temporal correlations.
"""

import random
from collections import deque
from typing import Tuple, List, Any


class ReplayBuffer:
    """Experience replay buffer using a double-ended queue (deque)."""

    def __init__(self, capacity: int) -> None:
        """Initializes the ReplayBuffer.

        Args:
            capacity: Maximum number of experiences to hold.
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Tuple[int, int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int, int],
        done: bool,
    ) -> None:
        """Adds a transition to the replay buffer.

        Args:
            state: The current state tuple.
            action: Action index selected by the agent.
            reward: Reward received for taking the action.
            next_state: The resulting next state tuple.
            done: A boolean flag indicating if the episode has terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> List[Tuple[Tuple[int, int, int, int, int], int, float, Tuple[int, int, int, int, int], bool]]:
        """Samples a random batch of experiences from the buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of transition tuples.
        """
        return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        """Clears all elements in the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Returns the current number of experiences stored in the buffer.

        Returns:
            int: The size of the buffer.
        """
        return len(self.buffer)
