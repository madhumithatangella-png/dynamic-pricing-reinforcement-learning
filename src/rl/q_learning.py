"""Q-Learning Agent implementation for Dynamic Pricing RL.

Implements tabular Q-learning with epsilon-greedy exploration, epsilon decay,
and serialization features.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.config import (
    PRICE_ACTIONS,
    MAX_INVENTORY,
    MAX_DAYS,
    ALPHA,
    GAMMA,
    EPSILON,
    EPSILON_DECAY,
    MIN_EPSILON,
    RL_RANDOM_SEED,
)

logger = get_logger("q_learning")


class QLearningAgent:
    """Tabular Q-Learning agent for dynamic pricing."""

    def __init__(
        self,
        price_actions: list = PRICE_ACTIONS,
        max_inventory: int = MAX_INVENTORY,
        max_days: int = MAX_DAYS,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
        min_epsilon: float = MIN_EPSILON,
        random_seed: int = RL_RANDOM_SEED,
    ) -> None:
        """Initializes the Q-Learning Agent.

        Args:
            price_actions: Available pricing actions.
            max_inventory: Maximum room capacity (state dim 1).
            max_days: Sales horizon length (state dim 2).
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration rate.
            epsilon_decay: Epsilon decay rate.
            min_epsilon: Minimum exploration rate.
            random_seed: Seed for reproducibility.
        """
        self.price_actions = price_actions
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.random_seed = random_seed

        # State Dimensions: (inventory+1, days+1, seasons(4), months(12), hotel_types(2))
        self.q_table_shape = (
            self.max_inventory + 1,
            self.max_days + 1,
            4,  # seasons
            12,  # months
            2,  # hotel types
            len(self.price_actions),
        )

        logger.info(f"Initializing Q-table with shape: {self.q_table_shape}")
        self.q_table = np.zeros(self.q_table_shape)

        self.rng = np.random.default_rng(self.random_seed)

    def choose_action(
        self, state: Tuple[int, int, int, int, int], use_epsilon: bool = True
    ) -> int:
        """Selects a pricing action index using epsilon-greedy strategy.

        Args:
            state: Encoded environment state tuple.
            use_epsilon: Whether to apply exploration.

        Returns:
            int: Selected action index.
        """
        if use_epsilon and (self.rng.random() < self.epsilon):
            # Explore: random action index
            return int(self.rng.integers(0, len(self.price_actions)))
        else:
            # Exploit: best known action index
            return int(np.argmax(self.q_table[state]))

    def update_q_table(
        self,
        state: Tuple[int, int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int, int],
        done: bool,
    ) -> float:
        """Performs Bellman update on the Q-table.

        Args:
            state: Encoded current state tuple.
            action: Action index chosen.
            reward: Step reward received.
            next_state: Encoded next state tuple.
            done: Termination flag.

        Returns:
            float: Updated Q-value for the (state, action) pair.
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        # Temporal difference update
        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        return float(self.q_table[state][action])

    def predict(self, state: Tuple[int, int, int, int, int]) -> int:
        """Greedy prediction of best action index (no exploration).

        Args:
            state: Encoded state tuple.

        Returns:
            int: Best action index.
        """
        return self.choose_action(state, use_epsilon=False)

    def decay_epsilon(self) -> None:
        """Decays the exploration rate epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath: Path) -> None:
        """Saves Q-table to a NumPy file.

        Args:
            filepath: Path to save the Q-table.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, self.q_table)
        logger.info(f"Q-table successfully saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Loads Q-table from a NumPy file.

        Args:
            filepath: Path to load the Q-table.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Q-table file not found at {filepath}")
        self.q_table = np.load(filepath)
        logger.info(f"Q-table successfully loaded from {filepath}. Shape: {self.q_table.shape}")

    def train(self, env: Any, episodes: int) -> pd.DataFrame:
        """Runs the complete training loop.

        Args:
            env: HotelPricingEnvironment instance.
            episodes: Number of episodes to train.

        Returns:
            pd.DataFrame: Log of training metrics.
        """
        metrics = []

        logger.info(f"Starting Q-Learning training for {episodes} episodes...")

        for ep in range(1, episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            # For Q-value tracking
            q_values_visited = []

            while not done:
                action_idx = self.choose_action(state, use_epsilon=True)
                q_values_visited.append(self.q_table[state][action_idx])

                next_state, reward, done, info = env.step(action_idx)
                self.update_q_table(state, action_idx, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            # Compute average Q-value visited in this episode
            avg_q = float(np.mean(q_values_visited)) if q_values_visited else 0.0

            # Record episode metrics
            metrics.append(
                {
                    "episode": ep,
                    "reward": total_reward,
                    "revenue": env.total_revenue,
                    "rooms_sold": env.rooms_sold,
                    "occupancy_rate": env.rooms_sold / env.max_inventory,
                    "epsilon": self.epsilon,
                    "average_q": avg_q,
                }
            )

            # Decay exploration rate
            self.decay_epsilon()

            if ep % 100 == 0 or ep == 1:
                logger.info(
                    f"Episode {ep:4d}/{episodes:4d} | Reward: {total_reward:8.2f} | "
                    f"Revenue: {env.total_revenue:8.0f} | Occupancy: {env.rooms_sold/env.max_inventory:5.2%} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        logger.info("Q-Learning training complete.")
        return pd.DataFrame(metrics)
