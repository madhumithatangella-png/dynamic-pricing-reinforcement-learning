"""Baseline Agents module for Hotel Dynamic Pricing.

Implements Fixed, Discount, and Random pricing heuristics to compare with Q-learning.
"""

from typing import Tuple
import numpy as np
from src.config import PRICE_ACTIONS, RL_RANDOM_SEED


class FixedPricingAgent:
    """Agent that always selects a fixed room price (typically median price)."""

    def __init__(
        self, price_actions: list = PRICE_ACTIONS, fixed_price: float = 5000.0
    ) -> None:
        """Initializes the FixedPricingAgent.

        Args:
            price_actions: Available pricing choices.
            fixed_price: Target fixed price.
        """
        self.price_actions = price_actions
        # Choose action index closest to fixed_price
        self.action_idx = int(np.argmin(np.abs(np.array(price_actions) - fixed_price)))

    def choose_action(self, state: Tuple[int, int, int, int, int]) -> int:
        """Selects the pre-configured fixed action index.

        Args:
            state: Environment state.

        Returns:
            int: Action index.
        """
        return self.action_idx


class DiscountPricingAgent:
    """Agent that reduces prices as the departure date draws closer."""

    def __init__(self, price_actions: list = PRICE_ACTIONS, max_days: int = 100) -> None:
        """Initializes the DiscountPricingAgent.

        Args:
            price_actions: Available pricing choices.
            max_days: Start booking horizon limit.
        """
        self.price_actions = price_actions
        self.max_days = max_days
        self.num_actions = len(price_actions)

    def choose_action(self, state: Tuple[int, int, int, int, int]) -> int:
        """Selects price based on days until departure.

        Starts with maximum price and discounts step-wise as time runs out.

        Args:
            state: State tuple where index 1 is days_until_departure.

        Returns:
            int: Action index.
        """
        days_left = state[1]

        # Dynamic pricing rule:
        # >60% time left: High price (index in upper third)
        # 30-60% time left: Medium price (index in middle third)
        # <30% time left: Discount price (index in lower third)
        time_ratio = days_left / self.max_days

        if time_ratio > 0.6:
            # Pick a high price action
            return min(self.num_actions - 1, int(self.num_actions * 0.8))
        elif time_ratio > 0.3:
            # Pick a medium price action
            return min(self.num_actions - 1, int(self.num_actions * 0.5))
        else:
            # Pick a low/discount price action
            return min(self.num_actions - 1, int(self.num_actions * 0.2))


class RandomPricingAgent:
    """Agent that randomly selects pricing actions at each step."""

    def __init__(
        self, price_actions: list = PRICE_ACTIONS, random_seed: int = RL_RANDOM_SEED
    ) -> None:
        """Initializes the RandomPricingAgent.

        Args:
            price_actions: Available pricing choices.
            random_seed: Random seed.
        """
        self.price_actions = price_actions
        self.rng = np.random.default_rng(random_seed)

    def choose_action(self, state: Tuple[int, int, int, int, int]) -> int:
        """Randomly chooses a price action index.

        Args:
            state: Environment state.

        Returns:
            int: Action index.
        """
        return int(self.rng.integers(0, len(self.price_actions)))
