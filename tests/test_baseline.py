"""Unit tests for baseline pricing agents."""

from src.rl.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent,
)


def test_fixed_pricing_agent() -> None:
    """Verifies FixedPricingAgent returns the target fixed action."""
    price_actions = [3000.0, 4000.0, 5000.0, 6000.0]
    agent = FixedPricingAgent(price_actions=price_actions, fixed_price=5000.0)

    # State: (inventory, days_left, season, month, hotel)
    state = (50, 100, 1, 6, 0)
    assert agent.choose_action(state) == 2  # 5000.0 index is 2

    # Even with different state, it should return the same action
    assert agent.choose_action((20, 5, 3, 11, 1)) == 2


def test_discount_pricing_agent() -> None:
    """Verifies DiscountPricingAgent reduces prices as departure day approaches."""
    price_actions = [3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0]
    agent = DiscountPricingAgent(price_actions=price_actions, max_days=100)

    # 1. High price when time ratio > 0.6 (e.g. days_left = 80)
    action_high = agent.choose_action((50, 80, 1, 6, 0))
    # ratio = 0.8 -> index int(9 * 0.8) = 7 (6500.0)
    assert action_high == 7

    # 2. Medium price when time ratio between 0.3 and 0.6 (e.g. days_left = 50)
    action_med = agent.choose_action((50, 50, 1, 6, 0))
    # ratio = 0.5 -> index int(9 * 0.5) = 4 (5000.0)
    assert action_med == 4

    # 3. Discount price when time ratio <= 0.3 (e.g. days_left = 10)
    action_low = agent.choose_action((50, 10, 1, 6, 0))
    # ratio = 0.1 -> index int(9 * 0.2) = 1 (3500.0)
    assert action_low == 1


def test_random_pricing_agent() -> None:
    """Verifies RandomPricingAgent returns valid actions within bounds."""
    price_actions = [3000.0, 4000.0, 5000.0]
    agent = RandomPricingAgent(price_actions=price_actions, random_seed=42)

    state = (50, 100, 1, 6, 0)
    for _ in range(50):
        action = agent.choose_action(state)
        assert 0 <= action < len(price_actions)
        assert isinstance(action, int)
