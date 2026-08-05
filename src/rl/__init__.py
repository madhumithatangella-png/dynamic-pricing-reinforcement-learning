"""RL package for Hotel Dynamic Pricing.

Contains environments, simulators, rewards, agents, and training utilities.
"""

from src.rl.demand_simulator import DemandSimulator
from src.rl.environment import HotelPricingEnvironment
from src.rl.reward import (
    calculate_revenue,
    calculate_occupancy_bonus,
    calculate_unsold_room_penalty,
    calculate_overpricing_penalty,
    compute_step_reward,
)
from src.rl.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent,
)
from src.rl.q_learning import QLearningAgent

__all__ = [
    "DemandSimulator",
    "HotelPricingEnvironment",
    "calculate_revenue",
    "calculate_occupancy_bonus",
    "calculate_unsold_room_penalty",
    "calculate_overpricing_penalty",
    "compute_step_reward",
    "FixedPricingAgent",
    "DiscountPricingAgent",
    "RandomPricingAgent",
    "QLearningAgent",
]
