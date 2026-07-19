"""Reward calculation module for Dynamic Pricing RL.

Exposes modular and reusable reward components for the RL environment.
"""

from typing import Dict, Any


def calculate_revenue(price: float, rooms_sold: int) -> float:
    """Calculates the raw revenue generated.

    Args:
        price: Price per room.
        rooms_sold: Number of rooms sold in the step.

    Returns:
        float: Revenue amount.
    """
    return float(price * rooms_sold)


def calculate_booking_bonus(rooms_sold: int, bonus_per_room: float = 10.0) -> float:
    """Calculates a bonus for each successful booking.

    Args:
        rooms_sold: Number of rooms sold.
        bonus_per_room: Bonus reward per room sold.

    Returns:
        float: Booking bonus.
    """
    return float(rooms_sold * bonus_per_room)


def calculate_occupancy_bonus(
    current_occupancy: int, max_inventory: int, multiplier: float = 20.0
) -> float:
    """Calculates a bonus based on current occupancy rate.

    Args:
        current_occupancy: Cumulative rooms sold.
        max_inventory: Total initial inventory.
        multiplier: Scaling factor for the occupancy bonus.

    Returns:
        float: Occupancy bonus.
    """
    if max_inventory <= 0:
        return 0.0
    occupancy_rate = current_occupancy / max_inventory
    return float(occupancy_rate * multiplier)


def calculate_unsold_room_penalty(
    remaining_rooms: int, penalty_per_room: float = 15.0
) -> float:
    """Calculates a penalty for unsold inventory when the booking window closes.

    Args:
        remaining_rooms: Number of rooms left unsold.
        penalty_per_room: Penalty multiplier per unsold room.

    Returns:
        float: Unsold inventory penalty (positive value representing the cost).
    """
    return float(remaining_rooms * penalty_per_room)


def calculate_overpricing_penalty(
    price: float, expected_price: float, price_scale: float = 50.0, multiplier: float = 2.0
) -> float:
    """Calculates a penalty for excessive pricing relative to expected historical rates.

    Args:
        price: Offered room price (unscaled).
        expected_price: Historical expected price (unscaled-equivalent to historical ADR).
        price_scale: Scaling factor to align price with historical ADR.
        multiplier: Penalty factor.

    Returns:
        float: Overpricing penalty.
    """
    offered_scaled = price / price_scale
    excess = max(0.0, offered_scaled - expected_price)
    return float(excess * multiplier)


def compute_step_reward(
    price: float,
    rooms_sold: int,
    is_terminal: bool,
    remaining_inventory: int,
    max_inventory: int,
    expected_price: float,
    price_scale: float = 50.0,
    revenue_scale: float = 100.0,
    booking_bonus_val: float = 10.0,
    occupancy_multiplier: float = 20.0,
    unsold_penalty_val: float = 15.0,
    overprice_multiplier: float = 2.0,
) -> float:
    """Aggregates all reward components into a single step reward signal.

    Args:
        price: Offered price.
        rooms_sold: Rooms sold in this step.
        is_terminal: Whether the episode has ended.
        remaining_inventory: Remaining rooms in inventory.
        max_inventory: Initial room count.
        expected_price: Expected historical price for the arrivals.
        price_scale: Action price scaling factor.
        revenue_scale: Divisor to scale down raw revenue for reinforcement learning.
        booking_bonus_val: Reward for each booking.
        occupancy_multiplier: Multiplier for occupancy rate bonus.
        unsold_penalty_val: Penalty per unsold room at the end of episode.
        overprice_multiplier: Multiplier for charging above expected pricing.

    Returns:
        float: Combined reward.
    """
    # 1. Scaled Revenue
    raw_rev = calculate_revenue(price, rooms_sold)
    scaled_rev = raw_rev / revenue_scale

    # 2. Booking Success Bonus
    booking_bonus = calculate_booking_bonus(rooms_sold, booking_bonus_val)

    # 3. Occupancy Bonus (computed on current occupancy state)
    current_occupancy = max_inventory - remaining_inventory
    occupancy_bonus = calculate_occupancy_bonus(
        current_occupancy, max_inventory, occupancy_multiplier
    )

    # 4. Overpricing Penalty
    overprice_penalty = calculate_overpricing_penalty(
        price, expected_price, price_scale, overprice_multiplier
    )

    # Combine step rewards
    step_reward = scaled_rev + booking_bonus + occupancy_bonus - overprice_penalty

    # 5. Terminal Unsold Room Penalty
    if is_terminal:
        unsold_penalty = calculate_unsold_room_penalty(remaining_inventory, unsold_penalty_val)
        step_reward -= unsold_penalty

    return float(step_reward)
