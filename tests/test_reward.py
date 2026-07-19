"""Unit tests for reinforcement learning reward functions."""

from src.rl.reward import (
    calculate_revenue,
    calculate_booking_bonus,
    calculate_occupancy_bonus,
    calculate_unsold_room_penalty,
    calculate_overpricing_penalty,
    compute_step_reward,
)


def test_calculate_revenue() -> None:
    """Verifies that revenue calculation is correct."""
    assert calculate_revenue(5000.0, 2) == 10000.0
    assert calculate_revenue(3000.0, 0) == 0.0


def test_calculate_booking_bonus() -> None:
    """Verifies that booking confirmations return correct bonus."""
    assert calculate_booking_bonus(3, 10.0) == 30.0
    assert calculate_booking_bonus(0, 10.0) == 0.0


def test_calculate_occupancy_bonus() -> None:
    """Verifies that occupancy-based bonus is correctly scaled."""
    assert calculate_occupancy_bonus(25, 50, 20.0) == 10.0  # 50% occupancy -> 10.0
    assert calculate_occupancy_bonus(0, 50, 20.0) == 0.0
    assert calculate_occupancy_bonus(50, 50, 20.0) == 20.0


def test_calculate_unsold_room_penalty() -> None:
    """Verifies that unsold rooms penalty is calculated correctly."""
    assert calculate_unsold_room_penalty(10, 15.0) == 150.0
    assert calculate_unsold_room_penalty(0, 15.0) == 0.0


def test_calculate_overpricing_penalty() -> None:
    """Verifies that charging above expected price triggers appropriate penalty."""
    # price 6000.0, expected_price 100.0, price_scale 50.0
    # price/scale = 120.0, excess = 20.0
    # penalty = 20 * 2.0 = 40.0
    assert calculate_overpricing_penalty(6000.0, 100.0, 50.0, 2.0) == 40.0

    # Under expected price: no penalty
    assert calculate_overpricing_penalty(4000.0, 100.0, 50.0, 2.0) == 0.0


def test_compute_step_reward() -> None:
    """Verifies combined step reward calculation works properly."""
    # Non-terminal step
    reward = compute_step_reward(
        price=5000.0,
        rooms_sold=1,
        is_terminal=False,
        remaining_inventory=49,
        max_inventory=50,
        expected_price=100.0,
        price_scale=50.0,
        revenue_scale=100.0,
        booking_bonus_val=10.0,
        occupancy_multiplier=20.0,
        unsold_penalty_val=15.0,
        overprice_multiplier=2.0,
    )
    # revenue: 5000 / 100 = 50.0
    # booking bonus: 1 * 10 = 10.0
    # occupancy: (50-49)/50 * 20 = 0.4
    # overpricing: 5000/50 = 100, expected=100 -> 0 penalty
    # expected total reward = 50 + 10 + 0.4 = 60.4
    assert round(reward, 2) == 60.40

    # Terminal step with unsold rooms
    reward_terminal = compute_step_reward(
        price=5000.0,
        rooms_sold=0,
        is_terminal=True,
        remaining_inventory=10,
        max_inventory=50,
        expected_price=100.0,
        price_scale=50.0,
        revenue_scale=100.0,
        booking_bonus_val=10.0,
        occupancy_multiplier=20.0,
        unsold_penalty_val=15.0,
        overprice_multiplier=2.0,
    )
    # revenue: 0
    # booking bonus: 0
    # occupancy: (50-10)/50 * 20 = 16.0
    # overpricing: 5000/50 = 100, expected = 100 -> 0
    # unsold penalty: 10 * 15.0 = 150.0
    # total = 16.0 - 150.0 = -134.0
    assert reward_terminal == -134.0
