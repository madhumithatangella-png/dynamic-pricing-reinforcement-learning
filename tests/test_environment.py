"""Unit tests for the HotelPricingEnvironment and StateEncoder."""

from pathlib import Path
import pytest
from src.rl.environment import HotelPricingEnvironment, StateEncoder
from src.rl.demand_simulator import DemandSimulator
from src.config import FEATURES_DATA_PATH, PRICE_ACTIONS


def test_state_encoder() -> None:
    """Verifies that StateEncoder converts categorical states accurately."""
    encoder = StateEncoder()

    # Spring, Resort Hotel, booking_month=4 (Spring is 0, Resort is 0, month mapped to 3)
    encoded = encoder.encode(
        remaining_inventory=50,
        days_until_departure=100,
        arrival_season="Spring",
        booking_month=4,
        hotel_type="Resort Hotel",
    )
    assert encoded == (50, 100, 0, 3, 0)

    # Winter, City Hotel, booking_month=12 (Winter is 3, City is 1, month mapped to 11)
    encoded_winter = encoder.encode(
        remaining_inventory=10,
        days_until_departure=5,
        arrival_season="Winter",
        booking_month=12,
        hotel_type="City Hotel",
    )
    assert encoded_winter == (10, 5, 3, 11, 1)


def test_environment_reset_and_step() -> None:
    """Verifies environment reset and step transitions."""
    if not FEATURES_DATA_PATH.exists():
        pytest.skip(f"Processed dataset not found at {FEATURES_DATA_PATH}, skipping integration test.")

    # Initialize environment
    env = HotelPricingEnvironment(
        data_path=str(FEATURES_DATA_PATH),
        max_inventory=50,
        max_days=100,
        price_actions=PRICE_ACTIONS,
        random_seed=42,
    )

    # Test Reset
    state = env.reset()
    assert len(state) == 5
    assert isinstance(state, tuple)
    assert state[0] == 50  # inventory
    assert state[1] == 100  # days until departure
    assert state[2] in [0, 1, 2, 3]  # season idx
    assert 0 <= state[3] < 12  # month idx
    assert state[4] in [0, 1]  # hotel idx

    # Test Step
    action_idx = 4  # median price
    next_state, reward, done, info = env.step(action_idx)

    assert len(next_state) == 5
    assert next_state[1] == 99  # days decremented
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "price" in info
    assert "arrivals" in info
    assert "rooms_sold_step" in info


def test_environment_done_condition() -> None:
    """Verifies that the episode ends when inventory or days left are 0."""
    if not FEATURES_DATA_PATH.exists():
        pytest.skip(f"Processed dataset not found at {FEATURES_DATA_PATH}, skipping test.")

    env = HotelPricingEnvironment(
        data_path=str(FEATURES_DATA_PATH),
        max_inventory=1000,
        max_days=3,
        price_actions=PRICE_ACTIONS,
        random_seed=42,
    )

    state = env.reset()
    assert state[0] == 1000  # inventory
    assert state[1] == 3  # days_left

    # Step 1
    state, reward, done, info = env.step(4)
    assert not done

    # Step 2
    state, reward, done, info = env.step(4)
    assert not done

    # Step 3 (reaches 3 days sales window limit)
    state, reward, done, info = env.step(4)
    assert done
    assert state[1] == 0
