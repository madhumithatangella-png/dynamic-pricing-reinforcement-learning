"""Hotel Pricing Environment for Reinforcement Learning.

Models the hotel room pricing problem as a Markov Decision Process (MDP),
simulating room bookings and pricing dynamics over a sales horizon.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.config import (
    FEATURES_DATA_PATH,
    PRICE_ACTIONS,
    MAX_INVENTORY,
    MAX_DAYS,
    RL_RANDOM_SEED,
)
from src.rl.demand_simulator import DemandSimulator
from src.rl.reward import compute_step_reward

logger = get_logger("environment")


class StateEncoder:
    """Encodes and decodes environment states to numerical formats for RL agents.

    Allows easy expansion of the state space by modifying mapping dictionaries.
    """

    def __init__(self) -> None:
        """Initializes the State Encoder with standard category-to-index mappings."""
        self.season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
        self.hotel_map = {"Resort Hotel": 0, "City Hotel": 1}
        self.reverse_season_map = {v: k for k, v in self.season_map.items()}
        self.reverse_hotel_map = {v: k for k, v in self.hotel_map.items()}

    def encode(
        self,
        remaining_inventory: int,
        days_until_departure: int,
        arrival_season: str,
        booking_month: int,
        hotel_type: str,
    ) -> Tuple[int, int, int, int, int]:
        """Converts raw state variables into a numerical tuple for agent lookup.

        Args:
            remaining_inventory: Count of rooms remaining.
            days_until_departure: Days left in the sales horizon.
            arrival_season: Season name ('Spring', 'Summer', etc.).
            booking_month: Calendar month (1 to 12).
            hotel_type: Hotel category ('Resort Hotel' or 'City Hotel').

        Returns:
            Tuple of encoded state elements:
            (inventory, days, season_idx, month_idx, hotel_idx)
        """
        season_idx = self.season_map.get(arrival_season, 0)
        hotel_idx = self.hotel_map.get(hotel_type, 0)
        month_idx = max(0, min(11, booking_month - 1))  # 1-12 mapped to 0-11

        return (
            int(remaining_inventory),
            int(days_until_departure),
            int(season_idx),
            int(month_idx),
            int(hotel_idx),
        )


class HotelPricingEnvironment:
    """Gym-like environment representing hotel room dynamic pricing scenario."""

    def __init__(
        self,
        data_path: str = str(FEATURES_DATA_PATH),
        demand_simulator: Optional[DemandSimulator] = None,
        max_inventory: int = MAX_INVENTORY,
        max_days: int = MAX_DAYS,
        price_actions: list = PRICE_ACTIONS,
        random_seed: int = RL_RANDOM_SEED,
    ) -> None:
        """Initializes the environment.

        Args:
            data_path: Path to processed feature CSV.
            demand_simulator: Pre-fitted DemandSimulator instance.
            max_inventory: Total starting room capacity.
            max_days: Booking horizon window length.
            price_actions: List of discrete pricing options.
            random_seed: Seed for reproducibility.
        """
        self.data_path = Path(data_path)
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.price_actions = price_actions
        self.random_seed = random_seed

        self.state_encoder = StateEncoder()

        # Load data to compute arrival rates and sample contexts
        logger.info(f"Loading data from {self.data_path} to initialize environment...")
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(
            subset=["hotel", "arrival_season", "booking_month", "arrival_date"]
        )

        # Initialize Demand Simulator
        if demand_simulator is None:
            logger.info("Initializing and fitting new Demand Simulator...")
            self.demand_simulator = DemandSimulator(
                data_path=self.data_path, random_seed=self.random_seed
            )
            self.demand_simulator.fit()
        else:
            self.demand_simulator = demand_simulator

        # Setup RNG
        self.rng = np.random.default_rng(self.random_seed)

        # Precompute daily booking request arrival rates per group
        # Arrival rate lambda = average bookings per arrival_date / max_days
        logger.info("Precomputing booking arrival rates per segment...")
        grouped_dates = self.df.groupby(["hotel", "arrival_season", "booking_month"])[
            "arrival_date"
        ].nunique()
        grouped_bookings = self.df.groupby(["hotel", "arrival_season", "booking_month"]).size()
        self.arrival_rates = (grouped_bookings / grouped_dates) / self.max_days
        self.arrival_rates = self.arrival_rates.to_dict()

        # State attributes
        self.remaining_inventory = 0
        self.days_until_departure = 0
        self.arrival_season = ""
        self.booking_month = 1
        self.hotel_type = ""

        # Episode tracking variables
        self.total_revenue = 0.0
        self.rooms_sold = 0
        self.total_arrivals = 0
        self.prices_chosen = []

    def reset(self) -> Tuple[int, int, int, int, int]:
        """Resets environment state to start a new episode.

        Samples a target hotel segment context from historical data.

        Returns:
            Tuple: The initial encoded state.
        """
        # Sample a random booking row from historical data to establish context
        random_idx = self.rng.integers(0, len(self.df))
        sample_row = self.df.iloc[random_idx]

        self.hotel_type = str(sample_row["hotel"])
        self.arrival_season = str(sample_row["arrival_season"])
        self.booking_month = int(sample_row["booking_month"])

        # Pre-filter and cache segment records as dictionaries for faster sampling
        segment_df = self.df[
            (self.df["hotel"] == self.hotel_type)
            & (self.df["arrival_season"] == self.arrival_season)
            & (self.df["booking_month"] == self.booking_month)
        ]
        if segment_df.empty:
            segment_df = self.df
        
        # Keep only features required by simulator
        cols_to_keep = self.demand_simulator.categorical_cols + self.demand_simulator.numerical_cols
        self.segment_records = segment_df[cols_to_keep].to_dict(orient="records")

        # Reset inventories and times
        self.remaining_inventory = self.max_inventory
        self.days_until_departure = self.max_days

        # Reset episode trackers
        self.total_revenue = 0.0
        self.rooms_sold = 0
        self.total_arrivals = 0
        self.prices_chosen = []

        return self.get_state()

    def get_state(self) -> Tuple[int, int, int, int, int]:
        """Returns the current encoded state.

        Returns:
            Tuple: Encoded state tuple.
        """
        return self.state_encoder.encode(
            remaining_inventory=self.remaining_inventory,
            days_until_departure=self.days_until_departure,
            arrival_season=self.arrival_season,
            booking_month=self.booking_month,
            hotel_type=self.hotel_type,
        )

    def step(self, action_idx: int) -> Tuple[Tuple[int, int, int, int, int], float, bool, Dict[str, Any]]:
        """Advances environment by one day using the offered pricing decision.

        Args:
            action_idx: Index of selected pricing action in PRICE_ACTIONS.

        Returns:
            Tuple of (next_encoded_state, reward, done, info_dict)
        """
        if not (0 <= action_idx < len(self.price_actions)):
            raise ValueError(f"Action index {action_idx} is out of bounds.")

        price = self.price_actions[action_idx]
        self.prices_chosen.append(price)

        # 1. Simulate arrivals
        # Lookup Poisson rate for current segment
        segment_key = (self.hotel_type, self.arrival_season, self.booking_month)
        # fallback rate: average 0.5 requests per day if segment has no rates
        lambda_rate = self.arrival_rates.get(segment_key, 0.5)

        # Draw number of request arrivals
        num_arrivals = self.rng.poisson(lambda_rate)
        self.total_arrivals += num_arrivals

        rooms_sold_this_step = 0
        expected_prices_list = []

        for _ in range(num_arrivals):
            if self.remaining_inventory <= 0:
                break

            # Sample customer features from pre-filtered list (extremely fast)
            cust_sample = self.rng.choice(self.segment_records).copy()

            # Align time features
            cust_sample["lead_time"] = self.days_until_departure
            cust_sample["booking_window"] = self.days_until_departure

            # Record expected ADR for pricing penalty calculation
            expected_price = self.demand_simulator.predict_expected_price(cust_sample)
            expected_prices_list.append(expected_price)

            # Check if booking is accepted
            is_booked = self.demand_simulator.simulate_booking(price, cust_sample)
            if is_booked:
                rooms_sold_this_step += 1
                self.remaining_inventory -= 1
                self.rooms_sold += 1
                self.total_revenue += price

        # Decrement time
        self.days_until_departure -= 1

        # Check termination conditions
        done = (self.remaining_inventory <= 0) or (self.days_until_departure <= 0)

        # Calculate mean expected price for overpricing penalty
        mean_expected_price = float(np.mean(expected_prices_list)) if expected_prices_list else 100.0

        # Calculate step reward using reward.py helper
        reward = compute_step_reward(
            price=price,
            rooms_sold=rooms_sold_this_step,
            is_terminal=done,
            remaining_inventory=self.remaining_inventory,
            max_inventory=self.max_inventory,
            expected_price=mean_expected_price,
        )

        info = {
            "price": price,
            "arrivals": num_arrivals,
            "rooms_sold_step": rooms_sold_this_step,
            "revenue_step": price * rooms_sold_this_step,
            "remaining_inventory": self.remaining_inventory,
            "days_until_departure": self.days_until_departure,
        }

        return self.get_state(), reward, done, info

    def render(self) -> None:
        """Prints current environment state diagnostics."""
        logger.info(
            f"Env: [Inventory: {self.remaining_inventory}/{self.max_inventory}] "
            f"[Days Left: {self.days_until_departure}] "
            f"[Hotel: {self.hotel_type}] [Season: {self.arrival_season}] "
            f"[Total Revenue: {self.total_revenue}]"
        )

    def close(self) -> None:
        """Performs cleanup if necessary."""
        pass
