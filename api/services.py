"""Services module implementing ModelManagerService for prediction, XAI, and diagnostics."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import time
import pandas as pd
import numpy as np
import torch
from src.config import (
    FEATURES_DATA_PATH,
    MODELS_DIR,
    PRICE_ACTIONS,
    MAX_INVENTORY,
    MAX_DAYS,
    DQN_RANDOM_SEED,
    BATCH_SIZE,
    MEMORY_SIZE,
    LEARNING_RATE,
    GAMMA,
    EPSILON,
    EPSILON_DECAY,
    MIN_EPSILON,
    HIDDEN_UNITS,
)
from src.logger import get_logger
from src.rl.demand_simulator import DemandSimulator
from src.rl.q_learning import QLearningAgent
from src.dqn.dqn_agent import DQNAgent

logger = get_logger("model_service")


class ModelManagerService:
    """Manages ML/RL models loading, pricing predictions, and XAI recommendations."""

    def __init__(self, data_path: Path = FEATURES_DATA_PATH, models_dir: Path = MODELS_DIR) -> None:
        """Initializes the service and fits demand simulators.

        Args:
            data_path: Path to processed features dataset.
            models_dir: Path containing trained agent checkpoints.
        """
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.price_actions = PRICE_ACTIONS
        self.max_inventory = MAX_INVENTORY
        self.max_days = MAX_DAYS

        # Metrics trackers
        self.prediction_count = 0
        self.total_latency_ms = 0.0
        self.total_inference_time_ms = 0.0
        self.prediction_history: List[Dict[str, Any]] = []

        # 1. Fit Demand Simulator on startup
        logger.info("Initializing Service: Fitting Demand Simulator...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed dataset features CSV not found at {self.data_path}")

        self.demand_simulator = DemandSimulator(data_path=self.data_path, random_seed=42)
        self.demand_simulator.fit()

        # Load context database for customer feature retrieval
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(subset=["hotel", "arrival_season", "booking_month"])

        # 2. Instantiate and Load Agents (Without retraining if missing)
        self.q_learning_agent: Optional[QLearningAgent] = None
        self.dqn_agent: Optional[DQNAgent] = None

        self.q_table_path = self.models_dir / "q_table.npy"
        self.dqn_model_path = self.models_dir / "dqn_model.pth"
        self.optimizer_path = self.models_dir / "optimizer.pth"

        # Load Q-Learning
        if self.q_table_path.exists():
            logger.info("Loading pre-trained Q-table model...")
            self.q_learning_agent = QLearningAgent(
                price_actions=self.price_actions,
                max_inventory=self.max_inventory,
                max_days=self.max_days,
            )
            self.q_learning_agent.load(self.q_table_path)
        else:
            logger.warning(f"Q-table model file not found at {self.q_table_path}")

        # Load DQN
        if self.dqn_model_path.exists():
            logger.info("Loading pre-trained DQN model...")
            self.dqn_agent = DQNAgent(
                state_dim=5,
                action_dim=len(self.price_actions),
                price_actions=self.price_actions,
                max_inventory=self.max_inventory,
                max_days=self.max_days,
                batch_size=BATCH_SIZE,
                memory_size=MEMORY_SIZE,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
                hidden_units=HIDDEN_UNITS,
                device="cpu",  # Load on CPU for inference stability
                random_seed=DQN_RANDOM_SEED,
            )
            self.dqn_agent.load(self.dqn_model_path, self.optimizer_path)
        else:
            logger.warning(f"DQN model file not found at {self.dqn_model_path}")

        # Active model preference (default to dqn if available, else q_learning, else raise)
        if self.dqn_agent is not None:
            self.active_model_name = "dqn"
        elif self.q_learning_agent is not None:
            self.active_model_name = "q_learning"
        else:
            self.active_model_name = "none"
            logger.warning("No pre-trained models found. Server health is OK, but predictions will fail.")

    def get_customer_features(self, hotel_type: str, arrival_season: str, booking_month: int, days_until_departure: int) -> Dict[str, Any]:
        """Retrieves or synthesizes a customer features dict matching context inputs."""
        segment_df = self.df[
            (self.df["hotel"] == hotel_type)
            & (self.df["arrival_season"] == arrival_season)
            & (self.df["booking_month"] == booking_month)
        ]
        if segment_df.empty:
            segment_df = self.df

        # Choose the first record as a baseline representative
        sample_row = segment_df.iloc[0].to_dict()
        cols = self.demand_simulator.categorical_cols + self.demand_simulator.numerical_cols
        cust_features = {k: sample_row[k] for k in cols if k in sample_row}
        cust_features["lead_time"] = days_until_departure
        cust_features["booking_window"] = days_until_departure
        return cust_features

    def get_inventory_status(self, remaining_inventory: int) -> str:
        """Returns inventory status categorization."""
        ratio = remaining_inventory / self.max_inventory
        if ratio <= 0.15:
            return "Critical"
        elif ratio <= 0.50:
            return "Moderate"
        else:
            return "High"

    def generate_recommendation_reason(
        self,
        price: float,
        expected_price: float,
        remaining_inventory: int,
        days_until_departure: int,
        arrival_season: str,
    ) -> str:
        """Generates XAI natural language reasoning for the recommendation."""
        inventory_status = self.get_inventory_status(remaining_inventory)
        time_ratio = days_until_departure / self.max_days

        if inventory_status == "Critical":
            return (
                f"Critical inventory level remaining ({remaining_inventory} rooms left). "
                f"Charging a high premium price of ₹{int(price)} to maximize margin."
            )
        
        if time_ratio <= 0.15:
            if price < expected_price:
                return (
                    f"Close to booking horizon limit ({days_until_departure} days left) with moderate/high unsold rooms. "
                    f"Price discounted to ₹{int(price)} to stimulate late conversions."
                )
            else:
                return (
                    f"Close to departure deadline, but inventory remains limited. "
                    f"Offered defensive price of ₹{int(price)} to preserve revenue."
                )

        if price > expected_price * 1.15:
            return (
                f"Strong seasonal booking demand ({arrival_season}) matching high segment ADR expectation. "
                f"Pricing set to premium ₹{int(price)}."
            )
        elif price < expected_price * 0.85:
            return (
                f"Moderate demand context with high inventory left. "
                f"Reduced price to ₹{int(price)} to increase occupancy rates."
            )
        else:
            return (
                f"Normal baseline market conditions for the {arrival_season} segment. "
                f"Room price aligned with average expected market price of ₹{int(price)}."
            )

    def predict_price(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrates predictions, performance metrics and XAI generation.

        Args:
            request_data: Dictonary containing state options.

        Returns:
            Dict: PredictionResponse dictionary.
        """
        start_time = time.time()

        remaining_inventory = request_data["remaining_inventory"]
        days_until_departure = request_data["days_until_departure"]
        arrival_season = request_data["arrival_season"]
        booking_month = request_data["booking_month"]
        hotel_type = request_data["hotel_type"]
        model_type = request_data.get("model_type") or self.active_model_name

        # Encode state tuple
        from src.rl.environment import StateEncoder
        encoder = StateEncoder()
        state = encoder.encode(
            remaining_inventory=remaining_inventory,
            days_until_departure=days_until_departure,
            arrival_season=arrival_season,
            booking_month=booking_month,
            hotel_type=hotel_type,
        )

        # Inferences select
        inference_start = time.time()
        if model_type == "dqn":
            if self.dqn_agent is None:
                raise FileNotFoundError("Trained DQN model state checkpoint file (.pth) is not loaded.")
            action_idx = self.dqn_agent.predict(state)
        elif model_type == "q_learning":
            if self.q_learning_agent is None:
                raise FileNotFoundError("Trained Q-Learning lookup table checkpoint file (.npy) is not loaded.")
            action_idx = self.q_learning_agent.predict(state)
        else:
            raise ValueError(f"Selected model type '{model_type}' is unavailable or not loaded.")
        inference_end = time.time()

        recommended_price = self.price_actions[action_idx]

        # Calculate booking probabilities and expected revenue
        customer_features = self.get_customer_features(
            hotel_type=hotel_type,
            arrival_season=arrival_season,
            booking_month=booking_month,
            days_until_departure=days_until_departure,
        )
        booking_prob = self.demand_simulator.get_booking_probability(recommended_price, customer_features)
        expected_price = self.demand_simulator.predict_expected_price(customer_features) * 50.0 # Scale expectation back to unscaled
        expected_revenue = recommended_price * booking_prob

        # Meta attributes
        occupancy_rate = (self.max_inventory - remaining_inventory) / self.max_inventory
        inventory_status = self.get_inventory_status(remaining_inventory)
        reason = self.generate_recommendation_reason(
            price=recommended_price,
            expected_price=expected_price,
            remaining_inventory=remaining_inventory,
            days_until_departure=days_until_departure,
            arrival_season=arrival_season,
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0
        inference_ms = (inference_end - inference_start) * 1000.0

        # Update metrics
        self.prediction_count += 1
        self.total_latency_ms += latency_ms
        self.total_inference_time_ms += inference_ms

        response_dict = {
            "recommended_price": recommended_price,
            "expected_revenue": expected_revenue,
            "booking_probability": booking_prob,
            "inventory_status": inventory_status,
            "occupancy_estimate": occupancy_rate,
            "reason": reason,
            "model_used": model_type,
        }

        # Save to history (limited to 50 items)
        self.prediction_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request": request_data,
            "response": response_dict,
            "latency_ms": latency_ms
        })
        if len(self.prediction_history) > 50:
            self.prediction_history.pop(0)

        return response_dict
