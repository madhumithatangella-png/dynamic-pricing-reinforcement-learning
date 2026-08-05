"""Demand Simulator component for Hotel Dynamic Pricing RL.

Uses historical booking data to estimate booking probabilities and expected prices
based on customer characteristics.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from src.logger import get_logger
from src.config import FEATURES_DATA_PATH, RL_RANDOM_SEED

logger = get_logger("demand_simulator")


class DemandSimulator:
    """Estimates booking behavior and expected price based on historical features."""

    def __init__(
        self,
        data_path: Path = FEATURES_DATA_PATH,
        random_seed: int = RL_RANDOM_SEED,
        price_sensitivity: float = 10.0,
        price_scale: float = 50.0,
    ) -> None:
        """Initializes the Demand Simulator.

        Args:
            data_path: Path to the processed features CSV.
            random_seed: Seed for reproducibility.
            price_sensitivity: Coefficient controlling price elasticity.
            price_scale: Scaling factor to align discrete pricing action space
              with historical ADR values.
        """
        self.data_path = Path(data_path)
        self.random_seed = random_seed
        self.price_sensitivity = price_sensitivity
        self.price_scale = price_scale

        self.categorical_cols = [
            "hotel",
            "arrival_season",
            "market_segment",
            "customer_type",
        ]
        self.numerical_cols = [
            "lead_time",
            "booking_month",
            "total_nights",
            "total_guests",
            "booking_window",
        ]

        self.regressor_pipeline: Optional[Pipeline] = None
        self.classifier_pipeline: Optional[Pipeline] = None
        self.rng = np.random.default_rng(self.random_seed)

    def fit(self) -> None:
        """Fits predictive models (Ridge + Logistic Regression) on historical data."""
        logger.info(f"Loading features data from {self.data_path} to fit demand models...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Drop rows with missing values in our target features
        required_cols = self.categorical_cols + self.numerical_cols + ["adr", "is_canceled"]
        df = df.dropna(subset=required_cols)

        # Feature matrix and targets
        X = df[self.categorical_cols + self.numerical_cols]
        y_price = df["adr"]
        y_confirm = 1 - df["is_canceled"]  # target: successful booking confirmation

        # Create preprocessing pipelines
        preprocessor_reg = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.numerical_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_cols,
                ),
            ]
        )

        preprocessor_cls = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.numerical_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_cols,
                ),
            ]
        )

        # Fit Regression for expected historical ADR (price)
        # We only fit on successful (non-canceled) bookings with positive adr
        confirmed_mask = (df["is_canceled"] == 0) & (df["adr"] > 0)
        X_reg = X[confirmed_mask]
        y_reg = y_price[confirmed_mask]

        logger.info(f"Fitting expected price Ridge regression model on {len(X_reg)} rows...")
        self.regressor_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_reg),
                ("regressor", Ridge(alpha=1.0, random_state=self.random_seed)),
            ]
        )
        self.regressor_pipeline.fit(X_reg, y_reg)

        # Fit Classification for baseline booking probability
        logger.info(f"Fitting baseline booking Logistic Regression on {len(X)} rows...")
        self.classifier_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_cls),
                ("classifier", LogisticRegression(max_iter=1000, random_state=self.random_seed)),
            ]
        )
        self.classifier_pipeline.fit(X, y_confirm)
        logger.info("Demand Simulator models fitted successfully.")

    def predict_expected_price(self, customer_features: Dict[str, Any]) -> float:
        """Estimates expected price for a set of customer features.

        Args:
            customer_features: Dictionary of historical feature values.

        Returns:
            float: Expected historical price (ADR).
        """
        if self.regressor_pipeline is None:
            raise RuntimeError("Demand Simulator model has not been fitted. Call fit() first.")

        df_input = pd.DataFrame([customer_features])
        pred = float(self.regressor_pipeline.predict(df_input)[0])
        return max(1.0, pred)  # Ensure expected price is strictly positive

    def get_booking_probability(
        self, price: float, customer_features: Dict[str, Any]
    ) -> float:
        """Calculates booking confirmation probability for a given price.

        Combines baseline booking likelihood with price elasticity.

        Args:
            price: Offered price (unscaled, from discrete price actions).
            customer_features: Dictionary of historical features.

        Returns:
            float: Booking probability between 0 and 1.
        """
        if self.classifier_pipeline is None or self.regressor_pipeline is None:
            raise RuntimeError("Models have not been fitted. Call fit() first.")

        # Scale offered price to match historical ADR scale
        offered_price_scaled = price / self.price_scale

        # Predict expected price
        expected_price = self.predict_expected_price(customer_features)

        # Predict baseline probability of confirmed booking
        df_input = pd.DataFrame([customer_features])
        baseline_prob = float(self.classifier_pipeline.predict_proba(df_input)[0][1])

        # Price acceptance factor modeled as a logistic curve relative to the price deviation
        # P(accept) = 1 / (1 + exp(sensitivity * (price_offered_scaled - expected_price) / expected_price))
        price_diff_ratio = (offered_price_scaled - expected_price) / expected_price
        price_accept_prob = 1.0 / (1.0 + np.exp(self.price_sensitivity * price_diff_ratio))

        booking_prob = baseline_prob * price_accept_prob
        return float(np.clip(booking_prob, 0.0, 1.0))

    def simulate_booking(self, price: float, customer_features: Dict[str, Any]) -> bool:
        """Simulates whether a booking occurs at the given price.

        Args:
            price: Offered price.
            customer_features: Dictionary of customer features.

        Returns:
            bool: True if booking succeeds, False otherwise.
        """
        prob = self.get_booking_probability(price, customer_features)
        return bool(self.rng.random() < prob)
