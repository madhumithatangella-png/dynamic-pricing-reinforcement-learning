"""Configuration module for Dynamic Pricing using RL.

Provides path resolution, project constants, feature lists, and validation/outlier
thresholds.
"""

from pathlib import Path
from typing import Dict, List, Any

# Project Root Detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hotel_bookings.csv"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "hotel_bookings_cleaned.csv"
FEATURES_DATA_PATH = DATA_DIR / "processed" / "hotel_bookings_features.csv"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_FILE_PATH = OUTPUT_DIR / "pipeline.log"
VALIDATION_JSON_PATH = REPORTS_DIR / "validation_report.json"
VALIDATION_TXT_PATH = REPORTS_DIR / "validation_report.txt"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure all directories exist
for directory in [DATA_DIR / "processed", FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42
RL_RANDOM_SEED = 42

# RL Configurations
PRICE_ACTIONS: List[float] = [3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0]
MAX_INVENTORY: int = 50
MAX_DAYS: int = 100
ALPHA: float = 0.1
GAMMA: float = 0.99
EPSILON: float = 1.0
EPSILON_DECAY: float = 0.995
MIN_EPSILON: float = 0.01
EPISODES: int = 1000
EVALUATION_EPISODES: int = 100

# Outlier Thresholds
OUTLIER_THRESHOLDS: Dict[str, Any] = {
    "adr_min": 0.0,
    "adr_max": 1000.0,
    "min_guests": 1,
}

# Feature Configurations
CATEGORICAL_COLS: List[str] = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "reservation_status",
]

NUMERICAL_COLS: List[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

DATE_COLS: List[str] = [
    "reservation_status_date",
]

ENGINEERED_COLS: List[str] = [
    "total_nights",
    "total_guests",
    "is_family",
    "arrival_season",
    "weekend_ratio",
    "booking_window",
    "is_weekend_only",
    "stay_length_category",
    "booking_month",
    "booking_year",
    "booking_dayofweek",
]
