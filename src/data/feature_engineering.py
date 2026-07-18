"""Feature Engineering module for Dynamic Pricing RL.

Engineers strategic features relating to demand, length of stay, guest types,
booking windows, and booking seasonality.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.config import FEATURES_DATA_PATH

logger = get_logger("feature_engineering")

# Season mapping for months (1-12)
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def engineer_features(
    df: pd.DataFrame, output_path: Path = FEATURES_DATA_PATH
) -> pd.DataFrame:
    """Applies feature engineering formulas to the dataset.

    Args:
        df: Preprocessed input DataFrame.
        output_path: Path where the dataset with engineered features will be
          saved.

    Returns:
        pd.DataFrame: DataFrame containing all the newly engineered features.
    """
    logger.info("Starting feature engineering step...")
    df_feat = df.copy()

    # Make sure arrival_date is parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat["arrival_date"]):
        df_feat["arrival_date"] = pd.to_datetime(df_feat["arrival_date"])

    # 1. total_nights
    df_feat["total_nights"] = (
        df_feat["stays_in_weekend_nights"] + df_feat["stays_in_week_nights"]
    )

    # 2. total_guests
    df_feat["total_guests"] = (
        df_feat["adults"] + df_feat["children"] + df_feat["babies"]
    )

    # 3. is_family
    df_feat["is_family"] = (
        ((df_feat["children"] > 0) | (df_feat["babies"] > 0)).astype(int)
    )

    # 4. arrival_season
    df_feat["arrival_season"] = df_feat["arrival_date"].dt.month.map(SEASON_MAP)

    # 5. weekend_ratio
    # Handle division by zero when total_nights is 0
    df_feat["weekend_ratio"] = np.where(
        df_feat["total_nights"] > 0,
        df_feat["stays_in_weekend_nights"] / df_feat["total_nights"],
        0.0,
    )

    # 6. booking_window (direct map from lead_time)
    df_feat["booking_window"] = df_feat["lead_time"]

    # 7. is_weekend_only
    df_feat["is_weekend_only"] = (
        (df_feat["stays_in_weekend_nights"] > 0)
        & (df_feat["stays_in_week_nights"] == 0)
    ).astype(int)

    # 8. stay_length_category
    conditions = [
        df_feat["total_nights"] == 0,
        (df_feat["total_nights"] >= 1) & (df_feat["total_nights"] <= 3),
        (df_feat["total_nights"] >= 4) & (df_feat["total_nights"] <= 7),
        df_feat["total_nights"] >= 8,
    ]
    choices = ["Zero", "Short", "Medium", "Long"]
    df_feat["stay_length_category"] = np.select(
        conditions, choices, default="Short"
    )

    # 9, 10, 11. booking date properties
    # booking_date = arrival_date - lead_time (in days)
    booking_dates = df_feat["arrival_date"] - pd.to_timedelta(
        df_feat["lead_time"], unit="D"
    )
    df_feat["booking_month"] = booking_dates.dt.month
    df_feat["booking_year"] = booking_dates.dt.year
    df_feat["booking_dayofweek"] = booking_dates.dt.dayofweek

    logger.info(
        f"Engineered {len(df_feat.columns) - len(df.columns)} new features. "
        f"Columns added: ['total_nights', 'total_guests', 'is_family', 'arrival_season', "
        f"'weekend_ratio', 'booking_window', 'is_weekend_only', 'stay_length_category', "
        f"'booking_month', 'booking_year', 'booking_dayofweek']"
    )

    # Save to processed directory
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(output_path, index=False)
        logger.info(
            f"Dataset with engineered features saved to: {output_path}. Shape: {df_feat.shape}"
        )
    except Exception as e:
        logger.error(f"Failed to write engineered features dataset to disk: {e}")
        raise

    return df_feat
