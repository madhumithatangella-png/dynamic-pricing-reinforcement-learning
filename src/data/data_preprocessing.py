"""Data Preprocessing module for Dynamic Pricing RL.

Fills missing values, creates unified datetime columns, and filters dataset
outliers.
"""

from pathlib import Path
import pandas as pd
from src.logger import get_logger
from src.config import CLEANED_DATA_PATH, OUTLIER_THRESHOLDS

# Opt-in to pandas future downcasting behavior to suppress warnings
pd.set_option("future.no_silent_downcasting", True)

logger = get_logger("data_preprocessing")

# Month name mapping to month numbers
MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def preprocess_data(
    df: pd.DataFrame, output_path: Path = CLEANED_DATA_PATH
) -> pd.DataFrame:
    """Preprocesses the raw dataset.

    Fills missing values, parses arrival dates, removes outliers, and saves the
    cleaned dataset.

    Args:
        df: Input raw pandas DataFrame.
        output_path: Path where the cleaned CSV will be written.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    logger.info("Starting preprocessing step...")
    df_clean = df.copy()

    # 1. Fill missing values
    # children -> 0
    if "children" in df_clean.columns:
        df_clean["children"] = df_clean["children"].fillna(0).astype(int)

    # country -> Unknown
    if "country" in df_clean.columns:
        df_clean["country"] = df_clean["country"].fillna("Unknown")

    # agent -> 0
    if "agent" in df_clean.columns:
        df_clean["agent"] = df_clean["agent"].fillna(0).astype(int)

    # company -> 0
    if "company" in df_clean.columns:
        df_clean["company"] = df_clean["company"].fillna(0).astype(int)

    logger.info("Missing values filled: children (0), country (Unknown), agent (0), company (0).")

    # 2. Create arrival_date column
    try:
        month_nums = df_clean["arrival_date_month"].map(MONTH_MAP)
        # Check if there are any unmapped months
        if month_nums.isna().any():
            unmapped = df_clean.loc[month_nums.isna(), "arrival_date_month"].unique()
            logger.warning(f"Unmapped month values found: {unmapped}")

        # Assemble string representation
        date_str = (
            df_clean["arrival_date_year"].astype(str)
            + "-"
            + month_nums.fillna(1).astype(int).astype(str)
            + "-"
            + df_clean["arrival_date_day_of_month"].astype(str)
        )
        df_clean["arrival_date"] = pd.to_datetime(date_str, format="%Y-%m-%d")
        logger.info("Created 'arrival_date' datetime column.")
    except Exception as e:
        err_msg = f"Failed to assemble and parse 'arrival_date': {e}"
        logger.error(err_msg)
        raise RuntimeError(err_msg) from e

    # 3. Outlier and invalid rows removal
    initial_rows = len(df_clean)

    # Calculate total guests for filtering purposes
    # Note: total_guests will be permanently engineered in the feature engineering module,
    # but we compute a helper column here to do validation filtering.
    temp_guests = df_clean["adults"] + df_clean["children"] + df_clean["babies"]

    # Filter rules
    adr_min = OUTLIER_THRESHOLDS.get("adr_min", 0.0)
    adr_max = OUTLIER_THRESHOLDS.get("adr_max", 1000.0)
    min_guests = OUTLIER_THRESHOLDS.get("min_guests", 1)

    # Apply masks
    valid_adr = (df_clean["adr"] >= adr_min) & (df_clean["adr"] <= adr_max)
    valid_guests = temp_guests >= min_guests

    # Outlier log details
    adr_outliers_count = (~valid_adr).sum()
    guests_outliers_count = (~valid_guests).sum()

    df_clean = df_clean[valid_adr & valid_guests]
    removed_rows = initial_rows - len(df_clean)

    logger.info(
        f"Filtered outliers: ADR < {adr_min} or ADR > {adr_max} removed "
        f"{adr_outliers_count} rows. Total Guests < {min_guests} removed "
        f"{guests_outliers_count} rows. "
        f"Total rows removed in preprocessing: {removed_rows} (out of {initial_rows})."
    )

    # 4. Save cleaned dataset
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        logger.info(
            f"Preprocessed and cleaned dataset saved to: {output_path}. Shape: {df_clean.shape}"
        )
    except Exception as e:
        logger.error(f"Failed to write preprocessed dataset to disk: {e}")
        raise

    return df_clean
