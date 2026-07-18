"""Integration tests for the data pipeline flow."""

from pathlib import Path
import pandas as pd
from src.data.data_preprocessing import preprocess_data
from src.data.feature_engineering import engineer_features


def test_full_pipeline_flow(tmp_path: Path) -> None:
    """Tests the preprocessing and feature engineering steps sequentially."""
    # Create raw data frame
    df_raw = pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel", "Resort Hotel"],
            "lead_time": [10, 20, 5],
            "arrival_date_year": [2015, 2016, 2017],
            "arrival_date_month": ["July", "December", "January"],
            "arrival_date_day_of_month": [1, 15, 10],
            "stays_in_weekend_nights": [1, 0, 2],
            "stays_in_week_nights": [2, 3, 0],
            "adults": [2, 1, 0],  # Third row has 0 adults -> total guests = 0
            "children": [1, 0, 0],
            "babies": [0, 0, 0],
            "adr": [120.0, 150.0, -5.0],  # Third row has negative ADR
            "country": ["PRT", None, "ESP"],
            "agent": [9.0, None, 14.0],
            "company": [None, None, None],
        }
    )

    clean_path = tmp_path / "cleaned.csv"
    feat_path = tmp_path / "features.csv"

    # Preprocess
    df_clean = preprocess_data(df_raw, clean_path)

    # 3 rows raw.
    # Row 0: Valid (adr=120, total guests = 3). Expected to keep.
    # Row 1: Valid (adr=150, total guests = 1). Expected to keep.
    # Row 2: Invalid (adr=-5, total guests = 0). Expected to filter.
    assert len(df_clean) == 2
    assert clean_path.exists()

    # Children, agent, company, country null fills
    assert df_clean.loc[0, "children"] == 1
    assert df_clean.loc[1, "children"] == 0
    assert df_clean.loc[0, "country"] == "PRT"
    assert df_clean.loc[1, "country"] == "Unknown"
    assert df_clean.loc[0, "agent"] == 9
    assert df_clean.loc[1, "agent"] == 0
    assert df_clean.loc[0, "company"] == 0

    # Date parsing
    assert df_clean.loc[0, "arrival_date"] == pd.Timestamp("2015-07-01")
    assert df_clean.loc[1, "arrival_date"] == pd.Timestamp("2016-12-15")

    # Feature Engineering
    df_feat = engineer_features(df_clean, feat_path)
    assert len(df_feat) == 2
    assert feat_path.exists()

    # Features verification
    assert df_feat.loc[0, "total_nights"] == 3
    assert df_feat.loc[1, "total_nights"] == 3
    assert df_feat.loc[0, "total_guests"] == 3
    assert df_feat.loc[1, "total_guests"] == 1
    assert df_feat.loc[0, "is_family"] == 1
    assert df_feat.loc[1, "is_family"] == 0
    assert df_feat.loc[0, "arrival_season"] == "Summer"
    assert df_feat.loc[1, "arrival_season"] == "Winter"
