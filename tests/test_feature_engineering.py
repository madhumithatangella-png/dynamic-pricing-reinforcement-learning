"""Unit tests for the feature engineering module."""

from pathlib import Path
import pandas as pd
from src.data.feature_engineering import engineer_features


def test_engineer_features(tmp_path: Path) -> None:
    """Verifies that all 11 target features are computed accurately."""
    df_preprocessed = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2015-07-01", "2015-12-15"]),
            "lead_time": [12, 0],
            "stays_in_weekend_nights": [2, 0],
            "stays_in_week_nights": [5, 0],
            "adults": [2, 1],
            "children": [1, 0],
            "babies": [0, 0],
        }
    )

    feat_path = tmp_path / "features.csv"
    df_feat = engineer_features(df_preprocessed, feat_path)

    # Verify column existence
    expected_cols = [
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
    for col in expected_cols:
        assert col in df_feat.columns

    # Row 0: stays_in_weekend_nights=2, stays_in_week_nights=5
    # total_nights = 7
    # total_guests = 2 + 1 + 0 = 3
    # is_family = True (1)
    # arrival_season = Summer (July)
    # weekend_ratio = 2/7 = 0.2857
    # booking_window = lead_time = 12
    # is_weekend_only = False (0)
    # stay_length_category = Medium (4 to 7 nights)
    # booking_date = 2015-07-01 - 12 days = 2015-06-19
    # booking_month = 6, booking_year = 2015
    assert df_feat.loc[0, "total_nights"] == 7
    assert df_feat.loc[0, "total_guests"] == 3
    assert df_feat.loc[0, "is_family"] == 1
    assert df_feat.loc[0, "arrival_season"] == "Summer"
    assert round(df_feat.loc[0, "weekend_ratio"], 4) == round(2 / 7, 4)
    assert df_feat.loc[0, "booking_window"] == 12
    assert df_feat.loc[0, "is_weekend_only"] == 0
    assert df_feat.loc[0, "stay_length_category"] == "Medium"
    assert df_feat.loc[0, "booking_month"] == 6
    assert df_feat.loc[0, "booking_year"] == 2015

    # Row 1: Zero nights stay, Winter, booking_date = 2015-12-15
    assert df_feat.loc[1, "total_nights"] == 0
    assert df_feat.loc[1, "total_guests"] == 1
    assert df_feat.loc[1, "is_family"] == 0
    assert df_feat.loc[1, "arrival_season"] == "Winter"
    assert df_feat.loc[1, "weekend_ratio"] == 0.0
    assert df_feat.loc[1, "is_weekend_only"] == 0
    assert df_feat.loc[1, "stay_length_category"] == "Zero"
    assert df_feat.loc[1, "booking_month"] == 12
    assert df_feat.loc[1, "booking_year"] == 2015

    # Check file exists
    assert feat_path.exists()
