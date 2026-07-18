"""Unit tests for the data loader module."""

from pathlib import Path
import pandas as pd
import pytest
from src.data.data_loader import load_raw_data


def test_load_raw_data_valid(tmp_path: Path) -> None:
    """Verifies that load_raw_data correctly loads a valid CSV file."""
    csv_file = tmp_path / "valid_bookings.csv"
    dummy_data = pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel"],
            "is_canceled": [0, 1],
            "lead_time": [10, 5],
        }
    )
    dummy_data.to_csv(csv_file, index=False)

    df = load_raw_data(csv_file)
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == ["hotel", "is_canceled", "lead_time"]


def test_load_raw_data_file_not_found() -> None:
    """Verifies that load_raw_data raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("non_existent_directory/file.csv"))


def test_load_raw_data_empty_file(tmp_path: Path) -> None:
    """Verifies that load_raw_data raises ValueError for empty files."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    with pytest.raises(ValueError):
        load_raw_data(empty_file)
