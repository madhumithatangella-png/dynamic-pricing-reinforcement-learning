"""Unit tests for the data validator module."""

from pathlib import Path
import pandas as pd
from src.data.data_validator import DataValidator


def test_data_validator_run(tmp_path: Path) -> None:
    """Verifies that DataValidator profiles correctly and creates output files."""
    validator = DataValidator(output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "hotel": ["City Hotel", "City Hotel", "Resort Hotel"],
            "adr": [120.0, 120.0, 80.0],
            "lead_time": [5, 5, 20],
            "adults": [2, 2, 1],
        }
    )

    report = validator.run_validation(df)

    # Assert report contents
    assert "shape" in report
    assert report["shape"]["rows"] == 3
    assert report["shape"]["columns"] == 4

    assert "duplicates" in report
    # Rows 0 and 1 are duplicate records
    assert report["duplicates"]["duplicate_count"] == 1

    assert "missing_values" in report
    assert report["missing_values"]["total_missing_cells"] == 0

    assert "column_datatypes" in report
    assert "basic_statistics" in report

    # Verify report files creation
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "validation_report.txt").exists()
