"""Data Validator module for Dynamic Pricing RL.

Performs schema checking, duplicates detection, missing value analysis, and
statistical profiling.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.logger import get_logger
from src.config import REPORTS_DIR

logger = get_logger("data_validator")


class DataValidator:
    """Performs validation checks and generates reports on data frames."""

    def __init__(self, output_dir: Path = REPORTS_DIR):
        """Initializes DataValidator with target reports directory.

        Args:
            output_dir: Path to directory where reports will be saved.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs all validation checks on the dataset.

        Args:
            df: The pandas DataFrame to validate.

        Returns:
            Dict[str, Any]: Validation report results dictionary.
        """
        logger.info("Starting validation checks on the dataset...")

        report: Dict[str, Any] = {}

        # 1. Dataset Shape
        num_rows, num_cols = df.shape
        report["shape"] = {"rows": num_rows, "columns": num_cols}
        logger.info(f"Dataset shape: {num_rows} rows, {num_cols} columns")

        # 2. Missing Values Analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (df.isnull().sum() / num_rows) * 100
        missing_data = {
            col: {
                "missing_count": int(count),
                "missing_percentage": float(missing_percentages[col]),
            }
            for col, count in missing_counts.items()
            if count > 0
        }
        report["missing_values"] = {
            "total_missing_cells": int(missing_counts.sum()),
            "columns_with_missing": missing_data,
        }
        logger.info(
            f"Total missing values: {missing_counts.sum()} across columns: "
            f"{list(missing_data.keys())}"
        )

        # 3. Duplicate Rows
        duplicate_count = int(df.duplicated().sum())
        report["duplicates"] = {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": float((duplicate_count / num_rows) * 100),
        }
        logger.info(
            f"Found {duplicate_count} duplicate rows ({report['duplicates']['duplicate_percentage']:.2f}%)"
        )

        # 4. Column Datatypes
        datatypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        report["column_datatypes"] = datatypes

        # 5. Basic Numeric Statistics (Profiling)
        numeric_df = df.select_dtypes(include=["number"])
        stats_summary = {}
        for col in numeric_df.columns:
            desc = numeric_df[col].describe()
            stats_summary[col] = {
                "mean": float(desc["mean"]) if not pd.isna(desc["mean"]) else 0.0,
                "std": float(desc["std"]) if not pd.isna(desc["std"]) else 0.0,
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }
        report["basic_statistics"] = stats_summary

        # Save report outputs
        self._save_reports(report)

        logger.info("Validation checks complete. Reports saved.")
        return report

    def _save_reports(self, report: Dict[str, Any]) -> None:
        """Saves validation report in JSON and TXT format.

        Args:
            report: Validation dictionary.
        """
        json_path = self.output_dir / "validation_report.json"
        txt_path = self.output_dir / "validation_report.txt"

        # Save JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Validation report (JSON) saved to: {json_path}")
        except Exception as e:
            logger.error(f"Failed to write validation JSON report: {e}")

        # Save TXT
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("            DATASET VALIDATION REPORT\n")
                f.write("=" * 60 + "\n\n")

                f.write("--- DATASET SUMMARY ---\n")
                f.write(f"Number of Rows:    {report['shape']['rows']}\n")
                f.write(f"Number of Columns: {report['shape']['columns']}\n")
                f.write(
                    f"Duplicate Rows:    {report['duplicates']['duplicate_count']} "
                    f"({report['duplicates']['duplicate_percentage']:.2f}%)\n"
                )
                f.write(
                    f"Total Missing:     {report['missing_values']['total_missing_cells']}\n\n"
                )

                f.write("--- MISSING VALUES PER COLUMN ---\n")
                if not report["missing_values"]["columns_with_missing"]:
                    f.write("No missing values found in the dataset.\n")
                else:
                    for col, data in report["missing_values"][
                        "columns_with_missing"
                    ].items():
                        f.write(
                            f"- {col}: {data['missing_count']} missing "
                            f"({data['missing_percentage']:.4f}%)\n"
                        )
                f.write("\n")

                f.write("--- DATATYPE SCHEMAS ---\n")
                for col, dtype in report["column_datatypes"].items():
                    f.write(f"- {col}: {dtype}\n")
                f.write("\n")

                f.write("--- STATISTICAL PROFILE FOR NUMERICAL COLUMNS ---\n")
                for col, stats in report["basic_statistics"].items():
                    f.write(f"\n[{col}]\n")
                    f.write(
                        f"  Mean: {stats['mean']:.4f} | Std Dev: {stats['std']:.4f}\n"
                    )
                    f.write(
                        f"  Min:  {stats['min']:.4f} | 25%: {stats['25%']:.4f} | 50%: {stats['50%']:.4f}\n"
                    )
                    f.write(f"  75%:  {stats['75%']:.4f} | Max: {stats['max']:.4f}\n")

            logger.info(f"Validation report (TXT) saved to: {txt_path}")
        except Exception as e:
            logger.error(f"Failed to write validation text report: {e}")
        return
