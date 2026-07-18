"""Main orchestrator script for the Dynamic Pricing RL data pipeline.

Loads, validates, cleans, engineers features, and generates plots.
"""

import sys
from src.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    FEATURES_DATA_PATH,
    FIGURES_DIR,
)
from src.data.data_loader import load_raw_data
from src.data.data_validator import DataValidator
from src.data.data_preprocessing import preprocess_data
from src.data.feature_engineering import engineer_features
from src.visualization.eda import generate_all_plots
from src.logger import get_logger

logger = get_logger("main_pipeline")


def run_pipeline() -> int:
    """Executes the complete Phase 1 pipeline end-to-end.

    Returns:
        int: Return code. 0 for success, non-zero for failure.
    """
    logger.info("=" * 60)
    logger.info("   STARTING DYNAMIC PRICING PIPELINE EXECUTION (PHASE 1)")
    logger.info("=" * 60)

    try:
        # Step 1: Load Data
        logger.info("[Pipeline Step 1] Loading raw data...")
        df_raw = load_raw_data(RAW_DATA_PATH)

        # Step 2: Validate Data
        logger.info("[Pipeline Step 2] Running data validation checks...")
        validator = DataValidator()
        validator.run_validation(df_raw)

        # Step 3: Preprocess Data (Fill nulls, parse dates, filter outliers)
        logger.info("[Pipeline Step 3] Preprocessing and cleaning dataset...")
        df_clean = preprocess_data(df_raw, CLEANED_DATA_PATH)

        # Step 4: Feature Engineering
        logger.info("[Pipeline Step 4] Engineering features...")
        df_feat = engineer_features(df_clean, FEATURES_DATA_PATH)

        # Step 5: Generate EDA Plots
        logger.info("[Pipeline Step 5] Generating EDA visualizations...")
        generate_all_plots(df_feat, FIGURES_DIR)

        logger.info("=" * 60)
        logger.info("   PIPELINE RUN SUCCESSFULLY COMPLETED!")
        logger.info(f"   Outputs produced:")
        logger.info(f"   - Cleaned Dataset:  {CLEANED_DATA_PATH}")
        logger.info(f"   - Features Dataset: {FEATURES_DATA_PATH}")
        logger.info(f"   - EDA Figures:      {FIGURES_DIR}/")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"Pipeline execution failed due to an error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
