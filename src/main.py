"""Main orchestrator script for the Dynamic Pricing RL data pipeline.

Supports Phase 1 (Data Processing & EDA) and Phase 2 (Reinforcement Learning).
"""

import argparse
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
from src.rl.trainer import run_training_flow
from src.rl.evaluator import run_evaluation_flow

logger = get_logger("main_pipeline")


def run_phase_1() -> int:
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


def run_phase_2() -> int:
    """Executes the complete Phase 2 Reinforcement Learning framework.

    Returns:
        int: Return code. 0 for success, non-zero for failure.
    """
    logger.info("=" * 60)
    logger.info("   STARTING DYNAMIC PRICING RL FRAMEWORK (PHASE 2)")
    logger.info("=" * 60)

    try:
        # Step 1: Train Q-Learning Agent
        agent, env, simulator = run_training_flow()

        # Step 2: Evaluate and compare Q-learning with Baselines
        df_summary = run_evaluation_flow(q_agent=agent, env=env)

        logger.info("Summary of Evaluation Results:")
        for idx, row in df_summary.iterrows():
            logger.info(
                f"Agent: {row['Agent']:18s} | Revenue: {row['Total Revenue']:10.1f} | "
                f"Occupancy: {row['Occupancy Rate']:6.2%} | Rooms Sold: {row['Rooms Sold']:4d}"
            )

        logger.info("=" * 60)
        logger.info("   PHASE 2 REINFORCEMENT LEARNING RUN SUCCESSFULLY COMPLETED!")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"Phase 2 execution failed due to an error: {e}")
        return 1


def main() -> None:
    """Parses command line arguments and routes execution to selected phase."""
    parser = argparse.ArgumentParser(
        description="Dynamic Pricing using Reinforcement Learning Orchestrator."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Specify phase to run. Phase 1 (Data/EDA) or Phase 2 (RL/Q-learning). Default is 1.",
    )
    args = parser.parse_args()

    if args.phase == 2:
        sys.exit(run_phase_2())
    else:
        sys.exit(run_phase_1())


if __name__ == "__main__":
    main()
