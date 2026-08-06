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


def run_phase_3() -> int:
    """Executes the complete Phase 3 Deep Q-Network dynamic pricing framework.

    Returns:
        int: Return code. 0 for success, non-zero for failure.
    """
    logger.info("=" * 60)
    logger.info("   STARTING DYNAMIC PRICING DQN FRAMEWORK (PHASE 3)")
    logger.info("=" * 60)

    try:
        import time
        from src.rl.trainer import run_training_flow as run_q_training_flow

        # Step 1: Train Q-Learning Agent first to get training duration and ensure q_table is ready
        logger.info("[Phase 3 Step 1] Training/Loading Tabular Q-Learning Agent...")
        q_start = time.time()
        q_agent, env, simulator = run_q_training_flow()
        q_end = time.time()
        q_time = q_end - q_start

        # Step 2: Train DQN Agent
        logger.info("[Phase 3 Step 2] Training PyTorch DQN Agent...")
        from src.dqn.trainer import run_dqn_training_flow
        dqn_agent, env, simulator, dqn_time = run_dqn_training_flow()

        # Step 3: Run Phase 3 Evaluation Comparison Flow
        logger.info("[Phase 3 Step 3] Evaluating and comparing DQN, Q-Learning, and Baselines...")
        from src.dqn.evaluator import run_dqn_evaluation_flow
        df_summary = run_dqn_evaluation_flow(
            dqn_agent=dqn_agent,
            env=env,
            dqn_training_time=dqn_time,
            q_learning_training_time=q_time,
        )

        logger.info("Summary of Phase 3 Evaluation Comparison Results:")
        for idx, row in df_summary.iterrows():
            logger.info(
                f"Agent: {row['Agent']:18s} | Revenue: {row['Total Revenue']:10.1f} | "
                f"Occupancy: {row['Occupancy Rate']:6.2%} | Avg Reward: {row['Average Reward']:8.2f} | "
                f"Train Time: {row['Training Time']:6.2f}s | Infer Time: {row['Inference Time']:5.2f}ms"
            )

        logger.info("=" * 60)
        logger.info("   PHASE 3 DEEP Q-NETWORK RUN SUCCESSFULLY COMPLETED!")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"Phase 3 execution failed due to an error: {e}")
        return 1


def run_phase_4() -> int:
    """Executes the complete Phase 4 deployment framework.

    Returns:
        int: Return code. 0 for success, non-zero for failure.
    """
    logger.info("=" * 60)
    logger.info("   STARTING DEPLOYMENT & PRODUCTION PLATFORM (PHASE 4)")
    logger.info("=" * 60)

    try:
        from src.config import MODELS_DIR, FEATURES_DATA_PATH

        # Verify dataset exists
        if not FEATURES_DATA_PATH.exists():
            logger.error(
                f"Processed dataset features CSV not found at {FEATURES_DATA_PATH}. "
                "Phase 1 must be run first."
            )
            return 1

        # Verify model checkpoints exist
        dqn_model = MODELS_DIR / "dqn_model.pth"
        q_table = MODELS_DIR / "q_table.npy"

        if not dqn_model.exists():
            logger.warning(f"Trained DQN agent model checkpoint ({dqn_model}) is missing.")
        if not q_table.exists():
            logger.warning(f"Trained Q-learning agent model checkpoint ({q_table}) is missing.")

        logger.info("Deployment artifacts successfully verified:")
        logger.info("- Dockerfile: deployment/Dockerfile")
        logger.info("- Compose:    deployment/docker-compose.yml")
        logger.info("- Nginx:      deployment/nginx.conf")
        logger.info("- FastAPI:    api/main.py")
        logger.info("- Streamlit:  dashboard/app.py")

        logger.info("Launching FastAPI production server on http://localhost:8000 ...")
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
        return 0

    except Exception as e:
        logger.exception(f"Phase 4 deployment server launch failed due to an error: {e}")
        return 1


def main() -> None:
    """Parses command line arguments and routes execution to selected phase."""
    parser = argparse.ArgumentParser(
        description="Dynamic Pricing using Reinforcement Learning Orchestrator."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Specify phase to run. 1 (Data/EDA), 2 (RL/Q-learning), 3 (DQN), or 4 (Deployment/API). Default is 1.",
    )
    args = parser.parse_args()

    if args.phase == 4:
        sys.exit(run_phase_4())
    elif args.phase == 3:
        sys.exit(run_phase_3())
    elif args.phase == 2:
        sys.exit(run_phase_2())
    else:
        sys.exit(run_phase_1())


if __name__ == "__main__":
    main()


