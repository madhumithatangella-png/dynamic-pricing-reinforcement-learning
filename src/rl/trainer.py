"""Trainer module for Dynamic Pricing RL.

Coordinates data loading, simulation fit, environment setup, Q-learning training,
metrics logging, serialization, and figure generation.
"""

from pathlib import Path
from typing import Tuple, Any
import pandas as pd
from src.logger import get_logger
from src.config import (
    FEATURES_DATA_PATH,
    EPISODES,
    MODELS_DIR,
    REPORTS_DIR,
    PRICE_ACTIONS,
    MAX_INVENTORY,
    MAX_DAYS,
    ALPHA,
    GAMMA,
    EPSILON,
    EPSILON_DECAY,
    MIN_EPSILON,
    RL_RANDOM_SEED,
)
from src.rl.demand_simulator import DemandSimulator
from src.rl.environment import HotelPricingEnvironment
from src.rl.q_learning import QLearningAgent
from src.rl.utils import (
    plot_learning_curve,
    plot_episode_rewards,
    plot_revenue_curve,
    plot_epsilon_decay,
    plot_average_q_value,
    plot_q_table_heatmap,
)

logger = get_logger("trainer")


def run_training_flow(
    data_path: Path = FEATURES_DATA_PATH,
    episodes: int = EPISODES,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    random_seed: int = RL_RANDOM_SEED,
) -> Tuple[QLearningAgent, HotelPricingEnvironment, DemandSimulator]:
    """Orchestrates the entire training pipeline for the Q-Learning agent.

    Args:
        data_path: Path to the processed features.
        episodes: Total episodes to train.
        models_dir: Folder to save trained models.
        reports_dir: Folder to save report CSV files.
        random_seed: Random seed.

    Returns:
        Tuple: (Trained QLearningAgent, HotelPricingEnvironment, DemandSimulator)
    """
    logger.info("=" * 60)
    logger.info("   STARTING Q-LEARNING AGENT TRAINING FLOW")
    logger.info("=" * 60)

    # 1. Fit Demand Simulator
    logger.info("Step 1: Fitting Demand Simulator on historical features...")
    demand_sim = DemandSimulator(data_path=data_path, random_seed=random_seed)
    demand_sim.fit()

    # 2. Create Environment
    logger.info("Step 2: Initializing Hotel Pricing Environment...")
    env = HotelPricingEnvironment(
        data_path=str(data_path),
        demand_simulator=demand_sim,
        max_inventory=MAX_INVENTORY,
        max_days=MAX_DAYS,
        price_actions=PRICE_ACTIONS,
        random_seed=random_seed,
    )

    # 3. Create Agent
    logger.info("Step 3: Initializing Q-Learning Agent...")
    agent = QLearningAgent(
        price_actions=PRICE_ACTIONS,
        max_inventory=MAX_INVENTORY,
        max_days=MAX_DAYS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
        random_seed=random_seed,
    )

    # 4. Train Agent
    logger.info("Step 4: Training agent...")
    df_metrics = agent.train(env, episodes)

    # 5. Save Outputs
    logger.info("Step 5: Saving model and training metrics...")
    model_path = models_dir / "q_table.npy"
    agent.save(model_path)

    metrics_path = reports_dir / "training_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # 6. Generate Graphs
    logger.info("Step 6: Plotting training metrics...")
    plot_learning_curve(df_metrics)
    plot_episode_rewards(df_metrics)
    plot_revenue_curve(df_metrics)
    plot_epsilon_decay(df_metrics)
    plot_average_q_value(df_metrics)
    plot_q_table_heatmap(agent.q_table)
    logger.info("All training visualizations successfully saved.")

    logger.info("=" * 60)
    logger.info("   Q-LEARNING AGENT TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return agent, env, demand_sim
