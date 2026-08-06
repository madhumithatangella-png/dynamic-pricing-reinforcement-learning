"""Trainer module for Phase 3 Deep Q Network (DQN) Dynamic Pricing.

Coordinates the data loading, simulation fit, environment setup, DQN agent initialization,
optimization training loops, target weight syncing, metric logging, checkpoint serialization,
and learning plot generations.
"""

from pathlib import Path
from typing import Tuple, Any, List
import time
import pandas as pd
import numpy as np
import torch
from src.logger import get_logger
from src.config import (
    FEATURES_DATA_PATH,
    DQN_EPISODES,
    BATCH_SIZE,
    MEMORY_SIZE,
    TARGET_UPDATE,
    LEARNING_RATE,
    HIDDEN_UNITS,
    DEVICE,
    SAVE_MODEL_EVERY,
    DQN_RANDOM_SEED,
    PRICE_ACTIONS,
    MAX_INVENTORY,
    MAX_DAYS,
    GAMMA,
    EPSILON,
    EPSILON_DECAY,
    MIN_EPSILON,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.rl.demand_simulator import DemandSimulator
from src.rl.environment import HotelPricingEnvironment
from src.dqn.dqn_agent import DQNAgent
from src.dqn.utils import (
    plot_dqn_learning_curve,
    plot_loss_curve,
    plot_dqn_episode_rewards,
    plot_dqn_revenue_curve,
)

logger = get_logger("dqn_trainer")


class DQNTrainer:
    """Trainer orchestrator for PyTorch Deep Q-Network dynamic pricing agent."""

    def __init__(self, env: HotelPricingEnvironment, agent: DQNAgent) -> None:
        """Initializes the DQNTrainer using dependency injection.

        Args:
            env: The HotelPricingEnvironment instance.
            agent: The DQNAgent instance.
        """
        self.env = env
        self.agent = agent

    def train(self, episodes: int, target_update_freq: int, models_dir: Path) -> pd.DataFrame:
        """Runs the optimization training loop over multiple episodes.

        Args:
            episodes: Total number of training episodes.
            target_update_freq: Episode interval for syncing target network weights.
            models_dir: Directory where model checkpoints will be saved.

        Returns:
            pd.DataFrame: DataFrame containing training metrics for all episodes.
        """
        metrics = []
        logger.info(f"Starting DQN training for {episodes} episodes...")

        for ep in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            episode_losses = []
            q_values_visited = []

            while not done:
                # Epsilon-greedy action index selection
                action_idx = self.agent.choose_action(state, use_epsilon=True)

                # Record expected Q-value for diagnostics
                with torch.no_grad():
                    state_tensor = self.agent._preprocess_state(state).unsqueeze(0)
                    q_values = self.agent.q_net(state_tensor)
                    q_values_visited.append(float(q_values[0, action_idx].item()))

                # Environment transition step
                next_state, reward, done, info = self.env.step(action_idx)

                # Store transition in replay buffer
                self.agent.remember(state, action_idx, reward, next_state, done)

                # Perform optimization step
                loss = self.agent.train_step()
                if loss > 0:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward
                steps += 1

            # Decay agent exploration probability
            self.agent.decay_epsilon()

            # Synchronize target network periodically
            if ep % target_update_freq == 0:
                self.agent.update_target_network()
                logger.info(f"Episode {ep:4d} | Target network weights synchronized.")

            # Periodic saving checkpoint
            if ep % SAVE_MODEL_EVERY == 0:
                self.agent.save(
                    models_dir / f"dqn_model_ep_{ep}.pth",
                    models_dir / f"optimizer_ep_{ep}.pth",
                )

            # Record training metrics
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_q = float(np.mean(q_values_visited)) if q_values_visited else 0.0

            metrics.append(
                {
                    "episode": ep,
                    "reward": total_reward,
                    "revenue": self.env.total_revenue,
                    "rooms_sold": self.env.rooms_sold,
                    "occupancy_rate": self.env.rooms_sold / self.env.max_inventory,
                    "epsilon": self.agent.epsilon,
                    "average_q": avg_q,
                    "loss": avg_loss,
                }
            )

            # Log metrics progress info
            if ep % 50 == 0 or ep == 1:
                logger.info(
                    f"Episode {ep:4d}/{episodes:4d} | Reward: {total_reward:8.2f} | "
                    f"Revenue: {self.env.total_revenue:8.0f} | "
                    f"Occupancy: {self.env.rooms_sold/self.env.max_inventory:5.2%} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | Avg Loss: {avg_loss:8.4f}"
                )

        logger.info("DQN Agent training complete.")
        return pd.DataFrame(metrics)


def run_dqn_training_flow(
    data_path: Path = FEATURES_DATA_PATH,
    episodes: int = DQN_EPISODES,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    random_seed: int = DQN_RANDOM_SEED,
) -> Tuple[DQNAgent, HotelPricingEnvironment, DemandSimulator, float]:
    """Orchestrates the entire training pipeline for the DQN agent.

    Args:
        data_path: Path to the processed dataset file.
        episodes: Total training episodes.
        models_dir: Folder path for saving state dictionaries.
        reports_dir: Folder path for saving evaluation reports.
        random_seed: Reproducibility random seed value.

    Returns:
        Tuple: (DQNAgent, HotelPricingEnvironment, DemandSimulator, float training_time)
    """
    logger.info("=" * 60)
    logger.info("   STARTING DEEP Q-NETWORK AGENT TRAINING FLOW")
    logger.info("=" * 60)

    # 1. Fit Simulator
    logger.info("Step 1: Fitting Demand Simulator on historical features...")
    demand_sim = DemandSimulator(data_path=data_path, random_seed=random_seed)
    demand_sim.fit()

    # 2. Setup Env
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
    logger.info("Step 3: Initializing DQNAgent with PyTorch network architectures...")
    agent = DQNAgent(
        state_dim=5,
        action_dim=len(PRICE_ACTIONS),
        price_actions=PRICE_ACTIONS,
        max_inventory=MAX_INVENTORY,
        max_days=MAX_DAYS,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
        hidden_units=HIDDEN_UNITS,
        device=DEVICE,
        random_seed=random_seed,
    )

    # 4. Train Agent (measure training duration)
    logger.info("Step 4: Starting optimization training phases...")
    trainer = DQNTrainer(env, agent)
    start_time = time.time()
    df_metrics = trainer.train(episodes, TARGET_UPDATE, models_dir)
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training DQN agent took {training_time:.2f} seconds.")

    # 5. Save Outputs
    logger.info("Step 5: Saving final state dictionaries and metrics CSV reports...")
    agent.save(models_dir / "dqn_model.pth", models_dir / "optimizer.pth")

    metrics_csv_path = reports_dir / "dqn_training_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Training metrics saved to: {metrics_csv_path}")

    # 6. Plot Training Visualization Curves
    logger.info("Step 6: Generating DQN training performance plots...")
    plot_dqn_learning_curve(df_metrics)
    plot_loss_curve(df_metrics)
    plot_dqn_episode_rewards(df_metrics)
    plot_dqn_revenue_curve(df_metrics)
    logger.info("DQN training visualizations generated successfully.")

    logger.info("=" * 60)
    logger.info("   DEEP Q-NETWORK TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return agent, env, demand_sim, training_time
