"""Integration tests verifying DQN Trainer and training flow execution."""

import pytest
from pathlib import Path
from src.config import FEATURES_DATA_PATH, PRICE_ACTIONS
from src.rl.demand_simulator import DemandSimulator
from src.rl.environment import HotelPricingEnvironment
from src.dqn.dqn_agent import DQNAgent
from src.dqn.trainer import DQNTrainer, run_dqn_training_flow


def test_dqn_trainer_execution(tmp_path: Path) -> None:
    """Verifies that DQNTrainer coordinates agent-env loops and stores metrics."""
    if not FEATURES_DATA_PATH.exists():
        pytest.skip("Processed features dataset CSV is missing, skipping integration test.")

    # Create short simulation
    simulator = DemandSimulator(data_path=FEATURES_DATA_PATH, random_seed=42)
    simulator.fit()

    env = HotelPricingEnvironment(
        data_path=str(FEATURES_DATA_PATH),
        demand_simulator=simulator,
        max_inventory=5,
        max_days=5,
        price_actions=PRICE_ACTIONS,
        random_seed=42,
    )

    agent = DQNAgent(
        state_dim=5,
        action_dim=len(PRICE_ACTIONS),
        price_actions=PRICE_ACTIONS,
        max_inventory=5,
        max_days=5,
        batch_size=2,
        memory_size=10,
        learning_rate=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9,
        min_epsilon=0.1,
        hidden_units=[8, 8],
        device="cpu",
        random_seed=42,
    )

    trainer = DQNTrainer(env, agent)
    # Train for 2 episodes
    df_metrics = trainer.train(episodes=2, target_update_freq=1, models_dir=tmp_path)

    assert len(df_metrics) == 2
    assert "episode" in df_metrics.columns
    assert "reward" in df_metrics.columns
    assert "loss" in df_metrics.columns
    assert "revenue" in df_metrics.columns
    assert "average_q" in df_metrics.columns
