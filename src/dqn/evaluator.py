"""Evaluator module for Phase 3 Deep Q Network (DQN).

Compares DQN against Tabular Q-learning, Fixed pricing, Discount pricing,
and Random pricing agents. Generates comparative tables and saving reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import pandas as pd
import numpy as np
import torch
from src.logger import get_logger
from src.config import REPORTS_DIR, PRICE_ACTIONS, EVALUATION_EPISODES, MODELS_DIR
from src.rl.environment import HotelPricingEnvironment
from src.rl.q_learning import QLearningAgent
from src.rl.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent,
)
from src.dqn.dqn_agent import DQNAgent
from src.dqn.utils import (
    plot_dqn_comparisons,
    plot_dqn_action_distribution,
    plot_dqn_price_distribution,
    plot_q_value_histogram,
)

logger = get_logger("dqn_evaluator")


def evaluate_agent(
    agent: Any, env: HotelPricingEnvironment, episodes: int
) -> Tuple[List[Dict[str, Any]], List[int], List[float], float]:
    """Evaluates a single agent over multiple episodes and measures inference time.

    Args:
        agent: Agent instance.
        env: Environment instance.
        episodes: Number of episodes to evaluate.

    Returns:
        Tuple: (episode_metrics_list, action_indices_chosen, actual_prices_chosen, avg_inference_time_ms)
    """
    episode_metrics = []
    actions_chosen = []
    prices_chosen = []

    start_time = time.time()
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Check decision making based on agent class type
            if isinstance(agent, DQNAgent):
                action_idx = agent.predict(state)
            elif isinstance(agent, QLearningAgent):
                action_idx = agent.choose_action(state, use_epsilon=False)
            else:
                action_idx = agent.choose_action(state)

            actions_chosen.append(action_idx)
            price = env.price_actions[action_idx]
            prices_chosen.append(price)

            next_state, reward, done, info = env.step(action_idx)
            state = next_state
            total_reward += reward

        # Calculate metrics for the episode
        occupancy = env.rooms_sold / env.max_inventory
        acceptance_rate = env.rooms_sold / env.total_arrivals if env.total_arrivals > 0 else 0.0

        episode_metrics.append(
            {
                "episode": ep,
                "revenue": env.total_revenue,
                "reward": total_reward,
                "rooms_sold": env.rooms_sold,
                "arrivals": env.total_arrivals,
                "occupancy_rate": occupancy,
                "acceptance_rate": acceptance_rate,
                "remaining_inventory": env.remaining_inventory,
            }
        )

    end_time = time.time()
    total_time = end_time - start_time
    # Avg inference time per episode in milliseconds
    avg_inference_time_ms = (total_time / episodes) * 1000.0

    return episode_metrics, actions_chosen, prices_chosen, avg_inference_time_ms


def run_dqn_evaluation_flow(
    dqn_agent: DQNAgent,
    env: HotelPricingEnvironment,
    dqn_training_time: float,
    q_learning_training_time: float = 0.0,
    episodes: int = EVALUATION_EPISODES,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Evaluates and compares DQN against Tabular Q-learning and baselines.

    Args:
        dqn_agent: Trained DQNAgent.
        env: HotelPricingEnvironment.
        dqn_training_time: Measured training duration of DQN.
        q_learning_training_time: Measured or estimated training duration of Q-learning.
        episodes: Number of evaluation episodes.
        reports_dir: Folder to save comparison report CSV files.

    Returns:
        pd.DataFrame: Summary comparison table.
    """
    logger.info("=" * 60)
    logger.info("   STARTING PERFORMANCE EVALUATION FLOW (PHASE 3)")
    logger.info("=" * 60)

    # 1. Instantiate other agents
    # Load QLearningAgent if exists, else initialize one
    q_agent = QLearningAgent(
        price_actions=env.price_actions,
        max_inventory=env.max_inventory,
        max_days=env.max_days,
    )
    q_table_path = MODELS_DIR / "q_table.npy"
    if q_table_path.exists():
        logger.info(f"Loading pre-trained Q-table from {q_table_path}...")
        q_agent.load(q_table_path)
    else:
        logger.warning(
            f"Pre-trained Q-table not found at {q_table_path}. "
            "Tabular agent will act randomly/untrained."
        )

    fixed_agent = FixedPricingAgent(price_actions=env.price_actions)
    discount_agent = DiscountPricingAgent(
        price_actions=env.price_actions, max_days=env.max_days
    )
    random_agent = RandomPricingAgent(price_actions=env.price_actions)

    agents = {
        "DQN": (dqn_agent, dqn_training_time),
        "Q-Learning": (q_agent, q_learning_training_time),
        "Fixed Pricing": (fixed_agent, 0.0),
        "Discount Pricing": (discount_agent, 0.0),
        "Random Pricing": (random_agent, 0.0),
    }

    results_summary = []
    agent_prices = {}
    dqn_actions = []
    dqn_q_values = []

    # Record DQN Q-values during evaluation for diagnostics histogram
    for ep in range(1, 11):  # Sample first 10 episodes to gather Q-values
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = dqn_agent._preprocess_state(state).unsqueeze(0)
                q_vals = dqn_agent.q_net(state_tensor)
                dqn_q_values.extend(q_vals[0].cpu().numpy().tolist())
            action = dqn_agent.predict(state)
            state, _, done, _ = env.step(action)

    for name, (agent, train_time) in agents.items():
        logger.info(f"Evaluating {name} agent for {episodes} episodes...")
        ep_metrics, actions, prices, inference_time = evaluate_agent(agent, env, episodes)

        agent_prices[name] = prices
        if name == "DQN":
            dqn_actions = actions

        # Aggregate metrics
        df_ep = pd.DataFrame(ep_metrics)
        avg_revenue = float(df_ep["revenue"].mean())
        total_revenue = float(df_ep["revenue"].sum())
        avg_reward = float(df_ep["reward"].mean())
        avg_occupancy = float(df_ep["occupancy_rate"].mean())
        avg_utilization = avg_occupancy
        avg_acceptance = float(df_ep["acceptance_rate"].mean())
        avg_price = float(np.mean(prices))
        total_sold = int(df_ep["rooms_sold"].sum())
        avg_sold = float(df_ep["rooms_sold"].mean())
        avg_remaining = float(df_ep["remaining_inventory"].mean())

        results_summary.append(
            {
                "Agent": name,
                "Total Revenue": total_revenue,
                "Revenue per Episode": avg_revenue,
                "Average Reward": avg_reward,
                "Booking Acceptance Rate": avg_acceptance,
                "Occupancy Rate": avg_occupancy,
                "Inventory Utilization": avg_utilization,
                "Average Room Price": avg_price,
                "Rooms Sold": total_sold,
                "Average Rooms Sold": avg_sold,
                "Remaining Inventory": avg_remaining,
                "Training Time": train_time,
                "Inference Time": inference_time,
            }
        )

    # Convert to DataFrame
    df_summary = pd.DataFrame(results_summary)

    # 2. Export reports
    eval_csv_path = reports_dir / "dqn_evaluation.csv"
    df_summary.to_csv(eval_csv_path, index=False)
    logger.info(f"Evaluation comparison report saved to {eval_csv_path}")

    # 3. Generate visualization plots
    logger.info("Generating comparative evaluation visualization plots...")
    plot_dqn_comparisons(df_summary)
    plot_dqn_action_distribution(dqn_actions, env.price_actions)
    plot_dqn_price_distribution(agent_prices)
    if dqn_q_values:
        plot_q_value_histogram(dqn_q_values)
    logger.info("All comparative figures saved to outputs/figures/")

    logger.info("=" * 60)
    logger.info("   EVALUATION FLOW (PHASE 3) COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return df_summary
