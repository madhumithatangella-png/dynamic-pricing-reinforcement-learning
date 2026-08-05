"""Evaluator module for Dynamic Pricing RL.

Compares Q-learning agent performance against fixed, discount, and random
pricing baselines. Generates tables and graphs.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import REPORTS_DIR, PRICE_ACTIONS, EVALUATION_EPISODES
from src.rl.environment import HotelPricingEnvironment
from src.rl.q_learning import QLearningAgent
from src.rl.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent,
)
from src.rl.utils import (
    plot_comparisons,
    plot_action_distribution,
    plot_price_distribution,
)

logger = get_logger("evaluator")


def evaluate_agent(
    agent: Any, env: HotelPricingEnvironment, episodes: int
) -> Tuple[List[Dict[str, Any]], List[int], List[float]]:
    """Evaluates a single agent over multiple episodes.

    Args:
        agent: Agent instance (needs choose_action).
        env: Environment instance.
        episodes: Number of episodes to evaluate.

    Returns:
        Tuple: (episode_metrics_list, action_indices_chosen, actual_prices_chosen)
    """
    episode_metrics = []
    actions_chosen = []
    prices_chosen = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Q-learning check: disable exploration during evaluation
            if isinstance(agent, QLearningAgent):
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

    return episode_metrics, actions_chosen, prices_chosen


def run_evaluation_flow(
    q_agent: QLearningAgent,
    env: HotelPricingEnvironment,
    episodes: int = EVALUATION_EPISODES,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Evaluates and compares Q-learning against Fixed, Discount, and Random pricing.

    Args:
        q_agent: Trained QLearningAgent.
        env: HotelPricingEnvironment.
        episodes: Number of evaluation episodes.
        reports_dir: Folder to save comparison CSV files.

    Returns:
        pd.DataFrame: Summary comparison table.
    """
    logger.info("=" * 60)
    logger.info("   STARTING PERFORMANCE EVALUATION FLOW")
    logger.info("=" * 60)

    # 1. Instantiate baseline agents
    fixed_agent = FixedPricingAgent(price_actions=env.price_actions)
    discount_agent = DiscountPricingAgent(
        price_actions=env.price_actions, max_days=env.max_days
    )
    random_agent = RandomPricingAgent(price_actions=env.price_actions)

    agents = {
        "Q-Learning": q_agent,
        "Fixed Pricing": fixed_agent,
        "Discount Pricing": discount_agent,
        "Random Pricing": random_agent,
    }

    results_summary = []
    agent_prices = {}
    q_learning_actions = []

    for name, agent in agents.items():
        logger.info(f"Evaluating {name} agent for {episodes} episodes...")
        ep_metrics, actions, prices = evaluate_agent(agent, env, episodes)

        agent_prices[name] = prices
        if name == "Q-Learning":
            q_learning_actions = actions

        # Aggregate metrics
        df_ep = pd.DataFrame(ep_metrics)
        avg_revenue = float(df_ep["revenue"].mean())
        total_revenue = float(df_ep["revenue"].sum())
        avg_reward = float(df_ep["reward"].mean())
        avg_occupancy = float(df_ep["occupancy_rate"].mean())
        avg_utilization = avg_occupancy  # utilization matches occupancy rate
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
            }
        )

    # Convert to DataFrame
    df_summary = pd.DataFrame(results_summary)

    # 2. Export reports
    # Export full evaluation summary
    eval_csv_path = reports_dir / "evaluation_results.csv"
    df_summary.to_csv(eval_csv_path, index=False)
    logger.info(f"Evaluation results exported to {eval_csv_path}")

    # Export baseline-only summary
    baseline_csv_path = reports_dir / "baseline_results.csv"
    df_baseline = df_summary[df_summary["Agent"] != "Q-Learning"]
    df_baseline.to_csv(baseline_csv_path, index=False)
    logger.info(f"Baseline results exported to {baseline_csv_path}")

    # 3. Generate visualization plots
    logger.info("Generating evaluation comparison visualizations...")
    plot_comparisons(df_summary)
    plot_action_distribution(q_learning_actions, env.price_actions)
    plot_price_distribution(agent_prices)
    logger.info("All evaluation visualizations successfully saved.")

    logger.info("=" * 60)
    logger.info("   EVALUATION WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return df_summary
