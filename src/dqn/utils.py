"""Visualization and plotting utilities for Phase 3 Deep Q Network (DQN).

Generates modern, premium-quality plots showing DQN learning progress,
loss decay, evaluation comparisons, action and price distributions, and Q-value ranges.
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, PRICE_ACTIONS

# Set high-quality styling matching Phase 2
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
    }
)


def plot_dqn_learning_curve(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots DQN training reward learning curve with rolling average.

    Args:
        df_metrics: DataFrame containing 'episode' and 'reward'.
        output_dir: Folder to save generated plot.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="reward",
        alpha=0.3,
        label="Raw Episode Reward",
        color="royalblue",
    )

    df_metrics["reward_rolling"] = df_metrics["reward"].rolling(window=50, min_periods=1).mean()
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="reward_rolling",
        linewidth=2.5,
        label="Rolling Average (50 Ep)",
        color="darkblue",
    )

    plt.title("DQN Agent Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_learning_curve.png", dpi=200)
    plt.close()


def plot_loss_curve(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots MSE training loss curve over episodes.

    Args:
        df_metrics: DataFrame containing 'episode' and 'loss'.
        output_dir: Folder to save generated plot.
    """
    plt.figure(figsize=(10, 4))
    # Filter out episodes that have no training step updates (e.g. before buffer fills)
    df_loss = df_metrics[df_metrics["loss"] > 0]
    if df_loss.empty:
        df_loss = df_metrics

    sns.lineplot(
        data=df_loss,
        x="episode",
        y="loss",
        color="crimson",
        alpha=0.4,
        label="Raw Loss",
    )

    df_loss["loss_rolling"] = df_loss["loss"].rolling(window=50, min_periods=1).mean()
    sns.lineplot(
        data=df_loss,
        x="episode",
        y="loss_rolling",
        linewidth=2.5,
        color="darkred",
        label="Rolling Average (50 Ep)",
    )

    plt.title("DQN Agent Training Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("MSE Loss")
    plt.yscale("log")  # Log scale since loss can vary by orders of magnitude
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


def plot_dqn_episode_rewards(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots raw reward progress over training.

    Args:
        df_metrics: DataFrame containing 'episode' and 'reward' columns.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df_metrics["episode"], df_metrics["reward"], color="teal", alpha=0.7)
    plt.title("DQN Episode Rewards Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_episode_rewards.png", dpi=200)
    plt.close()


def plot_dqn_revenue_curve(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots revenue per episode during training.

    Args:
        df_metrics: DataFrame containing 'episode' and 'revenue' columns.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="revenue",
        alpha=0.3,
        color="coral",
        label="Raw Revenue",
    )

    df_metrics["revenue_rolling"] = df_metrics["revenue"].rolling(window=50, min_periods=1).mean()
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="revenue_rolling",
        linewidth=2.5,
        color="darkorange",
        label="Rolling Average (50 Ep)",
    )

    plt.title("DQN Training Revenue Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue (INR)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_revenue_curve.png", dpi=200)
    plt.close()


def plot_dqn_action_distribution(
    actions_list: List[int], price_actions: list = PRICE_ACTIONS, output_dir: Path = FIGURES_DIR
) -> None:
    """Plots the frequency of price indices chosen by the DQN agent.

    Args:
        actions_list: List of selected action indices during evaluation.
        price_actions: Mapped price values.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(8, 5))
    unique, counts = np.unique(actions_list, return_counts=True)
    count_dict = dict(zip(unique, counts))

    frequencies = [count_dict.get(i, 0) for i in range(len(price_actions))]
    labels = [str(int(p)) for p in price_actions]

    sns.barplot(x=labels, y=frequencies, hue=labels, palette="coolwarm", legend=False)
    plt.title("DQN Pricing Action Choice Distribution")
    plt.xlabel("Price Option (INR)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_action_distribution.png", dpi=200)
    plt.close()


def plot_dqn_price_distribution(
    agent_prices: Dict[str, List[float]], output_dir: Path = FIGURES_DIR
) -> None:
    """Plots the boxplot price distributions of chosen prices by agent.

    Args:
        agent_prices: Dict mapping Agent Name -> list of prices chosen during evaluation.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(9, 5))

    # Convert dictionary to long-form DataFrame for Seaborn
    records = []
    for agent, prices in agent_prices.items():
        for p in prices:
            records.append({"Agent": agent, "Price": p})
    df = pd.DataFrame(records)

    sns.boxplot(data=df, x="Agent", y="Price", hue="Agent", palette="Set2")
    plt.title("Price Range Distributions Across Agents")
    plt.ylabel("Price Offered (INR)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_price_distribution.png", dpi=200)
    plt.close()


def plot_dqn_comparisons(
    df_eval: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> None:
    """Generates comparison bar charts for Revenue and Occupancy.

    Args:
        df_eval: Evaluation summary DataFrame containing agent performance metrics.
        output_dir: Path where figures are saved.
    """
    # 1. Revenue Comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_eval,
        x="Agent",
        y="Total Revenue",
        hue="Agent",
        palette="viridis",
        legend=False,
    )
    plt.title("Revenue Comparison Across Agents (Phase 3)")
    plt.ylabel("Total Revenue (INR)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_revenue_comparison.png", dpi=200)
    plt.close()

    # 2. Occupancy Comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_eval,
        x="Agent",
        y="Occupancy Rate",
        hue="Agent",
        palette="magma",
        legend=False,
    )
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.title("Average Occupancy Rate Comparison (Phase 3)")
    plt.ylabel("Occupancy Rate")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "dqn_occupancy_comparison.png", dpi=200)
    plt.close()


def plot_q_value_histogram(
    q_values: List[float], output_dir: Path = FIGURES_DIR
) -> None:
    """Generates a histogram of predicted Q-values for the DQN policy.

    Useful for diagnostic checks of Q-value scaling and stability.

    Args:
        q_values: List of policy network predicted Q-values.
        output_dir: Path to save generated plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(q_values, bins=30, kde=True, color="darkviolet")
    plt.title("DQN Predicted Q-Value Histogram (Evaluation Steps)")
    plt.xlabel("Predicted Q-Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "q_value_histogram.png", dpi=200)
    plt.close()
