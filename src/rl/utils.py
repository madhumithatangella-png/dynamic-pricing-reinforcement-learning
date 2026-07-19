"""Visualization and plotting utilities for Dynamic Pricing RL.

Generates beautiful, publication-ready graphs and heatmaps representing RL performance.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, PRICE_ACTIONS

# Set modern, premium aesthetic theme
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


def plot_learning_curve(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots the training reward learning curve with rolling average.

    Args:
        df_metrics: DataFrame containing 'episode' and 'reward' columns.
        output_dir: Path where figures are saved.
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

    # Rolling window of size 50
    df_metrics["reward_rolling"] = df_metrics["reward"].rolling(window=50, min_periods=1).mean()
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="reward_rolling",
        linewidth=2.5,
        label="Rolling Average (50 Ep)",
        color="darkblue",
    )

    plt.title("Q-Learning Agent Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=200)
    plt.close()


def plot_episode_rewards(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots raw reward progress over training.

    Args:
        df_metrics: DataFrame containing 'episode' and 'reward' columns.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df_metrics["episode"], df_metrics["reward"], color="teal", alpha=0.7)
    plt.title("Episode Rewards Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "episode_rewards.png", dpi=200)
    plt.close()


def plot_revenue_curve(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
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
        color="crimson",
        label="Raw Revenue",
    )

    df_metrics["revenue_rolling"] = df_metrics["revenue"].rolling(window=50, min_periods=1).mean()
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="revenue_rolling",
        linewidth=2.5,
        color="darkred",
        label="Rolling Average (50 Ep)",
    )

    plt.title("Training Revenue Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue (INR)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_curve.png", dpi=200)
    plt.close()


def plot_epsilon_decay(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots epsilon decay rate over episodes.

    Args:
        df_metrics: DataFrame containing 'episode' and 'epsilon' columns.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df_metrics["episode"], df_metrics["epsilon"], color="purple", linewidth=2)
    plt.title("Epsilon Decay Curve")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon (Exploration Rate)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_decay.png", dpi=200)
    plt.close()


def plot_average_q_value(df_metrics: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Plots the average Q-value visited per episode.

    Args:
        df_metrics: DataFrame containing 'episode' and 'average_q' columns.
        output_dir: Path where figures are saved.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df_metrics["episode"], df_metrics["average_q"], color="darkorange", linewidth=2)
    plt.title("Average Visited Q-Value over Training")
    plt.xlabel("Episode")
    plt.ylabel("Mean Q-Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "average_q_value.png", dpi=200)
    plt.close()


def plot_comparisons(
    df_eval: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> None:
    """Generates comparison bar charts for Revenue and Occupancy.

    Args:
        df_eval: Evaluation summary DataFrame containing agent performance metrics.
        output_dir: Path where figures are saved.
    """
    # 1. Revenue Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_eval,
        x="Agent",
        y="Total Revenue",
        hue="Agent",
        palette="viridis",
        legend=False,
    )
    plt.title("Revenue Comparison Across Agents")
    plt.ylabel("Total Revenue (INR)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_comparison.png", dpi=200)
    plt.close()

    # 2. Occupancy Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_eval,
        x="Agent",
        y="Occupancy Rate",
        hue="Agent",
        palette="magma",
        legend=False,
    )
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.title("Average Occupancy Rate Comparison")
    plt.ylabel("Occupancy Rate")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "occupancy_comparison.png", dpi=200)
    plt.close()


def plot_action_distribution(
    actions_list: List[int], price_actions: list = PRICE_ACTIONS, output_dir: Path = FIGURES_DIR
) -> None:
    """Plots the frequency of price indices chosen by the Q-learning agent.

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
    plt.title("Q-Learning Pricing Action Choice Distribution")
    plt.xlabel("Price Option (INR)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution.png", dpi=200)
    plt.close()


def plot_price_distribution(
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
    plt.savefig(output_dir / "price_distribution.png", dpi=200)
    plt.close()


def plot_q_table_heatmap(
    q_table: np.ndarray, price_actions: list = PRICE_ACTIONS, output_dir: Path = FIGURES_DIR
) -> None:
    """Generates a 2D heatmap slice of Q-values (mean across seasons/months/hotels).

    Visualizes Days Until Departure vs. Remaining Inventory.

    Args:
        q_table: Multi-dimensional NumPy Q-table.
        price_actions: Pricing action values.
        output_dir: Path where figures are saved.
    """
    # Average Q-table across seasons (axis 2), months (axis 3), hotel_types (axis 4)
    # Resulting shape: (inventory+1, days+1, actions)
    q_reduced = np.mean(q_table, axis=(2, 3, 4))

    # Compute optimal action index for each state coordinate: (inventory, days)
    # Shape: (inventory+1, days+1)
    best_actions = np.argmax(q_reduced, axis=-1)

    # Convert index to actual prices
    best_prices = np.zeros(best_actions.shape)
    for i in range(len(price_actions)):
        best_prices[best_actions == i] = price_actions[i]

    # Create Heatmap
    plt.figure(figsize=(10, 8))
    # We display remaining inventory on y-axis, days left on x-axis
    ax = sns.heatmap(
        best_prices,
        cmap="YlOrRd",
        cbar_kws={"label": "Optimal Offered Price (INR)"},
        xticklabels=10,
        yticklabels=5,
    )
    plt.title("Optimal Price Map Heatmap (Remaining Inventory vs. Days Left)")
    plt.xlabel("Days Until Departure (Booking Window)")
    plt.ylabel("Remaining Room Inventory")
    plt.gca().invert_yaxis()  # Put inventory 0 at the bottom
    plt.tight_layout()
    plt.savefig(output_dir / "q_table_heatmap.png", dpi=200)
    plt.close()
