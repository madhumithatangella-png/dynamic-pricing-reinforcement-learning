"""Charts utility module for rendering dashboard visualizations in Streamlit."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Set modern, premium aesthetic theme
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    }
)


def plot_revenue_comparison(df_comparison: pd.DataFrame) -> plt.Figure:
    """Generates revenue comparison bar chart.

    Args:
        df_comparison: DataFrame with columns 'Agent' and 'Total Revenue'.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df_comparison,
        x="Agent",
        y="Total Revenue",
        hue="Agent",
        palette="viridis",
        ax=ax,
        legend=False,
    )
    ax.set_title("Total Revenue Comparison Across Agents (INR)")
    ax.set_ylabel("Total Revenue (INR)")
    ax.set_xlabel("Pricing Agent")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def plot_occupancy_comparison(df_comparison: pd.DataFrame) -> plt.Figure:
    """Generates occupancy comparison bar chart.

    Args:
        df_comparison: DataFrame with columns 'Agent' and 'Occupancy Rate'.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df_comparison,
        x="Agent",
        y="Occupancy Rate",
        hue="Agent",
        palette="magma",
        ax=ax,
        legend=False,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Average Occupancy Rate Comparison")
    ax.set_ylabel("Occupancy Rate")
    ax.set_xlabel("Pricing Agent")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def plot_training_rewards(df_metrics: pd.DataFrame, rolling_window: int = 50) -> plt.Figure:
    """Generates training reward curve.

    Args:
        df_metrics: DataFrame containing 'episode' and 'reward'.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="reward",
        alpha=0.3,
        color="royalblue",
        label="Raw Reward",
        ax=ax,
    )
    df_metrics["reward_rolling"] = df_metrics["reward"].rolling(window=rolling_window, min_periods=1).mean()
    sns.lineplot(
        data=df_metrics,
        x="episode",
        y="reward_rolling",
        linewidth=2.0,
        color="darkblue",
        label=f"Rolling Average ({rolling_window} Ep)",
        ax=ax,
    )
    ax.set_title("Agent Reward Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_training_loss(df_metrics: pd.DataFrame, rolling_window: int = 50) -> plt.Figure:
    """Generates DQN loss curve.

    Args:
        df_metrics: DataFrame containing 'episode' and 'loss'.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df_loss = df_metrics[df_metrics["loss"] > 0]
    if df_loss.empty:
        df_loss = df_metrics

    sns.lineplot(
        data=df_loss,
        x="episode",
        y="loss",
        alpha=0.3,
        color="crimson",
        label="Raw Loss",
        ax=ax,
    )
    df_loss["loss_rolling"] = df_loss["loss"].rolling(window=rolling_window, min_periods=1).mean()
    sns.lineplot(
        data=df_loss,
        x="episode",
        y="loss_rolling",
        linewidth=2.0,
        color="darkred",
        label=f"Rolling Average ({rolling_window} Ep)",
        ax=ax,
    )
    ax.set_title("DQN Optimizer MSE Loss Curve (Log Scale)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_price_distribution_boxplot(agent_prices: Dict[str, List[float]]) -> plt.Figure:
    """Generates boxplot price distributions of chosen prices by agent.

    Args:
        agent_prices: Dict mapping Agent Name -> list of prices chosen during evaluation.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    records = []
    for agent, prices in agent_prices.items():
        for p in prices:
            records.append({"Agent": agent, "Price": p})
    df = pd.DataFrame(records)

    sns.boxplot(data=df, x="Agent", y="Price", hue="Agent", palette="Set2", ax=ax)
    ax.set_title("Price Distribution Comparison Across Agents")
    ax.set_ylabel("Offered Price (INR)")
    ax.set_xlabel("Pricing Agent")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig
