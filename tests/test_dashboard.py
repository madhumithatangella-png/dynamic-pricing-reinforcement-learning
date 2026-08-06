"""Unit tests verifying Streamlit dashboard charts compilation and pages structure."""

import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dashboard.charts import (
    plot_revenue_comparison,
    plot_occupancy_comparison,
    plot_training_rewards,
    plot_training_loss,
    plot_price_distribution_boxplot
)


def test_revenue_comparison_chart() -> None:
    """Verifies that plot_revenue_comparison builds a valid Matplotlib figure."""
    df = pd.DataFrame([
        {"Agent": "DQN", "Total Revenue": 10000.0},
        {"Agent": "Fixed", "Total Revenue": 5000.0}
    ])
    fig = plot_revenue_comparison(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_occupancy_comparison_chart() -> None:
    """Verifies that plot_occupancy_comparison builds a valid Matplotlib figure."""
    df = pd.DataFrame([
        {"Agent": "DQN", "Occupancy Rate": 0.6},
        {"Agent": "Fixed", "Occupancy Rate": 0.4}
    ])
    fig = plot_occupancy_comparison(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_training_rewards_chart() -> None:
    """Verifies that plot_training_rewards builds a valid Matplotlib figure."""
    df = pd.DataFrame([
        {"episode": 1, "reward": -10.0},
        {"episode": 2, "reward": 50.0}
    ])
    fig = plot_training_rewards(df, rolling_window=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_training_loss_chart() -> None:
    """Verifies that plot_training_loss builds a valid Matplotlib figure."""
    df = pd.DataFrame([
        {"episode": 1, "loss": 100.0},
        {"episode": 2, "loss": 10.0}
    ])
    fig = plot_training_loss(df, rolling_window=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_price_distribution_boxplot_chart() -> None:
    """Verifies that plot_price_distribution_boxplot builds a valid Matplotlib figure."""
    data = {
        "DQN": [3000.0, 4000.0, 5000.0],
        "Fixed": [5000.0, 5000.0, 5000.0]
    }
    fig = plot_price_distribution_boxplot(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dashboard_pages_importable() -> None:
    """Verifies dashboard files are well-formed and can be imported conceptually."""
    from dashboard.pages import dashboard, prediction, training, comparison, evaluation, system_status, about
    assert dashboard is not None
    assert prediction is not None
    assert training is not None
    assert comparison is not None
    assert evaluation is not None
    assert system_status is not None
    assert about is not None

