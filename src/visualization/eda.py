"""EDA Plotting functions for Dynamic Pricing RL.

Generates beautiful, production-quality visualizations to analyze bookings,
pricing, distributions, cancellations, and seasonality.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.logger import get_logger

logger = get_logger("eda")

# Set global plotting style for premium design feel
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    }
)


def plot_missing_values(df: pd.DataFrame, save_path: Path) -> None:
    """Generates and saves a missing values heatmap/bar chart."""
    plt.figure(figsize=(10, 6))
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        # If no missing values, plot an empty message chart
        plt.text(
            0.5,
            0.5,
            "No Missing Values Detected",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.title("Missing Values Profile")
    else:
        sns.barplot(x=missing.values, y=missing.index, palette="viridis")
        plt.title("Missing Values Count per Column")
        plt.xlabel("Number of Missing Values")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Missing Values plot to: {save_path}")


def plot_booking_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plots the distribution of booking outcomes (Canceled vs Checked-in)."""
    plt.figure(figsize=(6, 5))
    counts = df["is_canceled"].value_counts()
    labels = ["Not Canceled", "Canceled"]

    sns.barplot(
        x=counts.index.map({0: "Not Canceled", 1: "Canceled"}),
        y=counts.values,
        palette="crest",
    )
    plt.title("Distribution of Booking Cancellations")
    plt.xlabel("Booking Status")
    plt.ylabel("Number of Bookings")

    # Add percentages above bars
    total = len(df)
    for i, val in enumerate(counts.values):
        pct = (val / total) * 100
        plt.text(i, val + (total * 0.01), f"{pct:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Booking Distribution plot to: {save_path}")


def plot_hotel_type_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plots booking count distribution across hotel types."""
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="hotel", palette="flare")
    plt.title("Distribution of Bookings by Hotel Type")
    plt.xlabel("Hotel Type")
    plt.ylabel("Booking Count")

    # Add counts on top of bars
    counts = df["hotel"].value_counts()
    for i, val in enumerate(counts.values):
        plt.text(i, val + (len(df) * 0.01), f"{val:,}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Hotel Type Distribution plot to: {save_path}")


def plot_adr_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plots the distribution (histogram + KDE) of ADR (Average Daily Rate)."""
    plt.figure(figsize=(10, 6))
    # Filter extremely high rates for visualization clarity (e.g. adr > 500)
    vis_df = df[df["adr"] <= 500]

    sns.histplot(data=vis_df, x="adr", kde=True, bins=50, color="teal")
    plt.title("Distribution of Average Daily Rate (ADR) - Capped at $500")
    plt.xlabel("ADR ($)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved ADR Distribution plot to: {save_path}")


def plot_adr_by_hotel(df: pd.DataFrame, save_path: Path) -> None:
    """Boxplot comparing ADR across hotel types, split by cancellation."""
    plt.figure(figsize=(8, 6))
    # Filter for visualization
    vis_df = df[df["adr"] <= 500]

    sns.boxplot(
        data=vis_df,
        x="hotel",
        y="adr",
        hue="is_canceled",
        palette="Set2",
    )
    plt.title("ADR Distribution by Hotel Type and Cancellation Status")
    plt.xlabel("Hotel Type")
    plt.ylabel("ADR ($)")
    plt.legend(title="Canceled", labels=["No", "Yes"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved ADR by Hotel plot to: {save_path}")


def plot_cancellation_analysis(df: pd.DataFrame, save_path: Path) -> None:
    """Plots cancellation rates by market segment."""
    plt.figure(figsize=(10, 6))
    cancel_rates = (
        df.groupby("market_segment")["is_canceled"].mean().reset_index()
    )
    cancel_rates = cancel_rates.sort_values(by="is_canceled", ascending=False)

    sns.barplot(
        data=cancel_rates,
        x="is_canceled",
        y="market_segment",
        palette="coolwarm",
    )
    plt.title("Cancellation Rate by Market Segment")
    plt.xlabel("Cancellation Rate (Proportion)")
    plt.ylabel("Market Segment")

    # Add labels on bars
    for i, row in enumerate(cancel_rates.itertuples()):
        plt.text(
            row.is_canceled + 0.01,
            i,
            f"{row.is_canceled * 100:.1f}%",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Cancellation Analysis plot to: {save_path}")


def plot_monthly_bookings(df: pd.DataFrame, save_path: Path) -> None:
    """Plots booking count trends over months of arrival, ordered chronologically."""
    plt.figure(figsize=(12, 6))

    # Standard order of months
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    sns.countplot(
        data=df,
        x="arrival_date_month",
        order=month_order,
        hue="hotel",
        palette="magma",
    )
    plt.title("Booking Volume by Month and Hotel Type")
    plt.xlabel("Month")
    plt.ylabel("Number of Bookings")
    plt.xticks(rotation=45)
    plt.legend(title="Hotel Type")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Monthly Bookings plot to: {save_path}")


def plot_lead_time_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plots lead time distribution using histogram."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="lead_time", bins=50, kde=True, color="purple")
    plt.title("Distribution of Lead Time (Days before arrival)")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Lead Time Distribution plot to: {save_path}")


def plot_lead_time_vs_cancellation(df: pd.DataFrame, save_path: Path) -> None:
    """KDE plot comparing lead times for canceled vs non-canceled bookings."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df,
        x="lead_time",
        hue="is_canceled",
        fill=True,
        common_norm=False,
        palette="muted",
        alpha=0.5,
    )
    plt.title("Lead Time Distribution by Cancellation Status")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Density")
    plt.legend(title="Canceled", labels=["Yes", "No"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Lead Time vs Cancellation plot to: {save_path}")


def plot_market_segment(df: pd.DataFrame, save_path: Path) -> None:
    """Plots booking distribution by market segment."""
    plt.figure(figsize=(10, 6))
    segment_counts = df["market_segment"].value_counts().reset_index()

    sns.barplot(
        data=segment_counts,
        x="count",
        y="market_segment",
        palette="viridis",
    )
    plt.title("Distribution of Bookings by Market Segment")
    plt.xlabel("Booking Count")
    plt.ylabel("Market Segment")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Market Segment plot to: {save_path}")


def plot_customer_type(df: pd.DataFrame, save_path: Path) -> None:
    """Plots distribution of booking customer types."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y="customer_type", palette="rocket")
    plt.title("Distribution of Bookings by Customer Type")
    plt.xlabel("Booking Count")
    plt.ylabel("Customer Type")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Customer Type plot to: {save_path}")


def plot_correlation_matrix(df: pd.DataFrame, save_path: Path) -> None:
    """Plots correlation heatmap of numerical columns."""
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    # Drop columns that are completely constant to prevent NaN correlations
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    corr = numeric_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,
    )
    plt.title("Correlation Matrix of Numeric Features")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved Correlation Matrix plot to: {save_path}")


def generate_all_plots(df: pd.DataFrame, target_dir: Path) -> None:
    """Wrapper function to invoke and save all 12 target EDA plots.

    Args:
        df: Processed DataFrame (ideally containing engineered features).
        target_dir: Directory where figures should be saved.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating all 12 EDA figures inside: {target_dir}")

    plot_missing_values(df, target_dir / "missing_values.png")
    plot_booking_distribution(df, target_dir / "booking_distribution.png")
    plot_hotel_type_distribution(df, target_dir / "hotel_type_distribution.png")
    plot_adr_distribution(df, target_dir / "adr_distribution.png")
    plot_adr_by_hotel(df, target_dir / "adr_by_hotel.png")
    plot_cancellation_analysis(df, target_dir / "cancellation_analysis.png")
    plot_monthly_bookings(df, target_dir / "monthly_bookings.png")
    plot_lead_time_distribution(df, target_dir / "lead_time_distribution.png")
    plot_lead_time_vs_cancellation(
        df, target_dir / "lead_time_vs_cancellation.png"
    )
    plot_market_segment(df, target_dir / "market_segment.png")
    plot_customer_type(df, target_dir / "customer_type.png")
    plot_correlation_matrix(df, target_dir / "correlation_matrix.png")

    logger.info("All EDA figures generated successfully.")
