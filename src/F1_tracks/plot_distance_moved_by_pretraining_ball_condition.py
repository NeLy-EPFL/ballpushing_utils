#!/usr/bin/env python3
"""
Script to plot distance_moved by Pretraining and ball_condition for F1 dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_dataset(dataset_path):
    """Load the feather dataset and return the DataFrame."""
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def check_columns(df, required_cols):
    """Check if required columns exist and suggest alternatives if missing."""
    available_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing columns: {missing_cols}")

        # Suggest alternative column names
        for missing_col in missing_cols:
            if "distance" in missing_col.lower():
                distance_cols = [col for col in df.columns if "distance" in col.lower()]
                if distance_cols:
                    print(f"Available distance columns: {distance_cols}")
            elif "train" in missing_col.lower():
                train_cols = [col for col in df.columns if "train" in col.lower()]
                if train_cols:
                    print(f"Available training columns: {train_cols}")
            elif "ball" in missing_col.lower():
                ball_cols = [col for col in df.columns if "ball" in col.lower()]
                if ball_cols:
                    print(f"Available ball columns: {ball_cols}")

        return False

    return True


def create_distance_plot(df, output_dir):
    """Create plots showing distance_moved by Pretraining and ball_condition."""

    # Check required columns
    required_cols = ["distance_moved", "Pretraining", "ball_condition"]

    # Try to find alternative column names if exact matches don't exist
    column_mapping = {}
    for col in required_cols:
        if col in df.columns:
            column_mapping[col] = col
        else:
            # Try to find similar columns
            if "distance" in col.lower():
                distance_cols = [c for c in df.columns if "distance" in c.lower()]
                if distance_cols:
                    column_mapping[col] = distance_cols[0]
                    print(f"Using '{distance_cols[0]}' for '{col}'")
            elif "train" in col.lower():
                train_cols = [c for c in df.columns if "train" in c.lower()]
                if train_cols:
                    column_mapping[col] = train_cols[0]
                    print(f"Using '{train_cols[0]}' for '{col}'")
            elif "ball" in col.lower():
                ball_cols = [c for c in df.columns if "ball" in c.lower()]
                if ball_cols:
                    column_mapping[col] = ball_cols[0]
                    print(f"Using '{ball_cols[0]}' for '{col}'")

    # Check if we have all required columns (or their alternatives)
    if len(column_mapping) < len(required_cols):
        print("Cannot find all required columns. Available columns:")
        print(list(df.columns))
        return

    # Rename columns for consistency
    df_plot = df.copy()
    for original, mapped in column_mapping.items():
        if mapped != original:
            df_plot[original] = df_plot[mapped]

    # Remove rows with missing values in key columns
    df_plot = df_plot.dropna(subset=["distance_moved", "Pretraining", "ball_condition"])

    print(f"Data shape after removing NaNs: {df_plot.shape}")
    print(f"Unique Pretraining values: {df_plot['Pretraining'].unique()}")
    print(f"Unique ball_condition values: {df_plot['ball_condition'].unique()}")

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Distance Moved by Pretraining and Ball Condition", fontsize=16, fontweight="bold")

    # 1. Box plot
    ax1 = axes[0, 0]
    sns.boxplot(data=df_plot, x="Pretraining", y="distance_moved", hue="ball_condition", ax=ax1)
    ax1.set_title("Box Plot: Distance Moved by Pretraining and Ball Condition")
    ax1.set_ylabel("Distance Moved")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=df_plot, x="Pretraining", y="distance_moved", hue="ball_condition", ax=ax2)
    ax2.set_title("Violin Plot: Distance Moved by Pretraining and Ball Condition")
    ax2.set_ylabel("Distance Moved")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Bar plot with error bars
    ax3 = axes[1, 0]
    sns.barplot(data=df_plot, x="Pretraining", y="distance_moved", hue="ball_condition", ci=95, capsize=0.05, ax=ax3)
    ax3.set_title("Bar Plot: Mean Distance Moved (±95% CI)")
    ax3.set_ylabel("Distance Moved")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Strip plot with overlaid means
    ax4 = axes[1, 1]
    sns.stripplot(
        data=df_plot, x="Pretraining", y="distance_moved", hue="ball_condition", dodge=True, alpha=0.7, ax=ax4
    )
    sns.pointplot(
        data=df_plot,
        x="Pretraining",
        y="distance_moved",
        hue="ball_condition",
        dodge=True,
        join=False,
        markers="D",
        scale=1.5,
        ax=ax4,
    )
    ax4.set_title("Strip Plot: Individual Points with Means")
    ax4.set_ylabel("Distance Moved")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "distance_moved_by_pretraining_ball_condition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Show summary statistics
    print("\nSummary Statistics:")
    summary_stats = (
        df_plot.groupby(["Pretraining", "ball_condition"])["distance_moved"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .round(3)
    )
    print(summary_stats)

    # Statistical test (if scipy is available)
    try:
        from scipy import stats

        print("\nStatistical Tests:")
        # Test for each pretraining condition
        for pretraining in df_plot["Pretraining"].unique():
            subset = df_plot[df_plot["Pretraining"] == pretraining]
            conditions = subset["ball_condition"].unique()

            if len(conditions) >= 2:
                group1 = subset[subset["ball_condition"] == conditions[0]]["distance_moved"]
                group2 = subset[subset["ball_condition"] == conditions[1]]["distance_moved"]

                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                print(f"Mann-Whitney U test for {pretraining}:")
                print(f"  {conditions[0]} vs {conditions[1]}: p = {p_value:.4f}")

    except ImportError:
        print("scipy not available for statistical tests")

    plt.show()

    return output_path


def main():
    """Main function to run the plotting script."""

    # Dataset path
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Output directory
    output_dir = Path(__file__).parent

    # Load dataset
    df = load_dataset(dataset_path)
    if df is None:
        return

    # Create plots
    create_distance_plot(df, output_dir)


if __name__ == "__main__":
    main()
