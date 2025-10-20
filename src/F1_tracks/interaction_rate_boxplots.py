#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for interaction rate metrics.
1. overall_interaction_rate by Pretraining condition
2. overall_interaction_rate by F1_condition
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def create_boxplot_with_scatter(data, x_col, y_col, title, ax, color_mapping=None, order=None):
    """Create a boxplot with superimposed scatter plot."""

    # Create boxplot with no fill and black outline
    box_plot = sns.boxplot(data=data, x=x_col, y=y_col, ax=ax, order=order)

    # Style boxplot: no fill, black outline
    for patch in box_plot.patches:
        patch.set_facecolor("none")  # No fill
        patch.set_edgecolor("black")  # Black outline
        patch.set_linewidth(1.5)

    # Style the whiskers, caps, and medians to black
    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(1.5)

    # Superimpose scatter plot with specified colors
    if color_mapping and order:
        for i, condition in enumerate(order):
            condition_data = data[data[x_col] == condition]
            color = color_mapping.get(condition, "black")

            # Add jitter manually for better control
            y_values = condition_data[y_col].values
            x_positions = np.random.normal(i, 0.1, size=len(y_values))

            ax.scatter(x_positions, y_values, c=color, s=30, alpha=0.7, edgecolors="black", linewidth=0.5)
    else:
        # Fallback to default stripplot
        sns.stripplot(
            data=data, x=x_col, y=y_col, ax=ax, size=6, alpha=0.7, jitter=True, dodge=False, color="black", order=order
        )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    # Add sample size annotations
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        condition = label.get_text()
        n = len(data[data[x_col] == condition])
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")

    # Perform statistical tests if there are multiple groups
    unique_conditions = data[x_col].unique()
    if len(unique_conditions) >= 2:
        # Perform pairwise comparisons
        groups = [data[data[x_col] == condition][y_col] for condition in unique_conditions]

        if len(unique_conditions) == 2:
            # Mann-Whitney U test for two groups
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        else:
            # Kruskal-Wallis test for multiple groups
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

        # Add statistical test result to plot
        ax.text(
            0.02,
            0.98,
            f"{test_name}: p = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    return ax


def main():
    """Main function to create the boxplots with scatter plots."""

    # Dataset path
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Load dataset
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")
    else:
        print("Column 'Date' not found in dataset. Skipping date-based filtering.")

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Try to identify the correct column names
    interaction_rate_col = None
    pretraining_col = None
    f1_condition_col = None
    ball_condition_col = None

    # Look for interaction rate columns
    if "overall_interaction_rate" in df.columns:
        interaction_rate_col = "overall_interaction_rate"
    else:
        # Look for similar columns
        for col in df.columns:
            if "interaction" in col.lower() and "rate" in col.lower() and "overall" in col.lower():
                interaction_rate_col = col
                break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    # Look for F1_condition column
    for col in df.columns:
        if "f1_condition" in col.lower() or "f1condition" in col.lower():
            f1_condition_col = col
            break

    # Look for ball condition column - prioritize ball_condition over ball_identity
    if "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"
    elif "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"
    else:
        for col in df.columns:
            if "ball" in col.lower() and ("condition" in col.lower() or "identity" in col.lower()):
                ball_condition_col = col
                break

    # If exact matches not found, use manual mapping
    if pretraining_col is None:
        train_cols = [col for col in df.columns if "train" in col.lower()]
        if train_cols:
            pretraining_col = train_cols[0]
            print(f"Using '{pretraining_col}' as pretraining column")

    # Check if we found all required columns
    required_columns = {
        "interaction_rate": interaction_rate_col,
        "pretraining": pretraining_col,
        "f1_condition": f1_condition_col,
        "ball_condition": ball_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  Interaction rate: {interaction_rate_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  F1 condition: {f1_condition_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    df_clean = df[[interaction_rate_col, pretraining_col, f1_condition_col, ball_condition_col]]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, f1_condition_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the values
    sample_data = df_clean[[interaction_rate_col, ball_condition_col, pretraining_col, f1_condition_col]].head(10)
    print(f"\nSample data to verify values:")
    print(sample_data)

    # Try different possible values for test ball
    test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
    test_ball_data = pd.DataFrame()

    for test_val in test_ball_values:
        test_subset = df_clean[df_clean[ball_condition_col] == test_val]
        if not test_subset.empty:
            test_ball_data = test_subset
            print(f"Found test ball data using value: '{test_val}'")
            break

    # If no test ball found, use all data and warn user
    if test_ball_data.empty:
        print("No specific 'test' ball data found. Using all data.")
        print(f"Available ball conditions: {df_clean[ball_condition_col].value_counts()}")
        test_ball_data = df_clean

    print(f"Data shape for analysis: {test_ball_data.shape}")

    # Print unique values and missing data info
    print(f"Unique {pretraining_col} values: {test_ball_data[pretraining_col].unique()}")
    print(f"Unique {f1_condition_col} values: {test_ball_data[f1_condition_col].unique()}")

    # Check for missing values in the interaction rate metric
    print(f"\nMissing values analysis:")
    print(
        f"  {interaction_rate_col}: {test_ball_data[interaction_rate_col].isna().sum()} missing out of {len(test_ball_data)}"
    )

    # Show actual values for debugging
    print(f"\nSample of actual values:")
    print(f"{interaction_rate_col} (first 10 non-null values):")
    rate_sample = test_ball_data[interaction_rate_col].dropna().head(10)
    print(rate_sample.tolist() if not rate_sample.empty else "No non-null values found")

    # Check unique values to see if there are unexpected values
    print(f"\nUnique values in {interaction_rate_col} (first 20):")
    unique_rate_vals = test_ball_data[interaction_rate_col].unique()
    print(f"Found {len(unique_rate_vals)} unique values: {unique_rate_vals[:20]}...")

    # Show distribution by grouping variables
    print(f"\nData distribution by {pretraining_col}:")
    pretraining_dist = test_ball_data[pretraining_col].value_counts()
    print(pretraining_dist)

    print(f"\nData distribution by {f1_condition_col}:")
    f1_condition_dist = test_ball_data[f1_condition_col].value_counts()
    print(f1_condition_dist)

    # Check if we have valid data for the metric by grouping conditions
    print(f"\nValid (non-missing) data by pretraining condition:")
    valid_data = test_ball_data[test_ball_data[interaction_rate_col].notna()]
    if not valid_data.empty:
        valid_dist = valid_data[pretraining_col].value_counts()
        print(valid_dist)
    else:
        print("  No valid data found!")

    print(f"\nValid (non-missing) data by F1 condition:")
    if not valid_data.empty:
        valid_dist = valid_data[f1_condition_col].value_counts()
        print(valid_dist)
    else:
        print("  No valid data found!")

    # Prepare data for plotting - remove invalid values
    plot_data = test_ball_data.copy()

    # Remove only truly invalid values (inf, -inf, but keep finite numbers including 0)
    mask = pd.isna(plot_data[interaction_rate_col]) | np.isinf(plot_data[interaction_rate_col])
    plot_data_filtered = plot_data[~mask]

    print(f"\nAfter removing NaN/inf: {len(plot_data_filtered)} out of {len(test_ball_data)} remain")

    if len(plot_data_filtered) < 2:
        print(f"Insufficient valid data for plotting (only {len(plot_data_filtered)} valid points)")
        return

    # Create subplots - 1x2 grid for pretraining and F1_condition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Define color mappings
    pretraining_colors = {"n": "blue", "y": "orange"}

    f1_condition_colors = {"control": "blue", "pretrained": "orange", "pretrained_unlocked": "green"}

    # Define ordering
    pretraining_order = ["n", "y"]  # n first, then y
    f1_condition_order = ["control", "pretrained_unlocked", "pretrained"]  # control, pretrained_unlocked, pretrained

    # Get actual unique values and filter orders to only include existing values
    actual_pretraining_values = plot_data_filtered[pretraining_col].unique()
    actual_f1_values = plot_data_filtered[f1_condition_col].unique()

    # Filter orders to only include values that exist in the data
    filtered_pretraining_order = [val for val in pretraining_order if val in actual_pretraining_values]
    filtered_f1_order = [val for val in f1_condition_order if val in actual_f1_values]

    print(f"Pretraining values in data: {actual_pretraining_values}")
    print(f"F1 condition values in data: {actual_f1_values}")
    print(f"Using pretraining order: {filtered_pretraining_order}")
    print(f"Using F1 condition order: {filtered_f1_order}")

    print(f"Creating plots with {len(plot_data_filtered)} data points")

    # Show final distribution by grouping variables
    print(f"Final distribution by {pretraining_col}:")
    for pretraining_val in plot_data_filtered[pretraining_col].unique():
        count = len(plot_data_filtered[plot_data_filtered[pretraining_col] == pretraining_val])
        print(f"  {pretraining_val}: {count} data points")

    print(f"Final distribution by {f1_condition_col}:")
    for f1_condition_val in plot_data_filtered[f1_condition_col].unique():
        count = len(plot_data_filtered[plot_data_filtered[f1_condition_col] == f1_condition_val])
        print(f"  {f1_condition_val}: {count} data points")

    # Plot 1: interaction_rate by Pretraining condition
    create_boxplot_with_scatter(
        data=plot_data_filtered,
        x_col=pretraining_col,
        y_col=interaction_rate_col,
        title="Overall Interaction Rate by Pretraining Condition",
        ax=ax1,
        color_mapping=pretraining_colors,
        order=filtered_pretraining_order,
    )

    # Plot 2: interaction_rate by F1_condition
    create_boxplot_with_scatter(
        data=plot_data_filtered,
        x_col=f1_condition_col,
        y_col=interaction_rate_col,
        title="Overall Interaction Rate by F1 Condition",
        ax=ax2,
        color_mapping=f1_condition_colors,
        order=filtered_f1_order,
    )

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "interaction_rate_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if pretraining_col and interaction_rate_col:
        print(f"\n1. Interaction Rate by {pretraining_col}:")
        rate_data = plot_data_filtered.dropna(subset=[interaction_rate_col])
        if not rate_data.empty:
            rate_summary = (
                rate_data.groupby(pretraining_col)[interaction_rate_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(4)
            )
            # Reorder the summary to match plot order
            if set(filtered_pretraining_order).issubset(rate_summary.index):
                rate_summary = rate_summary.reindex(filtered_pretraining_order)
            print(rate_summary)
        else:
            print("No valid data for interaction_rate")

    if f1_condition_col and interaction_rate_col:
        print(f"\n2. Interaction Rate by {f1_condition_col}:")
        rate_data = plot_data_filtered.dropna(subset=[interaction_rate_col])
        if not rate_data.empty:
            rate_summary = (
                rate_data.groupby(f1_condition_col)[interaction_rate_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(4)
            )
            # Reorder the summary to match plot order
            if set(filtered_f1_order).issubset(rate_summary.index):
                rate_summary = rate_summary.reindex(filtered_f1_order)
            print(rate_summary)
        else:
            print("No valid data for interaction_rate")

    # Additional analysis: Two-way comparison
    print(f"\n3. Two-way breakdown by {pretraining_col} and {f1_condition_col}:")
    if not plot_data_filtered.empty:
        two_way_summary = (
            plot_data_filtered.groupby([pretraining_col, f1_condition_col])[interaction_rate_col]
            .agg(["count", "mean", "std", "median"])
            .round(4)
        )
        print(two_way_summary)

    plt.show()

    return plot_data_filtered


if __name__ == "__main__":
    data = main()
