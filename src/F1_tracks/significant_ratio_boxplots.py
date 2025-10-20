#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for significant_ratio metric.
Analyzes significant_ratio by Pretraining condition.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def create_boxplot_with_scatter(data, x_col, y_col, title, ax, color_palette=None):
    """Create a boxplot with superimposed scatter plot."""

    # Create boxplot
    box_plot = sns.boxplot(data=data, x=x_col, y=y_col, ax=ax, hue=x_col, palette=color_palette, legend=False)

    # Make boxplot transparent
    for patch in box_plot.patches:
        patch.set_alpha(0.7)

    # Superimpose scatter plot with jitter
    sns.stripplot(data=data, x=x_col, y=y_col, ax=ax, size=4, alpha=0.6, jitter=True, dodge=False, color="black")

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
    """Main function to create the boxplot with scatter plot."""

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
    significant_ratio_col = None
    pretraining_col = None
    ball_condition_col = None

    # Look for significant_ratio column
    if "significant_ratio" in df.columns:
        significant_ratio_col = "significant_ratio"
    else:
        for col in df.columns:
            if "significant" in col.lower() and "ratio" in col.lower():
                significant_ratio_col = col
                break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
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
        "significant_ratio": significant_ratio_col,
        "pretraining": pretraining_col,
        "ball_condition": ball_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  Significant ratio: {significant_ratio_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    df_clean = df[[significant_ratio_col, pretraining_col, ball_condition_col]]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the values
    sample_data = df_clean[[significant_ratio_col, ball_condition_col, pretraining_col]].head(10)
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

    # Check for missing values in the metric
    print(f"\nMissing values analysis:")
    print(
        f"  {significant_ratio_col}: {test_ball_data[significant_ratio_col].isna().sum()} missing out of {len(test_ball_data)}"
    )

    # Show actual values for debugging
    print(f"\nSample of actual values:")
    print(f"{significant_ratio_col} (first 10 non-null values):")
    ratio_sample = test_ball_data[significant_ratio_col].dropna().head(10)
    print(ratio_sample.tolist() if not ratio_sample.empty else "No non-null values found")

    # Check unique values to see range
    print(f"\nValue range for {significant_ratio_col}:")
    ratio_values = test_ball_data[significant_ratio_col].dropna()
    if not ratio_values.empty:
        print(f"Min: {ratio_values.min():.3f}, Max: {ratio_values.max():.3f}")
        print(f"Mean: {ratio_values.mean():.3f}, Std: {ratio_values.std():.3f}")
    else:
        print("No valid values found")

    # Show distribution by pretraining condition
    print(f"\nData distribution by {pretraining_col}:")
    pretraining_dist = test_ball_data[pretraining_col].value_counts()
    print(pretraining_dist)

    # Check if we have valid data by pretraining condition
    print(f"\nValid (non-missing) data by pretraining condition:")
    valid_data = test_ball_data[test_ball_data[significant_ratio_col].notna()]
    if not valid_data.empty:
        valid_dist = valid_data[pretraining_col].value_counts()
        print(valid_dist)
    else:
        print("  No valid data found!")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Set up color palette
    pretraining_palette = sns.color_palette("Set2", n_colors=len(test_ball_data[pretraining_col].unique()))

    # Plot significant_ratio by Pretraining condition (only for non-missing data)
    ratio_data = test_ball_data.dropna(subset=[significant_ratio_col])
    if not ratio_data.empty:
        create_boxplot_with_scatter(
            data=ratio_data,
            x_col=pretraining_col,
            y_col=significant_ratio_col,
            title="Significant Ratio by Pretraining Condition",
            ax=ax,
            color_palette=pretraining_palette,
        )
    else:
        ax.text(0.5, 0.5, "No valid data for significant_ratio", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Significant Ratio by Pretraining Condition")

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "significant_ratio_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if pretraining_col and significant_ratio_col:
        print(f"\nSignificant Ratio by {pretraining_col}:")
        ratio_data = test_ball_data.dropna(subset=[significant_ratio_col])
        if not ratio_data.empty:
            ratio_summary = (
                ratio_data.groupby(pretraining_col)[significant_ratio_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(ratio_summary)
        else:
            print("No valid data for significant_ratio")

    plt.show()

    return test_ball_data


if __name__ == "__main__":
    data = main()
