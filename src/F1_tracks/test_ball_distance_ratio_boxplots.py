#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for test ball distance_ratio.
1. Distance ratio for test ball by Pretraining condition
2. Distance ratio for test ball by F1_Condition
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
    ax.set_ylabel("Distance Ratio - Test Ball", fontsize=12)

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
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251001_13_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Load dataset
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Try to identify the correct column names
    distance_ratio_col = None
    pretraining_col = None
    f1_condition_col = None
    ball_condition_col = None

    # Look for distance_ratio column specifically
    if "distance_ratio" in df.columns:
        distance_ratio_col = "distance_ratio"
    else:
        # Look for any distance ratio column as fallback
        for col in df.columns:
            if "distance" in col.lower() and "ratio" in col.lower():
                distance_ratio_col = col
                break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    # Look for F1_condition column
    for col in df.columns:
        if "f1" in col.lower() and "condition" in col.lower():
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
    if distance_ratio_col is None:
        ratio_cols = [col for col in df.columns if "ratio" in col.lower()]
        if ratio_cols:
            distance_ratio_col = ratio_cols[0]
            print(f"Using '{distance_ratio_col}' as distance ratio column")

    if pretraining_col is None:
        train_cols = [col for col in df.columns if "train" in col.lower()]
        if train_cols:
            pretraining_col = train_cols[0]
            print(f"Using '{pretraining_col}' as pretraining column")

    if f1_condition_col is None:
        f1_cols = [col for col in df.columns if "f1" in col.lower()]
        if f1_cols:
            f1_condition_col = f1_cols[0]
            print(f"Using '{f1_condition_col}' as F1 condition column")

    # Check if we found all required columns
    required_columns = {
        "distance_ratio": distance_ratio_col,
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
    print(f"  Distance Ratio: {distance_ratio_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  F1 condition: {f1_condition_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    df_clean = df[[distance_ratio_col, pretraining_col, f1_condition_col, ball_condition_col]].dropna()

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the distance ratio values
    sample_data = df_clean[[distance_ratio_col, ball_condition_col]].head(10)
    print(f"\nSample data to verify distance ratio values:")
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

    # Print unique values
    print(f"Unique {pretraining_col} values: {test_ball_data[pretraining_col].unique()}")
    print(f"Unique {f1_condition_col} values: {test_ball_data[f1_condition_col].unique()}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set up color palettes
    pretraining_palette = sns.color_palette("Set2", n_colors=len(test_ball_data[pretraining_col].unique()))
    f1_palette = sns.color_palette("Set1", n_colors=len(test_ball_data[f1_condition_col].unique()))

    # Plot 1: Distance ratio by Pretraining condition
    create_boxplot_with_scatter(
        data=test_ball_data,
        x_col=pretraining_col,
        y_col=distance_ratio_col,
        title="Test Ball Distance Ratio by Pretraining Condition",
        ax=ax1,
        color_palette=pretraining_palette,
    )

    # Plot 2: Distance ratio by F1_Condition
    create_boxplot_with_scatter(
        data=test_ball_data,
        x_col=f1_condition_col,
        y_col=distance_ratio_col,
        title="Test Ball Distance Ratio by F1 Condition",
        ax=ax2,
        color_palette=f1_palette,
    )

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "test_ball_distance_ratio_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if pretraining_col and distance_ratio_col:
        print(f"\n1. Test Ball Distance Ratio by {pretraining_col}:")
        pretraining_summary = (
            test_ball_data.groupby(pretraining_col)[distance_ratio_col]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .round(3)
        )
        print(pretraining_summary)

    if f1_condition_col and distance_ratio_col:
        print(f"\n2. Test Ball Distance Ratio by {f1_condition_col}:")
        f1_summary = (
            test_ball_data.groupby(f1_condition_col)[distance_ratio_col]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .round(3)
        )
        print(f1_summary)

    plt.show()

    return test_ball_data


if __name__ == "__main__":
    data = main()
