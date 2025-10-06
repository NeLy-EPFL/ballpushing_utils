#!/usr/bin/env python3
"""
Script to analyze binary metrics for F1 dataset.
Creates comprehensive plots for has_finished, has_long_pauses, has_major, and has_significant
by Pretraining and F1_Condition.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact


def calculate_proportions_and_stats(data, group_col, binary_col):
    """Calculate proportions and statistical tests for binary data."""

    # Calculate proportions
    prop_data = data.groupby(group_col)[binary_col].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_positive"] = prop_data["sum"]
    prop_data["n_total"] = prop_data["count"]

    # Perform statistical test
    groups = data[group_col].unique()
    if len(groups) >= 2:
        # Create contingency table
        contingency = pd.crosstab(data[group_col], data[binary_col])

        if len(groups) == 2:
            # Fisher's exact test for 2x2 table
            if contingency.shape == (2, 2):
                try:
                    odds_ratio, p_value = fisher_exact(contingency)
                    test_name = "Fisher's exact"
                except:
                    # Fallback to chi-square
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    test_name = "Chi-square"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                test_name = "Chi-square"
        else:
            # Chi-square test for multiple groups
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            test_name = "Chi-square"
    else:
        p_value = np.nan
        test_name = "No test"

    return prop_data, p_value, test_name


def create_binary_barplot(data, group_col, binary_metrics, title_prefix, ax_row, fig):
    """Create bar plots for binary metrics."""

    n_metrics = len(binary_metrics)

    for i, metric in enumerate(binary_metrics):
        ax = ax_row[i]

        # Calculate proportions and statistics
        prop_data, p_value, test_name = calculate_proportions_and_stats(data, group_col, metric)

        # Create bar plot
        bars = ax.bar(prop_data[group_col], prop_data["proportion"], alpha=0.7, edgecolor="black", linewidth=1)

        # Color bars based on group
        colors = sns.color_palette("Set2", n_colors=len(prop_data))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

        # Add sample size and proportion labels on bars
        for j, (idx, row) in enumerate(prop_data.iterrows()):
            height = row["proportion"]
            ax.text(
                j,
                height + 0.01,
                f"{row['n_positive']}/{row['n_total']}\n({height:.2%})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Formatting
        ax.set_title(f"{title_prefix}\n{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, 1.1)

        # Add statistical test result
        if not np.isnan(p_value):
            ax.text(
                0.02,
                0.98,
                f"{test_name}\np = {p_value:.4f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        # Rotate x-axis labels if they're long
        if any(len(str(label)) > 8 for label in prop_data[group_col]):
            ax.tick_params(axis="x", rotation=45)


def create_heatmap(data, group_col, binary_metrics, title, ax):
    """Create a heatmap showing proportions of binary metrics by group."""

    # Calculate proportions for all metrics
    heatmap_data = []
    for metric in binary_metrics:
        prop_data, _, _ = calculate_proportions_and_stats(data, group_col, metric)
        proportions = dict(zip(prop_data[group_col], prop_data["proportion"]))
        heatmap_data.append(proportions)

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, index=[m.replace("_", " ").title() for m in binary_metrics])

    # Create heatmap
    sns.heatmap(heatmap_df, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Proportion"})

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Binary Metrics", fontsize=12)


def main():
    """Main function to create the binary metrics analysis."""

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

    # Define binary metrics to analyze
    binary_metrics = ["has_finished", "has_long_pauses", "has_major", "has_significant"]

    # Try to identify the correct column names
    pretraining_col = None
    f1_condition_col = None
    ball_condition_col = None

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

    # Check if binary metrics exist
    available_metrics = []
    for metric in binary_metrics:
        if metric in df.columns:
            available_metrics.append(metric)
        else:
            print(f"Warning: '{metric}' not found in dataset")

    if not available_metrics:
        print("No binary metrics found in dataset!")
        return

    print(f"Found binary metrics: {available_metrics}")

    # Check if we found required columns
    if pretraining_col is None:
        print("Could not find pretraining column")
        return
    if f1_condition_col is None:
        print("Could not find F1_condition column")
        return
    if ball_condition_col is None:
        print("Could not find ball_condition column")
        return

    print(f"Using columns:")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  F1 condition: {f1_condition_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Select relevant columns and clean data
    analysis_cols = [pretraining_col, f1_condition_col, ball_condition_col] + available_metrics
    df_clean = df[analysis_cols].dropna()

    print(f"Clean data shape before ball filtering: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Filter for test ball only
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

    # Use test ball data for analysis
    df_clean = test_ball_data
    print(f"Data shape for analysis (test ball only): {df_clean.shape}")

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_clean[pretraining_col].unique()}")
    print(f"Unique {f1_condition_col} values: {df_clean[f1_condition_col].unique()}")

    # Verify binary nature of metrics
    for metric in available_metrics:
        unique_vals = df_clean[metric].unique()
        print(f"{metric} unique values: {unique_vals}")
        if not set(unique_vals).issubset({0, 1, True, False}):
            print(f"Warning: {metric} may not be binary!")

    # Create comprehensive figure
    n_metrics = len(available_metrics)
    fig = plt.figure(figsize=(20, 16))

    # Create subplots: 2 rows of bar plots + 1 row of heatmaps
    gs = fig.add_gridspec(3, max(n_metrics, 2), height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

    # Row 1: Bar plots by Pretraining
    ax_pretraining = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    create_binary_barplot(
        df_clean, pretraining_col, available_metrics, "Binary Metrics by Pretraining", ax_pretraining, fig
    )

    # Row 2: Bar plots by F1_Condition
    ax_f1 = [fig.add_subplot(gs[1, i]) for i in range(n_metrics)]
    create_binary_barplot(df_clean, f1_condition_col, available_metrics, "Binary Metrics by F1 Condition", ax_f1, fig)

    # Row 3: Heatmaps
    ax_heatmap1 = fig.add_subplot(gs[2, 0])
    create_heatmap(df_clean, pretraining_col, available_metrics, "Test Ball Binary Metrics - Pretraining", ax_heatmap1)

    ax_heatmap2 = fig.add_subplot(gs[2, 1])
    create_heatmap(df_clean, f1_condition_col, available_metrics, "Test Ball Binary Metrics - F1 Condition", ax_heatmap2)

    # Overall title
    fig.suptitle("F1 Dataset: Binary Metrics Analysis (Test Ball Only)", fontsize=16, fontweight="bold", y=0.98)

    # Save the plot
    output_path = Path(__file__).parent / "binary_metrics_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print detailed summary statistics
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY STATISTICS")
    print("=" * 80)

    for group_col in [pretraining_col, f1_condition_col]:
        print(f"\n{group_col.upper()} ANALYSIS:")
        print("-" * 50)

        for metric in available_metrics:
            print(f"\n{metric}:")
            prop_data, p_value, test_name = calculate_proportions_and_stats(df_clean, group_col, metric)

            for _, row in prop_data.iterrows():
                print(f"  {row[group_col]}: {row['n_positive']}/{row['n_total']} ({row['proportion']:.2%})")

            print(f"  Statistical test: {test_name}, p = {p_value:.4f}")

    # Create summary table
    print(f"\n\nSUMMARY TABLE:")
    print("-" * 80)

    summary_table = []
    for group_col in [pretraining_col, f1_condition_col]:
        for metric in available_metrics:
            prop_data, p_value, test_name = calculate_proportions_and_stats(df_clean, group_col, metric)

            for _, row in prop_data.iterrows():
                summary_table.append(
                    {
                        "Grouping": group_col,
                        "Group": row[group_col],
                        "Metric": metric,
                        "Count": f"{row['n_positive']}/{row['n_total']}",
                        "Proportion": f"{row['proportion']:.2%}",
                        "P_value": f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
                    }
                )

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))

    plt.show()

    return df_clean


if __name__ == "__main__":
    data = main()
