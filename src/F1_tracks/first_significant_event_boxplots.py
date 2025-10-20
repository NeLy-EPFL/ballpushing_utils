#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for first significant event metrics.
1. first_significant_event by Pretraining condition
2. first_significant_event_time by Pretraining condition
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
    first_significant_event_col = None
    first_significant_event_time_col = None
    pretraining_col = None
    f1_condition_col = None
    ball_condition_col = None

    # Look for first_significant_event columns
    if "first_significant_event" in df.columns:
        first_significant_event_col = "first_significant_event"

    if "first_significant_event_time" in df.columns:
        first_significant_event_time_col = "first_significant_event_time"

    # If exact matches not found, look for similar columns
    if first_significant_event_col is None:
        for col in df.columns:
            if (
                "first" in col.lower()
                and "significant" in col.lower()
                and "event" in col.lower()
                and "time" not in col.lower()
            ):
                first_significant_event_col = col
                break

    if first_significant_event_time_col is None:
        for col in df.columns:
            if (
                "first" in col.lower()
                and "significant" in col.lower()
                and "event" in col.lower()
                and "time" in col.lower()
            ):
                first_significant_event_time_col = col
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
        "first_significant_event": first_significant_event_col,
        "first_significant_event_time": first_significant_event_time_col,
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
    print(f"  First significant event: {first_significant_event_col}")
    print(f"  First significant event time: {first_significant_event_time_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  F1 condition: {f1_condition_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    # Don't drop rows where first_significant_event cols are NaN yet - we'll handle this after ball filtering
    df_clean = df[
        [
            first_significant_event_col,
            first_significant_event_time_col,
            pretraining_col,
            f1_condition_col,
            ball_condition_col,
        ]
    ]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, f1_condition_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the values
    sample_data = df_clean[
        [
            first_significant_event_col,
            first_significant_event_time_col,
            ball_condition_col,
            pretraining_col,
            f1_condition_col,
        ]
    ].head(10)
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

    # Check for missing values in the metrics
    print(f"\nMissing values analysis:")
    print(
        f"  {first_significant_event_col}: {test_ball_data[first_significant_event_col].isna().sum()} missing out of {len(test_ball_data)}"
    )
    print(
        f"  {first_significant_event_time_col}: {test_ball_data[first_significant_event_time_col].isna().sum()} missing out of {len(test_ball_data)}"
    )

    # Show actual values for debugging
    print(f"\nSample of actual values:")
    print(f"{first_significant_event_col} (first 10 non-null values):")
    event_sample = test_ball_data[first_significant_event_col].dropna().head(10)
    print(event_sample.tolist() if not event_sample.empty else "No non-null values found")

    print(f"\n{first_significant_event_time_col} (first 10 non-null values):")
    time_sample = test_ball_data[first_significant_event_time_col].dropna().head(10)
    print(time_sample.tolist() if not time_sample.empty else "No non-null values found")

    # Check unique values to see if there are unexpected values
    print(f"\nUnique values in {first_significant_event_time_col}:")
    unique_time_vals = test_ball_data[first_significant_event_time_col].unique()
    print(f"Found {len(unique_time_vals)} unique values: {unique_time_vals[:20]}...")  # Show first 20

    # Show distribution by pretraining condition
    print(f"\nData distribution by {pretraining_col}:")
    pretraining_dist = test_ball_data[pretraining_col].value_counts()
    print(pretraining_dist)

    # Check if we have valid data for each metric by pretraining condition
    print(f"\nValid (non-missing) data by pretraining condition:")
    for metric in [first_significant_event_col, first_significant_event_time_col]:
        print(f"\n{metric}:")
        valid_data = test_ball_data[test_ball_data[metric].notna()]
        if not valid_data.empty:
            valid_dist = valid_data[pretraining_col].value_counts()
            print(valid_dist)
        else:
            print("  No valid data found!")

    # Also load has_significant to cross-check
    if "has_significant" in df.columns:
        print(f"\nCross-check with has_significant binary metric:")

        # Make sure we get has_significant for the same test ball filtered data
        # First get has_significant for the same ball condition filtering
        df_with_significant = df[[pretraining_col, f1_condition_col, ball_condition_col, "has_significant"]].dropna(
            subset=[pretraining_col, f1_condition_col, ball_condition_col]
        )

        # Apply the same test ball filtering as we did for the main data
        test_ball_significant_data = pd.DataFrame()
        for test_val in test_ball_values:
            test_subset = df_with_significant[df_with_significant[ball_condition_col] == test_val]
            if not test_subset.empty:
                test_ball_significant_data = test_subset
                print(f"Found test ball has_significant data using value: '{test_val}'")
                break

        if test_ball_significant_data.empty:
            print("No specific 'test' ball data found for has_significant. Using all data.")
            test_ball_significant_data = df_with_significant

        print("has_significant distribution by pretraining condition (test ball only):")
        for pretraining_val in test_ball_significant_data[pretraining_col].unique():
            subset = test_ball_significant_data[test_ball_significant_data[pretraining_col] == pretraining_val]
            has_significant_count = subset["has_significant"].sum()
            total_count = len(subset)
            print(f"  {pretraining_val}: {has_significant_count}/{total_count} have significant events")

        # NEW: Detailed analysis of rows with has_significant=1 (test ball only)
        print(f"\nDetailed analysis of rows with has_significant=1 (test ball only):")

        # Filter for rows with has_significant=1 in test ball data
        significant_events_data = test_ball_significant_data[test_ball_significant_data["has_significant"] == 1]

        if len(significant_events_data) > 0:
            print(f"Found {len(significant_events_data)} test ball rows with has_significant=1")

            # Now get the corresponding first_significant_event data for these specific rows
            # We need to match by index or find a way to align the data
            print(f"\nBreakdown by pretraining condition for has_significant=1 rows (test ball):")
            for pretraining_val in significant_events_data[pretraining_col].unique():
                subset = significant_events_data[significant_events_data[pretraining_col] == pretraining_val]
                print(f"\n{pretraining_val} (n={len(subset)}):")
                print(f"  These flies have has_significant=1 for test ball condition")

                # Now let's see if we can find corresponding first_significant_event data
                # by looking at the same pretraining condition in our main test_ball_data
                matching_main_data = test_ball_data[test_ball_data[pretraining_col] == pretraining_val]

                print(f"  Corresponding data in main dataset:")
                print(f"    Total flies with {pretraining_val}: {len(matching_main_data)}")

                if len(matching_main_data) > 0:
                    # Check first_significant_event values
                    event_valid = matching_main_data[first_significant_event_col].notna().sum()
                    event_missing = matching_main_data[first_significant_event_col].isna().sum()
                    print(f"    {first_significant_event_col}: {event_valid} valid, {event_missing} missing")
                    if event_valid > 0:
                        event_sample = matching_main_data[first_significant_event_col].dropna().head(5)
                        print(f"      Sample values: {event_sample.tolist()}")

                    # Check first_significant_event_time values
                    time_valid = matching_main_data[first_significant_event_time_col].notna().sum()
                    time_missing = matching_main_data[first_significant_event_time_col].isna().sum()
                    print(f"    {first_significant_event_time_col}: {time_valid} valid, {time_missing} missing")
                    if time_valid > 0:
                        time_sample = matching_main_data[first_significant_event_time_col].dropna().head(5)
                        print(f"      Sample values: {time_sample.tolist()}")
        else:
            print("No rows found with has_significant=1 in test ball data")

    # Create subplots - 2x2 grid for pretraining and F1_condition
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Define color mappings and ordering
    pretraining_colors = {"n": "steelblue", "y": "orange"}
    f1_condition_colors = {"control": "steelblue", "pretrained": "orange", "pretrained_unlocked": "lightgreen"}

    pretraining_order = ["n", "y"]
    f1_condition_order = ["control", "pretrained_unlocked", "pretrained"]

    # Plot 1: first_significant_event by Pretraining condition (only for non-missing data)
    event_data = test_ball_data.dropna(subset=[first_significant_event_col])
    if not event_data.empty:
        create_boxplot_with_scatter(
            data=event_data,
            x_col=pretraining_col,
            y_col=first_significant_event_col,
            title="First Significant Event by Pretraining Condition",
            ax=ax1,
            color_mapping=pretraining_colors,
            order=pretraining_order,
        )
    else:
        ax1.text(
            0.5, 0.5, "No valid data for first_significant_event", ha="center", va="center", transform=ax1.transAxes
        )
        ax1.set_title("First Significant Event by Pretraining Condition")

    # Plot 2: first_significant_event by F1_condition
    if not event_data.empty:
        create_boxplot_with_scatter(
            data=event_data,
            x_col=f1_condition_col,
            y_col=first_significant_event_col,
            title="First Significant Event by F1 Condition",
            ax=ax2,
            color_mapping=f1_condition_colors,
            order=f1_condition_order,
        )
    else:
        ax2.text(
            0.5, 0.5, "No valid data for first_significant_event", ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_title("First Significant Event by F1 Condition")

    # Plot 2: first_significant_event_time by Pretraining condition
    # First, let's see what data we have before any filtering
    print(f"\nAnalyzing {first_significant_event_time_col} data:")
    print(f"Total test ball data: {len(test_ball_data)}")

    # Check the data types and values
    print(f"Data type of {first_significant_event_time_col}: {test_ball_data[first_significant_event_time_col].dtype}")

    # Show distribution by pretraining before filtering
    print(f"Distribution by {pretraining_col} before filtering:")
    for pretraining_val in test_ball_data[pretraining_col].unique():
        subset = test_ball_data[test_ball_data[pretraining_col] == pretraining_val]
        non_null_count = subset[first_significant_event_time_col].count()  # count() excludes NaN
        total_count = len(subset)
        print(f"  {pretraining_val}: {non_null_count}/{total_count} non-null values")

    # Instead of dropping, let's check if the values are actually valid numbers
    time_data = test_ball_data.copy()

    # Remove only truly invalid values (inf, -inf, but keep finite numbers including 0)
    mask = pd.isna(time_data[first_significant_event_time_col]) | np.isinf(time_data[first_significant_event_time_col])

    time_data_filtered = time_data[~mask]

    print(f"After removing NaN/inf: {len(time_data_filtered)} out of {len(test_ball_data)} remain")

    # Plot 3: first_significant_event_time by Pretraining condition
    if len(time_data_filtered) >= 2:  # Need at least 2 points for meaningful plot
        print(f"Creating plot for {first_significant_event_time_col} with {len(time_data_filtered)} data points")

        # Show final distribution
        print(f"Final distribution by {pretraining_col}:")
        for pretraining_val in time_data_filtered[pretraining_col].unique():
            count = len(time_data_filtered[time_data_filtered[pretraining_col] == pretraining_val])
            print(f"  {pretraining_val}: {count} data points")

        create_boxplot_with_scatter(
            data=time_data_filtered,
            x_col=pretraining_col,
            y_col=first_significant_event_time_col,
            title="First Significant Event Time by Pretraining Condition",
            ax=ax3,
            color_mapping=pretraining_colors,
            order=pretraining_order,
        )
    else:
        print(
            f"Insufficient valid data for {first_significant_event_time_col} plot (only {len(time_data_filtered)} valid points)"
        )
        ax3.text(
            0.5,
            0.5,
            f"Insufficient valid data\n({len(time_data_filtered)} valid points)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.set_title("First Significant Event Time by Pretraining Condition")
        if pretraining_col:
            ax3.set_xlabel(pretraining_col.replace("_", " ").title())
        ax3.set_ylabel("First Significant Event Time")

    # Plot 4: first_significant_event_time by F1_condition
    if len(time_data_filtered) >= 2:
        create_boxplot_with_scatter(
            data=time_data_filtered,
            x_col=f1_condition_col,
            y_col=first_significant_event_time_col,
            title="First Significant Event Time by F1 Condition",
            ax=ax4,
            color_mapping=f1_condition_colors,
            order=f1_condition_order,
        )
    else:
        ax4.text(
            0.5,
            0.5,
            f"Insufficient valid data\n({len(time_data_filtered)} valid points)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.set_title("First Significant Event Time by F1 Condition")
        if f1_condition_col:
            ax4.set_xlabel(f1_condition_col.replace("_", " ").title())
        ax4.set_ylabel("First Significant Event Time")

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "first_significant_event_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if pretraining_col and first_significant_event_col:
        print(f"\n1. First Significant Event by {pretraining_col}:")
        event_data = test_ball_data.dropna(subset=[first_significant_event_col])
        if not event_data.empty:
            event_summary = (
                event_data.groupby(pretraining_col)[first_significant_event_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
                .reindex(pretraining_order)
            )
            print(event_summary)
        else:
            print("No valid data for first_significant_event")

    if f1_condition_col and first_significant_event_col:
        print(f"\n2. First Significant Event by {f1_condition_col}:")
        event_data = test_ball_data.dropna(subset=[first_significant_event_col])
        if not event_data.empty:
            event_summary = (
                event_data.groupby(f1_condition_col)[first_significant_event_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
                .reindex(f1_condition_order)
            )
            print(event_summary)
        else:
            print("No valid data for first_significant_event")

    if pretraining_col and first_significant_event_time_col:
        print(f"\n3. First Significant Event Time by {pretraining_col}:")
        time_data = test_ball_data.dropna(subset=[first_significant_event_time_col])
        if not time_data.empty:
            time_summary = (
                time_data.groupby(pretraining_col)[first_significant_event_time_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
                .reindex(pretraining_order)
            )
            print(time_summary)
        else:
            print("No valid data for first_significant_event_time")

    if f1_condition_col and first_significant_event_time_col:
        print(f"\n4. First Significant Event Time by {f1_condition_col}:")
        time_data = test_ball_data.dropna(subset=[first_significant_event_time_col])
        if not time_data.empty:
            time_summary = (
                time_data.groupby(f1_condition_col)[first_significant_event_time_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
                .reindex(f1_condition_order)
            )
            print(time_summary)
        else:
            print("No valid data for first_significant_event_time")

    # Additional analysis: correlation between event and time (only for valid data)
    correlation_data = test_ball_data.dropna(subset=[first_significant_event_col, first_significant_event_time_col])
    if not correlation_data.empty and len(correlation_data) > 1:
        correlation, p_value = stats.pearsonr(
            correlation_data[first_significant_event_col], correlation_data[first_significant_event_time_col]
        )
        print(f"\nCorrelation between first significant event and time (n={len(correlation_data)}):")
        print(f"Pearson r = {correlation:.3f}, p = {p_value:.4f}")
    else:
        print(f"\nInsufficient valid data for correlation analysis (n={len(correlation_data)})")

    plt.show()

    return test_ball_data


if __name__ == "__main__":
    data = main()
