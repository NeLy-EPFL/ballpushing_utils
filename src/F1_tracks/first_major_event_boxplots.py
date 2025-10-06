#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for first major event metrics.
1. first_major_event by Pretraining condition
2. first_major_event_time by Pretraining condition
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
    """Main function to create the boxplots with scatter plots."""

    # Dataset path
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251003_15_summary_F1_New_Data/summary/pooled_summary.feather"
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
    first_major_event_col = None
    first_major_event_time_col = None
    pretraining_col = None
    ball_condition_col = None

    # Look for first_major_event columns
    if "first_major_event" in df.columns:
        first_major_event_col = "first_major_event"

    if "first_major_event_time" in df.columns:
        first_major_event_time_col = "first_major_event_time"

    # If exact matches not found, look for similar columns
    if first_major_event_col is None:
        for col in df.columns:
            if (
                "first" in col.lower()
                and "major" in col.lower()
                and "event" in col.lower()
                and "time" not in col.lower()
            ):
                first_major_event_col = col
                break

    if first_major_event_time_col is None:
        for col in df.columns:
            if "first" in col.lower() and "major" in col.lower() and "event" in col.lower() and "time" in col.lower():
                first_major_event_time_col = col
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
        "first_major_event": first_major_event_col,
        "first_major_event_time": first_major_event_time_col,
        "pretraining": pretraining_col,
        "ball_condition": ball_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  First major event: {first_major_event_col}")
    print(f"  First major event time: {first_major_event_time_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    # Don't drop rows where first_major_event cols are NaN yet - we'll handle this after ball filtering
    df_clean = df[[first_major_event_col, first_major_event_time_col, pretraining_col, ball_condition_col]]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the values
    sample_data = df_clean[
        [first_major_event_col, first_major_event_time_col, ball_condition_col, pretraining_col]
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

    # Check for missing values in the metrics
    print(f"\nMissing values analysis:")
    print(
        f"  {first_major_event_col}: {test_ball_data[first_major_event_col].isna().sum()} missing out of {len(test_ball_data)}"
    )
    print(
        f"  {first_major_event_time_col}: {test_ball_data[first_major_event_time_col].isna().sum()} missing out of {len(test_ball_data)}"
    )

    # Show actual values for debugging
    print(f"\nSample of actual values:")
    print(f"{first_major_event_col} (first 10 non-null values):")
    event_sample = test_ball_data[first_major_event_col].dropna().head(10)
    print(event_sample.tolist() if not event_sample.empty else "No non-null values found")

    print(f"\n{first_major_event_time_col} (first 10 non-null values):")
    time_sample = test_ball_data[first_major_event_time_col].dropna().head(10)
    print(time_sample.tolist() if not time_sample.empty else "No non-null values found")

    # Check unique values to see if there are unexpected values
    print(f"\nUnique values in {first_major_event_time_col}:")
    unique_time_vals = test_ball_data[first_major_event_time_col].unique()
    print(f"Found {len(unique_time_vals)} unique values: {unique_time_vals[:20]}...")  # Show first 20

    # Show distribution by pretraining condition
    print(f"\nData distribution by {pretraining_col}:")
    pretraining_dist = test_ball_data[pretraining_col].value_counts()
    print(pretraining_dist)

    # Check if we have valid data for each metric by pretraining condition
    print(f"\nValid (non-missing) data by pretraining condition:")
    for metric in [first_major_event_col, first_major_event_time_col]:
        print(f"\n{metric}:")
        valid_data = test_ball_data[test_ball_data[metric].notna()]
        if not valid_data.empty:
            valid_dist = valid_data[pretraining_col].value_counts()
            print(valid_dist)
        else:
            print("  No valid data found!")

    # Also load has_major to cross-check
    if "has_major" in df.columns:
        print(f"\nCross-check with has_major binary metric:")

        # Make sure we get has_major for the same test ball filtered data
        # First get has_major for the same ball condition filtering
        df_with_major = df[[pretraining_col, ball_condition_col, "has_major"]].dropna(
            subset=[pretraining_col, ball_condition_col]
        )

        # Apply the same test ball filtering as we did for the main data
        test_ball_major_data = pd.DataFrame()
        for test_val in test_ball_values:
            test_subset = df_with_major[df_with_major[ball_condition_col] == test_val]
            if not test_subset.empty:
                test_ball_major_data = test_subset
                print(f"Found test ball has_major data using value: '{test_val}'")
                break

        if test_ball_major_data.empty:
            print("No specific 'test' ball data found for has_major. Using all data.")
            test_ball_major_data = df_with_major

        print("has_major distribution by pretraining condition (test ball only):")
        for pretraining_val in test_ball_major_data[pretraining_col].unique():
            subset = test_ball_major_data[test_ball_major_data[pretraining_col] == pretraining_val]
            has_major_count = subset["has_major"].sum()
            total_count = len(subset)
            print(f"  {pretraining_val}: {has_major_count}/{total_count} have major events")

        # NEW: Detailed analysis of rows with has_major=1 (test ball only)
        print(f"\nDetailed analysis of rows with has_major=1 (test ball only):")

        # Filter for rows with has_major=1 in test ball data
        major_events_data = test_ball_major_data[test_ball_major_data["has_major"] == 1]

        if len(major_events_data) > 0:
            print(f"Found {len(major_events_data)} test ball rows with has_major=1")

            # Now get the corresponding first_major_event data for these specific rows
            # We need to match by index or find a way to align the data
            print(f"\nBreakdown by pretraining condition for has_major=1 rows (test ball):")
            for pretraining_val in major_events_data[pretraining_col].unique():
                subset = major_events_data[major_events_data[pretraining_col] == pretraining_val]
                print(f"\n{pretraining_val} (n={len(subset)}):")
                print(f"  These flies have has_major=1 for test ball condition")

                # Now let's see if we can find corresponding first_major_event data
                # by looking at the same pretraining condition in our main test_ball_data
                matching_main_data = test_ball_data[test_ball_data[pretraining_col] == pretraining_val]

                print(f"  Corresponding data in main dataset:")
                print(f"    Total flies with {pretraining_val}: {len(matching_main_data)}")

                if len(matching_main_data) > 0:
                    # Check first_major_event values
                    event_valid = matching_main_data[first_major_event_col].notna().sum()
                    event_missing = matching_main_data[first_major_event_col].isna().sum()
                    print(f"    {first_major_event_col}: {event_valid} valid, {event_missing} missing")
                    if event_valid > 0:
                        event_sample = matching_main_data[first_major_event_col].dropna().head(5)
                        print(f"      Sample values: {event_sample.tolist()}")

                    # Check first_major_event_time values
                    time_valid = matching_main_data[first_major_event_time_col].notna().sum()
                    time_missing = matching_main_data[first_major_event_time_col].isna().sum()
                    print(f"    {first_major_event_time_col}: {time_valid} valid, {time_missing} missing")
                    if time_valid > 0:
                        time_sample = matching_main_data[first_major_event_time_col].dropna().head(5)
                        print(f"      Sample values: {time_sample.tolist()}")
        else:
            print("No rows found with has_major=1 in test ball data")  # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set up color palettes
    pretraining_palette = sns.color_palette("Set2", n_colors=len(test_ball_data[pretraining_col].unique()))

    # Plot 1: first_major_event by Pretraining condition (only for non-missing data)
    event_data = test_ball_data.dropna(subset=[first_major_event_col])
    if not event_data.empty:
        create_boxplot_with_scatter(
            data=event_data,
            x_col=pretraining_col,
            y_col=first_major_event_col,
            title="First Major Event by Pretraining Condition",
            ax=ax1,
            color_palette=pretraining_palette,
        )
    else:
        ax1.text(0.5, 0.5, "No valid data for first_major_event", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("First Major Event by Pretraining Condition")

    # Plot 2: first_major_event_time by Pretraining condition
    # First, let's see what data we have before any filtering
    print(f"\nAnalyzing {first_major_event_time_col} data:")
    print(f"Total test ball data: {len(test_ball_data)}")

    # Check the data types and values
    print(f"Data type of {first_major_event_time_col}: {test_ball_data[first_major_event_time_col].dtype}")

    # Show distribution by pretraining before filtering
    print(f"Distribution by {pretraining_col} before filtering:")
    for pretraining_val in test_ball_data[pretraining_col].unique():
        subset = test_ball_data[test_ball_data[pretraining_col] == pretraining_val]
        non_null_count = subset[first_major_event_time_col].count()  # count() excludes NaN
        total_count = len(subset)
        print(f"  {pretraining_val}: {non_null_count}/{total_count} non-null values")

    # Instead of dropping, let's check if the values are actually valid numbers
    time_data = test_ball_data.copy()

    # Remove only truly invalid values (inf, -inf, but keep finite numbers including 0)
    mask = pd.isna(time_data[first_major_event_time_col]) | np.isinf(time_data[first_major_event_time_col])

    time_data_filtered = time_data[~mask]

    print(f"After removing NaN/inf: {len(time_data_filtered)} out of {len(test_ball_data)} remain")

    if len(time_data_filtered) >= 2:  # Need at least 2 points for meaningful plot
        print(f"Creating plot for {first_major_event_time_col} with {len(time_data_filtered)} data points")

        # Show final distribution
        print(f"Final distribution by {pretraining_col}:")
        for pretraining_val in time_data_filtered[pretraining_col].unique():
            count = len(time_data_filtered[time_data_filtered[pretraining_col] == pretraining_val])
            print(f"  {pretraining_val}: {count} data points")

        create_boxplot_with_scatter(
            data=time_data_filtered,
            x_col=pretraining_col,
            y_col=first_major_event_time_col,
            title="First Major Event Time by Pretraining Condition",
            ax=ax2,
            color_palette=pretraining_palette,
        )
    else:
        print(
            f"Insufficient valid data for {first_major_event_time_col} plot (only {len(time_data_filtered)} valid points)"
        )
        ax2.text(
            0.5,
            0.5,
            f"Insufficient valid data\n({len(time_data_filtered)} valid points)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("First Major Event Time by Pretraining Condition")
        if pretraining_col:
            ax2.set_xlabel(pretraining_col.replace("_", " ").title())
        ax2.set_ylabel("First Major Event Time")

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "first_major_event_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if pretraining_col and first_major_event_col:
        print(f"\n1. First Major Event by {pretraining_col}:")
        event_data = test_ball_data.dropna(subset=[first_major_event_col])
        if not event_data.empty:
            event_summary = (
                event_data.groupby(pretraining_col)[first_major_event_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(event_summary)
        else:
            print("No valid data for first_major_event")

    if pretraining_col and first_major_event_time_col:
        print(f"\n2. First Major Event Time by {pretraining_col}:")
        time_data = test_ball_data.dropna(subset=[first_major_event_time_col])
        if not time_data.empty:
            time_summary = (
                time_data.groupby(pretraining_col)[first_major_event_time_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(time_summary)
        else:
            print("No valid data for first_major_event_time")

    # Additional analysis: correlation between event and time (only for valid data)
    correlation_data = test_ball_data.dropna(subset=[first_major_event_col, first_major_event_time_col])
    if not correlation_data.empty and len(correlation_data) > 1:
        correlation, p_value = stats.pearsonr(
            correlation_data[first_major_event_col], correlation_data[first_major_event_time_col]
        )
        print(f"\nCorrelation between first major event and time (n={len(correlation_data)}):")
        print(f"Pearson r = {correlation:.3f}, p = {p_value:.4f}")
    else:
        print(f"\nInsufficient valid data for correlation analysis (n={len(correlation_data)})")

    plt.show()

    return test_ball_data


if __name__ == "__main__":
    data = main()
