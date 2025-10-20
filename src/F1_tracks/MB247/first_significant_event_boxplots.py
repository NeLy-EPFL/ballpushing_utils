#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for first significant event metrics for MB247 dataset.
1. first_significant_event by Pretraining condition
2. first_significant_event by Genotype
3. first_significant_event_time by Pretraining condition
4. first_significant_event_time by Genotype
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import sys

# Add the path to the PCA package to import Config
sys.path.append("/home/matthias/ballpushing_utils/src")

try:
    from PCA.Config import color_dict as brain_region_colors_dict

    print("✅ Loaded brain region mappings: {} genotypes".format(len(brain_region_colors_dict)))
except ImportError as e:
    print(f"❌ Could not import brain region mappings: {e}")
    # Fallback color palette
    brain_region_colors_dict = {"MB": "#1f77b4", "Control": "#7f7f7f"}


def get_brain_region_for_genotype(genotype):
    """
    Map genotype to brain region using manual mapping for this dataset.
    """
    # Manual mapping for the specific genotypes in this dataset
    genotype_to_brain_region = {"TNTxMB247": "MB", "TNTxEmptyGal4": "Control"}

    return genotype_to_brain_region.get(genotype, "Unknown")


def create_genotype_color_mapping(unique_genotypes, unique_pretraining):
    """
    Create color mapping for genotypes based on brain regions and pretraining conditions.
    """
    color_mapping = {}

    for genotype in unique_genotypes:
        brain_region = get_brain_region_for_genotype(genotype)
        base_color = brain_region_colors_dict.get(brain_region, "#999999")

        for pretraining in unique_pretraining:
            key = f"{pretraining} + {genotype}"
            color_mapping[key] = base_color

    return color_mapping


def create_boxplot_with_scatter_combined(data, pretraining_col, genotype_col, y_col, title, ax, color_mapping=None):
    """Create a boxplot with superimposed scatter plot for combined pretraining + genotype."""

    # Create combined grouping variable
    data_copy = data.copy()
    data_copy["combined_group"] = data_copy[pretraining_col].astype(str) + " + " + data_copy[genotype_col].astype(str)

    # Create boxplot
    box_plot = sns.boxplot(data=data_copy, x="combined_group", y=y_col, ax=ax, palette=color_palette)

    # Make boxplot transparent
    for patch in box_plot.patches:
        patch.set_alpha(0.7)

    # Superimpose scatter plot with jitter
    sns.stripplot(
        data=data_copy, x="combined_group", y=y_col, ax=ax, size=4, alpha=0.6, jitter=True, dodge=False, color="black"
    )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Pretraining + Genotype", fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    # Add sample size annotations
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        condition = label.get_text()
        n = len(data_copy[data_copy["combined_group"] == condition])
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    # Perform statistical tests if there are multiple groups
    unique_conditions = data_copy["combined_group"].unique()
    if len(unique_conditions) >= 2:
        # Perform pairwise comparisons
        groups = [data_copy[data_copy["combined_group"] == condition][y_col] for condition in unique_conditions]

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

    # Dataset path - MB247 dataset
    dataset_path = "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251014_15_F1_coordinates_F1_TNT_MB247_Data/summary/pooled_summary.feather"

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
    genotype_col = None
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

    # Look for Genotype column
    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
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
        "genotype": genotype_col,
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
    print(f"  Genotype: {genotype_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    # Don't drop rows where first_significant_event cols are NaN yet - we'll handle this after ball filtering
    df_clean = df[
        [
            first_significant_event_col,
            first_significant_event_time_col,
            pretraining_col,
            genotype_col,
            ball_condition_col,
        ]
    ]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, genotype_col, ball_condition_col])

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
            genotype_col,
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
    print(f"Unique {genotype_col} values: {test_ball_data[genotype_col].unique()}")

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

    # Show distribution by grouping conditions
    print(f"\nData distribution by {pretraining_col}:")
    pretraining_dist = test_ball_data[pretraining_col].value_counts()
    print(pretraining_dist)

    print(f"\nData distribution by {genotype_col}:")
    genotype_dist = test_ball_data[genotype_col].value_counts()
    print(genotype_dist)

    # Check if we have valid data for each metric by grouping condition
    print(f"\nValid (non-missing) data by pretraining condition:")
    for metric in [first_significant_event_col, first_significant_event_time_col]:
        print(f"\n{metric}:")
        valid_data = test_ball_data[test_ball_data[metric].notna()]
        if not valid_data.empty:
            valid_dist = valid_data[pretraining_col].value_counts()
            print(valid_dist)
        else:
            print("  No valid data found!")

    print(f"\nValid (non-missing) data by genotype:")
    for metric in [first_significant_event_col, first_significant_event_time_col]:
        print(f"\n{metric}:")
        valid_data = test_ball_data[test_ball_data[metric].notna()]
        if not valid_data.empty:
            valid_dist = valid_data[genotype_col].value_counts()
            print(valid_dist)
        else:
            print("  No valid data found!")

    # Create subplots - 1x2 grid for event and time by combined pretraining + genotype
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Set up color palette for combined groups
    test_ball_data_copy = test_ball_data.copy()
    test_ball_data_copy["combined_group"] = (
        test_ball_data_copy[pretraining_col].astype(str) + " + " + test_ball_data_copy[genotype_col].astype(str)
    )
    n_combined_groups = len(test_ball_data_copy["combined_group"].unique())
    combined_palette = sns.color_palette("Set1", n_colors=n_combined_groups)

    # Plot 1: first_significant_event by Pretraining + Genotype (only for non-missing data)
    event_data = test_ball_data.dropna(subset=[first_significant_event_col])
    if not event_data.empty:
        create_boxplot_with_scatter_combined(
            data=event_data,
            pretraining_col=pretraining_col,
            genotype_col=genotype_col,
            y_col=first_significant_event_col,
            title="First Significant Event by Pretraining + Genotype",
            ax=ax1,
            color_palette=combined_palette,
        )
    else:
        ax1.text(
            0.5, 0.5, "No valid data for first_significant_event", ha="center", va="center", transform=ax1.transAxes
        )
        ax1.set_title("First Significant Event by Pretraining + Genotype")

    # Prepare time data - remove invalid values
    time_data = test_ball_data.copy()
    mask = pd.isna(time_data[first_significant_event_time_col]) | np.isinf(time_data[first_significant_event_time_col])
    time_data_filtered = time_data[~mask]

    print(f"\nAfter removing NaN/inf: {len(time_data_filtered)} out of {len(test_ball_data)} remain")

    # Plot 2: first_significant_event_time by Pretraining + Genotype
    if len(time_data_filtered) >= 2:  # Need at least 2 points for meaningful plot
        print(f"Creating plot for {first_significant_event_time_col} with {len(time_data_filtered)} data points")

        create_boxplot_with_scatter_combined(
            data=time_data_filtered,
            pretraining_col=pretraining_col,
            genotype_col=genotype_col,
            y_col=first_significant_event_time_col,
            title="First Significant Event Time by Pretraining + Genotype",
            ax=ax2,
            color_palette=combined_palette,
        )
    else:
        print(
            f"Insufficient valid data for {first_significant_event_time_col} plot (only {len(time_data_filtered)} valid points)"
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
        ax2.set_title("First Significant Event Time by Pretraining + Genotype")
        ax2.set_xlabel("Pretraining + Genotype")
        ax2.set_ylabel("First Significant Event Time")

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

    # Create combined grouping variable for analysis
    test_ball_data_combined = test_ball_data.copy()
    test_ball_data_combined["combined_group"] = (
        test_ball_data_combined[pretraining_col].astype(str) + " + " + test_ball_data_combined[genotype_col].astype(str)
    )

    if first_significant_event_col:
        print(f"\n1. First Significant Event by Pretraining + Genotype:")
        event_data = test_ball_data_combined.dropna(subset=[first_significant_event_col])
        if not event_data.empty:
            event_summary = (
                event_data.groupby("combined_group")[first_significant_event_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(event_summary)
        else:
            print("No valid data for first_significant_event")

    if first_significant_event_time_col:
        print(f"\n2. First Significant Event Time by Pretraining + Genotype:")
        time_data = test_ball_data_combined.dropna(subset=[first_significant_event_time_col])
        if not time_data.empty:
            time_summary = (
                time_data.groupby("combined_group")[first_significant_event_time_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(time_summary)
        else:
            print("No valid data for first_significant_event_time")

    # Statistical testing
    print("\n" + "=" * 60)
    print("STATISTICAL TESTING")
    print("=" * 60)

    from scipy.stats import mannwhitneyu, kruskal

    unique_groups = test_ball_data_combined["combined_group"].unique()

    if first_significant_event_col and len(unique_groups) >= 2:
        print(f"\nFirst Significant Event - Combined Group Analysis:")
        event_data = test_ball_data_combined.dropna(subset=[first_significant_event_col])

        if len(unique_groups) == 2:
            # Mann-Whitney U test for two groups
            group1_data = event_data[event_data["combined_group"] == unique_groups[0]][first_significant_event_col]
            group2_data = event_data[event_data["combined_group"] == unique_groups[1]][first_significant_event_col]

            if len(group1_data) > 0 and len(group2_data) > 0:
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
                print(f"Mann-Whitney U test: statistic={stat:.4f}, p-value={p_value:.4f}")
                print(f"Groups: {unique_groups[0]} (n={len(group1_data)}) vs {unique_groups[1]} (n={len(group2_data)})")

        elif len(unique_groups) > 2:
            # Kruskal-Wallis test for multiple groups
            group_data = [
                event_data[event_data["combined_group"] == group][first_significant_event_col].values
                for group in unique_groups
                if len(event_data[event_data["combined_group"] == group]) > 0
            ]

            if len(group_data) > 1 and all(len(g) > 0 for g in group_data):
                stat, p_value = kruskal(*group_data)
                print(f"Kruskal-Wallis test: statistic={stat:.4f}, p-value={p_value:.4f}")
                for i, group in enumerate(unique_groups):
                    n = len(event_data[event_data["combined_group"] == group])
                    print(f"  {group}: n={n}")

    if first_significant_event_time_col and len(unique_groups) >= 2:
        print(f"\nFirst Significant Event Time - Combined Group Analysis:")
        time_data = test_ball_data_combined.dropna(subset=[first_significant_event_time_col])

        if len(unique_groups) == 2:
            # Mann-Whitney U test for two groups
            group1_data = time_data[time_data["combined_group"] == unique_groups[0]][first_significant_event_time_col]
            group2_data = time_data[time_data["combined_group"] == unique_groups[1]][first_significant_event_time_col]

            if len(group1_data) > 0 and len(group2_data) > 0:
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
                print(f"Mann-Whitney U test: statistic={stat:.4f}, p-value={p_value:.4f}")
                print(f"Groups: {unique_groups[0]} (n={len(group1_data)}) vs {unique_groups[1]} (n={len(group2_data)})")

        elif len(unique_groups) > 2:
            # Kruskal-Wallis test for multiple groups
            group_data = [
                time_data[time_data["combined_group"] == group][first_significant_event_time_col].values
                for group in unique_groups
                if len(time_data[time_data["combined_group"] == group]) > 0
            ]

            if len(group_data) > 1 and all(len(g) > 0 for g in group_data):
                stat, p_value = kruskal(*group_data)
                print(f"Kruskal-Wallis test: statistic={stat:.4f}, p-value={p_value:.4f}")
                for i, group in enumerate(unique_groups):
                    n = len(time_data[time_data["combined_group"] == group])
                    print(f"  {group}: n={n}")

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
