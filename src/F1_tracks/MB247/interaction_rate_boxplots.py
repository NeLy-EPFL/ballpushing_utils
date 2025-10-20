#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for interaction rate metrics for MB247 dataset.
1. overall_interaction_rate by Pretraining condition
2. overall_interaction_rate by Genotype
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
    """Create a boxplot with superimposed scatter plot for combined pretraining + genotype groups."""

    # Create combined group variable - sort by genotype first, then pretraining
    data_copy = data.copy()
    data_copy["combined_group"] = data_copy[genotype_col].astype(str) + " + " + data_copy[pretraining_col].astype(str)

    # Get unique groups and sort them
    unique_groups = sorted(data_copy["combined_group"].unique())

    # Create color list for seaborn
    if color_mapping:
        # Update color mapping keys to match new format (genotype + pretraining)
        updated_color_mapping = {}
        for group in unique_groups:
            genotype, pretraining = group.split(" + ")
            old_key = f"{pretraining} + {genotype}"
            updated_color_mapping[group] = color_mapping.get(old_key, "#999999")
        colors = [updated_color_mapping.get(group, "#999999") for group in unique_groups]
    else:
        colors = None

    # Create boxplot
    box_plot = sns.boxplot(data=data_copy, x="combined_group", y=y_col, ax=ax, palette=colors, order=unique_groups)

    # Apply brain region styling to boxplots
    if color_mapping:
        for i, (patch, group) in enumerate(zip(box_plot.patches, unique_groups)):
            genotype, pretraining = group.split(" + ")
            old_key = f"{pretraining} + {genotype}"
            brain_region_color = color_mapping.get(old_key, "#999999")

            if pretraining == "n":
                # White fill with colored edge for 'n' pretraining
                patch.set_facecolor("white")
                patch.set_edgecolor(brain_region_color)
                patch.set_linewidth(2)
            else:
                # Colored fill for 'y' pretraining
                patch.set_facecolor(brain_region_color)
                patch.set_edgecolor(brain_region_color)
                patch.set_alpha(0.7)

    # Style the whiskers, caps, and medians
    for i, group in enumerate(unique_groups):
        genotype, pretraining = group.split(" + ")
        old_key = f"{pretraining} + {genotype}"
        brain_region_color = color_mapping.get(old_key, "#999999") if color_mapping else "#000000"

        # Style whiskers and caps
        whiskers = box_plot.lines[6 * i : 6 * i + 2]  # Two whiskers per box
        caps = box_plot.lines[6 * i + 2 : 6 * i + 4]  # Two caps per box
        median = box_plot.lines[6 * i + 4 : 6 * i + 5]  # One median per box

        for whisker in whiskers:
            whisker.set_color(brain_region_color)
            whisker.set_linewidth(1.5)
        for cap in caps:
            cap.set_color(brain_region_color)
            cap.set_linewidth(1.5)
        for med in median:
            med.set_color(brain_region_color)
            med.set_linewidth(2)

    # Superimpose scatter plot with jitter - keep black and slightly bigger
    sns.stripplot(
        data=data_copy,
        x="combined_group",
        y=y_col,
        ax=ax,
        size=6,
        alpha=0.7,
        jitter=True,
        dodge=False,
        color="black",
        order=unique_groups,
    )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Genotype + Pretraining", fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    # Add sample size annotations
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        condition = label.get_text()
        n = len(data_copy[data_copy["combined_group"] == condition])
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")

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
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251014_08_summary_F1_TNT_MB247_Data/summary/pooled_summary.feather"
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
    genotype_col = None
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
        "interaction_rate": interaction_rate_col,
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
    print(f"  Interaction rate: {interaction_rate_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Genotype: {genotype_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    df_clean = df[[interaction_rate_col, pretraining_col, genotype_col, ball_condition_col]]

    # Only drop rows where grouping variables are missing
    df_clean = df_clean.dropna(subset=[pretraining_col, genotype_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Check available ball condition values
    print(f"Available {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Debug: Show some sample data to verify the values
    sample_data = df_clean[[interaction_rate_col, ball_condition_col, pretraining_col, genotype_col]].head(10)
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

    print(f"\nData distribution by {genotype_col}:")
    genotype_dist = test_ball_data[genotype_col].value_counts()
    print(genotype_dist)

    # Check if we have valid data for the metric by grouping conditions
    print(f"\nValid (non-missing) data by pretraining condition:")
    valid_data = test_ball_data[test_ball_data[interaction_rate_col].notna()]
    if not valid_data.empty:
        valid_dist = valid_data[pretraining_col].value_counts()
        print(valid_dist)
    else:
        print("  No valid data found!")

    print(f"\nValid (non-missing) data by genotype:")
    if not valid_data.empty:
        valid_dist = valid_data[genotype_col].value_counts()
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

    # Create single plot for combined pretraining + genotype analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create color mapping based on brain regions
    unique_genotypes = plot_data_filtered[genotype_col].unique()
    unique_pretraining = plot_data_filtered[pretraining_col].unique()
    color_mapping = create_genotype_color_mapping(unique_genotypes, unique_pretraining)

    print(f"Creating combined plot with {len(plot_data_filtered)} data points")

    # Show final distribution by combined groups
    plot_data_combined = plot_data_filtered.copy()
    plot_data_combined["combined_group"] = (
        plot_data_combined[genotype_col].astype(str) + " + " + plot_data_combined[pretraining_col].astype(str)
    )
    print(f"Final distribution by combined groups:")
    for combined_group in sorted(plot_data_combined["combined_group"].unique()):
        count = len(plot_data_combined[plot_data_combined["combined_group"] == combined_group])
        print(f"  {combined_group}: {count} data points")

    # Plot: interaction_rate by Genotype + Pretraining
    create_boxplot_with_scatter_combined(
        data=plot_data_filtered,
        pretraining_col=pretraining_col,
        genotype_col=genotype_col,
        y_col=interaction_rate_col,
        title="Overall Interaction Rate by Genotype + Pretraining",
        ax=ax,
        color_mapping=color_mapping,
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

    # Create combined grouping variable for analysis - using new format (genotype + pretraining)
    plot_data_combined = plot_data_filtered.copy()
    plot_data_combined["combined_group"] = (
        plot_data_combined[genotype_col].astype(str) + " + " + plot_data_combined[pretraining_col].astype(str)
    )

    if interaction_rate_col:
        print(f"\nInteraction Rate by Genotype + Pretraining:")
        rate_data = plot_data_combined.dropna(subset=[interaction_rate_col])
        if not rate_data.empty:
            rate_summary = (
                rate_data.groupby("combined_group")[interaction_rate_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(4)
            )
            print(rate_summary)
        else:
            print("No valid data for interaction_rate")

    # Statistical testing
    print("\n" + "=" * 60)
    print("STATISTICAL TESTING")
    print("=" * 60)

    from scipy.stats import mannwhitneyu, kruskal

    unique_groups = plot_data_combined["combined_group"].unique()

    if interaction_rate_col and len(unique_groups) >= 2:
        print(f"\nInteraction Rate - Combined Group Analysis:")
        rate_data = plot_data_combined.dropna(subset=[interaction_rate_col])

        if len(unique_groups) == 2:
            # Mann-Whitney U test for two groups
            group1_data = rate_data[rate_data["combined_group"] == unique_groups[0]][interaction_rate_col]
            group2_data = rate_data[rate_data["combined_group"] == unique_groups[1]][interaction_rate_col]

            if len(group1_data) > 0 and len(group2_data) > 0:
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
                print(f"Mann-Whitney U test: statistic={stat:.4f}, p-value={p_value:.4f}")
                print(f"Groups: {unique_groups[0]} (n={len(group1_data)}) vs {unique_groups[1]} (n={len(group2_data)})")

        elif len(unique_groups) > 2:
            # Kruskal-Wallis test for multiple groups
            group_data = [
                rate_data[rate_data["combined_group"] == group][interaction_rate_col].values
                for group in unique_groups
                if len(rate_data[rate_data["combined_group"] == group]) > 0
            ]

            if len(group_data) > 1 and all(len(g) > 0 for g in group_data):
                stat, p_value = kruskal(*group_data)
                print(f"Kruskal-Wallis test: statistic={stat:.4f}, p-value={p_value:.4f}")
                for i, group in enumerate(unique_groups):
                    n = len(rate_data[rate_data["combined_group"] == group])
                    print(f"  {group}: n={n}")

    plt.show()

    return plot_data_filtered


if __name__ == "__main__":
    data = main()
