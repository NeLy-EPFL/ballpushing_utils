#!/usr/bin/env python3
"""
Script to create boxplots with superimposed scatter plots for pulled metric for TNT LC10-2 dataset.
Analyzes pulled (count of pull events) by Genotype and Pretraining condition.
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
    brain_region_colors_dict = {"Vision": "#2ca02c", "Control": "#7f7f7f"}


def get_brain_region_for_genotype(genotype):
    """
    Map genotype to brain region using manual mapping for this dataset.
    """
    # Manual mapping for the specific genotypes in this dataset
    genotype_to_brain_region = {
        "TNTxLC10-2": "Vision",
        "TNTxEmptyGal4": "Control",
        "TNTxEmptySplit": "Control",
    }
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
            # Mann-Whitney U test for two groups (better for count data)
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

    # Dataset path - LC10-2 TNT dataset
    dataset_path = "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251015_10_F1_coordinates_F1_TNT_LC10-2_Data/summary/pooled_summary.feather"

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

    # Try to identify the correct column names
    pulled_col = "pulled" if "pulled" in df.columns else None
    pretraining_col = None
    genotype_col = None
    ball_condition_col = None

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

    # Look for ball condition column
    if "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"
    elif "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"

    # Check if we found all required columns
    required_columns = {
        "pulled": pulled_col,
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
    print(f"  Pulled: {pulled_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Genotype: {genotype_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data and filter for test ball only
    df_clean = df[[pulled_col, pretraining_col, genotype_col, ball_condition_col]]
    df_clean = df_clean.dropna(subset=[pretraining_col, genotype_col, ball_condition_col])

    print(f"Data shape after removing missing grouping variables: {df_clean.shape}")

    # Filter for test ball
    test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
    test_ball_data = pd.DataFrame()

    for test_val in test_ball_values:
        test_subset = df_clean[df_clean[ball_condition_col] == test_val]
        if not test_subset.empty:
            test_ball_data = test_subset
            print(f"Found test ball data using value: '{test_val}'")
            break

    if test_ball_data.empty:
        print("No specific 'test' ball data found. Using all data.")
        test_ball_data = df_clean

    print(f"Data shape for analysis: {test_ball_data.shape}")
    print(f"Unique {pretraining_col} values: {test_ball_data[pretraining_col].unique()}")
    print(f"Unique {genotype_col} values: {test_ball_data[genotype_col].unique()}")

    # Check value range for count data interpretation
    print(f"\nValue range for {pulled_col}:")
    pulled_values = test_ball_data[pulled_col].dropna()
    if not pulled_values.empty:
        print(f"Min: {pulled_values.min():.0f}, Max: {pulled_values.max():.0f}")
        print(f"Mean: {pulled_values.mean():.3f}, Std: {pulled_values.std():.3f}")
        print(f"Median: {pulled_values.median():.0f}")

        # Count distribution
        print(f"Value counts for pulled events:")
        value_counts = pulled_values.value_counts().sort_index()
        for val, count in value_counts.head(10).items():  # Show first 10 values
            print(f"  {val:.0f} pulls: {count} flies ({100*count/len(pulled_values):.1f}%)")

    # Prepare data for plotting
    plot_data = test_ball_data.copy()
    mask = pd.isna(plot_data[pulled_col]) | np.isinf(plot_data[pulled_col])
    plot_data_filtered = plot_data[~mask]

    print(f"After removing NaN/inf: {len(plot_data_filtered)} out of {len(test_ball_data)} remain")

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

    # Plot: pulled by Genotype + Pretraining
    create_boxplot_with_scatter_combined(
        data=plot_data_filtered,
        pretraining_col=pretraining_col,
        genotype_col=genotype_col,
        y_col=pulled_col,
        title="Pulled Events (Count) by Genotype + Pretraining (TNT LC10-2)",
        ax=ax,
        color_mapping=color_mapping,
    )

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "pulled_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Create combined grouping variable for analysis
    plot_data_combined = plot_data_filtered.copy()
    plot_data_combined["combined_group"] = (
        plot_data_combined[genotype_col].astype(str) + " + " + plot_data_combined[pretraining_col].astype(str)
    )

    if pulled_col:
        print(f"\nPulled Events by Genotype + Pretraining:")
        pulled_data = plot_data_combined.dropna(subset=[pulled_col])
        if not pulled_data.empty:
            pulled_summary = (
                pulled_data.groupby("combined_group")[pulled_col]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .round(3)
            )
            print(pulled_summary)

            # Additional analysis: pulling behavior interpretation
            print(f"\nPulling Behavior Analysis:")
            for group in pulled_data["combined_group"].unique():
                subset = pulled_data[pulled_data["combined_group"] == group][pulled_col]
                total_flies = len(subset)
                flies_with_pulls = (subset > 0).sum()
                flies_no_pulls = (subset == 0).sum()
                print(f"  {group}:")
                print(
                    f"    Flies with pulls: {flies_with_pulls}/{total_flies} ({100*flies_with_pulls/total_flies:.1f}%)"
                )
                print(
                    f"    Flies with no pulls: {flies_no_pulls}/{total_flies} ({100*flies_no_pulls/total_flies:.1f}%)"
                )
                if flies_with_pulls > 0:
                    mean_among_pullers = subset[subset > 0].mean()
                    print(f"    Mean pulls among active pullers: {mean_among_pullers:.2f}")
        else:
            print("No valid data for pulled")

    plt.show()
    return plot_data_filtered


if __name__ == "__main__":
    data = main()
