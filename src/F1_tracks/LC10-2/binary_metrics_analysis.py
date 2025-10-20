#!/usr/bin/env python3
"""
Script to analyze binary metrics for MB247 dataset.
Creates comprehensive plots for has_finished, has_long_pauses, has_major, and has_significant
by Pretraining and Genotype.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from matplotlib.patches import Rectangle

# Add path to import Config for brain region mappings
sys.path.append("/home/matthias/ballpushing_utils/src")
try:
    from PCA import Config

    # Load brain region mappings
    nickname_to_brainregion = dict(zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"]))
    brain_region_color_dict = Config.color_dict
    HAS_BRAIN_REGIONS = True
    print(f"✅ Loaded brain region mappings: {len(nickname_to_brainregion)} genotypes")
except Exception as e:
    print(f"⚠️  Could not load brain region mappings from Config: {e}")
    nickname_to_brainregion = {}
    brain_region_color_dict = {}
    HAS_BRAIN_REGIONS = False


def get_brain_region_for_genotype(genotype):
    """Get brain region for a genotype, with manual mapping for MB247 dataset."""

    # Manual mapping for MB247 dataset genotypes
    manual_mapping = {
        "TNTxMB247": "MB",
        "TNTxEmptyGal4": "Control",
        "TNTxEmptySplit": "Control",
        "TNTxLC10-2": "Vision",
    }

    if genotype in manual_mapping:
        return manual_mapping[genotype]

    if not HAS_BRAIN_REGIONS:
        return "Unknown"

    # Direct lookup from Config
    if genotype in nickname_to_brainregion:
        return nickname_to_brainregion[genotype]

    # Try common variations
    variations = [
        genotype.replace("TNTx", ""),  # Remove TNTx prefix
        genotype.replace("x", ""),  # Remove x's
        genotype.replace("-", ""),  # Remove hyphens
        genotype.split("x")[-1] if "x" in genotype else genotype,  # Take last part after x
    ]

    for variation in variations:
        if variation in nickname_to_brainregion:
            return nickname_to_brainregion[variation]

    return "Unknown"


def get_color_for_brain_region(brain_region):
    """Get color for a brain region."""
    if not HAS_BRAIN_REGIONS or brain_region not in brain_region_color_dict:
        # Fallback colors if brain region mapping not available
        fallback_colors = {
            "Unknown": "#808080",  # Gray
            "MB": "#1f77b4",  # Blue
            "Control": "#ff7f0e",  # Orange
        }
        return fallback_colors.get(brain_region, "#808080")

    return brain_region_color_dict[brain_region]


def create_genotype_color_mapping(genotypes):
    """Create mapping of genotype -> color based on brain region."""
    genotype_to_color = {}

    for genotype in genotypes:
        brain_region = get_brain_region_for_genotype(genotype)
        color = get_color_for_brain_region(brain_region)
        genotype_to_color[genotype] = color

    return genotype_to_color


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


def create_binary_barplot_combined(data, pretraining_col, genotype_col, binary_metrics, title_prefix, ax_row, fig):
    """Create bar plots for binary metrics showing both pretraining and genotype."""

    n_metrics = len(binary_metrics)

    for i, metric in enumerate(binary_metrics):
        ax = ax_row[i]

        # Create combined grouping variable
        data_copy = data.copy()
        data_copy["combined_group"] = (
            data_copy[pretraining_col].astype(str) + " + " + data_copy[genotype_col].astype(str)
        )

        # Calculate proportions and statistics
        prop_data, p_value, test_name = calculate_proportions_and_stats(data_copy, "combined_group", metric)

        # Create bar plot with initial settings
        bars = ax.bar(range(len(prop_data)), prop_data["proportion"], alpha=1.0, linewidth=2)

        # Color bars based on brain region, with different fill styles for pretraining
        genotype_vals = [group.split(" + ")[1] for group in prop_data["combined_group"]]
        pretraining_vals = [group.split(" + ")[0] for group in prop_data["combined_group"]]

        # Create genotype to color mapping based on brain regions
        unique_genotypes = list(set(genotype_vals))
        genotype_color_map = create_genotype_color_mapping(unique_genotypes)

        for bar, group in zip(bars, prop_data["combined_group"]):
            pretraining_val = group.split(" + ")[0]
            genotype_val = group.split(" + ")[1]

            brain_region_color = genotype_color_map[genotype_val]

            if pretraining_val == "n":
                # No pretraining: white fill with colored outline
                bar.set_facecolor("white")
                bar.set_edgecolor(brain_region_color)
            else:
                # Pretraining: colored fill with same colored outline
                bar.set_facecolor(brain_region_color)
                bar.set_edgecolor(brain_region_color)

        # Add sample size and proportion labels on bars
        for j, (idx, row) in enumerate(prop_data.iterrows()):
            height = row["proportion"]
            ax.text(
                j,
                height + 0.01,
                f"{row['n_positive']}/{row['n_total']}\n({height:.2%})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Formatting
        ax.set_title(f"{title_prefix}\n{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Pretraining + Genotype", fontsize=10)
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, 1.1)

        # Set x-axis labels
        ax.set_xticks(range(len(prop_data)))
        ax.set_xticklabels(prop_data["combined_group"], rotation=45, ha="right")

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

        # Add legend for brain regions and pretraining patterns
        if i == 0:  # Only add legend to first subplot
            # Create legend elements for brain regions and pretraining patterns
            legend_elements = []

            # Brain region colors with different fill styles
            for genotype in unique_genotypes:
                brain_region = get_brain_region_for_genotype(genotype)
                color = genotype_color_map[genotype]

                # Colored fill for pretraining ('y')
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, label=f"{brain_region} (pretrained)")
                )
                # White fill with colored outline for no pretraining ('n')
                legend_elements.append(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=color,
                        linewidth=2,
                        label=f"{brain_region} (no pretraining)",
                    )
                )

            ax.legend(handles=legend_elements, title="Brain Region + Pretraining", loc="upper right", fontsize=8)


def create_heatmap_combined(data, pretraining_col, genotype_col, binary_metrics, title, ax):
    """Create a heatmap showing proportions of binary metrics by combined pretraining and genotype."""

    # Create combined grouping variable
    data_copy = data.copy()
    data_copy["combined_group"] = data_copy[pretraining_col].astype(str) + " + " + data_copy[genotype_col].astype(str)

    # Calculate proportions for all metrics
    heatmap_data = []
    for metric in binary_metrics:
        prop_data, _, _ = calculate_proportions_and_stats(data_copy, "combined_group", metric)
        proportions = dict(zip(prop_data["combined_group"], prop_data["proportion"]))
        heatmap_data.append(proportions)

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, index=[m.replace("_", " ").title() for m in binary_metrics])

    # Create heatmap
    sns.heatmap(heatmap_df, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Proportion"})

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Pretraining + Genotype", fontsize=12)
    ax.set_ylabel("Binary Metrics", fontsize=12)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)


def main():
    """Main function to create the binary metrics analysis."""

    # Dataset path - MB247 dataset
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

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Define binary metrics to analyze
    binary_metrics = ["has_finished", "has_long_pauses", "has_major", "has_significant"]

    # Try to identify the correct column names
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
    if genotype_col is None:
        print("Could not find Genotype column")
        return
    if ball_condition_col is None:
        print("Could not find ball_condition column")
        return

    print(f"Using columns:")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Genotype: {genotype_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Select relevant columns and clean data
    analysis_cols = [pretraining_col, genotype_col, ball_condition_col] + available_metrics
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
    print(f"Unique {genotype_col} values: {df_clean[genotype_col].unique()}")

    # Verify binary nature of metrics
    for metric in available_metrics:
        unique_vals = df_clean[metric].unique()
        print(f"{metric} unique values: {unique_vals}")
        if not set(unique_vals).issubset({0, 1, True, False}):
            print(f"Warning: {metric} may not be binary!")

    # Create comprehensive figure
    n_metrics = len(available_metrics)
    fig = plt.figure(figsize=(20, 12))

    # Create subplots: 1 row of bar plots + 1 row of heatmap
    gs = fig.add_gridspec(2, max(n_metrics, 1), height_ratios=[1, 0.6], hspace=0.3, wspace=0.3)

    # Row 1: Bar plots by combined Pretraining + Genotype
    ax_combined = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    create_binary_barplot_combined(
        df_clean,
        pretraining_col,
        genotype_col,
        available_metrics,
        "Binary Metrics by Pretraining + Genotype",
        ax_combined,
        fig,
    )

    # Row 2: Heatmap
    ax_heatmap = fig.add_subplot(gs[1, :])  # Span all columns
    create_heatmap_combined(
        df_clean,
        pretraining_col,
        genotype_col,
        available_metrics,
        "Test Ball Binary Metrics - Pretraining + Genotype",
        ax_heatmap,
    )

    # Overall title
    fig.suptitle("MB247 Dataset: Binary Metrics Analysis (Test Ball Only)", fontsize=16, fontweight="bold", y=0.98)

    # Save the plot
    output_path = Path(__file__).parent / "binary_metrics_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print detailed summary statistics
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY STATISTICS")
    print("=" * 80)

    # Create combined grouping variable for analysis
    df_clean_combined = df_clean.copy()
    df_clean_combined["combined_group"] = (
        df_clean_combined[pretraining_col].astype(str) + " + " + df_clean_combined[genotype_col].astype(str)
    )

    print(f"\nCOMBINED PRETRAINING + GENOTYPE ANALYSIS:")
    print("-" * 50)

    for metric in available_metrics:
        print(f"\n{metric}:")
        prop_data, p_value, test_name = calculate_proportions_and_stats(df_clean_combined, "combined_group", metric)

        for _, row in prop_data.iterrows():
            print(f"  {row['combined_group']}: {row['n_positive']}/{row['n_total']} ({row['proportion']:.2%})")

        print(f"  Statistical test: {test_name}, p = {p_value:.4f}")

    # Create summary table
    print(f"\n\nSUMMARY TABLE:")
    print("-" * 80)

    summary_table = []
    for metric in available_metrics:
        prop_data, p_value, test_name = calculate_proportions_and_stats(df_clean_combined, "combined_group", metric)

        for _, row in prop_data.iterrows():
            pretraining_val, genotype_val = row["combined_group"].split(" + ")
            summary_table.append(
                {
                    "Pretraining": pretraining_val,
                    "Genotype": genotype_val,
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
