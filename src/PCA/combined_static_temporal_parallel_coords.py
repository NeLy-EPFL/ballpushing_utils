#!/usr/bin/env python3
"""
Combined Static and Temporal Parallel Coordinates Plot for PCA Results
This script creates side-by-side parallel coordinates plots showing both static and temporal hits
with brain region color coding, using raw permutation test p-values for consistency with heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os
import ast
from scipy import stats

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config


def load_data_with_brain_regions():
    """Load PCA data and both static and temporal statistical results"""
    # Load PCA score data
    pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")

    # Load static results
    try:
        static_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
        print("Static statistical results loaded successfully")
    except FileNotFoundError:
        print("Static statistical results file not found.")
        static_results = None

    # Load temporal results
    try:
        temporal_results = pd.read_csv("fpca_temporal_stats_results_allmethods_tailoredctrls.csv")
        print("Temporal statistical results loaded successfully")
    except FileNotFoundError:
        print("Temporal statistical results file not found.")
        temporal_results = None

    return pca_data, static_results, temporal_results


def get_significance_from_permutation(stats_results, significance_threshold=0.05):
    """
    Extract significance information from permutation test p-values

    Parameters:
    - stats_results: DataFrame with statistical results
    - significance_threshold: Threshold for significance (default: 0.05)

    Returns:
    - Dictionary mapping genotype to significance information
    """

    if stats_results is None:
        return {}

    genotype_significance = {}

    for _, row in stats_results.iterrows():
        genotype = row["Nickname"]

        # Use raw permutation test p-values (same as heatmap)
        permutation_pval = row["Permutation_pval"]
        permutation_fdr_pval = row["Permutation_FDR_pval"]
        is_significant = permutation_pval < significance_threshold  # Use raw p-value like heatmap

        genotype_significance[genotype] = {
            "is_significant": is_significant,
            "permutation_pval": permutation_pval,
            "permutation_fdr_pval": permutation_fdr_pval,
            "min_pval": permutation_pval,  # For sorting consistency
            "test_type": "permutation",
        }

    return genotype_significance


def get_brain_region_colors():
    """Get the color dictionary from Config"""
    return Config.color_dict


def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrapped confidence intervals for a dataset

    Parameters:
    - data: array-like, the data to bootstrap
    - n_bootstrap: number of bootstrap samples
    - confidence_level: confidence level (0.95 for 95% CI)

    Returns:
    - lower_bound, upper_bound: confidence interval bounds
    """
    if len(data) < 2:
        return np.nan, np.nan

    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return lower_bound, upper_bound


def get_control_confidence_intervals(pca_data, available_pca_cols, n_bootstrap=1000):
    """
    Calculate bootstrapped confidence intervals for control genotypes

    Parameters:
    - pca_data: DataFrame with PCA scores and metadata
    - available_pca_cols: list of PCA column names
    - n_bootstrap: number of bootstrap samples

    Returns:
    - control_means: mean values for each PCA component
    - control_lower: lower CI bounds for each PCA component
    - control_upper: upper CI bounds for each PCA component
    """
    # Get control data (brain region 'Control')
    control_data = pca_data[pca_data["Brain region"] == "Control"]

    if len(control_data) == 0:
        print("Warning: No control data found")
        return None, None, None

    print(f"Control data: {len(control_data)} samples")

    control_means = []
    control_lower = []
    control_upper = []

    for col in available_pca_cols:
        data = control_data[col].values
        mean_val = np.mean(data)
        lower_ci, upper_ci = bootstrap_confidence_interval(data, n_bootstrap)

        control_means.append(mean_val)
        control_lower.append(lower_ci)
        control_upper.append(upper_ci)

    return control_means, control_lower, control_upper


def create_combined_parallel_plot(
    pca_data,
    static_results=None,
    temporal_results=None,
    n_components=6,
    significance_threshold=0.05,
    save_prefix="combined_static_temporal_parallel",
):
    """
    Create combined parallel coordinates plot showing both static and temporal hits

    Parameters:
    - pca_data: DataFrame with PCA scores and metadata
    - static_results: DataFrame with static statistical results
    - temporal_results: DataFrame with temporal statistical results
    - n_components: Number of PCA components to show (default: 6)
    - significance_threshold: Threshold for significance (default: 0.05)
    - save_prefix: Prefix for saved files
    """

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]
    available_pca_cols = [col for col in pca_cols if col in pca_data.columns]
    print(f"Using PCA components: {available_pca_cols}")

    # Get significance information for both static and temporal
    static_significance = get_significance_from_permutation(static_results, significance_threshold)
    temporal_significance = get_significance_from_permutation(temporal_results, significance_threshold)

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()

    # Calculate control confidence intervals
    control_means, control_lower, control_upper = get_control_confidence_intervals(
        pca_data, available_pca_cols, n_bootstrap=1000
    )

    # Get significant genotypes for both conditions
    static_significant = [g for g in static_significance.keys() if static_significance[g]["is_significant"]]
    temporal_significant = [g for g in temporal_significance.keys() if temporal_significance[g]["is_significant"]]

    print(f"Static significant genotypes: {len(static_significant)}")
    print(f"Temporal significant genotypes: {len(temporal_significant)}")

    # Filter to genotypes with sufficient samples
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 5].index.tolist()

    # Filter significant genotypes to those with sufficient samples
    static_to_plot = [g for g in static_significant if g in genotypes_with_samples]
    temporal_to_plot = [g for g in temporal_significant if g in genotypes_with_samples]

    print(f"Static genotypes to plot: {len(static_to_plot)}")
    print(f"Temporal genotypes to plot: {len(temporal_to_plot)}")

    # Create figure with two subplots side by side
    fig, (ax_static, ax_temporal) = plt.subplots(1, 2, figsize=(24, 10))

    # Function to get line properties based on brain region and significance
    def get_line_properties(genotype, brain_region, significance_dict):
        # Get brain region color
        base_color = brain_region_colors.get(brain_region, "#000000")  # Default to black

        # Get significance info
        if genotype in significance_dict:
            sig_info = significance_dict[genotype]
            is_significant = sig_info["is_significant"]
            min_pval = sig_info["min_pval"]

            # Line width based on significance strength
            if is_significant:
                if min_pval < 0.001:
                    linewidth = 4  # Very significant
                elif min_pval < 0.01:
                    linewidth = 3  # Significant
                else:
                    linewidth = 2  # Moderately significant
                alpha = 0.6  # Reduced opacity for better visibility
            else:
                linewidth = 1
                alpha = 0.3  # Reduced opacity for non-significant lines
        else:
            linewidth = 1
            alpha = 0.3  # Reduced opacity for non-significant lines

        return base_color, linewidth, alpha

    # Plot static results
    static_legend_info = {}
    static_plotted = []

    # Add control confidence interval shading with higher z-order for better visibility
    if control_means is not None:
        x_values = range(len(available_pca_cols))
        ax_static.fill_between(
            x_values, control_lower, control_upper, color="lightgray", alpha=0.4, label="Control 95% CI", zorder=2
        )
        ax_static.plot(
            x_values,
            control_means,
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label="Control Mean",
            zorder=3,
        )

    for genotype in static_to_plot:
        if genotype in pca_data["Nickname"].values:
            group = pca_data[pca_data["Nickname"] == genotype]
            brain_region = group["Brain region"].iloc[0]

            # Calculate means and standard errors
            means = [group[col].mean() for col in available_pca_cols]
            sems = [group[col].sem() for col in available_pca_cols]
            n_samples = len(group)

            color, linewidth, alpha = get_line_properties(genotype, brain_region, static_significance)

            # Plot mean line
            ax_static.plot(
                range(len(available_pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{genotype} (n={n_samples})",
                zorder=1,
            )

            # Add error bars
            ax_static.errorbar(
                range(len(available_pca_cols)),
                means,
                yerr=sems,
                color=color,
                alpha=0.3,
                capsize=2,
                linewidth=0.5,
                zorder=1,
            )

            static_plotted.append(genotype)

            # Collect legend info
            if brain_region not in static_legend_info:
                static_legend_info[brain_region] = color

    # Plot temporal results
    temporal_legend_info = {}
    temporal_plotted = []

    # Add control confidence interval shading with higher z-order for better visibility
    if control_means is not None:
        x_values = range(len(available_pca_cols))
        ax_temporal.fill_between(
            x_values, control_lower, control_upper, color="lightgray", alpha=0.4, label="Control 95% CI", zorder=2
        )
        ax_temporal.plot(
            x_values,
            control_means,
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label="Control Mean",
            zorder=3,
        )

    for genotype in temporal_to_plot:
        if genotype in pca_data["Nickname"].values:
            group = pca_data[pca_data["Nickname"] == genotype]
            brain_region = group["Brain region"].iloc[0]

            # Calculate means and standard errors
            means = [group[col].mean() for col in available_pca_cols]
            sems = [group[col].sem() for col in available_pca_cols]
            n_samples = len(group)

            color, linewidth, alpha = get_line_properties(genotype, brain_region, temporal_significance)

            # Plot mean line
            ax_temporal.plot(
                range(len(available_pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{genotype} (n={n_samples})",
                zorder=1,
            )

            # Add error bars
            ax_temporal.errorbar(
                range(len(available_pca_cols)),
                means,
                yerr=sems,
                color=color,
                alpha=0.3,
                capsize=2,
                linewidth=0.5,
                zorder=1,
            )

            temporal_plotted.append(genotype)

            # Collect legend info
            if brain_region not in temporal_legend_info:
                temporal_legend_info[brain_region] = color

    # Customize static plot
    ax_static.set_xticks(range(len(available_pca_cols)))
    ax_static.set_xticklabels(available_pca_cols, fontsize=12)
    ax_static.set_xlabel("PCA Components", fontsize=14)
    ax_static.set_ylabel("PCA Score", fontsize=14)
    ax_static.set_title(
        f"Static PCA - Parallel Coordinates\\n" f"{len(static_plotted)} Significant Genotypes (p < 0.05)", fontsize=16
    )
    ax_static.grid(True, alpha=0.3)
    ax_static.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Customize temporal plot
    ax_temporal.set_xticks(range(len(available_pca_cols)))
    ax_temporal.set_xticklabels(available_pca_cols, fontsize=12)
    ax_temporal.set_xlabel("PCA Components", fontsize=14)
    ax_temporal.set_ylabel("PCA Score", fontsize=14)
    ax_temporal.set_title(
        f"Temporal PCA - Parallel Coordinates\\n" f"{len(temporal_plotted)} Significant Genotypes (p < 0.05)",
        fontsize=16,
    )
    ax_temporal.grid(True, alpha=0.3)
    ax_temporal.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Add legend for brain regions (combine both plots)
    all_regions = set(list(static_legend_info.keys()) + list(temporal_legend_info.keys()))
    legend_elements = []

    # Add control elements first
    if control_means is not None:
        legend_elements.append(
            mlines.Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Control Mean")
        )
        legend_elements.append(
            mpatches.Rectangle((0, 0), 1, 1, facecolor="lightgray", alpha=0.3, label="Control 95% CI")
        )

    # Add brain region elements
    for brain_region in sorted(all_regions):
        color = brain_region_colors.get(brain_region, "#000000")
        legend_elements.append(mlines.Line2D([0], [0], color=color, linewidth=2, label=brain_region))

    # Place legend between the plots
    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(legend_elements), 6),
        title="Brain Regions & Controls",
    )

    # Add significance legend
    sig_legend_elements = [
        mlines.Line2D([0], [0], color="gray", linewidth=4, label="p < 0.001"),
        mlines.Line2D([0], [0], color="gray", linewidth=3, label="p < 0.01"),
        mlines.Line2D([0], [0], color="gray", linewidth=2, label="p < 0.05"),
        mlines.Line2D([0], [0], color="gray", linewidth=1, label="Not significant"),
    ]

    # Place significance legend on the right side
    fig.legend(handles=sig_legend_elements, loc="right", bbox_to_anchor=(1.02, 0.5), title="Significance Level")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)

    # Save plots
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")

    # Print summary
    print(f"\\n=== Combined Plot Summary ===")
    print(f"Static genotypes plotted: {len(static_plotted)}")
    print(f"Temporal genotypes plotted: {len(temporal_plotted)}")
    print(f"PCA components used: {len(available_pca_cols)}")

    # Print brain region breakdown
    print(f"\\n=== Static Brain Region Breakdown ===")
    static_region_stats = {}
    for genotype in static_plotted:
        if genotype in pca_data["Nickname"].values:
            brain_region = pca_data[pca_data["Nickname"] == genotype]["Brain region"].iloc[0]
            if brain_region not in static_region_stats:
                static_region_stats[brain_region] = 0
            static_region_stats[brain_region] += 1

    for region, count in sorted(static_region_stats.items()):
        print(f"{region}: {count} genotypes")

    print(f"\\n=== Temporal Brain Region Breakdown ===")
    temporal_region_stats = {}
    for genotype in temporal_plotted:
        if genotype in pca_data["Nickname"].values:
            brain_region = pca_data[pca_data["Nickname"] == genotype]["Brain region"].iloc[0]
            if brain_region not in temporal_region_stats:
                temporal_region_stats[brain_region] = 0
            temporal_region_stats[brain_region] += 1

    for region, count in sorted(temporal_region_stats.items()):
        print(f"{region}: {count} genotypes")

    # Print overlap analysis
    overlap = set(static_plotted) & set(temporal_plotted)
    static_only = set(static_plotted) - set(temporal_plotted)
    temporal_only = set(temporal_plotted) - set(static_plotted)

    print(f"\\n=== Overlap Analysis ===")
    print(f"Both static and temporal hits: {len(overlap)}")
    print(f"Static only: {len(static_only)}")
    print(f"Temporal only: {len(temporal_only)}")

    if overlap:
        print(f"\\nGenotypes significant in both:")
        for genotype in sorted(overlap):
            static_p = static_significance[genotype]["permutation_pval"]
            temporal_p = temporal_significance[genotype]["permutation_pval"]
            print(f"  {genotype}: static p={static_p:.4f}, temporal p={temporal_p:.4f}")

    return fig, static_plotted, temporal_plotted


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load data
    print("Loading data with brain region information...")
    pca_data, static_results, temporal_results = load_data_with_brain_regions()

    if static_results is not None:
        print(f"Static results shape: {static_results.shape}")
    if temporal_results is not None:
        print(f"Temporal results shape: {temporal_results.shape}")

    print(f"PCA data shape: {pca_data.shape}")
    print(f"Unique brain regions: {sorted(pca_data['Brain region'].unique())}")

    # Create combined parallel coordinates plot
    print("\\nCreating combined static and temporal parallel coordinates plot...")
    fig, static_plotted, temporal_plotted = create_combined_parallel_plot(
        pca_data,
        static_results,
        temporal_results,
        n_components=6,  # Use first 6 components
        significance_threshold=0.05,
        save_prefix="combined_static_temporal_parallel_coords",
    )

    plt.show()

    print("\\nPlot saved as 'combined_static_temporal_parallel_coords.png/pdf'")


if __name__ == "__main__":
    main()
