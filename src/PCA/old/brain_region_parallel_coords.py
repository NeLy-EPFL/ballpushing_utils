#!/usr/bin/env python3
"""
Brain Region Parallel Coordinates Plot for PCA Results
This script creates parallel coordinates plots with brain region color coding,
focusing on permutation test p-values, and using all 10 PCA components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os
import ast

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config


def load_data_with_brain_regions():
    """Load PCA data and statistical results with brain region information"""
    pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")

    try:
        stats_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
        print("Statistical results loaded successfully")
        print(f"Available columns: {stats_results.columns.tolist()}")
    except FileNotFoundError:
        print("Statistical results file not found. Run PCA_Static.py first.")
        stats_results = None

    return pca_data, stats_results


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


def create_brain_region_parallel_plot(
    pca_data,
    stats_results=None,
    n_components=6,
    significance_threshold=0.05,
    max_genotypes=30,
    save_prefix="brain_region_parallel",
    plot_mode="significant_only",
):
    """
    Create parallel coordinates plot with brain region color coding

    Parameters:
    - pca_data: DataFrame with PCA scores and metadata
    - stats_results: DataFrame with statistical results
    - n_components: Number of PCA components to show (default: 6)
    - significance_threshold: Threshold for significance (default: 0.05)
    - max_genotypes: Maximum number of genotypes to show
    - save_prefix: Prefix for saved files
    - plot_mode: 'significant_only' or 'mixed' (significant + top by sample count)
    """

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]
    available_pca_cols = [col for col in pca_cols if col in pca_data.columns]
    print(f"Using PCA components: {available_pca_cols}")

    # Get significance information
    genotype_significance = get_significance_from_permutation(stats_results, significance_threshold)

    # Debug: Print information about significance
    if stats_results is not None:
        print(f"Total genotypes in stats_results: {len(stats_results)}")
        print(f"Permutation_FDR_significant values: {stats_results['Permutation_FDR_significant'].value_counts()}")
        print(
            f"Number of significant genotypes: {len([g for g in genotype_significance.keys() if genotype_significance[g]['is_significant']])}"
        )
        print(
            f"Sample of significant genotypes: {[g for g in genotype_significance.keys() if genotype_significance[g]['is_significant']][:10]}"
        )
    else:
        print("stats_results is None!")

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()
    print(f"Brain region colors: {brain_region_colors}")

    # Filter to genotypes with sufficient samples and significance information
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 5].index.tolist()

    # Prioritize significant genotypes
    significant_genotypes = [g for g in genotype_significance.keys() if genotype_significance[g]["is_significant"]]

    # Select genotypes to plot based on mode
    if plot_mode == "significant_only":
        # Only plot significant genotypes
        genotypes_to_plot = [g for g in significant_genotypes if g in genotypes_with_samples]
        print(f"Plot mode: Significant genotypes only ({len(genotypes_to_plot)} genotypes)")
    else:
        # Mixed mode: significant + top by sample count
        if significant_genotypes:
            top_by_samples = genotype_counts.head(max_genotypes).index.tolist()
            genotypes_to_plot = list(set(significant_genotypes + top_by_samples))[:max_genotypes]
        else:
            genotypes_to_plot = genotypes_with_samples[:max_genotypes]
        print(f"Plot mode: Mixed (significant + top by sample count, {len(genotypes_to_plot)} genotypes)")

    # Ensure we have genotypes to plot
    if not genotypes_to_plot:
        print("No genotypes to plot! Check significance results.")
        return None, []

    # Filter data
    plot_data = pca_data[pca_data["Nickname"].isin(genotypes_to_plot)]

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[4, 1])

    # Main parallel coordinates plot
    ax_main = fig.add_subplot(gs[0, :])

    # Function to get line properties based on brain region and significance
    def get_line_properties(genotype, brain_region):
        # Get brain region color
        base_color = brain_region_colors.get(brain_region, "#000000")  # Default to black

        # Get significance info
        if genotype in genotype_significance:
            sig_info = genotype_significance[genotype]
            is_significant = sig_info["is_significant"]
            min_pval = sig_info["min_pval"]  # This is the permutation p-value

            # Line width based on significance strength
            if is_significant:
                if min_pval < 0.001:
                    linewidth = 4  # Very significant
                elif min_pval < 0.01:
                    linewidth = 3  # Significant
                else:
                    linewidth = 2  # Moderately significant
                alpha = 0.8
            else:
                linewidth = 1
                alpha = 0.5
        else:
            linewidth = 1
            alpha = 0.5

        return base_color, linewidth, alpha

    # Plot the lines
    plotted_genotypes = []
    legend_info = {}

    for genotype in genotypes_to_plot:
        if genotype in plot_data["Nickname"].values:
            group = plot_data[plot_data["Nickname"] == genotype]
            brain_region = group["Brain region"].iloc[0]

            # Calculate means and standard errors
            means = [group[col].mean() for col in available_pca_cols]
            sems = [group[col].sem() for col in available_pca_cols]
            n_samples = len(group)

            color, linewidth, alpha = get_line_properties(genotype, brain_region)

            # Plot mean line
            ax_main.plot(
                range(len(available_pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{genotype} (n={n_samples})",
            )

            # Add error bars for significant genotypes
            if genotype in genotype_significance and genotype_significance[genotype]["is_significant"]:
                ax_main.errorbar(
                    range(len(available_pca_cols)),
                    means,
                    yerr=sems,
                    color=color,
                    alpha=0.3,
                    capsize=2,
                    linewidth=0.5,
                    zorder=1,
                )

            plotted_genotypes.append(genotype)

            # Collect legend info
            if brain_region not in legend_info:
                legend_info[brain_region] = color

    # Customize main plot
    ax_main.set_xticks(range(len(available_pca_cols)))
    ax_main.set_xticklabels(available_pca_cols, fontsize=12)
    ax_main.set_xlabel("PCA Components", fontsize=14)
    ax_main.set_ylabel("PCA Score", fontsize=14)
    ax_main.set_title(
        f"Parallel Coordinates Plot - PCA Components 1-{len(available_pca_cols)}\\n"
        f"Brain Region Color Coding, Line Width = Permutation Test Significance",
        fontsize=16,
    )
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Add legend for brain regions
    legend_elements = []
    for brain_region, color in legend_info.items():
        legend_elements.append(mlines.Line2D([0], [0], color=color, linewidth=2, label=brain_region))

    ax_main.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1), title="Brain Regions")

    # Add significance legend
    sig_legend_elements = [
        mlines.Line2D([0], [0], color="gray", linewidth=4, label="p < 0.001"),
        mlines.Line2D([0], [0], color="gray", linewidth=3, label="p < 0.01"),
        mlines.Line2D([0], [0], color="gray", linewidth=2, label="p < 0.05"),
        mlines.Line2D([0], [0], color="gray", linewidth=1, label="Not significant"),
    ]
    ax_main.legend(handles=sig_legend_elements, loc="upper right", bbox_to_anchor=(1, 1), title="Significance Level")

    # Subplot 1: Brain region distribution
    ax_brain = fig.add_subplot(gs[1, 0])
    brain_region_counts = plot_data["Brain region"].value_counts()
    brain_colors = [brain_region_colors.get(region, "#000000") for region in brain_region_counts.index]

    bars = ax_brain.bar(range(len(brain_region_counts)), brain_region_counts.values, color=brain_colors, alpha=0.7)
    ax_brain.set_xticks(range(len(brain_region_counts)))
    ax_brain.set_xticklabels(brain_region_counts.index, rotation=45, ha="right", fontsize=10)
    ax_brain.set_ylabel("Number of Genotypes")
    ax_brain.set_title("Brain Region Distribution")
    ax_brain.grid(True, alpha=0.3)

    # Subplot 2: Significance distribution
    if genotype_significance:
        ax_sig = fig.add_subplot(gs[1, 1])

        # Count significant genotypes by brain region
        sig_by_region = {}
        for genotype in plotted_genotypes:
            if genotype in genotype_significance:
                group = plot_data[plot_data["Nickname"] == genotype]
                brain_region = group["Brain region"].iloc[0]
                is_sig = genotype_significance[genotype]["is_significant"]

                if brain_region not in sig_by_region:
                    sig_by_region[brain_region] = {"significant": 0, "total": 0}

                sig_by_region[brain_region]["total"] += 1
                if is_sig:
                    sig_by_region[brain_region]["significant"] += 1

        # Plot significance rates
        regions = list(sig_by_region.keys())
        sig_rates = [sig_by_region[region]["significant"] / sig_by_region[region]["total"] for region in regions]
        colors = [brain_region_colors.get(region, "#000000") for region in regions]

        bars = ax_sig.bar(range(len(regions)), sig_rates, color=colors, alpha=0.7)
        ax_sig.set_xticks(range(len(regions)))
        ax_sig.set_xticklabels(regions, rotation=45, ha="right", fontsize=10)
        ax_sig.set_ylabel("Significance Rate")
        ax_sig.set_title("Significance by Brain Region")
        ax_sig.set_ylim(0, 1)
        ax_sig.grid(True, alpha=0.3)

    # Subplot 3: Component variance
    ax_var = fig.add_subplot(gs[2, :])

    # Calculate relative variance for each component
    component_vars = [pca_data[col].var() for col in available_pca_cols]
    total_var = np.sum(component_vars)
    explained_ratios = [var / total_var for var in component_vars]

    bars = ax_var.bar(available_pca_cols, explained_ratios, alpha=0.7)
    ax_var.set_ylabel("Relative Variance")
    ax_var.set_title("Relative Variance Explained by Each Component")
    ax_var.grid(True, alpha=0.3)

    # Add percentages on bars
    for bar, ratio in zip(bars, explained_ratios):
        ax_var.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{ratio:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save plots
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")

    # Print summary
    print(f"\\n=== Plot Summary ===")
    print(f"Total genotypes plotted: {len(plotted_genotypes)}")
    print(f"PCA components used: {len(available_pca_cols)}")
    print(f"Brain regions represented: {len(legend_info)}")

    if genotype_significance:
        significant_plotted = [
            g for g in plotted_genotypes if g in genotype_significance and genotype_significance[g]["is_significant"]
        ]
        print(f"Significant genotypes plotted: {len(significant_plotted)}")

        # Categorize genotypes like in the heatmap
        static_hits = []
        no_hits = []

        for genotype in plotted_genotypes:
            if genotype in genotype_significance:
                is_significant = genotype_significance[genotype]["is_significant"]
                if is_significant:
                    static_hits.append(genotype)
                else:
                    no_hits.append(genotype)
            else:
                no_hits.append(genotype)

        print(f"\\n=== Genotype Categories ===")
        print(f"Static hits: {len(static_hits)}")
        print(f"No hits: {len(no_hits)}")

        if static_hits:
            print(f"\\n=== All Significant Genotypes (Static hits) ===")
            # Sort by permutation p-value
            static_hits_sorted = sorted(static_hits, key=lambda x: genotype_significance[x]["min_pval"])

            for i, genotype in enumerate(static_hits_sorted):
                sig_info = genotype_significance[genotype]
                brain_region = plot_data[plot_data["Nickname"] == genotype]["Brain region"].iloc[0]

                # Get sample count
                n_samples = len(plot_data[plot_data["Nickname"] == genotype])

                print(
                    f"{i+1:2d}. {genotype} ({brain_region}) - "
                    f"n={n_samples}, permutation p={sig_info['permutation_pval']:.4f}, "
                    f"FDR p={sig_info['permutation_fdr_pval']:.4f}"
                )

                # Show the permutation test details
                print(
                    f"    Permutation test: raw p={sig_info['permutation_pval']:.4f}, "
                    f"FDR-corrected p={sig_info['permutation_fdr_pval']:.4f}"
                )

        # Show brain region breakdown
        print(f"\\n=== Significance by Brain Region ===")
        region_stats = {}
        for genotype in plotted_genotypes:
            if genotype in plot_data["Nickname"].values:
                brain_region = plot_data[plot_data["Nickname"] == genotype]["Brain region"].iloc[0]
                is_sig = genotype in genotype_significance and genotype_significance[genotype]["is_significant"]

                if brain_region not in region_stats:
                    region_stats[brain_region] = {"total": 0, "significant": 0}

                region_stats[brain_region]["total"] += 1
                if is_sig:
                    region_stats[brain_region]["significant"] += 1

        for region, stats in sorted(region_stats.items()):
            rate = stats["significant"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{region}: {stats['significant']}/{stats['total']} ({rate:.2%})")

    return fig, plotted_genotypes


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load data
    print("Loading data with brain region information...")
    pca_data, stats_results = load_data_with_brain_regions()

    print(f"Data shape: {pca_data.shape}")
    print(f"Unique brain regions: {sorted(pca_data['Brain region'].unique())}")

    # Create brain region parallel coordinates plot
    print("\\nCreating brain region parallel coordinates plot...")
    fig, plotted_genotypes = create_brain_region_parallel_plot(
        pca_data,
        stats_results,
        n_components=6,  # Use first 6 components (95% variance)
        significance_threshold=0.05,
        max_genotypes=35,
        save_prefix="brain_region_parallel_coords",
        plot_mode="significant_only",  # Only plot significant genotypes
    )

    if fig is None:
        print("No plot generated. Exiting.")
        return

    plt.show()

    print("\\nPlot saved as 'brain_region_parallel_coords.png/pdf'")


if __name__ == "__main__":
    main()
