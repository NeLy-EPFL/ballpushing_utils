#!/usr/bin/env python3
"""
Statistical parallel coordinates plot for PCA results
This script creates parallel coordinates plots with statistical significance information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os


def load_data_with_stats():
    """Load PCA data and statistical results"""
    pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")

    try:
        stats_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
        print("Statistical results loaded successfully")
    except FileNotFoundError:
        print("Statistical results file not found. Creating plot without significance info.")
        stats_results = None

    return pca_data, stats_results


def get_significant_genotypes(stats_results):
    """Get different categories of significant genotypes"""
    if stats_results is None:
        return {}, {}

    significance_categories = {
        "all_methods": stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            & (stats_results["Permutation_FDR_significant"] == True)
            & (stats_results["Mahalanobis_FDR_significant"] == True)
        ]["Nickname"].tolist(),
        "mann_whitney": stats_results[stats_results["MannWhitney_any_dim_significant"] == True]["Nickname"].tolist(),
        "permutation": stats_results[stats_results["Permutation_FDR_significant"] == True]["Nickname"].tolist(),
        "mahalanobis": stats_results[stats_results["Mahalanobis_FDR_significant"] == True]["Nickname"].tolist(),
    }

    # Create a mapping of genotype to significance info
    genotype_significance = {}
    for _, row in stats_results.iterrows():
        genotype = row["Nickname"]
        genotype_significance[genotype] = {
            "mann_whitney": row["MannWhitney_any_dim_significant"],
            "permutation": row["Permutation_FDR_significant"],
            "mahalanobis": row["Mahalanobis_FDR_significant"],
            "permutation_pval": row["Permutation_FDR_pval"],
            "mahalanobis_pval": row["Mahalanobis_FDR_pval"],
        }

    return significance_categories, genotype_significance


def create_statistical_parallel_plot(
    pca_data, stats_results=None, n_components=6, focus_on_significant=True, max_genotypes=25
):
    """
    Create parallel coordinates plot with statistical significance information
    """

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]

    # Get significance information
    significance_categories, genotype_significance = get_significant_genotypes(stats_results)

    # Select genotypes to show
    if focus_on_significant and stats_results is not None:
        # Prioritize significant genotypes
        significant_any = set()
        for category in significance_categories.values():
            significant_any.update(category)

        # Add top genotypes by sample count
        genotype_counts = pca_data["Nickname"].value_counts()
        top_genotypes = genotype_counts.head(max_genotypes).index.tolist()

        genotypes_to_show = list(significant_any) + [g for g in top_genotypes if g not in significant_any]
        genotypes_to_show = genotypes_to_show[:max_genotypes]
    else:
        genotype_counts = pca_data["Nickname"].value_counts()
        genotypes_to_show = genotype_counts.head(max_genotypes).index.tolist()

    # Filter data
    plot_data = pca_data[pca_data["Nickname"].isin(genotypes_to_show)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define colors and line styles based on significance
    def get_line_properties(genotype):
        if stats_results is None or genotype not in genotype_significance:
            return "gray", 1, "-", 0.6

        sig_info = genotype_significance[genotype]

        # Color based on significance in different methods
        if sig_info["mann_whitney"] and sig_info["permutation"] and sig_info["mahalanobis"]:
            color = "red"  # Significant in all methods
            linewidth = 3
        elif sig_info["mann_whitney"] and sig_info["permutation"]:
            color = "orange"  # Significant in Mann-Whitney and Permutation
            linewidth = 2.5
        elif sig_info["mann_whitney"]:
            color = "blue"  # Significant in Mann-Whitney only
            linewidth = 2
        else:
            color = "gray"  # Not significant
            linewidth = 1

        return color, linewidth, "-", 0.8

    # Plot all genotypes
    legend_elements = []
    plotted_genotypes = []

    for genotype in genotypes_to_show:
        if genotype in plot_data["Nickname"].values:
            group = plot_data[plot_data["Nickname"] == genotype]
            means = [group[col].mean() for col in pca_cols]
            stds = [group[col].std() for col in pca_cols]

            color, linewidth, linestyle, alpha = get_line_properties(genotype)

            # Plot mean line
            line = ax.plot(
                range(len(pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                label=genotype,
            )[0]

            # Add error bars for significant genotypes
            if color != "gray":
                ax.errorbar(
                    range(len(pca_cols)), means, yerr=stds, color=color, alpha=0.3, capsize=2, linewidth=1, zorder=1
                )

            plotted_genotypes.append(genotype)

    # Customize plot
    ax.set_xticks(range(len(pca_cols)))
    ax.set_xticklabels(pca_cols, fontsize=12)
    ax.set_xlabel("PCA Components", fontsize=14)
    ax.set_ylabel("PCA Score", fontsize=14)
    ax.set_title(
        f"Parallel Coordinates Plot - PCA Components 1-{n_components}\n"
        f"Line thickness indicates statistical significance",
        fontsize=16,
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Create custom legend for significance levels
    if stats_results is not None:
        legend_elements = [
            mlines.Line2D([0], [0], color="red", linewidth=3, label="Significant in all methods"),
            mlines.Line2D([0], [0], color="orange", linewidth=2.5, label="Significant in MW & Permutation"),
            mlines.Line2D([0], [0], color="blue", linewidth=2, label="Significant in Mann-Whitney"),
            mlines.Line2D([0], [0], color="gray", linewidth=1, label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    return fig, ax, plotted_genotypes


def create_significance_summary(stats_results, save_name="significance_summary"):
    """Create a summary plot of statistical significance"""

    if stats_results is None:
        print("No statistical results available for summary")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Number of significant genotypes per method
    methods = ["MannWhitney_any_dim_significant", "Permutation_FDR_significant", "Mahalanobis_FDR_significant"]
    method_names = ["Mann-Whitney", "Permutation", "Mahalanobis"]

    counts = [stats_results[method].sum() for method in methods]

    bars = ax1.bar(method_names, counts, color=["blue", "green", "orange"], alpha=0.7)
    ax1.set_ylabel("Number of Significant Genotypes")
    ax1.set_title("Significant Genotypes by Statistical Method")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontsize=12
        )

    # Plot 2: P-value distribution
    pvals = stats_results["Permutation_FDR_pval"].dropna()
    ax2.hist(pvals, bins=20, alpha=0.7, color="green", edgecolor="black")
    ax2.axvline(x=0.05, color="red", linestyle="--", label="α = 0.05")
    ax2.set_xlabel("FDR-corrected P-values")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Permutation Test P-values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")

    return fig


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load data
    print("Loading data and statistical results...")
    pca_data, stats_results = load_data_with_stats()

    print(f"Data shape: {pca_data.shape}")
    print(f"Number of unique genotypes: {pca_data['Nickname'].nunique()}")

    # Create statistical parallel coordinates plot
    print("Creating statistical parallel coordinates plot...")
    fig1, ax1, plotted_genotypes = create_statistical_parallel_plot(
        pca_data, stats_results, n_components=6, focus_on_significant=True, max_genotypes=25
    )

    # Save the plot
    plt.figure(fig1.number)
    plt.savefig("parallel_coordinates_statistical.png", dpi=300, bbox_inches="tight")
    plt.savefig("parallel_coordinates_statistical.pdf", bbox_inches="tight")

    # Create significance summary
    if stats_results is not None:
        print("Creating significance summary...")
        fig2 = create_significance_summary(stats_results)

        # Print summary statistics
        print("\nStatistical Summary:")
        print(f"Total genotypes tested: {len(stats_results)}")
        print(f"Significant in Mann-Whitney: {stats_results['MannWhitney_any_dim_significant'].sum()}")
        print(f"Significant in Permutation test: {stats_results['Permutation_FDR_significant'].sum()}")
        print(f"Significant in Mahalanobis test: {stats_results['Mahalanobis_FDR_significant'].sum()}")

        # Most significant genotypes
        most_significant = stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            & (stats_results["Permutation_FDR_significant"] == True)
            & (stats_results["Mahalanobis_FDR_significant"] == True)
        ]["Nickname"].tolist()

        print(f"\nGenotypes significant in all methods ({len(most_significant)}):")
        for genotype in most_significant:
            print(f"  - {genotype}")

    print(f"\nPlots saved as 'parallel_coordinates_statistical.png/pdf'")
    print(f"Genotypes plotted: {len(plotted_genotypes)}")

    plt.show()


if __name__ == "__main__":
    main()
