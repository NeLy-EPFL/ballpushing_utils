#!/usr/bin/env python3
"""
Parallel coordinates plot for PCA results
This script creates a parallel coordinates visualization for PCA components,
showing different genotypes with their statistical significance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.colors import to_rgba
import sys
import os

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_data():
    """Load the PCA data and statistical results"""
    # Load PCA data
    pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")

    # Load statistical results
    try:
        stats_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
    except FileNotFoundError:
        print("Statistical results file not found. Run PCA_Static.py first.")
        stats_results = None

    return pca_data, stats_results


def get_significant_genotypes(stats_results, method="all"):
    """Get genotypes that are significant according to specified method"""
    if stats_results is None:
        return []

    if method == "all":
        # Significant in all three methods
        significant = stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            & (stats_results["Permutation_FDR_significant"] == True)
            & (stats_results["Mahalanobis_FDR_significant"] == True)
        ]
    elif method == "any":
        # Significant in any method
        significant = stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            | (stats_results["Permutation_FDR_significant"] == True)
            | (stats_results["Mahalanobis_FDR_significant"] == True)
        ]
    elif method == "mann_whitney":
        significant = stats_results[stats_results["MannWhitney_any_dim_significant"] == True]
    elif method == "permutation":
        significant = stats_results[stats_results["Permutation_FDR_significant"] == True]
    elif method == "mahalanobis":
        significant = stats_results[stats_results["Mahalanobis_FDR_significant"] == True]
    else:
        significant = pd.DataFrame()

    return significant["Nickname"].tolist() if not significant.empty else []


def create_parallel_coordinates_plot(
    pca_data,
    stats_results=None,
    n_components=5,
    significance_method="all",
    max_genotypes=20,
    show_individuals=True,
    alpha_individuals=0.3,
    show_means=True,
    highlight_significant=True,
):
    """
    Create a parallel coordinates plot for PCA components

    Parameters:
    - pca_data: DataFrame with PCA scores and metadata
    - stats_results: DataFrame with statistical results
    - n_components: Number of PCA components to show
    - significance_method: 'all', 'any', 'mann_whitney', 'permutation', 'mahalanobis'
    - max_genotypes: Maximum number of genotypes to show
    - show_individuals: Whether to show individual samples
    - alpha_individuals: Transparency for individual lines
    - show_means: Whether to show genotype means
    - highlight_significant: Whether to highlight significant genotypes
    """

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]

    # Get significant genotypes
    significant_genotypes = get_significant_genotypes(stats_results, significance_method)

    # Get genotype counts and select top ones
    genotype_counts = pca_data["Nickname"].value_counts()
    top_genotypes = genotype_counts.head(max_genotypes).index.tolist()

    # Combine significant and top genotypes
    genotypes_to_show = list(set(significant_genotypes + top_genotypes))

    # Filter data
    plot_data = pca_data[pca_data["Nickname"].isin(genotypes_to_show)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette
    unique_genotypes = plot_data["Nickname"].unique()
    colors = sns.color_palette("husl", len(unique_genotypes))
    color_map = dict(zip(unique_genotypes, colors))

    # Plot individual samples if requested
    if show_individuals:
        for i, (genotype, group) in enumerate(plot_data.groupby("Nickname")):
            color = color_map[genotype]
            is_significant = genotype in significant_genotypes

            # Adjust alpha based on significance
            alpha = alpha_individuals if not (highlight_significant and is_significant) else alpha_individuals * 2

            for _, row in group.iterrows():
                y_values = [row[col] for col in pca_cols]
                x_values = range(len(pca_cols))

                ax.plot(x_values, y_values, color=color, alpha=alpha, linewidth=0.5)

    # Plot means if requested
    if show_means:
        for genotype, group in plot_data.groupby("Nickname"):
            color = color_map[genotype]
            is_significant = genotype in significant_genotypes

            # Calculate means
            means = [group[col].mean() for col in pca_cols]
            stds = [group[col].std() for col in pca_cols]

            # Plot mean line
            linewidth = 3 if (highlight_significant and is_significant) else 2
            linestyle = "-" if (highlight_significant and is_significant) else "--"

            ax.plot(
                range(len(pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=genotype,
                alpha=0.8,
            )

            # Add error bars
            ax.errorbar(range(len(pca_cols)), means, yerr=stds, color=color, alpha=0.5, capsize=3, linewidth=1)

    # Customize the plot
    ax.set_xticks(range(len(pca_cols)))
    ax.set_xticklabels(pca_cols, rotation=0)
    ax.set_xlabel("PCA Components")
    ax.set_ylabel("PCA Score")
    ax.set_title(
        f"Parallel Coordinates Plot - PCA Components 1-{n_components}\n"
        f"Significant genotypes ({significance_method} method) highlighted"
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend (only for significant genotypes if there are many)
    if len(unique_genotypes) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    elif highlight_significant and significant_genotypes:
        # Only show legend for significant genotypes
        significant_handles = []
        for genotype in significant_genotypes:
            if genotype in color_map:
                significant_handles.append(
                    mlines.Line2D([0], [0], color=color_map[genotype], linewidth=3, label=genotype)
                )
        ax.legend(
            handles=significant_handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Significant Genotypes"
        )

    plt.tight_layout()
    return fig, ax


def create_summary_plot(pca_data, stats_results=None, n_components=5):
    """Create a summary plot showing variance explained and significance"""

    # Calculate variance explained (approximate from PCA results)
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Variance explained per component (approximate)
    total_vars = []
    for col in pca_cols:
        total_vars.append(pca_data[col].var())

    total_variance = sum(total_vars)
    explained_ratios = [var / total_variance for var in total_vars]

    ax1.bar(range(len(pca_cols)), explained_ratios, alpha=0.7)
    ax1.set_xticks(range(len(pca_cols)))
    ax1.set_xticklabels(pca_cols)
    ax1.set_ylabel("Relative Variance")
    ax1.set_title("Relative Variance per PCA Component")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of significant genotypes per method
    if stats_results is not None:
        methods = ["MannWhitney_any_dim_significant", "Permutation_FDR_significant", "Mahalanobis_FDR_significant"]
        method_names = ["Mann-Whitney", "Permutation", "Mahalanobis"]

        counts = []
        for method in methods:
            counts.append(stats_results[method].sum())

        bars = ax2.bar(method_names, counts, alpha=0.7)
        ax2.set_ylabel("Number of Significant Genotypes")
        ax2.set_title("Significant Genotypes by Statistical Method")
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count), ha="center", va="bottom")

    plt.tight_layout()
    return fig


def main():
    """Main function to create the parallel coordinates plot"""

    # Change to the PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load data
    print("Loading data...")
    pca_data, stats_results = load_data()

    print(f"Data shape: {pca_data.shape}")
    print(f"Number of unique genotypes: {pca_data['Nickname'].nunique()}")

    if stats_results is not None:
        print(f"Statistical results shape: {stats_results.shape}")
        # Print summary of significant genotypes
        for method in ["all", "any", "mann_whitney", "permutation", "mahalanobis"]:
            sig_genotypes = get_significant_genotypes(stats_results, method)
            print(f"Significant genotypes ({method}): {len(sig_genotypes)}")

    # Create parallel coordinates plot
    print("Creating parallel coordinates plot...")
    fig1, ax1 = create_parallel_coordinates_plot(
        pca_data,
        stats_results,
        n_components=6,  # Show first 6 components
        significance_method="any",  # Show genotypes significant in any method
        max_genotypes=30,  # Show top 30 genotypes by sample count
        show_individuals=True,
        alpha_individuals=0.2,
        show_means=True,
        highlight_significant=True,
    )

    # Save the plot
    plt.figure(fig1.number)
    plt.savefig("parallel_coordinates_pca.png", dpi=300, bbox_inches="tight")
    plt.savefig("parallel_coordinates_pca.pdf", bbox_inches="tight")

    # Create summary plot
    print("Creating summary plot...")
    fig2 = create_summary_plot(pca_data, stats_results, n_components=6)

    plt.figure(fig2.number)
    plt.savefig("pca_summary.png", dpi=300, bbox_inches="tight")
    plt.savefig("pca_summary.pdf", bbox_inches="tight")

    # Show plots
    plt.show()

    print("Plots saved as 'parallel_coordinates_pca.png/pdf' and 'pca_summary.png/pdf'")


if __name__ == "__main__":
    main()
