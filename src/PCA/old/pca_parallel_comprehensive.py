#!/usr/bin/env python3
"""
Comprehensive PCA Parallel Coordinates Visualization
This script creates high-quality parallel coordinates plots for PCA results with multiple options.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os


def create_comprehensive_plot(
    data_file="static_pca_with_metadata_tailoredctrls.feather",
    stats_file="static_pca_stats_results_allmethods_tailoredctrls.csv",
    n_components=6,
    plot_type="statistical",
    genotypes_of_interest=None,
    save_prefix="pca_parallel",
):
    """
    Create a comprehensive parallel coordinates plot

    Parameters:
    - data_file: Path to PCA data feather file
    - stats_file: Path to statistical results CSV file
    - n_components: Number of PCA components to show (default: 6)
    - plot_type: 'statistical', 'selected', or 'top_samples'
    - genotypes_of_interest: List of specific genotypes to highlight
    - save_prefix: Prefix for saved files
    """

    # Load data
    print(f"Loading data from {data_file}...")
    pca_data = pd.read_feather(data_file)

    try:
        stats_results = pd.read_csv(stats_file)
        print(f"Statistical results loaded: {len(stats_results)} genotypes tested")
    except FileNotFoundError:
        print("Statistical results not found. Using sample-based selection.")
        stats_results = None

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]
    print(f"Analyzing components: {pca_cols}")

    # Determine genotypes to plot based on plot_type
    if plot_type == "statistical" and stats_results is not None:
        # Prioritize statistically significant genotypes
        significant_all = stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            & (stats_results["Permutation_FDR_significant"] == True)
            & (stats_results["Mahalanobis_FDR_significant"] == True)
        ]["Nickname"].tolist()

        significant_any = stats_results[
            (stats_results["MannWhitney_any_dim_significant"] == True)
            | (stats_results["Permutation_FDR_significant"] == True)
            | (stats_results["Mahalanobis_FDR_significant"] == True)
        ]["Nickname"].tolist()

        # Add top genotypes by sample count
        genotype_counts = pca_data["Nickname"].value_counts()
        top_genotypes = genotype_counts.head(15).index.tolist()

        genotypes_to_plot = list(set(significant_any + top_genotypes))
        title_suffix = "Statistical Significance Highlighted"

    elif plot_type == "selected" and genotypes_of_interest is not None:
        # Use user-specified genotypes
        genotypes_to_plot = genotypes_of_interest
        title_suffix = "Selected Genotypes"

    else:
        # Use top genotypes by sample count
        genotype_counts = pca_data["Nickname"].value_counts()
        genotypes_to_plot = genotype_counts.head(20).index.tolist()
        title_suffix = "Top Genotypes by Sample Count"

    # Filter data
    plot_data = pca_data[pca_data["Nickname"].isin(genotypes_to_plot)]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 1])

    # Main parallel coordinates plot
    ax_main = fig.add_subplot(gs[0, :])

    # Helper function to get line properties
    def get_line_properties(genotype):
        if stats_results is None or genotype not in stats_results["Nickname"].values:
            return "gray", 1.5, "-", 0.7

        row = stats_results[stats_results["Nickname"] == genotype].iloc[0]

        # Determine significance level
        if (
            row["MannWhitney_any_dim_significant"]
            and row["Permutation_FDR_significant"]
            and row["Mahalanobis_FDR_significant"]
        ):
            return "red", 3, "-", 0.9  # Significant in all methods
        elif row["MannWhitney_any_dim_significant"] and row["Permutation_FDR_significant"]:
            return "orange", 2.5, "-", 0.8  # Significant in MW & Permutation
        elif row["MannWhitney_any_dim_significant"]:
            return "blue", 2, "-", 0.8  # Significant in Mann-Whitney
        else:
            return "gray", 1.5, "-", 0.6  # Not significant

    # Plot the lines
    plotted_genotypes = []
    significant_genotypes = []

    for genotype in genotypes_to_plot:
        if genotype in plot_data["Nickname"].values:
            group = plot_data[plot_data["Nickname"] == genotype]
            means = [group[col].mean() for col in pca_cols]
            stds = [group[col].std() for col in pca_cols]
            n_samples = len(group)

            color, linewidth, linestyle, alpha = get_line_properties(genotype)

            # Plot mean line
            ax_main.plot(
                range(len(pca_cols)),
                means,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                label=f"{genotype} (n={n_samples})",
            )

            # Add error bars for significant genotypes
            if color != "gray":
                ax_main.errorbar(
                    range(len(pca_cols)),
                    means,
                    yerr=stds / np.sqrt(n_samples),
                    color=color,
                    alpha=0.3,
                    capsize=2,
                    linewidth=1,
                    zorder=1,
                )
                significant_genotypes.append(genotype)

            plotted_genotypes.append(genotype)

    # Customize main plot
    ax_main.set_xticks(range(len(pca_cols)))
    ax_main.set_xticklabels(pca_cols, fontsize=12)
    ax_main.set_xlabel("PCA Components", fontsize=14)
    ax_main.set_ylabel("PCA Score", fontsize=14)
    ax_main.set_title(f"Parallel Coordinates Plot - PCA Components 1-{n_components}\\n{title_suffix}", fontsize=16)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Add legend for significance levels
    if stats_results is not None:
        legend_elements = [
            mlines.Line2D([0], [0], color="red", linewidth=3, label="Significant in all methods"),
            mlines.Line2D([0], [0], color="orange", linewidth=2.5, label="Significant in MW & Permutation"),
            mlines.Line2D([0], [0], color="blue", linewidth=2, label="Significant in Mann-Whitney"),
            mlines.Line2D([0], [0], color="gray", linewidth=1.5, label="Not significant"),
        ]
        ax_main.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    # Subplot 1: Sample count distribution
    ax_samples = fig.add_subplot(gs[1, 0])
    sample_counts = [len(plot_data[plot_data["Nickname"] == g]) for g in plotted_genotypes]
    bars = ax_samples.bar(range(len(plotted_genotypes)), sample_counts, alpha=0.7)
    ax_samples.set_xticks(range(len(plotted_genotypes)))
    ax_samples.set_xticklabels([g.split("(")[0][:15] for g in plotted_genotypes], rotation=45, ha="right", fontsize=8)
    ax_samples.set_ylabel("Sample Count")
    ax_samples.set_title("Sample Sizes")
    ax_samples.grid(True, alpha=0.3)

    # Subplot 2: Significance statistics
    if stats_results is not None:
        ax_stats = fig.add_subplot(gs[1, 1])
        methods = ["MannWhitney_any_dim_significant", "Permutation_FDR_significant", "Mahalanobis_FDR_significant"]
        method_names = ["Mann-Whitney", "Permutation", "Mahalanobis"]
        counts = [stats_results[method].sum() for method in methods]

        bars = ax_stats.bar(method_names, counts, color=["blue", "green", "orange"], alpha=0.7)
        ax_stats.set_ylabel("Significant Count")
        ax_stats.set_title("Significance by Method")
        ax_stats.grid(True, alpha=0.3)

        # Add counts on bars
        for bar, count in zip(bars, counts):
            ax_stats.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Subplot 3: PCA component variance (approximate)
    ax_variance = fig.add_subplot(gs[2, :])
    component_vars = [pca_data[col].var() for col in pca_cols]
    total_var = np.sum(component_vars)
    explained_ratios = [var / total_var for var in component_vars]

    bars = ax_variance.bar(pca_cols, explained_ratios, alpha=0.7)
    ax_variance.set_ylabel("Relative Variance")
    ax_variance.set_title("Relative Variance Explained by Each Component")
    ax_variance.grid(True, alpha=0.3)

    # Add percentages on bars
    for bar, ratio in zip(bars, explained_ratios):
        ax_variance.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{ratio:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plots
    plt.savefig(f"{save_prefix}_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}_comprehensive.pdf", bbox_inches="tight")

    # Print summary
    print(f"\\nPlot Summary:")
    print(f"- Total genotypes plotted: {len(plotted_genotypes)}")
    print(f"- Significant genotypes: {len(significant_genotypes)}")
    print(f"- Components analyzed: {n_components}")
    print(f"- Plot saved as: {save_prefix}_comprehensive.png/pdf")

    if stats_results is not None:
        print(f"\\nMost significant genotypes:")
        for genotype in significant_genotypes[:10]:  # Show top 10
            print(f"  - {genotype}")

    return fig, plotted_genotypes


def main():
    """Main function with multiple plot options"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Create statistical significance plot
    print("Creating comprehensive statistical plot...")
    fig1, genotypes1 = create_comprehensive_plot(plot_type="statistical", n_components=6, save_prefix="pca_statistical")

    # Create plot with selected genotypes of interest
    interesting_genotypes = [
        "Empty-Split",
        "Empty-Gal4",
        "TNTxPR",
        "MB247-Gal4",
        "LC10-2",
        "LC10-1",
        "LC9",
        "LC11",
        "LC4",
        "LC6",
        "DDC-gal4",
        "ORCO-GAL4",
        "OK107-Gal4",
        "GR63a-GAL4",
        "IR8a-GAL4",
        "IR25a-GAL4",
        "GH146-GAL4",
    ]

    print("\\nCreating selected genotypes plot...")
    fig2, genotypes2 = create_comprehensive_plot(
        plot_type="selected", genotypes_of_interest=interesting_genotypes, n_components=6, save_prefix="pca_selected"
    )

    plt.show()

    print("\\nAll plots created successfully!")


if __name__ == "__main__":
    main()
