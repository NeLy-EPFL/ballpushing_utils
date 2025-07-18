#!/usr/bin/env python3
"""
Simple parallel coordinates plot for PCA results
This script creates a focused parallel coordinates visualization for PCA components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os


def create_simple_parallel_plot(
    data_file, n_components=5, genotypes_to_highlight=None, save_name="parallel_coordinates_simple"
):
    """
    Create a simple parallel coordinates plot

    Parameters:
    - data_file: Path to the feather file with PCA data
    - n_components: Number of PCA components to show
    - genotypes_to_highlight: List of genotypes to highlight, or None for automatic selection
    - save_name: Name for saved files
    """

    # Load data
    pca_data = pd.read_feather(data_file)

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]

    # Get genotypes with sufficient samples
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 10].index.tolist()

    # Filter data to genotypes with sufficient samples
    plot_data = pca_data[pca_data["Nickname"].isin(genotypes_with_samples)]

    # Select genotypes to highlight
    if genotypes_to_highlight is None:
        # Select top 15 genotypes by sample count
        genotypes_to_highlight = genotype_counts.head(15).index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette for highlighted genotypes
    colors = sns.color_palette("husl", len(genotypes_to_highlight))
    color_map = dict(zip(genotypes_to_highlight, colors))

    # Plot all genotypes in light gray first
    for genotype, group in plot_data.groupby("Nickname"):
        if genotype not in genotypes_to_highlight:
            means = [group[col].mean() for col in pca_cols]
            ax.plot(range(len(pca_cols)), means, color="lightgray", alpha=0.5, linewidth=1, zorder=1)

    # Plot highlighted genotypes
    for genotype in genotypes_to_highlight:
        if genotype in plot_data["Nickname"].values:
            group = plot_data[plot_data["Nickname"] == genotype]
            means = [group[col].mean() for col in pca_cols]
            stds = [group[col].std() for col in pca_cols]

            # Plot mean line
            ax.plot(
                range(len(pca_cols)),
                means,
                color=color_map[genotype],
                linewidth=2.5,
                label=genotype,
                alpha=0.8,
                zorder=2,
            )

            # Add error bars
            ax.errorbar(
                range(len(pca_cols)),
                means,
                yerr=stds,
                color=color_map[genotype],
                alpha=0.4,
                capsize=3,
                linewidth=1,
                zorder=1,
            )

    # Customize plot
    ax.set_xticks(range(len(pca_cols)))
    ax.set_xticklabels(pca_cols, fontsize=12)
    ax.set_xlabel("PCA Components", fontsize=14)
    ax.set_ylabel("PCA Score", fontsize=14)
    ax.set_title(f"Parallel Coordinates Plot - First {n_components} PCA Components", fontsize=16)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save plot
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_name}.pdf", bbox_inches="tight")

    # Print summary
    print(f"Plot created with {len(genotypes_to_highlight)} highlighted genotypes")
    print(f"Total genotypes in dataset: {plot_data['Nickname'].nunique()}")
    print(f"Highlighted genotypes: {genotypes_to_highlight}")

    return fig, ax


def create_interactive_selection():
    """Interactive function to select genotypes and create plot"""

    # Load data to show available genotypes
    pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")

    # Show top genotypes by sample count
    genotype_counts = pca_data["Nickname"].value_counts()
    print("Top 20 genotypes by sample count:")
    print(genotype_counts.head(20))

    # Show some interesting genotypes (controls and common ones)
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

    available_interesting = [g for g in interesting_genotypes if g in pca_data["Nickname"].values]
    print(f"\nInteresting genotypes available: {available_interesting}")

    return pca_data, genotype_counts


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Check if data file exists
    data_file = "static_pca_with_metadata_tailoredctrls.feather"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run PCA_Static.py first.")
        return

    # Create a plot with automatic selection
    print("Creating parallel coordinates plot with automatic genotype selection...")
    fig1, ax1 = create_simple_parallel_plot(data_file, n_components=6, save_name="parallel_coordinates_auto")

    # Create a plot with specific genotypes of interest
    interesting_genotypes = [
        "Empty-Split",
        "Empty-Gal4",
        "TNTxPR",
        "MB247-Gal4",
        "LC10-2",
        "LC10-1",
        "LC9",
        "DDC-gal4",
        "OK107-Gal4",
        "GR63a-GAL4",
        "IR8a-GAL4",
        "IR25a-GAL4",
        "GH146-GAL4",
    ]

    print("\nCreating parallel coordinates plot with selected genotypes...")
    fig2, ax2 = create_simple_parallel_plot(
        data_file,
        n_components=6,
        genotypes_to_highlight=interesting_genotypes,
        save_name="parallel_coordinates_selected",
    )

    # Show available genotypes for user reference
    print("\n" + "=" * 50)
    print("Available genotypes summary:")
    pca_data, genotype_counts = create_interactive_selection()

    plt.show()


if __name__ == "__main__":
    main()
