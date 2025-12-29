#!/usr/bin/env python3
"""
Script to generate metrics heatmaps from Mann-Whitney U test results for learning mutants experiments.

This script creates two types of heatmaps:
1. Clustered heatmap with dendrograms showing hierarchical relationships
2. Simple heatmap sorted by significance categories

The heatmaps visualize Mann-Whitney U test results comparing each genotype against the control,
with color intensity representing statistical significance and direction of effect.

Usage:
    python plot_learning_mutants_metrics_heatmap.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from matplotlib.gridspec import GridSpec


def load_mannwhitney_statistics(stats_file):
    """
    Load Mann-Whitney U test statistics from CSV file.

    Parameters:
    -----------
    stats_file : str or Path
        Path to the genotype_mannwhitney_statistics.csv file

    Returns:
    --------
    pd.DataFrame
        Statistics DataFrame with columns: Metric, Control, Test, pval_corrected, direction, etc.
    """
    stats_df = pd.read_csv(stats_file)
    print(f"Loaded statistics from: {stats_file}")
    print(f"  Shape: {stats_df.shape}")
    print(f"  Metrics: {stats_df['Metric'].nunique()}")
    print(f"  Test genotypes: {sorted(stats_df['Test'].unique())}")
    print(f"  Control genotype: {stats_df['Control'].iloc[0]}")

    return stats_df


def build_significance_matrix(stats_df):
    """
    Build a matrix of significance values from Mann-Whitney statistics.

    The matrix values represent both direction and significance:
    - Positive values: metric increased in test vs control
    - Negative values: metric decreased in test vs control
    - Magnitude: statistical significance (larger = more significant)

    Encoding:
    - p < 0.001: weight = 1.0
    - 0.001 <= p < 0.05: weight = -log10(p) / 3.0 (ranges from ~0.43 to 1.0)
    - p >= 0.05: weight = 0.1 (minimal, non-significant)

    Parameters:
    -----------
    stats_df : pd.DataFrame
        Statistics DataFrame from Mann-Whitney tests

    Returns:
    --------
    pd.DataFrame
        Matrix with genotypes as rows and metrics as columns
    """
    # Get unique genotypes and metrics
    genotypes = sorted(stats_df["Test"].unique())
    metrics = sorted(stats_df["Metric"].unique())

    print(f"\nBuilding significance matrix:")
    print(f"  Genotypes: {len(genotypes)}")
    print(f"  Metrics: {len(metrics)}")

    # Initialize matrix
    matrix = pd.DataFrame(index=genotypes, columns=metrics, dtype=float)

    # Fill matrix with continuous gradient for all p-values
    for _, row in stats_df.iterrows():
        genotype = row["Test"]
        metric = row["Metric"]
        pval = row["pval_corrected"]
        effect_size = row["effect_size"]

        # Use continuous -log10 gradient for all p-values (not just significant)
        # This preserves information about near-significant effects
        # p=0.001 → weight=1.0, p=0.01 → 0.67, p=0.05 → 0.43, p=0.1 → 0.33, p=0.5 → 0.1
        pval = max(float(pval), 1e-10)  # Avoid log(0)
        log_p = -np.log10(pval)
        weight = min(log_p / 3.0, 1.0)  # Normalize so p=0.001 gives weight=1.0

        # Determine sign from effect_size (test_median - control_median)
        # This preserves direction even for non-significant results
        if effect_size > 0:
            sign = 1
        elif effect_size < 0:
            sign = -1
        else:
            sign = 0

        matrix.loc[genotype, metric] = sign * weight

    # Convert to numeric
    matrix = matrix.astype(float)

    # Print summary statistics
    print(f"\n  Matrix summary:")
    print(f"    Shape: {matrix.shape}")
    print(f"    Non-zero values: {(matrix != 0).sum().sum()}")
    print(f"    Positive values (increased): {(matrix > 0.1).sum().sum()}")
    print(f"    Negative values (decreased): {(matrix < -0.1).sum().sum()}")
    print(f"    Value range: [{matrix.min().min():.3f}, {matrix.max().max():.3f}]")

    return matrix


def plot_two_way_dendrogram(matrix, output_dir, filename_prefix="learning_mutants"):
    """
    Create a two-way clustered heatmap with dendrograms.

    Parameters:
    -----------
    matrix : pd.DataFrame
        Significance matrix (genotypes × metrics)
    output_dir : Path
        Directory to save the plot
    filename_prefix : str
        Prefix for output filename
    """
    print(f"\nCreating two-way dendrogram heatmap...")

    n_genotypes, n_metrics = matrix.shape
    print(f"  Matrix dimensions: {n_genotypes} genotypes × {n_metrics} metrics")

    # Calculate linkage for metrics (columns) - correlate metrics across genotypes
    metric_corr = matrix.corr()  # Correlate metrics (columns)
    print(f"  Metric correlation matrix shape: {metric_corr.shape}")

    # Convert correlation to distance
    metric_dist = 1 - metric_corr.abs()  # Distance = 1 - |correlation|

    # Handle numerical precision issues - clip to valid range and handle NaN/inf
    metric_dist = np.maximum(metric_dist, 0.0)
    metric_dist = np.nan_to_num(metric_dist, nan=1.0, posinf=1.0, neginf=0.0)

    # Convert to condensed distance matrix for linkage
    metric_dist_condensed = squareform(metric_dist, checks=False)

    # Additional safety check for finite values
    if not np.all(np.isfinite(metric_dist_condensed)):
        print("  ⚠️  Warning: Non-finite values detected in metric distance matrix, replacing with 1.0")
        metric_dist_condensed = np.nan_to_num(metric_dist_condensed, nan=1.0, posinf=1.0, neginf=0.0)

    metric_linkage = linkage(metric_dist_condensed, method="average")

    # Calculate linkage for genotypes (rows) if we have enough
    if n_genotypes > 1:
        genotype_corr = matrix.T.corr()  # Correlate genotypes (rows)
        genotype_dist = 1 - genotype_corr.abs()
        genotype_dist = np.maximum(genotype_dist, 0.0)
        genotype_dist = np.nan_to_num(genotype_dist, nan=1.0, posinf=1.0, neginf=0.0)
        genotype_dist_condensed = squareform(genotype_dist, checks=False)

        # Additional safety check
        if not np.all(np.isfinite(genotype_dist_condensed)):
            print("  ⚠️  Warning: Non-finite values detected in genotype distance matrix, replacing with 1.0")
            genotype_dist_condensed = np.nan_to_num(genotype_dist_condensed, nan=1.0, posinf=1.0, neginf=0.0)

        genotype_linkage = linkage(genotype_dist_condensed, method="average")
    else:
        genotype_linkage = None

    # Get dendrogram order
    metric_dendro = dendrogram(metric_linkage, no_plot=True)
    metric_order = metric_dendro["leaves"]

    if genotype_linkage is not None:
        genotype_dendro = dendrogram(genotype_linkage, no_plot=True)
        genotype_order = genotype_dendro["leaves"]
    else:
        genotype_order = [0]

    # Reorder matrix
    matrix_ordered = matrix.iloc[genotype_order, metric_order]

    # Create figure with GridSpec
    fig_height = max(8, n_genotypes * 1.0 + 4)
    fig_width = max(12, n_metrics * 0.7)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: [top dendro, spacing, heatmap]
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[2.0, 1.2, 5.0],
        width_ratios=[1.0, 8.0, 0.3],
        hspace=0.02,
        wspace=0.05,
    )

    # Axes
    ax_dendro_top = fig.add_subplot(gs[0, 1])  # Top dendrogram
    ax_main = fig.add_subplot(gs[2, 1])  # Heatmap
    ax_cbar = fig.add_subplot(gs[2, 2])  # Colorbar

    if genotype_linkage is not None:
        ax_dendro_left = fig.add_subplot(gs[2, 0])  # Left dendrogram

    # Plot dendrograms
    dendrogram(metric_linkage, ax=ax_dendro_top, orientation="top", no_labels=True, color_threshold=0)
    ax_dendro_top.axis("off")

    if genotype_linkage is not None:
        dendrogram(genotype_linkage, ax=ax_dendro_left, orientation="left", no_labels=True, color_threshold=0)
        ax_dendro_left.axis("off")

    # Plot heatmap
    im = ax_main.imshow(
        matrix_ordered.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        extent=[-0.5, n_metrics - 0.5, n_genotypes - 0.5, -0.5],
    )

    # Add significance stars
    for i in range(matrix_ordered.shape[0]):
        for j in range(matrix_ordered.shape[1]):
            value = matrix_ordered.iloc[i, j]
            abs_value = abs(value)

            # Determine significance level based on weight
            if abs_value >= 1.0:
                stars = "***"
            elif abs_value >= 0.43:
                stars = "**" if abs_value >= 0.67 else "*"
            else:
                stars = ""

            if stars:
                ax_main.text(
                    j,
                    i,
                    stars,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )

    # Set ticks and labels
    ax_main.set_xticks(range(n_metrics))
    ax_main.set_yticks(range(n_genotypes))
    ax_main.set_xticklabels(matrix_ordered.columns, rotation=45, ha="right", fontsize=9)
    ax_main.set_yticklabels(matrix_ordered.index, fontsize=9)

    ax_main.set_xlabel("Metrics", fontsize=12)
    ax_main.set_ylabel("Genotypes", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Significance & Direction", fontsize=10)
    cbar.set_ticks([-1, -0.67, -0.43, 0, 0.43, 0.67, 1])
    cbar.set_ticklabels(["***", "**", "*", "ns", "*", "**", "***"])

    # Title
    fig.suptitle(
        f"Learning Mutants: Mann-Whitney Significance Heatmap (Clustered)\n"
        f"{n_genotypes} genotypes × {n_metrics} metrics",
        fontsize=14,
        y=0.98,
    )

    # Save
    output_file_png = output_dir / f"{filename_prefix}_heatmap_clustered.png"
    output_file_pdf = output_dir / f"{filename_prefix}_heatmap_clustered.pdf"

    plt.savefig(output_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved clustered heatmap:")
    print(f"    {output_file_png}")
    print(f"    {output_file_pdf}")


def plot_simple_heatmap(matrix, stats_df, output_dir, filename_prefix="learning_mutants"):
    """
    Create a simple heatmap sorted by significance categories (no dendrogram).

    Genotypes are sorted by:
    1. Significantly increased metrics (most to least)
    2. Non-significant
    3. Significantly decreased metrics (least to most)

    Parameters:
    -----------
    matrix : pd.DataFrame
        Significance matrix (genotypes × metrics)
    stats_df : pd.DataFrame
        Original statistics for sorting
    output_dir : Path
        Directory to save the plot
    filename_prefix : str
        Prefix for output filename
    """
    print(f"\nCreating simple sorted heatmap...")

    n_genotypes, n_metrics = matrix.shape

    # Sort metrics by their overall significance pattern
    # Count significant results per metric
    metric_significance = {}
    for metric in matrix.columns:
        sig_increased = (matrix[metric] >= 0.43).sum()
        sig_decreased = (matrix[metric] <= -0.43).sum()
        total_sig = sig_increased + sig_decreased

        # Use total significance as primary sort, then ratio of increased vs decreased
        metric_significance[metric] = {
            "total_sig": total_sig,
            "increased": sig_increased,
            "decreased": sig_decreased,
            "net": sig_increased - sig_decreased,
        }

    # Sort metrics: most significant first, then by net direction
    sorted_metrics = sorted(
        metric_significance.keys(),
        key=lambda m: (
            -metric_significance[m]["total_sig"],  # More significant first
            -metric_significance[m]["net"],  # More increased first
        ),
    )

    # Reorder matrix
    matrix_ordered = matrix[sorted_metrics]

    # Create figure
    fig_height = max(6, n_genotypes * 0.8)
    fig_width = max(10, n_metrics * 0.5)

    fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))

    # Heatmap
    sns.heatmap(
        matrix_ordered,
        ax=ax_main,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=True,
        yticklabels=True,
    )

    # Add significance stars
    for i in range(matrix_ordered.shape[0]):
        for j in range(matrix_ordered.shape[1]):
            value = matrix_ordered.iloc[i, j]
            abs_value = abs(value)

            # Determine significance level based on weight
            # weight = 1.0 for p < 0.001, ~0.43-1.0 for p < 0.05
            if abs_value >= 1.0:
                stars = "***"
            elif abs_value >= 0.43:  # approximately p < 0.05
                stars = "**" if abs_value >= 0.67 else "*"
            else:
                stars = ""

            if stars:
                ax_main.text(
                    j + 0.5,
                    i + 0.5,
                    stars,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )

    # Labels
    ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=9)

    # Create custom colorbar
    from matplotlib.patches import Rectangle
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Add colorbar axes on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    norm = Normalize(vmin=-1, vmax=1)
    cbar = ColorbarBase(cbar_ax, cmap=cm.RdBu_r, norm=norm, orientation="vertical")
    cbar.set_label("Significance & Direction", fontsize=10)
    cbar.set_ticks([-1, -0.67, -0.43, 0, 0.43, 0.67, 1])
    cbar.set_ticklabels(["***", "**", "*", "ns", "*", "**", "***"])

    # Title
    plt.suptitle(
        f"Learning Mutants: Mann-Whitney Significance Heatmap\n"
        f"{n_genotypes} genotypes × {n_metrics} metrics (sorted by significance)",
        fontsize=14,
        x=0.5,
        y=0.98,
    )

    ax_main.set_xlabel("Metrics", fontsize=12)
    ax_main.set_ylabel("Genotypes", fontsize=12)

    # Save
    output_file_png = output_dir / f"{filename_prefix}_heatmap_simple.png"
    output_file_pdf = output_dir / f"{filename_prefix}_heatmap_simple.pdf"

    plt.savefig(output_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved simple heatmap:")
    print(f"    {output_file_png}")
    print(f"    {output_file_pdf}")


def main():
    """Main function to generate heatmaps from Mann-Whitney statistics."""
    print("=" * 80)
    print("LEARNING MUTANTS METRICS HEATMAP GENERATION")
    print("=" * 80)

    # Define paths
    base_dir = Path("/mnt/upramdya_data/MD/Learning_mutants/Plots/Summary_metrics/Genotype_Mannwhitney")
    stats_file = base_dir / "genotype_mannwhitney_statistics.csv"
    output_dir = base_dir / "heatmaps"

    # Check if statistics file exists
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        print("Please run the Mann-Whitney analysis first.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load statistics
    stats_df = load_mannwhitney_statistics(stats_file)

    # Build significance matrix
    matrix = build_significance_matrix(stats_df)

    # Generate heatmaps
    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS")
    print("=" * 80)

    # Two-way dendrogram heatmap
    plot_two_way_dendrogram(matrix, output_dir, filename_prefix="learning_mutants")

    # Simple sorted heatmap
    plot_simple_heatmap(matrix, stats_df, output_dir, filename_prefix="learning_mutants")

    print("\n" + "=" * 80)
    print("✅ HEATMAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
