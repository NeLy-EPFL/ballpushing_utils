#!/usr/bin/env python3
"""
Script to generate metrics heatmaps from Mann-Whitney U test results for dissected experiments.

This script creates two types of heatmaps:
1. Clustered heatmap with dendrograms showing hierarchical relationships
2. Simple heatmap sorted by significance categories

The heatmaps visualize Mann-Whitney U test results comparing dissected vs non-dissected groups,
with color intensity representing statistical significance and direction of effect.

Usage:
    python plot_dissected_metrics_heatmap.py
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
        Path to the dissected_mannwhitney_statistics.csv file

    Returns:
    --------
    pd.DataFrame
        Statistics DataFrame with columns: Metric, Control, Test, pval, direction, etc.
    """
    stats_df = pd.read_csv(stats_file)
    print(f"Loaded statistics from: {stats_file}")
    print(f"  Shape: {stats_df.shape}")
    print(f"  Metrics: {stats_df['Metric'].nunique()}")
    print(f"  Test group: {stats_df['Test'].iloc[0]}")
    print(f"  Control group: {stats_df['Control'].iloc[0]}")

    return stats_df


def build_significance_matrix(stats_df):
    """
    Build a matrix of significance values from Mann-Whitney statistics.

    For dissected experiments, we only have 2 groups (dissected vs non-dissected),
    so the matrix will be 1 row × N metrics.

    The matrix values represent both direction and significance using continuous gradient:
    - Sign based on effect_size (test_median - control_median)
    - Magnitude: -log10(p_value) / 3.0 for continuous gradient

    This preserves information about near-significant effects:
    - p=0.001 → weight=1.0 (dark)
    - p=0.01 → weight=0.67 (medium-dark)
    - p=0.05 → weight=0.43 (medium)
    - p=0.1 → weight=0.33 (light)
    - p=0.5 → weight=0.1 (very light)

    Parameters:
    -----------
    stats_df : pd.DataFrame
        Statistics DataFrame from Mann-Whitney tests

    Returns:
    --------
    pd.DataFrame
        Matrix with groups as rows and metrics as columns
    """
    # Get unique groups and metrics
    test_group = stats_df["Test"].iloc[0]  # Should be "dissected"
    control_group = stats_df["Control"].iloc[0]  # Should be "non-dissected"
    metrics = sorted(stats_df["Metric"].unique())

    print(f"\nBuilding significance matrix:")
    print(f"  Test group: {test_group}")
    print(f"  Control group: {control_group}")
    print(f"  Metrics: {len(metrics)}")

    # Initialize matrix - only 1 row for the test group
    matrix = pd.DataFrame(index=[test_group], columns=metrics, dtype=float)

    # Fill matrix with continuous gradient for all p-values
    for _, row in stats_df.iterrows():
        metric = row["Metric"]
        pval = row["pval"]
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

        matrix.loc[test_group, metric] = sign * weight

    # Convert to numeric
    matrix = matrix.astype(float)

    # Print summary statistics
    print(f"\n  Matrix summary:")
    print(f"    Shape: {matrix.shape}")
    print(f"    Non-zero values: {(matrix != 0).sum().sum()}")
    print(f"    Positive values (increased): {(matrix > 0.01).sum().sum()}")
    print(f"    Negative values (decreased): {(matrix < -0.01).sum().sum()}")
    print(f"    Value range: [{matrix.min().min():.3f}, {matrix.max().max():.3f}]")

    return matrix


def plot_simple_heatmap(matrix, stats_df, output_dir, filename_prefix="dissected"):
    """
    Create a simple heatmap sorted by significance categories (no dendrogram).

    For dissected experiments with only 1 row, this creates a clear visualization
    of all metrics showing their significance and direction.

    Parameters:
    -----------
    matrix : pd.DataFrame
        Significance matrix (1 row × metrics)
    stats_df : pd.DataFrame
        Original statistics for sorting
    output_dir : Path
        Directory to save the plot
    filename_prefix : str
        Prefix for output filename
    """
    print(f"\nCreating simple sorted heatmap...")

    n_groups, n_metrics = matrix.shape

    # Sort metrics by their significance and direction
    metric_scores = {}
    for metric in matrix.columns:
        value = matrix[metric].iloc[0]
        abs_value = abs(value)

        # Sort by: 1) significance strength (abs value), 2) direction (positive first)
        metric_scores[metric] = (abs_value, value)

    # Sort metrics: most significant first, then by direction
    sorted_metrics = sorted(
        metric_scores.keys(),
        key=lambda m: (-metric_scores[m][0], -metric_scores[m][1]),
    )

    # Reorder matrix
    matrix_ordered = matrix[sorted_metrics]

    # Create figure
    fig_height = max(4, n_groups * 1.5)
    fig_width = max(12, n_metrics * 0.5)

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
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=11)

    # Create custom colorbar
    from matplotlib.patches import Rectangle
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Add colorbar axes on the right
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

    norm = Normalize(vmin=-1, vmax=1)
    cbar = ColorbarBase(cbar_ax, cmap=cm.RdBu_r, norm=norm, orientation="vertical")
    cbar.set_label("Significance & Direction", fontsize=10)
    cbar.set_ticks([-1, -0.67, -0.43, 0, 0.43, 0.67, 1])
    cbar.set_ticklabels(["***\n(lower)", "**", "*", "ns", "*", "**", "***\n(higher)"])

    # Title
    test_group = matrix_ordered.index[0]
    control_group = stats_df["Control"].iloc[0]

    plt.suptitle(
        f"Dissected Experiments: Mann-Whitney Significance Heatmap\n"
        f"{test_group} vs {control_group} ({n_metrics} metrics, sorted by significance)",
        fontsize=14,
        x=0.5,
        y=0.98,
    )

    ax_main.set_xlabel("Metrics", fontsize=12)
    ax_main.set_ylabel("Group", fontsize=12)

    # Save
    output_file_png = output_dir / f"{filename_prefix}_heatmap_simple.png"
    output_file_pdf = output_dir / f"{filename_prefix}_heatmap_simple.pdf"

    plt.savefig(output_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved simple heatmap:")
    print(f"    {output_file_png}")
    print(f"    {output_file_pdf}")


def plot_metrics_barplot(matrix, stats_df, output_dir, filename_prefix="dissected"):
    """
    Create a horizontal barplot showing significance for all metrics.

    This is particularly useful for single-group comparisons where a heatmap
    would only have one row.

    Parameters:
    -----------
    matrix : pd.DataFrame
        Significance matrix (1 row × metrics)
    stats_df : pd.DataFrame
        Original statistics for annotations
    output_dir : Path
        Directory to save the plot
    filename_prefix : str
        Prefix for output filename
    """
    print(f"\nCreating metrics barplot...")

    # Extract values and sort by absolute value (most significant first)
    values = matrix.iloc[0].to_dict()
    sorted_metrics = sorted(values.keys(), key=lambda m: abs(values[m]), reverse=True)
    sorted_values = [values[m] for m in sorted_metrics]

    # Create figure
    fig_height = max(8, len(sorted_metrics) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Color bars by direction and significance
    colors = []
    for val in sorted_values:
        if val > 0.43:  # Significantly increased
            colors.append("#d73027")  # Red
        elif val > 0:  # Trending increased
            colors.append("#fc8d59")  # Light red
        elif val < -0.43:  # Significantly decreased
            colors.append("#4575b4")  # Blue
        elif val < 0:  # Trending decreased
            colors.append("#91bfdb")  # Light blue
        else:
            colors.append("#cccccc")  # Gray for no change

    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_metrics))
    bars = ax.barh(y_pos, sorted_values, color=colors, alpha=0.8)

    # Add significance stars
    for i, (metric, val) in enumerate(zip(sorted_metrics, sorted_values)):
        abs_val = abs(val)
        if abs_val >= 1.0:
            stars = "***"
        elif abs_val >= 0.67:
            stars = "**"
        elif abs_val >= 0.43:
            stars = "*"
        else:
            stars = ""

        if stars:
            x_pos = val + (0.05 if val > 0 else -0.05)
            ax.text(
                x_pos,
                i,
                stars,
                ha="left" if val > 0 else "right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_metrics, fontsize=9)
    ax.set_xlabel("Signed -log10(p-value)", fontsize=12)
    ax.set_ylabel("Metrics", fontsize=12)

    # Add vertical line at x=0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    # Add significance threshold lines
    ax.axvline(x=0.43, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.43, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Set x-axis limits
    max_abs = max(abs(min(sorted_values)), abs(max(sorted_values)))
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

    # Grid
    ax.grid(axis="x", alpha=0.3)

    # Title
    test_group = matrix.index[0]
    control_group = stats_df["Control"].iloc[0]

    ax.set_title(
        f"Dissected Experiments: Metric Significance Barplot\n"
        f"{test_group} vs {control_group} (sorted by significance)",
        fontsize=14,
        pad=20,
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d73027", label="Significantly Higher (p<0.05)"),
        Patch(facecolor="#fc8d59", label="Trending Higher (p≥0.05)"),
        Patch(facecolor="#91bfdb", label="Trending Lower (p≥0.05)"),
        Patch(facecolor="#4575b4", label="Significantly Lower (p<0.05)"),
        Patch(facecolor="none", label=""),
        Patch(facecolor="none", label="Significance: * p<0.05, ** p<0.01, *** p<0.001"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    # Save
    output_file_png = output_dir / f"{filename_prefix}_metrics_barplot.png"
    output_file_pdf = output_dir / f"{filename_prefix}_metrics_barplot.pdf"

    plt.savefig(output_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved metrics barplot:")
    print(f"    {output_file_png}")
    print(f"    {output_file_pdf}")


def main():
    """Main function to generate heatmaps from Mann-Whitney statistics."""
    print("=" * 80)
    print("DISSECTED EXPERIMENTS METRICS HEATMAP GENERATION")
    print("=" * 80)

    # Define paths
    base_dir = Path("/mnt/upramdya_data/MD/Antennae_dissection/Plots/Summary/Dissected_Mannwhitney")
    stats_file = base_dir / "dissected_mannwhitney_statistics.csv"
    output_dir = base_dir / "heatmaps"

    # Check if statistics file exists
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        print("Please run the Mann-Whitney analysis first (run_mannwhitney_dissected.py).")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load statistics
    stats_df = load_mannwhitney_statistics(stats_file)

    # Build significance matrix
    matrix = build_significance_matrix(stats_df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Simple sorted heatmap
    plot_simple_heatmap(matrix, stats_df, output_dir, filename_prefix="dissected")

    # Barplot (better for single-group comparison)
    plot_metrics_barplot(matrix, stats_df, output_dir, filename_prefix="dissected")

    print("\n" + "=" * 80)
    print("✅ HEATMAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
