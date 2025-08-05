#!/usr/bin/env python3
"""
Scatterplot Matrix for PCA Results
This script creates a pairwise scatterplot matrix showing all combinations of PCA components
with brain region color coding and significance highlighting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os
from itertools import combinations
from scipy import stats
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse, Circle
from sklearn.covariance import EmpiricalCovariance

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
    """Extract significance information from permutation test p-values"""
    if stats_results is None:
        return {}

    genotype_significance = {}

    for _, row in stats_results.iterrows():
        genotype = row["Nickname"]

        # Use raw permutation test p-values (same as heatmap)
        permutation_pval = row["Permutation_pval"]
        is_significant = permutation_pval < significance_threshold

        genotype_significance[genotype] = {
            "is_significant": is_significant,
            "permutation_pval": permutation_pval,
            "min_pval": permutation_pval,
            "test_type": "permutation",
        }

    return genotype_significance


def get_brain_region_colors():
    """Get the color dictionary from Config"""
    return Config.color_dict


def plot_control_convex_hull(ax, control_x, control_y, alpha=0.2, color="lightgray"):
    """Plot convex hull around control points"""
    if len(control_x) < 3:  # Need at least 3 points for convex hull
        return None

    points = np.column_stack([control_x, control_y])
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon

        ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha, label="Control Hull")
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.5, linewidth=1)
        return hull
    except:
        return None


def plot_control_confidence_ellipse(ax, control_x, control_y, n_std=2, alpha=0.2, color="lightblue"):
    """Plot confidence ellipse around control points"""
    if len(control_x) < 3:  # Need at least 3 points
        return None

    points = np.column_stack([control_x, control_y])

    # Calculate covariance matrix
    cov = EmpiricalCovariance().fit(points)
    mean = np.mean(points, axis=0)

    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov.covariance_)

    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])

    # Create ellipse
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle, facecolor=color, alpha=alpha, edgecolor=color, linewidth=1
    )
    ax.add_patch(ellipse)

    return ellipse


def plot_all_flies_confidence_ellipse(ax, all_x, all_y, confidence=0.95, alpha=0.2, color="lightblue"):
    """Plot confidence ellipse using provided data points"""
    if len(all_x) < 3:  # Need at least 3 points
        return None

    points = np.column_stack([all_x, all_y])

    # Calculate covariance matrix
    cov = EmpiricalCovariance().fit(points)
    mean = np.mean(points, axis=0)

    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov.covariance_)

    # Calculate confidence interval multiplier for given confidence level
    from scipy.stats import chi2

    chi2_val = chi2.ppf(confidence, df=2)

    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * np.sqrt(chi2_val * eigenvals[0])
    height = 2 * np.sqrt(chi2_val * eigenvals[1])

    # Create ellipse
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle, facecolor=color, alpha=alpha, edgecolor=color, linewidth=1
    )
    ax.add_patch(ellipse)

    return ellipse


def plot_control_ellipses_by_line(ax, pca_data, pc1, pc2, confidence=0.95, alpha=0.2):
    """Plot separate confidence ellipses for each control line"""
    control_lines = {
        "Empty-Split": "#ffb7b7",  # Light gray
        "Empty-Gal4": "#C6F7B0",  # Medium gray
        "TNTxPR": "#B4EAFF",  # Dark gray
    }

    ellipses = {}

    for control_name, color in control_lines.items():
        # Get data for this specific control line
        control_data = pca_data[pca_data["Nickname"] == control_name]

        if len(control_data) >= 3:  # Need at least 3 points
            control_x = control_data[pc2].values
            control_y = control_data[pc1].values

            # Plot confidence ellipse for this control line
            ellipse = plot_all_flies_confidence_ellipse(
                ax, control_x, control_y, confidence=confidence, alpha=alpha, color=color
            )
            ellipses[control_name] = ellipse

            # Plot individual control points
            ax.scatter(
                control_x, control_y, c=color, alpha=0.6, s=20, marker="x", label=f"{control_name} points", linewidth=1
            )

    return ellipses


def plot_control_percentile_area(ax, control_x, control_y, percentile=95, alpha=0.2, color="lightgreen"):
    """Plot area containing specified percentile of control points"""
    if len(control_x) < 5:  # Need enough points for percentile calculation
        return None

    # Calculate distances from center
    center_x, center_y = np.mean(control_x), np.mean(control_y)
    distances = np.sqrt((control_x - center_x) ** 2 + (control_y - center_y) ** 2)

    # Find radius that contains the specified percentile
    radius = np.percentile(distances, percentile)

    # Draw circle
    circle = Circle(
        (float(center_x), float(center_y)), float(radius), facecolor=color, alpha=alpha, edgecolor=color, linewidth=1
    )
    ax.add_patch(circle)

    return circle


def create_scatterplot_matrix(
    pca_data,
    static_results=None,
    temporal_results=None,
    n_components=6,
    significance_threshold=0.05,
    save_prefix="pca_scatterplot_matrix",
):
    """
    Create scatterplot matrix showing all pairwise combinations of PCA components

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

    # Get significance information
    static_significance = get_significance_from_permutation(static_results, significance_threshold)
    temporal_significance = get_significance_from_permutation(temporal_results, significance_threshold)

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()

    # Filter to genotypes with sufficient samples
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 5].index.tolist()

    # Get significant genotypes
    static_significant = [
        g
        for g in static_significance.keys()
        if static_significance[g]["is_significant"] and g in genotypes_with_samples
    ]
    temporal_significant = [
        g
        for g in temporal_significance.keys()
        if temporal_significance[g]["is_significant"] and g in genotypes_with_samples
    ]

    # Combine all significant genotypes
    all_significant = list(set(static_significant + temporal_significant))

    print(f"Static significant genotypes: {len(static_significant)}")
    print(f"Temporal significant genotypes: {len(temporal_significant)}")
    print(f"Total unique significant genotypes: {len(all_significant)}")

    # Create figure with subplots
    n_pcs = len(available_pca_cols)
    fig, axes = plt.subplots(n_pcs, n_pcs, figsize=(4 * n_pcs, 4 * n_pcs))

    # Plot each pairwise combination
    for i, pc1 in enumerate(available_pca_cols):
        for j, pc2 in enumerate(available_pca_cols):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show histogram/density
                # Plot controls first
                control_data = pca_data[pca_data["Brain region"] == "Control"]
                if len(control_data) > 0:
                    ax.hist(control_data[pc1], bins=20, alpha=0.3, color="lightgray", label="Control", density=True)

                # Plot significant genotypes
                for genotype in all_significant:
                    if genotype in pca_data["Nickname"].values:
                        group = pca_data[pca_data["Nickname"] == genotype]
                        brain_region = group["Brain region"].iloc[0]
                        color = brain_region_colors.get(brain_region, "#000000")

                        # Different styles for static vs temporal
                        if genotype in static_significant and genotype in temporal_significant:
                            alpha = 0.8
                            linestyle = "-"
                        elif genotype in static_significant:
                            alpha = 0.6
                            linestyle = "--"
                        else:  # temporal only
                            alpha = 0.6
                            linestyle = ":"

                        ax.hist(
                            group[pc1],
                            bins=20,
                            alpha=alpha,
                            color=color,
                            histtype="step",
                            linewidth=2,
                            linestyle=linestyle,
                            density=True,
                        )

                ax.set_xlabel(pc1)
                ax.set_ylabel("Density")

            else:
                # Off-diagonal: scatter plot
                # Plot separate confidence ellipses for each control line
                plot_control_ellipses_by_line(ax, pca_data, pc1, pc2, confidence=0.95, alpha=0.2)

                # Plot significant genotypes
                for genotype in all_significant:
                    if genotype in pca_data["Nickname"].values:
                        group = pca_data[pca_data["Nickname"] == genotype]
                        brain_region = group["Brain region"].iloc[0]
                        color = brain_region_colors.get(brain_region, "#000000")

                        # Different markers and sizes for static vs temporal
                        if genotype in static_significant and genotype in temporal_significant:
                            # Both: circle, larger
                            marker = "o"
                            size = 60
                            alpha = 0.8
                            edgecolor = "black"
                            linewidth = 1
                        elif genotype in static_significant:
                            # Static only: square
                            marker = "s"
                            size = 50
                            alpha = 0.7
                            edgecolor = "black"
                            linewidth = 0.5
                        else:  # temporal only
                            # Temporal only: triangle
                            marker = "^"
                            size = 50
                            alpha = 0.7
                            edgecolor = "black"
                            linewidth = 0.5

                        # Plot mean point for the genotype
                        mean_pc1 = group[pc1].mean()
                        mean_pc2 = group[pc2].mean()

                        ax.scatter(
                            mean_pc2,
                            mean_pc1,
                            c=color,
                            s=size,
                            alpha=alpha,
                            marker=marker,
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                        )

                ax.set_xlabel(pc2)
                ax.set_ylabel(pc1)

            # Add grid
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Create custom legends
    # Brain region legend
    brain_region_handles = []
    all_regions = set()
    for genotype in all_significant:
        if genotype in pca_data["Nickname"].values:
            brain_region = pca_data[pca_data["Nickname"] == genotype]["Brain region"].iloc[0]
            all_regions.add(brain_region)

    for brain_region in sorted(all_regions):
        color = brain_region_colors.get(brain_region, "#000000")
        brain_region_handles.append(
            mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=brain_region)
        )

    # Add control to brain region legend
    brain_region_handles.append(
        mlines.Line2D([0], [0], marker="x", color="w", markerfacecolor="gray", markersize=8, label="Control Points")
    )

    # Control lines legend - separate ellipse for each control line
    control_area_handles = [
        mpatches.Patch(color="#ffb7b7", alpha=0.2, label="Empty-Split (95% CI)"),
        mpatches.Patch(color="#C6F7B0", alpha=0.2, label="Empty-Gal4 (95% CI)"),
        mpatches.Patch(color="#B4EAFF", alpha=0.2, label="TNTxPR (95% CI)"),
    ]

    # Significance type legend
    sig_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Both Static & Temporal",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Static Only",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Temporal Only",
        ),
    ]

    # Add legends
    brain_legend = fig.legend(
        handles=brain_region_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), title="Brain Regions", ncol=1
    )
    sig_legend = fig.legend(
        handles=sig_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), title="Significance Type", ncol=1
    )
    control_legend = fig.legend(
        handles=control_area_handles, loc="lower left", bbox_to_anchor=(0.02, 0.02), title="Control Areas", ncol=1
    )

    # Add all legends to the figure
    fig.add_artist(brain_legend)
    fig.add_artist(sig_legend)

    plt.suptitle(
        f"PCA Scatterplot Matrix\n{len(all_significant)} Significant Genotypes\nSeparate 95% confidence ellipses for each control line",
        fontsize=16,
        y=0.95,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save plots
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")

    # Print summary
    print(f"\n=== Scatterplot Matrix Summary ===")
    print(f"Total significant genotypes plotted: {len(all_significant)}")
    print(f"Static only: {len(set(static_significant) - set(temporal_significant))}")
    print(f"Temporal only: {len(set(temporal_significant) - set(static_significant))}")
    print(f"Both static and temporal: {len(set(static_significant) & set(temporal_significant))}")

    # Print brain region breakdown
    print(f"\n=== Brain Region Breakdown ===")
    region_stats = {}
    for genotype in all_significant:
        if genotype in pca_data["Nickname"].values:
            brain_region = pca_data[pca_data["Nickname"] == genotype]["Brain region"].iloc[0]
            if brain_region not in region_stats:
                region_stats[brain_region] = 0
            region_stats[brain_region] += 1

    for region, count in sorted(region_stats.items()):
        print(f"{region}: {count} genotypes")

    return fig, all_significant


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

    # Create scatterplot matrix
    print("\nCreating PCA scatterplot matrix...")
    fig, significant_genotypes = create_scatterplot_matrix(
        pca_data,
        static_results,
        temporal_results,
        n_components=6,  # Use first 6 components
        significance_threshold=0.05,
        save_prefix="pca_scatterplot_matrix",
    )

    plt.show()

    print("\nPlot saved as 'pca_scatterplot_matrix.png/pdf'")


if __name__ == "__main__":
    main()
