#!/usr/bin/env python3
"""
PCA Boxplots with Significance Annotations
This script creates boxplots with superimposed scatterplots for each PC component,
showing controls vs hits with significance annotations based on statistical test results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os
import ast
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config


def load_static_data_and_results():
    """Load static PCA data and statistical results"""
    try:
        pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")
        print("Static PCA data loaded successfully")
    except FileNotFoundError:
        print("Static PCA data file not found.")
        return None, None

    try:
        static_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
        print("Static statistical results loaded successfully")
    except FileNotFoundError:
        print("Static statistical results file not found.")
        static_results = None

    return pca_data, static_results


def load_temporal_data_and_results():
    """Load temporal PCA data and statistical results"""
    try:
        pca_data = pd.read_feather("fpca_temporal_with_metadata_tailoredctrls.feather")
        print("Temporal PCA data loaded successfully")
    except FileNotFoundError:
        print("Temporal PCA data file not found.")
        return None, None

    try:
        temporal_results = pd.read_csv("fpca_temporal_stats_results_allmethods_tailoredctrls.csv")
        print("Temporal statistical results loaded successfully")
    except FileNotFoundError:
        print("Temporal statistical results file not found.")
        temporal_results = None

    return pca_data, temporal_results


def get_brain_region_colors():
    """Get the color dictionary from Config"""
    return Config.color_dict


def get_significant_genotypes(stats_results, significance_threshold=0.05):
    """Get significant genotypes from statistical results"""
    if stats_results is None:
        return set()

    significant_genotypes = set()
    for _, row in stats_results.iterrows():
        if row["Permutation_pval"] < significance_threshold:
            significant_genotypes.add(row["Nickname"])

    return significant_genotypes


def get_mannwhitney_significance(stats_results):
    """Get Mann-Whitney significance for individual PC components from statistical results"""
    if stats_results is None:
        return {}

    genotype_pc_significance = {}

    for _, row in stats_results.iterrows():
        genotype = row["Nickname"]

        # Parse the MannWhitney_significant_dims column
        try:
            if pd.isna(row["MannWhitney_significant_dims"]) or row["MannWhitney_significant_dims"] == "[]":
                significant_dims = []
            else:
                significant_dims = ast.literal_eval(row["MannWhitney_significant_dims"])
        except (ValueError, SyntaxError):
            significant_dims = []

        genotype_pc_significance[genotype] = set(significant_dims)

    return genotype_pc_significance


def create_pc_boxplots(pca_data, stats_results, analysis_type="static", n_components=6):
    """
    Create boxplots for each PC showing individual genotypes (hits and controls) with significance annotations

    Parameters:
    - pca_data: DataFrame with PCA scores and metadata
    - stats_results: DataFrame with statistical results
    - analysis_type: "static" or "temporal"
    - n_components: Number of PCA components to plot
    """

    # Get available PCA components based on analysis type
    if analysis_type == "temporal":
        pca_cols = [f"FPCA{i+1}" for i in range(n_components)]
    else:
        pca_cols = [f"PCA{i+1}" for i in range(n_components)]

    available_pca_cols = [col for col in pca_cols if col in pca_data.columns]

    print(f"Creating {analysis_type} boxplots for components: {available_pca_cols}")

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()

    # Get significant genotypes (for filtering)
    significant_genotypes = get_significant_genotypes(stats_results)
    print(f"Found {len(significant_genotypes)} significant genotypes")

    # Get Mann-Whitney significance for individual PC components
    mannwhitney_significance = get_mannwhitney_significance(stats_results)

    # Get controls
    control_genotypes = pca_data[pca_data["Brain region"] == "Control"]["Nickname"].unique()
    print(f"Found {len(control_genotypes)} control genotypes: {list(control_genotypes)}")

    # Combine significant hits and controls for plotting
    all_genotypes = list(control_genotypes) + list(significant_genotypes)

    # Create figure with subplots
    n_pcs = len(available_pca_cols)
    if n_pcs == 0:
        print(f"No {analysis_type} PCA components found!")
        return None

    fig, axes = plt.subplots(n_pcs, 1, figsize=(max(16, len(all_genotypes) * 0.8), 4 * n_pcs))

    # Handle single subplot case
    if n_pcs == 1:
        axes = [axes]

    for i, pc in enumerate(available_pca_cols):
        ax = axes[i]

        # Prepare data for this PC
        plot_data = []
        x_positions = []
        x_labels = []
        colors = []
        is_significant_for_pc = []

        for idx, genotype in enumerate(all_genotypes):
            genotype_data = pca_data[pca_data["Nickname"] == genotype]
            if len(genotype_data) > 0:
                # Get brain region and color
                brain_region = genotype_data["Brain region"].iloc[0]
                color = brain_region_colors.get(brain_region, "#000000")

                # Check if this genotype is significant for this specific PC
                pc_significant = pc in mannwhitney_significance.get(genotype, set())

                # Add data for boxplot
                values = genotype_data[pc].values
                plot_data.append(values)
                x_positions.append(idx)
                x_labels.append(genotype)
                colors.append(color)
                is_significant_for_pc.append(pc_significant)

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f"No data for {pc}", ha="center", va="center", transform=ax.transAxes)
            continue

        # Create boxplot with no fill and black lines
        bp = ax.boxplot(
            plot_data,
            positions=x_positions,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
            medianprops=dict(color="black", linewidth=1.5),
        )

        # Fill boxes with brain region color if significant for this PC
        for patch, color, is_sig in zip(bp["boxes"], colors, is_significant_for_pc):
            if is_sig:
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
            else:
                patch.set_facecolor("none")

        # Add individual points with jitter, colored by brain region
        np.random.seed(42)  # For reproducible jitter

        for idx, (genotype, values, color) in enumerate(zip(all_genotypes, plot_data, colors)):
            if len(values) > 0:
                jitter = np.random.normal(0, 0.1, len(values))
                ax.scatter(
                    np.full(len(values), idx) + jitter,
                    values,
                    c=color,
                    alpha=0.7,
                    s=25,
                    marker="o",
                    edgecolor="black",
                    linewidth=0.3,
                )

        # Customize axis
        ax.set_title(f"{pc} ({analysis_type.capitalize()})", fontsize=14, fontweight="bold")
        ax.set_ylabel("PC Score", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Set x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # Add sample size annotations
        y_min = min([min(vals) for vals in plot_data if len(vals) > 0])
        y_max = max([max(vals) for vals in plot_data if len(vals) > 0])
        y_range = y_max - y_min

        for idx, (genotype, values) in enumerate(zip(all_genotypes, plot_data)):
            ax.text(idx, y_min - 0.05 * y_range, f"n={len(values)}", ha="center", va="top", fontsize=8, alpha=0.7)

        # Add vertical line to separate controls from hits
        if len(control_genotypes) > 0:
            separation_line = len(control_genotypes) - 0.5
            ax.axvline(x=separation_line, color="gray", linestyle="--", alpha=0.5, linewidth=1)

            # Add labels for control and hits sections
            control_center = (len(control_genotypes) - 1) / 2
            hits_center = len(control_genotypes) + (len(significant_genotypes) - 1) / 2

            ax.text(
                control_center,
                y_max + 0.05 * y_range,
                "Controls",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                alpha=0.7,
            )
            if len(significant_genotypes) > 0:
                ax.text(
                    hits_center,
                    y_max + 0.05 * y_range,
                    "Hits",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    alpha=0.7,
                )

    # Create legends
    # Brain region legend
    brain_region_handles = []
    all_regions = set()

    # Add controls
    for control_genotype in control_genotypes:
        control_data = pca_data[pca_data["Nickname"] == control_genotype]
        if len(control_data) > 0:
            brain_region = control_data["Brain region"].iloc[0]
            all_regions.add(brain_region)

    # Add hits
    for genotype in significant_genotypes:
        if genotype in pca_data["Nickname"].values:
            brain_region = pca_data[pca_data["Nickname"] == genotype]["Brain region"].iloc[0]
            all_regions.add(brain_region)

    for brain_region in sorted(all_regions):
        color = brain_region_colors.get(brain_region, "#000000")
        brain_region_handles.append(
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=brain_region,
            )
        )

    # Significance legend
    sig_handles = [
        mpatches.Patch(color="gray", alpha=0.4, label="Significant for this PC/FPC (Mann-Whitney)"),
        mpatches.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", label="Not significant for this PC/FPC"),
    ]

    # Add legends
    if brain_region_handles:
        brain_legend = fig.legend(
            handles=brain_region_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), title="Brain Regions", ncol=1
        )
        fig.add_artist(brain_legend)

    sig_legend = fig.legend(
        handles=sig_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), title="Significance", ncol=1
    )

    # Add title and adjust layout
    plt.suptitle(
        f"{analysis_type.capitalize()} PCA Component Boxplots by Genotype\n"
        f"{len(control_genotypes)} Controls, {len(significant_genotypes)} Significant Hits",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)

    return fig


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load static data
    print("Loading static PCA data...")
    static_data, static_results = load_static_data_and_results()

    if static_data is not None:
        print(f"Static PCA data shape: {static_data.shape}")
        print(f"Static unique brain regions: {sorted(static_data['Brain region'].unique())}")

        # Create static boxplots
        print("\nCreating static PCA boxplots...")
        static_fig = create_pc_boxplots(static_data, static_results, analysis_type="static", n_components=6)

        # Save static plots
        if static_fig is not None:
            static_fig.savefig("static_pca_boxplots_with_significance.png", dpi=300, bbox_inches="tight")
            static_fig.savefig("static_pca_boxplots_with_significance.pdf", bbox_inches="tight")
            print("Static plots saved as 'static_pca_boxplots_with_significance.png/pdf'")
        else:
            print("Failed to create static plots")

    # Load temporal data
    print("\nLoading temporal PCA data...")
    temporal_data, temporal_results = load_temporal_data_and_results()

    if temporal_data is not None:
        print(f"Temporal PCA data shape: {temporal_data.shape}")
        print(f"Temporal unique brain regions: {sorted(temporal_data['Brain region'].unique())}")

        # Create temporal boxplots
        print("\nCreating temporal PCA boxplots...")
        temporal_fig = create_pc_boxplots(temporal_data, temporal_results, analysis_type="temporal", n_components=6)

        # Save temporal plots
        if temporal_fig is not None:
            temporal_fig.savefig("temporal_pca_boxplots_with_significance.png", dpi=300, bbox_inches="tight")
            temporal_fig.savefig("temporal_pca_boxplots_with_significance.pdf", bbox_inches="tight")
            print("Temporal plots saved as 'temporal_pca_boxplots_with_significance.png/pdf'")
        else:
            print("Failed to create temporal plots")

    # Show plots
    plt.show()

    print("\nBoxplot generation complete!")


if __name__ == "__main__":
    main()
