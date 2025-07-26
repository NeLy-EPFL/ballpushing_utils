#!/usr/bin/env python3
"""
Interactive Scatterplot Matrix for Individual Fly PCA Results using HoloViews
This script creates an interactive pairwise scatterplot matrix showing all combinations of PCA components
with individual fly data points instead of genotype averages.
"""

import pandas as pd
import numpy as np
import sys
import os
from itertools import combinations
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
import colorcet as cc
from bokeh.palettes import Category10, all_palettes
from scipy.spatial import ConvexHull
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2

# Enable Bokeh backend for HoloViews
hv.extension("bokeh")

# Add the parent directory to the path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config

# Plot configuration templates
hv_main = {
    "scatter": {
        "size": 6,  # Smaller points for individual flies
        "alpha": 0.4,  # Further reduced alpha for individual flies (more crowded)
        "line_color": "black",
        "line_width": 0.3,
    },
    "plot": {
        "width": 400,
        "height": 400,
        "show_grid": True,
        "fontscale": 1.2,
        "tools": ["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        "active_tools": ["wheel_zoom"],
    },
    "ellipse": {
        "line_width": 2,
        "alpha": 0.2,
    },
    "layout": {
        "shared_axes": False,
        "merge_tools": True,
    },
}


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


def get_control_assignment(nickname, pca_data=None):
    """Get the control assignment for a given genotype nickname"""
    try:
        # If we have the PCA data, we can look up the Split value directly
        if pca_data is not None and "Split" in pca_data.columns:
            nickname_data = pca_data[pca_data["Nickname"] == nickname]
            if len(nickname_data) > 0:
                split_value = nickname_data["Split"].iloc[0]
            else:
                # Fallback to SplitRegistry lookup
                split_data = Config.SplitRegistry[Config.SplitRegistry["Nickname"] == nickname]
                if len(split_data) > 0:
                    split_value = split_data["Split"].iloc[0]
                else:
                    return "Unknown"
        else:
            # Fallback to SplitRegistry lookup
            split_data = Config.SplitRegistry[Config.SplitRegistry["Nickname"] == nickname]
            if len(split_data) > 0:
                split_value = split_data["Split"].iloc[0]
            else:
                return "Unknown"

        # Map Split values to readable control names
        split_mapping = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}

        return split_mapping.get(split_value, f"Unknown ({split_value})")

    except (KeyError, AttributeError, IndexError) as e:
        print(f"Warning: Could not determine control assignment for {nickname}: {e}")
        return "Unknown"


def create_confidence_ellipse_data(data_x, data_y, confidence=0.95):
    """Create ellipse data points for plotting confidence intervals"""
    if len(data_x) < 3:
        return None, None

    points = np.column_stack([data_x, data_y])

    # Calculate covariance matrix
    cov = EmpiricalCovariance().fit(points)
    mean = np.mean(points, axis=0)

    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov.covariance_)

    # Calculate confidence interval multiplier for given confidence level
    chi2_val = chi2.ppf(confidence, df=2)

    # Calculate ellipse parameters
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    width = 2 * np.sqrt(chi2_val * eigenvals[0])
    height = 2 * np.sqrt(chi2_val * eigenvals[1])

    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_points = np.column_stack([width / 2 * np.cos(theta), height / 2 * np.sin(theta)])

    # Rotate ellipse
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    ellipse_points = ellipse_points @ rotation_matrix.T
    ellipse_points += mean

    return ellipse_points[:, 0], ellipse_points[:, 1]


def create_control_ellipses(pca_data, pc_x, pc_y, confidence=0.95):
    """Create confidence ellipses for each control line"""
    control_lines = {
        "Empty-Split": "#ffb7b7",
        "Empty-Gal4": "#C6F7B0",
        "TNTxPR": "#B4EAFF",
    }

    ellipse_curves = []

    for control_name, color in control_lines.items():
        # Get data for this specific control line
        control_data = pca_data[pca_data["Nickname"] == control_name]

        if len(control_data) >= 3:
            control_x = control_data[pc_x].values
            control_y = control_data[pc_y].values

            # Create ellipse data
            ellipse_x, ellipse_y = create_confidence_ellipse_data(control_x, control_y, confidence=confidence)

            if ellipse_x is not None:
                # Create ellipse curve
                ellipse_df = pd.DataFrame({pc_x: ellipse_x, pc_y: ellipse_y, "control_line": control_name})

                ellipse_curve = hv.Curve(ellipse_df, kdims=[pc_x], vdims=[pc_y]).opts(
                    color=color, line_width=2, alpha=0.8, line_dash="dashed"
                )
                ellipse_curves.append(ellipse_curve)

    return ellipse_curves


def prepare_individual_fly_data(pca_data, static_significance, temporal_significance, genotypes_with_samples):
    """Prepare individual fly data for plotting with significance markers and metadata"""

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

    # Filter data to significant genotypes only
    significant_data = pca_data[pca_data["Nickname"].isin(all_significant)].copy()

    # Rename Brain region column to avoid issues with spaces in hover tooltips
    significant_data = significant_data.rename(columns={"Brain region": "Brain_region"})

    # Add significance type information
    def get_significance_type(nickname):
        if nickname in static_significant and nickname in temporal_significant:
            return "Both Static & Temporal"
        elif nickname in static_significant:
            return "Static Only"
        elif nickname in temporal_significant:
            return "Temporal Only"
        else:
            return "Not Significant"

    def get_marker_style(nickname):
        if nickname in static_significant and nickname in temporal_significant:
            return "circle"  # Both
        elif nickname in static_significant:
            return "square"  # Static only
        elif nickname in temporal_significant:
            return "triangle"  # Temporal only
        else:
            return "circle"

    def get_marker_size(nickname):
        if nickname in static_significant and nickname in temporal_significant:
            return 8  # Both - slightly larger
        else:
            return 6  # Single significance type

    significant_data["significance_type"] = significant_data["Nickname"].apply(get_significance_type)
    significant_data["marker_style"] = significant_data["Nickname"].apply(get_marker_style)
    significant_data["marker_size"] = significant_data["Nickname"].apply(get_marker_size)

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()
    significant_data["brain_region_color"] = significant_data["Brain_region"].map(brain_region_colors)

    # Add control assignment information using the Split column from the data
    def get_control_from_split(split_value):
        split_mapping = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}
        return split_mapping.get(split_value, f"Unknown ({split_value})")

    significant_data["control_assignment"] = significant_data["Split"].apply(get_control_from_split)

    # Add statistical significance values
    static_pvals = {
        g: static_significance[g]["permutation_pval"] for g in static_significance.keys() if g in all_significant
    }
    temporal_pvals = {
        g: temporal_significance[g]["permutation_pval"] for g in temporal_significance.keys() if g in all_significant
    }

    significant_data["static_pval"] = significant_data["Nickname"].map(static_pvals).fillna(1.0)
    significant_data["temporal_pval"] = significant_data["Nickname"].map(temporal_pvals).fillna(1.0)

    # Create fly identifier for better tracking
    significant_data["fly_id"] = significant_data["Nickname"] + "_" + significant_data["fly"].astype(str)

    return significant_data, all_significant, static_significant, temporal_significant


def create_interactive_scatterplot_individual(
    fly_data, pca_data, pc_x_col, pc_y_col, pc_x_label, pc_y_label, plot_options=hv_main
):
    """Create an interactive scatterplot for individual flies"""

    # Define comprehensive hover tooltips
    hover_tooltips = [
        # ('Fly ID', '@fly_id'),
        ("Genotype", "@Nickname"),
        ("Brain Region", "@Brain_region"),
        ("Control", "@control_assignment"),
        ("Significance", "@significance_type"),
        ("Fly ID", "@fly"),
        # ("Experiment", "@experiment"),
        (f"{pc_x_label}", f"@{pc_x_col}{{0.000}}"),
        (f"{pc_y_label}", f"@{pc_y_col}{{0.000}}"),
        ("Static p-val", "@static_pval{0.0000}"),
        ("Temporal p-val", "@temporal_pval{0.0000}"),
    ]

    hover = HoverTool(tooltips=hover_tooltips)

    # Create separate scatter plots for each significance type to control marker shapes
    plot_elements = []

    # Both static and temporal - circles (slightly larger)
    both_data = fly_data[fly_data["significance_type"] == "Both Static & Temporal"]
    if len(both_data) > 0:
        scatter_both = hv.Scatter(
            both_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "fly_id",
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "fly",
                "experiment",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
        ).opts(
            color="brain_region_color",
            size=8,  # Slightly larger for both
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=0.5,
            marker="circle",
            tools=[hover],
        )
        plot_elements.append(scatter_both)

    # Static only - squares
    static_data = fly_data[fly_data["significance_type"] == "Static Only"]
    if len(static_data) > 0:
        scatter_static = hv.Scatter(
            static_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "fly_id",
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "fly",
                "experiment",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
        ).opts(
            color="brain_region_color",
            size=6,
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=0.3,
            marker="square",
            tools=[hover],
        )
        plot_elements.append(scatter_static)

    # Temporal only - triangles
    temporal_data = fly_data[fly_data["significance_type"] == "Temporal Only"]
    if len(temporal_data) > 0:
        scatter_temporal = hv.Scatter(
            temporal_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "fly_id",
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "fly",
                "experiment",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
        ).opts(
            color="brain_region_color",
            size=6,
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=0.3,
            marker="triangle",
            tools=[hover],
        )
        plot_elements.append(scatter_temporal)

    # Create control ellipses (based on all data, not just individual flies)
    ellipse_curves = create_control_ellipses(pca_data, pc_x_col, pc_y_col)
    if ellipse_curves:
        plot_elements.extend(ellipse_curves)

    # Add reference lines at zero
    zero_h = hv.HLine(0).opts(color="black", alpha=0.3, line_width=1)
    zero_v = hv.VLine(0).opts(color="black", alpha=0.3, line_width=1)

    plot_elements.extend([zero_h, zero_v])

    # Create the overlay and apply common options
    plot = hv.Overlay(plot_elements).opts(
        width=plot_options["plot"]["width"],
        height=plot_options["plot"]["height"],
        show_grid=plot_options["plot"]["show_grid"],
        fontscale=plot_options["plot"]["fontscale"],
        xlabel=pc_x_label,
        ylabel=pc_y_label,
        tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        active_tools=["wheel_zoom"],
    )

    return plot


def create_interactive_histogram_individual(fly_data, pc_col, pc_label, plot_options=hv_main):
    """Create an interactive histogram for individual flies"""

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()

    # Get unique brain regions in the data
    unique_regions = sorted(fly_data["Brain_region"].unique())

    # Calculate appropriate bins for the histogram
    pc_values = fly_data[pc_col].values
    n_bins = max(8, min(15, int(np.sqrt(len(pc_values)) * 1.5)))  # More bins for individual data

    # Create common bin edges for all histograms
    bin_edges = np.linspace(pc_values.min(), pc_values.max(), n_bins + 1)

    # Create histograms for each brain region using the same bin edges
    plot_elements = []

    for brain_region in unique_regions:
        region_data = fly_data[fly_data["Brain_region"] == brain_region]
        if len(region_data) > 0:
            color = brain_region_colors.get(brain_region, "#000000")

            # Create histogram for this brain region with consistent bins
            counts, edges = np.histogram(region_data[pc_col].values, bins=bin_edges)

            hist = hv.Histogram((edges, counts), kdims=[pc_label], vdims=["Frequency"]).opts(
                fill_color=color,
                fill_alpha=0.35,  # Increased alpha between previous (0.2) and original (0.5)
                line_color=color,
                line_width=2,
                # Removed line_dash to use normal solid lines
                tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
            )
            plot_elements.append(hist)

    # Add reference line at zero
    zero_v = hv.VLine(0).opts(color="black", alpha=0.5, line_width=2)
    plot_elements.append(zero_v)

    # Create the overlay
    plot = hv.Overlay(plot_elements).opts(
        width=plot_options["plot"]["width"],
        height=plot_options["plot"]["height"],
        show_grid=plot_options["plot"]["show_grid"],
        fontscale=plot_options["plot"]["fontscale"],
        xlabel=pc_label,
        ylabel="Frequency",
        tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        show_legend=False,
    )

    return plot


def create_interactive_scatterplot_matrix_individual(
    pca_data,
    static_results=None,
    temporal_results=None,
    n_components=6,
    significance_threshold=0.05,
    plot_options=hv_main,
    save_prefix="interactive_pca_scatterplot_matrix_individual",
):
    """
    Create interactive scatterplot matrix for individual flies using HoloViews
    """

    # Select PCA components
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]  # Internal column names
    pc_labels = [f"PC{i+1}" for i in range(n_components)]  # Display labels
    available_pca_cols = [col for col in pca_cols if col in pca_data.columns]
    available_pc_labels = [f"PC{i+1}" for i, col in enumerate(pca_cols) if col in pca_data.columns]
    print(f"Using PCA components: {available_pca_cols} (displayed as {available_pc_labels})")

    # Get significance information
    static_significance = get_significance_from_permutation(static_results, significance_threshold)
    temporal_significance = get_significance_from_permutation(temporal_results, significance_threshold)

    # Filter to genotypes with sufficient samples
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 5].index.tolist()

    # Prepare individual fly data
    fly_data, all_significant, static_significant, temporal_significant = prepare_individual_fly_data(
        pca_data, static_significance, temporal_significance, genotypes_with_samples
    )

    print(f"Static significant genotypes: {len(static_significant)}")
    print(f"Temporal significant genotypes: {len(temporal_significant)}")
    print(f"Total unique significant genotypes: {len(all_significant)}")
    print(f"Total individual flies plotted: {len(fly_data)}")

    # Create matrix of plots
    plots = {}
    n_pcs = len(available_pca_cols)

    for i, (pc_y_col, pc_y_label) in enumerate(zip(available_pca_cols, available_pc_labels)):
        for j, (pc_x_col, pc_x_label) in enumerate(zip(available_pca_cols, available_pc_labels)):

            if i == j:
                # Diagonal: show distribution as histogram
                plot = create_interactive_histogram_individual(fly_data, pc_x_col, pc_x_label, plot_options)
                plots[(i, j)] = plot
            else:
                # Off-diagonal: scatter plot
                plot = create_interactive_scatterplot_individual(
                    fly_data, pca_data, pc_x_col, pc_y_col, pc_x_label, pc_y_label, plot_options
                )
                plots[(i, j)] = plot

    # Create layout as a proper matrix
    # Create a list of all plots in row-major order for matrix layout
    all_plots = []
    for i in range(n_pcs):
        for j in range(n_pcs):
            all_plots.append(plots[(i, j)])

    # Create matrix layout
    matrix_layout = hv.Layout(all_plots).cols(n_pcs).opts(**plot_options["layout"])

    final_layout = matrix_layout

    # Print summary
    print(f"\n=== Individual Fly Scatterplot Matrix Summary ===")
    print(f"Total significant genotypes plotted: {len(all_significant)}")
    print(f"Static only: {len(set(static_significant) - set(temporal_significant))}")
    print(f"Temporal only: {len(set(temporal_significant) - set(static_significant))}")
    print(f"Both static and temporal: {len(set(static_significant) & set(temporal_significant))}")

    # Print brain region breakdown
    print(f"\n=== Brain Region Breakdown (Individual Flies) ===")
    region_stats = fly_data["Brain_region"].value_counts().to_dict()
    for region, count in sorted(region_stats.items()):
        print(f"{region}: {count} flies")

    return final_layout, fly_data, all_significant, static_significant, temporal_significant


def save_interactive_plot(plot, filename):
    """Save the interactive plot as HTML"""
    hv.save(plot, filename)
    print(f"Interactive plot saved as '{filename}'")


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

    # Create interactive scatterplot matrix for individual flies
    print("\nCreating interactive PCA scatterplot matrix for individual flies...")

    # Choose plot style
    plot_style = hv_main

    final_layout, fly_data, all_significant, static_significant, temporal_significant = (
        create_interactive_scatterplot_matrix_individual(
            pca_data,
            static_results,
            temporal_results,
            n_components=6,  # Use first 6 components
            significance_threshold=0.05,
            plot_options=plot_style,
            save_prefix="interactive_pca_scatterplot_matrix_individual",
        )
    )

    # Save the interactive plot
    save_interactive_plot(final_layout, "interactive_pca_scatterplot_matrix_individual.html")

    print("Files saved:")
    print("- Individual flies plot: interactive_pca_scatterplot_matrix_individual.html")
    print("- Note: Use the legend from the main script (pca_legend_combined.png)")

    return final_layout


if __name__ == "__main__":
    matrix_layout = main()
