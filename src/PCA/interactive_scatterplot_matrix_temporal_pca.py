#!/usr/bin/env python3
"""
Interactive Scatterplot Matrix for Temporal PCA (fPCA) Results using HoloViews
This script creates an interactive pairwise scatterplot matrix showing all combinations of temporal PCA components
with brain region color coding, significance highlighting, and hover tooltips with metadata.
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
        "size": 8,
        "alpha": 0.5,  # Reduced alpha for better visibility when crowded
        "line_color": "black",
        "line_width": 0.5,
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

hv_presentation = {
    "scatter": {
        "size": 12,
        "alpha": 0.8,
        "line_color": "black",
        "line_width": 1,
    },
    "plot": {
        "width": 500,
        "height": 500,
        "show_grid": True,
        "fontscale": 1.5,
        "tools": ["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        "active_tools": ["wheel_zoom"],
    },
    "ellipse": {
        "line_width": 3,
        "alpha": 0.3,
    },
    "layout": {
        "shared_axes": False,
        "merge_tools": True,
    },
}


def load_temporal_data_with_brain_regions():
    """Load temporal PCA (fPCA) data and statistical results"""
    # Load temporal PCA score data
    pca_data = pd.read_feather("fpca_temporal_with_metadata_tailoredctrls.feather")

    # Load static results (for comparison/additional context)
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


def prepare_plotting_data(pca_data, static_significance, temporal_significance, genotypes_with_samples):
    """Prepare data for plotting with significance markers and metadata"""

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
            return 12  # Both - larger
        else:
            return 10  # Single significance type

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

    # Group by genotype and calculate means for plotting - use FPCA columns
    fpca_cols = [col for col in significant_data.columns if col.startswith("FPCA")]
    genotype_means = (
        significant_data.groupby(
            [
                "Nickname",
                "Brain_region",
                "significance_type",
                "marker_style",
                "marker_size",
                "brain_region_color",
                "control_assignment",
            ]
        )
        .agg({col: "mean" for col in fpca_cols})
        .reset_index()
    )

    # Add additional metadata for tooltips
    genotype_counts = significant_data.groupby("Nickname").size().reset_index(name="n_flies")
    genotype_means = genotype_means.merge(genotype_counts, on="Nickname")

    # Add statistical significance values
    static_pvals = {
        g: static_significance[g]["permutation_pval"] for g in static_significance.keys() if g in all_significant
    }
    temporal_pvals = {
        g: temporal_significance[g]["permutation_pval"] for g in temporal_significance.keys() if g in all_significant
    }

    genotype_means["static_pval"] = genotype_means["Nickname"].map(static_pvals).fillna(1.0)
    genotype_means["temporal_pval"] = genotype_means["Nickname"].map(temporal_pvals).fillna(1.0)

    return genotype_means, all_significant, static_significant, temporal_significant


def create_interactive_scatterplot(
    genotype_means,
    pca_data,
    pc_x_col,
    pc_y_col,
    pc_x_label,
    pc_y_label,
    plot_options=hv_main,
    show_legend=False,
    legend_type=None,
):
    """Create an interactive scatterplot for a pair of temporal PCA components"""

    # Define comprehensive hover tooltips
    hover_tooltips = [
        ("Genotype", "@Nickname"),
        ("Brain Region", "@Brain_region"),  # Fixed: use underscore instead of space
        ("Control", "@control_assignment"),
        ("Significance", "@significance_type"),
        ("N Flies", "@n_flies"),
        (f"{pc_x_label}", f"@{pc_x_col}{{0.000}}"),
        (f"{pc_y_label}", f"@{pc_y_col}{{0.000}}"),
        ("Static p-val", "@static_pval{0.0000}"),
        ("Temporal p-val", "@temporal_pval{0.0000}"),
    ]

    hover = HoverTool(tooltips=hover_tooltips)

    # Create separate scatter plots for each significance type to control marker shapes
    plot_elements = []

    # Both static and temporal - circles (larger)
    both_data = genotype_means[genotype_means["significance_type"] == "Both Static & Temporal"]
    if len(both_data) > 0:
        scatter_both = hv.Scatter(
            both_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "n_flies",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
            label="Both Static & Temporal" if show_legend else "",
        ).opts(
            color="brain_region_color",
            size=12,  # Larger for both
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=1,
            marker="circle",
            tools=[hover],
        )
        plot_elements.append(scatter_both)

    # Static only - squares
    static_data = genotype_means[genotype_means["significance_type"] == "Static Only"]
    if len(static_data) > 0:
        scatter_static = hv.Scatter(
            static_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "n_flies",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
            label="Static Only" if show_legend else "",
        ).opts(
            color="brain_region_color",
            size=10,
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=0.5,
            marker="square",
            tools=[hover],
        )
        plot_elements.append(scatter_static)

    # Temporal only - triangles
    temporal_data = genotype_means[genotype_means["significance_type"] == "Temporal Only"]
    if len(temporal_data) > 0:
        scatter_temporal = hv.Scatter(
            temporal_data,
            kdims=[pc_x_col],
            vdims=[
                pc_y_col,
                "Nickname",
                "Brain_region",
                "control_assignment",
                "significance_type",
                "n_flies",
                "brain_region_color",
                "marker_style",
                "marker_size",
                "static_pval",
                "temporal_pval",
            ],
            label="Temporal Only" if show_legend else "",
        ).opts(
            color="brain_region_color",
            size=10,
            alpha=plot_options["scatter"]["alpha"],
            line_color="black",
            line_width=0.5,
            marker="triangle",
            tools=[hover],
        )
        plot_elements.append(scatter_temporal)

    # Create control ellipses
    ellipse_curves = create_control_ellipses(pca_data, pc_x_col, pc_y_col)
    if ellipse_curves:
        plot_elements.extend(ellipse_curves)

    # Add reference lines at zero
    zero_h = hv.HLine(0).opts(color="black", alpha=0.3, line_width=1)
    zero_v = hv.VLine(0).opts(color="black", alpha=0.3, line_width=1)

    plot_elements.extend([zero_h, zero_v])

    # Create the overlay and apply common options
    plot_opts = {
        "width": plot_options["plot"]["width"],
        "height": plot_options["plot"]["height"],
        "show_grid": plot_options["plot"]["show_grid"],
        "fontscale": plot_options["plot"]["fontscale"],
        "xlabel": pc_x_label,
        "ylabel": pc_y_label,
        "tools": ["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        "active_tools": ["wheel_zoom"],
        "show_legend": show_legend,
    }

    if show_legend:
        plot_opts["legend_position"] = "right"

    plot = hv.Overlay(plot_elements).opts(**plot_opts)

    return plot


def create_interactive_histogram(genotype_means, pc_col, pc_label, plot_options=hv_main, show_legend=False):
    """Create an interactive histogram for a single temporal PCA component using overlapping histograms with better styling"""

    # Get brain region colors
    brain_region_colors = get_brain_region_colors()

    # Get unique brain regions in the data
    unique_regions = sorted(genotype_means["Brain_region"].unique())

    # Calculate appropriate bins for the histogram
    pc_values = genotype_means[pc_col].values
    n_bins = max(6, min(12, int(np.sqrt(len(pc_values)))))  # Fewer bins for cleaner display

    # Create common bin edges for all histograms
    bin_edges = np.linspace(pc_values.min(), pc_values.max(), n_bins + 1)

    # Create histograms for each brain region using the same bin edges
    plot_elements = []

    for brain_region in unique_regions:
        region_data = genotype_means[genotype_means["Brain_region"] == brain_region]
        if len(region_data) > 0:
            color = brain_region_colors.get(brain_region, "#000000")

            # Create histogram for this brain region with consistent bins
            counts, edges = np.histogram(region_data[pc_col].values, bins=bin_edges)

            hist = hv.Histogram((edges, counts), kdims=[pc_label], vdims=["Frequency"], label=brain_region).opts(
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
    plot_opts = {
        "width": plot_options["plot"]["width"],
        "height": plot_options["plot"]["height"],
        "show_grid": plot_options["plot"]["show_grid"],
        "fontscale": plot_options["plot"]["fontscale"],
        "xlabel": pc_label,
        "ylabel": "Frequency",
        "tools": ["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        "show_legend": show_legend,
    }

    if show_legend:
        plot_opts["legend_position"] = "right"

    plot = hv.Overlay(plot_elements).opts(**plot_opts)

    return plot


def create_interactive_scatterplot_matrix(
    pca_data,
    static_results=None,
    temporal_results=None,
    n_components=6,
    significance_threshold=0.05,
    plot_options=hv_main,
    save_prefix="interactive_temporal_pca_scatterplot_matrix",
):
    """
    Create interactive scatterplot matrix using HoloViews for temporal PCA

    Parameters:
    - pca_data: DataFrame with temporal PCA scores and metadata
    - static_results: DataFrame with static statistical results
    - temporal_results: DataFrame with temporal statistical results
    - n_components: Number of temporal PCA components to show (default: 6)
    - significance_threshold: Threshold for significance (default: 0.05)
    - plot_options: Dictionary with plot styling options
    - save_prefix: Prefix for saved files
    """

    # Select temporal PCA components
    fpca_cols = [f"FPCA{i+1}" for i in range(n_components)]  # Internal column names (FPCA not fPCA)
    fpc_labels = [f"fPC{i+1}" for i in range(n_components)]  # Display labels (functional PC)
    available_fpca_cols = [col for col in fpca_cols if col in pca_data.columns]
    available_fpc_labels = [f"fPC{i+1}" for i, col in enumerate(fpca_cols) if col in pca_data.columns]
    print(f"Using temporal PCA components: {available_fpca_cols} (displayed as {available_fpc_labels})")

    # Get significance information
    static_significance = get_significance_from_permutation(static_results, significance_threshold)
    temporal_significance = get_significance_from_permutation(temporal_results, significance_threshold)

    # Filter to genotypes with sufficient samples
    genotype_counts = pca_data["Nickname"].value_counts()
    genotypes_with_samples = genotype_counts[genotype_counts >= 5].index.tolist()

    # Prepare plotting data
    genotype_means, all_significant, static_significant, temporal_significant = prepare_plotting_data(
        pca_data, static_significance, temporal_significance, genotypes_with_samples
    )

    print(f"Static significant genotypes: {len(static_significant)}")
    print(f"Temporal significant genotypes: {len(temporal_significant)}")
    print(f"Total unique significant genotypes: {len(all_significant)}")

    # Create matrix of plots
    plots = {}
    n_pcs = len(available_fpca_cols)

    for i, (pc_y_col, pc_y_label) in enumerate(zip(available_fpca_cols, available_fpc_labels)):
        for j, (pc_x_col, pc_x_label) in enumerate(zip(available_fpca_cols, available_fpc_labels)):

            if i == j:
                # Diagonal: show distribution as histogram - no legends
                plot = create_interactive_histogram(
                    genotype_means, pc_x_col, pc_x_label, plot_options, show_legend=False
                )
                plots[(i, j)] = plot
            else:
                # Off-diagonal: scatter plot - no legends
                plot = create_interactive_scatterplot(
                    genotype_means,
                    pca_data,
                    pc_x_col,
                    pc_y_col,
                    pc_x_label,
                    pc_y_label,
                    plot_options,
                    show_legend=False,
                    legend_type=None,
                )
                plots[(i, j)] = plot

    # Create layout as a proper matrix
    # Create a list of all plots in row-major order for matrix layout
    all_plots = []
    for i in range(n_pcs):
        for j in range(n_pcs):
            all_plots.append(plots[(i, j)])

    # Create matrix layout - keep this as the main 6x6 matrix
    matrix_layout = hv.Layout(all_plots).cols(n_pcs).opts(**plot_options["layout"])

    # Return just the matrix without external legends to maintain 6x6 structure
    final_layout = matrix_layout

    # Print summary
    print(f"\n=== Interactive Temporal PCA Scatterplot Matrix Summary ===")
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

    return final_layout, genotype_means, all_significant, static_significant, temporal_significant


def save_plots_multiple_formats(plot, base_filename):
    """Save plots in multiple formats for different use cases"""
    saved_files = []

    # Save HTML (always works)
    try:
        html_filename = f"{base_filename}.html"
        hv.save(plot, html_filename, fmt="html")
        saved_files.append(html_filename)
        print(f"✓ Interactive web version: {html_filename}")
    except Exception as e:
        print(f"✗ Could not save HTML: {e}")

    # Save PNG (requires selenium and webdriver)
    try:
        png_filename = f"{base_filename}.png"
        hv.save(plot, png_filename, fmt="png")
        saved_files.append(png_filename)
        print(f"✓ Static raster for presentations: {png_filename}")
    except Exception as e:
        print(f"✗ Could not save PNG: {e}")
        print(f"  To enable PNG export, you can:")
        print(f"  1. Use the HTML version and take screenshots")
        print(f"  2. Export individual plots from the browser")
        print(f"  3. Use print-to-PDF in browser, then convert to PNG")

    # Information about vector graphics
    print(f"ℹ For vector graphics (EPS/SVG):")
    print(f"  • Use browser's print-to-PDF for high quality")
    print(f"  • Convert PDF to EPS/SVG using external tools")
    print(f"  • Or extract individual plots for vector export")

    return saved_files


def main():
    """Main function"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    # Load temporal data
    print("Loading temporal PCA data with brain region information...")
    pca_data, static_results, temporal_results = load_temporal_data_with_brain_regions()

    if static_results is not None:
        print(f"Static results shape: {static_results.shape}")
    if temporal_results is not None:
        print(f"Temporal results shape: {temporal_results.shape}")

    print(f"Temporal PCA data shape: {pca_data.shape}")
    print(f"Temporal PCA data columns: {list(pca_data.columns)}")
    print(f"Unique brain regions: {sorted(pca_data['Brain region'].unique())}")
    print(f"Sample brain region values: {pca_data['Brain region'].head().tolist()}")

    # Create interactive scatterplot matrix
    print("\nCreating interactive temporal PCA scatterplot matrix...")

    # Choose plot style - you can switch between hv_main and hv_presentation
    plot_style = hv_main  # or hv_presentation for larger plots

    final_layout, genotype_means, all_significant, static_significant, temporal_significant = (
        create_interactive_scatterplot_matrix(
            pca_data,
            static_results,
            temporal_results,
            n_components=6,  # Use first 6 components
            significance_threshold=0.05,
            plot_options=plot_style,
            save_prefix="interactive_temporal_pca_scatterplot_matrix",
        )
    )

    # Save the interactive plot in multiple formats
    print("\nSaving plots in multiple formats...")
    saved_files = save_plots_multiple_formats(final_layout, "interactive_temporal_pca_scatterplot_matrix")

    # Create matplotlib legend instead of HTML
    print("Creating matplotlib legend figure...")
    import subprocess

    try:
        subprocess.run(["python", "create_pca_legends.py"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not create legend figure automatically.")
        print("Run 'python create_pca_legends.py' manually to create legend.")

    print("\nFiles saved:")
    for file in saved_files:
        print(f"- Plot: {file}")
    print("- Legend: pca_legend_combined.png")

    # Display the plot (if running in Jupyter or similar environment)
    return final_layout

    print("\nInteractive temporal PCA plot saved with multiple formats")
    print("Open the HTML file in a web browser to interact with the plot.")


if __name__ == "__main__":
    matrix_layout = main()
