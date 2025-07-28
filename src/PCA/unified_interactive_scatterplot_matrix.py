#!/usr/bin/env python3
"""
Unified Interactive Scatterplot Matrix Generator for PCA Results using HoloViews
Creates interactive plots with brain region color coding, significance highlighting, and hover tooltips.
Supports all plot types through configuration: individual/genotype × static/temporal PCA.
Exports as HTML for sharing and PNG for publications.
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
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

# Add the parent directory to the path to import Config and plot_configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCA import Config

# Import plot configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from plot_configurations import get_config, CONTROL_ELLIPSES, SIGNIFICANCE_MARKERS

def load_data(config):
    """Load PCA data and statistical results based on configuration"""
    # Load PCA score data
    pca_data = pd.read_feather(config["pca_data_file"])
    print(f"Loaded PCA data: {config['pca_data_file']} ({len(pca_data)} rows)")

    # Load static results
    static_results = None
    try:
        static_results = pd.read_csv(config["static_results_file"])
        print(f"Static statistical results loaded: {config['static_results_file']}")
    except FileNotFoundError:
        print(f"Static statistical results file not found: {config['static_results_file']}")

    # Load temporal results
    temporal_results = None
    try:
        temporal_results = pd.read_csv(config["temporal_results_file"])
        print(f"Temporal statistical results loaded: {config['temporal_results_file']}")
    except FileNotFoundError:
        print(f"Temporal statistical results file not found: {config['temporal_results_file']}")

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

def create_confidence_ellipse_data(data_x, data_y, confidence=0.95):
    """Create ellipse data points for plotting confidence intervals"""
    if len(data_x) < 3:
        return None

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

    return ellipse_points

def create_ellipses_for_controls(pca_data, x_col, y_col, config):
    """Create confidence ellipses for control genotypes"""
    ellipse_plots = []

    for control_name, color in CONTROL_ELLIPSES.items():
        control_data = pca_data[pca_data["Nickname"] == control_name]

        if len(control_data) >= 3:
            ellipse_points = create_confidence_ellipse_data(
                control_data[x_col].values,
                control_data[y_col].values
            )

            if ellipse_points is not None:
                ellipse_df = pd.DataFrame(ellipse_points, columns=[x_col, y_col])
                ellipse_style = config["ellipse"]

                ellipse_plot = hv.Path(ellipse_df).opts(
                    color=color,
                    alpha=ellipse_style["alpha"],
                    line_width=ellipse_style["line_width"],
                    tools=[]
                )
                ellipse_plots.append(ellipse_plot)

    return ellipse_plots

def prepare_plotting_data(pca_data, static_results, temporal_results, config):
    """Prepare data for plotting based on configuration"""
    # Get significance information
    static_significance = get_significance_from_permutation(static_results, 0.05)
    temporal_significance = get_significance_from_permutation(temporal_results, 0.05)

    if config["significance_filter"] == "static":
        # Use only static significance
        significant_genotypes = [
            genotype for genotype, sig_info in static_significance.items()
            if sig_info["is_significant"]
        ]
        print(f"Using static significance: {len(significant_genotypes)} genotypes")

    elif config["significance_filter"] == "temporal":
        # Use only temporal significance
        significant_genotypes = [
            genotype for genotype, sig_info in temporal_significance.items()
            if sig_info["is_significant"]
        ]
        print(f"Using temporal significance: {len(significant_genotypes)} genotypes")

    elif config["significance_filter"] == "combined":
        # Use combined significance
        static_significant = [
            genotype for genotype, sig_info in static_significance.items()
            if sig_info["is_significant"]
        ]
        temporal_significant = [
            genotype for genotype, sig_info in temporal_significance.items()
            if sig_info["is_significant"]
        ]
        significant_genotypes = list(set(static_significant + temporal_significant))
        print(f"Using combined significance: {len(static_significant)} static + {len(temporal_significant)} temporal = {len(significant_genotypes)} unique")
    else:
        raise ValueError(f"Unknown significance_filter: {config['significance_filter']}")

    if config["aggregation_level"] == "individual":
        # Filter to significant individual flies
        plot_data = pca_data[pca_data["Nickname"].isin(significant_genotypes)]
        print(f"Individual flies from significant genotypes: {len(plot_data)}")

    elif config["aggregation_level"] == "genotype":
        # Calculate genotype averages for significant genotypes only
        significant_data = pca_data[pca_data["Nickname"].isin(significant_genotypes)]

        # Group by genotype and calculate means
        pca_columns = config["pca_columns"]

        # Ensure Brain region column is renamed consistently
        if "Brain region" in significant_data.columns and "Brain_region" not in significant_data.columns:
            significant_data = significant_data.rename(columns={"Brain region": "Brain_region"})

        grouping_columns = ["Nickname", "Brain_region"]
        plot_data = significant_data.groupby(grouping_columns)[pca_columns].mean().reset_index()
        print(f"Genotype averages for significant genotypes: {len(plot_data)}")
    else:
        raise ValueError(f"Unknown aggregation_level: {config['aggregation_level']}")

    # Ensure Brain_region column exists
    if "Brain region" in plot_data.columns:
        plot_data = plot_data.rename(columns={"Brain region": "Brain_region"})

    # Add control assignment information
    plot_data["control_assignment"] = plot_data["Nickname"].apply(
        lambda x: "Control" if x in ['mCherry', 'GFP-1', 'GFP', 'Empty-split'] else "Hit"
    )

    # Add sample size information (N flies) for genotype level
    if config["aggregation_level"] == "genotype":
        # Calculate N flies per genotype-brain_region combination
        # First ensure Brain region is renamed consistently
        pca_data_copy = pca_data.copy()
        if "Brain region" in pca_data_copy.columns and "Brain_region" not in pca_data_copy.columns:
            pca_data_copy = pca_data_copy.rename(columns={"Brain region": "Brain_region"})

        n_flies_info = pca_data_copy.groupby(["Nickname", "Brain_region"]).size().reset_index(name="n_flies")
        plot_data = plot_data.merge(n_flies_info, on=["Nickname", "Brain_region"], how="left")
    else:
        # For individual flies, N flies is 1
        plot_data["n_flies"] = 1

    # Add p-value information using the correct field name
    def get_pval_info(nickname):
        static_pval = static_significance.get(nickname, {}).get("permutation_pval", None)
        temporal_pval = temporal_significance.get(nickname, {}).get("permutation_pval", None)
        return static_pval, temporal_pval

    pval_info = plot_data["Nickname"].apply(get_pval_info)
    plot_data["static_pval"] = [p[0] for p in pval_info]
    plot_data["temporal_pval"] = [p[1] for p in pval_info]

    # Add significance markers - this is the key addition!
    def get_significance_type(nickname):
        in_static = nickname in [g for g, s in static_significance.items() if s["is_significant"]]
        in_temporal = nickname in [g for g, s in temporal_significance.items() if s["is_significant"]]

        if in_static and in_temporal:
            return "both"
        elif in_static:
            return "static"
        elif in_temporal:
            return "temporal"
        else:
            return "static"  # fallback, shouldn't happen

    def get_marker_style(nickname):
        sig_type = get_significance_type(nickname)
        return SIGNIFICANCE_MARKERS[sig_type]["interactive"]

    def get_marker_size(nickname, base_size):
        sig_type = get_significance_type(nickname)
        multiplier = SIGNIFICANCE_MARKERS[sig_type]["size_multiplier"]
        return base_size * multiplier

    plot_data["significance_type"] = plot_data["Nickname"].apply(get_significance_type)
    plot_data["marker_style"] = plot_data["Nickname"].apply(get_marker_style)
    plot_data["marker_size"] = plot_data["Nickname"].apply(lambda x: get_marker_size(x, config["scatter"]["size"]))

    return plot_data

def create_scatter_plot(plot_data, x_col, y_col, x_label, y_label, config):
    """Create a single scatter plot with the given configuration"""
    brain_region_colors = Config.color_dict
    scatter_style = config["scatter"]
    plot_style = config["plot"]

    # Create hover tool with comprehensive information
    tooltips = [
        ("Genotype", "@Nickname"),
        ("Brain Region", "@Brain_region"),
        ("Control", "@control_assignment"),
        ("Significance", "@significance_type"),
        (x_label, f"@{{{x_col}}}{{0.000}}"),
        (y_label, f"@{{{y_col}}}{{0.000}}")
    ]

    # Add N flies for genotype aggregation
    if config["aggregation_level"] == "genotype":
        tooltips.insert(4, ("N Flies", "@n_flies"))
    elif config["aggregation_level"] == "individual":
        # Add fly ID for individual level data if available
        tooltips.insert(4, ("Fly ID", "@fly"))
    # Add p-values if available
    tooltips.extend([
        ("Static p-val", "@static_pval{0.000}"),
        ("Temporal p-val", "@temporal_pval{0.000}")
    ])

    hover = HoverTool(tooltips=tooltips)

    # Create separate scatter plots for each significance type to control marker shapes
    scatter_plots = []

    for sig_type in ["both", "static", "temporal"]:
        sig_data = plot_data[plot_data["significance_type"] == sig_type]
        if len(sig_data) == 0:
            continue

        marker_style = SIGNIFICANCE_MARKERS[sig_type]["interactive"]

        for brain_region in sig_data["Brain_region"].unique():
            region_data = sig_data[sig_data["Brain_region"] == brain_region]
            if len(region_data) > 0:
                color = brain_region_colors.get(brain_region, "#000000")

                # All tooltip fields must be included in vdims for HoverTool to work
                vdims_list = [
                    y_col,
                    "Nickname",
                    "Brain_region",
                    "control_assignment",
                    "significance_type",
                    "marker_size",
                    "n_flies",
                    "static_pval",
                    "temporal_pval"
                ]

                # Add fly ID for individual level data
                if config["aggregation_level"] == "individual" and "fly" in region_data.columns:
                    vdims_list.append("fly")

                scatter = hv.Scatter(region_data, kdims=[x_col], vdims=vdims_list).opts(
                    color=color,
                    size="marker_size",  # Use the calculated marker size
                    alpha=scatter_style["alpha"],
                    line_color=scatter_style["line_color"],
                    line_width=scatter_style["line_width"],
                    marker=marker_style,
                    tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
                    active_tools=plot_style["active_tools"],
                    width=plot_style["width"],
                    height=plot_style["height"],
                    show_grid=plot_style["show_grid"],
                    fontscale=plot_style["fontscale"],
                    xlabel=x_label,
                    ylabel=y_label,
                    show_legend=config["layout"]["show_legend"]  # No title, no legend
                )
                scatter_plots.append(scatter)

    # Combine scatter plots
    combined_scatter = scatter_plots[0]
    for scatter in scatter_plots[1:]:
        combined_scatter = combined_scatter * scatter

    return combined_scatter

def create_histogram_plot(plot_data, col, label, config):
    """Create a histogram plot with the given configuration"""
    brain_region_colors = Config.color_dict
    plot_style = config["plot"]
    hist_style = config["histogram"]

    # Calculate bins using the configuration formula
    pc_values = plot_data[col].values
    n = len(pc_values)
    bins_formula = hist_style["bins_formula"]
    n_bins = eval(bins_formula)

    # Create common bin edges for all histograms to ensure homogeneous binning
    bin_edges = np.linspace(pc_values.min(), pc_values.max(), n_bins + 1)

    # Create histogram plots by brain region
    hist_plots = []
    for brain_region in plot_data["Brain_region"].unique():
        region_data = plot_data[plot_data["Brain_region"] == brain_region]
        if len(region_data) > 0:
            color = brain_region_colors.get(brain_region, "#000000")

            # Create histogram for this brain region with consistent bins (matching original format)
            counts, edges = np.histogram(region_data[col], bins=bin_edges, density=hist_style["density"])

            hist = hv.Histogram((edges, counts), kdims=[label], vdims=["Frequency"]).opts(
                fill_color=color,
                fill_alpha=hist_style["alpha"],
                line_color=color,
                line_width=2,
                width=plot_style["width"],
                height=plot_style["height"],
                show_grid=plot_style["show_grid"],
                fontscale=plot_style["fontscale"],
                xlabel=label,
                ylabel="Density" if hist_style["density"] else "Count",
                show_legend=config["layout"]["show_legend"],  # No title, no legend
                tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"]
            )
            hist_plots.append(hist)

    # Combine histogram plots
    combined_hist = hist_plots[0]
    for hist in hist_plots[1:]:
        combined_hist = combined_hist * hist

    # Add reference line at zero
    ref_line = hv.VLine(0).opts(color='black', alpha=0.5, line_width=2)
    combined_hist = combined_hist * ref_line

    return combined_hist

def create_interactive_scatterplot_matrix(pca_data, static_results, temporal_results, config, n_components=6):
    """Create interactive scatterplot matrix using HoloViews"""

    # Prepare plotting data
    plot_data = prepare_plotting_data(pca_data, static_results, temporal_results, config)

    # Select PCA components
    available_pca_cols = [col for col in config["pca_columns"][:n_components] if col in plot_data.columns]
    available_pc_labels = config["pc_labels"][:len(available_pca_cols)]
    print(f"Using PCA components: {available_pca_cols} (displayed as {available_pc_labels})")

    # Create matrix of plots
    plots = {}
    n_pcs = len(available_pca_cols)

    for i, (pc_y_col, pc_y_label) in enumerate(zip(available_pca_cols, available_pc_labels)):
        for j, (pc_x_col, pc_x_label) in enumerate(zip(available_pca_cols, available_pc_labels)):

            if i == j:
                # Diagonal: histogram
                plot = create_histogram_plot(plot_data, pc_x_col, pc_x_label, config)
            else:
                # Off-diagonal: scatter plot
                plot = create_scatter_plot(plot_data, pc_x_col, pc_y_col, pc_x_label, pc_y_label, config)

                # Add control ellipses for scatter plots
                ellipses = create_ellipses_for_controls(pca_data, pc_x_col, pc_y_col, config)
                for ellipse in ellipses:
                    plot = plot * ellipse

                # Add reference lines at zero
                h_line = hv.HLine(0).opts(color='black', alpha=0.3, line_width=1)
                v_line = hv.VLine(0).opts(color='black', alpha=0.3, line_width=1)
                plot = plot * h_line * v_line

            plots[(i, j)] = plot

    # Create layout
    layout_opts = opts.Layout(
        shared_axes=config["layout"]["shared_axes"]
    )

    # Create matrix layout
    matrix_layout = hv.Layout([plots[(i, j)] for i in range(n_pcs) for j in range(n_pcs)]).cols(n_pcs).opts(layout_opts)

    return matrix_layout

def save_interactive_plot(plot, base_filename, formats=['html', 'png']):
    """Save interactive plot in multiple formats in organized directories"""
    saved_files = []

    # Ensure the organized directory structure exists
    for fmt in formats:
        os.makedirs(f"pca_matrices/{fmt}", exist_ok=True)

    for fmt in formats:
        try:
            filename = f"pca_matrices/{fmt}/{base_filename}.{fmt}"
            if fmt == 'html':
                hv.save(plot, filename)
            elif fmt == 'png':
                hv.save(plot, filename, fmt='png')
            saved_files.append(filename)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"✗ Could not save {fmt.upper()}: {e}")

    return saved_files

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Generate unified interactive PCA scatterplot matrices")
    parser.add_argument("plot_type",
                       choices=["individual_static", "genotype_static", "individual_temporal", "genotype_temporal"],
                       help="Type of plot to generate")
    parser.add_argument("--n-components", type=int, default=6,
                       help="Number of PCA components to include (default: 6)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--formats", nargs="+", default=['html'],
                       choices=['html'],
                       help="Output formats (default: html only)")

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.plot_type, style_type="interactive")

    if args.verbose:
        print(f"Creating {config['description']}...")
        print(f"Data source: {config['data_source']}")
        print(f"Aggregation level: {config['aggregation_level']}")
        print(f"Significance filter: {config['significance_filter']}")

    # Load data
    pca_data, static_results, temporal_results = load_data(config)

    # Create the plot
    plot = create_interactive_scatterplot_matrix(pca_data, static_results, temporal_results, config, args.n_components)

    # Save the plots
    base_filename = config["output_filename"]
    saved_files = save_interactive_plot(plot, base_filename, args.formats)

    print(f"\nInteractive plots created: {config['description']}")
    print(f"Files saved:")
    for file in saved_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            file_type = "interactive web page" if file.endswith('.html') else "static image"
            print(f"- {file} ({file_type})")

    # Show the plot
    if not args.verbose:
        import bokeh.plotting
        bokeh.plotting.show(hv.render(plot))

if __name__ == "__main__":
    main()
