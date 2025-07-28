#!/usr/bin/env python3
"""
Unified Static Scatterplot Matrix Generator for PCA Results
Creates publication-quality static plots using matplotlib/seaborn that exactly match the interactive version's style.
Supports all plot types through configuration: individual/genotype × static/temporal PCA.
Exports as PNG, SVG, and PDF for different use cases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import sys
import os
import argparse
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2
import matplotlib.patches as patches

# Set up matplotlib and seaborn for publication quality
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
})

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
    """Create ellipse data points for plotting confidence intervals - exact match to interactive"""
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

def create_control_ellipses(pca_data, pc_x, pc_y, ax, config, confidence=0.95):
    """Create confidence ellipses for each control line - exact match to interactive"""
    for control_name, color in CONTROL_ELLIPSES.items():
        # Get data for this specific control line
        control_data = pca_data[pca_data["Nickname"] == control_name]

        if len(control_data) >= 3:
            control_x = control_data[pc_x].values
            control_y = control_data[pc_y].values

            # Create ellipse data
            ellipse_x, ellipse_y = create_confidence_ellipse_data(control_x, control_y, confidence=confidence)

            if ellipse_x is not None:
                # Add ellipse to plot using config styling
                ellipse_style = config["ellipse"]
                ax.plot(ellipse_x, ellipse_y,
                       color=color,
                       alpha=ellipse_style["alpha"],
                       linewidth=ellipse_style["linewidth"])

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
        return SIGNIFICANCE_MARKERS[sig_type]["static"]

    def get_marker_size(nickname, base_size):
        sig_type = get_significance_type(nickname)
        multiplier = SIGNIFICANCE_MARKERS[sig_type]["size_multiplier"]
        return base_size * multiplier

    plot_data["significance_type"] = plot_data["Nickname"].apply(get_significance_type)
    plot_data["marker_style"] = plot_data["Nickname"].apply(get_marker_style)
    plot_data["marker_size"] = plot_data["Nickname"].apply(lambda x: get_marker_size(x, config["scatter"]["s"]))

    return plot_data

def create_static_scatterplot_matrix(pca_data, static_results, temporal_results, config, n_components=6):
    """Create static scatterplot matrix using seaborn and matplotlib"""

    # Prepare plotting data
    plot_data = prepare_plotting_data(pca_data, static_results, temporal_results, config)

    # Select PCA components
    available_pca_cols = [col for col in config["pca_columns"][:n_components] if col in plot_data.columns]
    available_pc_labels = config["pc_labels"][:len(available_pca_cols)]
    print(f"Using PCA components: {available_pca_cols} (displayed as {available_pc_labels})")

    # Get brain region colors
    brain_region_colors = Config.color_dict

    # Create figure with subplots
    n_pcs = len(available_pca_cols)
    figsize = config["figure"]["figsize"]
    fig, axes = plt.subplots(n_pcs, n_pcs, figsize=figsize)

    for i, (pc_y_col, pc_y_label) in enumerate(zip(available_pca_cols, available_pc_labels)):
        for j, (pc_x_col, pc_x_label) in enumerate(zip(available_pca_cols, available_pc_labels)):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show distribution by brain region
                pc_values = plot_data[pc_x_col].values

                # Calculate bins using the configuration formula
                bins_formula = config["histogram"]["bins_formula"]
                n_bins = eval(bins_formula)

                # Create common bin edges for all histograms
                bin_edges = np.linspace(pc_values.min(), pc_values.max(), n_bins + 1)

                unique_regions = plot_data["Brain_region"].unique()
                for brain_region in unique_regions:
                    region_data = plot_data[plot_data["Brain_region"] == brain_region]
                    if len(region_data) > 0:
                        color = brain_region_colors.get(brain_region, "#000000")
                        hist_style = config["histogram"]

                        # Use the same bins for consistent histogram appearance
                        ax.hist(region_data[pc_x_col], bins=bin_edges,
                               alpha=hist_style["alpha"],
                               color=color,
                               density=hist_style["density"],
                               edgecolor=color,
                               linewidth=hist_style["linewidth"])

                # Add reference line at zero
                ref_style = config["reference_lines"]["diagonal_axvline"]
                ax.axvline(x=0, **ref_style)

                ax.set_xlabel(pc_x_label)
                ax.set_ylabel('Density')

            else:
                # Off-diagonal: scatter plot by brain region with significance markers
                unique_regions = plot_data["Brain_region"].unique()
                scatter_style = config["scatter"]

                for brain_region in unique_regions:
                    region_data = plot_data[plot_data["Brain_region"] == brain_region]
                    if len(region_data) > 0:
                        color = brain_region_colors.get(brain_region, "#000000")

                        # Group by marker style for this brain region
                        for marker_style in region_data["marker_style"].unique():
                            marker_data = region_data[region_data["marker_style"] == marker_style]
                            if len(marker_data) > 0:
                                # Use the marker-specific size
                                marker_sizes = marker_data["marker_size"].values
                                ax.scatter(marker_data[pc_x_col], marker_data[pc_y_col],
                                         color=color,
                                         s=marker_sizes,
                                         alpha=scatter_style["alpha"],
                                         edgecolors=scatter_style["edgecolors"],
                                         linewidth=scatter_style["linewidth"],
                                         marker=marker_style)

                # Create control ellipses (based on all data, not just significant flies)
                create_control_ellipses(pca_data, pc_x_col, pc_y_col, ax, config)

                # Add reference lines at zero
                ref_style = config["reference_lines"]
                ax.axhline(y=0, **ref_style["axhline"])
                ax.axvline(x=0, **ref_style["axvline"])

                ax.set_xlabel(pc_x_label)
                ax.set_ylabel(pc_y_label)

            # Remove individual legends (match interactive: show_legend=False)
            if ax.get_legend():
                ax.legend().set_visible(False)

    # NO LEGEND - match interactive script exactly (show_legend=False)
    # The interactive script explicitly sets show_legend=False and states:
    # "Note: Use the legend from the main script (pca_legend_combined.png)"

    plt.tight_layout()
    return fig

def save_plots_multiple_formats(fig, base_filename):
    """Save plots in multiple publication-ready formats in organized directories"""
    saved_files = []
    formats = ['png', 'svg', 'pdf', 'eps']

    # Ensure the organized directory structure exists
    for fmt in formats:
        os.makedirs(f"pca_matrices/{fmt}", exist_ok=True)

    for fmt in formats:
        try:
            filename = f"pca_matrices/{fmt}/{base_filename}.{fmt}"
            fig.savefig(filename, format=fmt, bbox_inches='tight', dpi=300)
            saved_files.append(filename)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"✗ Could not save {fmt.upper()}: {e}")

    return saved_files

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Generate unified static PCA scatterplot matrices")
    parser.add_argument("plot_type",
                       choices=["individual_static", "genotype_static", "individual_temporal", "genotype_temporal"],
                       help="Type of plot to generate")
    parser.add_argument("--n-components", type=int, default=6,
                       help="Number of PCA components to include (default: 6)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.plot_type, style_type="static")

    if args.verbose:
        print(f"Creating {config['description']}...")
        print(f"Data source: {config['data_source']}")
        print(f"Aggregation level: {config['aggregation_level']}")
        print(f"Significance filter: {config['significance_filter']}")

    # Load data
    pca_data, static_results, temporal_results = load_data(config)

    # Create the plot
    fig = create_static_scatterplot_matrix(pca_data, static_results, temporal_results, config, args.n_components)

    # Save the plots
    base_filename = config["output_filename"]
    saved_files = save_plots_multiple_formats(fig, base_filename)

    print(f"\nStatic plots created: {config['description']}")
    print(f"Files saved:")
    for file in saved_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            file_type = "raster" if file.endswith('.png') else "vector - for Illustrator" if file.endswith('.svg') else "vector - for publications"
            print(f"- {file} ({file_type})")

    if not args.verbose:
        plt.show()

if __name__ == "__main__":
    main()
