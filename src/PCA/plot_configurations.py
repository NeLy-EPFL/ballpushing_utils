#!/usr/bin/env python3
"""
Configuration definitions for unified PCA scatterplot matrix generation.
This file defines all the different plot types and their styling parameters.
"""

import os
import pandas as pd
from pathlib import Path

# Data source configurations
DATA_SOURCES = {
    "static": {
        # Auto-detect Sparse PCA vs regular PCA files
        "pca_data_file": "src/PCA/static_sparsepca_with_metadata_tailoredctrls.feather",  # Updated for Sparse PCA
        "pca_data_file_fallback": "src/PCA/static_pca_with_metadata_tailoredctrls.feather",  # Fallback to regular PCA
        "static_results_file": "src/PCA/static_sparsepca_stats_results_allmethods_tailoredctrls.csv",  # Updated for Sparse PCA
        "static_results_file_fallback": "src/PCA/static_pca_stats_results_allmethods_tailoredctrls.csv",  # Fallback
        "temporal_results_file": "src/PCA/fpca_temporal_stats_results_allmethods_tailoredctrls.csv",
        "pca_columns": [
            "SparsePCA1",
            "SparsePCA2",
            "SparsePCA3",
            "SparsePCA4",
            "SparsePCA5",
            "SparsePCA6",
        ],  # Updated for Sparse PCA
        "pca_columns_fallback": ["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6"],  # Fallback for regular PCA
        "pc_labels": ["sPC1", "sPC2", "sPC3", "sPC4", "sPC5", "sPC6"],  # Updated labels for Sparse PCA
    },
    "temporal": {
        "pca_data_file": "src/PCA/fpca_temporal_with_metadata_tailoredctrls.feather",
        "static_results_file": "src/PCA/static_sparsepca_stats_results_allmethods_tailoredctrls.csv",  # Updated for Sparse PCA
        "static_results_file_fallback": "src/PCA/static_pca_stats_results_allmethods_tailoredctrls.csv",  # Fallback
        "temporal_results_file": "src/PCA/fpca_temporal_stats_results_allmethods_tailoredctrls.csv",
        "pca_columns": ["FPCA1", "FPCA2", "FPCA3", "FPCA4", "FPCA5", "FPCA6"],  # Corrected column names
        "pc_labels": ["fPC1", "fPC2", "fPC3", "fPC4", "fPC5", "fPC6"],  # Display labels for functional PCA
    },
}

# Interactive (HoloViews) styling configurations
INTERACTIVE_STYLES = {
    "individual_static": {
        "description": "Individual flies - Static PCA",
        "data_source": "static",
        "aggregation_level": "individual",
        "significance_filter": "static",  # Use only static significance
        "scatter": {
            "size": 6,
            "alpha": 0.4,
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
            "alpha": 0.4,  # Increased from 0.2 for better visibility
        },
        "layout": {
            "shared_axes": False,
            "show_legend": False,  # Match interactive scripts exactly
        },
        "histogram": {
            "bins_formula": "max(6, min(10, int(np.sqrt(n))))",  # More conservative binning for better visibility
            "alpha": 0.35,
            "density": True,
        },
    },
    "genotype_static": {
        "description": "Genotype averages - Static PCA",
        "data_source": "static",
        "aggregation_level": "genotype",
        "significance_filter": "static",  # Use only static significance
        "scatter": {
            "size": 8,
            "alpha": 0.5,
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
            "alpha": 0.4,  # Increased from 0.2 for better visibility
        },
        "layout": {
            "shared_axes": False,
            "show_legend": False,
        },
        "histogram": {
            "bins_formula": "max(6, min(12, int(np.sqrt(n))))",  # Genotype averages formula
            "alpha": 0.35,
            "density": True,
        },
        "significance_styles": {
            "size": 12,
            "alpha": 0.8,
            "line_color": "black",
            "line_width": 1,
        },
        "convex_hull": {
            "line_width": 3,
            "alpha": 0.3,
        },
    },
    "individual_temporal": {
        "description": "Individual flies - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "individual",
        "significance_filter": "static",
        "scatter": {
            "size": 4,
            "alpha": 0.4,
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
            "alpha": 0.4,  # Increased from 0.2 for better visibility
        },
        "layout": {
            "shared_axes": False,
            "show_legend": False,
        },
        "histogram": {
            "bins_formula": "max(6, min(10, int(np.sqrt(n))))",  # More conservative binning for better visibility
            "alpha": 0.35,
            "density": True,
        },
        "significance_styles": {
            "size": 6,
            "alpha": 0.6,
            "line_color": "black",
            "line_width": 0.5,
        },
        "convex_hull": {
            "line_width": 3,
            "alpha": 0.3,
        },
    },
    "genotype_temporal": {
        "description": "Genotype averages - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "genotype",
        "significance_filter": "static",
        "scatter": {
            "size": 8,
            "alpha": 0.5,
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
            "alpha": 0.4,  # Increased from 0.2 for better visibility
        },
        "layout": {
            "shared_axes": False,
            "show_legend": False,
        },
        "histogram": {
            "bins_formula": "max(6, min(12, int(np.sqrt(n))))",  # Genotype averages formula
            "alpha": 0.35,
            "density": True,
        },
        "significance_styles": {
            "size": 12,
            "alpha": 0.8,
            "line_color": "black",
            "line_width": 1,
        },
        "convex_hull": {
            "line_width": 3,
            "alpha": 0.3,
        },
    },
}

# Static (Matplotlib/Seaborn) styling configurations - derived from interactive
STATIC_STYLES = {
    "individual_static": {
        "description": "Individual flies - Static PCA",
        "data_source": "static",
        "aggregation_level": "individual",
        "significance_filter": "static",
        "scatter": {
            "s": 12,  # matplotlib uses 's' for size - increased from 6
            "alpha": 0.4,
            "edgecolors": "black",
            "linewidth": 0.3,
        },
        "histogram": {
            "bins_formula": "max(8, min(15, int(np.sqrt(len(pc_values)) * 1.5)))",
            "alpha": 0.35,
            "density": True,
            "edgecolor": None,  # Will be set to color
            "linewidth": 2,
        },
        "reference_lines": {
            "axhline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "axvline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "diagonal_axvline": {"color": "black", "alpha": 0.5, "linewidth": 2},  # For diagonal plots
        },
        "ellipse": {
            "color": None,  # Will be set to control line color
            "alpha": 0.4,  # Increased from 0.2 for better visibility
            "linewidth": 2,
        },
        "figure": {
            "figsize": (20, 20),
            "dpi": 300,
        },
        "layout": {
            "show_legend": False,  # Critical: match interactive exactly
        },
    },
    "genotype_static": {
        "description": "Genotype averages - Static PCA",
        "data_source": "static",
        "aggregation_level": "genotype",
        "significance_filter": "static",
        "scatter": {
            "s": 16,  # Increased from 8
            "alpha": 0.5,
            "edgecolors": "black",
            "linewidth": 0.5,
        },
        "histogram": {
            "bins_formula": "max(6, min(12, int(np.sqrt(len(pc_values)))))",
            "alpha": 0.35,
            "density": True,
            "edgecolor": None,
            "linewidth": 2,
        },
        "reference_lines": {
            "axhline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "axvline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "diagonal_axvline": {"color": "black", "alpha": 0.5, "linewidth": 2},
        },
        "ellipse": {
            "color": None,
            "alpha": 0.4,  # Increased from 0.2 for better visibility
            "linewidth": 2,
        },
        "figure": {
            "figsize": (20, 20),
            "dpi": 300,
        },
        "layout": {
            "show_legend": False,
        },
    },
    "individual_temporal": {
        "description": "Individual flies - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "individual",
        "significance_filter": "static",
        "scatter": {
            "s": 8,  # Increased from 4
            "alpha": 0.4,
            "edgecolors": "black",
            "linewidth": 0.3,
        },
        "histogram": {
            "bins_formula": "max(8, min(15, int(np.sqrt(len(pc_values)) * 1.5)))",
            "alpha": 0.35,
            "density": True,
            "edgecolor": None,
            "linewidth": 2,
        },
        "reference_lines": {
            "axhline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "axvline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "diagonal_axvline": {"color": "black", "alpha": 0.5, "linewidth": 2},
        },
        "ellipse": {
            "color": None,
            "alpha": 0.4,  # Increased from 0.2 for better visibility
            "linewidth": 2,
        },
        "figure": {
            "figsize": (20, 20),
            "dpi": 300,
        },
        "layout": {
            "show_legend": False,
        },
    },
    "genotype_temporal": {
        "description": "Genotype averages - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "genotype",
        "significance_filter": "static",
        "scatter": {
            "s": 16,  # Increased from 8
            "alpha": 0.5,
            "edgecolors": "black",
            "linewidth": 0.5,
        },
        "histogram": {
            "bins_formula": "max(6, min(12, int(np.sqrt(len(pc_values)))))",
            "alpha": 0.35,
            "density": True,
            "edgecolor": None,
            "linewidth": 2,
        },
        "reference_lines": {
            "axhline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "axvline": {"color": "black", "alpha": 0.3, "linewidth": 1},
            "diagonal_axvline": {"color": "black", "alpha": 0.5, "linewidth": 2},
        },
        "ellipse": {
            "color": None,
            "alpha": 0.4,  # Increased from 0.2 for better visibility
            "linewidth": 2,
        },
        "figure": {
            "figsize": (20, 20),
            "dpi": 300,
        },
        "layout": {
            "show_legend": False,
        },
    },
}

# Output filename templates
OUTPUT_TEMPLATES = {
    "interactive": {
        "individual_static": "interactive_pca_scatterplot_matrix_individual",
        "genotype_static": "interactive_pca_scatterplot_matrix_genotype_averages",
        "individual_temporal": "interactive_temporal_pca_scatterplot_matrix_individual",
        "genotype_temporal": "interactive_temporal_pca_scatterplot_matrix_genotype_averages",
    },
    "static": {
        "individual_static": "static_pca_scatterplot_matrix_individual",
        "genotype_static": "static_pca_scatterplot_matrix_genotype_averages",
        "individual_temporal": "static_temporal_pca_scatterplot_matrix_individual",
        "genotype_temporal": "static_temporal_pca_scatterplot_matrix_genotype_averages",
    },
}

# Marker styles for significance (simplified for static-only analysis)
SIGNIFICANCE_MARKERS = {
    "significant": {
        "interactive": "circle",
        "static": "o",  # matplotlib circle
        "size_multiplier": 1.2,  # Larger for significant points
        "description": "Statistically significant",
    },
    "non_significant": {
        "interactive": "circle",
        "static": "o",  # same shape, different color/alpha will distinguish
        "size_multiplier": 1.0,
        "description": "Not statistically significant",
    },
}

# Control line ellipse configurations (same for all plot types)
CONTROL_ELLIPSES = {
    "Empty-Split": "#ffb7b7",
    "Empty-Gal4": "#C6F7B0",
    "TNTxPR": "#B4EAFF",
}


def get_config(plot_type, style_type="interactive"):
    """
    Get configuration for a specific plot type and style.

    Parameters:
    -----------
    plot_type : str
        One of: "individual_static", "genotype_static", "individual_temporal", "genotype_temporal"
    style_type : str, default "interactive"
        Either "interactive" or "static"

    Returns:
    --------
    dict: Complete configuration for the specified plot type
    """
    if style_type == "interactive":
        config = INTERACTIVE_STYLES[plot_type].copy()
    elif style_type == "static":
        config = STATIC_STYLES[plot_type].copy()
    else:
        raise ValueError(f"Unknown style_type: {style_type}")

    # Add data source configuration
    data_source_key = config["data_source"]
    config.update(DATA_SOURCES[data_source_key])

    # Add output template
    config["output_filename"] = OUTPUT_TEMPLATES[style_type][plot_type]

    return config


def list_available_configs():
    """List all available plot configurations."""
    print("Available plot configurations:")
    print("\nInteractive (HoloViews):")
    for plot_type in INTERACTIVE_STYLES:
        config = INTERACTIVE_STYLES[plot_type]
        print(f"  {plot_type}: {config['description']}")

    print("\nStatic (Matplotlib/Seaborn):")
    for plot_type in STATIC_STYLES:
        config = STATIC_STYLES[plot_type]
        print(f"  {plot_type}: {config['description']}")


def get_dynamic_data_source(pca_type="static", workspace_root="."):
    """
    Dynamically determine which PCA files are available and return appropriate configuration.

    Args:
        pca_type: "static" or "temporal"
        workspace_root: Path to workspace root directory

    Returns:
        dict: Updated data source configuration with actual file paths and column names
    """
    if pca_type not in DATA_SOURCES:
        raise ValueError(f"Unknown PCA type: {pca_type}")

    config = DATA_SOURCES[pca_type].copy()
    workspace_path = Path(workspace_root)

    # For static PCA, check if Sparse PCA files exist and get actual column names
    if pca_type == "static":
        # Check for Sparse PCA file first
        sparse_pca_file = workspace_path / config["pca_data_file"]
        fallback_pca_file = workspace_path / config["pca_data_file_fallback"]

        actual_file = None
        is_sparse = False

        if sparse_pca_file.exists():
            print(f"Using Sparse PCA files: {config['pca_data_file']}")
            actual_file = sparse_pca_file
            is_sparse = True
        elif fallback_pca_file.exists():
            print(f"Sparse PCA files not found, falling back to regular PCA: {config['pca_data_file_fallback']}")
            # Switch to regular PCA configuration
            config["pca_data_file"] = config["pca_data_file_fallback"]
            config["static_results_file"] = config["static_results_file_fallback"]
            actual_file = fallback_pca_file
            is_sparse = False
        else:
            raise FileNotFoundError(
                f"Neither Sparse PCA nor regular PCA files found at:\n  - {sparse_pca_file}\n  - {fallback_pca_file}"
            )

        # Dynamically detect available PCA columns
        try:
            df = pd.read_feather(actual_file)
            if is_sparse:
                # Look for SparsePCA columns
                pca_cols = [col for col in df.columns if col.startswith("SparsePCA") and col[-1].isdigit()]
                pca_cols = sorted(pca_cols, key=lambda x: int(x.replace("SparsePCA", "")))
                # Generate PC labels (PC1, PC2, etc. - no "s" prefix)
                pc_labels = [f"PC{i+1}" for i in range(len(pca_cols))]
            else:
                # Look for regular PCA columns
                pca_cols = [
                    col
                    for col in df.columns
                    if col.startswith("PCA") and not col.startswith("SparsePCA") and col[-1].isdigit()
                ]
                pca_cols = sorted(pca_cols, key=lambda x: int(x.replace("PCA", "")))
                # Generate PC labels
                pc_labels = [f"PC{i+1}" for i in range(len(pca_cols))]

            config["pca_columns"] = pca_cols
            config["pc_labels"] = pc_labels
            print(f"Detected {len(pca_cols)} PCA components: {pca_cols}")

        except Exception as e:
            print(f"Warning: Could not read PCA file to detect columns: {e}")
            # Fall back to default configuration
            pass

    # For temporal, detect FPCA columns dynamically
    elif pca_type == "temporal":
        temporal_file = workspace_path / config["pca_data_file"]
        if temporal_file.exists():
            try:
                df = pd.read_feather(temporal_file)
                fpca_cols = [col for col in df.columns if col.startswith("FPCA") and col[-1].isdigit()]
                fpca_cols = sorted(fpca_cols, key=lambda x: int(x.replace("FPCA", "")))
                fpc_labels = [f"fPC{i+1}" for i in range(len(fpca_cols))]

                config["pca_columns"] = fpca_cols
                config["pc_labels"] = fpc_labels
                print(f"Detected {len(fpca_cols)} functional PCA components: {fpca_cols}")

            except Exception as e:
                print(f"Warning: Could not read temporal PCA file to detect columns: {e}")

    return config


def get_available_pca_columns(pca_type="static", workspace_root="."):
    """
    Load actual data and return available PCA column names.
    Useful for determining how many PCs were actually generated.

    Args:
        pca_type: "static" or "temporal"
        workspace_root: Path to workspace root directory

    Returns:
        list: Available PCA column names
    """
    config = get_dynamic_data_source(pca_type, workspace_root)
    data_file = Path(workspace_root) / config["pca_data_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"PCA data file not found: {data_file}")

    # Load just the column names
    df = pd.read_feather(data_file)

    # Filter for PCA columns
    if pca_type == "static":
        # Look for either SparsePCA or PCA columns
        pca_cols = [col for col in df.columns if col.startswith(("SparsePCA", "PCA")) and col[-1].isdigit()]
    else:  # temporal
        pca_cols = [col for col in df.columns if col.startswith("FPCA") and col[-1].isdigit()]

    return sorted(pca_cols, key=lambda x: int(x[-1]))


if __name__ == "__main__":
    list_available_configs()
