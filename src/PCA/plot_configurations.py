#!/usr/bin/env python3
"""
Configuration definitions for unified PCA scatterplot matrix generation.
This file defines all the different plot types and their styling parameters.
"""

# Data source configurations
DATA_SOURCES = {
    "static": {
        "pca_data_file": "src/PCA/static_pca_with_metadata_tailoredctrls.feather",
        "static_results_file": "src/PCA/static_pca_stats_results_allmethods_tailoredctrls.csv",
        "temporal_results_file": "src/PCA/fpca_temporal_stats_results_allmethods_tailoredctrls.csv",
        "pca_columns": ["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6"],  # Internal column names
        "pc_labels": ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]  # Display labels
    },
    "temporal": {
        "pca_data_file": "src/PCA/fpca_temporal_with_metadata_tailoredctrls.feather",
        "static_results_file": "src/PCA/static_pca_stats_results_allmethods_tailoredctrls.csv",
        "temporal_results_file": "src/PCA/fpca_temporal_stats_results_allmethods_tailoredctrls.csv",
        "pca_columns": ["FPCA1", "FPCA2", "FPCA3", "FPCA4", "FPCA5", "FPCA6"],  # Corrected column names
        "pc_labels": ["fPC1", "fPC2", "fPC3", "fPC4", "fPC5", "fPC6"]  # Display labels for functional PCA
    }
}

# Interactive (HoloViews) styling configurations
INTERACTIVE_STYLES = {
    "individual_static": {
        "description": "Individual flies - Static PCA",
        "data_source": "static",
        "aggregation_level": "individual",
        "significance_filter": "combined",  # Use both static and temporal significance
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
        }
    },
    "genotype_static": {
        "description": "Genotype averages - Static PCA",
        "data_source": "static",
        "aggregation_level": "genotype",
        "significance_filter": "combined",  # Use only combined significance
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
        }
    },
    "individual_temporal": {
        "description": "Individual flies - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "individual",
        "significance_filter": "combined",
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
        }
    },
    "genotype_temporal": {
        "description": "Genotype averages - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "genotype",
        "significance_filter": "combined",
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
        }
    }
}

# Static (Matplotlib/Seaborn) styling configurations - derived from interactive
STATIC_STYLES = {
    "individual_static": {
        "description": "Individual flies - Static PCA",
        "data_source": "static",
        "aggregation_level": "individual",
        "significance_filter": "combined",
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
        }
    },
    "genotype_static": {
        "description": "Genotype averages - Static PCA",
        "data_source": "static",
        "aggregation_level": "genotype",
        "significance_filter": "combined",
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
        }
    },
    "individual_temporal": {
        "description": "Individual flies - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "individual",
        "significance_filter": "combined",
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
        }
    },
    "genotype_temporal": {
        "description": "Genotype averages - Temporal PCA (fPCA)",
        "data_source": "temporal",
        "aggregation_level": "genotype",
        "significance_filter": "combined",
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
        }
    }
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
    }
}

# Marker styles for significance types
SIGNIFICANCE_MARKERS = {
    "both": {
        "interactive": "circle",
        "static": "o",  # matplotlib circle
        "size_multiplier": 1.2,  # Larger for both significance
        "description": "Both static and temporal significance"
    },
    "static": {
        "interactive": "square",
        "static": "s",  # matplotlib square
        "size_multiplier": 1.0,
        "description": "Static significance only"
    },
    "temporal": {
        "interactive": "triangle",
        "static": "^",  # matplotlib triangle
        "size_multiplier": 1.0,
        "description": "Temporal significance only"
    }
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

if __name__ == "__main__":
    list_available_configs()
