#!/usr/bin/env python3
"""
F1 PCA Analysis Script - Control Mode

Performs Principal Component Analysis on selected behavioral metrics from F1 control experiments.
Analyzes data in two ways:
1. By F1_condition (control, pretrained, pretrained_unlocked)
2. By Pretraining (naive vs pretrained)

Selected metrics for PCA:
- ball_proximity_proportion_200px: Time spent near ball
- distance_moved: Total ball distance moved
- max_distance: Maximum ball distance achieved
- nb_events: Number of interaction events
- has_major: Binary - achieved major event
- has_finished: Binary - completed task
- significant_ratio: Proportion of significant events

Usage:
    python f1_pca_analysis_control.py
    python f1_pca_analysis_control.py --show
    python f1_pca_analysis_control.py --n-components 3
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path
DATASET_PATH = (
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260104_12_F1_coordinates_F1_New_Data/summary/pooled_summary.feather"
)
OUTPUT_DIR = "/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New/PCA"

# F1 conditions to analyze
SELECTED_F1_CONDITIONS = [
    "control",
    "pretrained",
    "pretrained_unlocked",
]

# Metrics for PCA
PCA_METRICS = [
    "ball_proximity_proportion_200px",
    "distance_moved",
    "max_distance",
    "nb_events",
    "has_major",
    "has_finished",
    "significant_ratio",
]

# Elegant names for display
METRIC_DISPLAY_NAMES = {
    "ball_proximity_proportion_200px": "Ball Proximity",
    "distance_moved": "Distance Moved",
    "max_distance": "Max Distance",
    "nb_events": "N Events",
    "has_major": "Has Major",
    "has_finished": "Finished Task",
    "significant_ratio": "Significant Ratio",
}

# Pixel to mm conversion
PIXELS_PER_MM = 500 / 30

# Filter for test ball only
FILTER_TEST_BALL = True
TEST_BALL_VALUES = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]

# Plot styling
FIGURE_SIZE_SINGLE = (14, 8)
FIGURE_SIZE_MULTI = (14, 10)
FIGURE_SIZE_LOADING = (12, 8)
DPI = 300
PLOT_FORMAT = "png"
SCATTER_SIZE = 40
SCATTER_ALPHA = 0.6

# Colors for F1 conditions
F1_CONDITION_COLORS = {
    "control": "orange",
    "pretrained": "steelblue",
    "pretrained_unlocked": "lightgreen",
}

# Colors for Pretraining (match F1_condition colors)
PRETRAINING_COLORS = {
    "n": "orange",  # Naive = control
    "y": "steelblue",  # Pretrained = pretrained
}

ALPHA = 0.05


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def load_dataset():
    """Load dataset from feather file"""
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_feather(DATASET_PATH)
    print(f"  Loaded {len(df)} rows")
    return df


def detect_columns(df):
    """Detect f1_condition and pretraining columns"""
    f1_condition_col = None
    pretraining_col = None

    for col in df.columns:
        if "f1" in col.lower() and "condition" in col.lower() and f1_condition_col is None:
            f1_condition_col = col
        if "pretrain" in col.lower() and pretraining_col is None:
            pretraining_col = col

    if f1_condition_col is None:
        raise ValueError("Could not find F1_condition column")
    if pretraining_col is None:
        raise ValueError("Could not find pretraining column")

    print(f"  F1 Condition column: {f1_condition_col}")
    print(f"  Pretraining column: {pretraining_col}")

    return f1_condition_col, pretraining_col


def filter_dataset(df, f1_condition_col):
    """Filter dataset for selected conditions and test ball"""
    # Filter for selected F1 conditions
    if SELECTED_F1_CONDITIONS:
        df = df[df[f1_condition_col].isin(SELECTED_F1_CONDITIONS)].copy()
        print(f"  Filtered for {len(SELECTED_F1_CONDITIONS)} F1 conditions: {SELECTED_F1_CONDITIONS}")

    # Filter for test ball
    if FILTER_TEST_BALL:
        ball_col = "ball_identity"
        if ball_col not in df.columns:
            print(f"  Warning: '{ball_col}' column not found, skipping ball filter")
        else:
            print(f"  Available ball identities: {df[ball_col].unique()}")

            test_ball_data = pd.DataFrame()
            for test_val in TEST_BALL_VALUES:
                subset = df[df[ball_col] == test_val]
                if len(subset) > 0:
                    test_ball_data = pd.concat([test_ball_data, subset])

            if test_ball_data.empty:
                print(f"  Warning: No test ball data found")
            else:
                df = test_ball_data
                print(f"  Filtered to test ball: {len(df)} rows")

    print(f"  Final dataset size: {len(df)} rows")
    return df


def prepare_pca_data(df, metrics=None, grouping_col=None):
    """
    Prepare data for PCA analysis.

    Returns:
        X: Standardized feature matrix
        scaler: The fitted StandardScaler
        valid_metrics: List of metrics actually used
        metadata_df: DataFrame with metadata
    """
    if metrics is None:
        metrics = PCA_METRICS

    print(f"\nPreparing data for PCA...")
    print(f"  Requested metrics: {metrics}")

    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    missing_metrics = [m for m in metrics if m not in df.columns]

    if missing_metrics:
        print(f"  ⚠️  Missing metrics: {missing_metrics}")

    if len(available_metrics) < 2:
        raise ValueError(f"Need at least 2 metrics for PCA, only found: {available_metrics}")

    print(f"  Using {len(available_metrics)} metrics: {available_metrics}")

    # Extract feature matrix
    X = df[available_metrics].copy()

    # Store metadata
    metadata_cols = ["fly", "Pretraining"]
    if grouping_col and grouping_col in df.columns:
        metadata_cols.append(grouping_col)
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata_df = df[metadata_cols].copy()

    # Handle missing values
    n_before = len(X)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    metadata_df = metadata_df[valid_mask]
    n_after = len(X)

    if n_after < n_before:
        print(f"  Dropped {n_before - n_after} rows with missing values ({n_after} remaining)")

    if len(X) < 10:
        raise ValueError(f"Too few valid samples for PCA: {len(X)}")

    # Convert distance metrics from pixels to mm
    for metric in available_metrics:
        if "distance" in metric.lower() and "proximity" not in metric.lower():
            X[metric] = X[metric] / PIXELS_PER_MM
            print(f"  Converted {metric} from pixels to mm")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Final data shape: {X_scaled.shape}")
    print(f"  Features standardized (mean=0, std=1)")

    return X_scaled, scaler, available_metrics, metadata_df


# ============================================================================
# PCA ANALYSIS
# ============================================================================


def perform_pca(X, n_components=3):
    """Perform PCA analysis"""
    print(f"\nPerforming PCA with {n_components} components...")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(f"  Explained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var*100:.2f}%")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    return pca, X_pca


def create_pca_dataframe(X_pca, metadata_df, pca, metrics):
    """Create DataFrame with PCA results and metadata"""
    n_components = X_pca.shape[1]

    # Create PC columns
    pc_cols = {f"PC{i+1}": X_pca[:, i] for i in range(n_components)}
    pca_df = pd.DataFrame(pc_cols)

    # Add metadata
    for col in metadata_df.columns:
        pca_df[col] = metadata_df[col].values

    # Add explained variance as attributes
    pca_df.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    pca_df.attrs["components"] = pca.components_
    pca_df.attrs["feature_names"] = metrics

    return pca_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_explained_variance(pca, output_dir):
    """Plot explained variance by component"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)

    # Individual explained variance
    ax1.bar(
        components, pca.explained_variance_ratio_ * 100, color="steelblue", alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Explained Variance by Component", fontsize=14, fontweight="bold")
    ax1.set_xticks(components)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (comp, var) in enumerate(zip(components, pca.explained_variance_ratio_)):
        ax1.text(comp, var * 100 + 1, f"{var*100:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(
        components,
        cumsum,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax2.axhline(y=80, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="80% threshold")
    ax2.axhline(y=90, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="90% threshold")
    ax2.set_xlabel("Number of Components", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Explained Variance", fontsize=14, fontweight="bold")
    ax2.set_xticks(components)
    ax2.set_ylim([0, 105])
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(loc="lower right")

    for i, (comp, var) in enumerate(zip(components, cumsum)):
        ax2.text(comp, var + 2, f"{var:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / f"pca_explained_variance.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    return fig


def plot_loadings_heatmap(pca, metrics, output_dir):
    """Plot heatmap of PCA loadings"""
    n_components = pca.components_.shape[0]

    # Create DataFrame of loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=[METRIC_DISPLAY_NAMES.get(m, m) for m in metrics],
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LOADING)

    sns.heatmap(
        loadings,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Loading"},
        linewidths=0.5,
        linecolor="gray",
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("PCA Loadings: Metric Contributions to Principal Components", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")

    col_labels = [f"PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_components)]
    ax.set_xticklabels(col_labels, rotation=0)

    plt.tight_layout()

    output_path = Path(output_dir) / f"pca_loadings_heatmap.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    print("\n  Top contributors to each PC:")
    for i in range(n_components):
        pc_loadings = loadings[f"PC{i+1}"].abs().sort_values(ascending=False)
        print(f"    PC{i+1}: {pc_loadings.head(3).to_dict()}")

    return fig


def get_color(value, grouping_type):
    """Get color for a value based on grouping type"""
    if grouping_type == "f1_condition":
        return F1_CONDITION_COLORS.get(value, "#808080")
    elif grouping_type == "pretraining":
        return PRETRAINING_COLORS.get(str(value).lower(), "#808080")
    return "#808080"


def plot_pc_boxplot(pca_df, pc_name, grouping_col, selected_values, grouping_type, output_dir):
    """Create boxplot for a single principal component with between-group comparisons"""
    if pc_name not in pca_df.columns:
        print(f"  Warning: {pc_name} not found in data")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    # Get values in specified order
    all_values = pca_df[grouping_col].unique()
    group_values = [v for v in selected_values if v in all_values]

    # Single box per group
    group_spacing = 1.5
    box_width = 0.8

    all_data = []
    all_positions = []
    all_colors = []
    xtick_positions = []
    xtick_labels = []

    for i, group_val in enumerate(group_values):
        data = pca_df[pca_df[grouping_col] == group_val][pc_name].values
        all_data.append(data)
        all_positions.append(i * group_spacing)
        all_colors.append(get_color(group_val, grouping_type))
        xtick_positions.append(i * group_spacing)
        xtick_labels.append(group_val)

    # Perform pairwise Mann-Whitney tests between groups
    mw_results = []
    if len(group_values) >= 2:
        for idx1, idx2 in combinations(range(len(group_values)), 2):
            group1_data = all_data[idx1]
            group2_data = all_data[idx2]
            if len(group1_data) >= 2 and len(group2_data) >= 2:
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
                mw_results.append(
                    {
                        "group1": group_values[idx1],
                        "group2": group_values[idx2],
                        "idx1": idx1,
                        "idx2": idx2,
                        "p_value": p_value,
                        "n1": len(group1_data),
                        "n2": len(group2_data),
                    }
                )

    # FDR correction
    if mw_results:
        p_values = [r["p_value"] for r in mw_results]
        _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
        for i, result in enumerate(mw_results):
            result["p_corrected"] = p_corrected[i]

    # Create boxplots
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Style boxes
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    # Add jittered scatter points
    for data, pos, color in zip(all_data, all_positions, all_colors):
        if len(data) > 0:
            np.random.seed(42)
            jitter = np.random.normal(0, box_width * 0.15, size=len(data))
            ax.scatter(
                pos + jitter,
                data,
                alpha=SCATTER_ALPHA,
                s=SCATTER_SIZE,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    # Add significance annotations with brackets
    if mw_results:
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        bracket_height = y_max + y_range * 0.05
        bracket_increment = y_range * 0.12

        for i, result in enumerate(mw_results):
            idx1 = result["idx1"]
            idx2 = result["idx2"]

            x1 = idx1 * group_spacing
            x2 = idx2 * group_spacing
            y_bracket = bracket_height + i * bracket_increment

            p_corr = result["p_corrected"]
            if p_corr < 0.001:
                sig_marker = "***"
            elif p_corr < 0.01:
                sig_marker = "**"
            elif p_corr < 0.05:
                sig_marker = "*"
            else:
                sig_marker = "ns"

            # Draw bracket
            ax.plot(
                [x1, x1, x2, x2],
                [y_bracket - y_range * 0.02, y_bracket, y_bracket, y_bracket - y_range * 0.02],
                "k-",
                linewidth=1.5,
            )
            ax.text(
                (x1 + x2) / 2,
                y_bracket + y_range * 0.01,
                sig_marker,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Expand y-axis to accommodate brackets
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + y_range * 0.15 * len(mw_results))

    # Set labels
    ax.set_xticks(xtick_positions)
    labels_with_n = [f"{val}\n(n = {len(data)})" for val, data in zip(group_values, all_data)]
    ax.set_xticklabels(labels_with_n, rotation=45, ha="right", fontsize=9)

    # Get explained variance
    pc_idx = int(pc_name[2:]) - 1
    var_explained = pca_df.attrs["explained_variance_ratio"][pc_idx] * 100

    ax.set_ylabel(f"{pc_name} Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{pc_name} ({var_explained:.1f}% variance explained)", fontsize=14, fontweight="bold")

    xlabel = "F1 Condition" if grouping_type == "f1_condition" else "Pretraining"
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

    # Add legend for significance
    if mw_results:
        legend_elements = [
            plt.Line2D([0], [0], color="none", marker="$***$", markersize=10, label="p < 0.001 (FDR)"),
            plt.Line2D([0], [0], color="none", marker="$**$", markersize=10, label="p < 0.01 (FDR)"),
            plt.Line2D([0], [0], color="none", marker="$*$", markersize=10, label="p < 0.05 (FDR)"),
            plt.Line2D([0], [0], color="none", marker="$ns$", markersize=10, label="not significant"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = Path(output_dir) / f"{pc_name}_boxplot.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    # Print statistics
    if mw_results:
        print(f"  Statistics for {pc_name}:")
        for result in mw_results:
            print(
                f"    {result['group1']} vs {result['group2']}: p = {result['p_corrected']:.4f} "
                f"(n = {result['n1']}, {result['n2']})"
            )

    return fig


def plot_pca_scatter(pca_df, pc1, pc2, grouping_col, selected_values, grouping_type, output_dir, suffix=""):
    """Create scatter plot of two principal components"""
    if pc1 not in pca_df.columns or pc2 not in pca_df.columns:
        print(f"  Warning: {pc1} or {pc2} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get values in specified order
    all_values = pca_df[grouping_col].unique()
    group_values = [v for v in selected_values if v in all_values]

    # Plot each group
    for group_val in group_values:
        mask = pca_df[grouping_col] == group_val
        subset = pca_df[mask]

        if len(subset) == 0:
            continue

        color = get_color(group_val, grouping_type)
        label = f"{group_val} (n={len(subset)})"

        ax.scatter(
            subset[pc1],
            subset[pc2],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=[color],
            edgecolors="black",
            linewidths=0.5,
            marker="o",
            label=label,
            zorder=3,
        )

    # Get explained variance
    pc1_idx = int(pc1[2:]) - 1
    pc2_idx = int(pc2[2:]) - 1
    var1 = pca_df.attrs["explained_variance_ratio"][pc1_idx] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][pc2_idx] * 100

    ax.set_xlabel(f"{pc1} ({var1:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({var2:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_title(f"PCA: {pc1} vs {pc2}", fontsize=14, fontweight="bold")

    # Add origin lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    ax.grid(alpha=0.3, linestyle="--", zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir) / f"pca_scatter_{pc1}_vs_{pc2}{suffix}.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    return fig


def plot_bivariate_confidence_ellipses(pca_df, pc1, pc2, grouping_col, selected_values, grouping_type, output_dir):
    """
    Plot two PCs with 95% confidence ellipses for each condition.
    Ellipses visualize the bivariate distribution and separation between groups.
    """
    from matplotlib.patches import Ellipse
    import matplotlib.patches as mpatches

    if pc1 not in pca_df.columns or pc2 not in pca_df.columns:
        print(f"  Warning: {pc1} or {pc2} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    def confidence_ellipse(x, y, ax, n_std=1.96, facecolor="none", **kwargs):
        """Draw confidence ellipse of bivariate normal distribution."""
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = plt.matplotlib.transforms.Affine2D().scale(scale_x, scale_y).translate(x.mean(), y.mean())
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    # Get values in specified order
    all_values = pca_df[grouping_col].unique()
    group_values = [v for v in selected_values if v in all_values]

    # Plot each group with ellipse
    for group_val in group_values:
        mask = pca_df[grouping_col] == group_val
        subset = pca_df[mask]

        if len(subset) < 3:
            continue

        color = get_color(group_val, grouping_type)
        label = f"{group_val} (n={len(subset)})"

        # Plot points
        ax.scatter(
            subset[pc1],
            subset[pc2],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=[color],
            edgecolors="black",
            linewidths=0.5,
            label=label,
            zorder=3,
        )

        # Add confidence ellipse
        confidence_ellipse(
            subset[pc1].values, subset[pc2].values, ax, edgecolor=color, linestyle="-", linewidth=2, alpha=0.7, zorder=2
        )

    # Get explained variance
    pc1_idx = int(pc1[2:]) - 1
    pc2_idx = int(pc2[2:]) - 1
    var1 = pca_df.attrs["explained_variance_ratio"][pc1_idx] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][pc2_idx] * 100

    ax.set_xlabel(f"{pc1} ({var1:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({var2:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_title(f"Bivariate Analysis: {pc1} vs {pc2}\n95% Confidence Ellipses", fontsize=14, fontweight="bold")

    # Add origin lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    ax.grid(alpha=0.3, linestyle="--", zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / f"bivariate_{pc1}_vs_{pc2}_ellipses.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    return fig


def plot_bivariate_vector_shift(pca_df, pc1, pc2, grouping_col, selected_values, grouping_type, output_dir):
    """
    Vector-only plot showing condition centroids and shifts.
    For control mode, shows centroid positions for each F1 condition.
    """
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get values in specified order
    all_values = pca_df[grouping_col].unique()
    group_values = [v for v in selected_values if v in all_values]

    # Add reference arrow showing expected improvement direction (+PC1, +PC2)
    ax.arrow(
        0,
        0,
        2,
        1,
        head_width=0.15,
        head_length=0.2,
        fc="lightgreen",
        ec="darkgreen",
        linewidth=3,
        alpha=0.3,
        label="Improvement direction\n(+PC1, +PC2)",
        zorder=1,
    )

    # Plot centroids and connections for each condition
    centroids = {}
    for group_val in group_values:
        mask = pca_df[grouping_col] == group_val
        subset = pca_df[mask]

        if len(subset) < 2:
            continue

        centroid = [subset[pc1].mean(), subset[pc2].mean()]
        centroids[group_val] = centroid

        color = get_color(group_val, grouping_type)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=400,
            c=[color],
            edgecolors="black",
            linewidths=2,
            label=f"{group_val} (n={len(subset)})",
            zorder=3,
            marker="o",
            alpha=0.8,
        )

    # Draw arrows between consecutive conditions (for pretraining comparisons)
    if grouping_type == "pretraining" and len(centroids) >= 2:
        # Order: n (naive) -> y (pretrained)
        if "n" in centroids and "y" in centroids:
            start = centroids["n"]
            end = centroids["y"]
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                head_width=0.15,
                head_length=0.2,
                fc="red",
                ec="darkred",
                linewidth=2.5,
                alpha=0.6,
                zorder=2,
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)

    pc1_var = pca_df.attrs["explained_variance_ratio"][int(pc1[2]) - 1] * 100
    pc2_var = pca_df.attrs["explained_variance_ratio"][int(pc2[2]) - 1] * 100

    ax.set_xlabel(f"{pc1} ({pc1_var:.1f}% variance)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({pc2_var:.1f}% variance)", fontsize=13, fontweight="bold")
    ax.set_title(f"Condition Centroids: {pc1} vs {pc2}", fontsize=14, fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    output_path = Path(output_dir) / f"bivariate_{pc1}_vs_{pc2}_vector_shift.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    return fig


def plot_3d_pca(pca_df, grouping_col, selected_values, grouping_type, output_dir):
    """Create 3D scatter plot of first three PCs"""
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns or "PC3" not in pca_df.columns:
        print(f"  Warning: Not enough PCs for 3D plot")
        return None

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get values in specified order
    all_values = pca_df[grouping_col].unique()
    group_values = [v for v in selected_values if v in all_values]

    # Plot each group
    for group_val in group_values:
        mask = pca_df[grouping_col] == group_val
        subset = pca_df[mask]

        if len(subset) == 0:
            continue

        color = get_color(group_val, grouping_type)
        label = f"{group_val} (n={len(subset)})"

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=[color],
            edgecolors="black",
            linewidths=0.5,
            marker="o",
            label=label,
            depthshade=True,
        )

    # Get explained variance
    var1 = pca_df.attrs["explained_variance_ratio"][0] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][1] * 100
    var3 = pca_df.attrs["explained_variance_ratio"][2] * 100

    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10, fontweight="bold")
    ax.set_zlabel(f"PC3 ({var3:.1f}%)", fontsize=10, fontweight="bold")
    ax.set_title("3D PCA: PC1 vs PC2 vs PC3", fontsize=14, fontweight="bold", pad=20)

    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir) / f"pca_3d_scatter.{PLOT_FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path.name}")

    return fig


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="F1 PCA Analysis - Control Mode")
    parser.add_argument("--n-components", type=int, default=3, help="Number of principal components to compute")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print("=" * 80)
    print("F1 PCA ANALYSIS - CONTROL MODE")
    print("=" * 80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load and prepare data
    df = load_dataset()
    f1_condition_col, pretraining_col = detect_columns(df)
    df = filter_dataset(df, f1_condition_col)

    # Run PCA (once, using F1_condition as grouping for metadata)
    X_scaled, scaler, valid_metrics, metadata_df = prepare_pca_data(df, PCA_METRICS, grouping_col=f1_condition_col)
    pca, X_pca = perform_pca(X_scaled, n_components=args.n_components)
    pca_df = create_pca_dataframe(X_pca, metadata_df, pca, valid_metrics)

    # Save variance and loadings plots (once at root level)
    print("\n" + "=" * 80)
    print("GENERATING SHARED PLOTS")
    print("=" * 80)
    plot_explained_variance(pca, output_dir)
    plot_loadings_heatmap(pca, valid_metrics, output_dir)

    # Define two analyses
    analyses = [
        {
            "name": "by_f1_condition",
            "grouping_col": f1_condition_col,
            "grouping_type": "f1_condition",
            "selected_values": SELECTED_F1_CONDITIONS,
            "label": "F1 Condition",
        },
        {
            "name": "by_pretraining",
            "grouping_col": pretraining_col,
            "grouping_type": "pretraining",
            "selected_values": sorted(pca_df[pretraining_col].unique()),
            "label": "Pretraining",
        },
    ]

    # Run each analysis
    for analysis in analyses:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: {analysis['label']}")
        print("=" * 80)

        # Create subdirectory
        analysis_dir = output_dir / analysis["name"]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {analysis_dir}")

        # Generate plots
        print("\nGenerating boxplots...")
        for i in range(1, args.n_components + 1):
            plot_pc_boxplot(
                pca_df,
                f"PC{i}",
                analysis["grouping_col"],
                analysis["selected_values"],
                analysis["grouping_type"],
                analysis_dir,
            )

        print("\nGenerating 2D scatter plots...")
        plot_pca_scatter(
            pca_df,
            "PC1",
            "PC2",
            analysis["grouping_col"],
            analysis["selected_values"],
            analysis["grouping_type"],
            analysis_dir,
        )
        plot_pca_scatter(
            pca_df,
            "PC1",
            "PC3",
            analysis["grouping_col"],
            analysis["selected_values"],
            analysis["grouping_type"],
            analysis_dir,
            suffix="_alt",
        )

        print("\nGenerating bivariate plots...")
        plot_bivariate_confidence_ellipses(
            pca_df,
            "PC1",
            "PC2",
            analysis["grouping_col"],
            analysis["selected_values"],
            analysis["grouping_type"],
            analysis_dir,
        )
        plot_bivariate_confidence_ellipses(
            pca_df,
            "PC1",
            "PC3",
            analysis["grouping_col"],
            analysis["selected_values"],
            analysis["grouping_type"],
            analysis_dir,
        )
        plot_bivariate_vector_shift(
            pca_df,
            "PC1",
            "PC2",
            analysis["grouping_col"],
            analysis["selected_values"],
            analysis["grouping_type"],
            analysis_dir,
        )

        if args.n_components >= 3:
            print("\nGenerating 3D scatter plot...")
            plot_3d_pca(
                pca_df, analysis["grouping_col"], analysis["selected_values"], analysis["grouping_type"], analysis_dir
            )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
