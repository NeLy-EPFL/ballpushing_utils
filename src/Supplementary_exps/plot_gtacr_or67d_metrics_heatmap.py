#!/usr/bin/env python3
"""
Generate detailed metrics heatmap with dendrograms for Gtacr OR67d experiments.

This script creates comprehensive visualizations showing:
- Two-way hierarchical clustering (genotypes and metrics)
- Color-coded Cohen's d effect sizes
- Brain region annotations
- All genotypes included (no filtering by consistency scores)

The comparison is inverted to show: GtacrxOR67d (test) vs Controls
where Controls = GtacrxEmptyGal4 and PRxOR67d

Usage:
    python plot_gtacr_or67d_metrics_heatmap.py [--stats-file PATH] [--output-dir PATH]

Arguments:
    --stats-file: Path to Mann-Whitney statistics CSV (default: auto-detect from Genotype_Mannwhitney folder)
    --output-dir: Directory to save plots (default: same as stats file)
"""

import sys
import os
import argparse
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, Normalize

warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from Config.config import Config

    nickname_to_brainregion = dict(zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"]))
    color_dict = Config.color_dict
except Exception as e:
    print(f"⚠️  Could not load region mapping from Config: {e}")
    nickname_to_brainregion = {}
    color_dict = {}


# Try to reuse metric display names and canonical metrics order from the PCA pipeline
try:
    from PCA.plot_detailed_metric_statistics import METRIC_DISPLAY_NAMES as PCA_METRIC_DISPLAY_NAMES
    from PCA.plot_detailed_metric_statistics import METRICS_PATH as PCA_METRICS_PATH
except Exception:
    PCA_METRIC_DISPLAY_NAMES = {}
    PCA_METRICS_PATH = None

# Clip Cohen's d color scale to this absolute value (set to None to disable)
CLIP_EFFECTS = 1.5


def get_display_name(metric_name):
    """Return informative display name for a metric (fallback to original)."""
    return PCA_METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def load_canonical_metric_order():
    """Attempt to load canonical metric ordering used by PCA plots.

    Searches a few likely locations and returns a list of metric keys in order.
    Returns empty list if none found.
    """
    candidates = []
    if PCA_METRICS_PATH:
        candidates.append(Path(PCA_METRICS_PATH))

    # Common paths relative to repository src/PCA/metrics_lists
    base = Path(__file__).parent.parent / "PCA" / "metrics_lists"
    candidates.extend(
        [
            base / "final_metrics_for_pca_alt.txt",
            base / "final_metrics_for_pca.txt",
            base / "final_metrics_for_pca_list.txt",
        ]
    )

    # Also check for a canonical metrics file explicitly named 'canonical_metrics_order.txt'
    candidates.insert(0, base / "canonical_metrics_order.txt")

    for p in candidates:
        try:
            if p and p.exists():
                with p.open("r") as fh:
                    lines = [l.strip() for l in fh if l.strip()]
                    if lines:
                        return lines
        except Exception:
            continue

    return []


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate metrics heatmap with dendrograms for Gtacr OR67d experiments"
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="Path to Mann-Whitney statistics CSV (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as stats file)",
    )
    parser.add_argument(
        "--control-genotype",
        type=str,
        default="GtacrxOR67d",
        help="Name of control genotype (default: GtacrxOR67d)",
    )
    return parser.parse_args()


def find_stats_files():
    """Auto-detect Mann-Whitney statistics files in ATR subdirectories"""
    base_path = Path("/mnt/upramdya_data/MD/Gtacr/Plots/summaries/Genotype_Mannwhitney")

    # Look for ATR subdirectories
    atr_dirs = sorted([d for d in base_path.glob("ATR_*") if d.is_dir()])

    if not atr_dirs:
        # Fallback to old structure (no ATR subdirectories)
        stats_file = base_path / "genotype_mannwhitney_statistics.csv"
        if stats_file.exists():
            return [("all", stats_file)]
        else:
            raise FileNotFoundError(
                f"Could not find statistics file at {stats_file}. "
                "Please run the Mann-Whitney analysis first or specify --stats-file"
            )

    # Find stats files in ATR subdirectories
    stats_files = []
    for atr_dir in atr_dirs:
        stats_file = atr_dir / "genotype_mannwhitney_statistics.csv"
        if stats_file.exists():
            atr_condition = atr_dir.name.replace("ATR_", "")
            stats_files.append((atr_condition, stats_file))

    if not stats_files:
        raise FileNotFoundError(
            f"Could not find statistics files in ATR subdirectories at {base_path}. "
            "Please run the Mann-Whitney analysis first or specify --stats-file"
        )

    return stats_files


def load_statistics(stats_file):
    """Load Mann-Whitney U test statistics"""
    print(f"📊 Loading statistics from: {stats_file}")

    stats_df = pd.read_csv(stats_file)
    print(f"   Loaded {len(stats_df)} statistical comparisons")
    print(f"   Metrics: {stats_df['Metric'].nunique()}")
    print(f"   Genotypes: {stats_df['Test'].nunique()}")

    return stats_df


def load_raw_data():
    """
    Load the raw summary data to calculate proper Cohen's d effect sizes.
    Applies genotype harmonization.

    Returns
    -------
    pd.DataFrame
        Raw data with all metrics
    """
    data_path = "/mnt/upramdya_data/MD/Gtacr/Datasets/251219_10_summary_OR67d_Gtacr_Data/summary/pooled_summary.feather"
    print(f"   Loading raw data from: {data_path}")
    df = pd.read_feather(data_path)
    print(f"   Loaded {len(df)} rows")

    # Harmonize genotype naming (synonyms)
    if "Genotype" in df.columns:
        genotype_mapping = {
            "OGS32xOR67d": "GtacrxOR67d",
            "OGS32xEmptyGal4": "GtacrxEmptyGal4",
        }
        df["Genotype"] = df["Genotype"].replace(genotype_mapping)
        print(f"   Applied genotype harmonization")

    # Handle ATR column if present
    if "ATR" in df.columns:
        if pd.api.types.is_categorical_dtype(df["ATR"]):
            df["ATR"] = df["ATR"].astype(str)
        df["ATR"] = df["ATR"].fillna("yes")

    return df


def calculate_cohens_d(group1_values, group2_values):
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_standard_deviation

    Parameters
    ----------
    group1_values : array-like
        Test group values (GtacrxOR67d)
    group2_values : array-like
        Control group values

    Returns
    -------
    float
        Cohen's d effect size (positive = group1 > group2)
    """
    n1, n2 = len(group1_values), len(group2_values)
    if n1 < 2 or n2 < 2:
        return 0.0

    # Remove NaN values
    group1_clean = group1_values.dropna() if hasattr(group1_values, "dropna") else pd.Series(group1_values).dropna()
    group2_clean = group2_values.dropna() if hasattr(group2_values, "dropna") else pd.Series(group2_values).dropna()

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return 0.0

    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    var1, var2 = np.var(group1_clean, ddof=1), np.var(group2_clean, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(group1_clean), len(group2_clean)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def build_significance_matrix(stats_df, control_genotype, raw_data, atr_condition=None):
    """
    Build matrix where:
    - Rows = control genotypes (what was "Test" in stats file)
    - Columns = metrics
    - Values = Cohen's d effect sizes (GtacrxOR67d vs control)

    Sign: +1 if GtacrxOR67d > control, -1 if GtacrxOR67d < control
    Magnitude: Cohen's d effect size (standardized difference)

    IMPORTANT: The stats file has GtacrxOR67d as "Control" and the actual controls
    as "Test", so we invert the comparison to show GtacrxOR67d vs Controls.

    Parameters
    ----------
    atr_condition : str, optional
        Filter raw data by ATR condition (e.g., "yes", "no")
    """
    print("   Calculating proper Cohen's d effect sizes from raw data...")

    # Filter raw data by ATR condition if specified
    if atr_condition and "ATR" in raw_data.columns:
        raw_data = raw_data[raw_data["ATR"] == atr_condition].copy()
        print(f"   Filtered to ATR={atr_condition}: {len(raw_data)} rows")

    # Get unique genotypes (what was called "Test" in stats, but are actually controls)
    # and metrics
    genotypes = sorted(stats_df["Test"].unique())
    metrics = sorted(stats_df["Metric"].unique())

    # Rename genotypes to reflect the inverted comparison
    genotype_labels = [f"GtacrxOR67d vs {g}" for g in genotypes]

    # Initialize matrix with new labels
    matrix = pd.DataFrame(0.0, index=genotype_labels, columns=metrics)

    # Calculate Cohen's d for each genotype-metric combination
    for control_name in genotypes:
        for metric in metrics:
            # Get raw data for GtacrxOR67d (test group)
            test_data = raw_data[raw_data["Genotype"] == "GtacrxOR67d"][metric]

            # Get raw data for the control genotype
            control_data = raw_data[raw_data["Genotype"] == control_name][metric]

            # Calculate Cohen's d (GtacrxOR67d - control)
            if len(test_data) > 0 and len(control_data) > 0:
                cohens_d_value = calculate_cohens_d(test_data, control_data)
            else:
                cohens_d_value = 0.0

            # Store with new label
            genotype_label = f"GtacrxOR67d vs {control_name}"
            matrix.loc[genotype_label, metric] = cohens_d_value

    print(f"   Cohen's d range: [{matrix.values.min():.3f}, {matrix.values.max():.3f}]")
    # Try to apply canonical metric ordering (if available) so downstream plots
    # use the same metric order across scripts.
    canonical = load_canonical_metric_order()
    if canonical:
        # Keep only canonical metrics that are present, preserve their order
        ordered = [m for m in canonical if m in matrix.columns]
        remaining = [m for m in matrix.columns if m not in ordered]
        new_cols = ordered + remaining
        matrix = matrix.loc[:, new_cols]
        print(f"   Applied canonical metric order: {len(ordered)} metrics reordered")

    return matrix


def colour_y_ticklabels(ax, nickname_to_region, color_dict):
    """Paint y-tick labels according to genotype's brain region"""
    for tick in ax.get_yticklabels():
        label_text = tick.get_text()
        region = nickname_to_region.get(label_text, "Unknown")
        color = color_dict.get(region, "black")
        tick.set_color(color)


def plot_two_way_dendrogram(
    matrix,
    stats_df,
    output_dir,
    control_genotype,
    nickname_to_region=None,
    color_dict_regions=None,
    fig_size=(20, 12),
    raw_data=None,
):
    """
    Create two-way dendrogram heatmap:
    - Rows (genotypes) clustered by their metric profiles
    - Columns (metrics) clustered by correlation

    Parameters
    ----------
    stats_df : pd.DataFrame
        Original statistics dataframe (needed for p-values for stars)
    """
    print(f"\n🎨 Creating two-way dendrogram heatmap...")

    if matrix.empty:
        print("❌ Empty matrix, cannot create dendrogram")
        return

    # Compute linkages
    print("   Computing hierarchical clustering...")

    # Row clustering (genotypes) - only if we have more than 2 genotypes
    # With only 2 genotypes, clustering doesn't make sense
    if matrix.shape[0] > 2:
        row_linkage = linkage(pdist(matrix.values, metric="euclidean"), method="ward")
        row_dendro = dendrogram(row_linkage, no_plot=True)
        row_order = row_dendro["leaves"]
    else:
        row_linkage = None
        row_order = list(range(matrix.shape[0]))

    # Column clustering (metrics) - use correlation distance
    # First, check for a canonical metric ordering file. If present, use it
    # for column ordering and skip column dendrogram (canonical was produced
    # by the PCA plotting routine based on correlation clustering).
    canonical = load_canonical_metric_order()
    if canonical:
        # Determine column order indices from canonical list
        col_order = [matrix.columns.get_loc(m) for m in canonical if m in matrix.columns]
        col_linkage = None
    elif matrix.shape[1] > 1:
        # Compute correlation matrix between metrics (columns)
        corr_matrix = matrix.corr()
        corr_matrix = corr_matrix.fillna(0.0)

        # Convert to distance
        distance_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(distance_matrix, 0.0)

        # Fix any negative values due to numerical precision
        distance_matrix = np.maximum(distance_matrix, 0.0)

        # Get condensed distance matrix
        n = distance_matrix.shape[0]
        col_distances = distance_matrix[np.triu_indices(n, k=1)]

        col_linkage = linkage(col_distances, method="average")
        col_dendro = dendrogram(col_linkage, no_plot=True)
        col_order = col_dendro["leaves"]
    else:
        col_linkage = None
        col_order = [0]

    # Reorder matrix
    matrix_ordered = matrix.iloc[row_order, col_order]

    # Calculate figure size based on number of metrics and genotypes
    # More metrics = wider figure, more genotypes = taller figure
    n_metrics = len(matrix_ordered.columns)
    n_genotypes = len(matrix_ordered.index)

    # Dynamic sizing: ensure enough space for labels (more generous spacing)
    fig_width = max(20, n_metrics * 0.7)  # Increased from 0.5 to 0.7
    fig_height = max(10, n_genotypes * 1.0)  # Increased from 0.8 to 1.0

    # Create figure with GridSpec - proper spacing for all elements
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        3,
        3,  # 3 rows: dendrogram, metric labels, heatmap; 3 cols: genotype labels, main, colorbar
        width_ratios=[2.0, 10.0, 0.4],  # More space for heatmap
        height_ratios=[2.0, 1.2, 5.0],  # More space for dendrogram and labels
        wspace=0.03,
        hspace=0.08,  # Small gap between sections to prevent overlap
    )

    # Create axes
    ax_top_dendro = fig.add_subplot(gs[0, 1])
    ax_metric_labels = fig.add_subplot(gs[1, 1])  # Separate axis for metric labels
    ax_genotype_labels = fig.add_subplot(gs[2, 0])
    ax_hm = fig.add_subplot(gs[2, 1])
    ax_cbar = fig.add_subplot(gs[2, 2])

    # Heatmap first (so we can sync dendrogram to it)
    # Set extent to have proper column centering for dendrogram alignment
    n_cols = matrix_ordered.shape[1]
    n_rows = matrix_ordered.shape[0]

    # Set vmin/vmax based on actual Cohen's d range
    max_abs_d = max(abs(matrix_ordered.values.min()), abs(matrix_ordered.values.max()))
    vmin, vmax = -max_abs_d, max_abs_d

    im = ax_hm.imshow(
        matrix_ordered.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5],  # Proper extent for alignment
    )

    # Add gridlines
    for i in range(matrix_ordered.shape[0] + 1):
        ax_hm.axhline(i - 0.5, color="gray", linewidth=0.5)
    for j in range(matrix_ordered.shape[1] + 1):
        ax_hm.axvline(j - 0.5, color="gray", linewidth=0.5)

    # Add significance stars (based on p-values from stats_df)
    # Need to map back to original stats to get p-values
    for i in range(matrix_ordered.shape[0]):
        for j in range(matrix_ordered.shape[1]):
            genotype_label = matrix_ordered.index[i]
            metric = matrix_ordered.columns[j]

            # Extract original control name from label "GtacrxOR67d vs [control]"
            control_name = genotype_label.replace("GtacrxOR67d vs ", "")

            # Look up p-value from stats_df
            mask = (stats_df["Test"] == control_name) & (stats_df["Metric"] == metric)
            if mask.any():
                p_val = stats_df.loc[mask, "pval_corrected"].values[0]

                # Determine significance stars
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = ""

                if stars:
                    # Choose annotation color based on background intensity
                    bg_val = matrix_ordered.values[i, j]
                    text_color = "white" if abs(bg_val) >= 0.5 else "black"
                    ax_hm.text(
                        j,
                        i,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=text_color,
                        fontweight="bold",
                    )

    # Remove all tick labels from heatmap
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    ax_hm.tick_params(axis="both", which="both", length=0)

    # Top dendrogram (metrics) - create AFTER heatmap
    if matrix.shape[1] > 1 and col_linkage is not None:
        dg = dendrogram(
            col_linkage,
            ax=ax_top_dendro,
            orientation="top",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="C0",
        )
        # Don't sync xlim - dendrogram has its own coordinate system
        # We'll handle alignment through the metric labels
    ax_top_dendro.axis("off")

    # Metric labels in separate axis - positioned between dendrogram and heatmap
    ax_metric_labels.axis("off")

    # Sync xlim with heatmap to ensure alignment
    ax_metric_labels.set_xlim(ax_hm.get_xlim())
    ax_metric_labels.set_ylim(0, 1)

    # Position labels at column centers - matching heatmap extent
    # Heatmap columns are centered at 0, 1, 2, ..., n-1
    x_positions = np.arange(len(matrix_ordered.columns))
    metric_labels_list = list(matrix_ordered.columns)
    display_labels = [get_display_name(l) for l in metric_labels_list]

    for x_pos, label in zip(x_positions, display_labels):
        ax_metric_labels.text(
            x_pos,
            0.2,  # Lower in the axis to be closer to heatmap
            label,
            ha="right",
            va="top",
            fontsize=7,
            rotation=45,
        )  # Genotype labels (left side)
    ax_genotype_labels.axis("off")
    ax_genotype_labels.set_ylim(ax_hm.get_ylim())

    # Match the y-positions with heatmap rows
    y_positions = np.arange(len(matrix_ordered.index))
    genotype_labels = list(matrix_ordered.index)

    for y_pos, genotype in zip(y_positions, genotype_labels):
        # Get color based on brain region
        region = nickname_to_region.get(genotype, "Unknown") if nickname_to_region else "Unknown"
        color = color_dict_regions.get(region, "black") if color_dict_regions else "black"

        # Add sample size annotation if raw_data provided
        label_text = genotype
        if raw_data is not None and "Genotype" in raw_data.columns:
            try:
                # Genotype labels are expected in the form "<left> vs <right>"
                # e.g. "GtacrxOR67d vs SomeControl". Split robustly and count
                # rows for each side in the provided (ATR-filtered) raw_data.
                left, right = genotype.split(" vs ", 1)
                left_n = int(raw_data[raw_data["Genotype"] == left].shape[0])
                right_n = int(raw_data[raw_data["Genotype"] == right].shape[0])
                label_text = f"{genotype}\n(n = {left_n} - {right_n})"
            except Exception:
                # If parsing/counting fails, silently skip sample-size annotation
                pass

        ax_genotype_labels.text(
            0.95,
            y_pos,
            label_text,
            ha="right",
            va="center",
            fontsize=10,
            color=color,
        )

    # Colorbar - set range based on actual Cohen's d values (with optional clipping)
    max_abs_d = max(abs(matrix_ordered.values.min()), abs(matrix_ordered.values.max()))
    vmin, vmax = -max_abs_d, max_abs_d
    if CLIP_EFFECTS is not None:
        clip = CLIP_EFFECTS
        if max_abs_d > clip:
            vmin, vmax = -clip, clip

    norm = Normalize(vmin=vmin, vmax=vmax)
    pos = ax_cbar.get_position()
    new_height = pos.height * 0.35
    new_y = pos.y0 + (pos.height - new_height) / 2
    ax_cbar.set_position((pos.x0, new_y, pos.width, new_height))

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label("Cohen's d\n(Blue: Lower, Red: Higher)", fontsize=8)

    # Add Cohen's d interpretation (show clipped edges explicitly when clipping active)
    tick_positions = np.linspace(vmin, vmax, 5)
    tick_labels = [f"{pos:.2f}" for pos in tick_positions]
    if CLIP_EFFECTS is not None and max_abs_d > CLIP_EFFECTS:
        tick_labels[0] = f"< {vmin:.2f}"
        tick_labels[-1] = f"> {vmax:.2f}"
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=7)

    # Title
    fig.suptitle(
        f"GtacrxOR67d vs Controls - Metrics Heatmap (Cohen's d Effect Sizes)\n"
        f"({len(matrix_ordered)} comparisons × {len(matrix_ordered.columns)} metrics)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_file = output_dir / "gtacr_or67d_metrics_dendrogram.png"
    pdf_file = output_dir / "gtacr_or67d_metrics_dendrogram.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")

    print(f"   💾 Saved heatmap: {png_file}")
    print(f"   💾 Saved heatmap: {pdf_file}")

    # Save matrix
    matrix_file = output_dir / "gtacr_or67d_significance_matrix.csv"
    matrix_ordered.to_csv(matrix_file)
    print(f"   💾 Saved matrix: {matrix_file}")

    plt.close()

    return matrix_ordered


def plot_simple_heatmap(
    matrix,
    stats_df,
    output_dir,
    control_genotype,
    nickname_to_region=None,
    color_dict_regions=None,
    fig_size=(16, 12),
    raw_data=None,
):
    """
    Create simple heatmap without dendrograms:
    - Genotypes sorted by brain region
    - Metrics sorted by correlation

    Parameters
    ----------
    stats_df : pd.DataFrame
        Original statistics dataframe (needed for p-values for stars)
    """
    print(f"\n🎨 Creating simple heatmap...")

    if matrix.empty:
        print("❌ Empty matrix, cannot create heatmap")
        return

    # Sort genotypes by brain region
    if nickname_to_region:
        genotype_regions = pd.DataFrame(
            {"genotype": matrix.index, "region": [nickname_to_region.get(g, "Unknown") for g in matrix.index]}
        )
        genotype_regions = genotype_regions.sort_values(["region", "genotype"])
        row_order = genotype_regions["genotype"].tolist()
    else:
        row_order = sorted(matrix.index)

    # Sort metrics by correlation
    # Prefer canonical metric order if available
    canonical = load_canonical_metric_order()
    if canonical and len([m for m in canonical if m in matrix.columns]) > 0:
        col_order = [m for m in canonical if m in matrix.columns]
        # append any metrics not in canonical list at the end
        remaining = [m for m in matrix.columns if m not in col_order]
        col_order = col_order + remaining
    elif matrix.shape[1] > 1:
        corr_matrix = matrix.corr()
        corr_matrix = corr_matrix.fillna(0.0)

        distance_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(distance_matrix, 0.0)

        # Fix any negative values due to numerical precision
        distance_matrix = np.maximum(distance_matrix, 0.0)

        n = distance_matrix.shape[0]
        col_distances = distance_matrix[np.triu_indices(n, k=1)]

        col_linkage = linkage(col_distances, method="average")
        col_dendro = dendrogram(col_linkage, no_plot=True)
        col_order = [matrix.columns[i] for i in col_dendro["leaves"]]
    else:
        col_order = list(matrix.columns)

    # Reorder matrix
    matrix_ordered = matrix.loc[row_order, col_order]

    # Create figure
    fig, (ax_main, ax_cbar) = plt.subplots(
        1, 2, figsize=fig_size, gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05}
    )

    # Set vmin/vmax based on Cohen's d range
    max_abs_d = max(abs(matrix_ordered.values.min()), abs(matrix_ordered.values.max()))
    vmin, vmax = -max_abs_d, max_abs_d
    if CLIP_EFFECTS is not None:
        clip = CLIP_EFFECTS
        if max_abs_d > clip:
            vmin, vmax = -clip, clip

    # Heatmap
    sns.heatmap(
        matrix_ordered,
        ax=ax_main,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=True,
        yticklabels=True,
    )

    # Add significance stars (based on p-values from stats_df)
    for i in range(matrix_ordered.shape[0]):
        for j in range(matrix_ordered.shape[1]):
            genotype_label = matrix_ordered.index[i]
            metric = matrix_ordered.columns[j]

            # Extract original control name from label "GtacrxOR67d vs [control]"
            control_name = genotype_label.replace("GtacrxOR67d vs ", "")

            # Look up p-value from stats_df
            mask = (stats_df["Test"] == control_name) & (stats_df["Metric"] == metric)
            if mask.any():
                p_val = stats_df.loc[mask, "pval_corrected"].values[0]

                # Determine significance stars
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = ""

                if stars:
                    # Choose annotation color based on background intensity
                    try:
                        bg_val = matrix_ordered.values[i, j]
                    except Exception:
                        bg_val = 0
                    text_color = "white" if abs(bg_val) >= 0.5 else "black"
                    ax_main.text(
                        j + 0.5,
                        i + 0.5,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=text_color,
                        fontweight="bold",
                    )

    # Labels - replace metric names with informative display names
    ax_main.set_xticklabels(
        [get_display_name(l.get_text()) for l in ax_main.get_xticklabels()], rotation=45, ha="right", fontsize=9
    )

    # Add sample sizes to y-tick labels if raw_data provided
    y_labels = []
    for tick in ax_main.get_yticklabels():
        genotype = tick.get_text()
        label_text = genotype

        # Try to add sample size annotation
        if raw_data is not None and "Genotype" in raw_data.columns:
            try:
                # Split genotype label (format: "left vs right")
                left, right = genotype.split(" vs ", 1)
                left_n = int(raw_data[raw_data["Genotype"] == left].shape[0])
                right_n = int(raw_data[raw_data["Genotype"] == right].shape[0])
                label_text = f"{genotype}\n(n = {left_n} - {right_n})"
            except Exception:
                pass

        y_labels.append(label_text)

    ax_main.set_yticklabels(y_labels, rotation=0, fontsize=9)

    # Color genotype labels by brain region
    if nickname_to_region and color_dict_regions:
        for tick in ax_main.get_yticklabels():
            label_text = tick.get_text().split("\n")[0]  # Get genotype name (before newline)
            region = nickname_to_region.get(label_text, "Unknown")
            color = color_dict_regions.get(region, "black")
            tick.set_color(color)

    # Colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label("Cohen's d\n(Blue: Lower, Red: Higher)", fontsize=10)

    tick_positions = np.linspace(vmin, vmax, 5)
    tick_labels = [f"{pos:.2f}" for pos in tick_positions]
    if CLIP_EFFECTS is not None and max_abs_d > CLIP_EFFECTS:
        tick_labels[0] = f"< {vmin:.2f}"
        tick_labels[-1] = f"> {vmax:.2f}"
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    # Title
    ax_main.set_title(
        f"GtacrxOR67d vs Controls - Simple Heatmap (Cohen's d Effect Sizes)\n"
        f"(Comparisons by brain region, Metrics by correlation)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_main.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Comparisons (GtacrxOR67d vs Controls)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    png_file = output_dir / "gtacr_or67d_metrics_simple_heatmap.png"
    pdf_file = output_dir / "gtacr_or67d_metrics_simple_heatmap.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")

    print(f"   💾 Saved simple heatmap: {png_file}")
    print(f"   💾 Saved simple heatmap: {pdf_file}")

    plt.close()


def main():
    """Main execution"""
    args = parse_args()

    print(f"\n{'='*60}")
    print("GTACR OR67D - METRICS HEATMAP GENERATION")
    print(f"{'='*60}\n")

    # Find stats files (may be multiple ATR conditions)
    if args.stats_file:
        stats_files = [("specified", Path(args.stats_file))]
    else:
        stats_files = find_stats_files()

    print(f"📊 Found {len(stats_files)} statistics file(s) to process:")
    for atr_cond, stats_file in stats_files:
        print(f"   ATR={atr_cond}: {stats_file}")

    # Load raw data once (will be filtered per ATR condition)
    print(f"\n📊 Loading raw data for Cohen's d calculation...")
    raw_data = load_raw_data()

    # Process each ATR condition
    for atr_condition, stats_file in stats_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING ATR={atr_condition}")
        print(f"{'='*80}\n")

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / f"ATR_{atr_condition}"
        else:
            output_dir = stats_file.parent / "heatmaps"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}\n")

        # Load statistics
        stats_df = load_statistics(stats_file)

        # Build significance matrix for this ATR condition
        print(f"\n🔨 Building significance matrix for ATR={atr_condition}...")
        matrix = build_significance_matrix(
            stats_df, args.control_genotype, raw_data, atr_condition=atr_condition if atr_condition != "all" else None
        )
        print(f"   Matrix shape: {matrix.shape} (genotypes × metrics)")
        print(f"   Genotypes: {len(matrix)}")
        print(f"   Metrics: {len(matrix.columns)}")

        # Generate plots
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS FOR ATR={atr_condition}")
        print(f"{'='*60}")

        # Two-way dendrogram
        # Filter raw_data to the ATR condition used for the matrix so sample
        # size annotations match the effect-size calculations.
        if atr_condition != "all" and raw_data is not None and "ATR" in raw_data.columns:
            filtered_raw = raw_data[raw_data["ATR"] == atr_condition].copy()
        else:
            filtered_raw = raw_data.copy() if raw_data is not None else None

        plot_two_way_dendrogram(
            matrix,
            stats_df,
            output_dir,
            args.control_genotype,
            nickname_to_region=nickname_to_brainregion,
            color_dict_regions=color_dict,
            raw_data=filtered_raw,
        )

        # canonical metric order is produced by the PCA plotting routine;
        # nothing to save here.

        # Simple heatmap
        plot_simple_heatmap(
            matrix,
            stats_df,
            output_dir,
            args.control_genotype,
            nickname_to_region=nickname_to_brainregion,
            color_dict_regions=color_dict,
            raw_data=filtered_raw,
        )

        print(f"\n✅ ATR={atr_condition} heatmaps complete")
        print(f"📁 Saved to: {output_dir}")

    print(f"\n{'='*60}")
    print("✅ ALL HEATMAP GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(stats_files)} ATR condition(s)")


if __name__ == "__main__":
    main()
