#!/usr/bin/env python3
"""
Generate alternative dendrogram visualizations for metric effect sizes.

This script focuses on dendrogram-based plots and creates four variants:
1) Grayscale heatmap of absolute Cohen's d (directionless)
2) Grayscale heatmap of absolute Cohen's d + red/blue tile outlines for direction
3) Bubble plot with fixed red/blue colors and circle size = absolute Cohen's d
4) Bubble plot in grayscale with circle size = absolute Cohen's d

Input files are expected from plot_detailed_metric_statistics.py outputs:
- metric_stats_results_*.csv
- metric_correlations.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist

try:
    import Config

    HAS_CONFIG = True
except Exception:
    Config = None
    HAS_CONFIG = False


# Configure editable text in vector outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


DEFAULT_INPUT_DIR = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Detailed_metrics_statistics"
CELL_SIZE = 10.0
CELL_CENTER = CELL_SIZE / 2.0


METRIC_DISPLAY_NAMES = {
    "pulling_ratio": "Proportion pull vs push",
    "distance_ratio": "Dist. ball moved / corridor length",
    "distance_moved": "Dist. ball moved",
    "pulled": "Signif. pulling events (#)",
    "max_event": "Event max. ball displ.",
    "number_of_pauses": "Long pauses (#)",
    "first_major_event": "First major event",
    "significant_ratio": "Fraction signif. events",
    "max_distance": "Max ball displacement",
    "chamber_ratio": "Fraction time in chamber",
    "nb_events": "Ball contact events (#)",
    "persistence_at_end": "Fraction time near end",
    "time_chamber_beginning": "Time in chamber early",
    "normalized_speed": "Normalized speed",
    "first_major_event_time": "First major event time",
    "max_event_time": "Max event time",
    "nb_freeze": "Short pauses (#)",
    "flailing": "Front leg movement",
    "speed_during_interactions": "Speed during contact",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing ball",
    "interaction_persistence": "Avg interaction duration",
    "chamber_exit_time": "Time to first chamber exit",
    "speed_trend": "Speed trend",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dendrogram effect-size variants")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing metric statistics and correlation CSV outputs",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to metric_stats_results_*.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--correlation-csv",
        default=None,
        help="Path to metric_correlations.csv (defaults to <input-dir>/metric_correlations.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output plots (defaults to <input-dir>/dendrogram_variants)",
    )
    parser.add_argument(
        "--clip-effects",
        type=float,
        default=1.5,
        help="Clip Cohen's d to +/- this value before plotting (set <=0 to disable)",
    )
    parser.add_argument(
        "--bubble-unclipped",
        action="store_true",
        default=True,
        help="Use unclipped Cohen's d for bubble-size variants (default: enabled)",
    )
    parser.add_argument(
        "--row-linkage",
        default="ward",
        choices=["ward", "single", "complete", "average"],
        help="Linkage method for row clustering",
    )
    parser.add_argument(
        "--col-linkage",
        default="ward",
        choices=["ward", "single", "complete", "average"],
        help="Linkage method for column clustering",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=22.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=16.0,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--metric-rotation",
        type=float,
        default=55.0,
        help="Rotation in degrees for metric labels",
    )
    return parser.parse_args()


def _find_latest_results_csv(input_dir: Path) -> Path:
    candidates = sorted(input_dir.glob("metric_stats_results_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No metric_stats_results_*.csv found in {input_dir}")
    return candidates[-1]


def _load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    input_dir = Path(args.input_dir)
    if args.results_csv:
        results_csv = Path(args.results_csv)
    else:
        results_csv = _find_latest_results_csv(input_dir)

    if args.correlation_csv:
        correlation_csv = Path(args.correlation_csv)
    else:
        correlation_csv = input_dir / "metric_correlations.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    if not correlation_csv.exists():
        raise FileNotFoundError(f"Correlation CSV not found: {correlation_csv}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "dendrogram_variants")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_csv)
    corr_df = pd.read_csv(correlation_csv, index_col=0)

    if "genotype" not in results_df.columns:
        raise ValueError("Results CSV must contain a 'genotype' column")

    print(f"Using results CSV: {results_csv}")
    print(f"Using correlation CSV: {correlation_csv}")
    print(f"Output directory: {output_dir}")

    return results_df, corr_df, results_csv, correlation_csv, output_dir


def _get_metric_names(results_df: pd.DataFrame, corr_df: pd.DataFrame) -> List[str]:
    corr_metrics = set(corr_df.columns.tolist())
    metric_names: List[str] = []

    for col in results_df.columns:
        if col.endswith("_cohens_d"):
            metric = col[: -len("_cohens_d")]
            if metric in corr_metrics:
                metric_names.append(metric)

    metric_names = sorted(set(metric_names))
    if not metric_names:
        raise ValueError("No metric columns ending with '_cohens_d' matched correlation matrix columns")

    return metric_names


def _build_signed_matrix(
    results_df: pd.DataFrame,
    metric_names: List[str],
    clip_effects: float | None,
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=results_df["genotype"].astype(str).values, columns=metric_names, dtype=float)

    for _, row in results_df.iterrows():
        genotype = str(row["genotype"])
        for metric in metric_names:
            d_col = f"{metric}_cohens_d"
            value = row.get(d_col, np.nan)
            if pd.isna(value):
                value = 0.0
            value = float(value)
            if clip_effects is not None:
                value = float(np.clip(value, -clip_effects, clip_effects))
            matrix.loc[genotype, metric] = value

    matrix = matrix.fillna(0.0)
    return matrix


def _compute_orders(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    row_mode: str,
    row_linkage: str,
    col_linkage: str,
):
    if row_mode not in {"signed", "absolute"}:
        raise ValueError("row_mode must be 'signed' or 'absolute'")

    row_data = signed_matrix.values if row_mode == "signed" else np.abs(signed_matrix.values)

    row_Z = None
    if row_data.shape[0] > 1:
        row_Z = linkage(pdist(row_data, metric="euclidean"), method=row_linkage)
        row_order_idx = leaves_list(row_Z)
    else:
        row_order_idx = np.array([0])

    corr_subset = corr_df.loc[metric_names, metric_names].copy().fillna(0.0)
    col_distance = 1.0 - np.abs(corr_subset.values)
    np.fill_diagonal(col_distance, 0.0)
    col_distance = np.where(np.isfinite(col_distance), col_distance, 1.0)

    col_Z = None
    if col_distance.shape[0] > 1:
        condensed = col_distance[np.triu_indices(col_distance.shape[0], k=1)]
        condensed = np.where(np.isfinite(condensed), condensed, 1.0)
        col_Z = linkage(condensed, method=col_linkage)
        col_order_idx = leaves_list(col_Z)
    else:
        col_order_idx = np.array([0])

    row_order = signed_matrix.index[row_order_idx].tolist()
    col_order = [metric_names[i] for i in col_order_idx]
    return row_order, col_order, row_Z, col_Z


def _metric_label(metric_name: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def _get_brain_region_mappings() -> Tuple[dict, dict]:
    nickname_to_region = {}
    region_to_color = {}
    if HAS_CONFIG and Config is not None:
        try:
            split_registry = getattr(Config, "SplitRegistry", None)
            if split_registry is not None:
                nickname_to_region = dict(
                    zip(
                        split_registry["Nickname"],
                        split_registry["Simplified region"],
                    )
                )
            region_to_color = dict(getattr(Config, "color_dict", {}))
        except Exception:
            pass
    return nickname_to_region, region_to_color


def _setup_layout(fig_size: Tuple[float, float]):
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1.7, 1.2, 8.5, 0.35],
        height_ratios=[1.4, 8.5],
        wspace=0.06,
        hspace=0.03,
    )

    ax_top = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_main = fig.add_subplot(gs[1, 2])
    ax_cbar = fig.add_subplot(gs[1, 3])
    return fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar


def _draw_dendrograms(ax_top, ax_left, row_Z, col_Z):
    if col_Z is not None:
        dendrogram(col_Z, orientation="top", no_labels=True, color_threshold=None, ax=ax_top)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    if row_Z is not None:
        dendrogram(row_Z, orientation="left", no_labels=True, color_threshold=None, ax=ax_left)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for s in ax_left.spines.values():
        s.set_visible(False)


def _draw_label_axis(ax_labels, row_order: List[str], nickname_to_region: dict, region_to_color: dict):
    n_rows = len(row_order)
    ax_labels.set_xlim(0.0, 1.0)
    ax_labels.set_ylim(n_rows * CELL_SIZE, 0.0)
    ax_labels.axis("off")

    for i, genotype in enumerate(row_order):
        region = nickname_to_region.get(genotype, None)
        text_color = region_to_color.get(region, "black")
        ax_labels.text(
            0.99,
            i * CELL_SIZE + CELL_CENTER,
            genotype,
            ha="right",
            va="center",
            fontsize=9,
            color=text_color,
        )


def _finalize_main_axis(
    ax_main,
    col_order: List[str],
    metric_rotation: float,
):
    n_cols = len(col_order)
    x_centers = np.arange(n_cols) * CELL_SIZE + CELL_CENTER
    ax_main.set_xticks(x_centers)
    ax_main.set_yticks([])
    ax_main.set_xticklabels([_metric_label(m) for m in col_order], rotation=metric_rotation, ha="right", fontsize=9)
    ax_main.set_xlabel("Metrics", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("", fontsize=11, fontweight="bold")


def _add_cell_grid(ax_main, n_rows: int, n_cols: int, color: str = "0.85", lw: float = 0.35):
    ax_main.set_xticks(np.arange(0.0, (n_cols + 1) * CELL_SIZE, CELL_SIZE), minor=True)
    ax_main.set_yticks(np.arange(0.0, (n_rows + 1) * CELL_SIZE, CELL_SIZE), minor=True)
    ax_main.grid(which="minor", color=color, linewidth=lw)
    ax_main.tick_params(which="minor", bottom=False, left=False)


def _set_main_limits(ax_main, n_rows: int, n_cols: int):
    ax_main.set_xlim(0.0, n_cols * CELL_SIZE)
    ax_main.set_ylim(n_rows * CELL_SIZE, 0.0)
    ax_main.set_aspect("auto")


def _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main):
    ax_top.set_xlim(ax_main.get_xlim())
    ax_left.set_ylim(ax_main.get_ylim())
    ax_labels.set_ylim(ax_main.get_ylim())


def _save_figure(fig, output_dir: Path, stem: str):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")


def _save_orders_csv(output_dir: Path, stem: str, row_order: List[str], col_order: List[str]):
    pd.Series(row_order, name="genotype").to_csv(output_dir / f"{stem}_row_order.csv", index=False)
    pd.Series(col_order, name="metric").to_csv(output_dir / f"{stem}_col_order.csv", index=False)


def _gray_bin_color(abs_d: float) -> str:
    if abs_d < 0.2:
        return "#ffffff"
    if abs_d < 0.5:
        return "#e0e0e0"
    if abs_d < 0.8:
        return "#b3b3b3"
    if abs_d < 1.2:
        return "#808080"
    return "#4d4d4d"


def _draw_solid_grayscale_tiles(ax_main, abs_values: np.ndarray):
    n_rows, n_cols = abs_values.shape
    _set_main_limits(ax_main, n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            face = _gray_bin_color(float(abs_values[i, j]))
            rect = Rectangle(
                (j * CELL_SIZE, i * CELL_SIZE),
                CELL_SIZE,
                CELL_SIZE,
                facecolor=face,
                edgecolor="white",
                linewidth=1.0,
            )
            ax_main.add_patch(rect)


def _add_absd_bin_legend(ax_cbar):
    ax_cbar.axis("off")
    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="0.5", label="|d| < 0.2"),
        Rectangle((0, 0), 1, 1, facecolor="#e0e0e0", edgecolor="0.5", label="0.2 <= |d| < 0.5"),
        Rectangle((0, 0), 1, 1, facecolor="#b3b3b3", edgecolor="0.5", label="0.5 <= |d| < 0.8"),
        Rectangle((0, 0), 1, 1, facecolor="#808080", edgecolor="0.5", label="0.8 <= |d| < 1.2"),
        Rectangle((0, 0), 1, 1, facecolor="#4d4d4d", edgecolor="0.5", label="|d| >= 1.2"),
    ]
    ax_cbar.legend(
        handles=legend_items,
        title="Absolute Cohen's d",
        loc="center left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )


def plot_variant_1_abs_grayscale(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    output_dir: Path,
    row_linkage: str,
    col_linkage: str,
    fig_size: Tuple[float, float],
    metric_rotation: float,
    stem_suffix: str = "",
):
    row_order, col_order, row_Z, col_Z = _compute_orders(
        signed_matrix,
        corr_df,
        metric_names,
        row_mode="absolute",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    ordered = signed_matrix.loc[row_order, col_order]
    abs_values = np.abs(ordered.values)
    nickname_to_region, region_to_color = _get_brain_region_mappings()

    fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar = _setup_layout(fig_size)
    _draw_dendrograms(ax_top, ax_left, row_Z, col_Z)
    _draw_label_axis(ax_labels, row_order, nickname_to_region, region_to_color)

    _draw_solid_grayscale_tiles(ax_main, abs_values)
    _finalize_main_axis(ax_main, col_order, metric_rotation)
    _add_absd_bin_legend(ax_cbar)
    _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main)

    ax_top.set_title(
        "Dendrogram + Grayscale Absolute Effect Size (clustering uses |d|)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    stem = f"metric_dendrogram_absd_grayscale_heatmap{stem_suffix}"
    _save_figure(fig, output_dir, stem)
    _save_orders_csv(output_dir, stem, row_order, col_order)


def plot_variant_2_abs_grayscale_with_sign_outline(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    output_dir: Path,
    row_linkage: str,
    col_linkage: str,
    fig_size: Tuple[float, float],
    metric_rotation: float,
    stem_suffix: str = "",
):
    row_order, col_order, row_Z, col_Z = _compute_orders(
        signed_matrix,
        corr_df,
        metric_names,
        row_mode="absolute",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    ordered = signed_matrix.loc[row_order, col_order]
    abs_values = np.abs(ordered.values)
    nickname_to_region, region_to_color = _get_brain_region_mappings()

    fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar = _setup_layout(fig_size)
    _draw_dendrograms(ax_top, ax_left, row_Z, col_Z)
    _draw_label_axis(ax_labels, row_order, nickname_to_region, region_to_color)

    _draw_solid_grayscale_tiles(ax_main, abs_values)

    pos_color = "#b2182b"
    neg_color = "#2166ac"
    neutral_color = "#666666"

    for i in range(ordered.shape[0]):
        for j in range(ordered.shape[1]):
            val = ordered.iloc[i, j]
            if val > 0:
                edge = pos_color
            elif val < 0:
                edge = neg_color
            else:
                edge = neutral_color

            rect = Rectangle(
                (j * CELL_SIZE, i * CELL_SIZE),
                CELL_SIZE,
                CELL_SIZE,
                fill=False,
                edgecolor=edge,
                linewidth=1.2,
            )
            ax_main.add_patch(rect)

    _finalize_main_axis(ax_main, col_order, metric_rotation)
    _add_absd_bin_legend(ax_cbar)
    _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main)

    legend_items = [
        Line2D([0], [0], color=pos_color, lw=2, label="Positive direction"),
        Line2D([0], [0], color=neg_color, lw=2, label="Negative direction"),
    ]
    ax_main.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    ax_top.set_title(
        "Grayscale |d| + Direction Outlines (clustering uses |d|)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    stem = f"metric_dendrogram_absd_grayscale_outline_signed{stem_suffix}"
    _save_figure(fig, output_dir, stem)
    _save_orders_csv(output_dir, stem, row_order, col_order)


def plot_variant_1b_abs_grayscale_signed_clustering(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    output_dir: Path,
    row_linkage: str,
    col_linkage: str,
    fig_size: Tuple[float, float],
    metric_rotation: float,
    stem_suffix: str = "",
):
    """Same as variant 1 but clusters on signed d instead of absolute d."""
    row_order, col_order, row_Z, col_Z = _compute_orders(
        signed_matrix,
        corr_df,
        metric_names,
        row_mode="signed",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    ordered = signed_matrix.loc[row_order, col_order]
    abs_values = np.abs(ordered.values)
    nickname_to_region, region_to_color = _get_brain_region_mappings()

    fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar = _setup_layout(fig_size)
    _draw_dendrograms(ax_top, ax_left, row_Z, col_Z)
    _draw_label_axis(ax_labels, row_order, nickname_to_region, region_to_color)

    _draw_solid_grayscale_tiles(ax_main, abs_values)
    _finalize_main_axis(ax_main, col_order, metric_rotation)
    _add_absd_bin_legend(ax_cbar)
    _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main)

    ax_top.set_title(
        "Dendrogram + Grayscale Absolute Effect Size (clustering uses signed d)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    stem = f"metric_dendrogram_absd_grayscale_heatmap_signed_clustering{stem_suffix}"
    _save_figure(fig, output_dir, stem)
    _save_orders_csv(output_dir, stem, row_order, col_order)


def _plot_bubble_matrix(
    ordered: pd.DataFrame,
    ax_main,
    vmax: float,
    color_mode: str,
):
    n_rows, n_cols = ordered.shape
    _set_main_limits(ax_main, n_rows, n_cols)

    x_vals = []
    y_vals = []
    sizes = []
    colors = []

    min_area = 8.0
    max_area = 650.0

    for i in range(n_rows):
        for j in range(n_cols):
            value = float(ordered.iloc[i, j])
            abs_value = abs(value)
            if abs_value <= 0:
                continue

            size_ratio = 0.0 if vmax <= 0 else min(abs_value / vmax, 1.0)
            area = min_area + (max_area - min_area) * size_ratio

            x_vals.append(j * CELL_SIZE + CELL_CENTER)
            y_vals.append(i * CELL_SIZE + CELL_CENTER)
            sizes.append(area)

            if color_mode == "signed_color":
                colors.append("#c43c39" if value > 0 else "#2f6db2")
            elif color_mode == "grayscale":
                colors.append("0.35")
            else:
                raise ValueError("Unknown bubble color_mode")

    if x_vals:
        ax_main.scatter(x_vals, y_vals, s=sizes, c=colors, marker="o", linewidths=0.0, alpha=0.92, zorder=3)

    _add_cell_grid(ax_main, n_rows, n_cols, color="0.88", lw=0.35)


def _add_bubble_legends(ax_main, vmax: float, signed: bool):
    if vmax <= 0:
        return

    size_levels = [0.2, 0.5, 1.0]
    size_labels = [f"|d|={vmax * lvl:.2f}" for lvl in size_levels]

    marker_sizes = [(0.03 + (0.44 - 0.03) * lvl) * 45.0 for lvl in size_levels]

    size_handles = [
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="0.3", markersize=ms, linewidth=0)
        for ms in marker_sizes
    ]

    size_legend = ax_main.legend(
        handles=size_handles,
        labels=size_labels,
        title="Circle size",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    ax_main.add_artist(size_legend)

    if signed:
        sign_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c43c39", markersize=8, label="Positive"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f6db2", markersize=8, label="Negative"),
        ]
        ax_main.legend(
            handles=sign_handles,
            title="Direction",
            loc="upper left",
            bbox_to_anchor=(1.01, 0.73),
            fontsize=8,
            title_fontsize=9,
            frameon=False,
        )


def plot_variant_3_bubbles_color(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    output_dir: Path,
    row_linkage: str,
    col_linkage: str,
    fig_size: Tuple[float, float],
    metric_rotation: float,
    stem_suffix: str = "",
):
    row_order, col_order, row_Z, col_Z = _compute_orders(
        signed_matrix,
        corr_df,
        metric_names,
        row_mode="signed",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    ordered = signed_matrix.loc[row_order, col_order]
    vmax = max(float(np.abs(ordered.values).max()), 1e-6)

    nickname_to_region, region_to_color = _get_brain_region_mappings()

    fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar = _setup_layout(fig_size)
    _draw_dendrograms(ax_top, ax_left, row_Z, col_Z)
    _draw_label_axis(ax_labels, row_order, nickname_to_region, region_to_color)

    _plot_bubble_matrix(ordered, ax_main, vmax=vmax, color_mode="signed_color")
    _finalize_main_axis(ax_main, col_order, metric_rotation)

    ax_cbar.axis("off")
    _add_bubble_legends(ax_main, vmax=vmax, signed=True)
    _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main)

    ax_top.set_title(
        "Dendrogram Bubble Plot (fixed red/blue, size = |d|, clustering uses signed d)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    stem = f"metric_dendrogram_bubble_signed_color{stem_suffix}"
    _save_figure(fig, output_dir, stem)
    _save_orders_csv(output_dir, stem, row_order, col_order)


def plot_variant_4_bubbles_grayscale(
    signed_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    metric_names: List[str],
    output_dir: Path,
    row_linkage: str,
    col_linkage: str,
    fig_size: Tuple[float, float],
    metric_rotation: float,
    stem_suffix: str = "",
):
    row_order, col_order, row_Z, col_Z = _compute_orders(
        signed_matrix,
        corr_df,
        metric_names,
        row_mode="absolute",
        row_linkage=row_linkage,
        col_linkage=col_linkage,
    )

    ordered = signed_matrix.loc[row_order, col_order]
    vmax = max(float(np.abs(ordered.values).max()), 1e-6)

    nickname_to_region, region_to_color = _get_brain_region_mappings()

    fig, ax_top, ax_left, ax_labels, ax_main, ax_cbar = _setup_layout(fig_size)
    _draw_dendrograms(ax_top, ax_left, row_Z, col_Z)
    _draw_label_axis(ax_labels, row_order, nickname_to_region, region_to_color)

    _plot_bubble_matrix(ordered, ax_main, vmax=vmax, color_mode="grayscale")
    _finalize_main_axis(ax_main, col_order, metric_rotation)

    ax_cbar.axis("off")
    _add_bubble_legends(ax_main, vmax=vmax, signed=False)
    _sync_dendrogram_layout(ax_top, ax_left, ax_labels, ax_main)

    ax_top.set_title(
        "Dendrogram Bubble Plot (grayscale, size = |d|, clustering uses |d|)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    stem = f"metric_dendrogram_bubble_abs_grayscale{stem_suffix}"
    _save_figure(fig, output_dir, stem)
    _save_orders_csv(output_dir, stem, row_order, col_order)


def main() -> None:
    args = _parse_args()

    clip_effects = args.clip_effects if args.clip_effects and args.clip_effects > 0 else None
    fig_size = (args.fig_width, args.fig_height)

    results_df, corr_df, _, _, output_dir = _load_inputs(args)
    metric_names = _get_metric_names(results_df, corr_df)

    signed_matrix = _build_signed_matrix(
        results_df=results_df,
        metric_names=metric_names,
        clip_effects=clip_effects,
    )

    signed_matrix_unclipped = _build_signed_matrix(
        results_df=results_df,
        metric_names=metric_names,
        clip_effects=None,
    )

    print(f"Genotypes: {signed_matrix.shape[0]}")
    print(f"Metrics: {signed_matrix.shape[1]}")
    if clip_effects is None:
        print("Effect clipping: disabled")
    else:
        print(f"Effect clipping: +/-{clip_effects}")
    print("Rendering both effect-size versions: clipped and unclipped")

    for matrix_label, matrix_data in [
        ("clipped", signed_matrix),
        ("unclipped", signed_matrix_unclipped),
    ]:
        suffix = f"_{matrix_label}"
        print(f"  -> {matrix_label}")

        plot_variant_1_abs_grayscale(
            signed_matrix=matrix_data,
            corr_df=corr_df,
            metric_names=metric_names,
            output_dir=output_dir,
            row_linkage=args.row_linkage,
            col_linkage=args.col_linkage,
            fig_size=fig_size,
            metric_rotation=args.metric_rotation,
            stem_suffix=suffix,
        )

        plot_variant_1b_abs_grayscale_signed_clustering(
            signed_matrix=matrix_data,
            corr_df=corr_df,
            metric_names=metric_names,
            output_dir=output_dir,
            row_linkage=args.row_linkage,
            col_linkage=args.col_linkage,
            fig_size=fig_size,
            metric_rotation=args.metric_rotation,
            stem_suffix=suffix,
        )

        plot_variant_2_abs_grayscale_with_sign_outline(
            signed_matrix=matrix_data,
            corr_df=corr_df,
            metric_names=metric_names,
            output_dir=output_dir,
            row_linkage=args.row_linkage,
            col_linkage=args.col_linkage,
            fig_size=fig_size,
            metric_rotation=args.metric_rotation,
            stem_suffix=suffix,
        )

        plot_variant_3_bubbles_color(
            signed_matrix=matrix_data,
            corr_df=corr_df,
            metric_names=metric_names,
            output_dir=output_dir,
            row_linkage=args.row_linkage,
            col_linkage=args.col_linkage,
            fig_size=fig_size,
            metric_rotation=args.metric_rotation,
            stem_suffix=suffix,
        )

        plot_variant_4_bubbles_grayscale(
            signed_matrix=matrix_data,
            corr_df=corr_df,
            metric_names=metric_names,
            output_dir=output_dir,
            row_linkage=args.row_linkage,
            col_linkage=args.col_linkage,
            fig_size=fig_size,
            metric_rotation=args.metric_rotation,
            stem_suffix=suffix,
        )

    print("Done.")


if __name__ == "__main__":
    main()
