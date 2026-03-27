#!/usr/bin/env python3
"""
Plot F1 ball coordinates over time with bin-wise permutation tests and FDR correction.

- Separate layout for training and test phases
- Comparison: control genotype vs test genotype
- Statistics: permutation test per time bin + Benjamini-Hochberg FDR per phase
- Test phase overlay: naive vs pretrained lines per genotype
- Additional outputs: test-only naive vs pretrained plots per genotype

Default test dataset is the small file with TNTxEmptyGal4 and TNTxMB247.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PIXELS_PER_MM = 500 / 30

DEFAULT_DATASET = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260304_15_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/"
    "251007_F1_TNT_Videos_Checked_F1_coordinates.feather"
)

DEFAULT_COORDINATES_DIR = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260304_15_F1_coordinates_F1_TNT_Full_Data/F1_coordinates"
)

DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Trajectory_plots")

DEFAULT_CONTROL = "TNTxEmptySplit"
DEFAULT_TEST = "TNTxMB247"

PREFERRED_SELECTED_GENOTYPES = [
    "TNTxEmptySplit",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

GENOTYPE_COLORS = {
    "TNTxEmptyGal4": "#7f7f7f",
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxPR": "#7f7f7f",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "PRxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "PRxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
    "PRxLC16-1": "#ff7f0e",
}

PHASE_TO_DISTANCE_COL = {
    "training": "training_ball_euclidean_distance",
    "test": "test_ball_euclidean_distance",
}

PRETRAINING_GROUP_COL = "pretraining_group"
PRETRAINED_VALUES = {"y", "yes", "true", "1", "pretrained", "trained"}
NAIVE_VALUES = {"n", "no", "false", "0", "naive", "untrained"}
PRETRAINING_STYLES = {
    "pretrained": {"linestyle": "-", "line_alpha": 1.0, "fill_alpha": 0.2, "label": "pretrained"},
    "naive": {"linestyle": "--", "line_alpha": 0.55, "fill_alpha": 0.08, "label": "naive"},
}


def get_genotype_column_name(df: pd.DataFrame) -> str:
    genotype_col = "Genotype" if "Genotype" in df.columns else ("genotype" if "genotype" in df.columns else None)
    if genotype_col is None:
        raise ValueError("Could not find genotype column (Genotype/genotype)")
    return genotype_col


def get_genotype_color(genotype: str) -> str:
    return GENOTYPE_COLORS.get(genotype, "#444444")


def get_pretraining_column_name(df: pd.DataFrame) -> str | None:
    if "Pretraining" in df.columns:
        return "Pretraining"
    if "pretraining" in df.columns:
        return "pretraining"
    return None


def normalize_pretraining_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in PRETRAINED_VALUES:
        return "pretrained"
    if text in NAIVE_VALUES:
        return "naive"
    return text


def load_f1_coordinates_incrementally(
    coordinates_dir: Path,
    downsample_seconds: int,
    pretraining_value: str,
    control_genotype: str,
    test_genotype: str | None,
) -> pd.DataFrame:
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    files = sorted(
        f for f in coordinates_dir.glob("*F1_coordinates.feather") if not f.name.lower().startswith("pooled_")
    )
    if not files:
        raise FileNotFoundError(f"No feather files found in {coordinates_dir}")

    print(f"Loading coordinates incrementally from: {coordinates_dir}")
    print(f"Files found: {len(files)}")

    all_downsampled = []
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}")
        try:
            df = pd.read_feather(file_path)
        except Exception as e:
            print(f"  Skipping file (read error): {e}")
            continue

        try:
            genotype_col = get_genotype_column_name(df)
        except ValueError as e:
            print(f"  Skipping file: {e}")
            continue

        pretraining_col = get_pretraining_column_name(df)

        required = [
            "fly",
            "phase",
            "phase_time",
            "training_ball_euclidean_distance",
            "test_ball_euclidean_distance",
            genotype_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  Skipping file (missing columns): {missing}")
            continue

        optional_cols = [pretraining_col] if pretraining_col else []
        keep_cols = required + [c for c in optional_cols if c in df.columns]
        chunk = df[keep_cols].copy()

        # Normalize optional pretraining column name for downstream operations
        if pretraining_col and pretraining_col in chunk.columns and pretraining_col != "Pretraining":
            chunk["Pretraining"] = chunk[pretraining_col]
            pretraining_col = "Pretraining"

        # Optional pretraining filter
        if pretraining_value.lower() != "all" and pretraining_col and pretraining_col in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk[pretraining_col].astype(str).str.lower() == pretraining_value.lower()].copy()
            print(f"  Pretraining filter: {before} -> {len(chunk)} rows")

        # Optional genotype prefilter if a single pair is requested
        if test_genotype:
            before = len(chunk)
            chunk = chunk[chunk[genotype_col].astype(str).isin([control_genotype, test_genotype])].copy()
            print(f"  Genotype pair filter: {before} -> {len(chunk)} rows")

        if chunk.empty:
            print("  No rows left after filtering")
            continue

        required_notna_cols = ["fly", "phase", "phase_time", genotype_col]
        chunk = chunk.loc[chunk[required_notna_cols].notna().all(axis=1)].copy()
        if chunk.empty:
            print("  No rows left after NaN filtering")
            continue

        # Make fly IDs unique across files to prevent collisions
        chunk["fly"] = file_path.stem + "::" + chunk["fly"].astype(str)

        # Downsample to one datapoint every N seconds per fly and phase
        chunk["phase"] = chunk["phase"].astype(str).str.lower()
        chunk = chunk[chunk["phase"].isin(["training", "test"])].copy()
        if chunk.empty:
            print("  No training/test rows")
            continue

        chunk["phase_time_sec"] = np.floor(chunk["phase_time"].astype(float) / float(downsample_seconds)).astype(int)

        agg_cols = {
            "phase_time": "mean",
            "training_ball_euclidean_distance": "mean",
            "test_ball_euclidean_distance": "mean",
            genotype_col: "first",
        }
        if pretraining_col and pretraining_col in chunk.columns:
            agg_cols[pretraining_col] = "first"

        downsampled = (
            chunk.groupby(["fly", "phase", "phase_time_sec"], as_index=False)
            .agg(agg_cols)
            .rename(columns={"phase_time": "phase_time"})
        )

        print(f"  Downsampled: {len(chunk)} -> {len(downsampled)} rows")
        all_downsampled.append(downsampled)

    if not all_downsampled:
        raise ValueError("No usable rows loaded from coordinates directory")

    combined = pd.concat(all_downsampled, ignore_index=True)
    print(f"Combined downsampled shape: {combined.shape}")
    return combined


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan

    observed = np.mean(y) - np.mean(x)
    pooled = np.concatenate([x, y])
    n_x = len(x)

    null_dist = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(pooled)
        null_dist[i] = np.mean(perm[n_x:]) - np.mean(perm[:n_x])

    p = (np.sum(np.abs(null_dist) >= np.abs(observed)) + 1) / (n_permutations + 1)
    return float(p)


def preprocess_phase_data(
    df: pd.DataFrame,
    phase: str,
    genotype_col: str,
    fly_col: str,
    n_bins: int,
    training_cutoff_seconds: float,
    normalize_time: bool = False,
    min_flies_per_genotype: int = 0,
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if phase not in PHASE_TO_DISTANCE_COL:
        raise ValueError(f"Unknown phase: {phase}")

    distance_col = PHASE_TO_DISTANCE_COL[phase]
    if group_cols is None:
        group_cols = [genotype_col]
    if not group_cols:
        raise ValueError("group_cols must contain at least one grouping column")

    required = ["phase", "phase_time", fly_col, distance_col] + group_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for phase '{phase}': {missing}")

    phase_df = df[df["phase"].astype(str).str.lower() == phase].copy()

    if phase == "training":
        phase_df = phase_df[phase_df["phase_time"] <= training_cutoff_seconds].copy()

    required_notna_cols = ["phase_time", distance_col, fly_col] + group_cols
    phase_df = phase_df.loc[phase_df[required_notna_cols].notna().all(axis=1)]
    if phase_df.empty:
        return pd.DataFrame(), np.array([])

    phase_df["distance_mm"] = phase_df[distance_col].astype(float) / PIXELS_PER_MM

    if normalize_time:
        fly_max = phase_df.groupby(fly_col)["phase_time"].transform("max")
        fly_max = fly_max.replace(0, np.nan)
        phase_df["time_axis"] = (phase_df["phase_time"].astype(float) / fly_max.astype(float)) * 100.0
        x_min, x_max = 0.0, 100.0
    else:
        phase_df["time_axis"] = phase_df["phase_time"].astype(float) / 60.0
        x_min = float(phase_df["time_axis"].min())
        x_max = float(phase_df["time_axis"].max())

    time_min = x_min
    time_max = x_max
    if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max <= time_min:
        return pd.DataFrame(), np.array([])

    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    phase_df["time_bin"] = pd.cut(phase_df["time_axis"], bins=bin_edges, labels=False, include_lowest=True)
    phase_df = phase_df.dropna(subset=["time_bin"])
    if phase_df.empty:
        return pd.DataFrame(), np.array([])

    phase_df["time_bin"] = phase_df["time_bin"].astype(int)

    # Use per-fly mean within each bin for stats and plotting
    binned = (
        phase_df.groupby(group_cols + [fly_col, "time_bin"], as_index=False)["distance_mm"]
        .mean()
        .rename(columns={"distance_mm": "avg_distance_mm"})
    )

    if min_flies_per_genotype > 0:
        counts = binned.groupby(group_cols + ["time_bin"])[fly_col].nunique().reset_index(name="n_flies")
        per_bin_min = counts.groupby("time_bin")["n_flies"].min().reset_index(name="min_n_flies")
        valid_bins = per_bin_min[per_bin_min["min_n_flies"] >= min_flies_per_genotype]["time_bin"]
        binned = binned[binned["time_bin"].isin(valid_bins)].copy()
        if binned.empty:
            return pd.DataFrame(), np.array([])

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    centers_map = dict(enumerate(centers))
    binned["bin_center"] = binned["time_bin"].map(centers_map)

    observed_bins = sorted(binned["time_bin"].unique().tolist())
    centers_out = np.array([centers_map[b] for b in observed_bins], dtype=float)

    return binned, centers_out


def compute_binwise_stats(
    binned: pd.DataFrame,
    genotype_col: str,
    fly_col: str,
    control_genotype: str,
    test_genotype: str,
    n_permutations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    rows = []
    for bin_idx in sorted(binned["time_bin"].unique().tolist()):
        sub = binned[binned["time_bin"] == bin_idx]
        x = sub[sub[genotype_col] == control_genotype]["avg_distance_mm"].dropna().values
        y = sub[sub[genotype_col] == test_genotype]["avg_distance_mm"].dropna().values

        if len(x) == 0 and len(y) == 0:
            continue

        center = float(sub["bin_center"].iloc[0])
        n_control_flies = int(sub[sub[genotype_col] == control_genotype][fly_col].nunique())
        n_test_flies = int(sub[sub[genotype_col] == test_genotype][fly_col].nunique())

        mean_x = float(np.nanmean(x)) if len(x) else np.nan
        mean_y = float(np.nanmean(y)) if len(y) else np.nan
        diff = mean_y - mean_x if np.isfinite(mean_x) and np.isfinite(mean_y) else np.nan
        p = permutation_pvalue(x, y, n_permutations=n_permutations, rng=rng)

        rows.append(
            {
                "time_bin": int(bin_idx),
                "bin_center": float(center),
                "n_control": int(len(x)),
                "n_test": int(len(y)),
                "n_control_flies": n_control_flies,
                "n_test_flies": n_test_flies,
                "mean_control": mean_x,
                "mean_test": mean_y,
                "mean_diff_test_minus_control": diff,
                "p_value": p,
            }
        )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    valid = stats_df["p_value"].notna()
    stats_df["p_fdr"] = np.nan
    if valid.any():
        _, p_corr, _, _ = multipletests(stats_df.loc[valid, "p_value"].values, method="fdr_bh")
        stats_df.loc[valid, "p_fdr"] = p_corr

    stats_df["sig"] = stats_df["p_fdr"].apply(lambda p: significance_stars(p) if pd.notna(p) else "")
    return stats_df


def compute_binwise_pretraining_stats(
    binned: pd.DataFrame,
    pretraining_group_col: str,
    fly_col: str,
    n_permutations: int,
    naive_label: str = "naive",
    pretrained_label: str = "pretrained",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    rows = []
    for bin_idx in sorted(binned["time_bin"].unique().tolist()):
        sub = binned[binned["time_bin"] == bin_idx]
        x = sub[sub[pretraining_group_col] == naive_label]["avg_distance_mm"].dropna().values
        y = sub[sub[pretraining_group_col] == pretrained_label]["avg_distance_mm"].dropna().values

        if len(x) == 0 and len(y) == 0:
            continue

        center = float(sub["bin_center"].iloc[0])
        n_naive_flies = int(sub[sub[pretraining_group_col] == naive_label][fly_col].nunique())
        n_pretrained_flies = int(sub[sub[pretraining_group_col] == pretrained_label][fly_col].nunique())

        mean_x = float(np.nanmean(x)) if len(x) else np.nan
        mean_y = float(np.nanmean(y)) if len(y) else np.nan
        diff = mean_y - mean_x if np.isfinite(mean_x) and np.isfinite(mean_y) else np.nan
        p = permutation_pvalue(x, y, n_permutations=n_permutations, rng=rng)

        rows.append(
            {
                "time_bin": int(bin_idx),
                "bin_center": float(center),
                "n_naive": int(len(x)),
                "n_pretrained": int(len(y)),
                "n_naive_flies": n_naive_flies,
                "n_pretrained_flies": n_pretrained_flies,
                "mean_naive": mean_x,
                "mean_pretrained": mean_y,
                "mean_diff_pretrained_minus_naive": diff,
                "p_value": p,
            }
        )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    valid = stats_df["p_value"].notna()
    stats_df["p_fdr"] = np.nan
    if valid.any():
        _, p_corr, _, _ = multipletests(stats_df.loc[valid, "p_value"].values, method="fdr_bh")
        stats_df.loc[valid, "p_fdr"] = p_corr

    stats_df["sig"] = stats_df["p_fdr"].apply(lambda p: significance_stars(p) if pd.notna(p) else "")
    return stats_df


def annotate_significance_stars(ax, stats_df: pd.DataFrame):
    if stats_df is None or stats_df.empty:
        return

    y_min, y_max = ax.get_ylim()
    y_range = max(1e-6, y_max - y_min)
    y_star = y_max + 0.06 * y_range

    for _, row in stats_df.iterrows():
        star = row.get("sig", "")
        if not star:
            continue
        ax.text(row["bin_center"], y_star, star, ha="center", va="bottom", fontsize=11, color="red", fontweight="bold")

    ax.set_ylim(y_min, y_max + 0.18 * y_range)


def collect_unique_legend_handles_labels(axes) -> tuple[list, list]:
    seen = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen[label] = handle
    return list(seen.values()), list(seen.keys())


def add_line_with_ci_filtered(
    ax,
    binned: pd.DataFrame,
    filters: dict[str, str],
    label: str,
    color: str,
    linestyle: str = "-",
    line_alpha: float = 1.0,
    fill_alpha: float = 0.2,
):
    sub = binned
    for col, value in filters.items():
        sub = sub[sub[col] == value]

    if sub.empty:
        return False

    summary = sub.groupby("time_bin", as_index=False).agg(
        bin_center=("bin_center", "first"), mean=("avg_distance_mm", "mean"), sem=("avg_distance_mm", sem)
    )
    summary = cast(pd.DataFrame, summary).sort_values(by="time_bin")

    x = summary["bin_center"].values
    y = summary["mean"].values
    y_sem = summary["sem"].fillna(0).values

    ax.plot(x, y, color=color, linewidth=2.5, label=label, linestyle=linestyle, alpha=line_alpha)
    ax.fill_between(x, y - y_sem, y + y_sem, color=color, alpha=fill_alpha)
    return True


def add_line_with_ci(ax, binned: pd.DataFrame, genotype: str, genotype_col: str, label: str, color: str):
    add_line_with_ci_filtered(ax, binned, {genotype_col: genotype}, label=label, color=color)


def plot_phase_panel(
    ax,
    binned: pd.DataFrame,
    stats_df: pd.DataFrame,
    genotype_col: str,
    control_genotype: str,
    test_genotype: str,
    phase: str,
    control_color: str,
    test_color: str,
    x_label: str,
    test_pretraining_binned: pd.DataFrame | None = None,
    pretraining_group_col: str = PRETRAINING_GROUP_COL,
):
    plotted_test_pretraining = False
    if (
        phase == "test"
        and test_pretraining_binned is not None
        and not test_pretraining_binned.empty
        and pretraining_group_col in test_pretraining_binned.columns
    ):
        for genotype, role, color in [
            (control_genotype, "control", control_color),
            (test_genotype, "test", test_color),
        ]:
            for pretraining_label in ["pretrained", "naive"]:
                style = PRETRAINING_STYLES[pretraining_label]
                label = f"{genotype} ({role}, {style['label']})"
                added = add_line_with_ci_filtered(
                    ax,
                    test_pretraining_binned,
                    {genotype_col: genotype, pretraining_group_col: pretraining_label},
                    label=label,
                    color=color,
                    linestyle=style["linestyle"],
                    line_alpha=style["line_alpha"],
                    fill_alpha=style["fill_alpha"],
                )
                plotted_test_pretraining = plotted_test_pretraining or added

    if not plotted_test_pretraining:
        add_line_with_ci(
            ax,
            binned,
            control_genotype,
            genotype_col,
            f"{control_genotype} (control)",
            control_color,
        )
        add_line_with_ci(
            ax,
            binned,
            test_genotype,
            genotype_col,
            f"{test_genotype} (test)",
            test_color,
        )

    ax.set_title(f"{phase.capitalize()} phase", fontsize=11)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Ball distance (mm)", fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotate_significance_stars(ax, stats_df)


def plot_test_pretraining_panel(
    ax,
    binned: pd.DataFrame,
    stats_df: pd.DataFrame,
    pretraining_group_col: str,
    genotype_label: str,
    genotype_color: str,
    x_label: str,
):
    for pretraining_label in ["pretrained", "naive"]:
        style = PRETRAINING_STYLES[pretraining_label]
        add_line_with_ci_filtered(
            ax,
            binned,
            {pretraining_group_col: pretraining_label},
            label=style["label"],
            color=genotype_color,
            linestyle=style["linestyle"],
            line_alpha=style["line_alpha"],
            fill_alpha=style["fill_alpha"],
        )

    ax.set_title(f"{genotype_label} | Test phase", fontsize=11)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Ball distance (mm)", fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotate_significance_stars(ax, stats_df)


def main():
    parser = argparse.ArgumentParser(description="F1 ball coordinates over time (training/test) with permutation+FDR")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a single F1_coordinates feather (optional; if omitted, --coordinates-dir is used)",
    )
    parser.add_argument(
        "--coordinates-dir",
        type=str,
        default=str(DEFAULT_COORDINATES_DIR),
        help="Directory with per-file F1_coordinates feathers to load incrementally",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--control-genotype", type=str, default=DEFAULT_CONTROL, help="Control genotype")
    parser.add_argument(
        "--test-genotype",
        type=str,
        default=None,
        help="Single test genotype. If omitted, all genotypes (except control) are compared to control.",
    )
    parser.add_argument("--n-bins", type=int, default=12, help="Number of time bins")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count per bin")
    parser.add_argument(
        "--training-cutoff-seconds", type=float, default=3600.0, help="Training phase max time in seconds"
    )
    parser.add_argument(
        "--pretraining-value",
        type=str,
        default="all",
        help="Pretraining filter value (default: all). Use 'all' to compare naive and pretrained.",
    )
    parser.add_argument(
        "--downsample-seconds",
        type=int,
        default=1,
        help="Downsample step in seconds when loading from --coordinates-dir (default: 1)",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument(
        "--trim-min-flies",
        type=int,
        default=10,
        help="Minimum flies per genotype required in a bin for trimmed-raw mode (default: 10)",
    )
    args = parser.parse_args()

    if args.downsample_seconds < 1:
        raise ValueError("--downsample-seconds must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        print(f"Loading single dataset: {dataset_path}")
        df = pd.read_feather(dataset_path)
    else:
        df = load_f1_coordinates_incrementally(
            coordinates_dir=Path(args.coordinates_dir),
            downsample_seconds=args.downsample_seconds,
            pretraining_value=args.pretraining_value,
            control_genotype=args.control_genotype,
            test_genotype=args.test_genotype,
        )

    genotype_col = get_genotype_column_name(df)
    if "fly" not in df.columns:
        raise ValueError("Missing 'fly' column")

    pretraining_col = get_pretraining_column_name(df)

    if args.pretraining_value.lower() != "all" and pretraining_col and pretraining_col in df.columns:
        keep = df[df[pretraining_col].astype(str).str.lower() == args.pretraining_value.lower()].copy()
    else:
        keep = df.copy()

    available_genotypes = sorted(keep[genotype_col].dropna().astype(str).unique().tolist())
    selected_available = [g for g in PREFERRED_SELECTED_GENOTYPES if g in available_genotypes]

    if not selected_available:
        raise ValueError(
            "None of the preferred selected genotypes are available after filtering. "
            f"Available: {available_genotypes}"
        )

    keep = keep[keep[genotype_col].isin(selected_available)].copy()
    available_genotypes = sorted(keep[genotype_col].dropna().astype(str).unique().tolist())

    if args.control_genotype not in available_genotypes:
        raise ValueError(
            f"Control genotype '{args.control_genotype}' not found after filtering. Available: {available_genotypes}"
        )

    if args.test_genotype:
        if args.test_genotype == "TNTxEmptyGal4":
            raise ValueError("TNTxEmptyGal4 is excluded from this analysis. Use selected F1 TNT genotypes.")
        genotype_pairs = [(args.control_genotype, args.test_genotype)]
    else:
        genotype_pairs = [(args.control_genotype, g) for g in available_genotypes if g != args.control_genotype]

    pair_genotypes = sorted({g for pair in genotype_pairs for g in pair})
    keep = keep[keep[genotype_col].isin(pair_genotypes)].copy()
    if keep.empty:
        raise ValueError("No rows left after genotype filtering")

    keep_pretraining: pd.DataFrame = pd.DataFrame()
    if pretraining_col and pretraining_col in keep.columns:
        keep[PRETRAINING_GROUP_COL] = keep[pretraining_col].apply(normalize_pretraining_value)
        print("Pretraining groups after filtering:")
        print(keep[PRETRAINING_GROUP_COL].value_counts(dropna=False).to_string())

        keep_pretraining = keep[keep[PRETRAINING_GROUP_COL].isin(["pretrained", "naive"])].copy()
        if keep_pretraining.empty:
            print("No recognized naive/pretrained rows available; pretraining comparison plots will be skipped.")
    else:
        print("No Pretraining column found; pretraining comparison plots will be skipped.")

    print(f"Rows after genotype filter: {len(keep)}")
    print(keep[genotype_col].value_counts())
    print(f"Genotype pairs: {genotype_pairs}")

    analysis_modes = [
        {
            "name": "raw",
            "normalize_time": False,
            "min_flies": 0,
            "x_label": "Time (min)",
        },
        {
            "name": "normalized_percent",
            "normalize_time": True,
            "min_flies": 0,
            "x_label": "Time (% of phase)",
        },
        {
            "name": "trimmed_raw",
            "normalize_time": False,
            "min_flies": args.trim_min_flies,
            "x_label": "Time (min)",
        },
    ]

    for mode in analysis_modes:
        mode_name = mode["name"]
        print(f"\nRunning mode: {mode_name}")
        all_phase_stats = []
        all_pretraining_stats = []

        # One figure per pair
        for control_genotype, test_genotype in genotype_pairs:
            pair_df = keep[keep[genotype_col].isin([control_genotype, test_genotype])].copy()
            pair_pretraining_df = cast(
                pd.DataFrame,
                keep_pretraining.loc[keep_pretraining[genotype_col].isin([control_genotype, test_genotype])].copy(),
            )
            if pair_df.empty:
                continue

            control_color = get_genotype_color(control_genotype)
            test_color = get_genotype_color(test_genotype)

            fig_pair, axes_pair = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ax, phase in zip(axes_pair, ["training", "test"]):
                binned, _ = preprocess_phase_data(
                    pair_df,
                    phase=phase,
                    genotype_col=genotype_col,
                    fly_col="fly",
                    n_bins=args.n_bins,
                    training_cutoff_seconds=args.training_cutoff_seconds,
                    normalize_time=mode["normalize_time"],
                    min_flies_per_genotype=mode["min_flies"],
                )

                test_pretraining_binned = None
                if phase == "test" and not pair_pretraining_df.empty:
                    test_pretraining_binned, _ = preprocess_phase_data(
                        cast(pd.DataFrame, pair_pretraining_df),
                        phase="test",
                        genotype_col=genotype_col,
                        fly_col="fly",
                        n_bins=args.n_bins,
                        training_cutoff_seconds=args.training_cutoff_seconds,
                        normalize_time=mode["normalize_time"],
                        min_flies_per_genotype=mode["min_flies"],
                        group_cols=[genotype_col, PRETRAINING_GROUP_COL],
                    )

                if binned.empty:
                    ax.set_title(f"{phase.capitalize()} phase (no data)")
                    ax.axis("off")
                    continue

                stats_df = compute_binwise_stats(
                    binned,
                    genotype_col=genotype_col,
                    fly_col="fly",
                    control_genotype=control_genotype,
                    test_genotype=test_genotype,
                    n_permutations=args.n_permutations,
                )

                stats_df["phase"] = phase
                stats_df["control_genotype"] = control_genotype
                stats_df["test_genotype"] = test_genotype
                stats_df["analysis_mode"] = mode_name
                all_phase_stats.append(stats_df)

                plot_phase_panel(
                    ax,
                    binned,
                    stats_df,
                    genotype_col=genotype_col,
                    control_genotype=control_genotype,
                    test_genotype=test_genotype,
                    phase=phase,
                    control_color=control_color,
                    test_color=test_color,
                    x_label=mode["x_label"],
                    test_pretraining_binned=test_pretraining_binned,
                )

            handles, labels = collect_unique_legend_handles_labels(axes_pair.ravel())
            if handles:
                fig_pair.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=min(4, max(2, len(labels))),
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.02),
                )

            fig_pair.suptitle(
                f"F1 ball trajectories over time ({mode_name})\n"
                f"{test_genotype} vs {control_genotype} (permutation + FDR)",
                fontsize=12,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.95))

            pair_stem = f"f1_ball_coordinates_over_time_{mode_name}_{test_genotype}_vs_{control_genotype}".replace(
                "/", "_"
            )
            for ext in ["png", "pdf", "svg"]:
                out = output_dir / f"{pair_stem}.{ext}"
                plt.savefig(out, dpi=300, bbox_inches="tight", format=ext)

            if args.show:
                plt.show()
            plt.close(fig_pair)

        # Combined layout: one row per pair, columns = training/test
        n_pairs = len(genotype_pairs)
        fig_h = max(4.2 * n_pairs, 5.0)
        fig_combined, axes_combined = plt.subplots(
            n_pairs, 2, figsize=(14, fig_h), squeeze=False, sharex=False, sharey=False
        )

        for row_idx, (control_genotype, test_genotype) in enumerate(genotype_pairs):
            pair_df = keep[keep[genotype_col].isin([control_genotype, test_genotype])].copy()
            control_color = get_genotype_color(control_genotype)
            test_color = get_genotype_color(test_genotype)
            for col_idx, phase in enumerate(["training", "test"]):
                ax = cast(Axes, axes_combined[row_idx, col_idx])

                binned, _ = preprocess_phase_data(
                    pair_df,
                    phase=phase,
                    genotype_col=genotype_col,
                    fly_col="fly",
                    n_bins=args.n_bins,
                    training_cutoff_seconds=args.training_cutoff_seconds,
                    normalize_time=mode["normalize_time"],
                    min_flies_per_genotype=mode["min_flies"],
                )

                test_pretraining_binned = None
                if phase == "test" and not keep_pretraining.empty:
                    pair_pretraining_df = cast(
                        pd.DataFrame,
                        keep_pretraining.loc[
                            keep_pretraining[genotype_col].isin([control_genotype, test_genotype])
                        ].copy(),
                    )
                    if not pair_pretraining_df.empty:
                        test_pretraining_binned, _ = preprocess_phase_data(
                            cast(pd.DataFrame, pair_pretraining_df),
                            phase="test",
                            genotype_col=genotype_col,
                            fly_col="fly",
                            n_bins=args.n_bins,
                            training_cutoff_seconds=args.training_cutoff_seconds,
                            normalize_time=mode["normalize_time"],
                            min_flies_per_genotype=mode["min_flies"],
                            group_cols=[genotype_col, PRETRAINING_GROUP_COL],
                        )

                if binned.empty:
                    ax.set_title(f"{test_genotype} vs {control_genotype} | {phase} (no data)")
                    ax.axis("off")
                    continue

                stats_df = compute_binwise_stats(
                    binned,
                    genotype_col=genotype_col,
                    fly_col="fly",
                    control_genotype=control_genotype,
                    test_genotype=test_genotype,
                    n_permutations=args.n_permutations,
                )

                plot_phase_panel(
                    ax,
                    binned,
                    stats_df,
                    genotype_col=genotype_col,
                    control_genotype=control_genotype,
                    test_genotype=test_genotype,
                    phase=phase,
                    control_color=control_color,
                    test_color=test_color,
                    x_label=mode["x_label"],
                    test_pretraining_binned=test_pretraining_binned,
                )
                ax.set_title(f"{test_genotype} vs {control_genotype} | {phase}", fontsize=10)

        handles, labels = collect_unique_legend_handles_labels(axes_combined.ravel())
        if handles:
            fig_combined.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(4, max(2, len(labels))),
                frameon=False,
                bbox_to_anchor=(0.5, 1.005),
            )

        pretrain_label = (
            "all pretraining" if args.pretraining_value.lower() == "all" else f"pretraining={args.pretraining_value}"
        )
        trim_label = f", min_flies={mode['min_flies']}" if mode["min_flies"] > 0 else ""
        fig_combined.suptitle(
            f"F1 trajectories by genotype pair ({pretrain_label}, mode={mode_name}{trim_label})\n"
            f"Rows: genotype pairs, Columns: training/test",
            fontsize=12,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))

        combined_stem = f"f1_ball_coordinates_over_time_{mode_name}_combined_pairs_training_test"
        for ext in ["png", "pdf", "svg"]:
            out = output_dir / f"{combined_stem}.{ext}"
            plt.savefig(out, dpi=300, bbox_inches="tight", format=ext)

        if args.show:
            plt.show()
        plt.close(fig_combined)

        # Pretraining effect at test phase: one plot per genotype (including control)
        if not keep_pretraining.empty:
            for genotype in pair_genotypes:
                geno_df: pd.DataFrame = keep_pretraining.loc[keep_pretraining[genotype_col] == genotype].copy()
                if geno_df.empty:
                    continue

                binned_pretraining, _ = preprocess_phase_data(
                    cast(pd.DataFrame, geno_df),
                    phase="test",
                    genotype_col=genotype_col,
                    fly_col="fly",
                    n_bins=args.n_bins,
                    training_cutoff_seconds=args.training_cutoff_seconds,
                    normalize_time=mode["normalize_time"],
                    min_flies_per_genotype=mode["min_flies"],
                    group_cols=[PRETRAINING_GROUP_COL],
                )
                if binned_pretraining.empty:
                    print(f"Skipping pretraining test plot for {genotype}: no binned test data")
                    continue

                present_groups = set(binned_pretraining[PRETRAINING_GROUP_COL].astype(str).unique().tolist())
                if not {"naive", "pretrained"}.issubset(present_groups):
                    print(
                        f"Skipping pretraining test plot for {genotype}: requires both naive and pretrained "
                        f"(found: {sorted(present_groups)})"
                    )
                    continue

                stats_pretraining = compute_binwise_pretraining_stats(
                    binned_pretraining,
                    pretraining_group_col=PRETRAINING_GROUP_COL,
                    fly_col="fly",
                    n_permutations=args.n_permutations,
                )
                stats_pretraining["analysis_mode"] = mode_name
                stats_pretraining["genotype"] = genotype
                all_pretraining_stats.append(stats_pretraining)

                fig_pt, ax_pt = plt.subplots(1, 1, figsize=(6.8, 4.8))
                plot_test_pretraining_panel(
                    ax_pt,
                    binned_pretraining,
                    stats_pretraining,
                    pretraining_group_col=PRETRAINING_GROUP_COL,
                    genotype_label=genotype,
                    genotype_color=get_genotype_color(genotype),
                    x_label=mode["x_label"],
                )

                handles_pt, labels_pt = collect_unique_legend_handles_labels([ax_pt])
                if handles_pt:
                    fig_pt.legend(
                        handles_pt,
                        labels_pt,
                        loc="upper center",
                        ncol=2,
                        frameon=False,
                        bbox_to_anchor=(0.5, 1.03),
                    )

                fig_pt.suptitle(
                    f"Test phase pretraining effect ({mode_name})\n{genotype}: pretrained vs naive",
                    fontsize=11,
                )
                plt.tight_layout(rect=(0, 0, 1, 0.93))

                geno_stem = f"f1_ball_coordinates_test_pretraining_{mode_name}_{genotype}".replace("/", "_")
                for ext in ["png", "pdf", "svg"]:
                    out = output_dir / f"{geno_stem}.{ext}"
                    plt.savefig(out, dpi=300, bbox_inches="tight", format=ext)

                if args.show:
                    plt.show()
                plt.close(fig_pt)

            if all_pretraining_stats:
                stats_pretraining_all = pd.concat(all_pretraining_stats, ignore_index=True)
                stats_pretraining_path = (
                    output_dir
                    / f"f1_ball_coordinates_test_pretraining_{mode_name}_by_genotype_binwise_permutation_fdr.csv"
                )
                stats_pretraining_all.to_csv(stats_pretraining_path, index=False)
                print(f"Saved pretraining stats: {stats_pretraining_path}")
                print(
                    stats_pretraining_all[
                        [
                            "analysis_mode",
                            "genotype",
                            "time_bin",
                            "n_naive",
                            "n_pretrained",
                            "n_naive_flies",
                            "n_pretrained_flies",
                            "p_value",
                            "p_fdr",
                            "sig",
                        ]
                    ].to_string(index=False)
                )

        if all_phase_stats:
            stats_all = pd.concat(all_phase_stats, ignore_index=True)
            stats_path = output_dir / f"f1_ball_coordinates_over_time_{mode_name}_all_pairs_binwise_permutation_fdr.csv"
            stats_all.to_csv(stats_path, index=False)
            print(f"Saved stats: {stats_path}")
            print(
                stats_all[
                    [
                        "analysis_mode",
                        "control_genotype",
                        "test_genotype",
                        "phase",
                        "time_bin",
                        "n_control",
                        "n_test",
                        "n_control_flies",
                        "n_test_flies",
                        "p_value",
                        "p_fdr",
                        "sig",
                    ]
                ].to_string(index=False)
            )

    print("Done.")


if __name__ == "__main__":
    main()
