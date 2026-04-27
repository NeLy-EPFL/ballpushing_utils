#!/usr/bin/env python3
"""Extended Data Figure 2 (a-c) — Fly speed during ball immobility affordance experiments.

Panels:
    (a) Average walking speed over the entire 2-hour experiment
        - 10-minute bins
        - Permutation tests with FDR correction per bin
    (b) Average walking speed in 20-minute window around unlock time
        - 100-second bins
        - Permutation tests with FDR correction per bin
    (c) Individual fly speed changes (pre vs post unlock)
        - Pre-unlocking window: 56:40-60:00 min
        - Post-unlocking window: 63:20-66:40 min
        - Two-sample permutation test comparing change magnitudes

Usage:
    python edfigure2_abc_speeds.py [--test]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import set_illustrator_style
from ballpushing_utils.stats import permutation_test

# Matplotlib styling
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"

# Dataset paths
COORDINATES_PATH = dataset(
    "MagnetBlock/Datasets/260127_15_coordinates_magnet_block_folders_Data/coordinates/pooled_coordinates.feather"
)
SUMMARY_PATH = dataset(
    "MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"
)

# Constants
N_PERMUTATIONS = 10_000
FPS = 29
PIXELS_TO_MM_S = 0.06  # Convert pixels/frame to mm/s (30 fps, 500px = 30mm)
UNLOCK_TIME_S = 60 * 60  # Gate opening at 60 minutes
UNLOCK_TIME_MIN = 60.0

# Panel (a) parameters - Full range
PANEL_A_BIN_SIZE_MIN = 10.0

# Panel (b) parameters - Window around unlock
PANEL_B_WINDOW_MIN = 10.0  # ±10 minutes around unlock
PANEL_B_BIN_SIZE_S = 100.0

# Panel (c) parameters - Pre/post comparison
PANEL_C_PRE_START_MIN = 56.0 + 40.0 / 60.0  # 56:40
PANEL_C_PRE_END_MIN = 60.0  # 60:00
PANEL_C_POST_START_MIN = 63.0 + 20.0 / 60.0  # 63:20
PANEL_C_POST_END_MIN = 66.0 + 40.0 / 60.0  # 66:40

# Colors (matching other magnetblock plots)
COLORS = {"n": "#faa41a", "y": "#3953A4"}  # control=orange, experimental=blue
LABELS = {"n": "No access to ball", "y": "Access to immobile ball"}


def load_data(test_mode: bool = False):
    """Load coordinates and subset to valid flies from summary."""
    print(f"\nLoading coordinates from:\n  {COORDINATES_PATH}")
    coords = pd.read_feather(COORDINATES_PATH)

    if test_mode:
        # Sample subset for testing
        sample_flies = coords["fly"].unique()[:10]
        coords = coords[coords["fly"].isin(sample_flies)].copy()
        print(f"Test mode: using {len(sample_flies)} flies")

    print(f"  Shape: {coords.shape}")

    # Load summary to get valid flies
    print(f"Loading summary from:\n  {SUMMARY_PATH}")
    summary = pd.read_feather(SUMMARY_PATH)
    valid_flies = summary["fly"].unique()

    # Subset coordinates to valid flies
    coords = coords[coords["fly"].isin(valid_flies)].copy()
    print(f"  After filtering to valid flies: {coords.shape}")

    # Check required columns
    required = ["time", "y_fly_0", "Magnet", "fly"]
    missing = [c for c in required if c not in coords.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Magnet groups: {sorted(coords['Magnet'].unique())}")
    for group in sorted(coords["Magnet"].unique()):
        n = coords[coords["Magnet"] == group]["fly"].nunique()
        print(f"    {group}: {n} flies")

    return coords


def calculate_speed(df: pd.DataFrame, rolling_window: int = 150) -> pd.DataFrame:
    """Calculate per-fly speed with rolling window smoothing."""
    data = df.copy()

    # Sort by fly and time for proper per-fly diff
    data = data.sort_values(by=["fly", "time"]).reset_index(drop=True)

    # Calculate speed, speed, and smooth all in one pass per fly
    def calc_per_fly_speed(group):
        # Get fly ID from the group (name is the grouping key value)
        fly_id = group["fly"].iloc[0] if "fly" in group.columns else group.name
        group = group.sort_values("time").reset_index(drop=True)
        group["speed_raw"] = group["y_fly_0"].diff() / group["time"].diff()
        group["speed_raw"] = group["speed_raw"].abs()
        group["speed_mm_s"] = group["speed_raw"] * PIXELS_TO_MM_S
        # Smooth within each fly
        group["speed_mm_s_smooth"] = (
            group["speed_mm_s"].rolling(window=rolling_window, min_periods=1, center=True).mean()
        )
        # Ensure fly column is preserved
        if "fly" not in group.columns:
            group["fly"] = fly_id
        return group

    data = data.groupby("fly", group_keys=False).apply(calc_per_fly_speed)

    return data


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(boot_sample)
    alpha = 100 - ci
    lower = np.percentile(boot_means, alpha / 2)
    upper = np.percentile(boot_means, 100 - alpha / 2)
    return lower, upper


def bootstrap_diff_ci(
    control_vals: np.ndarray, test_vals: np.ndarray, n_bootstrap: int = 1000, ci: float = 95
) -> tuple[float, float]:
    """Bootstrap confidence interval for the difference in means between two independent groups."""
    if len(control_vals) == 0 or len(test_vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        control_sample = rng.choice(control_vals, size=len(control_vals), replace=True)
        test_sample = rng.choice(test_vals, size=len(test_vals), replace=True)
        boot_diffs[i] = np.mean(test_sample) - np.mean(control_sample)
    alpha = 100 - ci
    lower = np.percentile(boot_diffs, alpha / 2)
    upper = np.percentile(boot_diffs, 100 - alpha / 2)
    return lower, upper


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values)
    valid_mask = ~np.isnan(p_values)

    p_adj = np.full_like(p_values, np.nan)
    significant = np.zeros(len(p_values), dtype=bool)

    if not valid_mask.any():
        return p_adj, significant

    valid_p = p_values[valid_mask]
    n = len(valid_p)
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # Compute adjusted p-values: p_adj[i] = min(p[i] * n / (i+1), 1)
    # Apply cumulative minimum from right to left to ensure monotonicity
    p_adj_sorted = np.minimum(sorted_p * n / (np.arange(1, n + 1)), 1.0)
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]

    # Unsort to original order
    p_adj_valid = np.empty(n)
    p_adj_valid[sorted_idx] = p_adj_sorted
    p_adj[valid_mask] = p_adj_valid

    # Find largest i where p[i] <= (i+1)/n * alpha
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passing = sorted_p <= thresholds

    if passing.any():
        max_idx = np.where(passing)[0][-1]
        significant_sorted = np.zeros(n, dtype=bool)
        significant_sorted[: max_idx + 1] = True

        # Unsort
        significant_valid = np.zeros(n, dtype=bool)
        significant_valid[sorted_idx] = significant_sorted
        significant[valid_mask] = significant_valid

    return p_adj, significant


def compute_binned_stats_per_fly(
    df: pd.DataFrame,
    bin_size_min: float,
    metric_col: str = "speed_mm_s_smooth",
) -> pd.DataFrame:
    """Bin data and compute per-fly averages within each bin."""
    df = df.copy()
    df["time_min"] = df["time"] / 60.0

    # Create bins
    time_max = df["time_min"].max()
    bin_edges = np.arange(0, time_max + bin_size_min, bin_size_min)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df["bin_idx"] = pd.cut(df["time_min"], bins=bin_edges, labels=False, include_lowest=True)

    # Average within each fly and bin
    grouped = df.groupby(["Magnet", "fly", "bin_idx"])[metric_col].mean().reset_index()
    grouped["bin_center_min"] = grouped["bin_idx"].map(dict(enumerate(bin_centers)))

    return grouped


def continuous_line_data(
    df: pd.DataFrame,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    resample_s: float = 1.0,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Compute per-second mean ± bootstrap CI of speed across flies.

    Workflow:
      1. Optional time-window filter.
      2. Bin raw frames into ``resample_s``-second buckets.
      3. Within each bucket average per fly (removes 29-fps over-sampling).
      4. Bootstrap across flies to get 95 % CI.

    Returns a dict keyed by Magnet value (``"n"`` / ``"y"``) containing
    ``times_min``, ``means``, ``ci_lo``, ``ci_hi`` arrays.
    """
    df = df.copy()
    if time_start_s is not None:
        df = df[df["time"] >= time_start_s]
    if time_end_s is not None:
        df = df[df["time"] <= time_end_s]

    # Bin to 1-second resolution
    df["time_bin_s"] = np.floor(df["time"] / resample_s) * resample_s

    # Per-fly mean per second
    per_fly = df.groupby(["Magnet", "fly", "time_bin_s"])["speed_mm_s_smooth"].mean().reset_index()

    result = {}
    for group in ["n", "y"]:
        gdata = per_fly[per_fly["Magnet"] == group]
        times_s = np.sort(gdata["time_bin_s"].unique())
        if len(times_s) == 0:
            result[group] = dict(
                times_min=np.array([]),
                means=np.array([]),
                ci_lo=np.array([]),
                ci_hi=np.array([]),
            )
            continue

        # Pivot: rows = time_bin, cols = fly
        pivot = gdata.pivot(index="time_bin_s", columns="fly", values="speed_mm_s_smooth").reindex(times_s)
        fly_vals = pivot.values  # shape: (n_times, n_flies), NaN where fly absent
        means = np.nanmean(fly_vals, axis=1)

        n_times = len(times_s)
        ci_lo = np.empty(n_times)
        ci_hi = np.empty(n_times)
        rng = np.random.default_rng(seed)
        for i in range(n_times):
            row = fly_vals[i]
            valid = row[~np.isnan(row)]
            if len(valid) < 2:
                ci_lo[i] = ci_hi[i] = means[i]
            else:
                boot = rng.choice(valid, size=(n_bootstrap, len(valid)), replace=True).mean(axis=1)
                ci_lo[i] = float(np.percentile(boot, 2.5))
                ci_hi[i] = float(np.percentile(boot, 97.5))

        result[group] = dict(
            times_min=times_s / 60.0,
            means=means,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
    return result


def compute_binned_stats_per_fly_seconds(
    df: pd.DataFrame,
    bin_size_s: float,
    time_start_s: float,
    time_end_s: float,
    metric_col: str = "speed_mm_s_smooth",
) -> pd.DataFrame:
    """Bin data in seconds and compute per-fly averages within each bin."""
    df = df.copy()

    # Filter to time window
    df = df[(df["time"] >= time_start_s) & (df["time"] <= time_end_s)].copy()

    # Create bins relative to window start
    bin_edges = np.arange(time_start_s, time_end_s + bin_size_s, bin_size_s)
    bin_centers_s = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_min = (bin_centers_s - UNLOCK_TIME_S) / 60.0  # Relative to unlock

    df["bin_idx"] = pd.cut(df["time"], bins=bin_edges, labels=False, include_lowest=True)

    # Average within each fly and bin
    grouped = df.groupby(["Magnet", "fly", "bin_idx"])[metric_col].mean().reset_index()
    grouped["bin_center_min"] = grouped["bin_idx"].map(dict(enumerate(bin_centers_min)))

    return grouped


def permutation_test_per_bin(
    binned_df: pd.DataFrame,
    metric_col: str,
    control_group: str = "n",
    test_group: str = "y",
    n_permutations: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run permutation test for each bin with FDR correction."""
    bins = sorted(binned_df["bin_idx"].unique())
    results = []

    print(f"\n  Running permutation tests per bin ({n_permutations} permutations)...")
    for bin_idx in tqdm(bins, desc="  Bins"):
        bin_data = binned_df[binned_df["bin_idx"] == bin_idx]

        control_vals = bin_data[bin_data["Magnet"] == control_group][metric_col].values
        test_vals = bin_data[bin_data["Magnet"] == test_group][metric_col].values

        if len(control_vals) < 2 or len(test_vals) < 2:
            results.append(
                {
                    "bin_idx": bin_idx,
                    "bin_center_min": bin_data["bin_center_min"].iloc[0],
                    "n_control": len(control_vals),
                    "n_test": len(test_vals),
                    "mean_control": np.mean(control_vals) if len(control_vals) > 0 else np.nan,
                    "mean_test": np.mean(test_vals) if len(test_vals) > 0 else np.nan,
                    "p_value_raw": np.nan,
                    "p_value_adj": np.nan,
                    "significant": False,
                }
            )
            continue

        # Use ballpushing_utils.stats.permutation_test
        perm_result = permutation_test(
            control_vals,
            test_vals,
            statistic="mean",
            n_permutations=n_permutations,
            seed=42,
        )

        ci_lower, ci_upper = bootstrap_diff_ci(control_vals, test_vals)

        results.append(
            {
                "bin_idx": bin_idx,
                "bin_center_min": bin_data["bin_center_min"].iloc[0],
                "n_control": len(control_vals),
                "n_test": len(test_vals),
                "mean_control": np.mean(control_vals),
                "mean_test": np.mean(test_vals),
                "mean_diff": perm_result.observed_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value_raw": perm_result.p_value,
                "p_value_adj": np.nan,  # Will be filled after FDR correction
                "significant": False,
            }
        )

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    p_adj, significant = benjamini_hochberg_correction(results_df["p_value_raw"].values, alpha=alpha)
    results_df["p_value_adj"] = p_adj
    results_df["significant"] = significant

    n_sig = significant.sum()
    print(f"    Significant bins after FDR correction: {n_sig}/{len(bins)} (α={alpha})")

    return results_df


def plot_panel_a(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    test_mode: bool = False,
) -> None:
    """Plot panel (a): Full 2-hour speed comparison with 10-minute bins."""
    print("\n=== Panel (a): Full 2-hour speed ===")

    # Compute binned stats
    binned = compute_binned_stats_per_fly(df, PANEL_A_BIN_SIZE_MIN)

    # Run permutation tests
    perm_results = permutation_test_per_bin(
        binned,
        metric_col="speed_mm_s_smooth",
        n_permutations=N_PERMUTATIONS,
    )

    # Save stats
    stats_csv = output_dir / "panel_a_full_speed_stats.csv"
    perm_results.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")

    # --- Plot: continuous per-second line (mirrors original create_speed_plot) ---
    print("  Computing per-second mean ± CI for continuous line...")
    line_data = continuous_line_data(df, resample_s=1.0, n_bootstrap=200, seed=42)

    # Figure size matches original (245 mm × 120 mm)
    set_illustrator_style()
    fig, ax = plt.subplots(figsize=(245 / 25.4, 120 / 25.4))

    n_flies = {g: df[df["Magnet"] == g]["fly"].nunique() for g in ["n", "y"]}
    for group in ["n", "y"]:
        ld = line_data[group]
        label = f"{LABELS[group]} (n = {n_flies[group]})"
        ax.plot(ld["times_min"], ld["means"], color=COLORS[group], lw=1.5, label=label)
        ax.fill_between(
            ld["times_min"],
            ld["ci_lo"],
            ld["ci_hi"],
            color=COLORS[group],
            alpha=0.3,
            rasterized=True,
            zorder=0,
        )

    # Bin-boundary dotted lines (10-min bins) and significance stars
    time_min_s = df["time"].min()
    time_max_s = df["time"].max()
    n_bins = round((time_max_s - time_min_s) / (PANEL_A_BIN_SIZE_MIN * 60))
    bin_edges_s = np.linspace(time_min_s, time_max_s, n_bins + 1)
    bin_edges_min = bin_edges_s / 60.0
    bin_centers_min = (bin_edges_min[:-1] + bin_edges_min[1:]) / 2
    for edge in bin_edges_min:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, lw=0.8, zorder=1)

    y_min_plot, y_max_plot = ax.get_ylim()
    y_rng = y_max_plot - y_min_plot
    for i, row in perm_results[perm_results["significant"]].iterrows():
        # Map result's bin_center_min to the nearest bin_centers_min index
        midpoint = bin_centers_min[np.argmin(np.abs(bin_centers_min - row["bin_center_min"]))]
        p = row["p_value_adj"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.text(
            midpoint,
            y_max_plot + 0.04 * y_rng,
            star,
            ha="center",
            va="bottom",
            fontsize=11,
            color="red",
            fontweight="bold",
        )

    # Red dashed vertical line at unlock
    ax.axvline(UNLOCK_TIME_MIN, color="red", linestyle="dashed", lw=2, alpha=0.9, zorder=5)

    ax.set_xlim(bin_edges_min[0], bin_edges_min[-1])
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel("Average speed (mm/s)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(False)

    plt.tight_layout()

    # Save
    fig_pdf = output_dir / "panel_a_full_speed.pdf"
    fig_png = output_dir / "panel_a_full_speed.png"
    fig.savefig(fig_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(fig_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {fig_pdf}")
    print(f"  Saved: {fig_png}")


def plot_panel_b(
    df: pd.DataFrame,
    output_dir: Path,
    test_mode: bool = False,
) -> None:
    """Plot panel (b): 20-minute window around unlock with 100s bins."""
    print("\n=== Panel (b): Window around unlock ===")

    # Compute binned stats in seconds
    window_start_s = UNLOCK_TIME_S - PANEL_B_WINDOW_MIN * 60
    window_end_s = UNLOCK_TIME_S + PANEL_B_WINDOW_MIN * 60

    binned = compute_binned_stats_per_fly_seconds(
        df,
        PANEL_B_BIN_SIZE_S,
        window_start_s,
        window_end_s,
    )

    # Run permutation tests
    perm_results = permutation_test_per_bin(
        binned,
        metric_col="speed_mm_s_smooth",
        n_permutations=N_PERMUTATIONS,
    )

    # Save stats
    stats_csv = output_dir / "panel_b_window_speed_stats.csv"
    perm_results.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")

    # --- Plot: continuous per-second line within the ±10-min window ---
    print("  Computing per-second mean ± CI for continuous line (window)...")
    line_data = continuous_line_data(
        df,
        time_start_s=window_start_s,
        time_end_s=window_end_s,
        resample_s=1.0,
        n_bootstrap=200,
        seed=42,
    )

    set_illustrator_style()
    fig, ax = plt.subplots(figsize=(245 / 25.4, 120 / 25.4))

    n_flies = {g: df[df["Magnet"] == g]["fly"].nunique() for g in ["n", "y"]}
    for group in ["n", "y"]:
        ld = line_data[group]
        # Convert absolute time → relative to unlock (minutes)
        times_rel_min = ld["times_min"] - UNLOCK_TIME_MIN
        label = f"{LABELS[group]} (n = {n_flies[group]})"
        ax.plot(times_rel_min, ld["means"], color=COLORS[group], lw=1.5, label=label)
        ax.fill_between(
            times_rel_min,
            ld["ci_lo"],
            ld["ci_hi"],
            color=COLORS[group],
            alpha=0.3,
            rasterized=True,
            zorder=0,
        )

    # Bin-boundary dotted lines (100-second bins)
    bin_edges_s = np.arange(window_start_s, window_end_s + PANEL_B_BIN_SIZE_S, PANEL_B_BIN_SIZE_S)
    bin_edges_rel_min = (bin_edges_s - UNLOCK_TIME_S) / 60.0
    bin_centers_rel_min = (bin_edges_rel_min[:-1] + bin_edges_rel_min[1:]) / 2
    for edge in bin_edges_rel_min:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, lw=0.8, zorder=1)

    y_min_plot, y_max_plot = ax.get_ylim()
    y_rng = y_max_plot - y_min_plot
    for i, row in perm_results[perm_results["significant"]].iterrows():
        midpoint = bin_centers_rel_min[np.argmin(np.abs(bin_centers_rel_min - row["bin_center_min"]))]
        p = row["p_value_adj"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.text(
            midpoint,
            y_max_plot + 0.04 * y_rng,
            star,
            ha="center",
            va="bottom",
            fontsize=11,
            color="red",
            fontweight="bold",
        )

    # Red dashed vertical line at unlock (x = 0)
    ax.axvline(0, color="red", linestyle="dashed", lw=2, alpha=0.9, zorder=5)

    ax.set_xlim(bin_edges_rel_min[0], bin_edges_rel_min[-1])
    ax.set_xlabel("Time relative to unlock (min)", fontsize=11)
    ax.set_ylabel("Average speed (mm/s)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(False)

    plt.tight_layout()

    # Save
    fig_pdf = output_dir / "panel_b_window_speed.pdf"
    fig_png = output_dir / "panel_b_window_speed.png"
    fig.savefig(fig_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(fig_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {fig_pdf}")
    print(f"  Saved: {fig_png}")


def plot_panel_c(
    df: pd.DataFrame,
    output_dir: Path,
    test_mode: bool = False,
) -> None:
    """Plot panel (c): Individual fly pre/post comparison."""
    print("\n=== Panel (c): Pre/post comparison ===")

    # Compute pre and post medians per fly
    df = df.copy()
    df["time_min"] = df["time"] / 60.0

    pre_data = df[(df["time_min"] >= PANEL_C_PRE_START_MIN) & (df["time_min"] <= PANEL_C_PRE_END_MIN)]
    post_data = df[(df["time_min"] >= PANEL_C_POST_START_MIN) & (df["time_min"] <= PANEL_C_POST_END_MIN)]

    pre_medians = pre_data.groupby(["Magnet", "fly"])["speed_mm_s_smooth"].median().reset_index()
    pre_medians.columns = ["Magnet", "fly", "pre_median"]

    post_medians = post_data.groupby(["Magnet", "fly"])["speed_mm_s_smooth"].median().reset_index()
    post_medians.columns = ["Magnet", "fly", "post_median"]

    # Merge
    change_df = pre_medians.merge(post_medians, on=["Magnet", "fly"])
    change_df["change"] = change_df["post_median"] - change_df["pre_median"]

    # Permutation test comparing change magnitudes
    control_changes = change_df[change_df["Magnet"] == "n"]["change"].values
    test_changes = change_df[change_df["Magnet"] == "y"]["change"].values

    perm_result = permutation_test(
        control_changes,
        test_changes,
        statistic="median",
        n_permutations=N_PERMUTATIONS,
        seed=42,
    )

    print(f"  Control (n={len(control_changes)}): median change = {np.median(control_changes):.3f} mm/s")
    print(f"  Test (n={len(test_changes)}): median change = {np.median(test_changes):.3f} mm/s")
    print(f"  Permutation test: p = {perm_result.p_value:.4f}")

    # Save stats
    stats_df = pd.DataFrame(
        [
            {
                "metric": "speed_change_mm_s",
                "control_group": "n",
                "test_group": "y",
                "control_n": len(control_changes),
                "test_n": len(test_changes),
                "control_median": float(np.median(control_changes)),
                "test_median": float(np.median(test_changes)),
                "median_diff": perm_result.observed_diff,
                "p_value": perm_result.p_value,
                "n_permutations": N_PERMUTATIONS,
            }
        ]
    )
    stats_csv = output_dir / "panel_c_prepost_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")

    # Save per-fly data
    change_csv = output_dir / "panel_c_prepost_per_fly.csv"
    change_df.to_csv(change_csv, index=False)
    print(f"  Saved: {change_csv}")

    # --- Plot: two subplots, one per condition (mirrors plot_pre_post_by_group) ---
    # Build MM:SS labels matching original
    def _mmss(minutes: float) -> str:
        m = int(minutes)
        s = int(round((minutes - m) * 60))
        return f"{m}:{s:02d}"

    pre_label = f"{_mmss(PANEL_C_PRE_START_MIN)}-{_mmss(PANEL_C_PRE_END_MIN)} min"
    post_label = f"{_mmss(PANEL_C_POST_START_MIN)}-{_mmss(PANEL_C_POST_END_MIN)} min"

    set_illustrator_style()
    fig, axes = plt.subplots(1, 2, figsize=(245 / 25.4, 120 / 25.4), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    for ax, group in zip(axes, ["n", "y"]):
        color = COLORS[group]
        gdf = change_df[change_df["Magnet"] == group]
        n = len(gdf)

        # Per-fly lines with markers
        for _, row in gdf.iterrows():
            ax.plot(
                [0, 1],
                [row["pre_median"], row["post_median"]],
                marker="o",
                color=color,
                alpha=0.7,
                linewidth=1,
                markersize=4,
            )

        # Group median overlay (black, no marker)
        med_pre = gdf["pre_median"].median()
        med_post = gdf["post_median"].median()
        ax.plot([0, 1], [med_pre, med_post], color="black", linewidth=2.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([pre_label, post_label], fontsize=9)
        ax.set_xlim(-0.4, 1.4)
        title_str = ("No access to ball" if group == "n" else "Access to immobile ball")
        ax.set_title(f"{title_str}\n(n={n})", fontsize=10, color=color, fontweight="bold")
        ax.set_ylabel("Median average speed (mm/s)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.grid(False)

    # Between-group annotation
    p = perm_result.p_value
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"
    fig.suptitle(
        f"Between-group change: {sig} (p = {p:.3f})",
        y=1.02, fontsize=9, color="red",
    )

    plt.tight_layout()

    # Save
    fig_pdf = output_dir / "panel_c_prepost_comparison.pdf"
    fig_png = output_dir / "panel_c_prepost_comparison.png"
    fig.savefig(fig_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(fig_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {fig_pdf}")
    print(f"  Saved: {fig_png}")


def main(test_mode: bool = False) -> None:
    """Generate all panels for Extended Data Figure 2 (a-c)."""
    print("\n" + "=" * 60)
    print("EXTENDED DATA FIGURE 2 (a-c): FLY SPEED ANALYSIS")
    print("=" * 60)

    # Set up output directory
    output_dir = figure_output_dir("EDFigure2", __file__, create=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    df = load_data(test_mode=test_mode)

    # Calculate speeds
    print("\nCalculating speeds...")
    df = calculate_speed(df, rolling_window=150)
    print(f"  Speed data shape: {df.shape}")

    # Generate each panel
    plot_panel_a(df, None, output_dir, test_mode=test_mode)
    plot_panel_b(df, output_dir, test_mode=test_mode)
    plot_panel_c(df, output_dir, test_mode=test_mode)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Data Figure 2 (a-c) — Fly speed analysis")
    parser.add_argument("--test", action="store_true", help="Run in test mode with subset of data")
    args = parser.parse_args()

    main(test_mode=args.test)
