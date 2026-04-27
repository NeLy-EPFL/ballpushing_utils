#!/usr/bin/env python3
"""
Script to analyze interaction rates and duration (persistence) over time
for MagnetBlock experiments.

Two complementary analyses are produced:

1. **Interaction rate** (events / min per configurable bin)
   — how often does the fly visit the ball?

2. **Interaction duration** (mean seconds per event per configurable bin)
   — how long does the fly stay on the ball per visit (persistence)?

Both analyses use:
- Time bins of configurable width (default 10 sec = 0.1667 min) over the full 120 min.
- A between-group permutation test (Magnet=y vs Magnet=n) per bin.
- A per-fly linear-regression trend test over the first hour
  (one-sample permutation test of the slope distribution vs. zero).

Key design choices:
- An "event" is counted once: the column carries the same integer for every frame of
  an event, so we deduplicate by (fly, event_id) and take the earliest frame as onset.
- Duration = number of frames belonging to an event / FPS (29 Hz) → seconds.
- Bins where a fly had no events contribute NaN to the duration mean
  (rather than 0), so the duration reflects only flies that actually interacted.

Usage:
    python plot_magnetblock_interaction_rate.py [--bin-size N] [--n-permutations N]
                                                [--output-dir PATH] [--focus-hours N]

Arguments:
    --bin-size         : Bin duration in minutes (default: 0.1667 = 10 sec)
    --n-permutations   : Number of permutations for statistical testing (default: 10000)
    --output-dir       : Directory to save output (default: same folder as the dataset)
    --focus-hours      : Hours of data to use for the trend analysis (default: 1)
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global matplotlib styling — consistent with other MagnetBlock scripts
# ---------------------------------------------------------------------------
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable PDFs
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Acquisition frame rate (frames per second)
FPS = 29
# Pixel → mm conversion (500 px = 30 mm corridor width)
PIXELS_PER_MM = 500 / 30
DEFAULT_BIN_SIZE_MIN = 10.0 / 60.0


def bootstrap_ci_difference(group1, group2, n_bootstrap=2000, ci=95, random_state=42):
    """Bootstrap CI for mean(group2) - mean(group1)."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs[i] = np.mean(s2) - np.mean(s1)

    alpha = 100 - ci
    return float(np.percentile(diffs, alpha / 2)), float(np.percentile(diffs, 100 - alpha / 2))


def cohens_d(group1, group2):
    """Cohen's d for (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return float((np.mean(group2) - np.mean(group1)) / pooled_std)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_p_value(p_value: float, n_permutations: int) -> str:
    min_p = 1.0 / n_permutations
    if p_value <= min_p:
        return f"p ≤ {min_p:.0e}"
    elif p_value < 0.001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.4f}"


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: str) -> pd.DataFrame:
    print(f"Loading dataset from:\n  {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"  Shape: {df.shape}")

    required = ["time", "interaction_event_onset", "Magnet", "fly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Magnet groups: {sorted(df['Magnet'].unique())}")
    print(f"  Unique flies: {df['fly'].nunique()}")
    for g in sorted(df["Magnet"].unique()):
        n = df[df["Magnet"] == g]["fly"].nunique()
        print(f"    Magnet={g}: {n} flies")
    print(f"  Time range: {df['time'].min():.0f} – {df['time'].max():.0f} s " f"({df['time'].max() / 60:.1f} min)")

    # Pre-compute fly-ball Euclidean distance (mm) on every frame
    df["fly_ball_dist_mm"] = (
        np.sqrt((df["x_fly_0"] - df["x_ball_0"]) ** 2 + (df["y_fly_0"] - df["y_ball_0"]) ** 2) / PIXELS_PER_MM
    )
    print(
        f"  fly_ball_dist_mm: mean={df['fly_ball_dist_mm'].mean():.2f} mm, "
        f"min={df['fly_ball_dist_mm'].min():.2f}, max={df['fly_ball_dist_mm'].max():.2f}"
    )
    return df


# ---------------------------------------------------------------------------
# Event extraction & binning
# ---------------------------------------------------------------------------


def extract_event_onsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per unique interaction event per fly.

    The column ``interaction_event_onset`` carries the same integer across ALL
    frames that belong to a given event (e.g. [NaN, NaN, 1, 1, 1, NaN, 2, 2, …]).
    We therefore group by (fly, event_id) and keep only the first frame (minimum
    time) so that each event contributes exactly one count.
    """
    non_nan = df[df["interaction_event_onset"].notna()].copy()
    # One row per event per fly — earliest frame of each event
    onsets = (
        non_nan.sort_values("time").groupby(["fly", "interaction_event_onset"], sort=False).first().reset_index()
    ).copy()
    onsets["time_min"] = onsets["time"] / 60.0
    print(f"\nTotal unique interaction events across all flies: {len(onsets)}")
    return onsets


def compute_rates_per_fly(
    onsets: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = DEFAULT_BIN_SIZE_MIN,
) -> pd.DataFrame:
    """
    For each fly, count interactions per time bin and normalise to events/min.

    Parameters
    ----------
    onsets      : DataFrame of onset rows only (has 'time_min', 'fly', 'Magnet')
    df_full     : Full dataset (used to determine which bins a fly was observed in)
    bin_size_min: Bin width in minutes

    Returns
    -------
    DataFrame with columns [fly, Magnet, bin_start_min, bin_center_min,
                             event_count, rate_per_min]
    """
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    rows = []
    fly_meta = df_full.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")

    for fly, fly_onsets in onsets.groupby("fly"):
        magnet = fly_meta.loc[fly, "Magnet"]
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            count = ((fly_onsets["time_min"] >= b_start) & (fly_onsets["time_min"] < b_end)).sum()
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_start,
                    bin_center_min=bin_centers[i],
                    event_count=int(count),
                    rate_per_min=count / bin_size_min,
                )
            )

    # Include flies that have NO onsets at all (rate = 0 in every bin)
    flies_with_onsets = set(onsets["fly"].unique())
    all_flies = set(df_full["fly"].unique())
    for fly in all_flies - flies_with_onsets:
        magnet = fly_meta.loc[fly, "Magnet"]
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_start,
                    bin_center_min=bin_centers[i],
                    event_count=0,
                    rate_per_min=0.0,
                )
            )

    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


def compute_durations_per_fly(
    df: pd.DataFrame,
    bin_size_min: float = 5.0,
    fps: float = FPS,
) -> pd.DataFrame:
    """
    For each interaction event, compute its duration (in seconds) from the raw
    frame-level data, then assign it to a time bin based on its onset time and
    average durations per fly per bin.

    Parameters
    ----------
    df           : Full dataset (all rows, including non-event rows)
    bin_size_min : Bin width in minutes
    fps          : Recording frame rate (default: FPS constant)

    Returns
    -------
    DataFrame with columns [fly, Magnet, bin_idx, bin_start_min,
                             bin_center_min, n_events, mean_duration_s,
                             total_duration_s]
    """
    # Work only on rows that belong to an event.
    # IMPORTANT: use `interaction_event` (not `interaction_event_onset`).
    # `interaction_event` carries the same integer across ALL frames of an event,
    # so grouping on it gives the true frame count (= duration).
    # `interaction_event_onset` only marks the single onset frame → always 1 row.
    event_rows = df[df["interaction_event"].notna()].copy()

    # Per-event stats: onset time (min) + duration (frames → seconds)
    event_stats = (
        event_rows.groupby(["fly", "interaction_event"])
        .agg(
            onset_time_s=("time", "min"),  # earliest frame = onset
            duration_frames=("time", "count"),  # number of frames = duration
            Magnet=("Magnet", "first"),
        )
        .reset_index()
    )
    event_stats["onset_time_min"] = event_stats["onset_time_s"] / 60.0
    event_stats["duration_s"] = event_stats["duration_frames"] / fps

    total_events = len(event_stats)
    print(f"\nTotal unique interaction events (for duration analysis): {total_events}")
    print(
        f"  Mean duration: {event_stats['duration_s'].mean():.2f} s  "
        f"(median: {event_stats['duration_s'].median():.2f} s)"
    )

    # Build time bins
    time_max_min = df["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")
    all_flies = set(df["fly"].unique())

    rows = []
    for fly in all_flies:
        magnet = fly_meta.loc[fly, "Magnet"]
        fly_events = event_stats[event_stats["fly"] == fly]
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            mask = (fly_events["onset_time_min"] >= b_start) & (fly_events["onset_time_min"] < b_end)
            bin_events = fly_events[mask]
            n_ev = len(bin_events)
            mean_dur = bin_events["duration_s"].mean() if n_ev > 0 else np.nan
            total_dur = bin_events["duration_s"].sum() if n_ev > 0 else 0.0
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_start,
                    bin_center_min=float(bin_centers[i]),
                    n_events=n_ev,
                    mean_duration_s=mean_dur,
                    total_duration_s=total_dur,
                )
            )

    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


def compute_proximity_per_fly(
    df: pd.DataFrame,
    bin_size_min: float = 5.0,
) -> pd.DataFrame:
    """
    Compute mean fly-ball distance (mm) per fly per time bin, restricted to
    frames that belong to an interaction event (``interaction_event`` not NaN).

    This focuses on how close the fly gets to the ball *while interacting*,
    capturing approach depth rather than general proximity.
    Bins where a fly has no events contribute NaN (not zero).
    Lower values = fly gets closer to ball during interactions = deeper engagement.

    Parameters
    ----------
    df           : Full dataset with the pre-computed ``fly_ball_dist_mm`` column
    bin_size_min : Bin width in minutes

    Returns
    -------
    DataFrame with columns [fly, Magnet, bin_idx, bin_start_min,
                             bin_center_min, mean_dist_mm, median_dist_mm]
    """
    if "fly_ball_dist_mm" not in df.columns:
        raise ValueError("Column 'fly_ball_dist_mm' not found. Run load_dataset() first.")

    # Keep only frames inside an interaction event
    event_df = df[df["interaction_event"].notna()].copy()
    event_df["time_min"] = event_df["time"] / 60.0

    time_max_min = df["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")

    # Assign each event frame to its bin
    event_df["bin_idx"] = pd.cut(
        event_df["time_min"], bins=bin_edges, labels=range(n_bins), include_lowest=True
    ).astype(int)

    # Group by fly + bin: mean/median distance during events only
    grouped = (
        event_df.groupby(["fly", "bin_idx"])["fly_ball_dist_mm"]
        .agg(mean_dist_mm="mean", median_dist_mm="median")
        .reset_index()
    )

    # Re-index to include all (fly, bin) combinations — missing bins → NaN
    all_flies = df["fly"].unique()
    full_index = pd.MultiIndex.from_product([all_flies, range(n_bins)], names=["fly", "bin_idx"])
    grouped = grouped.set_index(["fly", "bin_idx"]).reindex(full_index).reset_index()

    # Add metadata
    grouped["Magnet"] = grouped["fly"].map(fly_meta["Magnet"])
    grouped["bin_start_min"] = grouped["bin_idx"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_center_min"] = grouped["bin_idx"].map(dict(enumerate(bin_centers)))

    n_events = event_df["interaction_event"].notna().sum()
    print(f"\nProximity (during events only): {n_events} event frames used")
    print(f"  Overall mean dist during events: {grouped['mean_dist_mm'].mean():.2f} mm")

    return grouped.sort_values(["fly", "bin_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------


def permutation_test_between_groups(
    rate_df: pd.DataFrame,
    group_col: str = "Magnet",
    control_val: str = "n",
    test_val: str = "y",
    metric: str = "rate_per_min",
    n_permutations: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """
    Permutation test (two-group, per time bin): Magnet=y vs Magnet=n.
    No FDR correction for a single pair.
    """
    bins = sorted(rate_df["bin_idx"].unique())
    results = {
        k: []
        for k in [
            "bin_idx",
            "bin_center_min",
            "obs_diff",
            "p_value",
            "control_mean",
            "test_mean",
        ]
    }

    print(f"\nPermutation test: {test_val} vs {control_val} ({n_permutations} permutations per bin)...")
    for b in tqdm(bins, desc="  bins"):
        bdata = rate_df[rate_df["bin_idx"] == b]
        ctrl = bdata[bdata[group_col] == control_val][metric].values
        test = bdata[bdata[group_col] == test_val][metric].values

        if len(ctrl) == 0 or len(test) == 0:
            results["bin_idx"].append(b)
            results["bin_center_min"].append(bdata["bin_center_min"].iloc[0])
            results["obs_diff"].append(np.nan)
            results["p_value"].append(np.nan)
            results["control_mean"].append(np.nan)
            results["test_mean"].append(np.nan)
            continue

        obs_diff = np.mean(test) - np.mean(ctrl)
        combined = np.concatenate([ctrl, test])
        n_ctrl = len(ctrl)

        perm_diffs = np.empty(n_permutations)
        for j in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[j] = np.mean(combined[n_ctrl:]) - np.mean(combined[:n_ctrl])

        p = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

        results["bin_idx"].append(b)
        results["bin_center_min"].append(float(bdata["bin_center_min"].iloc[0]))
        results["obs_diff"].append(obs_diff)
        results["p_value"].append(p)
        results["control_mean"].append(float(np.mean(ctrl)))
        results["test_mean"].append(float(np.mean(test)))

    results_df = pd.DataFrame(results)
    results_df["p_value_fdr"] = np.nan
    valid_mask = results_df["p_value"].notna()
    if valid_mask.any():
        _, p_corr, _, _ = multipletests(results_df.loc[valid_mask, "p_value"].values, method="fdr_bh")
        results_df.loc[valid_mask, "p_value_fdr"] = p_corr
    results_df["significant_fdr"] = results_df["p_value_fdr"] < alpha

    sig = (results_df["p_value"] < alpha).sum()
    sig_fdr = results_df["significant_fdr"].sum()
    print(f"  Significant bins (raw p<0.05): {sig}/{len(bins)}")
    print(f"  Significant bins (FDR q<0.05): {sig_fdr}/{len(bins)}")
    return results_df


def export_interaction_rate_gate_opening_csv(
    rate_df: pd.DataFrame,
    between_results: pd.DataFrame,
    output_dir: Path,
    focus_hours: float,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Export detailed per-bin interaction-rate stats relative to gate opening."""
    gate_min = focus_hours * 60.0
    rows = []

    for _, row in between_results.iterrows():
        bin_idx = int(row["bin_idx"])
        bdata = rate_df[rate_df["bin_idx"] == bin_idx]

        control_vals = bdata[bdata["Magnet"] == "n"]["rate_per_min"].dropna().values
        test_vals = bdata[bdata["Magnet"] == "y"]["rate_per_min"].dropna().values

        n_control = int(len(control_vals))
        n_test = int(len(test_vals))
        mean_control = float(np.mean(control_vals)) if n_control > 0 else np.nan
        mean_test = float(np.mean(test_vals)) if n_test > 0 else np.nan
        diff = mean_test - mean_control if np.isfinite(mean_control) and np.isfinite(mean_test) else np.nan

        if n_control > 0 and n_test > 0:
            ci_lower, ci_upper = bootstrap_ci_difference(
                control_vals,
                test_vals,
                n_bootstrap=n_bootstrap,
                ci=95,
                random_state=42 + bin_idx,
            )
        else:
            ci_lower, ci_upper = np.nan, np.nan

        if np.isfinite(mean_control) and mean_control != 0 and np.isfinite(diff):
            pct_change = (diff / mean_control) * 100.0
            pct_ci_lower = (ci_lower / mean_control) * 100.0 if np.isfinite(ci_lower) else np.nan
            pct_ci_upper = (ci_upper / mean_control) * 100.0 if np.isfinite(ci_upper) else np.nan
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan

        bin_center = float(row["bin_center_min"])
        bin_start = float(bdata["bin_start_min"].iloc[0]) if len(bdata) > 0 else np.nan
        # Back-calculate bin end from center/start where available
        if np.isfinite(bin_center) and np.isfinite(bin_start):
            bin_end = float(2 * bin_center - bin_start)
        else:
            bin_end = np.nan

        rows.append(
            {
                "Time_Bin": bin_idx,
                "Bin_Start_min": bin_start,
                "Bin_End_min": bin_end,
                "Bin_Center_min": bin_center,
                "Gate_Opening_min": gate_min,
                "Minutes_From_Gate": bin_center - gate_min,
                "Gate_Phase": "pre_gate" if bin_center < gate_min else "post_gate",
                "ConditionA": "n",
                "ConditionB": "y",
                "N_Control": n_control,
                "N_Test": n_test,
                "Mean_Control": mean_control,
                "Mean_Test": mean_test,
                "Difference_Test_minus_Control": diff,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "Pct_Change": pct_change,
                "Pct_CI_Lower": pct_ci_lower,
                "Pct_CI_Upper": pct_ci_upper,
                "Cohens_D": cohens_d(control_vals, test_vals) if n_control > 1 and n_test > 1 else np.nan,
                "P_Value_Raw": row.get("p_value", np.nan),
                "P_Value_FDR": row.get("p_value_fdr", np.nan),
                "Significant_FDR": bool(row.get("significant_fdr", False)),
                "N_Bootstrap": int(n_bootstrap),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = output_dir / "interaction_rate_gate_opening_detailed.csv"
    out_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  Saved: {out_csv}")
    return out_df


def export_interaction_rate_unlocking_csvs(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    output_dir: Path,
    bin_size_min: float,
    n_permutations: int,
    window_min: float = 60.0,
    gate_open_min: float | None = None,
    gate_close_min: float | None = None,
    n_bootstrap: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export interaction-rate CSVs aligned to first successful ball unlocking event."""
    if first_push_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    merged = rate_df.merge(first_push_df[["fly", "Magnet", "t_first_push_min"]], on=["fly", "Magnet"], how="inner")
    if gate_open_min is not None:
        merged = merged[merged["bin_center_min"] >= gate_open_min].copy()
    if gate_close_min is not None:
        merged = merged[merged["bin_center_min"] <= gate_close_min].copy()
    merged["Minutes_From_First_Unlock"] = merged["bin_center_min"] - merged["t_first_push_min"]
    merged["Rel_Bin_min"] = (merged["Minutes_From_First_Unlock"] / bin_size_min).round() * bin_size_min
    aligned = merged[merged["Rel_Bin_min"].abs() <= window_min].copy()

    per_fly_csv = output_dir / "interaction_rate_aligned_to_first_unlock_per_fly.csv"
    aligned.sort_values(["Magnet", "fly", "Rel_Bin_min"]).to_csv(per_fly_csv, index=False, float_format="%.6f")
    print(f"  Saved: {per_fly_csv}")

    rows = []
    rng = np.random.default_rng(42)
    rel_bins = sorted(aligned["Rel_Bin_min"].dropna().unique().tolist())
    for rel_bin in rel_bins:
        b = aligned[aligned["Rel_Bin_min"] == rel_bin]
        ctrl = b[b["Magnet"] == "n"]["rate_per_min"].dropna().values
        test = b[b["Magnet"] == "y"]["rate_per_min"].dropna().values
        if len(ctrl) == 0 or len(test) == 0:
            p_raw = np.nan
            obs_diff = np.nan
            ci_lower, ci_upper = np.nan, np.nan
            mean_ctrl, mean_test = np.nan, np.nan
            pct_change, pct_ci_lower, pct_ci_upper = np.nan, np.nan, np.nan
        else:
            mean_ctrl = float(np.mean(ctrl))
            mean_test = float(np.mean(test))
            obs_diff = mean_test - mean_ctrl

            combined = np.concatenate([ctrl, test])
            n_ctrl = len(ctrl)
            perm_diffs = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                rng.shuffle(combined)
                perm_diffs[i] = np.mean(combined[n_ctrl:]) - np.mean(combined[:n_ctrl])
            p_raw = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

            ci_lower, ci_upper = bootstrap_ci_difference(
                ctrl,
                test,
                n_bootstrap=n_bootstrap,
                ci=95,
                random_state=4242 + int(rel_bin * 10),
            )

            if mean_ctrl != 0:
                pct_change = (obs_diff / mean_ctrl) * 100.0
                pct_ci_lower = (ci_lower / mean_ctrl) * 100.0 if np.isfinite(ci_lower) else np.nan
                pct_ci_upper = (ci_upper / mean_ctrl) * 100.0 if np.isfinite(ci_upper) else np.nan
            else:
                pct_change, pct_ci_lower, pct_ci_upper = np.nan, np.nan, np.nan

        rows.append(
            {
                "Rel_Bin_min": rel_bin,
                "Unlock_Phase": "pre_unlock" if rel_bin < 0 else "post_unlock",
                "ConditionA": "n",
                "ConditionB": "y",
                "N_Control": int(len(ctrl)),
                "N_Test": int(len(test)),
                "Mean_Control": mean_ctrl,
                "Mean_Test": mean_test,
                "Difference_Test_minus_Control": obs_diff,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "Pct_Change": pct_change,
                "Pct_CI_Lower": pct_ci_lower,
                "Pct_CI_Upper": pct_ci_upper,
                "Cohens_D": cohens_d(ctrl, test) if len(ctrl) > 1 and len(test) > 1 else np.nan,
                "P_Value_Raw": p_raw,
                "N_Permutations": int(n_permutations),
                "N_Bootstrap": int(n_bootstrap),
            }
        )

    between_df = pd.DataFrame(rows)
    valid = between_df["P_Value_Raw"].notna()
    between_df["P_Value_FDR"] = np.nan
    if valid.any():
        _, p_corr, _, _ = multipletests(between_df.loc[valid, "P_Value_Raw"].values, method="fdr_bh")
        between_df.loc[valid, "P_Value_FDR"] = p_corr
    between_df["Significant_FDR"] = between_df["P_Value_FDR"] < 0.05

    between_csv = output_dir / "interaction_rate_aligned_to_first_unlock_between_group_stats.csv"
    between_df.to_csv(between_csv, index=False, float_format="%.6f")
    print(f"  Saved: {between_csv}")
    return aligned, between_df


def trend_test_per_group(
    rate_df: pd.DataFrame,
    focus_hours: float = 1.0,
    start_hours: float = 0.0,
    group_col: str = "Magnet",
    metric: str = "rate_per_min",
    n_permutations: int = 10000,
) -> dict:
    """
    For each Magnet group, fit a per-fly linear regression of rate ~ time_bin
    over the window [start_hours, focus_hours]. Then test whether the
    distribution of slopes is significantly different from zero (one-sample
    permutation test).

    Returns a dict with results per group.
    """
    start_min = start_hours * 60.0
    focus_min = focus_hours * 60.0
    sub = rate_df[(rate_df["bin_center_min"] >= start_min) & (rate_df["bin_center_min"] <= focus_min)].copy()

    group_results = {}
    for group, gdf in sub.groupby(group_col):
        slopes = []
        for fly, fdf in gdf.groupby("fly"):
            x = fdf["bin_center_min"].values
            y = fdf[metric].values
            if len(x) < 2:
                continue
            slope, intercept, r, p, se = stats.linregress(x, y)
            slopes.append(slope)

        slopes = np.array(slopes)
        observed_mean_slope = np.mean(slopes)

        # Permutation: is the mean slope different from zero?
        # Under H0, signs are random → permute signs
        perm_means = np.empty(n_permutations)
        for j in range(n_permutations):
            signs = np.random.choice([-1, 1], size=len(slopes))
            perm_means[j] = np.mean(signs * np.abs(slopes))

        p_value = np.mean(np.abs(perm_means) >= np.abs(observed_mean_slope))

        group_results[group] = {
            "group": group,
            "n_flies": len(slopes),
            "slopes": slopes,
            "mean_slope": observed_mean_slope,
            "median_slope": np.median(slopes),
            "sem_slope": np.std(slopes, ddof=1) / np.sqrt(len(slopes)),
            "p_value": p_value,
        }

        direction = "DECREASING" if observed_mean_slope < 0 else "INCREASING"
        print(
            f"  Magnet={group}: mean slope = {observed_mean_slope:.5f} events/min²"
            f"  [{direction}], p = {format_p_value(p_value, n_permutations)}"
        )

    return group_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_palette(groups):
    """Return a dict mapping 'y'→blue, 'n'→orange (or generic colors)."""
    color_map = {"y": "#3953A4", "n": "#faa41a"}
    return {g: color_map.get(str(g), f"C{i}") for i, g in enumerate(sorted(groups))}


def label_group(g, n):
    if g == "y":
        return f"Magnet block (n={n})"
    elif g == "n":
        return f"Control (n={n})"
    return f"{g} (n={n})"


def plot_interaction_rate(
    rate_df: pd.DataFrame,
    between_results: pd.DataFrame,
    trend_results: dict,
    focus_hours: float,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Top : event rate (mean ± 95 % CI from seaborn) vs time, both groups,
            with significance annotations per bin.
      Bottom: slope distribution (violin + individual points) per group,
              to illustrate whether the within-group trend is significant.
    """
    groups = sorted(rate_df["Magnet"].unique())
    n_per_group = {g: rate_df[rate_df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}

    rate_df = rate_df.copy()
    rate_df["Group"] = rate_df["Magnet"].map(label_map)
    palette_labels = {label_map[g]: palette[g] for g in groups}

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.35)

    # ---- Top panel: rate over time ----------------------------------------
    ax_rate = fig.add_subplot(gs[0, :])

    sns.lineplot(
        data=rate_df,
        x="bin_center_min",
        y="rate_per_min",
        hue="Group",
        palette=palette_labels,
        errorbar=("ci", 95),
        ax=ax_rate,
    )

    # Significance annotations (between-group test) — only show asterisks for significant bins
    y_min, y_max = ax_rate.get_ylim()
    y_range = y_max - y_min
    ax_rate.set_ylim(y_min, y_max + 0.12 * y_range)
    y_star = y_max + 0.05 * y_range

    for _, row in between_results.iterrows():
        bx = row["bin_center_min"]
        p = row["p_value"]
        if pd.isna(p):
            continue
        stars = sig_stars(p)
        if stars != "ns":
            ax_rate.text(bx, y_star, stars, ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")

    # Mark focus-hour boundary
    ax_rate.axvline(
        focus_hours * 60, color="black", linestyle="--", lw=1.5, alpha=0.7, label=f"{int(focus_hours)}h boundary"
    )

    ax_rate.set_xlabel("Time (min)", fontsize=12)
    ax_rate.set_ylabel("Interaction rate (events / min)", fontsize=12)
    ax_rate.set_title("Interaction rate over time — Magnet block vs Control", fontsize=13)
    ax_rate.legend(loc="upper right", fontsize=10)

    # ---- Bottom-left: slope violin per group --------------------------------
    ax_slope = fig.add_subplot(gs[1, 0])

    slope_rows = []
    for g, res in trend_results.items():
        for s in res["slopes"]:
            slope_rows.append({"Magnet": g, "Group": label_map[g], "slope": s})
    slope_df = pd.DataFrame(slope_rows)

    sns.violinplot(
        data=slope_df,
        x="Group",
        y="slope",
        palette=palette_labels,
        inner=None,
        ax=ax_slope,
        cut=0,
    )
    sns.stripplot(
        data=slope_df,
        x="Group",
        y="slope",
        palette=palette_labels,
        size=3,
        alpha=0.6,
        jitter=True,
        ax=ax_slope,
    )
    ax_slope.axhline(0, color="black", lw=1, linestyle="--")

    # Annotate p-values on the violin plot
    for i, g in enumerate(sorted(trend_results.keys())):
        res = trend_results[g]
        p = res["p_value"]
        stars = sig_stars(p)
        p_str = format_p_value(p, n_permutations)
        y_annot = slope_df["slope"].max() * 1.05
        ax_slope.text(
            i, y_annot, f"{stars}\n{p_str}", ha="center", va="bottom", fontsize=8, color="red" if p < 0.05 else "gray"
        )

    ax_slope.set_xlabel("")
    ax_slope.set_ylabel("Slope (events/min per min)", fontsize=10)
    ax_slope.set_title(f"Per-fly trend slope\n(first {int(focus_hours * 60)} min)", fontsize=11)
    ax_slope.tick_params(axis="x", labelsize=9)

    # ---- Bottom-right: mean slopes with error bars --------------------------
    ax_mean = fig.add_subplot(gs[1, 1])

    x_pos = []
    x_labels = []
    for i, g in enumerate(sorted(trend_results.keys())):
        res = trend_results[g]
        ax_mean.bar(
            i,
            res["mean_slope"],
            yerr=res["sem_slope"],
            color=palette[g],
            alpha=0.8,
            capsize=5,
            width=0.5,
        )
        x_pos.append(i)
        x_labels.append(label_map[g])

    ax_mean.axhline(0, color="black", lw=1, linestyle="--")
    ax_mean.set_xticks(x_pos)
    ax_mean.set_xticklabels(x_labels, fontsize=9)
    ax_mean.set_ylabel("Mean slope ± SEM", fontsize=10)
    ax_mean.set_title(f"Mean trend (first {int(focus_hours * 60)} min)", fontsize=11)

    plt.suptitle(
        "MagnetBlock — Interaction Rate Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # Save
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")

    plt.close(fig)


def plot_interaction_duration(
    dur_df: pd.DataFrame,
    between_results: pd.DataFrame,
    trend_results: dict,
    focus_hours: float,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
) -> None:
    """
    Two-panel figure for interaction *duration* (mean seconds per event):
      Top    : mean duration per bin (mean ± 95 % CI) vs time, both groups,
               with per-bin between-group significance annotations.
      Bottom : per-fly slope distributions for the first-hour trend.

    Flies with no events in a bin contribute NaN and are excluded from
    that bin's mean (rather than zero-inflating it); this reflects true
    persistence of flies that do interact.
    """
    groups = sorted(dur_df["Magnet"].unique())
    n_per_group = {g: dur_df[dur_df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}

    dur_df = dur_df.copy()
    dur_df["Group"] = dur_df["Magnet"].map(label_map)
    palette_labels = {label_map[g]: palette[g] for g in groups}

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.35)

    # ---- Top panel: mean duration over time ----------------------------------
    ax_dur = fig.add_subplot(gs[0, :])

    # Drop bins where a fly had no events (NaN duration) so seaborn CI is
    # computed only over flies that actually interacted in that bin
    plot_data = dur_df.dropna(subset=["mean_duration_s"])

    sns.lineplot(
        data=plot_data,
        x="bin_center_min",
        y="mean_duration_s",
        hue="Group",
        palette=palette_labels,
        errorbar=("ci", 95),
        ax=ax_dur,
    )

    # Significance annotations — only show asterisks for significant bins
    y_min, y_max = ax_dur.get_ylim()
    y_range = y_max - y_min
    ax_dur.set_ylim(y_min, y_max + 0.12 * y_range)
    y_star = y_max + 0.05 * y_range

    for _, row in between_results.iterrows():
        bx = row["bin_center_min"]
        p = row["p_value"]
        if pd.isna(p):
            continue
        stars = sig_stars(p)
        if stars != "ns":
            ax_dur.text(bx, y_star, stars, ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")

    # Mark focus-hour boundary
    ax_dur.axvline(
        focus_hours * 60,
        color="black",
        linestyle="--",
        lw=1.5,
        alpha=0.7,
        label=f"{int(focus_hours)}h boundary",
    )

    ax_dur.set_xlabel("Time (min)", fontsize=12)
    ax_dur.set_ylabel("Mean interaction duration (s)", fontsize=12)
    ax_dur.set_title(
        "Interaction duration over time — Magnet block vs Control\n"
        "(bins with no events for a fly excluded from that fly's mean)",
        fontsize=13,
    )
    ax_dur.legend(loc="upper right", fontsize=10)

    # ---- Bottom-left: slope violin per group --------------------------------
    ax_slope = fig.add_subplot(gs[1, 0])

    slope_rows = []
    for g, res in trend_results.items():
        for s in res["slopes"]:
            slope_rows.append({"Magnet": g, "Group": label_map[g], "slope": s})
    slope_df = pd.DataFrame(slope_rows)

    if not slope_df.empty:
        sns.violinplot(
            data=slope_df,
            x="Group",
            y="slope",
            palette=palette_labels,
            inner=None,
            ax=ax_slope,
            cut=0,
        )
        sns.stripplot(
            data=slope_df,
            x="Group",
            y="slope",
            palette=palette_labels,
            size=3,
            alpha=0.6,
            jitter=True,
            ax=ax_slope,
        )
        ax_slope.axhline(0, color="black", lw=1, linestyle="--")

        for i, g in enumerate(sorted(trend_results.keys())):
            res = trend_results[g]
            p = res["p_value"]
            stars = sig_stars(p)
            p_str = format_p_value(p, n_permutations)
            y_annot = slope_df["slope"].max() * 1.05
            ax_slope.text(
                i,
                y_annot,
                f"{stars}\n{p_str}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red" if p < 0.05 else "gray",
            )

    ax_slope.set_xlabel("")
    ax_slope.set_ylabel("Slope (s per min)", fontsize=10)
    ax_slope.set_title(
        f"Per-fly duration trend slope\n(first {int(focus_hours * 60)} min)",
        fontsize=11,
    )
    ax_slope.tick_params(axis="x", labelsize=9)

    # ---- Bottom-right: mean slopes with error bars --------------------------
    ax_mean = fig.add_subplot(gs[1, 1])

    x_pos = []
    x_labels = []
    for i, g in enumerate(sorted(trend_results.keys())):
        res = trend_results[g]
        ax_mean.bar(
            i,
            res["mean_slope"],
            yerr=res["sem_slope"],
            color=palette[g],
            alpha=0.8,
            capsize=5,
            width=0.5,
        )
        x_pos.append(i)
        x_labels.append(label_map[g])

    ax_mean.axhline(0, color="black", lw=1, linestyle="--")
    ax_mean.set_xticks(x_pos)
    ax_mean.set_xticklabels(x_labels, fontsize=9)
    ax_mean.set_ylabel("Mean slope ± SEM (s/min)", fontsize=10)
    ax_mean.set_title(
        f"Mean duration trend (first {int(focus_hours * 60)} min)",
        fontsize=11,
    )

    plt.suptitle(
        "MagnetBlock — Interaction Duration (Persistence) Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistics export
# ---------------------------------------------------------------------------


def save_stats(
    between_results: pd.DataFrame,
    trend_results: dict,
    rate_df: pd.DataFrame,
    output_dir: Path,
    n_permutations: int,
    focus_hours: float,
    bin_size_min: float,
) -> None:
    """Write CSV and Markdown summaries."""

    # Between-group test CSV
    csv_between = output_dir / "interaction_rate_between_group_stats.csv"
    between_results.to_csv(csv_between, index=False)
    print(f"  Saved: {csv_between}")

    # Trend test CSV
    trend_rows = []
    for g, res in trend_results.items():
        trend_rows.append(
            {
                "Magnet": g,
                "n_flies": res["n_flies"],
                "mean_slope": res["mean_slope"],
                "median_slope": res["median_slope"],
                "sem_slope": res["sem_slope"],
                "p_value": res["p_value"],
                "significant": sig_stars(res["p_value"]),
            }
        )
    csv_trend = output_dir / "interaction_rate_trend_stats.csv"
    pd.DataFrame(trend_rows).to_csv(csv_trend, index=False)
    print(f"  Saved: {csv_trend}")

    # Per-fly per-bin rates CSV
    csv_rates = output_dir / "interaction_rate_per_fly_per_bin.csv"
    rate_df.to_csv(csv_rates, index=False)
    print(f"  Saved: {csv_rates}")

    # Markdown summary
    md_path = output_dir / "interaction_rate_statistics.md"
    with open(md_path, "w") as f:
        f.write("# MagnetBlock Interaction Rate Analysis\n\n")
        f.write(f"**Bin size:** {bin_size_min} min\n\n")
        f.write(f"**Permutations:** {n_permutations}\n\n")
        f.write(f"**Trend analysis window:** first {int(focus_hours * 60)} min\n\n")

        f.write("## Between-group test (Magnet=y vs Magnet=n) per time bin\n\n")
        f.write("| Bin center (min) | Control mean (ev/min) | Magnet mean (ev/min) | Diff | P-value | Sig |\n")
        f.write("|-----------------|----------------------|---------------------|------|---------|-----|\n")
        for _, row in between_results.iterrows():
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations) if not pd.isna(p) else "—"
            stars = sig_stars(p) if not pd.isna(p) else "—"
            f.write(
                f"| {row['bin_center_min']:.1f} "
                f"| {row['control_mean']:.4f} "
                f"| {row['test_mean']:.4f} "
                f"| {row['obs_diff']:.4f} "
                f"| {p_str} "
                f"| {stars} |\n"
            )

        f.write("\n## Trend test (first hour, per fly linear regression)\n\n")
        f.write("Slopes are in units of **events / min²** (change in rate per minute).\n\n")
        f.write("| Group | N flies | Mean slope | Median slope | SEM | P-value | Sig |\n")
        f.write("|-------|---------|------------|--------------|-----|---------|-----|\n")
        for row in trend_rows:
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations)
            f.write(
                f"| {row['Magnet']} "
                f"| {row['n_flies']} "
                f"| {row['mean_slope']:.6f} "
                f"| {row['median_slope']:.6f} "
                f"| {row['sem_slope']:.6f} "
                f"| {p_str} "
                f"| {row['significant']} |\n"
            )

    print(f"  Saved: {md_path}")


def save_duration_stats(
    between_results: pd.DataFrame,
    trend_results: dict,
    dur_df: pd.DataFrame,
    output_dir: Path,
    n_permutations: int,
    focus_hours: float,
    bin_size_min: float,
) -> None:
    """Write CSV and Markdown summaries for the duration analysis."""

    csv_between = output_dir / "interaction_duration_between_group_stats.csv"
    between_results.to_csv(csv_between, index=False)
    print(f"  Saved: {csv_between}")

    trend_rows = []
    for g, res in trend_results.items():
        trend_rows.append(
            {
                "Magnet": g,
                "n_flies": res["n_flies"],
                "mean_slope": res["mean_slope"],
                "median_slope": res["median_slope"],
                "sem_slope": res["sem_slope"],
                "p_value": res["p_value"],
                "significant": sig_stars(res["p_value"]),
            }
        )
    csv_trend = output_dir / "interaction_duration_trend_stats.csv"
    pd.DataFrame(trend_rows).to_csv(csv_trend, index=False)
    print(f"  Saved: {csv_trend}")

    csv_dur = output_dir / "interaction_duration_per_fly_per_bin.csv"
    dur_df.to_csv(csv_dur, index=False)
    print(f"  Saved: {csv_dur}")

    md_path = output_dir / "interaction_duration_statistics.md"
    with open(md_path, "w") as f:
        f.write("# MagnetBlock Interaction Duration (Persistence) Analysis\n\n")
        f.write(f"**Bin size:** {bin_size_min} min\n\n")
        f.write(f"**Frame rate:** {FPS} Hz\n\n")
        f.write(f"**Permutations:** {n_permutations}\n\n")
        f.write(f"**Trend analysis window:** first {int(focus_hours * 60)} min\n\n")
        f.write(
            "> **Note:** bins where a fly had no events contribute NaN and are\n"
            "> excluded from that fly's mean for that bin. The between-group test\n"
            "> compares mean durations only among flies that interacted in each bin.\n\n"
        )

        f.write("## Between-group test per time bin\n\n")
        f.write("| Bin center (min) | Control mean (s) | Magnet mean (s) | Diff (s) | P-value | Sig |\n")
        f.write("|-----------------|-----------------|----------------|----------|---------|-----|\n")
        for _, row in between_results.iterrows():
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations) if not pd.isna(p) else "—"
            stars = sig_stars(p) if not pd.isna(p) else "—"
            f.write(
                f"| {row['bin_center_min']:.1f} "
                f"| {row['control_mean']:.3f} "
                f"| {row['test_mean']:.3f} "
                f"| {row['obs_diff']:.3f} "
                f"| {p_str} "
                f"| {stars} |\n"
            )

        f.write("\n## Trend test (first hour, per-fly linear regression)\n\n")
        f.write("Slopes are in **s/min** (change in mean duration per minute of experiment time).\n\n")
        f.write("| Group | N flies | Mean slope | Median slope | SEM | P-value | Sig |\n")
        f.write("|-------|---------|------------|--------------|-----|---------|-----|\n")
        for row in trend_rows:
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations)
            f.write(
                f"| {row['Magnet']} "
                f"| {row['n_flies']} "
                f"| {row['mean_slope']:.6f} "
                f"| {row['median_slope']:.6f} "
                f"| {row['sem_slope']:.6f} "
                f"| {p_str} "
                f"| {row['significant']} |\n"
            )

    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# Individual-fly spaghetti plots (Magnet=y only)
# ---------------------------------------------------------------------------


def _spaghetti_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    focus_hours: float,
    bin_size_min: float,
    ylabel: str,
    title: str,
    color: str = "#3953A4",
) -> None:
    """Draw one individual-fly spaghetti panel onto *ax*."""
    flies = sub["fly"].unique()
    for fly in flies:
        fd = sub[sub["fly"] == fly].dropna(subset=[y_col]).sort_values(x_col)
        ax.plot(
            fd[x_col],
            fd[y_col],
            color=color,
            alpha=0.35,
            lw=0.9,
        )

    # Bold mean line on top
    mean_line = sub.groupby(x_col)[y_col].mean().reset_index()
    ax.plot(
        mean_line[x_col],
        mean_line[y_col],
        color="black",
        lw=2.5,
        zorder=10,
        label=f"Mean (n={len(flies)} flies)",
    )

    # Focus-hour boundary
    ax.axvline(
        focus_hours * 60,
        color="red",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label=f"{int(focus_hours)}h boundary",
    )

    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")


def plot_individual_rate(
    rate_df: pd.DataFrame,
    focus_hours: float,
    bin_size_min: float,
    output_path: Path,
) -> None:
    """
    Individual-fly interaction-rate lines for Magnet=y only.
    Each fly is a semi-transparent line; the group mean is a bold black line.
    No confidence intervals.
    """
    sub = rate_df[rate_df["Magnet"] == "y"].copy()
    n_flies = sub["fly"].nunique()

    fig, ax = plt.subplots(figsize=(13, 5))

    _spaghetti_panel(
        ax=ax,
        sub=sub,
        x_col="bin_center_min",
        y_col="rate_per_min",
        focus_hours=focus_hours,
        bin_size_min=bin_size_min,
        ylabel="Interaction rate (events / min)",
        title=f"Magnet block — individual fly interaction rates (n={n_flies})",
        color="#3953A4",
    )

    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_individual_duration(
    dur_df: pd.DataFrame,
    focus_hours: float,
    bin_size_min: float,
    output_path: Path,
) -> None:
    """
    Individual-fly interaction-duration lines for Magnet=y only.
    Each fly is a semi-transparent line; the group mean is a bold black line.
    Bins where a fly had no events are gaps (NaN), not zeros.
    No confidence intervals.
    """
    sub = dur_df[dur_df["Magnet"] == "y"].copy()
    n_flies = sub["fly"].nunique()

    fig, ax = plt.subplots(figsize=(13, 5))

    _spaghetti_panel(
        ax=ax,
        sub=sub,
        x_col="bin_center_min",
        y_col="mean_duration_s",
        focus_hours=focus_hours,
        bin_size_min=bin_size_min,
        ylabel="Mean interaction duration (s)",
        title=f"Magnet block — individual fly interaction duration (n={n_flies})\n"
        "(gaps = no events in that bin for that fly)",
        color="#3953A4",
    )

    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_proximity(
    prox_df: pd.DataFrame,
    between_results: pd.DataFrame,
    trend_results: dict,
    focus_hours: float,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
) -> None:
    """
    Two-panel figure for mean fly-ball distance over time:
      Top    : mean distance (mean ± 95 % CI) vs time, both groups,
               with per-bin between-group significance annotations.
               Lower = fly closer to ball = more engaged.
      Bottom : per-fly slope distributions for the first-hour trend.
    """
    groups = sorted(prox_df["Magnet"].unique())
    n_per_group = {g: prox_df[prox_df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}

    prox_df = prox_df.copy()
    prox_df["Group"] = prox_df["Magnet"].map(label_map)
    palette_labels = {label_map[g]: palette[g] for g in groups}

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.35)

    # ---- Top panel: distance over time --------------------------------------
    ax_prox = fig.add_subplot(gs[0, :])

    sns.lineplot(
        data=prox_df,
        x="bin_center_min",
        y="mean_dist_mm",
        hue="Group",
        palette=palette_labels,
        errorbar=("ci", 95),
        ax=ax_prox,
    )

    y_min, y_max = ax_prox.get_ylim()
    y_range = y_max - y_min
    ax_prox.set_ylim(y_min, y_max + 0.12 * y_range)
    y_star = y_max + 0.05 * y_range

    for _, row in between_results.iterrows():
        bx = row["bin_center_min"]
        p = row["p_value"]
        if pd.isna(p):
            continue
        stars = sig_stars(p)
        if stars != "ns":
            ax_prox.text(bx, y_star, stars, ha="center", va="bottom", fontsize=14, color="red", fontweight="bold")
    ax_prox.axvline(
        focus_hours * 60,
        color="black",
        linestyle="--",
        lw=1.5,
        alpha=0.7,
        label=f"{int(focus_hours)}h boundary",
    )

    ax_prox.set_xlabel("Time (min)", fontsize=12)
    ax_prox.set_ylabel("Mean fly-ball distance during events (mm)", fontsize=12)
    ax_prox.set_title(
        "Fly-ball proximity during interactions — Magnet block vs Control\n"
        "(lower = fly gets closer to ball while interacting = deeper engagement)",
        fontsize=13,
    )
    ax_prox.legend(loc="upper right", fontsize=10)

    # ---- Bottom-left: slope violin ------------------------------------------
    ax_slope = fig.add_subplot(gs[1, 0])

    slope_rows = []
    for g, res in trend_results.items():
        for s in res["slopes"]:
            slope_rows.append({"Magnet": g, "Group": label_map[g], "slope": s})
    slope_df = pd.DataFrame(slope_rows)

    if not slope_df.empty:
        sns.violinplot(
            data=slope_df,
            x="Group",
            y="slope",
            palette=palette_labels,
            inner=None,
            ax=ax_slope,
            cut=0,
        )
        sns.stripplot(
            data=slope_df,
            x="Group",
            y="slope",
            palette=palette_labels,
            size=3,
            alpha=0.6,
            jitter=True,
            ax=ax_slope,
        )
        ax_slope.axhline(0, color="black", lw=1, linestyle="--")

        for i, g in enumerate(sorted(trend_results.keys())):
            res = trend_results[g]
            p = res["p_value"]
            stars = sig_stars(p)
            p_str = format_p_value(p, n_permutations)
            y_annot = slope_df["slope"].max() * 1.05
            ax_slope.text(
                i,
                y_annot,
                f"{stars}\n{p_str}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red" if p < 0.05 else "gray",
            )

    ax_slope.set_xlabel("")
    ax_slope.set_ylabel("Slope (mm / min)", fontsize=10)
    ax_slope.set_title(
        f"Per-fly distance trend slope\n(first {int(focus_hours * 60)} min)\n"
        "positive = moving away, negative = approaching",
        fontsize=10,
    )
    ax_slope.tick_params(axis="x", labelsize=9)

    # ---- Bottom-right: mean slopes bar chart --------------------------------
    ax_mean = fig.add_subplot(gs[1, 1])

    x_pos, x_labels = [], []
    for i, g in enumerate(sorted(trend_results.keys())):
        res = trend_results[g]
        ax_mean.bar(
            i,
            res["mean_slope"],
            yerr=res["sem_slope"],
            color=palette[g],
            alpha=0.8,
            capsize=5,
            width=0.5,
        )
        x_pos.append(i)
        x_labels.append(label_map[g])

    ax_mean.axhline(0, color="black", lw=1, linestyle="--")
    ax_mean.set_xticks(x_pos)
    ax_mean.set_xticklabels(x_labels, fontsize=9)
    ax_mean.set_ylabel("Mean slope ± SEM (mm/min)", fontsize=10)
    ax_mean.set_title(
        f"Mean distance trend (first {int(focus_hours * 60)} min)",
        fontsize=11,
    )

    plt.suptitle(
        "MagnetBlock — Fly-Ball Proximity (Engagement) Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_individual_proximity(
    prox_df: pd.DataFrame,
    focus_hours: float,
    bin_size_min: float,
    output_path: Path,
) -> None:
    """
    Individual-fly fly-ball distance lines for Magnet=y only.
    Each fly is a semi-transparent line; the group mean is bold black.
    No confidence intervals.
    """
    sub = prox_df[prox_df["Magnet"] == "y"].copy()
    n_flies = sub["fly"].nunique()

    fig, ax = plt.subplots(figsize=(13, 5))

    _spaghetti_panel(
        ax=ax,
        sub=sub,
        x_col="bin_center_min",
        y_col="mean_dist_mm",
        focus_hours=focus_hours,
        bin_size_min=bin_size_min,
        ylabel="Mean fly-ball distance during events (mm)",
        title=f"Magnet block — individual fly proximity during interactions (n={n_flies})\n"
        "(lower = closer during events = deeper engagement; gaps = no events in bin)",
        color="#3953A4",
    )

    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def save_proximity_stats(
    between_results: pd.DataFrame,
    trend_results: dict,
    prox_df: pd.DataFrame,
    output_dir: Path,
    n_permutations: int,
    focus_hours: float,
    bin_size_min: float,
) -> None:
    """Write CSV and Markdown summaries for the proximity analysis."""

    csv_between = output_dir / "proximity_between_group_stats.csv"
    between_results.to_csv(csv_between, index=False)
    print(f"  Saved: {csv_between}")

    trend_rows = []
    for g, res in trend_results.items():
        trend_rows.append(
            {
                "Magnet": g,
                "n_flies": res["n_flies"],
                "mean_slope": res["mean_slope"],
                "median_slope": res["median_slope"],
                "sem_slope": res["sem_slope"],
                "p_value": res["p_value"],
                "significant": sig_stars(res["p_value"]),
            }
        )
    csv_trend = output_dir / "proximity_trend_stats.csv"
    pd.DataFrame(trend_rows).to_csv(csv_trend, index=False)
    print(f"  Saved: {csv_trend}")

    csv_prox = output_dir / "proximity_per_fly_per_bin.csv"
    prox_df.to_csv(csv_prox, index=False)
    print(f"  Saved: {csv_prox}")

    md_path = output_dir / "proximity_statistics.md"
    with open(md_path, "w") as f:
        f.write("# MagnetBlock Fly-Ball Proximity (Engagement) Analysis\n\n")
        f.write(f"**Bin size:** {bin_size_min} min\n\n")
        f.write(
            f"**Distance computed from:** x_fly_0/y_fly_0 vs x_ball_0/y_ball_0, **during interaction events only**\n\n"
        )
        f.write(f"**Pixel → mm:** 500 px = 30 mm\n\n")
        f.write(f"**Permutations:** {n_permutations}\n\n")
        f.write(f"**Trend analysis window:** first {int(focus_hours * 60)} min\n\n")
        f.write(
            "> Lower distance = fly gets closer to ball while interacting = deeper engagement.\n"
            "> Bins where a fly had no events contribute NaN and are excluded from that bin's mean.\n\n"
        )

        f.write("## Between-group test per time bin\n\n")
        f.write("| Bin center (min) | Control mean (mm) | Magnet mean (mm) | Diff (mm) | P-value | Sig |\n")
        f.write("|-----------------|-------------------|-----------------|-----------|---------|-----|\n")
        for _, row in between_results.iterrows():
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations) if not pd.isna(p) else "—"
            stars = sig_stars(p) if not pd.isna(p) else "—"
            f.write(
                f"| {row['bin_center_min']:.1f} "
                f"| {row['control_mean']:.3f} "
                f"| {row['test_mean']:.3f} "
                f"| {row['obs_diff']:.3f} "
                f"| {p_str} "
                f"| {stars} |\n"
            )

        f.write("\n## Trend test (first hour, per-fly linear regression)\n\n")
        f.write("Slopes are in **mm/min** (positive = moving away from ball over time).\n\n")
        f.write("| Group | N flies | Mean slope | Median slope | SEM | P-value | Sig |\n")
        f.write("|-------|---------|------------|--------------|-----|---------|-----|\n")
        for row in trend_rows:
            p = row["p_value"]
            p_str = format_p_value(p, n_permutations)
            f.write(
                f"| {row['Magnet']} "
                f"| {row['n_flies']} "
                f"| {row['mean_slope']:.6f} "
                f"| {row['median_slope']:.6f} "
                f"| {row['sem_slope']:.6f} "
                f"| {p_str} "
                f"| {row['significant']} |\n"
            )
    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# Summary dashboard — all metrics in one figure
# ---------------------------------------------------------------------------


def plot_summary_dashboard(
    rate_df: pd.DataFrame,
    dur_df: pd.DataFrame,
    prox_df: pd.DataFrame,
    iei_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    vel_df: pd.DataFrame,
    chamber_df: pd.DataFrame,
    rate_between: pd.DataFrame,
    dur_between: pd.DataFrame,
    prox_between: pd.DataFrame,
    iei_between: pd.DataFrame,
    disp_between: pd.DataFrame,
    vel_between: pd.DataFrame,
    chamber_between: pd.DataFrame,
    focus_hours: float,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
) -> None:
    """
    Summary figure showing all metrics (one row each) side by side:

    Left column  — between-group comparison (mean ± 95 % CI, both groups)
                   with per-bin significance annotations.
    Right column — Magnet=y individual flies (spaghetti + bold mean).

    Rows: rate, duration, proximity, IEI, ball displacement,
          approach speed, time in chamber.
    """
    groups = sorted(rate_df["Magnet"].unique())
    n_per_group = {g: rate_df[rate_df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}
    palette_labels = {label_map[g]: palette[g] for g in groups}

    # Metric specs: (df, y_col, between_df, ylabel, dropna_col)
    metrics = [
        (rate_df, "rate_per_min", rate_between, "Rate (events/min)", None),
        (dur_df, "mean_duration_s", dur_between, "Duration (s)", "mean_duration_s"),
        (prox_df, "mean_dist_mm", prox_between, "Fly-ball dist. (mm)", "mean_dist_mm"),
        (iei_df, "mean_iei_s", iei_between, "IEI (s)", "mean_iei_s"),
        (disp_df, "mean_displacement_px", disp_between, "Ball displacement (px)", "mean_displacement_px"),
        (vel_df, "mean_approach_speed_px_s", vel_between, "Approach speed (px/s)", "mean_approach_speed_px_s"),
        (chamber_df, "fraction_in_chamber", chamber_between, "Fraction in chamber", "fraction_in_chamber"),
    ]
    metric_titles = [
        "Interaction rate",
        "Interaction duration",
        "Fly-ball proximity (during events)",
        "Inter-event interval",
        "Ball displacement per event",
        "Approach speed (2s pre-event)",
        "Time in chamber (r=50px)",
    ]

    n_rows = len(metrics)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(18, 4 * n_rows),
        sharex="col",
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.28)

    for row_idx, (df, y_col, btwn, ylabel, dropna_col) in enumerate(metrics):
        ax_grp = axes[row_idx, 0]  # left: group comparison
        ax_ind = axes[row_idx, 1]  # right: individual flies (Magnet=y)

        # ---- Left: group mean ± CI ----------------------------------------
        plot_df = df.copy()
        if dropna_col:
            plot_df = plot_df.dropna(subset=[dropna_col])
        plot_df["Group"] = plot_df["Magnet"].map(label_map)

        sns.lineplot(
            data=plot_df,
            x="bin_center_min",
            y=y_col,
            hue="Group",
            palette=palette_labels,
            errorbar=("ci", 95),
            ax=ax_grp,
            legend=(row_idx == 0),  # legend only on first row
        )

        # Significance annotations
        y_bot, y_top = ax_grp.get_ylim()
        y_rng = y_top - y_bot
        ax_grp.set_ylim(y_bot, y_top + 0.12 * y_rng)
        y_star = y_top + 0.04 * y_rng

        for _, brow in btwn.iterrows():
            bx = brow["bin_center_min"]
            p = brow["p_value"]
            if pd.isna(p):
                continue
            stars = sig_stars(p)
            if stars != "ns":
                ax_grp.text(bx, y_star, stars, ha="center", va="bottom", fontsize=11, color="red", fontweight="bold")

        ax_grp.axvline(focus_hours * 60, color="black", linestyle="--", lw=1.2, alpha=0.7)

        ax_grp.set_ylabel(ylabel, fontsize=10)
        ax_grp.set_title(f"{metric_titles[row_idx]} — group comparison", fontsize=10)
        if row_idx == n_rows - 1:
            ax_grp.set_xlabel("Time (min)", fontsize=10)
        else:
            ax_grp.set_xlabel("")

        # ---- Right: individual flies (Magnet=y) ----------------------------
        sub = df[df["Magnet"] == "y"].copy()
        n_flies = sub["fly"].nunique()

        _spaghetti_panel(
            ax=ax_ind,
            sub=sub,
            x_col="bin_center_min",
            y_col=y_col,
            focus_hours=focus_hours,
            bin_size_min=bin_size_min,
            ylabel=ylabel,
            title=f"{metric_titles[row_idx]} — Magnet=y individual flies (n={n_flies})",
            color="#3953A4",
        )
        ax_ind.get_legend().remove()  # legend already shown, keep compact
        if row_idx == n_rows - 1:
            ax_ind.set_xlabel("Time (min)", fontsize=10)
        else:
            ax_ind.set_xlabel("")
        ax_ind.set_title(f"{metric_titles[row_idx]} — individual flies (Magnet=y)", fontsize=10)

    # Global legend from first group-comparison panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(groups) + 1,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )

    # Column headers
    axes[0, 0].annotate(
        "Both groups (mean ± 95% CI)",
        xy=(0.5, 1.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].annotate(
        "Magnet=y only (individual flies)",
        xy=(0.5, 1.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "MagnetBlock — All engagement metrics summary",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# IEI · Ball displacement · Approach speed · First-push alignment
# ---------------------------------------------------------------------------


def compute_iei_per_fly(
    df: pd.DataFrame,
    bin_size_min: float = 5.0,
) -> pd.DataFrame:
    """
    Compute mean inter-event interval (s) per fly per time bin.
    IEI = time from one event onset to the next within the same fly.
    Bins with fewer than 2 consecutive events for a fly → NaN.
    """
    event_rows = df[df["interaction_event"].notna()].copy()
    event_onsets = (
        event_rows.groupby(["fly", "interaction_event"])
        .agg(onset_time_s=("time", "min"), Magnet=("Magnet", "first"))
        .reset_index()
        .sort_values(["fly", "onset_time_s"])
    )
    event_onsets["iei_s"] = event_onsets.groupby("fly")["onset_time_s"].diff()
    event_onsets["onset_time_min"] = event_onsets["onset_time_s"] / 60.0

    time_max_min = df["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")
    all_flies = df["fly"].unique()

    rows = []
    for fly in all_flies:
        magnet = fly_meta.loc[fly, "Magnet"]
        fly_ieis = event_onsets[(event_onsets["fly"] == fly) & event_onsets["iei_s"].notna()]
        for i in range(n_bins):
            b_s, b_e = bin_edges[i], bin_edges[i + 1]
            mask = (fly_ieis["onset_time_min"] >= b_s) & (fly_ieis["onset_time_min"] < b_e)
            vals = fly_ieis[mask]["iei_s"]
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_s,
                    bin_center_min=float(bin_centers[i]),
                    mean_iei_s=vals.mean() if len(vals) else np.nan,
                    n_intervals=len(vals),
                )
            )
    out = pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)
    valid = out["mean_iei_s"].dropna()
    print(f"\nIEI: overall mean = {valid.mean():.1f} s, median = {valid.median():.1f} s")
    return out


def _generic_group_plot(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    between_results: pd.DataFrame,
    trend_h1: dict,
    trend_h2: dict,
    focus_hours: float,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
    dropna: bool = True,
) -> None:
    """Generic group-comparison lineplot for any per-fly per-bin metric."""
    groups = sorted(df["Magnet"].unique())
    n_per_group = {g: df[df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}
    palette_labels = {label_map[g]: palette[g] for g in groups}
    plot_df = df.dropna(subset=[y_col]) if dropna else df.copy()
    plot_df = plot_df.copy()
    plot_df["Group"] = plot_df["Magnet"].map(label_map)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(
        data=plot_df,
        x="bin_center_min",
        y=y_col,
        hue="Group",
        palette=palette_labels,
        errorbar=("ci", 95),
        ax=ax,
    )
    y_bot, y_top = ax.get_ylim()
    y_rng = y_top - y_bot
    ax.set_ylim(y_bot, y_top + 0.12 * y_rng)
    y_star = y_top + 0.04 * y_rng

    for _, row in between_results.iterrows():
        bx = row["bin_center_min"]
        p = row["p_value"]
        if pd.isna(p):
            continue
        stars = sig_stars(p)
        if stars != "ns":
            ax.text(bx, y_star, stars, ha="center", va="bottom", fontsize=11, color="red", fontweight="bold")

    ax.axvline(
        focus_hours * 60, color="black", linestyle="--", lw=1.5, alpha=0.7, label=f"{int(focus_hours)}h boundary"
    )

    annotation_y = 0.04
    for trend_dict, phase_label in [(trend_h1, "h1"), (trend_h2, "h2")]:
        for group, res in trend_dict.items():
            g_label = label_map.get(group, group)
            p_str = format_p_value(res["p_value"], n_permutations)
            direction = "↓" if res["mean_slope"] < 0 else "↑"
            ax.annotate(
                f"[{phase_label}] {g_label}: slope={res['mean_slope']:.4f}{direction}, {p_str}",
                xy=(0.02, annotation_y),
                xycoords="axes fraction",
                fontsize=7,
                color=palette.get(group, "black"),
            )
            annotation_y += 0.07

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _generic_individual_plot(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    focus_hours: float,
    bin_size_min: float,
    output_path: Path,
    dropna: bool = True,
) -> None:
    """Generic Magnet=y spaghetti plot for any per-fly per-bin metric."""
    sub = df[df["Magnet"] == "y"].copy()
    if dropna:
        sub = sub.dropna(subset=[y_col])
    fig, ax = plt.subplots(figsize=(14, 5))
    _spaghetti_panel(
        ax=ax,
        sub=sub,
        x_col="bin_center_min",
        y_col=y_col,
        focus_hours=focus_hours,
        bin_size_min=bin_size_min,
        ylabel=ylabel,
        title=title,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9, loc="upper right")
    fig.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_boxplot_per_bin(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    focus_hours: float,
    output_path: Path,
    dropna: bool = True,
) -> None:
    """Two-panel figure (Magnet=y | Magnet=n) with per-bin box-and-whisker plots
    and jittered individual fly data points overlaid on top.

    Parameters
    ----------
    df : DataFrame with columns ``bin_center_min``, ``Magnet``, ``fly``, *y_col*.
    y_col : column to plot on the y-axis.
    ylabel : y-axis label.
    title : figure suptitle.
    focus_hours : draws a dashed boundary at this many hours.
    output_path : saved as ``.pdf`` and ``.png`` (extension is replaced).
    dropna : drop NaN rows for *y_col* before plotting.
    """
    if dropna:
        df = df.dropna(subset=[y_col])
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    magnet_vals = ["y", "n"]
    group_labels = {"y": "Magnet=y (active)", "n": "Magnet=n (sham)"}
    colors = {"y": "#3953A4", "n": "#faa41a"}

    for ax, mag in zip(axes, magnet_vals):
        sub = df[df["Magnet"] == mag].copy()
        bins = sorted(sub["bin_center_min"].unique())
        n_bins = len(bins)
        positions = list(range(n_bins))
        bin_data = [sub.loc[sub["bin_center_min"] == b, y_col].values for b in bins]

        # Speed-style boxplot: unfilled boxes with black outlines
        ax.boxplot(
            bin_data,
            positions=positions,
            patch_artist=True,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
            medianprops=dict(color="black", lw=1.5),
            whiskerprops=dict(color="black", alpha=1.0, linewidth=1.5),
            capprops=dict(color="black", alpha=1.0, linewidth=1.5),
            flierprops=dict(marker="", linestyle="none"),
            widths=0.55,
        )

        # Jittered scatter
        for pos, vals in zip(positions, bin_data):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.28, 0.28, size=len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                s=20,
                alpha=0.5,
                color=colors[mag],
                zorder=3,
                linewidths=0,
            )

        # x-axis ticks — show every ~10th bin to avoid crowding
        tick_step = max(1, n_bins // 10)
        tick_positions = positions[::tick_step]
        tick_labels = [f"{bins[i]:.0f}" for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

        # Focus-hours dashed boundary (convert minutes → bin index)
        focus_min = focus_hours * 60
        focus_idx = next((i for i, b in enumerate(bins) if b >= focus_min), None)
        if focus_idx is not None and focus_idx > 0:
            ax.axvline(
                focus_idx - 0.5,
                color="black",
                linestyle="--",
                lw=1.5,
                alpha=0.7,
                label=f"{int(focus_hours)}h boundary",
            )
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("Time (min)", fontsize=11)
        if mag == "y":
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(group_labels[mag], fontsize=12)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    for suffix in [".pdf", ".png", ".svg"]:
        p = Path(output_path).with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def compute_ball_displacement_per_event(
    df: pd.DataFrame,
    fps: float = FPS,
) -> pd.DataFrame:
    """
    Per interaction event compute:
      - ``displacement_px``     : Euclidean distance from ball start→end position
      - ``max_displacement_px`` : max distance the ball reached from its start
                                  position at any point during the event
      - ``onset_time_s/min``, ``duration_s``

    Net displacement (start→end) detects sustained pushes; max displacement also
    catches brief deflections that the ball recovered from.
    """
    event_rows = df[df["interaction_event"].notna()].copy()
    event_rows = event_rows.sort_values(["fly", "interaction_event", "time"])

    agg = (
        event_rows.groupby(["fly", "interaction_event"])
        .agg(
            onset_time_s=("time", "first"),
            ball_x_start=("x_ball_0", "first"),
            ball_y_start=("y_ball_0", "first"),
            ball_x_end=("x_ball_0", "last"),
            ball_y_end=("y_ball_0", "last"),
            duration_frames=("time", "count"),
            Magnet=("Magnet", "first"),
        )
        .reset_index()
    )
    agg["displacement_px"] = np.sqrt(
        (agg["ball_x_end"] - agg["ball_x_start"]) ** 2 + (agg["ball_y_end"] - agg["ball_y_start"]) ** 2
    )
    agg["duration_s"] = agg["duration_frames"] / fps
    agg["onset_time_min"] = agg["onset_time_s"] / 60.0

    # Max displacement: per-frame distance from event start
    event_rows = event_rows.merge(
        agg[["fly", "interaction_event", "ball_x_start", "ball_y_start"]],
        on=["fly", "interaction_event"],
        how="left",
    )
    event_rows["dist_from_start"] = np.sqrt(
        (event_rows["x_ball_0"] - event_rows["ball_x_start"]) ** 2
        + (event_rows["y_ball_0"] - event_rows["ball_y_start"]) ** 2
    )
    max_disp = (
        event_rows.groupby(["fly", "interaction_event"])["dist_from_start"]
        .max()
        .reset_index()
        .rename(columns={"dist_from_start": "max_displacement_px"})
    )
    result = agg.merge(max_disp, on=["fly", "interaction_event"])

    n_total = len(result)
    n_over_5 = (result["max_displacement_px"] > 5).sum()
    print(f"\nBall displacement: {n_total} events")
    print(f"  Mean net displacement:   {result['displacement_px'].mean():.2f} px")
    print(f"  Mean max displacement:   {result['max_displacement_px'].mean():.2f} px")
    print(f"  Events >5 px (max disp): {n_over_5} ({100 * n_over_5 / n_total:.1f}%)")

    return result[
        [
            "fly",
            "Magnet",
            "interaction_event",
            "onset_time_s",
            "onset_time_min",
            "duration_s",
            "displacement_px",
            "max_displacement_px",
        ]
    ].reset_index(drop=True)


def bin_displacement_per_fly(
    event_disp_df: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = 5.0,
    metric: str = "max_displacement_px",
) -> pd.DataFrame:
    """Average per-event ball displacement metric into time bins per fly."""
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")
    all_flies = df_full["fly"].unique()

    rows = []
    for fly in all_flies:
        magnet = fly_meta.loc[fly, "Magnet"]
        fly_ev = event_disp_df[event_disp_df["fly"] == fly]
        for i in range(n_bins):
            b_s, b_e = bin_edges[i], bin_edges[i + 1]
            mask = (fly_ev["onset_time_min"] >= b_s) & (fly_ev["onset_time_min"] < b_e)
            vals = fly_ev[mask][metric]
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_s,
                    bin_center_min=float(bin_centers[i]),
                    mean_displacement_px=vals.mean() if len(vals) else np.nan,
                )
            )
    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


def compute_approach_speed_per_event(
    df: pd.DataFrame,
    fps: float = FPS,
    pre_window_s: float = 2.0,
) -> pd.DataFrame:
    """
    For each interaction event, compute mean fly approach speed (px/s) in the
    ``pre_window_s`` seconds immediately before the event onset.

    Speed is the instantaneous Euclidean displacement between consecutive
    frames multiplied by FPS.  Returns an event-level DataFrame.
    """
    wdf = df[["fly", "time", "x_fly_0", "y_fly_0", "Magnet", "interaction_event"]].sort_values(["fly", "time"]).copy()
    grp = wdf.groupby("fly", sort=False)
    dx = grp["x_fly_0"].diff()
    dy = grp["y_fly_0"].diff()
    wdf["fly_speed_px_s"] = np.sqrt(dx**2 + dy**2) * fps

    event_rows = df[df["interaction_event"].notna()].copy()
    event_onsets = (
        event_rows.groupby(["fly", "interaction_event"])
        .agg(onset_time_s=("time", "min"), Magnet=("Magnet", "first"))
        .reset_index()
    )

    results = []
    for fly, fly_data in wdf.groupby("fly", sort=False):
        fly_data = fly_data.sort_values("time")
        times = fly_data["time"].values
        speeds = fly_data["fly_speed_px_s"].values
        fly_ev = event_onsets[event_onsets["fly"] == fly]
        for onset_t, magnet, ev_id in zip(fly_ev["onset_time_s"], fly_ev["Magnet"], fly_ev["interaction_event"]):
            idx_end = int(np.searchsorted(times, onset_t, side="left"))
            idx_start = int(np.searchsorted(times, onset_t - pre_window_s, side="left"))
            pre_speeds = speeds[idx_start:idx_end]
            mean_speed = float(np.nanmean(pre_speeds)) if len(pre_speeds) > 0 else np.nan
            results.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    interaction_event=ev_id,
                    onset_time_s=float(onset_t),
                    onset_time_min=float(onset_t) / 60.0,
                    approach_speed_px_s=mean_speed,
                )
            )

    result = pd.DataFrame(results)
    print(
        f"\nApproach speed ({pre_window_s}s pre-event window): {len(result)} events, "
        f"mean = {result['approach_speed_px_s'].mean():.2f} px/s"
    )
    return result


def bin_approach_speed_per_fly(
    vel_df: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = 5.0,
) -> pd.DataFrame:
    """Average approach speed (px/s) into time bins per fly."""
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")
    all_flies = df_full["fly"].unique()

    rows = []
    for fly in all_flies:
        magnet = fly_meta.loc[fly, "Magnet"]
        fly_ev = vel_df[vel_df["fly"] == fly]
        for i in range(n_bins):
            b_s, b_e = bin_edges[i], bin_edges[i + 1]
            mask = (fly_ev["onset_time_min"] >= b_s) & (fly_ev["onset_time_min"] < b_e)
            vals = fly_ev[mask]["approach_speed_px_s"]
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_s,
                    bin_center_min=float(bin_centers[i]),
                    mean_approach_speed_px_s=vals.mean() if len(vals) else np.nan,
                )
            )
    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


def compute_first_successful_push(
    event_disp_df: pd.DataFrame,
    focus_hours: float = 1.0,
    threshold_px: float = 5.0,
) -> pd.DataFrame:
    """
    For each fly, find the first event in hour 2 (onset_time_s > focus_hours*3600)
    where ``max_displacement_px > threshold_px``.

    Returns one row per qualifying fly with columns
    [fly, Magnet, t_first_push_s, t_first_push_min, first_push_event_id].
    """
    cutoff_s = focus_hours * 3600.0
    hour2 = event_disp_df[event_disp_df["onset_time_s"] > cutoff_s].copy()
    successful = hour2[hour2["max_displacement_px"] > threshold_px].copy()
    successful = successful.sort_values(["fly", "onset_time_s"])
    first_push = successful.groupby("fly").first().reset_index()
    first_push = first_push.rename(
        columns={
            "onset_time_s": "t_first_push_s",
            "onset_time_min": "t_first_push_min",
            "interaction_event": "first_push_event_id",
        }
    )
    n = len(first_push)
    print(f"\nFirst successful push (>{threshold_px}px, post-hour-{focus_hours}): {n} flies")
    if n:
        elapsed = (first_push["t_first_push_s"] - cutoff_s).mean() / 60
        print(f"  Mean time into hour 2 before first push: {elapsed:.1f} min")
    return first_push[["fly", "Magnet", "t_first_push_s", "t_first_push_min", "first_push_event_id"]]


def plot_aligned_to_first_push(
    rate_df: pd.DataFrame,
    dur_df: pd.DataFrame,
    prox_df: pd.DataFrame,
    iei_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    bin_size_min: float = 5.0,
    window_min: float = 60.0,
    focus_hours: float = 1.0,
    threshold_px: float = 5.0,
    output_path: Path = Path("aligned_to_first_push"),
) -> None:
    """
    Key discriminating analysis: align flies per Magnet condition to their
    first successful push in hour 2 and plot rate/duration/proximity/IEI as a
    function of time relative to that event.

    Layout: 4 rows (metrics) × 2 columns (Magnet=y | Magnet=n).

    Model-based prediction  : near-instantaneous jump at t=0 (first push).
    Model-free prediction   : gradual increase from start of hour 2,
                              independent of when the push occurred.

    x-axis : minutes relative to first successful push (negative = before push)
    """
    palette = make_palette(["y", "n"])
    groups = [("y", "Magnet=y (blocked)", palette["y"]), ("n", "Magnet=n (control)", palette["n"])]

    metric_specs = [
        (rate_df, "rate_per_min", "Rate (events/min)"),
        (dur_df, "mean_duration_s", "Duration (s)"),
        (prox_df, "mean_dist_mm", "Fly-ball distance (mm)"),
        (iei_df, "mean_iei_s", "Inter-event interval (s)"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.45, wspace=0.30)

    for col_idx, (magnet_val, group_label, color) in enumerate(groups):
        push_sub = first_push_df[first_push_df["Magnet"] == magnet_val]
        flies = set(push_sub["fly"])
        n_flies = len(flies)

        for row_idx, (metric_df, y_col, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]

            sub = metric_df[metric_df["fly"].isin(flies)].dropna(subset=[y_col]).copy()
            sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
            # Compute relative time, then snap to the nearest bin_size_min
            # multiple so all flies share the same relative-time grid before
            # averaging (avoids spiky means from per-fly float offsets).
            sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
            sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
            aligned = sub[sub["rel_bin"].abs() <= window_min]

            if not aligned.empty:
                n_per_bin = aligned.groupby("rel_bin")["fly"].nunique()
                n_label = int(n_per_bin.max()) if not n_per_bin.empty else n_flies
                # Bootstrap 95% CI per relative bin (1000 resamples)
                bs_rng = np.random.default_rng(42)
                bins_sorted = np.sort(aligned["rel_bin"].unique())
                means, ci_lo, ci_hi = [], [], []
                for rb in bins_sorted:
                    vals = aligned.loc[aligned["rel_bin"] == rb, y_col].values
                    if len(vals) == 0:
                        means.append(np.nan)
                        ci_lo.append(np.nan)
                        ci_hi.append(np.nan)
                        continue
                    boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
                    means.append(float(np.mean(vals)))
                    ci_lo.append(float(np.percentile(boot, 2.5)))
                    ci_hi.append(float(np.percentile(boot, 97.5)))
                means = np.array(means)
                ci_lo = np.array(ci_lo)
                ci_hi = np.array(ci_hi)
                ax.fill_between(
                    bins_sorted,
                    ci_lo,
                    ci_hi,
                    color=color,
                    alpha=0.25,
                    zorder=5,
                )
                ax.plot(
                    bins_sorted,
                    means,
                    color=color,
                    lw=2.5,
                    zorder=10,
                    label=f"Mean ± 95% CI (n≤{n_label})",
                )

            ax.axvline(0, color="red", linestyle="--", lw=2, alpha=0.9, label="First successful push")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{ylabel}\n{group_label} (n={n_flies})", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            if row_idx == 3:
                ax.set_xlabel("Time relative to first push (min)", fontsize=10)
            else:
                ax.set_xlabel("")

    # Column headers
    axes[0, 0].annotate(
        "Magnet=y (blocked)",
        xy=(0.5, 1.22),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=palette["y"],
    )
    axes[0, 1].annotate(
        "Magnet=n (control)",
        xy=(0.5, 1.22),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=palette["n"],
    )

    fig.suptitle(
        f"Metrics aligned to first successful push in hour 2\n"
        f"(window ±{int(window_min)} min; threshold >{threshold_px:.0f} px ball displacement)",
        fontsize=12,
        fontweight="bold",
    )
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_interaction_rate_aligned_to_first_push(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    bin_size_min: float = 5.0,
    window_min: float = 10.0,
    gate_open_min: float | None = 60.0,
    gate_close_min: float | None = 120.0,
    output_path: Path = Path("aligned_to_first_push_5px"),
) -> None:
    """Two-panel plot: interaction rate aligned to first successful push for each group.

    This is a rate-only version intended for figure compositing.
    """
    if gate_open_min is None or gate_close_min is None:
        raise ValueError(
            "plot_interaction_rate_aligned_to_first_push requires gate_open_min and gate_close_min; "
            "pass explicit second-hour bounds (e.g., 60 and 120)."
        )

    # Enforce fixed visualization window for this analysis, regardless of caller input.
    window_min_effective = 10.0
    if not np.isclose(float(window_min), window_min_effective):
        print(
            "  [aligned-rate] overriding requested window_min="
            f"{window_min} with enforced ±{window_min_effective:.0f} min"
        )

    palette = make_palette(["y", "n"])
    groups = [
        ("n", "No access to ball", palette["n"]),
        ("y", "Access to immobile ball", palette["y"]),
    ]

    # Keep final canvas size matched to the compact two-panel pre/post figure.
    fig_w_mm = ((2.0 * 55.0) + 18.0) * 1.3
    fig_h_mm = (73.0 + 16.0) * 1.3
    fig, axes = plt.subplots(1, 2, figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), sharey=True)
    fig.subplots_adjust(wspace=0.25, top=0.82, bottom=0.22)

    gate_text = (
        f", gate={gate_open_min:.0f}-{gate_close_min:.0f} min"
        if gate_open_min is not None and gate_close_min is not None
        else ""
    )
    fig.suptitle(
        f"First-push aligned interaction rate (window=±{window_min_effective:.0f} min{gate_text})",
        fontsize=11,
        y=0.98,
    )

    metadata_rows = []

    for col_idx, (magnet_val, group_label, color) in enumerate(groups):
        ax = axes[col_idx]
        push_sub = first_push_df[first_push_df["Magnet"] == magnet_val]
        flies = set(push_sub["fly"])
        n_flies = len(flies)

        sub = rate_df[rate_df["fly"].isin(flies)].dropna(subset=["rate_per_min"]).copy()
        if gate_open_min is not None:
            sub = sub[sub["bin_center_min"] >= gate_open_min].copy()
        if gate_close_min is not None:
            sub = sub[sub["bin_center_min"] <= gate_close_min].copy()
        sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
        sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
        sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
        aligned = sub[sub["rel_bin"].abs() <= window_min_effective]
        if aligned.empty:
            metadata_rows.append(
                {
                    "group": magnet_val,
                    "n_flies": n_flies,
                    "rel_bin_min": np.nan,
                    "rel_bin_max": np.nan,
                    "n_rel_bins": 0,
                }
            )
        else:
            metadata_rows.append(
                {
                    "group": magnet_val,
                    "n_flies": n_flies,
                    "rel_bin_min": float(aligned["rel_bin"].min()),
                    "rel_bin_max": float(aligned["rel_bin"].max()),
                    "n_rel_bins": int(aligned["rel_bin"].nunique()),
                }
            )

        if not aligned.empty:
            bs_rng = np.random.default_rng(42)
            bins_sorted = np.sort(aligned["rel_bin"].unique())
            means, ci_lo, ci_hi = [], [], []
            for rb in bins_sorted:
                vals = aligned.loc[aligned["rel_bin"] == rb, "rate_per_min"].values
                if len(vals) == 0:
                    means.append(np.nan)
                    ci_lo.append(np.nan)
                    ci_hi.append(np.nan)
                    continue
                boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
                means.append(float(np.mean(vals)))
                ci_lo.append(float(np.percentile(boot, 2.5)))
                ci_hi.append(float(np.percentile(boot, 97.5)))

            means = np.array(means)
            ci_lo = np.array(ci_lo)
            ci_hi = np.array(ci_hi)
            ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25, zorder=4)
            ax.plot(bins_sorted, means, color=color, lw=2.5, zorder=10)

        ax.axvline(0, color="red", linestyle="--", lw=2, alpha=0.9)
        ax.set_xlim(-window_min_effective, window_min_effective)
        if window_min_effective <= 10:
            ax.set_xticks(np.arange(-window_min_effective, window_min_effective + 0.1, 5.0))
        ax.set_xlabel("Time relative to first push (min)", fontsize=10)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Keep the group label only in the colored header (with sample size).
        ax.annotate(
            f"{group_label} (n={n_flies})",
            xy=(0.5, 1.06),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    axes[0].set_ylabel("Interaction rate (events / min)", fontsize=10)

    for suffix in [".pdf", ".png", ".svg"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")

    # Sidecar metadata to verify exactly what was plotted.
    meta_path = output_path.with_name(output_path.name + "_metadata.txt")
    with open(meta_path, "w") as f:
        f.write("first_push_aligned_interaction_rate_metadata\n")
        f.write(f"window_min={window_min_effective}\n")
        f.write(f"gate_open_min={gate_open_min}\n")
        f.write(f"gate_close_min={gate_close_min}\n")
        f.write("xlim_enforced=[-{0},{0}]\n".format(window_min_effective))
        for row in metadata_rows:
            f.write(
                "group={group}, n_flies={n_flies}, rel_bin_min={rel_bin_min}, "
                "rel_bin_max={rel_bin_max}, n_rel_bins={n_rel_bins}\n".format(**row)
            )
    print(f"  Saved: {meta_path}")
    plt.close(fig)


def plot_ball_position_from_first_push(
    df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    window_min: float = 60.0,
    downsample_s: float = 1.0,
    n_bootstrap: int = 1000,
    output_path: Path = Path("ball_position_from_first_push"),
) -> None:
    """
    Align each fly to its first successful push (t=0) and plot the mean
    relative ball displacement trajectory from t=0 onward, grouped by Magnet.

    Relative ball displacement is computed per fly as Euclidean distance (px)
    from the ball position at t=0 (the push-alignment frame):

      rel_disp_px(t) = || ball_xy(t) - ball_xy(t0) ||

    Curves are downsampled to one datapoint per second per fly and displayed as
    group mean ± bootstrapped 95% CI.
    """
    required = ["fly", "Magnet", "time", "x_ball_0", "y_ball_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ball-position alignment: {missing}")

    if first_push_df.empty:
        print("  No flies with first-push timestamps; skipping aligned ball-position plot.")
        return

    window_s = float(window_min) * 60.0
    rows = []

    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        magnet = push_row["Magnet"]
        t0 = float(push_row["t_first_push_s"])

        fly_df = df[df["fly"] == fly].sort_values("time").copy()
        if fly_df.empty:
            continue

        idx0 = (fly_df["time"] - t0).abs().idxmin()
        x0 = float(fly_df.loc[idx0, "x_ball_0"])
        y0 = float(fly_df.loc[idx0, "y_ball_0"])

        aligned = fly_df[(fly_df["time"] >= t0) & (fly_df["time"] <= t0 + window_s)].copy()
        if aligned.empty:
            continue

        aligned["rel_time_s"] = aligned["time"] - t0
        aligned["rel_sec"] = np.floor(aligned["rel_time_s"] / downsample_s).astype(int)
        aligned["rel_disp_px"] = np.sqrt((aligned["x_ball_0"] - x0) ** 2 + (aligned["y_ball_0"] - y0) ** 2)

        per_sec = (
            aligned.groupby("rel_sec", as_index=False)["rel_disp_px"]
            .mean()
            .rename(columns={"rel_disp_px": "rel_disp_px_mean"})
        )
        per_sec["fly"] = fly
        per_sec["Magnet"] = magnet
        rows.append(per_sec)

    if not rows:
        print("  No aligned ball-position data available; skipping plot.")
        return

    aligned_df = pd.concat(rows, ignore_index=True)
    palette = make_palette(["y", "n"])
    group_specs = [
        ("y", "Magnet=y (blocked)"),
        ("n", "Magnet=n (control)"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bs_rng = np.random.default_rng(42)

    for magnet, label in group_specs:
        gdf = aligned_df[aligned_df["Magnet"] == magnet].copy()
        if gdf.empty:
            continue

        n_flies = gdf["fly"].nunique()
        secs = np.sort(gdf["rel_sec"].unique())

        means, ci_lo, ci_hi = [], [], []
        for sec in secs:
            vals = gdf[gdf["rel_sec"] == sec]["rel_disp_px_mean"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue

            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        x_min = secs / 60.0
        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        ax.fill_between(x_min, ci_lo, ci_hi, color=palette.get(magnet, "gray"), alpha=0.25)
        ax.plot(
            x_min,
            means,
            color=palette.get(magnet, "gray"),
            lw=2.5,
            label=f"{label} (n={n_flies}) — mean ± 95% CI",
        )

    ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First successful push")
    ax.set_xlim(left=0)
    ax.set_xlabel("Time from first successful push (min)", fontsize=12)
    ax.set_ylabel("Relative ball displacement from t=0 (px)", fontsize=12)
    ax.set_title("Ball position trajectory from first push onward", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def compute_first_push_magnitude(
    event_disp_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one-row-per-fly table containing first-push magnitude metrics.

    Magnitude is taken from the first successful push event identified in
    ``first_push_df`` and includes both peak and net displacement.
    """
    if first_push_df.empty:
        return pd.DataFrame(
            columns=["fly", "Magnet", "first_push_event_id", "first_push_max_disp_px", "first_push_net_disp_px"]
        )

    merged = first_push_df.merge(
        event_disp_df[["fly", "interaction_event", "max_displacement_px", "displacement_px"]],
        left_on=["fly", "first_push_event_id"],
        right_on=["fly", "interaction_event"],
        how="left",
    )

    out = merged.rename(
        columns={
            "max_displacement_px": "first_push_max_disp_px",
            "displacement_px": "first_push_net_disp_px",
        }
    )[
        [
            "fly",
            "Magnet",
            "first_push_event_id",
            "first_push_max_disp_px",
            "first_push_net_disp_px",
        ]
    ].copy()

    miss = out["first_push_max_disp_px"].isna().sum()
    if miss:
        print(f"  Warning: {miss} first-push events missing displacement info after merge.")
    return out


def test_first_push_magnitude(
    first_push_mag_df: pd.DataFrame,
    n_permutations: int = 10000,
    metric_col: str = "first_push_max_disp_px",
) -> dict:
    """Two-sample permutation test for first-push magnitude (Magnet=y vs n)."""
    rng = np.random.default_rng(0)
    y_vals = first_push_mag_df[first_push_mag_df["Magnet"] == "y"][metric_col].dropna().values
    n_vals = first_push_mag_df[first_push_mag_df["Magnet"] == "n"][metric_col].dropna().values

    result = {
        "metric": metric_col,
        "n_y": int(len(y_vals)),
        "n_n": int(len(n_vals)),
        "mean_y": float(np.mean(y_vals)) if len(y_vals) else np.nan,
        "mean_n": float(np.mean(n_vals)) if len(n_vals) else np.nan,
        "obs_diff": np.nan,
        "p_value": np.nan,
    }

    if len(y_vals) < 3 or len(n_vals) < 3:
        print("  Too few flies per group for first-push magnitude permutation test (need >=3 each).")
        return result

    obs_diff = float(np.mean(y_vals) - np.mean(n_vals))
    combined = np.concatenate([y_vals, n_vals])
    ny = len(y_vals)
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diffs[i] = np.mean(perm[:ny]) - np.mean(perm[ny:])
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

    result["obs_diff"] = obs_diff
    result["p_value"] = p_value
    direction = "↑" if obs_diff >= 0 else "↓"
    print(
        "  First-push magnitude between groups: "
        f"Δ(y-n)={obs_diff:+.3f} px {direction}, p={format_p_value(p_value, n_permutations)}"
    )
    return result


def plot_first_push_magnitude(
    first_push_mag_df: pd.DataFrame,
    stats: dict,
    metric_col: str = "first_push_max_disp_px",
    ylabel: str = "First push magnitude (max displacement, px)",
    n_permutations: int = 10000,
    output_path: Path = Path("first_push_magnitude"),
) -> None:
    """Boxplot + jitter for first-push magnitude by Magnet condition."""
    palette = make_palette(["y", "n"])
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    group_meta = [("y", "Magnet=y"), ("n", "Magnet=n")]

    vals_by_group = []
    labels = []
    for grp, glabel in group_meta:
        vals = first_push_mag_df[first_push_mag_df["Magnet"] == grp][metric_col].dropna().values
        vals_by_group.append(vals)
        labels.append(f"{glabel}\n(n={len(vals)})")

    bp = ax.boxplot(
        vals_by_group,
        positions=[0, 1],
        patch_artist=True,
        widths=0.45,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(color="black", alpha=1.0, linewidth=1.5),
        capprops=dict(color="black", alpha=1.0, linewidth=1.5),
        flierprops=dict(marker="", linestyle="none"),
    )
    _ = bp

    for pos, (grp, _) in enumerate(group_meta):
        vals = vals_by_group[pos]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            s=26,
            color=palette[grp],
            alpha=0.75,
            zorder=3,
            edgecolors="none",
        )

    y_all = (
        np.concatenate([v for v in vals_by_group if len(v) > 0])
        if any(len(v) > 0 for v in vals_by_group)
        else np.array([0.0])
    )
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_rng = y_max - y_min if y_max != y_min else 1.0
    bar_y = y_max + 0.08 * y_rng
    txt_y = y_max + 0.11 * y_rng
    ax.set_ylim(y_min - 0.05 * y_rng, y_max + 0.25 * y_rng)

    p = stats.get("p_value", np.nan)
    if not np.isnan(p):
        ax.plot([0, 1], [bar_y, bar_y], color="black", lw=1.5)
        stars = sig_stars(p)
        p_str = format_p_value(p, n_permutations)
        ax.text(0.5, txt_y, f"{stars}\np={p_str}", ha="center", va="bottom", fontsize=10, color="red")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("First successful push magnitude by condition", fontsize=12, fontweight="bold")

    fig.tight_layout()
    for suffix in [".pdf", ".png", ".svg"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p_out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-push rate change analysis
# ---------------------------------------------------------------------------


def compute_post_push_rate_change(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    focus_hours: float = 1.0,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
) -> pd.DataFrame:
    """
    Per fly: compute mean interaction rate in a pre-push and post-push window,
    aligned to each fly's first successful push.

    Pre-window  : rel_min in [-pre_window_min[0], -pre_window_min[1]]
                  (avoids t=0 artifact on both sides)
    Post-window : rel_min in [+post_window_min[0], +post_window_min[1]]

    For Magnet=y flies the pre-window is additionally clipped so that only
    bins that fall *after* the barrier removal (bin_center_min >= focus_hours*60)
    are counted, removing hour-1 baseline probing contamination.

    Returns one row per qualifying fly with columns:
    fly, Magnet, t_first_push_min, pre_mean, post_mean, delta, n_pre_bins, n_post_bins
    """
    cutoff_min = focus_hours * 60.0

    # Infer bin_size_min from the spacing between bin centers
    all_bins = np.sort(rate_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 5.0

    rows = []
    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        magnet = push_row["Magnet"]
        t_push = push_row["t_first_push_min"]

        fly_data = rate_df[rate_df["fly"] == fly].dropna(subset=["rate_per_min"]).copy()

        # Snap relative times to the bin grid
        fly_data["rel_bin"] = ((fly_data["bin_center_min"] - t_push) / bin_size_min).round() * bin_size_min

        # Pre-window: e.g. rel_bin in [-10, -1]
        pre_far = -pre_window_min[0]
        pre_near = -pre_window_min[1]
        pre_data = fly_data[(fly_data["rel_bin"] >= pre_far) & (fly_data["rel_bin"] <= pre_near)]
        # For blocked flies restrict pre-window to hour 2 only
        if magnet == "y":
            pre_data = pre_data[pre_data["bin_center_min"] >= cutoff_min]

        # Post-window: e.g. rel_bin in [+1, +10]
        post_near = post_window_min[0]
        post_far = post_window_min[1]
        post_data = fly_data[(fly_data["rel_bin"] >= post_near) & (fly_data["rel_bin"] <= post_far)]

        if pre_data.empty or post_data.empty:
            continue  # insufficient data on one side — skip fly

        pre_mean = float(pre_data["rate_per_min"].mean())
        post_mean = float(post_data["rate_per_min"].mean())
        rows.append(
            dict(
                fly=fly,
                Magnet=magnet,
                t_first_push_min=t_push,
                pre_mean=pre_mean,
                post_mean=post_mean,
                delta=post_mean - pre_mean,
                n_pre_bins=len(pre_data),
                n_post_bins=len(post_data),
            )
        )

    result = pd.DataFrame(rows)
    for grp, sub in result.groupby("Magnet"):
        print(f"  Post-push change ({grp}): n={len(sub)}, " f"mean delta={sub['delta'].mean():.3f} events/min")
    return result


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng) -> float:
    """Two-sided sign-permutation test of H0: mean == 0."""
    obs = np.abs(np.mean(scores))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(scores))
        if np.abs(np.mean(scores * signs)) >= obs:
            count += 1
    return count / n_permutations


def test_post_push_change(
    change_df: pd.DataFrame,
    n_permutations: int = 10000,
) -> dict:
    """
    1. One-sample sign-permutation test per group: is delta != 0?
    2. Two-sample permutation test between groups: is delta_y != delta_n?

    Returns a dict with keys 'y', 'n', 'between' each containing p_value.
    """
    rng = np.random.default_rng(0)
    results = {}

    for group in ["y", "n"]:
        scores = change_df[change_df["Magnet"] == group]["delta"].values
        if len(scores) < 3:
            results[group] = {"p_value": np.nan, "obs_mean": np.nan, "n": len(scores)}
            continue
        obs_mean = float(np.mean(scores))
        p = _one_sample_sign_perm(scores, n_permutations, rng)
        results[group] = {"p_value": p, "obs_mean": obs_mean, "n": len(scores)}
        direction = "↑" if obs_mean >= 0 else "↓"
        print(f"  [{group}] mean delta = {obs_mean:+.4f}  " f"{direction}  p = {format_p_value(p, n_permutations)}")

    y_scores = change_df[change_df["Magnet"] == "y"]["delta"].values
    n_scores = change_df[change_df["Magnet"] == "n"]["delta"].values
    if len(y_scores) >= 3 and len(n_scores) >= 3:
        obs_diff = float(np.mean(y_scores) - np.mean(n_scores))
        combined = np.concatenate([y_scores, n_scores])
        ny = len(y_scores)
        perm_diffs = np.empty(n_permutations)
        for j in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diffs[j] = np.mean(perm[:ny]) - np.mean(perm[ny:])
        p_between = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
        results["between"] = {"p_value": p_between, "obs_diff": obs_diff}
        print(f"  [between] diff = {obs_diff:+.4f}  " f"p = {format_p_value(p_between, n_permutations)}")
    else:
        results["between"] = {"p_value": np.nan, "obs_diff": np.nan}

    return results


def plot_post_push_rate_change(
    change_df: pd.DataFrame,
    stats: dict,
    focus_hours: float = 1.0,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
    n_permutations: int = 10000,
    output_path: Path = Path("post_push_rate_change"),
) -> None:
    """
    Three-panel figure:
      Left  : Magnet=y — paired pre/post boxplot + jittered dots + linking lines
      Middle : Magnet=n — same
      Right  : Delta (post-pre) distribution, both groups, with between-group test
    """
    palette = make_palette(["y", "n"])
    rng = np.random.default_rng(42)

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    group_meta = [
        ("n", "No access to ball", palette["n"]),
        ("y", "Access to immobile ball", palette["y"]),
    ]

    # Compute shared y-limits across both Magnet groups for panels 0 & 1
    all_paired = change_df.dropna(subset=["pre_mean", "post_mean"])
    shared_y_min = min(all_paired["pre_mean"].min(), all_paired["post_mean"].min())
    shared_y_max = max(all_paired["pre_mean"].max(), all_paired["post_mean"].max())
    shared_y_rng = shared_y_max - shared_y_min
    # Reserve 30% headroom above the data for the significance annotation + suptitle gap
    shared_y_top = shared_y_max + 0.30 * shared_y_rng
    shared_y_bot = shared_y_min - 0.05 * shared_y_rng

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    # Leave room at top for suptitle so it never overlaps annotation
    fig.subplots_adjust(wspace=0.40, top=0.82)

    # ── Panels 0 & 1: paired pre / post per group ─────────────────────────
    for ax, (grp, glabel, color) in zip(axes[:2], group_meta):
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["pre_mean", "post_mean"])
        pre_vals = sub["pre_mean"].values
        post_vals = sub["post_mean"].values
        n = len(sub)

        # Box plots at x=0 and x=1
        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=[0, 1],
            patch_artist=True,
            widths=0.35,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black", alpha=1.0, linewidth=1.5),
            capprops=dict(color="black", alpha=1.0, linewidth=1.5),
            flierprops=dict(marker="", linestyle="none"),
        )
        _ = bp  # suppress unused warning

        # Jitter and lines
        jitter = rng.uniform(-0.07, 0.07, size=n)
        for i, (pv, sv, jit) in enumerate(zip(pre_vals, post_vals, jitter)):
            ax.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(0 + jitter, pre_vals, s=22, color=color, alpha=0.7, zorder=3, edgecolors="none")
        ax.scatter(1 + jitter, post_vals, s=22, color=color, alpha=0.7, zorder=3, edgecolors="none")

        # Significance annotation placed at a fixed shared height
        p = stats.get(grp, {}).get("p_value", np.nan)
        stars = sig_stars(p) if not np.isnan(p) else ""
        annot_y = shared_y_max + 0.12 * shared_y_rng
        ax.annotate(
            "",
            xy=(1, annot_y),
            xytext=(0, annot_y),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )
        star_color = "red" if (not np.isnan(p) and p < 0.05) else "gray"
        star_txt = stars if stars and stars != "ns" else f"p={p:.3f}" if not np.isnan(p) else "n.d."
        ax.text(
            0.5,
            annot_y + 0.02 * shared_y_rng,
            star_txt,
            ha="center",
            va="bottom",
            fontsize=12,
            color=star_color,
            fontweight="bold" if stars and stars != "ns" else "normal",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([pre_label, post_label], fontsize=10)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(shared_y_bot, shared_y_top)
        ax.set_ylabel("Interaction rate (events / min)", fontsize=11)
        ax.set_title(f"{glabel}\n(n={n})", fontsize=11)

    # ── Panel 2: delta distributions, both groups ─────────────────────────
    ax_d = axes[2]
    positions = {"n": 0, "y": 1}
    for grp, glabel, color in group_meta:
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["delta"])
        vals = sub["delta"].values
        pos = positions[grp]
        n = len(vals)
        ax_d.boxplot(
            [vals],
            positions=[pos],
            patch_artist=True,
            widths=0.35,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black", alpha=1.0, linewidth=1.5),
            capprops=dict(color="black", alpha=1.0, linewidth=1.5),
            flierprops=dict(marker="", linestyle="none"),
        )
        jitter = rng.uniform(-0.10, 0.10, size=n)
        ax_d.scatter(pos + jitter, vals, s=22, color=color, alpha=0.7, zorder=3, edgecolors="none")

    ax_d.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    # Between-group annotation — compute limits first so annotation never overflows
    delta_vals = change_df["delta"].dropna().values
    d_top = float(np.nanmax(delta_vals)) if len(delta_vals) else 1.0
    d_bot = float(np.nanmin(delta_vals)) if len(delta_vals) else -1.0
    d_rng = d_top - d_bot if d_top != d_bot else 1.0
    annot_y_d = d_top + 0.12 * d_rng
    # Reserve 30% headroom above annotation for suptitle clearance
    ax_d.set_ylim(d_bot - 0.05 * d_rng, d_top + 0.30 * d_rng)

    p_b = stats.get("between", {}).get("p_value", np.nan)
    if not np.isnan(p_b):
        ax_d.annotate(
            "",
            xy=(1, annot_y_d),
            xytext=(0, annot_y_d),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )
        stars_b = sig_stars(p_b)
        star_color = "red" if p_b < 0.05 else "gray"
        star_txt = stars_b if stars_b != "ns" else f"p={p_b:.3f}"
        ax_d.text(
            0.5,
            annot_y_d + 0.02 * d_rng,
            star_txt,
            ha="center",
            va="bottom",
            fontsize=12,
            color=star_color,
            fontweight="bold" if stars_b != "ns" else "normal",
        )

    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(
        [
            f"No access to ball\n(n={len(change_df[change_df['Magnet']=='n'])})",
            f"Access to immobile ball\n(n={len(change_df[change_df['Magnet']=='y'])})",
        ],
        fontsize=10,
    )
    ax_d.set_xlim(-0.6, 1.6)
    ax_d.set_ylabel("Δ rate (post − pre, events/min)", fontsize=11)
    ax_d.set_title("Change score\n(between-group test)", fontsize=11)

    pre_str = f"-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min"
    post_str = f"+{int(post_window_min[0])} to +{int(post_window_min[1])} min"
    fig.suptitle(
        f"Post-push interaction rate change (pre: {pre_str}, post: {post_str})\n"
        f"Magnet=y pre-window restricted to hour 2 (>={int(focus_hours*60)} min)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    for suffix in [".pdf", ".png", ".svg"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p_out}")
    plt.close(fig)


def plot_post_push_rate_change_two_panel(
    change_df: pd.DataFrame,
    stats: dict,
    focus_hours: float = 1.0,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
    n_permutations: int = 10000,
    subplot_width_mm: float = 55.0,
    subplot_height_mm: float = 73.0,
    outer_padding_width_mm: float = 18.0,
    outer_padding_height_mm: float = 16.0,
    output_path: Path = Path("post_push_rate_change_two_panel_small"),
) -> None:
    """Compact two-panel figure with only within-group pre/post changes.

    Panels:
      Left  : Magnet=y paired pre/post
      Right : Magnet=n paired pre/post
    """
    palette = make_palette(["y", "n"])
    rng = np.random.default_rng(42)

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    group_meta = [
        ("n", "No access to ball", palette["n"]),
        ("y", "Access to immobile ball", palette["y"]),
    ]

    all_paired = change_df.dropna(subset=["pre_mean", "post_mean"])
    shared_y_min = min(all_paired["pre_mean"].min(), all_paired["post_mean"].min())
    shared_y_max = max(all_paired["pre_mean"].max(), all_paired["post_mean"].max())
    shared_y_rng = shared_y_max - shared_y_min if shared_y_max != shared_y_min else 1.0
    shared_y_top = shared_y_max + 0.28 * shared_y_rng
    shared_y_bot = shared_y_min - 0.05 * shared_y_rng

    # Treat subplot_width/height as the intended plotting window size.
    # Add outer canvas padding so titles/labels do not crowd the compact figure,
    # then enlarge the overall canvas by 30%.
    fig_w_mm = ((2.0 * subplot_width_mm) + outer_padding_width_mm) * 1.3
    fig_h_mm = (subplot_height_mm + outer_padding_height_mm) * 1.3
    fig, axes = plt.subplots(1, 2, figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), sharey=True)
    fig.subplots_adjust(wspace=0.24, top=0.82, bottom=0.22)

    for ax, (grp, glabel, color) in zip(axes, group_meta):
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["pre_mean", "post_mean"])
        pre_vals = sub["pre_mean"].values
        post_vals = sub["post_mean"].values
        n = len(sub)

        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=[0, 1],
            patch_artist=True,
            widths=0.35,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black", alpha=1.0, linewidth=1.5),
            capprops=dict(color="black", alpha=1.0, linewidth=1.5),
            flierprops=dict(marker="", linestyle="none"),
        )
        _ = bp

        jitter = rng.uniform(-0.07, 0.07, size=n)
        for pv, sv, jit in zip(pre_vals, post_vals, jitter):
            ax.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(0 + jitter, pre_vals, s=22, color=color, alpha=0.7, zorder=3, edgecolors="none")
        ax.scatter(1 + jitter, post_vals, s=22, color=color, alpha=0.7, zorder=3, edgecolors="none")

        p = stats.get(grp, {}).get("p_value", np.nan)
        stars = sig_stars(p) if not np.isnan(p) else ""
        annot_y = shared_y_max + 0.11 * shared_y_rng
        ax.annotate(
            "",
            xy=(1, annot_y),
            xytext=(0, annot_y),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )

        star_color = "red" if (not np.isnan(p) and p < 0.05) else "gray"
        if np.isnan(p):
            star_txt = "n.d."
        elif stars != "ns":
            star_txt = stars
        else:
            star_txt = f"p={format_p_value(p, n_permutations)}"

        ax.text(
            0.5,
            annot_y + 0.018 * shared_y_rng,
            star_txt,
            ha="center",
            va="bottom",
            fontsize=12,
            color=star_color,
            fontweight="bold" if (stars and stars != "ns") else "normal",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([pre_label, post_label], fontsize=10)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(shared_y_bot, shared_y_top)
        ax.set_title(f"{glabel}\n(n={n})", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Interaction rate (events / min)", fontsize=11)
    axes[1].set_ylabel("")

    for suffix in [".pdf", ".png", ".svg"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p_out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time spent in chamber (proximity to start position)
# ---------------------------------------------------------------------------


def compute_time_in_chamber_per_fly(
    df: pd.DataFrame,
    fps: float = FPS,
    bin_size_min: float = 5.0,
    chamber_radius_px: float = 50.0,
) -> pd.DataFrame:
    """
    For each fly, compute the fraction of frames (and total seconds) per time
    bin where the fly is within ``chamber_radius_px`` pixels of *its own
    starting position* (position at the earliest recorded time point).

    "Start position" = mean fly position over the first 5 seconds of recording
    (averaged to reduce noise from tracking jitter at t=0).

    Parameters
    ----------
    df                : full dataset
    fps               : frames per second
    bin_size_min      : bin width in minutes
    chamber_radius_px : radius threshold in pixels (default 50)

    Returns
    -------
    DataFrame with columns [fly, Magnet, bin_idx, bin_start_min,
                             bin_center_min, fraction_in_chamber,
                             time_in_chamber_s, total_frames]
    """
    time_max_min = df["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")

    # Assign bin index once for the whole dataframe
    df = df.copy()
    df["time_min"] = df["time"] / 60.0
    df["bin_idx"] = pd.cut(df["time_min"], bins=bin_edges, labels=range(n_bins), include_lowest=True).astype("Int64")

    rows = []
    for fly, fly_data in df.groupby("fly", sort=False):
        magnet = fly_meta.loc[fly, "Magnet"]
        fly_data = fly_data.sort_values("time")

        # Start position: mean over first 5 s to reduce jitter
        init_frames = fly_data[fly_data["time"] <= fly_data["time"].min() + 5.0]
        x0 = init_frames["x_fly_0"].mean()
        y0 = init_frames["y_fly_0"].mean()

        # Distance from start position on every frame
        fly_data = fly_data.copy()
        fly_data["dist_from_start"] = np.sqrt((fly_data["x_fly_0"] - x0) ** 2 + (fly_data["y_fly_0"] - y0) ** 2)
        fly_data["in_chamber"] = fly_data["dist_from_start"] <= chamber_radius_px

        for i in range(n_bins):
            bin_frames = fly_data[fly_data["bin_idx"] == i]
            n_total = len(bin_frames)
            n_in = int(bin_frames["in_chamber"].sum())
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=float(bin_edges[i]),
                    bin_center_min=float(bin_centers[i]),
                    fraction_in_chamber=n_in / n_total if n_total > 0 else np.nan,
                    time_in_chamber_s=n_in / fps,
                    total_frames=n_total,
                )
            )

    result = pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)
    overall = result["fraction_in_chamber"].mean()
    print(
        f"\nTime in chamber (r={chamber_radius_px}px from start): "
        f"overall mean fraction = {overall:.3f} ({overall * 100:.1f}%)"
    )
    return result


# ---------------------------------------------------------------------------
# Event-rank analysis (x = sequential event number per fly)
# ---------------------------------------------------------------------------


def compute_metrics_by_event_rank(
    df: pd.DataFrame,
    fps: float = FPS,
    focus_hours: float = 1.0,
    event_disp_df: pd.DataFrame | None = None,
    vel_event_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each fly, rank interaction events in temporal order **within two
    phases** — before and after ``focus_hours`` — and compute per-event
    metrics:

    - ``duration_s``             : event length in seconds
    - ``mean_dist_mm``           : mean fly-ball distance *during* the event
    - ``inter_event_interval_s`` : time from the previous event onset to this
                                   event onset in seconds (NaN for rank 1
                                   within each phase)
    - ``displacement_px``        : net ball displacement start→end (optional)
    - ``max_displacement_px``    : peak ball displacement during event (optional)
    - ``approach_speed_px_s``    : mean fly speed in 2 s pre-event (optional)
    - ``phase``                  : ``"pre"`` (≤ focus_hours) or ``"post"``
    - ``event_rank``             : 1-indexed rank within the fly × phase

    Returns a DataFrame with one row per (fly, phase, event_rank).
    """
    if "fly_ball_dist_mm" not in df.columns:
        raise ValueError("Column 'fly_ball_dist_mm' not found. Run load_dataset() first.")

    event_rows = df[df["interaction_event"].notna()].copy()

    event_stats = (
        event_rows.groupby(["fly", "interaction_event"])
        .agg(
            onset_time_s=("time", "min"),
            duration_frames=("time", "count"),
            mean_dist_mm=("fly_ball_dist_mm", "mean"),
            Magnet=("Magnet", "first"),
        )
        .reset_index()
    )
    event_stats["duration_s"] = event_stats["duration_frames"] / fps

    # Label phase
    cutoff_s = focus_hours * 3600.0
    event_stats["phase"] = np.where(event_stats["onset_time_s"] <= cutoff_s, "pre", "post")

    # Rank events within each fly × phase by onset time (1-indexed)
    event_stats = event_stats.sort_values(["fly", "onset_time_s"])
    event_stats["event_rank"] = event_stats.groupby(["fly", "phase"]).cumcount() + 1

    # Inter-event interval within each fly × phase
    event_stats["inter_event_interval_s"] = event_stats.groupby(["fly", "phase"])["onset_time_s"].diff()

    n_flies = event_stats["fly"].nunique()
    for phase in ["pre", "post"]:
        sub = event_stats[event_stats["phase"] == phase]
        print(f"  Phase '{phase}': {len(sub)} events, " f"max rank = {sub['event_rank'].max()}")
    print(f"\nEvent-rank table: {len(event_stats)} events total, {n_flies} flies")

    # Optionally merge ball displacement
    if event_disp_df is not None:
        disp_cols = ["fly", "interaction_event", "displacement_px", "max_displacement_px"]
        available = [c for c in disp_cols if c in event_disp_df.columns]
        event_stats = event_stats.merge(event_disp_df[available], on=["fly", "interaction_event"], how="left")

    # Optionally merge approach speed
    if vel_event_df is not None:
        vel_cols = ["fly", "interaction_event", "approach_speed_px_s"]
        available_v = [c for c in vel_cols if c in vel_event_df.columns]
        event_stats = event_stats.merge(vel_event_df[available_v], on=["fly", "interaction_event"], how="left")

    keep_cols = [
        "fly",
        "Magnet",
        "phase",
        "event_rank",
        "onset_time_s",
        "duration_s",
        "mean_dist_mm",
        "inter_event_interval_s",
    ]
    for extra in ["displacement_px", "max_displacement_px", "approach_speed_px_s"]:
        if extra in event_stats.columns:
            keep_cols.append(extra)

    return event_stats[keep_cols].reset_index(drop=True)


def _plot_event_rank_phase(
    data: pd.DataFrame,
    phase: str,
    phase_label: str,
    max_rank: int,
    groups: list,
    palette_labels: dict,
    label_map: dict,
    output_path: Path,
) -> None:
    """Produce one event-rank figure (n_metrics rows × 2 cols) for a single phase."""
    sub_phase = data[data["phase"] == phase].copy()
    sub_phase["Group"] = sub_phase["Magnet"].map(label_map)

    metrics = [
        ("duration_s", "Duration (s)", "Interaction duration"),
        ("mean_dist_mm", "Fly-ball distance (mm)", "Fly-ball proximity"),
        ("inter_event_interval_s", "Inter-event interval (s)", "Inter-event interval"),
        ("displacement_px", "Ball displacement (px)", "Ball displacement (net)"),
        ("max_displacement_px", "Max ball displacement (px)", "Ball displacement (peak)"),
        ("approach_speed_px_s", "Approach speed (px/s)", "Fly approach speed"),
    ]
    # Only keep metrics whose column is actually present in the data
    metrics = [(col, lbl, ttl) for col, lbl, ttl in metrics if col in sub_phase.columns]
    n_rows = len(metrics)

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows), sharex="col")
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D
    fig.subplots_adjust(hspace=0.45, wspace=0.30)

    for row_idx, (y_col, ylabel, title) in enumerate(metrics):
        ax_grp = axes[row_idx, 0]
        ax_ind = axes[row_idx, 1]

        sub = sub_phase.dropna(subset=[y_col])

        # ---- Left: group mean ± CI ----------------------------------------
        sns.lineplot(
            data=sub,
            x="event_rank",
            y=y_col,
            hue="Group",
            palette=palette_labels,
            errorbar=("ci", 95),
            ax=ax_grp,
            legend=(row_idx == 0),
        )
        ax_grp.set_xlabel("Event rank (within phase)" if row_idx == n_rows - 1 else "", fontsize=10)
        ax_grp.set_ylabel(ylabel, fontsize=10)
        ax_grp.set_title(f"{title} — group comparison", fontsize=10)

        # ---- Right: individual flies (Magnet=y) ---------------------------
        sub_y = sub_phase[sub_phase["Magnet"] == "y"].dropna(subset=[y_col])
        n_flies = sub_y["fly"].nunique()
        for fly in sub_y["fly"].unique():
            fd = sub_y[sub_y["fly"] == fly].sort_values("event_rank")
            ax_ind.plot(fd["event_rank"], fd[y_col], color="#3953A4", alpha=0.3, lw=0.9)
        mean_line = sub_y.groupby("event_rank")[y_col].mean().reset_index()
        ax_ind.plot(
            mean_line["event_rank"],
            mean_line[y_col],
            color="black",
            lw=2.5,
            zorder=10,
            label=f"Mean (n={n_flies})",
        )
        ax_ind.set_xlabel("Event rank (within phase)" if row_idx == n_rows - 1 else "", fontsize=10)
        ax_ind.set_ylabel(ylabel, fontsize=10)
        ax_ind.set_title(f"{title} — Magnet=y individual flies (n={n_flies})", fontsize=10)
        ax_ind.legend(fontsize=8, loc="upper right")

    # Global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0, 0].get_legend():
        axes[0, 0].get_legend().remove()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(groups) + 1, fontsize=9, bbox_to_anchor=(0.5, 1.01), frameon=False
    )

    axes[0, 0].annotate(
        "Both groups (mean ± 95% CI)",
        xy=(0.5, 1.16),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].annotate(
        "Magnet=y — individual flies",
        xy=(0.5, 1.16),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        f"MagnetBlock — metrics by event number ({phase_label}, ranks 1–{max_rank})",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_event_rank_overview(
    event_rank_df: pd.DataFrame,
    max_rank: int | None = None,
    focus_hours: float = 1.0,
    output_dir: Path = Path("."),
) -> None:
    """
    Produce two 3 × 2 figures — one per phase — with x = sequential event
    number *within that phase*.

    Phases:
      ``pre``  — events with onset ≤ focus_hours (magnet active)
      ``post`` — events with onset > focus_hours (magnet off)

    Left column  : both Magnet groups, mean ± 95 % CI
    Right column : Magnet=y individual flies (spaghetti + bold mean)

    Rows:
      1  Interaction duration (s)
      2  Fly-ball distance during events (mm)
      3  Inter-event interval (s)
    """
    groups = sorted(event_rank_df["Magnet"].unique())
    n_per_group = {g: event_rank_df[event_rank_df["Magnet"] == g]["fly"].nunique() for g in groups}
    palette = make_palette(groups)
    label_map = {g: label_group(g, n_per_group[g]) for g in groups}
    palette_labels = {label_map[g]: palette[g] for g in groups}

    phase_specs = [
        ("pre", f"first {int(focus_hours * 60)} min — magnet ON"),
        ("post", f"after {int(focus_hours * 60)} min — magnet OFF"),
    ]

    for phase, phase_label in phase_specs:
        phase_data = event_rank_df[event_rank_df["phase"] == phase]
        if phase_data.empty:
            print(f"  No events for phase '{phase}', skipping.")
            continue

        # Per-phase cap
        if max_rank is None:
            per_fly_max = phase_data.groupby("fly")["event_rank"].max()
            cap = int(np.percentile(per_fly_max, 90))
        else:
            cap = max_rank

        capped = phase_data[phase_data["event_rank"] <= cap].copy()
        out = output_dir / f"event_rank_overview_{phase}"
        _plot_event_rank_phase(
            data=capped,
            phase=phase,
            phase_label=phase_label,
            max_rank=cap,
            groups=groups,
            palette_labels=palette_labels,
            label_map=label_map,
            output_path=out,
        )


# ---------------------------------------------------------------------------
# CSV export — aligned to first push (5 px)
# ---------------------------------------------------------------------------


def export_aligned_to_first_push_csv(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    output_dir: Path,
    bin_size_min: float,
    n_permutations: int,
    window_min: float = 60.0,
    gate_open_min: float | None = None,
    gate_close_min: float | None = None,
    n_bootstrap: int = 2000,
    tag: str = "5px",
) -> pd.DataFrame:
    """Export comparison-level stats for the interaction-rate plot aligned to
    first successful push.

    Exported file
    -------------
    aligned_to_first_push_{tag}_stats.csv
        One row per relative-time bin with sample sizes, effect size,
        test name, p-values (raw + FDR-corrected), and 95% bootstrap CI.
    """
    if first_push_df.empty:
        print("  No first-push data — skipping aligned CSV export.")
        return pd.DataFrame()

    merged = rate_df.merge(
        first_push_df[["fly", "Magnet", "t_first_push_min"]],
        on=["fly", "Magnet"],
        how="inner",
    )
    if gate_open_min is not None:
        merged = merged[merged["bin_center_min"] >= gate_open_min].copy()
    if gate_close_min is not None:
        merged = merged[merged["bin_center_min"] <= gate_close_min].copy()
    merged["rel_time_min"] = merged["bin_center_min"] - merged["t_first_push_min"]
    merged["rel_bin"] = (merged["rel_time_min"] / bin_size_min).round() * bin_size_min
    aligned = merged[merged["rel_bin"].abs() <= window_min].copy()

    # ── Between-group stats CSV ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    rel_bins = sorted(aligned["rel_bin"].dropna().unique().tolist())
    rows = []
    for rel_bin in rel_bins:
        b = aligned[aligned["rel_bin"] == rel_bin]
        ctrl = b[b["Magnet"] == "n"]["rate_per_min"].dropna().values
        test = b[b["Magnet"] == "y"]["rate_per_min"].dropna().values

        n_ctrl = int(len(ctrl))
        n_test = int(len(test))

        if n_ctrl == 0 or n_test == 0:
            p_raw = np.nan
            obs_diff = np.nan
            ci_lower, ci_upper = np.nan, np.nan
            mean_ctrl = float(np.mean(ctrl)) if n_ctrl > 0 else np.nan
            mean_test = float(np.mean(test)) if n_test > 0 else np.nan
            pct_change, pct_ci_lower, pct_ci_upper = np.nan, np.nan, np.nan
            d = np.nan
        else:
            mean_ctrl = float(np.mean(ctrl))
            mean_test = float(np.mean(test))
            obs_diff = mean_test - mean_ctrl

            combined = np.concatenate([ctrl, test])
            n_c = len(ctrl)
            perm_diffs = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                rng.shuffle(combined)
                perm_diffs[i] = np.mean(combined[n_c:]) - np.mean(combined[:n_c])
            p_raw = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

            ci_lower, ci_upper = bootstrap_ci_difference(
                ctrl,
                test,
                n_bootstrap=n_bootstrap,
                ci=95,
                random_state=4242 + int(rel_bin * 10),
            )

            if mean_ctrl != 0 and np.isfinite(mean_ctrl):
                pct_change = (obs_diff / mean_ctrl) * 100.0
                pct_ci_lower = (ci_lower / mean_ctrl) * 100.0 if np.isfinite(ci_lower) else np.nan
                pct_ci_upper = (ci_upper / mean_ctrl) * 100.0 if np.isfinite(ci_upper) else np.nan
            else:
                pct_change, pct_ci_lower, pct_ci_upper = np.nan, np.nan, np.nan

            d = cohens_d(ctrl, test) if n_ctrl > 1 and n_test > 1 else np.nan

        rows.append(
            {
                "Comparison": "between-group (Magnet=y vs Magnet=n)",
                "Test_Type": "two-sample permutation test",
                "Rel_Bin_min": rel_bin,
                "Push_Phase": "pre_push" if rel_bin < 0 else "post_push",
                "ConditionA": "n",
                "ConditionB": "y",
                "N_Control": n_ctrl,
                "N_Test": n_test,
                "Mean_Control": mean_ctrl,
                "Mean_Test": mean_test,
                "Difference_Test_minus_Control": obs_diff,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "Pct_Change": pct_change,
                "Pct_CI_Lower": pct_ci_lower,
                "Pct_CI_Upper": pct_ci_upper,
                "Effect_Size": d,
                "Effect_Size_Type": "Cohens_d",
                "P_Value_Raw": p_raw,
                "Test_Used": "two-sample permutation test",
                "N_Permutations": int(n_permutations),
                "N_Bootstrap": int(n_bootstrap),
            }
        )

    between_df = pd.DataFrame(rows)
    valid = between_df["P_Value_Raw"].notna()
    between_df["P_Value_FDR"] = np.nan
    if valid.any():
        _, p_corr, _, _ = multipletests(between_df.loc[valid, "P_Value_Raw"].values, method="fdr_bh")
        between_df.loc[valid, "P_Value_FDR"] = p_corr
    between_df["Significant_FDR"] = between_df["P_Value_FDR"] < 0.05

    between_csv = output_dir / f"aligned_to_first_push_{tag}_stats.csv"
    between_df.to_csv(between_csv, index=False, float_format="%.6f")
    print(f"  Saved: {between_csv}")
    return between_df


# ---------------------------------------------------------------------------
# CSV export — post-push rate change
# ---------------------------------------------------------------------------


def export_post_push_rate_change_csv(
    change_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
    pre_window_min: tuple,
    post_window_min: tuple,
    n_permutations: int,
    n_bootstrap: int = 2000,
    out_tag: str = "post_push_rate_change",
) -> None:
    """Export summary statistics for the post-push rate change analysis.

    Exported files
    --------------
    {out_tag}_stats.csv
        Three-row summary: within-group sign-permutation tests for each
        Magnet condition plus the between-group permutation test, with
        effect sizes (Cohen's d) and bootstrap 95 % CI.
    """
    if change_df.empty:
        print(f"  No change data — skipping {out_tag} CSV export.")
        return

    def _safe_adjusted_p(p: float) -> float:
        """Avoid reporting exact zero for permutation p-values.

        With finite permutations, the practical lower bound is 1/(N+1).
        """
        if p is None or not np.isfinite(p):
            return np.nan
        p = float(p)
        if p <= 0:
            return 1.0 / (n_permutations + 1.0)
        return p

    def _bootstrap_paired_change(pre_vals: np.ndarray, post_vals: np.ndarray) -> tuple[float, float, float, float]:
        """Bootstrap CI for paired raw and percent change means.

        Returns
        -------
        (raw_ci_lo, raw_ci_hi, pct_ci_lo, pct_ci_hi)
        """
        n = len(pre_vals)
        if n < 2:
            return np.nan, np.nan, np.nan, np.nan

        raw_boot = np.empty(n_bootstrap, dtype=float)
        pct_boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            pre_s = pre_vals[idx]
            post_s = post_vals[idx]
            raw_boot[i] = np.mean(post_s - pre_s)

            # Percent change is computed fly-wise then averaged
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_vals = ((post_s - pre_s) / pre_s) * 100.0
            pct_vals = pct_vals[np.isfinite(pct_vals)]
            pct_boot[i] = np.mean(pct_vals) if len(pct_vals) > 0 else np.nan

        raw_ci_lo = float(np.nanpercentile(raw_boot, 2.5))
        raw_ci_hi = float(np.nanpercentile(raw_boot, 97.5))
        pct_ci_lo = float(np.nanpercentile(pct_boot, 2.5))
        pct_ci_hi = float(np.nanpercentile(pct_boot, 97.5))
        return raw_ci_lo, raw_ci_hi, pct_ci_lo, pct_ci_hi

    # ── Summary stats CSV ────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    summary_rows = []

    # Within-group tests
    for grp in ["y", "n"]:
        grp_info = stats.get(grp, {})
        grp_df = change_df[change_df["Magnet"] == grp].dropna(subset=["pre_mean", "post_mean", "delta"]).copy()
        pre_vals = grp_df["pre_mean"].values
        post_vals = grp_df["post_mean"].values
        scores = grp_df["delta"].values
        n = int(len(scores))
        obs_mean = grp_info.get("obs_mean", np.nan)
        p_raw = _safe_adjusted_p(grp_info.get("p_value", np.nan))

        mean_pre = float(np.mean(pre_vals)) if n > 0 else np.nan
        mean_post = float(np.mean(post_vals)) if n > 0 else np.nan
        mean_change_raw = float(np.mean(scores)) if n > 0 else np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            pct_vals = ((post_vals - pre_vals) / pre_vals) * 100.0
        pct_vals = pct_vals[np.isfinite(pct_vals)]
        mean_change_pct = float(np.mean(pct_vals)) if len(pct_vals) > 0 else np.nan

        # Effect size: one-sample Cohen's d vs. 0 (d_z = mean/std)
        if n >= 2:
            std_scores = float(np.std(scores, ddof=1))
            d_z = float(obs_mean / std_scores) if std_scores > 0 and np.isfinite(obs_mean) else np.nan
        else:
            d_z = np.nan

        # Bootstrap 95% CI for paired raw and percent change
        ci_raw_lo, ci_raw_hi, ci_pct_lo, ci_pct_hi = _bootstrap_paired_change(pre_vals, post_vals)

        summary_rows.append(
            {
                "Comparison": f"Magnet={grp} within-group",
                "Test_Type": "one-sample sign permutation test",
                "ConditionA": "0 (null)",
                "ConditionB": f"Magnet={grp}",
                "N_Group": n,
                "N_Control": np.nan,
                "N_Test": np.nan,
                "Mean_Pre": mean_pre,
                "Mean_Post": mean_post,
                "Observed_Statistic": obs_mean,
                "Mean_Change_Raw": mean_change_raw,
                "CI_Change_Raw_Lower": ci_raw_lo,
                "CI_Change_Raw_Upper": ci_raw_hi,
                "Mean_Change_Pct": mean_change_pct,
                "CI_Change_Pct_Lower": ci_pct_lo,
                "CI_Change_Pct_Upper": ci_pct_hi,
                "Effect_Size": d_z,
                "Effect_Size_Type": "Cohens_d_z",
                "P_Value_Raw": p_raw,
                "Test_Used": "one-sample sign permutation test (H0: mean delta == 0)",
                "N_Permutations": int(n_permutations),
                "N_Bootstrap": int(n_bootstrap),
                "Pre_Window_Far_min": pre_window_min[0],
                "Pre_Window_Near_min": pre_window_min[1],
                "Post_Window_Near_min": post_window_min[0],
                "Post_Window_Far_min": post_window_min[1],
            }
        )

    # Between-group test
    between_info = stats.get("between", {})
    obs_diff = between_info.get("obs_diff", np.nan)
    p_between = _safe_adjusted_p(between_info.get("p_value", np.nan))

    y_scores = change_df[change_df["Magnet"] == "y"]["delta"].dropna().values
    n_scores = change_df[change_df["Magnet"] == "n"]["delta"].dropna().values
    y_pre = change_df[change_df["Magnet"] == "y"]["pre_mean"].dropna().values
    y_post = change_df[change_df["Magnet"] == "y"]["post_mean"].dropna().values
    n_pre = change_df[change_df["Magnet"] == "n"]["pre_mean"].dropna().values
    n_post = change_df[change_df["Magnet"] == "n"]["post_mean"].dropna().values
    n_y = int(len(y_scores))
    n_n = int(len(n_scores))

    mean_pre_y = float(np.mean(y_pre)) if n_y > 0 else np.nan
    mean_post_y = float(np.mean(y_post)) if n_y > 0 else np.nan
    mean_pre_n = float(np.mean(n_pre)) if n_n > 0 else np.nan
    mean_post_n = float(np.mean(n_post)) if n_n > 0 else np.nan

    mean_change_y = float(np.mean(y_scores)) if n_y > 0 else np.nan
    mean_change_n = float(np.mean(n_scores)) if n_n > 0 else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        y_pct = ((y_post - y_pre) / y_pre) * 100.0
        n_pct = ((n_post - n_pre) / n_pre) * 100.0
    y_pct = y_pct[np.isfinite(y_pct)]
    n_pct = n_pct[np.isfinite(n_pct)]
    mean_pct_y = float(np.mean(y_pct)) if len(y_pct) > 0 else np.nan
    mean_pct_n = float(np.mean(n_pct)) if len(n_pct) > 0 else np.nan
    diff_pct = mean_pct_y - mean_pct_n if np.isfinite(mean_pct_y) and np.isfinite(mean_pct_n) else np.nan

    # Bootstrap CI for the difference in means
    if n_y >= 2 and n_n >= 2:
        ci_lower_b, ci_upper_b = bootstrap_ci_difference(
            n_scores, y_scores, n_bootstrap=n_bootstrap, ci=95, random_state=99
        )
        d_between = cohens_d(n_scores, y_scores)

        pct_boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            yi = rng.integers(0, n_y, size=n_y)
            ni = rng.integers(0, n_n, size=n_n)

            y_pre_s, y_post_s = y_pre[yi], y_post[yi]
            n_pre_s, n_post_s = n_pre[ni], n_post[ni]

            with np.errstate(divide="ignore", invalid="ignore"):
                yp = ((y_post_s - y_pre_s) / y_pre_s) * 100.0
                npct = ((n_post_s - n_pre_s) / n_pre_s) * 100.0
            yp = yp[np.isfinite(yp)]
            npct = npct[np.isfinite(npct)]

            y_m = np.mean(yp) if len(yp) > 0 else np.nan
            n_m = np.mean(npct) if len(npct) > 0 else np.nan
            pct_boot[i] = y_m - n_m if np.isfinite(y_m) and np.isfinite(n_m) else np.nan

        ci_pct_b_lo = float(np.nanpercentile(pct_boot, 2.5))
        ci_pct_b_hi = float(np.nanpercentile(pct_boot, 97.5))
    else:
        ci_lower_b, ci_upper_b = np.nan, np.nan
        ci_pct_b_lo, ci_pct_b_hi = np.nan, np.nan
        d_between = np.nan

    summary_rows.append(
        {
            "Comparison": "between-group (Magnet=y vs Magnet=n)",
            "Test_Type": "two-sample permutation test",
            "ConditionA": "Magnet=n",
            "ConditionB": "Magnet=y",
            "N_Group": n_y + n_n,
            "N_Control": n_n,
            "N_Test": n_y,
            "Mean_Pre_Control": mean_pre_n,
            "Mean_Post_Control": mean_post_n,
            "Mean_Pre_Test": mean_pre_y,
            "Mean_Post_Test": mean_post_y,
            "Observed_Statistic": obs_diff,
            "Mean_Change_Raw_Control": mean_change_n,
            "Mean_Change_Raw_Test": mean_change_y,
            "Difference_In_Change_Raw": obs_diff,
            "CI_Change_Raw_Lower": ci_lower_b,
            "CI_Change_Raw_Upper": ci_upper_b,
            "Mean_Change_Pct_Control": mean_pct_n,
            "Mean_Change_Pct_Test": mean_pct_y,
            "Difference_In_Change_Pct": diff_pct,
            "CI_Change_Pct_Lower": ci_pct_b_lo,
            "CI_Change_Pct_Upper": ci_pct_b_hi,
            "Effect_Size": d_between,
            "Effect_Size_Type": "Cohens_d",
            "P_Value_Raw": p_between,
            "Test_Used": "two-sample permutation test (H0: mean_delta_y == mean_delta_n)",
            "N_Permutations": int(n_permutations),
            "N_Bootstrap": int(n_bootstrap),
            "Pre_Window_Far_min": pre_window_min[0],
            "Pre_Window_Near_min": pre_window_min[1],
            "Post_Window_Near_min": post_window_min[0],
            "Post_Window_Far_min": post_window_min[1],
        }
    )

    stats_df = pd.DataFrame(summary_rows)

    # Benjamini-Hochberg correction across the comparisons in this table
    valid = stats_df["P_Value_Raw"].notna()
    stats_df["P_Value_Adjusted"] = np.nan
    stats_df["P_Value_Adjustment_Method"] = "Benjamini-Hochberg"
    if valid.any():
        _, p_corr, _, _ = multipletests(stats_df.loc[valid, "P_Value_Raw"].values, method="fdr_bh")
        stats_df.loc[valid, "P_Value_Adjusted"] = p_corr
    stats_df["Significant_Adjusted"] = stats_df["P_Value_Adjusted"] < 0.05

    # Backward-compatible aliases used by older downstream scripts
    stats_df["P_Value_FDR"] = stats_df["P_Value_Adjusted"]
    stats_df["Significant_FDR"] = stats_df["Significant_Adjusted"]

    stats_csv = output_dir / f"{out_tag}_stats.csv"
    # Keep full numeric precision (incl. scientific notation) to avoid very
    # small p-values being displayed as 0 due fixed-decimal rounding.
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interaction rate analysis for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=DEFAULT_BIN_SIZE_MIN,
        help="Time bin size in minutes (default: 0.1667 = 10 seconds)",
    )
    parser.add_argument(
        "--bin-size-sec",
        type=float,
        default=None,
        help="Time bin size in seconds (overrides --bin-size when set)",
    )
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: next to dataset)")
    parser.add_argument("--focus-hours", type=float, default=1.0, help="Hours to use for trend analysis (default: 1.0)")
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Cap event rank axis at this value (default: 90th-percentile of per-fly max rank)",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help=(
            "Run the full legacy pipeline (all metrics and dashboards). "
            "Default behavior runs only first-push-centered interaction-rate curves "
            "and pre/post push boxplots."
        ),
    )
    args = parser.parse_args()
    if args.bin_size_sec is not None:
        if args.bin_size_sec <= 0:
            raise ValueError("--bin-size-sec must be > 0")
        args.bin_size = args.bin_size_sec / 60.0
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    # This companion script is dedicated to the broader non-rate metrics workflow.
    # Force full-analysis mode so users don't accidentally run the lightweight rate-only path.
    args.full_analysis = True

    dataset_path = (
        "/mnt/upramdya_data/MD/MagnetBlock/Datasets/"
        "260224_12_summary_magnet_block_folders_Data_Full/"
        "coordinates/pooled_coordinates.feather"
    )

    if args.output_dir is None:
        output_dir = Path(dataset_path).parent.parent / "Plots" / "interaction_rate"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAGNETBLOCK — INTERACTION RATE ANALYSIS")
    print("=" * 60)
    print(f"Bin size     : {args.bin_size:.6f} min ({args.bin_size * 60.0:.1f} s)")
    print(f"Permutations : {args.n_permutations}")
    print(f"Trend window : first {int(args.focus_hours * 60)} min")
    print("Mode         : " + ("full analysis" if args.full_analysis else "interaction-rate rework only (default)"))
    print(f"Output dir   : {output_dir}")
    print("=" * 60)

    df = load_dataset(dataset_path)

    # For Magnet=n flies the ball is physically inaccessible before focus_hours
    # (barrier not yet removed).  Any detected events in that period are tracking
    # artefacts; zero them out so ALL downstream metrics ignore them.
    cutoff_s = args.focus_hours * 3600.0
    pre_sham_mask = (df["Magnet"] == "n") & (df["time"] < cutoff_s)
    n_artefact_frames = int(pre_sham_mask.sum())
    event_cols = [c for c in ["interaction_event", "interaction_event_onset"] if c in df.columns]
    df.loc[pre_sham_mask, event_cols] = np.nan
    print(
        f"\n[Artefact removal] Zeroed event columns for Magnet=n, t < {args.focus_hours:.1f} h "
        f"({n_artefact_frames:,} frames, columns: {event_cols})"
    )

    onsets = extract_event_onsets(df)

    print(f"\nComputing per-fly event rates (bin size = {args.bin_size} min)...")
    rate_df = compute_rates_per_fly(onsets, df, bin_size_min=args.bin_size)
    print(f"Rate table shape: {rate_df.shape}")
    print(rate_df.groupby(["Magnet", "bin_idx"])["rate_per_min"].mean().unstack().round(3).to_string())

    # ------------------------------------------------------------------ #
    # Default lightweight mode: only the interaction-rate rework figures
    # ------------------------------------------------------------------ #
    if not args.full_analysis:
        print("\n" + "=" * 60)
        print("DEFAULT MODE: FIRST-PUSH-CENTERED RATE + PRE/POST BOXPLOTS")
        print("=" * 60)

        print("\nComputing ball displacement per event (for first-push detection)...")
        event_disp_df = compute_ball_displacement_per_event(df, fps=FPS)

        print("\nComputing first successful push (threshold >5px)...")
        first_push_df_5px = compute_first_successful_push(
            event_disp_df,
            focus_hours=args.focus_hours,
            threshold_px=5.0,
        )

        if len(first_push_df_5px) < 3:
            print("  Too few flies with a qualifying push — skipping default interaction-rate rework plots.")
            return

        print("\nGenerating centered-on-first-push interaction-rate curve...")
        plot_interaction_rate_aligned_to_first_push(
            rate_df=rate_df,
            first_push_df=first_push_df_5px,
            bin_size_min=args.bin_size,
            window_min=10.0,
            gate_open_min=args.focus_hours * 60.0,
            gate_close_min=120.0,
            output_path=output_dir / "aligned_to_first_push_5px",
        )

        print("\nExporting aligned-to-first-push statistics CSV...")
        export_aligned_to_first_push_csv(
            rate_df=rate_df,
            first_push_df=first_push_df_5px,
            output_dir=output_dir,
            bin_size_min=args.bin_size,
            n_permutations=args.n_permutations,
            window_min=10.0,
            gate_open_min=args.focus_hours * 60.0,
            gate_close_min=120.0,
            n_bootstrap=2000,
            tag="5px",
        )

        for pre_win, post_win, out_tag in [
            ((10.0, 1.0), (1.0, 10.0), "post_push_rate_change_10min"),
            ((5.0, 1.0), (1.0, 5.0), "post_push_rate_change_5min"),
            ((3.0, 1.0), (1.0, 3.0), "post_push_rate_change_3min"),
        ]:
            pre_str = f"-{int(pre_win[0])} to -{int(pre_win[1])} min"
            post_str = f"+{int(post_win[0])} to +{int(post_win[1])} min"
            print(f"\nComputing pre/post rate change per fly (pre: {pre_str}, post: {post_str})...")
            change_df = compute_post_push_rate_change(
                rate_df=rate_df,
                first_push_df=first_push_df_5px,
                focus_hours=args.focus_hours,
                pre_window_min=pre_win,
                post_window_min=post_win,
            )
            if len(change_df) < 3:
                print(f"  Too few flies with both pre and post data for {out_tag} — skipping.")
                continue

            print("\nRunning permutation tests on change scores...")
            change_stats = test_post_push_change(change_df, n_permutations=args.n_permutations)

            print("\nGenerating post-push rate change plots...")
            plot_post_push_rate_change(
                change_df=change_df,
                stats=change_stats,
                focus_hours=args.focus_hours,
                pre_window_min=pre_win,
                post_window_min=post_win,
                n_permutations=args.n_permutations,
                output_path=output_dir / out_tag,
            )
            plot_post_push_rate_change_two_panel(
                change_df=change_df,
                stats=change_stats,
                focus_hours=args.focus_hours,
                pre_window_min=pre_win,
                post_window_min=post_win,
                n_permutations=args.n_permutations,
                subplot_width_mm=55.0,
                subplot_height_mm=73.0,
                output_path=output_dir / f"{out_tag}_two_panel_small",
            )

            print("\nExporting post-push rate change CSV statistics...")
            export_post_push_rate_change_csv(
                change_df=change_df,
                stats=change_stats,
                output_dir=output_dir,
                pre_window_min=pre_win,
                post_window_min=post_win,
                n_permutations=args.n_permutations,
                n_bootstrap=2000,
                out_tag=out_tag,
            )

        print("\nDefault interaction-rate rework outputs completed.")
        return

    # Between-group permutation test per time bin
    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST (Magnet=y vs Magnet=n) per time bin")
    print("-" * 60)
    between_results = permutation_test_between_groups(
        rate_df,
        control_val="n",
        test_val="y",
        n_permutations=args.n_permutations,
    )

    # Trend test (first hour)
    print("\n" + "-" * 60)
    print(f"TREND TEST — slope per fly, first {int(args.focus_hours * 60)} min")
    print("-" * 60)
    trend_results = trend_test_per_group(
        rate_df,
        focus_hours=args.focus_hours,
        n_permutations=args.n_permutations,
    )

    print(f"TREND TEST — rate slope per fly, hour 2 ({int(args.focus_hours * 60)}–120 min)")
    print("-" * 60)
    trend_results_h2 = trend_test_per_group(
        rate_df,
        start_hours=args.focus_hours,
        focus_hours=2.0,
        n_permutations=args.n_permutations,
    )

    # Plot
    print("\nGenerating rate plot...")
    plot_path = output_dir / "interaction_rate_over_time"
    plot_interaction_rate(
        rate_df=rate_df,
        between_results=between_results,
        trend_results=trend_results,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=plot_path,
    )

    print("\nGenerating individual-fly rate plot (Magnet=y)...")
    plot_individual_rate(
        rate_df=rate_df,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "interaction_rate_individual_flies",
    )

    # Save statistics
    print("\nSaving rate statistics...")
    save_stats(
        between_results=between_results,
        trend_results=trend_results,
        rate_df=rate_df,
        output_dir=output_dir,
        n_permutations=args.n_permutations,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
    )

    print("\nSaving gate-opening detailed interaction-rate CSV...")
    export_interaction_rate_gate_opening_csv(
        rate_df=rate_df,
        between_results=between_results,
        output_dir=output_dir,
        focus_hours=args.focus_hours,
        n_bootstrap=2000,
    )

    # ------------------------------------------------------------------ #
    # Duration (persistence) analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("INTERACTION DURATION (PERSISTENCE) ANALYSIS")
    print("=" * 60)

    print(f"\nComputing per-event durations (bin size = {args.bin_size} min, fps = {FPS})...")
    dur_df = compute_durations_per_fly(df, bin_size_min=args.bin_size, fps=FPS)
    print(f"Duration table shape: {dur_df.shape}")
    print(
        dur_df.dropna(subset=["mean_duration_s"])
        .groupby(["Magnet", "bin_idx"])["mean_duration_s"]
        .mean()
        .unstack()
        .round(2)
        .to_string()
    )

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — duration per time bin")
    print("-" * 60)
    dur_between = permutation_test_between_groups(
        dur_df.dropna(subset=["mean_duration_s"]),
        control_val="n",
        test_val="y",
        metric="mean_duration_s",
        n_permutations=args.n_permutations,
    )

    print("\n" + "-" * 60)
    print(f"TREND TEST — duration slope per fly, first {int(args.focus_hours * 60)} min")
    print("-" * 60)
    dur_trend = trend_test_per_group(
        dur_df.dropna(subset=["mean_duration_s"]),
        focus_hours=args.focus_hours,
        metric="mean_duration_s",
        n_permutations=args.n_permutations,
    )

    print(f"TREND TEST — duration slope per fly, hour 2 ({int(args.focus_hours * 60)}–120 min)")
    print("-" * 60)
    dur_trend_h2 = trend_test_per_group(
        dur_df.dropna(subset=["mean_duration_s"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="mean_duration_s",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating duration plot...")
    dur_plot_path = output_dir / "interaction_duration_over_time"
    plot_interaction_duration(
        dur_df=dur_df,
        between_results=dur_between,
        trend_results=dur_trend,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=dur_plot_path,
    )

    print("\nGenerating individual-fly duration plot (Magnet=y)...")
    plot_individual_duration(
        dur_df=dur_df,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "interaction_duration_individual_flies",
    )

    print("\nSaving duration statistics...")
    save_duration_stats(
        between_results=dur_between,
        trend_results=dur_trend,
        dur_df=dur_df,
        output_dir=output_dir,
        n_permutations=args.n_permutations,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
    )

    # ------------------------------------------------------------------ #
    # Proximity (fly-ball distance) analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FLY-BALL PROXIMITY (ENGAGEMENT) ANALYSIS")
    print("=" * 60)

    print(f"\nComputing per-fly proximity (bin size = {args.bin_size} min)...")
    prox_df = compute_proximity_per_fly(df, bin_size_min=args.bin_size)
    print(prox_df.groupby(["Magnet", "bin_idx"])["mean_dist_mm"].mean().unstack().round(2).to_string())

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — fly-ball distance per time bin")
    print("-" * 60)
    prox_between = permutation_test_between_groups(
        prox_df.dropna(subset=["mean_dist_mm"]),
        control_val="n",
        test_val="y",
        metric="mean_dist_mm",
        n_permutations=args.n_permutations,
    )

    print("\n" + "-" * 60)
    print(f"TREND TEST — distance slope per fly, first {int(args.focus_hours * 60)} min")
    print("-" * 60)
    prox_trend = trend_test_per_group(
        prox_df.dropna(subset=["mean_dist_mm"]),
        focus_hours=args.focus_hours,
        metric="mean_dist_mm",
        n_permutations=args.n_permutations,
    )

    print(f"TREND TEST — proximity slope per fly, hour 2 ({int(args.focus_hours * 60)}–120 min)")
    print("-" * 60)
    prox_trend_h2 = trend_test_per_group(
        prox_df.dropna(subset=["mean_dist_mm"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="mean_dist_mm",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating proximity plot...")
    plot_proximity(
        prox_df=prox_df,
        between_results=prox_between,
        trend_results=prox_trend,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "proximity_over_time",
    )

    print("\nGenerating individual-fly proximity plot (Magnet=y)...")
    plot_individual_proximity(
        prox_df=prox_df,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "proximity_individual_flies",
    )

    print("\nSaving proximity statistics...")
    save_proximity_stats(
        between_results=prox_between,
        trend_results=prox_trend,
        prox_df=prox_df,
        output_dir=output_dir,
        n_permutations=args.n_permutations,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
    )

    # --- Summary dashboard moved to after all metrics are computed ----------

    # ------------------------------------------------------------------ #
    # Inter-event interval (IEI) analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("INTER-EVENT INTERVAL (IEI) ANALYSIS")
    print("=" * 60)

    print(f"\nComputing IEI per fly (bin size = {args.bin_size} min)...")
    iei_df = compute_iei_per_fly(df, bin_size_min=args.bin_size)

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — IEI per time bin")
    print("-" * 60)
    iei_between = permutation_test_between_groups(
        iei_df.dropna(subset=["mean_iei_s"]),
        control_val="n",
        test_val="y",
        metric="mean_iei_s",
        n_permutations=args.n_permutations,
    )

    print(f"\nTREND TEST — IEI slope per fly, hour 1")
    iei_trend_h1 = trend_test_per_group(
        iei_df.dropna(subset=["mean_iei_s"]),
        focus_hours=args.focus_hours,
        metric="mean_iei_s",
        n_permutations=args.n_permutations,
    )
    print(f"TREND TEST — IEI slope per fly, hour 2")
    iei_trend_h2 = trend_test_per_group(
        iei_df.dropna(subset=["mean_iei_s"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="mean_iei_s",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating IEI plots...")
    _generic_group_plot(
        df=iei_df,
        y_col="mean_iei_s",
        ylabel="Inter-event interval (s)",
        title="Inter-event interval over time (MagnetBlock)",
        between_results=iei_between,
        trend_h1=iei_trend_h1,
        trend_h2=iei_trend_h2,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "iei_over_time",
    )
    _generic_individual_plot(
        df=iei_df,
        y_col="mean_iei_s",
        ylabel="Inter-event interval (s)",
        title="IEI over time — Magnet=y individual flies",
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "iei_individual_flies",
    )
    plot_boxplot_per_bin(
        df=iei_df,
        y_col="mean_iei_s",
        ylabel="Inter-event interval (s)",
        title="IEI per bin — individual fly distribution (MagnetBlock)",
        focus_hours=args.focus_hours,
        output_path=output_dir / "iei_boxplot_per_bin",
    )

    # ------------------------------------------------------------------ #
    # Ball displacement per event
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BALL DISPLACEMENT PER EVENT")
    print("=" * 60)

    print("\nComputing ball displacement per event...")
    event_disp_df = compute_ball_displacement_per_event(df, fps=FPS)

    print(f"\nBinning displacement (bin size = {args.bin_size} min)...")
    disp_df = bin_displacement_per_fly(event_disp_df, df_full=df, bin_size_min=args.bin_size)

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — ball displacement per time bin")
    print("-" * 60)
    disp_between = permutation_test_between_groups(
        disp_df.dropna(subset=["mean_displacement_px"]),
        control_val="n",
        test_val="y",
        metric="mean_displacement_px",
        n_permutations=args.n_permutations,
    )

    print("\nTREND TEST — displacement slope per fly, hour 1")
    disp_trend_h1 = trend_test_per_group(
        disp_df.dropna(subset=["mean_displacement_px"]),
        focus_hours=args.focus_hours,
        metric="mean_displacement_px",
        n_permutations=args.n_permutations,
    )
    print("TREND TEST — displacement slope per fly, hour 2")
    disp_trend_h2 = trend_test_per_group(
        disp_df.dropna(subset=["mean_displacement_px"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="mean_displacement_px",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating ball displacement plots...")
    _generic_group_plot(
        df=disp_df,
        y_col="mean_displacement_px",
        ylabel="Ball displacement per event (px)",
        title="Ball displacement per event over time (MagnetBlock)",
        between_results=disp_between,
        trend_h1=disp_trend_h1,
        trend_h2=disp_trend_h2,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "ball_displacement_over_time",
    )
    _generic_individual_plot(
        df=disp_df,
        y_col="mean_displacement_px",
        ylabel="Ball displacement per event (px)",
        title="Ball displacement — Magnet=y individual flies",
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "ball_displacement_individual_flies",
    )

    # ------------------------------------------------------------------ #
    # Approach speed
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("APPROACH SPEED ANALYSIS")
    print("=" * 60)

    print("\nComputing approach speed per event (2s pre-event window)...")
    vel_event_df = compute_approach_speed_per_event(df, fps=FPS, pre_window_s=2.0)

    print(f"\nBinning approach speed (bin size = {args.bin_size} min)...")
    vel_df = bin_approach_speed_per_fly(vel_event_df, df_full=df, bin_size_min=args.bin_size)

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — approach speed per time bin")
    print("-" * 60)
    vel_between = permutation_test_between_groups(
        vel_df.dropna(subset=["mean_approach_speed_px_s"]),
        control_val="n",
        test_val="y",
        metric="mean_approach_speed_px_s",
        n_permutations=args.n_permutations,
    )

    print("\nTREND TEST — approach speed slope per fly, hour 1")
    vel_trend_h1 = trend_test_per_group(
        vel_df.dropna(subset=["mean_approach_speed_px_s"]),
        focus_hours=args.focus_hours,
        metric="mean_approach_speed_px_s",
        n_permutations=args.n_permutations,
    )
    print("TREND TEST — approach speed slope per fly, hour 2")
    vel_trend_h2 = trend_test_per_group(
        vel_df.dropna(subset=["mean_approach_speed_px_s"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="mean_approach_speed_px_s",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating approach speed plots...")
    _generic_group_plot(
        df=vel_df,
        y_col="mean_approach_speed_px_s",
        ylabel="Approach speed (px/s)",
        title="Fly approach speed before events over time (MagnetBlock)",
        between_results=vel_between,
        trend_h1=vel_trend_h1,
        trend_h2=vel_trend_h2,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "approach_speed_over_time",
    )
    _generic_individual_plot(
        df=vel_df,
        y_col="mean_approach_speed_px_s",
        ylabel="Approach speed (px/s)",
        title="Approach speed — Magnet=y individual flies",
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "approach_speed_individual_flies",
    )

    # ------------------------------------------------------------------ #
    # First successful push + alignment analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FIRST SUCCESSFUL PUSH & ALIGNMENT ANALYSIS")
    print("=" * 60)

    for threshold_px in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tag = f"{int(threshold_px)}px"
        print(f"\nComputing first successful push (threshold >{tag})...")
        first_push_df = compute_first_successful_push(
            event_disp_df, focus_hours=args.focus_hours, threshold_px=threshold_px
        )
        if len(first_push_df) >= 3:
            if threshold_px == 5.0:
                print(f"  Generating rate-only alignment plot for threshold >{tag}...")
                plot_interaction_rate_aligned_to_first_push(
                    rate_df=rate_df,
                    first_push_df=first_push_df,
                    bin_size_min=args.bin_size,
                    window_min=10.0,
                    gate_open_min=args.focus_hours * 60.0,
                    gate_close_min=120.0,
                    output_path=output_dir / f"aligned_to_first_push_{tag}",
                )
                print(f"  Exporting aligned-to-first-push CSV statistics for threshold >{tag}...")
                export_aligned_to_first_push_csv(
                    rate_df=rate_df,
                    first_push_df=first_push_df,
                    output_dir=output_dir,
                    bin_size_min=args.bin_size,
                    n_permutations=args.n_permutations,
                    window_min=10.0,
                    gate_open_min=args.focus_hours * 60.0,
                    gate_close_min=120.0,
                    n_bootstrap=2000,
                    tag=tag,
                )
            else:
                print(f"  Generating alignment plot for threshold >{tag}...")
                plot_aligned_to_first_push(
                    rate_df=rate_df,
                    dur_df=dur_df,
                    prox_df=prox_df,
                    iei_df=iei_df,
                    first_push_df=first_push_df,
                    bin_size_min=args.bin_size,
                    window_min=60.0,
                    focus_hours=args.focus_hours,
                    threshold_px=threshold_px,
                    output_path=output_dir / f"aligned_to_first_push_{tag}",
                )
        else:
            print(f"  Too few flies with a >{tag} push — skipping alignment plot.")

    # ------------------------------------------------------------------ #
    # Post-push rate change analysis (5 px threshold)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("POST-PUSH RATE CHANGE ANALYSIS")
    print("=" * 60)

    first_push_df_5px = compute_first_successful_push(event_disp_df, focus_hours=args.focus_hours, threshold_px=5.0)
    if len(first_push_df_5px) >= 3:
        print("\nSaving first-unlock-aligned interaction-rate detailed CSVs...")
        export_interaction_rate_unlocking_csvs(
            rate_df=rate_df,
            first_push_df=first_push_df_5px,
            output_dir=output_dir,
            bin_size_min=args.bin_size,
            n_permutations=args.n_permutations,
            window_min=10.0,
            gate_open_min=args.focus_hours * 60.0,
            gate_close_min=120.0,
            n_bootstrap=2000,
        )

        print("\nComputing first-push magnitude per fly (5 px threshold cohort)...")
        first_push_mag_df = compute_first_push_magnitude(event_disp_df, first_push_df_5px)

        print("\nRunning between-group permutation test on first-push magnitude...")
        first_push_mag_stats = test_first_push_magnitude(
            first_push_mag_df,
            n_permutations=args.n_permutations,
            metric_col="first_push_max_disp_px",
        )

        print("\nGenerating first-push magnitude boxplot...")
        plot_first_push_magnitude(
            first_push_mag_df,
            stats=first_push_mag_stats,
            metric_col="first_push_max_disp_px",
            ylabel="First push magnitude (max displacement, px)",
            n_permutations=args.n_permutations,
            output_path=output_dir / "first_push_magnitude",
        )

        print("\nGenerating first-push-aligned ball-position trajectory plot (1 Hz, bootstrap CI)...")
        plot_ball_position_from_first_push(
            df=df,
            first_push_df=first_push_df_5px,
            window_min=60.0,
            downsample_s=1.0,
            n_bootstrap=1000,
            output_path=output_dir / "ball_position_from_first_push",
        )

        for pre_win, post_win, out_tag in [
            ((10.0, 1.0), (1.0, 10.0), "post_push_rate_change_10min"),
            ((5.0, 1.0), (1.0, 5.0), "post_push_rate_change_5min"),
            ((3.0, 1.0), (1.0, 3.0), "post_push_rate_change_3min"),
        ]:
            pre_str = f"-{int(pre_win[0])} to -{int(pre_win[1])} min"
            post_str = f"+{int(post_win[0])} to +{int(post_win[1])} min"
            print(f"\nComputing pre/post rate change per fly (pre: {pre_str}, post: {post_str})...")
            change_df = compute_post_push_rate_change(
                rate_df=rate_df,
                first_push_df=first_push_df_5px,
                focus_hours=args.focus_hours,
                pre_window_min=pre_win,
                post_window_min=post_win,
            )
            if len(change_df) >= 3:
                print("\nRunning permutation tests on change scores...")
                change_stats = test_post_push_change(change_df, n_permutations=args.n_permutations)
                print("\nGenerating post-push rate change plot...")
                plot_post_push_rate_change(
                    change_df=change_df,
                    stats=change_stats,
                    focus_hours=args.focus_hours,
                    pre_window_min=pre_win,
                    post_window_min=post_win,
                    n_permutations=args.n_permutations,
                    output_path=output_dir / out_tag,
                )
                print("\nGenerating compact two-panel post-push change plot...")
                plot_post_push_rate_change_two_panel(
                    change_df=change_df,
                    stats=change_stats,
                    focus_hours=args.focus_hours,
                    pre_window_min=pre_win,
                    post_window_min=post_win,
                    n_permutations=args.n_permutations,
                    subplot_width_mm=60.0,
                    subplot_height_mm=90.0,
                    output_path=output_dir / f"{out_tag}_two_panel_small",
                )
                print("\nExporting post-push rate change CSV statistics...")
                export_post_push_rate_change_csv(
                    change_df=change_df,
                    stats=change_stats,
                    output_dir=output_dir,
                    pre_window_min=pre_win,
                    post_window_min=post_win,
                    n_permutations=args.n_permutations,
                    n_bootstrap=2000,
                    out_tag=out_tag,
                )
            else:
                print(f"  Too few flies with both pre and post data for {out_tag} — skipping.")
    else:
        print("  Too few flies with a qualifying push — skipping.")

    # ------------------------------------------------------------------ #
    # Time in chamber analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("TIME IN CHAMBER ANALYSIS")
    print("=" * 60)

    print(f"\nComputing time in chamber (radius=50px, bin size = {args.bin_size} min)...")
    chamber_df = compute_time_in_chamber_per_fly(df, fps=FPS, bin_size_min=args.bin_size, chamber_radius_px=50.0)

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP TEST — time in chamber per time bin")
    print("-" * 60)
    chamber_between = permutation_test_between_groups(
        chamber_df.dropna(subset=["fraction_in_chamber"]),
        control_val="n",
        test_val="y",
        metric="fraction_in_chamber",
        n_permutations=args.n_permutations,
    )

    print("\nTREND TEST — chamber fraction slope per fly, hour 1")
    chamber_trend_h1 = trend_test_per_group(
        chamber_df.dropna(subset=["fraction_in_chamber"]),
        focus_hours=args.focus_hours,
        metric="fraction_in_chamber",
        n_permutations=args.n_permutations,
    )
    print("TREND TEST — chamber fraction slope per fly, hour 2")
    chamber_trend_h2 = trend_test_per_group(
        chamber_df.dropna(subset=["fraction_in_chamber"]),
        start_hours=args.focus_hours,
        focus_hours=2.0,
        metric="fraction_in_chamber",
        n_permutations=args.n_permutations,
    )

    print("\nGenerating time-in-chamber plots...")
    _generic_group_plot(
        df=chamber_df,
        y_col="fraction_in_chamber",
        ylabel="Fraction of time in chamber (r=50px)",
        title="Time spent near start position over time (MagnetBlock)",
        between_results=chamber_between,
        trend_h1=chamber_trend_h1,
        trend_h2=chamber_trend_h2,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "time_in_chamber_over_time",
    )
    _generic_individual_plot(
        df=chamber_df,
        y_col="fraction_in_chamber",
        ylabel="Fraction of time in chamber (r=50px)",
        title="Time in chamber — Magnet=y individual flies",
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        output_path=output_dir / "time_in_chamber_individual_flies",
    )
    plot_boxplot_per_bin(
        df=chamber_df,
        y_col="fraction_in_chamber",
        ylabel="Fraction of time in chamber (r=50px)",
        title="Time in chamber per bin — individual fly distribution (MagnetBlock)",
        focus_hours=args.focus_hours,
        output_path=output_dir / "chamber_boxplot_per_bin",
    )

    # --- Summary dashboard (all metrics) ------------------------------------
    print("\nGenerating summary dashboard…")
    plot_summary_dashboard(
        rate_df=rate_df,
        dur_df=dur_df,
        prox_df=prox_df,
        iei_df=iei_df,
        disp_df=disp_df,
        vel_df=vel_df,
        chamber_df=chamber_df,
        rate_between=between_results,
        dur_between=dur_between,
        prox_between=prox_between,
        iei_between=iei_between,
        disp_between=disp_between,
        vel_between=vel_between,
        chamber_between=chamber_between,
        focus_hours=args.focus_hours,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "summary_dashboard",
    )

    # ------------------------------------------------------------------ #
    # Event-rank analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("EVENT-RANK ANALYSIS")
    print("=" * 60)

    print("\nComputing per-fly metrics by sequential event number...")
    event_rank_df = compute_metrics_by_event_rank(
        df,
        fps=FPS,
        focus_hours=args.focus_hours,
        event_disp_df=event_disp_df,
        vel_event_df=vel_event_df,
    )

    print("\nGenerating event-rank overview plots (pre / post phase)...")
    plot_event_rank_overview(
        event_rank_df=event_rank_df,
        max_rank=args.max_rank,
        focus_hours=args.focus_hours,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
