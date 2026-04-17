#!/usr/bin/env python3
"""
Analyze interaction rates around the first significant push for wildtype
(starved / no-water) control flies.

Companion script to plot_magnetblock_interaction_rate.py.  Because these are
open-arena animals with no gate-opening protocol the **full dataset** is used
with no hour-2 restriction.

Analyses:
  1. Interaction rate over time (mean ± 95 % CI) + per-fly slope trend test
  2. Individual-fly spaghetti plot
  3. Interaction rate aligned to first significant push (5 px threshold)
     – windows ±10, ±20, ±30 min
  4. Pre/post push rate change (paired boxplot + delta distribution)

Usage:
    python plot_wildtype_push_rate.py
    python plot_wildtype_push_rate.py --bin-size 1 --n-permutations 10000
    python plot_wildtype_push_rate.py --n-flies 53  # sample a subset
    python plot_wildtype_push_rate.py --coordinates-dir /path/to/dir \\
        --filter FeedingState=starved_noWater --filter Light=on
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global matplotlib styling
# ---------------------------------------------------------------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Acquisition frame rate (frames per second)
FPS = 29
# Pixel → mm conversion
PIXELS_PER_MM = 500 / 30

# Default paths
DEFAULT_COORDINATES_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets"
    "/260220_10_summary_control_folders_Data/coordinates"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/wildtype_push_rate"
)

# Visual identity for this single group
GROUP_COLOR = "#8DA0CB"  # starved_noWater blue
GROUP_LABEL = "starved_noWater"


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


def load_wildtype_data(
    coordinates_dir: Path,
    filters: dict[str, str],
    n_flies: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load all *_coordinates.feather files in *coordinates_dir*, filter by
    column values, and return a concatenated DataFrame.

    If *n_flies* == 0 (default) **all** matching flies are used.
    Otherwise a random sample of *n_flies* is drawn.

    A synthetic ``Magnet`` column (value = 'wildtype') is added so that
    shared analysis functions work without modification.
    """
    feather_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in {coordinates_dir}")

    print(f"Found {len(feather_files)} feather file(s) in {coordinates_dir}")

    chunks = []
    for fp in feather_files:
        df = pd.read_feather(fp)
        mask = pd.Series(True, index=df.index)
        for col, val in filters.items():
            if col not in df.columns:
                print(f"  Warning: column '{col}' missing in {fp.name} — skipping filter")
                continue
            mask &= df[col].astype(str) == str(val)
        matching = df[mask]
        if not matching.empty:
            chunks.append(matching)

    if not chunks:
        raise ValueError(f"No rows matched filters {filters} across all feather files.")

    combined = pd.concat(chunks, ignore_index=True)

    all_flies = list(combined["fly"].dropna().unique())
    print(f"Flies matching filters: {len(all_flies)}")

    if n_flies == 0 or n_flies >= len(all_flies):
        if n_flies != 0 and n_flies > len(all_flies):
            print(f"Warning: only {len(all_flies)} flies available — using all.")
        sampled_flies = all_flies
    else:
        rng = random.Random(seed)
        sampled_flies = rng.sample(all_flies, n_flies)

    print(f"Using {len(sampled_flies)} flies.")
    data = combined[combined["fly"].isin(sampled_flies)].copy()

    # Add synthetic Magnet column so shared analysis functions work unchanged
    data["Magnet"] = "wildtype"

    required = ["time", "interaction_event_onset", "fly"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns after loading: {missing}")

    print(f"  Shape: {data.shape}")
    print(f"  Unique flies: {data['fly'].nunique()}")
    print(f"  Time range: {data['time'].min():.0f} – {data['time'].max():.0f} s "
          f"({data['time'].max() / 60:.1f} min)")
    return data


# ---------------------------------------------------------------------------
# Event extraction & rate binning   (identical logic to magnetblock script)
# ---------------------------------------------------------------------------


def extract_event_onsets(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique interaction event per fly (first frame only)."""
    non_nan = df[df["interaction_event_onset"].notna()].copy()
    onsets = (
        non_nan.sort_values("time")
        .groupby(["fly", "interaction_event_onset"], sort=False)
        .first()
        .reset_index()
    ).copy()
    onsets["time_min"] = onsets["time"] / 60.0
    print(f"\nTotal unique interaction events across all flies: {len(onsets)}")
    return onsets


def compute_rates_per_fly(
    onsets: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = 1.0,
) -> pd.DataFrame:
    """
    Count interactions per time bin per fly and normalise to events / min.
    Returns DataFrame with columns [fly, Magnet, bin_idx, bin_start_min,
    bin_center_min, event_count, rate_per_min].
    """
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "Magnet"]].set_index("fly")
    rows = []

    for fly, fly_onsets in onsets.groupby("fly"):
        magnet = fly_meta.loc[fly, "Magnet"]
        for i in range(n_bins):
            b_start, b_end = bin_edges[i], bin_edges[i + 1]
            count = ((fly_onsets["time_min"] >= b_start) & (fly_onsets["time_min"] < b_end)).sum()
            rows.append(dict(
                fly=fly, Magnet=magnet, bin_idx=i,
                bin_start_min=b_start, bin_center_min=bin_centers[i],
                event_count=int(count), rate_per_min=count / bin_size_min,
            ))

    # Flies with no events at all get zeros
    flies_with_onsets = set(onsets["fly"].unique())
    for fly in set(df_full["fly"].unique()) - flies_with_onsets:
        magnet = fly_meta.loc[fly, "Magnet"]
        for i in range(n_bins):
            rows.append(dict(
                fly=fly, Magnet=magnet, bin_idx=i,
                bin_start_min=bin_edges[i], bin_center_min=bin_centers[i],
                event_count=0, rate_per_min=0.0,
            ))

    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def trend_test_single_group(
    rate_df: pd.DataFrame,
    metric: str = "rate_per_min",
    n_permutations: int = 10000,
) -> dict:
    """
    Fit per-fly linear regression of rate ~ time_bin over the full dataset.
    Test whether the distribution of slopes differs from zero (sign permutation).
    """
    slopes = []
    for fly, fdf in rate_df.groupby("fly"):
        x = fdf["bin_center_min"].values
        y = fdf[metric].values
        if len(x) < 2:
            continue
        slope, *_ = stats.linregress(x, y)
        slopes.append(slope)

    slopes = np.array(slopes)
    obs_mean = float(np.mean(slopes))

    perm_means = np.empty(n_permutations)
    rng = np.random.default_rng(42)
    for j in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(slopes))
        perm_means[j] = np.mean(signs * np.abs(slopes))

    p_value = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))
    direction = "INCREASING" if obs_mean >= 0 else "DECREASING"
    print(f"  Trend: mean slope = {obs_mean:.5f} events/min²  [{direction}]"
          f"  p = {format_p_value(p_value, n_permutations)}")
    return {
        "slopes": slopes, "mean_slope": obs_mean,
        "median_slope": float(np.median(slopes)),
        "sem_slope": float(np.std(slopes, ddof=1) / np.sqrt(len(slopes))),
        "p_value": p_value, "n_flies": len(slopes),
    }


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng) -> float:
    """Two-sided sign-permutation test of H0: mean == 0."""
    obs = np.abs(np.mean(scores))
    count = sum(
        np.abs(np.mean(scores * rng.choice([-1, 1], size=len(scores)))) >= obs
        for _ in range(n_permutations)
    )
    return count / n_permutations


def test_post_push_change_single(
    change_df: pd.DataFrame,
    n_permutations: int = 10000,
) -> dict:
    """One-sample sign-permutation test: is mean delta != 0?"""
    rng = np.random.default_rng(0)
    scores = change_df["delta"].dropna().values
    if len(scores) < 3:
        return {"p_value": np.nan, "obs_mean": np.nan, "n": len(scores)}
    obs_mean = float(np.mean(scores))
    p = _one_sample_sign_perm(scores, n_permutations, rng)
    direction = "↑" if obs_mean >= 0 else "↓"
    print(f"  mean delta = {obs_mean:+.4f}  {direction}  p = {format_p_value(p, n_permutations)}")
    return {"p_value": p, "obs_mean": obs_mean, "n": len(scores)}


# ---------------------------------------------------------------------------
# Ball displacement & first-push detection  (ported from magnetblock script)
# ---------------------------------------------------------------------------


def compute_ball_displacement_per_event(
    df: pd.DataFrame,
    fps: float = FPS,
) -> pd.DataFrame:
    """Per interaction event: net displacement, max displacement, onset time."""
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
        (agg["ball_x_end"] - agg["ball_x_start"]) ** 2
        + (agg["ball_y_end"] - agg["ball_y_start"]) ** 2
    )
    agg["duration_s"] = agg["duration_frames"] / fps
    agg["onset_time_min"] = agg["onset_time_s"] / 60.0

    # Max displacement from event start
    event_rows = event_rows.merge(
        agg[["fly", "interaction_event", "ball_x_start", "ball_y_start"]],
        on=["fly", "interaction_event"], how="left",
    )
    event_rows["dist_from_start"] = np.sqrt(
        (event_rows["x_ball_0"] - event_rows["ball_x_start"]) ** 2
        + (event_rows["y_ball_0"] - event_rows["ball_y_start"]) ** 2
    )
    max_disp = (
        event_rows.groupby(["fly", "interaction_event"])["dist_from_start"]
        .max().reset_index()
        .rename(columns={"dist_from_start": "max_displacement_px"})
    )
    result = agg.merge(max_disp, on=["fly", "interaction_event"])

    n_total = len(result)
    n_over_5 = (result["max_displacement_px"] > 5).sum()
    print(f"\nBall displacement: {n_total} events")
    print(f"  Mean max displacement: {result['max_displacement_px'].mean():.2f} px")
    print(f"  Events >5 px (max):    {n_over_5} ({100 * n_over_5 / n_total:.1f}%)")
    return result[["fly", "Magnet", "interaction_event", "onset_time_s",
                   "onset_time_min", "duration_s", "displacement_px",
                   "max_displacement_px"]].reset_index(drop=True)


def compute_first_successful_push(
    event_disp_df: pd.DataFrame,
    threshold_px: float = 5.0,
) -> pd.DataFrame:
    """
    For each fly find the first event with max_displacement_px > threshold_px.
    Uses the full dataset (no hour-2 restriction).
    Returns one row per qualifying fly: [fly, Magnet, t_first_push_s,
    t_first_push_min, first_push_event_id].
    """
    successful = event_disp_df[event_disp_df["max_displacement_px"] > threshold_px].copy()
    successful = successful.sort_values(["fly", "onset_time_s"])
    first_push = successful.groupby("fly").first().reset_index()
    first_push = first_push.rename(columns={
        "onset_time_s": "t_first_push_s",
        "onset_time_min": "t_first_push_min",
        "interaction_event": "first_push_event_id",
    })
    n = len(first_push)
    print(f"\nFirst successful push (>{threshold_px}px, full dataset): {n} flies")
    if n:
        print(f"  Median time of first push: {first_push['t_first_push_min'].median():.1f} min")
    return first_push[["fly", "Magnet", "t_first_push_s", "t_first_push_min", "first_push_event_id"]]


# ---------------------------------------------------------------------------
# Pre/post rate change computation
# ---------------------------------------------------------------------------


def compute_post_push_rate_change(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
) -> pd.DataFrame:
    """
    Per fly: mean rate in a pre- and post-push window aligned to first push.
    No hour-2 restriction (full dataset).
    Returns [fly, Magnet, t_first_push_min, pre_mean, post_mean, delta,
             n_pre_bins, n_post_bins].
    """
    all_bins = np.sort(rate_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 1.0

    rows = []
    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        magnet = push_row["Magnet"]
        t_push = push_row["t_first_push_min"]

        fly_data = rate_df[rate_df["fly"] == fly].dropna(subset=["rate_per_min"]).copy()
        fly_data["rel_bin"] = ((fly_data["bin_center_min"] - t_push) / bin_size_min).round() * bin_size_min

        pre_far, pre_near = -pre_window_min[0], -pre_window_min[1]
        pre_data = fly_data[(fly_data["rel_bin"] >= pre_far) & (fly_data["rel_bin"] <= pre_near)]

        post_near, post_far = post_window_min[0], post_window_min[1]
        post_data = fly_data[(fly_data["rel_bin"] >= post_near) & (fly_data["rel_bin"] <= post_far)]

        if pre_data.empty or post_data.empty:
            continue

        pre_mean = float(pre_data["rate_per_min"].mean())
        post_mean = float(post_data["rate_per_min"].mean())
        rows.append(dict(
            fly=fly, Magnet=magnet, t_first_push_min=t_push,
            pre_mean=pre_mean, post_mean=post_mean,
            delta=post_mean - pre_mean,
            n_pre_bins=len(pre_data), n_post_bins=len(post_data),
        ))

    result = pd.DataFrame(rows)
    if result.empty:
        print("  No flies had sufficient pre- and post-push data — returning empty DataFrame.")
    else:
        print(f"  n flies with pre+post data: {len(result)},  mean delta = {result['delta'].mean():.3f} events/min")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rate_over_time(
    rate_df: pd.DataFrame,
    trend_result: dict,
    bin_size_min: float,
    n_permutations: int,
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Top : mean ± 95 % CI interaction rate over time.
      Bottom : per-fly slope distribution (violin + points).
    """
    n_flies = rate_df["fly"].nunique()

    box_props = dict(linewidth=1.5, edgecolor="black")
    whisker_props = dict(linewidth=1.5, color="black")
    cap_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2, color="black")

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.45)

    # ── Top: rate over time ───────────────────────────────────────────────
    ax_rate = fig.add_subplot(gs[0])
    sns.lineplot(
        data=rate_df, x="bin_center_min", y="rate_per_min",
        color=GROUP_COLOR, errorbar=("ci", 95), ax=ax_rate,
        label=f"{GROUP_LABEL} (n={n_flies})",
    )
    ax_rate.set_xlabel("Time (min)", fontsize=12)
    ax_rate.set_ylabel("Interaction rate (events / min)", fontsize=12)
    ax_rate.set_title("Interaction rate over time — wildtype starved animals", fontsize=13)
    ax_rate.legend(loc="upper right", fontsize=10)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["left"].set_linewidth(1.5)
    ax_rate.spines["bottom"].set_linewidth(1.5)
    ax_rate.tick_params(direction="out", length=4, width=1.5)
    ax_rate.set_facecolor("white")
    ax_rate.grid(False)

    # ── Bottom: slope violin ──────────────────────────────────────────────
    ax_slope = fig.add_subplot(gs[1])
    slopes = trend_result["slopes"]
    bp = ax_slope.boxplot(
        [slopes], positions=[0], patch_artist=True, widths=0.5,
        boxprops=box_props, whiskerprops=whisker_props,
        capprops=cap_props, medianprops=median_props, showfliers=False,
    )
    bp["boxes"][0].set_facecolor("none")
    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.10, 0.10, size=len(slopes))
    ax_slope.scatter(jit, slopes, s=25, color=GROUP_COLOR, alpha=0.6, zorder=3, edgecolors="none")
    ax_slope.axhline(0, color="black", lw=1, linestyle="--", alpha=0.7)

    p = trend_result["p_value"]
    stars = sig_stars(p)
    y_top = max(slopes.max(), 0) + 0.02 * max(abs(slopes.max()), abs(slopes.min()), 1e-9)
    ax_slope.text(
        0, y_top, f"{stars}\n{format_p_value(p, n_permutations)}",
        ha="center", va="bottom", fontsize=10,
        color="red" if p < 0.05 else "gray",
    )
    ax_slope.set_xticks([0])
    ax_slope.set_xticklabels([f"{GROUP_LABEL}\n(n={len(slopes)})"])
    ax_slope.set_ylabel("Slope (events/min²)", fontsize=10)
    ax_slope.set_title("Per-fly trend slope (full recording)", fontsize=11)
    ax_slope.spines["top"].set_visible(False)
    ax_slope.spines["right"].set_visible(False)
    ax_slope.spines["left"].set_linewidth(1.5)
    ax_slope.spines["bottom"].set_linewidth(1.5)
    ax_slope.tick_params(direction="out", length=4, width=1.5)
    ax_slope.set_facecolor("white")
    ax_slope.grid(False)

    fig.suptitle("Wildtype — Interaction Rate Analysis", fontsize=14, fontweight="bold", y=1.01)
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_path.with_suffix(suffix), dpi=200, bbox_inches="tight")
        print(f"  Saved: {output_path.with_suffix(suffix)}")
    plt.close(fig)


def plot_individual_rate(
    rate_df: pd.DataFrame,
    bin_size_min: float,
    output_path: Path,
) -> None:
    """Individual-fly rate lines (semi-transparent) with bold mean on top."""
    n_flies = rate_df["fly"].nunique()
    fig, ax = plt.subplots(figsize=(13, 5))

    for fly, fdf in rate_df.groupby("fly"):
        fd = fdf.dropna(subset=["rate_per_min"]).sort_values("bin_center_min")
        ax.plot(fd["bin_center_min"], fd["rate_per_min"],
                color=GROUP_COLOR, alpha=0.25, lw=0.8)

    mean_line = rate_df.groupby("bin_center_min")["rate_per_min"].mean().reset_index()
    ax.plot(mean_line["bin_center_min"], mean_line["rate_per_min"],
            color="black", lw=2.5, zorder=10, label=f"Mean (n={n_flies})")

    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel("Interaction rate (events / min)", fontsize=11)
    ax.set_title(f"Individual fly interaction rates — {GROUP_LABEL} (n={n_flies})", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(direction="out", length=4, width=1.5)
    ax.set_facecolor("white")
    ax.grid(False)

    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_path.with_suffix(suffix), dpi=200, bbox_inches="tight")
        print(f"  Saved: {output_path.with_suffix(suffix)}")
    plt.close(fig)


def plot_aligned_to_first_push(
    rate_df: pd.DataFrame,
    push_datasets: list,
    bin_size_min: float = 1.0,
    window_min: float = 10.0,
    output_path: Path = Path("aligned_to_first_push"),
) -> None:
    """
    Align flies to their first significant push and plot interaction rate
    vs. relative time.  Single column (no Magnet split).

    push_datasets : list of (first_push_df, threshold_px, row_label) tuples.
                    Each tuple produces one row.
    x-axis: ±window_min around first push.
    """
    n_rows = len(push_datasets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 5 * n_rows), squeeze=False)
    fig.subplots_adjust(hspace=0.55)

    for row_idx, (first_push_df, threshold_px, row_label) in enumerate(push_datasets):
        ax = axes[row_idx][0]
        flies = set(first_push_df["fly"])
        n_flies = len(flies)

        sub = rate_df[rate_df["fly"].isin(flies)].copy()
        sub = sub.merge(first_push_df[["fly", "t_first_push_min"]], on="fly", how="left")
        sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
        sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
        aligned = sub[sub["rel_bin"].abs() <= window_min]

        if not aligned.empty:
            bs_rng = np.random.default_rng(42)
            bins_sorted = np.sort(aligned["rel_bin"].unique())
            n_per_bin = aligned.groupby("rel_bin")["fly"].nunique()
            n_label = int(n_per_bin.max()) if not n_per_bin.empty else n_flies
            means, ci_lo, ci_hi = [], [], []
            for rb in bins_sorted:
                vals = aligned.loc[aligned["rel_bin"] == rb, "rate_per_min"].values
                if len(vals) == 0:
                    means.append(np.nan)
                    ci_lo.append(np.nan)
                    ci_hi.append(np.nan)
                    continue
                boot = np.array([
                    bs_rng.choice(vals, size=len(vals), replace=True).mean()
                    for _ in range(1000)
                ])
                means.append(float(np.mean(vals)))
                ci_lo.append(float(np.percentile(boot, 2.5)))
                ci_hi.append(float(np.percentile(boot, 97.5)))
            means = np.array(means)
            ci_lo = np.array(ci_lo)
            ci_hi = np.array(ci_hi)
            ax.fill_between(bins_sorted, ci_lo, ci_hi, color=GROUP_COLOR, alpha=0.30, zorder=5)
            ax.plot(bins_sorted, means, color=GROUP_COLOR, lw=2.5, zorder=10,
                    label=f"Mean ± 95% CI (n≤{n_label})")

        ax.axvline(0, color="red", linestyle="--", lw=2, alpha=0.9, label="First significant push")
        ax.set_xlim(-window_min, window_min)
        ax.set_xlabel("Time relative to first push (min)", fontsize=10)
        ax.set_ylabel("Interaction rate (events / min)", fontsize=10)
        ax.set_title(f"{row_label} — {GROUP_LABEL} (n={n_flies})", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(direction="out", length=4, width=1.5)
        ax.set_facecolor("white")
        ax.grid(False)

    fig.suptitle(
        f"Interaction rate aligned to first significant push\n"
        f"(window ±{int(window_min)} min — full recording)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_path.with_suffix(suffix), dpi=200, bbox_inches="tight")
        print(f"  Saved: {output_path.with_suffix(suffix)}")
    plt.close(fig)


def plot_post_push_rate_change(
    change_df: pd.DataFrame,
    stats: dict,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
    n_permutations: int = 10000,
    output_path: Path = Path("post_push_rate_change"),
) -> None:
    """
    Two-panel figure:
      Left  : paired pre/post boxplot + jittered dots + linking lines.
      Right : delta (post − pre) distribution.
    Styling matches run_permutations_magnetblock.py.
    """
    rng = np.random.default_rng(42)

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    box_props = dict(linewidth=1.5, edgecolor="black")
    whisker_props = dict(linewidth=1.5, color="black")
    cap_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2, color="black")

    sub = change_df.dropna(subset=["pre_mean", "post_mean"])
    pre_vals = sub["pre_mean"].values
    post_vals = sub["post_mean"].values
    n = len(sub)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.subplots_adjust(wspace=0.45, top=0.82)

    # ── Left: paired pre/post ─────────────────────────────────────────────
    ax0 = axes[0]
    bp = ax0.boxplot(
        [pre_vals, post_vals], positions=[0, 1], patch_artist=True, widths=0.5,
        boxprops=box_props, whiskerprops=whisker_props,
        capprops=cap_props, medianprops=median_props, showfliers=False,
    )
    bp["boxes"][0].set_facecolor("none")
    bp["boxes"][1].set_facecolor("none")

    jitter = rng.uniform(-0.08, 0.08, size=n)
    for pv, sv, jit in zip(pre_vals, post_vals, jitter):
        ax0.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8, zorder=2)
    ax0.scatter(0 + jitter, pre_vals, s=25, color=GROUP_COLOR, alpha=0.6, zorder=3, edgecolors="none")
    ax0.scatter(1 + jitter, post_vals, s=25, color=GROUP_COLOR, alpha=0.6, zorder=3, edgecolors="none")

    y_max0 = max(pre_vals.max(), post_vals.max()) if n > 0 else 1.0
    y_min0 = min(pre_vals.min(), post_vals.min()) if n > 0 else 0.0
    y_rng0 = y_max0 - y_min0 if y_max0 != y_min0 else 1.0
    annot_y0 = y_max0 + 0.12 * y_rng0

    p = stats.get("p_value", np.nan)
    stars = sig_stars(p) if not np.isnan(p) else "n.d."
    ax0.annotate("", xy=(1, annot_y0), xytext=(0, annot_y0),
                 arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    star_color = "red" if (not np.isnan(p) and p < 0.05) else "gray"
    ax0.text(0.5, annot_y0 + 0.02 * y_rng0, stars, ha="center", va="bottom",
             fontsize=12, color=star_color,
             fontweight="bold" if stars not in ("ns", "n.d.") else "normal")
    p_text = f"p = {format_p_value(p, n_permutations)}" if not np.isnan(p) else "p = n.d."
    ax0.text(0.02, 0.98, p_text, transform=ax0.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="left",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1))

    ax0.set_xticks([0, 1])
    ax0.set_xticklabels([pre_label, post_label], fontsize=10)
    ax0.set_xlim(-0.6, 1.6)
    ax0.set_ylim(y_min0 - 0.05 * y_rng0, y_max0 + 0.30 * y_rng0)
    ax0.set_ylabel("Interaction rate (events / min)", fontsize=11)
    ax0.set_title(f"{GROUP_LABEL}\n(n={n})", fontsize=11)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_linewidth(1.5)
    ax0.spines["bottom"].set_linewidth(1.5)
    ax0.tick_params(direction="out", length=4, width=1.5)
    ax0.set_facecolor("white")
    ax0.grid(False)

    # ── Right: delta distribution ─────────────────────────────────────────
    ax1 = axes[1]
    delta_vals = change_df["delta"].dropna().values
    n_d = len(delta_vals)
    bp_d = ax1.boxplot(
        [delta_vals], positions=[0], patch_artist=True, widths=0.5,
        boxprops=box_props, whiskerprops=whisker_props,
        capprops=cap_props, medianprops=median_props, showfliers=False,
    )
    bp_d["boxes"][0].set_facecolor("none")
    jit_d = rng.uniform(-0.10, 0.10, size=n_d)
    ax1.scatter(jit_d, delta_vals, s=25, color=GROUP_COLOR, alpha=0.6, zorder=3, edgecolors="none")
    ax1.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    d_top = float(np.nanmax(delta_vals)) if n_d else 1.0
    d_bot = float(np.nanmin(delta_vals)) if n_d else -1.0
    d_rng = d_top - d_bot if d_top != d_bot else 1.0
    annot_yd = d_top + 0.12 * d_rng
    ax1.set_ylim(d_bot - 0.05 * d_rng, d_top + 0.30 * d_rng)

    p_d = stats.get("p_value", np.nan)
    stars_d = sig_stars(p_d) if not np.isnan(p_d) else "n.d."
    ax1.text(0, annot_yd, stars_d, ha="center", va="bottom",
             fontsize=12, color=("red" if (not np.isnan(p_d) and p_d < 0.05) else "gray"),
             fontweight="bold" if stars_d not in ("ns", "n.d.") else "normal")
    p_text_d = f"p = {format_p_value(p_d, n_permutations)}" if not np.isnan(p_d) else "p = n.d."
    ax1.text(0.02, 0.98, p_text_d, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="left",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1))

    ax1.set_xticks([0])
    ax1.set_xticklabels([f"{GROUP_LABEL}\n(n={n_d})"], fontsize=10)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylabel("Δ rate (post − pre, events/min)", fontsize=11)
    ax1.set_title("Rate change (Δ post − pre)", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.tick_params(direction="out", length=4, width=1.5)
    ax1.set_facecolor("white")
    ax1.grid(False)

    pre_str = f"-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min"
    post_str = f"+{int(post_window_min[0])} to +{int(post_window_min[1])} min"
    fig.suptitle(
        f"Post-push interaction rate change (pre: {pre_str}, post: {post_str})\n"
        f"{GROUP_LABEL} — full recording",
        fontsize=12, fontweight="bold", y=0.98,
    )
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_path.with_suffix(suffix), dpi=200, bbox_inches="tight")
        print(f"  Saved: {output_path.with_suffix(suffix)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interaction rate around first significant push — wildtype control animals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coordinates-dir", type=Path, default=DEFAULT_COORDINATES_DIR,
        help=f"Directory of *_coordinates.feather files (default: {DEFAULT_COORDINATES_DIR})",
    )
    parser.add_argument(
        "--filter", action="append", dest="filters", metavar="COL=VAL", default=[],
        help="Column filter as Key=Value (repeatable), e.g. --filter FeedingState=starved_noWater",
    )
    parser.add_argument(
        "--n-flies", type=int, default=0,
        help="Number of flies to sample (default: 0 = use all matching flies)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fly sampling (default: 42)",
    )
    parser.add_argument(
        "--bin-size", type=float, default=1.0,
        help="Time bin size in minutes (default: 1.0)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=10000,
        help="Permutations for statistical testing (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: DEFAULT_OUTPUT_DIR)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Parse filters ---------------------------------------------------
    filters: dict[str, str] = {}
    for item in args.filters:
        if "=" not in item:
            raise ValueError(f"Filter must be COL=VAL, got: {item!r}")
        col, val = item.split("=", 1)
        filters[col] = val

    # Apply default filters if none provided
    if not filters:
        filters = {"FeedingState": "starved_noWater", "Light": "on"}
        print(f"No --filter arguments given; using defaults: {filters}")

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ---- Load data -------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_wildtype_data(
        coordinates_dir=args.coordinates_dir,
        filters=filters,
        n_flies=args.n_flies,
        seed=args.seed,
    )

    # ---- Event extraction & rate binning ---------------------------------
    print("\n" + "=" * 60)
    print("INTERACTION RATE ANALYSIS")
    print("=" * 60)
    onsets = extract_event_onsets(df)
    rate_df = compute_rates_per_fly(onsets, df, bin_size_min=args.bin_size)
    print(f"  Rate table shape: {rate_df.shape}")

    # -- Overall rate over time + trend ------------------------------------
    print("\nRunning per-fly slope trend test...")
    trend_result = trend_test_single_group(rate_df, n_permutations=args.n_permutations)

    print("\nGenerating interaction rate plot...")
    plot_rate_over_time(
        rate_df=rate_df,
        trend_result=trend_result,
        bin_size_min=args.bin_size,
        n_permutations=args.n_permutations,
        output_path=output_dir / "interaction_rate_over_time",
    )

    print("\nGenerating individual-fly spaghetti plot...")
    plot_individual_rate(
        rate_df=rate_df,
        bin_size_min=args.bin_size,
        output_path=output_dir / "interaction_rate_individual",
    )

    # ---- Ball displacement & first-push alignment -----------------------
    print("\n" + "=" * 60)
    print("FIRST-PUSH ALIGNMENT ANALYSIS")
    print("=" * 60)

    event_disp_df = compute_ball_displacement_per_event(df, fps=FPS)

    first_push_df_5px = compute_first_successful_push(event_disp_df, threshold_px=5.0)
    first_push_df_20px = compute_first_successful_push(event_disp_df, threshold_px=20.0)

    # -- Aligned rate curves — 3 window widths ----------------------------
    print("\nGenerating aligned-to-first-push rate plots...")
    for window_min in [10.0, 20.0, 30.0]:
        win_tag = f"{int(window_min)}min"

        # 5 px only
        if len(first_push_df_5px) >= 3:
            plot_aligned_to_first_push(
                rate_df=rate_df,
                push_datasets=[(first_push_df_5px, 5.0, "Significant push (>5 px)")],
                bin_size_min=args.bin_size,
                window_min=window_min,
                output_path=output_dir / f"aligned_to_first_push_5px_{win_tag}",
            )

        # Combined 5px + 20px
        valid_rows = []
        if len(first_push_df_5px) >= 3:
            valid_rows.append((first_push_df_5px, 5.0, "Significant push (>5 px)"))
        if len(first_push_df_20px) >= 3:
            valid_rows.append((first_push_df_20px, 20.0, "Major push (>20 px)"))
        if valid_rows:
            plot_aligned_to_first_push(
                rate_df=rate_df,
                push_datasets=valid_rows,
                bin_size_min=args.bin_size,
                window_min=window_min,
                output_path=output_dir / f"aligned_to_first_push_combined_{win_tag}",
            )

    # ---- Pre/post rate change -------------------------------------------
    print("\n" + "=" * 60)
    print("PRE/POST PUSH RATE CHANGE ANALYSIS")
    print("=" * 60)

    if len(first_push_df_5px) >= 3:
        for pre_win, post_win, out_tag in [
            ((10.0, 1.0), (1.0, 10.0), "post_push_rate_change_10min"),
            ((5.0, 1.0), (1.0, 5.0), "post_push_rate_change_5min"),
        ]:
            pre_str = f"-{int(pre_win[0])} to -{int(pre_win[1])} min"
            post_str = f"+{int(post_win[0])} to +{int(post_win[1])} min"
            print(f"\nComputing pre/post rate change (pre: {pre_str}, post: {post_str})...")
            change_df = compute_post_push_rate_change(
                rate_df=rate_df,
                first_push_df=first_push_df_5px,
                pre_window_min=pre_win,
                post_window_min=post_win,
            )
            if len(change_df) >= 3:
                print("Running permutation test on change scores...")
                change_stats = test_post_push_change_single(change_df, n_permutations=args.n_permutations)
                print("Generating pre/post rate change plot...")
                plot_post_push_rate_change(
                    change_df=change_df,
                    stats=change_stats,
                    pre_window_min=pre_win,
                    post_window_min=post_win,
                    n_permutations=args.n_permutations,
                    output_path=output_dir / out_tag,
                )
            else:
                print(f"  Too few flies with both pre and post data for {out_tag} — skipping.")
    else:
        print("  Too few flies with a qualifying push (5 px) — skipping pre/post analysis.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
