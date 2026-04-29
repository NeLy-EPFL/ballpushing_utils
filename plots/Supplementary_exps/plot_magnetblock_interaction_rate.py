#!/usr/bin/env python3
"""
Script to analyze interaction rates over time for MagnetBlock experiments.

Analyses:

1. **Interaction rate** (events / min per 5-min bin)
   — how often does the fly visit the ball?

Includes:
- Time bins of configurable width (default 5 min) over the full 120 min.
- A between-group permutation test (Magnet=y vs Magnet=n) per bin.
- A per-fly linear-regression trend test over the first hour
  (one-sample permutation test of the slope distribution vs. zero).
- First successful push detection (hour 2) and aligned pre/post comparisons.

Key design choices:
- An "event" is counted once: the column carries the same integer for every frame of
  an event, so we deduplicate by (fly, event_id) and take the earliest frame as onset.
- For aligned-to-push and pre/post plots, only data from hour 2 (after focus_hours)
  is used, and the x-axis is constrained to ±10 min around the first push.

Usage:
    python plot_magnetblock_interaction_rate.py [--bin-size N] [--n-permutations N]
                                                [--output-dir PATH] [--focus-hours N]

Arguments:
    --bin-size         : Bin duration in minutes (default: 1)
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
from tqdm import tqdm
from ballpushing_utils import read_feather

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
    df = read_feather(dataset_path)
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
    bin_size_min: float = 5.0,
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
    sig = (results_df["p_value"] < alpha).sum()
    print(f"  Significant bins: {sig}/{len(bins)}")
    return results_df


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


# ---------------------------------------------------------------------------
# Ball displacement · First-push alignment
# ---------------------------------------------------------------------------


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
    push_datasets: list,
    bin_size_min: float = 0.5,
    window_min: float = 10.0,
    focus_hours: float = 1.0,
    output_path: Path = Path("aligned_to_first_push"),
) -> None:
    """
    Align flies to their first successful push in hour 2 and plot interaction rate
    as a function of time relative to that event.

    Parameters
    ----------
    push_datasets : list of (first_push_df, threshold_px, row_label) tuples.
                    Each tuple produces one row of panels (Magnet=y | Magnet=n).

    Only bins after focus_hours are used (hour 2 data only).
    Layout: n_rows × 2 columns (Magnet=y | Magnet=n).
    x-axis: minutes relative to first push, constrained to ±window_min.
    """
    palette = make_palette(["y", "n"])
    groups = [("n", "Magnet=n (control)", palette["n"]), ("y", "Magnet=y (blocked)", palette["y"])]
    cutoff_min = focus_hours * 60.0
    n_rows = len(push_datasets)

    _fig_w = 170 / 25.4  # 170 mm
    _fig_h = 89 / 25.4   # 89 mm per row
    fig, axes = plt.subplots(n_rows, 2, figsize=(_fig_w, _fig_h * n_rows), squeeze=False)
    fig.subplots_adjust(wspace=0.35, hspace=0.55)

    for row_idx, (first_push_df, threshold_px, row_label) in enumerate(push_datasets):
        for col_idx, (magnet_val, group_label, color) in enumerate(groups):
            push_sub = first_push_df[first_push_df["Magnet"] == magnet_val]
            flies = set(push_sub["fly"])
            n_flies = len(flies)
            ax = axes[row_idx][col_idx]

            # Filter to hour 2 only, then align to first push
            sub = rate_df[rate_df["fly"].isin(flies)].copy()
            sub = sub[sub["bin_center_min"] >= cutoff_min]
            sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
            sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
            sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
            aligned = sub[sub["rel_bin"].abs() <= window_min]

            if not aligned.empty:
                n_per_bin = aligned.groupby("rel_bin")["fly"].nunique()
                n_label = int(n_per_bin.max()) if not n_per_bin.empty else n_flies
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
                ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25, zorder=5)
                ax.plot(bins_sorted, means, color=color, lw=2.5, zorder=10)

            ax.axvline(0, color="red", linestyle="--", lw=2, alpha=0.9)
            ax.set_xlim(-window_min, window_min)
            ax.set_ylabel("Interaction rate (events / min)" if col_idx == 0 else "", fontsize=11)
            col_title = (
                f"Access to immobile ball (n={n_flies})" if magnet_val == "y"
                else f"No access to ball (n={n_flies})"
            )
            ax.set_title(col_title, fontsize=11, color=color, fontweight="bold", pad=4)
            ax.set_xlabel("Time relative to first push (min)", fontsize=10)
            ax.tick_params(labelsize=9, width=1.5, length=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.set_facecolor("white")
            ax.grid(False)

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
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
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(color="black", alpha=0.8),
        capprops=dict(color="black", alpha=0.8),
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
    for suffix in [".pdf", ".png"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=200, bbox_inches="tight")
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

    For Magnet=y flies the pre-window is restricted to hour 2 to remove
    hour-1 baseline probing contamination. The same restriction is applied to
    Magnet=n flies for a symmetric comparison.

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
        # Restrict pre-window to hour 2 only for all groups (avoid contamination
        # from the barrier-active phase in blocked flies and ensure symmetric
        # comparison for control flies).
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
    if result.empty:
        print("  No flies had sufficient pre- and post-push data — returning empty DataFrame.")
        return result
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
        ("y", "Magnet=y (blocked)", palette["y"]),
        ("n", "Magnet=n (control)", palette["n"]),
    ]

    # Shared box styling — matching run_permutations_magnetblock.py
    box_props = dict(linewidth=1.5, edgecolor="black")
    whisker_props = dict(linewidth=1.5, color="black")
    cap_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2, color="black")

    # Compute shared y-limits across both Magnet groups for panels 0 & 1
    all_paired = change_df.dropna(subset=["pre_mean", "post_mean"])
    shared_y_min = min(all_paired["pre_mean"].min(), all_paired["post_mean"].min())
    shared_y_max = max(all_paired["pre_mean"].max(), all_paired["post_mean"].max())
    shared_y_rng = shared_y_max - shared_y_min
    # Reserve 30% headroom above the data for the significance annotation + suptitle gap
    shared_y_top = shared_y_max + 0.30 * shared_y_rng
    shared_y_bot = shared_y_min - 0.05 * shared_y_rng

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.subplots_adjust(wspace=0.40, top=0.82)

    # ── Panels 0 & 1: paired pre / post per group ─────────────────────────
    for ax, (grp, glabel, color) in zip(axes[:2], group_meta):
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["pre_mean", "post_mean"])
        pre_vals = sub["pre_mean"].values
        post_vals = sub["post_mean"].values
        n = len(sub)

        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=[0, 1],
            patch_artist=True,
            widths=0.5,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor("none")
        bp["boxes"][1].set_facecolor("none")

        # Jitter and linking lines
        jitter = rng.uniform(-0.08, 0.08, size=n)
        for pv, sv, jit in zip(pre_vals, post_vals, jitter):
            ax.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(0 + jitter, pre_vals, s=25, color=color, alpha=0.6, zorder=3, edgecolors="none")
        ax.scatter(1 + jitter, post_vals, s=25, color=color, alpha=0.6, zorder=3, edgecolors="none")

        # Significance bar + stars at a fixed shared height
        p = stats.get(grp, {}).get("p_value", np.nan)
        stars = sig_stars(p) if not np.isnan(p) else "n.d."
        annot_y = shared_y_max + 0.12 * shared_y_rng
        ax.annotate(
            "",
            xy=(1, annot_y),
            xytext=(0, annot_y),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )
        star_color = "red" if (not np.isnan(p) and p < 0.05) else "gray"
        ax.text(
            0.5,
            annot_y + 0.02 * shared_y_rng,
            stars,
            ha="center",
            va="bottom",
            fontsize=12,
            color=star_color,
            fontweight="bold" if stars not in ("ns", "n.d.") else "normal",
        )

        # p-value text in top-left corner
        p_text = f"p = {format_p_value(p, n_permutations)}" if not np.isnan(p) else "p = n.d."
        ax.text(
            0.02, 0.98, p_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([pre_label, post_label], fontsize=10)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(shared_y_bot, shared_y_top)
        ax.set_ylabel("Interaction rate (events / min)", fontsize=11)
        ax.set_title(f"{glabel}\n(n={n})", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)
        ax.set_facecolor("white")
        ax.grid(False)

    # ── Panel 2: delta distributions, both groups ─────────────────────────
    ax_d = axes[2]
    positions = {"y": 0, "n": 1}
    for grp, glabel, color in group_meta:
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["delta"])
        vals = sub["delta"].values
        pos = positions[grp]
        n = len(vals)
        bp_d = ax_d.boxplot(
            [vals],
            positions=[pos],
            patch_artist=True,
            widths=0.5,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            showfliers=False,
        )
        bp_d["boxes"][0].set_facecolor("none")
        jitter = rng.uniform(-0.10, 0.10, size=n)
        ax_d.scatter(pos + jitter, vals, s=25, color=color, alpha=0.6, zorder=3, edgecolors="none")

    ax_d.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    # Between-group annotation — compute limits first so annotation never overflows
    delta_vals = change_df["delta"].dropna().values
    d_top = float(np.nanmax(delta_vals)) if len(delta_vals) else 1.0
    d_bot = float(np.nanmin(delta_vals)) if len(delta_vals) else -1.0
    d_rng = d_top - d_bot if d_top != d_bot else 1.0
    annot_y_d = d_top + 0.12 * d_rng
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
        ax_d.text(
            0.5,
            annot_y_d + 0.02 * d_rng,
            stars_b,
            ha="center",
            va="bottom",
            fontsize=12,
            color=star_color,
            fontweight="bold" if stars_b != "ns" else "normal",
        )
        p_text_b = f"p = {format_p_value(p_b, n_permutations)}"
        ax_d.text(
            0.02, 0.98, p_text_b, transform=ax_d.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
        )

    n_y = len(change_df[change_df["Magnet"] == "y"]["delta"].dropna())
    n_n = len(change_df[change_df["Magnet"] == "n"]["delta"].dropna())
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"Blocked\n(n={n_y})", f"Control\n(n={n_n})"], fontsize=10)
    ax_d.set_xlim(-0.6, 1.6)
    ax_d.set_ylabel("Δ rate (post − pre, events/min)", fontsize=11)
    ax_d.set_title("Change score\n(between-group test)", fontsize=11)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_linewidth(1.5)
    ax_d.spines["bottom"].set_linewidth(1.5)
    ax_d.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)
    ax_d.set_facecolor("white")
    ax_d.grid(False)

    pre_str = f"-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min"
    post_str = f"+{int(post_window_min[0])} to +{int(post_window_min[1])} min"
    fig.suptitle(
        f"Post-push interaction rate change (pre: {pre_str}, post: {post_str})\n"
        f"Magnet=y pre-window restricted to hour 2 (>={int(focus_hours*60)} min)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    for suffix in [".pdf", ".png"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p_out}")
    plt.close(fig)


def plot_post_push_rate_change_blocked(
    change_df: pd.DataFrame,
    stats: dict,
    focus_hours: float = 1.0,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
    n_permutations: int = 10000,
    output_path: Path = Path("post_push_rate_change_blocked"),
) -> None:
    """
    Two-panel figure showing:
      Left  : Blocked group (Magnet=y) only — paired pre/post boxplot + linking lines
      Right  : Delta (post-pre) comparison — control (Magnet=n) vs blocked (Magnet=y)
    """
    palette = make_palette(["y", "n"])
    rng = np.random.default_rng(42)

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    # Shared box styling — matching run_permutations_magnetblock.py
    box_props = dict(linewidth=1.5, edgecolor="black")
    whisker_props = dict(linewidth=1.5, color="black")
    cap_props = dict(linewidth=1.5, color="black")
    median_props = dict(linewidth=2, color="black")

    fig, axes = plt.subplots(1, 2, figsize=(170 / 25.4, 89 / 25.4))
    fig.subplots_adjust(wspace=0.50)

    # ── Panel 0: Magnet=y pre/post paired ──────────────────────────────────
    ax0 = axes[0]
    color_y = palette["y"]
    sub_y = change_df[change_df["Magnet"] == "y"].dropna(subset=["pre_mean", "post_mean"])
    pre_vals = sub_y["pre_mean"].values
    post_vals = sub_y["post_mean"].values
    n_y = len(sub_y)

    bp = ax0.boxplot(
        [pre_vals, post_vals],
        positions=[0, 1],
        patch_artist=True,
        widths=0.5,
        boxprops=box_props,
        whiskerprops=whisker_props,
        capprops=cap_props,
        medianprops=median_props,
        showfliers=False,
    )
    bp["boxes"][0].set_facecolor("none")
    bp["boxes"][1].set_facecolor("none")

    jitter = rng.uniform(-0.08, 0.08, size=n_y)
    for pv, sv, jit in zip(pre_vals, post_vals, jitter):
        ax0.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8, zorder=2)
    ax0.scatter(0 + jitter, pre_vals, s=25, color=color_y, alpha=0.6, zorder=3, edgecolors="none")
    ax0.scatter(1 + jitter, post_vals, s=25, color=color_y, alpha=0.6, zorder=3, edgecolors="none")

    y_max0 = max(pre_vals.max(), post_vals.max()) if n_y > 0 else 1.0
    y_min0 = min(pre_vals.min(), post_vals.min()) if n_y > 0 else 0.0
    y_rng0 = y_max0 - y_min0 if y_max0 != y_min0 else 1.0
    annot_y0 = y_max0 + 0.12 * y_rng0

    p_y = stats.get("y", {}).get("p_value", np.nan)
    stars_y = sig_stars(p_y) if not np.isnan(p_y) else "n.d."
    ax0.annotate(
        "",
        xy=(1, annot_y0),
        xytext=(0, annot_y0),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
    )
    star_color_y = "red" if (not np.isnan(p_y) and p_y < 0.05) else "gray"
    ax0.text(
        0.5, annot_y0 + 0.02 * y_rng0, stars_y, ha="center", va="bottom",
        fontsize=12, color=star_color_y,
        fontweight="bold" if stars_y not in ("ns", "n.d.") else "normal",
    )

    ax0.set_xticks([0, 1])
    ax0.set_xticklabels([pre_label, post_label], fontsize=10)
    ax0.set_xlim(-0.6, 1.6)
    ax0.set_ylim(y_min0 - 0.05 * y_rng0, y_max0 + 0.30 * y_rng0)
    ax0.set_ylabel("Interaction rate (events / min)", fontsize=11)
    ax0.set_title(f"Access to immobile ball\n(n={n_y})", fontsize=11, color=palette["y"], fontweight="bold")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_linewidth(1.5)
    ax0.spines["bottom"].set_linewidth(1.5)
    ax0.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)
    ax0.set_facecolor("white")
    ax0.grid(False)

    # ── Panel 1: delta — control (pos 0) vs blocked (pos 1) ───────────────
    ax1 = axes[1]
    for pos, grp, color in [(0, "n", palette["n"]), (1, "y", palette["y"])]:
        sub = change_df[change_df["Magnet"] == grp].dropna(subset=["delta"])
        vals = sub["delta"].values
        n = len(vals)
        bp_d = ax1.boxplot(
            [vals],
            positions=[pos],
            patch_artist=True,
            widths=0.5,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            showfliers=False,
        )
        bp_d["boxes"][0].set_facecolor("none")
        jitter = rng.uniform(-0.10, 0.10, size=n)
        ax1.scatter(pos + jitter, vals, s=25, color=color, alpha=0.6, zorder=3, edgecolors="none")

    ax1.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    delta_vals = change_df["delta"].dropna().values
    d_top = float(np.nanmax(delta_vals)) if len(delta_vals) else 1.0
    d_bot = float(np.nanmin(delta_vals)) if len(delta_vals) else -1.0
    d_rng = d_top - d_bot if d_top != d_bot else 1.0
    annot_y_d = d_top + 0.12 * d_rng
    ax1.set_ylim(d_bot - 0.05 * d_rng, d_top + 0.30 * d_rng)

    p_b = stats.get("between", {}).get("p_value", np.nan)
    if not np.isnan(p_b):
        ax1.annotate(
            "",
            xy=(1, annot_y_d),
            xytext=(0, annot_y_d),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )
        stars_b = sig_stars(p_b)
        star_color = "red" if p_b < 0.05 else "gray"
        ax1.text(
            0.5, annot_y_d + 0.02 * d_rng, stars_b, ha="center", va="bottom",
            fontsize=12, color=star_color,
            fontweight="bold" if stars_b != "ns" else "normal",
        )

    n_n_d = len(change_df[change_df["Magnet"] == "n"]["delta"].dropna())
    n_y_d = len(change_df[change_df["Magnet"] == "y"]["delta"].dropna())
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(
        [f"No access\nto ball\n(n={n_n_d})", f"Access to\nimmobile ball\n(n={n_y_d})"],
        fontsize=10,
    )
    ax1.set_xlim(-0.6, 1.6)
    ax1.set_ylabel("Δ rate (post − pre, events/min)", fontsize=11)
    ax1.set_title("Pre-post change\ncomparison with control", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)
    ax1.set_facecolor("white")
    ax1.grid(False)

    for suffix in [".pdf", ".png"]:
        p_out = output_path.with_suffix(suffix)
        fig.savefig(p_out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p_out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Interaction rate analysis for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bin-size", type=float, default=1.0, help="Time bin size in minutes (default: 1)")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: next to dataset)")
    parser.add_argument("--focus-hours", type=float, default=1.0, help="Hours to use for trend analysis (default: 1.0)")
    args = parser.parse_args()

    dataset_path = (
        "/mnt/upramdya_data/MD/MagnetBlock/Datasets/"
        "260224_12_summary_magnet_block_folders_Data_Full/"
        "coordinates/pooled_coordinates.feather"
    )

    if args.output_dir is None:
        output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock")/ "Plots" / "interaction_rate"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAGNETBLOCK — INTERACTION RATE ANALYSIS")
    print("=" * 60)
    print(f"Bin size     : {args.bin_size} min")
    print(f"Permutations : {args.n_permutations}")
    print(f"Trend window : first {int(args.focus_hours * 60)} min")
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

    # ------------------------------------------------------------------ #
    # First successful push + alignment analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FIRST SUCCESSFUL PUSH & ALIGNMENT ANALYSIS")
    print("=" * 60)

    print("\nComputing ball displacement per event...")
    event_disp_df = compute_ball_displacement_per_event(df, fps=FPS)

    first_push_df_5px = compute_first_successful_push(event_disp_df, focus_hours=args.focus_hours, threshold_px=5.0)
    first_push_df_20px = compute_first_successful_push(event_disp_df, focus_hours=args.focus_hours, threshold_px=20.0)

    # Individual threshold plots (5 px only) — three window widths
    if len(first_push_df_5px) >= 3:
        print("\nGenerating alignment plots for threshold >5px...")
        for window_min in [10.0, 20.0, 30.0]:
            win_tag = f"{int(window_min)}min"
            plot_aligned_to_first_push(
                rate_df=rate_df,
                push_datasets=[(first_push_df_5px, 5.0, "Significant push (>5 px)")],
                bin_size_min=args.bin_size,
                window_min=window_min,
                focus_hours=args.focus_hours,
                output_path=output_dir / f"aligned_to_first_push_5px_{win_tag}",
            )
    else:
        print("  Too few flies with a >5px push — skipping alignment plots.")

    # Combined figure: significant push (≥5 px) on top, major push (≥20 px) below
    print("\nGenerating combined significant + major push alignment plots...")
    valid_rows = []
    if len(first_push_df_5px) >= 3:
        valid_rows.append((first_push_df_5px, 5.0, "Significant push (>5 px)"))
    if len(first_push_df_20px) >= 3:
        valid_rows.append((first_push_df_20px, 20.0, "Major push (>20 px)"))
    if valid_rows:
        for window_min in [10.0, 20.0, 30.0]:
            win_tag = f"{int(window_min)}min"
            plot_aligned_to_first_push(
                rate_df=rate_df,
                push_datasets=valid_rows,
                bin_size_min=args.bin_size,
                window_min=window_min,
                focus_hours=args.focus_hours,
                output_path=output_dir / f"aligned_to_first_push_combined_{win_tag}",
            )
    else:
        print("  Too few flies for combined alignment plot — skipping.")

    # ------------------------------------------------------------------ #
    # Post-push rate change analysis (5 px threshold)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("POST-PUSH RATE CHANGE ANALYSIS")
    print("=" * 60)

    # first_push_df_5px already computed above
    if len(first_push_df_5px) >= 3:
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
                print("\nGenerating blocked-only post-push rate change plot...")
                plot_post_push_rate_change_blocked(
                    change_df=change_df,
                    stats=change_stats,
                    focus_hours=args.focus_hours,
                    pre_window_min=pre_win,
                    post_window_min=post_win,
                    n_permutations=args.n_permutations,
                    output_path=output_dir / f"{out_tag}_blocked",
                )
            else:
                print(f"  Too few flies with both pre and post data for {out_tag} — skipping.")
    else:
        print("  Too few flies with a qualifying push — skipping.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
