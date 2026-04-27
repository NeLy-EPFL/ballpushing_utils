#!/usr/bin/env python3
"""Extended Data Figure 2 (d-g) - Interaction rates during ball immobility affordance experiments.

Panels:
    (d) Interaction rates for experimental animals (Magnet=y), centered on first significant push
    (e) Interaction rates for control animals (Magnet=n), centered on first significant push
    (f) Pre vs Post comparison for experimental animals only
    (g) Pre vs Post comparison for both groups (boxplots + delta distribution)

Methodology closely follows plot_magnetblock_interaction_rate.py:
  - Rates computed in absolute 1-min bins over the full recording
  - Alignment to first push done within plot functions (integer relative bins including 0)
  - Pre/post windows: +-10 min (excluding +-1 min gap)
  - Pre-window restricted to hour 2 (>60 min) for both groups
  - One-sample sign-permutation test on mean delta (not median)

Usage:
    python edfigure2_defg_interaction_rates.py [--test]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import set_illustrator_style

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"

DATASET_PATH = dataset(
    "MagnetBlock/Datasets/260224_12_summary_magnet_block_folders_Data_Full/coordinates/pooled_coordinates.feather"
)

N_PERMUTATIONS = 10_000
FPS = 29
PIXELS_PER_MM = 500 / 30
FOCUS_HOURS = 1.0
PUSH_THRESHOLD_PX = 5.0
BIN_SIZE_MIN = 1.0
WINDOW_MIN = 10.0
PRE_WINDOW = (10.0, 1.0)
POST_WINDOW = (1.0, 10.0)
COLORS = {"n": "#faa41a", "y": "#3953A4"}
LABELS = {"n": "No access to ball", "y": "Access to immobile ball"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(test_mode: bool = False) -> pd.DataFrame:
    print(f"\nLoading dataset from:\n  {DATASET_PATH}")
    df = pd.read_feather(DATASET_PATH)
    if test_mode:
        sample_flies = df["fly"].unique()[:10]
        df = df[df["fly"].isin(sample_flies)].copy()
        print(f"  Test mode: using {len(sample_flies)} flies")
    print(f"  Shape: {df.shape}")
    required = ["time", "interaction_event_onset", "interaction_event", "Magnet", "fly", "x_ball_0", "y_ball_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"  Magnet groups: {sorted(df['Magnet'].unique())}")
    for group in sorted(df["Magnet"].unique()):
        n = df[df["Magnet"] == group]["fly"].nunique()
        print(f"    {group}: {n} flies")
    return df


# ---------------------------------------------------------------------------
# Event extraction & binning
# ---------------------------------------------------------------------------


def extract_event_onsets(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique interaction event per fly (earliest frame)."""
    non_nan = df[df["interaction_event_onset"].notna()].copy()
    onsets = (
        non_nan.sort_values("time").groupby(["fly", "interaction_event_onset"], sort=False).first().reset_index()
    ).copy()
    onsets["time_min"] = onsets["time"] / 60.0
    print(f"\n  Total unique interaction events: {len(onsets)}")
    return onsets


def compute_rates_per_fly(
    onsets: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = 1.0,
) -> pd.DataFrame:
    """
    Compute per-fly interaction rates in absolute time bins.
    Mirrors original compute_rates_per_fly - bin_center_min is absolute time.
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
            count = int(((fly_onsets["time_min"] >= b_start) & (fly_onsets["time_min"] < b_end)).sum())
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=b_start,
                    bin_center_min=bin_centers[i],
                    event_count=count,
                    rate_per_min=count / bin_size_min,
                )
            )

    # Include flies with no events at all
    flies_with_onsets = set(onsets["fly"].unique())
    all_flies = set(df_full["fly"].unique())
    for fly in all_flies - flies_with_onsets:
        magnet = fly_meta.loc[fly, "Magnet"]
        for i in range(n_bins):
            rows.append(
                dict(
                    fly=fly,
                    Magnet=magnet,
                    bin_idx=i,
                    bin_start_min=bin_edges[i],
                    bin_center_min=bin_centers[i],
                    event_count=0,
                    rate_per_min=0.0,
                )
            )

    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ball displacement & first push detection
# ---------------------------------------------------------------------------


def compute_ball_displacement_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """Compute net and max displacement per interaction event."""
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
    agg["duration_s"] = agg["duration_frames"] / FPS
    agg["onset_time_min"] = agg["onset_time_s"] / 60.0

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
    print(f"  Event displacement data shape: {result.shape}")
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


def find_first_significant_push(
    event_disp_df: pd.DataFrame,
    focus_hours: float = 1.0,
    threshold_px: float = 5.0,
) -> pd.DataFrame:
    """Find the first event in hour 2 (onset_time_s > cutoff) with max_displacement > threshold.
    Uses strict > for cutoff to match original compute_first_successful_push."""
    cutoff_s = focus_hours * 3600.0
    hour2 = event_disp_df[event_disp_df["onset_time_s"] > cutoff_s].copy()
    significant = hour2[hour2["max_displacement_px"] > threshold_px].copy()
    significant = significant.sort_values(["fly", "onset_time_s"])
    first_push = significant.groupby("fly").first().reset_index()
    first_push = first_push.rename(
        columns={
            "onset_time_s": "t_first_push_s",
            "onset_time_min": "t_first_push_min",
        }
    )

    print(f"\n  Flies with first significant push (>{threshold_px}px, after hour {focus_hours}):")
    for group in sorted(first_push["Magnet"].unique()):
        n = len(first_push[first_push["Magnet"] == group])
        print(f"    {group}: {n} flies")

    return first_push[["fly", "Magnet", "t_first_push_s", "t_first_push_min"]].copy()


# ---------------------------------------------------------------------------
# Pre/post rate change (mirrors original compute_post_push_rate_change)
# ---------------------------------------------------------------------------


def compute_post_push_rate_change(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    focus_hours: float = 1.0,
    pre_window_min: tuple = (10.0, 1.0),
    post_window_min: tuple = (1.0, 10.0),
) -> pd.DataFrame:
    """
    Per fly: compute mean interaction rate in pre and post windows around the first push.
    rate_df has ABSOLUTE bin_center_min values.
    Mirrors original compute_post_push_rate_change exactly.
    """
    cutoff_min = focus_hours * 60.0
    all_bins = np.sort(rate_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 1.0

    rows = []
    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        magnet = push_row["Magnet"]
        t_push = push_row["t_first_push_min"]

        fly_data = rate_df[rate_df["fly"] == fly].dropna(subset=["rate_per_min"]).copy()

        # Snap to bin grid: rel_bin = round((abs_bin_center - t_push) / bin_size) * bin_size
        fly_data["rel_bin"] = ((fly_data["bin_center_min"] - t_push) / bin_size_min).round() * bin_size_min

        # Pre-window: e.g. rel_bin in [-10, -1]
        pre_far = -pre_window_min[0]
        pre_near = -pre_window_min[1]
        pre_data = fly_data[(fly_data["rel_bin"] >= pre_far) & (fly_data["rel_bin"] <= pre_near)]
        # Restrict pre-window to hour 2 (absolute bin_center_min >= cutoff)
        pre_data = pre_data[pre_data["bin_center_min"] >= cutoff_min]

        # Post-window: e.g. rel_bin in [+1, +10]
        post_near = post_window_min[0]
        post_far = post_window_min[1]
        post_data = fly_data[(fly_data["rel_bin"] >= post_near) & (fly_data["rel_bin"] <= post_far)]

        if pre_data.empty or post_data.empty:
            continue

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
        print("  WARNING: No flies had sufficient pre- and post-push data")
    return result


# ---------------------------------------------------------------------------
# Statistical testing (mirrors original _one_sample_sign_perm / test_post_push_change)
# ---------------------------------------------------------------------------


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng) -> float:
    """Two-sided sign-permutation test of H0: mean == 0. Exactly mirrors original."""
    obs = np.abs(np.mean(scores))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(scores))
        if np.abs(np.mean(scores * signs)) >= obs:
            count += 1
    return count / n_permutations


def test_post_push_change(change_df: pd.DataFrame, n_permutations: int = 10_000) -> dict:
    """
    1. One-sample sign-permutation test per group: is mean(delta) != 0?
    2. Two-sample permutation test between groups.
    Mirrors original test_post_push_change.

    Each test uses its own independently-seeded RNG so results are fully
    reproducible and do not depend on how many permutations previous tests consumed.
    """
    results = {}

    for group in ["y", "n"]:
        rng = np.random.default_rng({"y": 0, "n": 1}[group])
        scores = change_df[change_df["Magnet"] == group]["delta"].values
        if len(scores) < 3:
            results[group] = {"p_value": np.nan, "obs_mean": np.nan, "n": len(scores)}
            continue
        obs_mean = float(np.mean(scores))
        p = _one_sample_sign_perm(scores, n_permutations, rng)
        results[group] = {"p_value": p, "obs_mean": obs_mean, "n": len(scores)}
        direction = "up" if obs_mean >= 0 else "down"
        print(f"  [{group}] mean delta = {obs_mean:+.4f}  ({direction})  p = {p:.4f}")

    y_scores = change_df[change_df["Magnet"] == "y"]["delta"].values
    n_scores = change_df[change_df["Magnet"] == "n"]["delta"].values
    if len(y_scores) >= 3 and len(n_scores) >= 3:
        rng_between = np.random.default_rng(2)
        obs_diff = float(np.mean(y_scores) - np.mean(n_scores))
        combined = np.concatenate([y_scores, n_scores])
        ny = len(y_scores)
        perm_diffs = np.empty(n_permutations)
        for j in range(n_permutations):
            perm = rng_between.permutation(combined)
            perm_diffs[j] = np.mean(perm[:ny]) - np.mean(perm[ny:])
        p_between = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
        results["between"] = {"p_value": p_between, "obs_diff": obs_diff}
        print(f"  [between] diff = {obs_diff:+.4f}  p = {p_between:.4f}")
    else:
        results["between"] = {"p_value": np.nan, "obs_diff": np.nan}

    return results


# ---------------------------------------------------------------------------
# Helpers for alignment and bootstrap CI
# ---------------------------------------------------------------------------


def _align_to_push(
    rate_df: pd.DataFrame,
    push_sub: pd.DataFrame,
    cutoff_min: float,
    bin_size_min: float,
    window_min: float,
) -> pd.DataFrame:
    """
    Filter rate_df to hour 2, align to each fly's first push, return with rel_bin column.
    Mirrors plot_aligned_to_first_push logic from original.
    """
    flies = set(push_sub["fly"])
    sub = rate_df[rate_df["fly"].isin(flies)].copy()
    sub = sub[sub["bin_center_min"] >= cutoff_min]  # Hour 2 only
    sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
    sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
    sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
    return sub[sub["rel_bin"].abs() <= window_min]


def _bootstrap_ci_per_bin(
    aligned: pd.DataFrame,
    bins_sorted: np.ndarray,
    n_bootstrap: int = 1000,
    rng=None,
):
    """Bootstrap mean +/- 95% CI per relative bin. Mirrors original."""
    if rng is None:
        rng = np.random.default_rng(42)
    means, ci_lo, ci_hi = [], [], []
    for rb in bins_sorted:
        vals = aligned.loc[aligned["rel_bin"] == rb, "rate_per_min"].values
        if len(vals) == 0:
            means.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            continue
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
        means.append(float(np.mean(vals)))
        ci_lo.append(float(np.percentile(boot, 2.5)))
        ci_hi.append(float(np.percentile(boot, 97.5)))
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def _p_str(p: float) -> str:
    min_p = 1.0 / N_PERMUTATIONS
    if p <= min_p:
        return f"<{min_p:.4f}"
    return f"{p:.4f}"


# ---------------------------------------------------------------------------
# Panel plots
# ---------------------------------------------------------------------------


def plot_panels_d_e(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Panels (d) and (e): Interaction rates aligned to each fly's first significant push.

    Mirrors plot_aligned_to_first_push from original:
      - rate_df has absolute bin_center_min values
      - Align by subtracting t_first_push_min, snap to integer bins
      - Red dashed vertical line at x=0 (the first push)
      - Mean +/- bootstrapped 95% CI
    """
    print("\n=== Panels (d-e): Aligned interaction rates ===")
    cutoff_min = FOCUS_HOURS * 60.0
    bs_rng = np.random.default_rng(42)

    for magnet_val, panel_name in [("y", "d"), ("n", "e")]:
        push_sub = first_push_df[first_push_df["Magnet"] == magnet_val]
        n_flies = len(push_sub)
        color = COLORS[magnet_val]

        aligned = _align_to_push(rate_df, push_sub, cutoff_min, BIN_SIZE_MIN, WINDOW_MIN)
        bins_sorted = np.sort(aligned["rel_bin"].unique())
        means, ci_lo, ci_hi = _bootstrap_ci_per_bin(aligned, bins_sorted, rng=bs_rng)

        # Save stats
        stats_df = pd.DataFrame({"rel_bin": bins_sorted, "mean_rate": means, "ci_lower": ci_lo, "ci_upper": ci_hi})
        stats_csv = output_dir / f"panel_{panel_name}_aligned_rates_stats.csv"
        stats_df.to_csv(stats_csv, index=False)

        # Plot
        set_illustrator_style()
        fig, ax = plt.subplots(figsize=(170 / 25.4, 89 / 25.4))
        ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25, zorder=5)
        ax.plot(bins_sorted, means, color=color, lw=2.5, zorder=10)

        # Red dashed line at first push (x=0) -- matches original
        ax.axvline(0, color="red", linestyle="--", lw=2, alpha=0.9)
        ax.set_xlim(-WINDOW_MIN, WINDOW_MIN)

        col_title = (
            f"Access to immobile ball (n={n_flies})" if magnet_val == "y" else f"No access to ball (n={n_flies})"
        )
        ax.set_title(col_title, fontsize=11, color=color, fontweight="bold", pad=4)
        ax.set_xlabel("Time relative to first push (min)", fontsize=10)
        ax.set_ylabel("Interaction rate (events / min)", fontsize=10)
        ax.tick_params(labelsize=9, width=1.5, length=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.set_facecolor("white")
        ax.grid(False)
        plt.tight_layout()

        for suffix in [".pdf", ".png"]:
            p_out = (output_dir / f"panel_{panel_name}_aligned_rates").with_suffix(suffix)
            fig.savefig(p_out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"\n  Panel ({panel_name}): {LABELS[magnet_val]} (n={n_flies})")
        print(f"    Saved: {output_dir}/panel_{panel_name}_aligned_rates.pdf")


def _paired_prepost_ax(ax, gdf, color, label, stats_group, rng):
    """Draw a paired pre/post boxplot on ax. Returns y_max for annotation."""
    positions = [0, 1]
    data_to_plot = [gdf["pre_mean"].values, gdf["post_mean"].values]
    n = len(gdf)

    ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    for pos, data in zip(positions, data_to_plot):
        x = rng.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=20, color=color, zorder=5)
    for _, row in gdf.iterrows():
        ax.plot([0, 1], [row["pre_mean"], row["post_mean"]], color="gray", alpha=0.3, linewidth=0.5, zorder=3)

    y_all = np.concatenate(data_to_plot)
    y_max = float(np.nanmax(y_all))
    y_min_val = float(np.nanmin(y_all))
    y_rng = y_max - y_min_val if y_max != y_min_val else 1.0
    ax.set_ylim(y_min_val - 0.05 * y_rng, y_max + 0.30 * y_rng)

    p = stats_group.get("p_value", np.nan)
    if not np.isnan(p):
        bar_y = y_max + 0.08 * y_rng
        txt_y = y_max + 0.12 * y_rng
        ax.plot([0, 1], [bar_y, bar_y], color="black", lw=1.5)
        ax.text(0.5, txt_y, f"{_sig_label(p)}\np={_p_str(p)}", ha="center", va="bottom", fontsize=9, color="red")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Pre", "Post"], fontsize=10)
    ax.set_ylabel("Interaction rate (events / min)", fontsize=10)
    ax.set_title(f"{label}\n(n={n})", fontsize=10, color=color, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return y_max


def plot_panel_f(
    change_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
) -> None:
    """Panel (f): Pre vs Post paired boxplot for experimental animals (Magnet=y only)."""
    print("\n=== Panel (f): Pre vs Post (experimental) ===")

    exp_df = change_df[change_df["Magnet"] == "y"].copy()
    if exp_df.empty:
        print("  WARNING: No experimental flies with sufficient data -- skipping panel (f)")
        return

    n = len(exp_df)
    p_value = stats.get("y", {}).get("p_value", np.nan)
    obs_mean = stats.get("y", {}).get("obs_mean", np.nan)
    print(f"  Experimental group (n={n}): mean delta={obs_mean:.3f}, p={p_value:.4f}")

    stats_csv = output_dir / "panel_f_prepost_experimental_stats.csv"
    pd.DataFrame(
        [
            {
                "group": "y",
                "n": n,
                "mean_pre": exp_df["pre_mean"].mean(),
                "mean_post": exp_df["post_mean"].mean(),
                "mean_delta": obs_mean,
                "p_value": p_value,
            }
        ]
    ).to_csv(stats_csv, index=False)
    exp_df.to_csv(output_dir / "panel_f_prepost_experimental_per_fly.csv", index=False)

    set_illustrator_style()
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(4, 5))
    _paired_prepost_ax(ax, exp_df, COLORS["y"], LABELS["y"], stats.get("y", {}), rng)
    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        fig.savefig((output_dir / "panel_f_prepost_experimental").with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/panel_f_prepost_experimental.pdf")


def plot_panel_g(
    change_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
) -> None:
    """
    Panel (g): Three-panel figure:
      Left  -- Magnet=y paired pre/post
      Middle -- Magnet=n paired pre/post
      Right  -- Delta distributions with between-group test annotation
    Mirrors plot_post_push_rate_change from original.
    """
    print("\n=== Panel (g): Pre vs Post (both groups) ===")
    if change_df.empty:
        print("  WARNING: change_df is empty -- skipping panel (g)")
        return

    for group in ["n", "y"]:
        gdf = change_df[change_df["Magnet"] == group]
        p = stats.get(group, {}).get("p_value", np.nan)
        m = stats.get(group, {}).get("obs_mean", np.nan)
        print(f"  {LABELS[group]} (n={len(gdf)}): mean delta={m:.3f}, p={p:.4f}")
    p_between = stats.get("between", {}).get("p_value", np.nan)
    print(f"  Between groups: p={p_between:.4f}")

    pd.DataFrame(
        [
            {
                "group": g,
                "label": LABELS[g],
                "n": len(change_df[change_df["Magnet"] == g]),
                "mean_delta": stats.get(g, {}).get("obs_mean", np.nan),
                "one_sample_p": stats.get(g, {}).get("p_value", np.nan),
                "between_group_p": p_between,
            }
            for g in ["n", "y"]
        ]
    ).to_csv(output_dir / "panel_g_stats.csv", index=False)
    change_df.to_csv(output_dir / "panel_g_per_fly.csv", index=False)

    set_illustrator_style()
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for magnet_val, ax in zip(["y", "n"], axes[:2]):
        gdf = change_df[change_df["Magnet"] == magnet_val]
        _paired_prepost_ax(ax, gdf, COLORS[magnet_val], LABELS[magnet_val], stats.get(magnet_val, {}), rng)

    # Right panel: delta distributions with between-group annotation
    ax_delta = axes[2]
    for pos, magnet_val in enumerate(["y", "n"]):
        gdf = change_df[change_df["Magnet"] == magnet_val]
        deltas = gdf["delta"].values
        color = COLORS[magnet_val]
        ax_delta.boxplot(
            [deltas],
            positions=[pos],
            widths=0.4,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        ax_delta.scatter(rng.normal(pos, 0.04, size=len(deltas)), deltas, alpha=0.5, s=20, color=color, zorder=5)

    ax_delta.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    if not np.isnan(p_between):
        y_all_d = change_df["delta"].values
        y_max_d = float(np.nanmax(y_all_d))
        y_min_d = float(np.nanmin(y_all_d))
        y_rng_d = y_max_d - y_min_d if y_max_d != y_min_d else 1.0
        ax_delta.set_ylim(y_min_d - 0.05 * y_rng_d, y_max_d + 0.30 * y_rng_d)
        bar_y_d = y_max_d + 0.08 * y_rng_d
        txt_y_d = y_max_d + 0.12 * y_rng_d
        ax_delta.plot([0, 1], [bar_y_d, bar_y_d], color="black", lw=1.5)
        ax_delta.text(
            0.5,
            txt_y_d,
            f"{_sig_label(p_between)}\np={_p_str(p_between)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
        )

    ax_delta.set_xticks([0, 1])
    ax_delta.set_xticklabels([LABELS["y"], LABELS["n"]], fontsize=9)
    ax_delta.set_ylabel("Delta rate (Post - Pre, events / min)", fontsize=10)
    ax_delta.set_title(f"Between groups\n(p={p_between:.3f})", fontsize=10)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        fig.savefig((output_dir / "panel_g_prepost_between_groups").with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/panel_g_prepost_between_groups.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(test_mode: bool = False) -> None:
    print("\n" + "=" * 60)
    print("EXTENDED DATA FIGURE 2 (d-g): INTERACTION RATE ANALYSIS")
    print("=" * 60)

    output_dir = figure_output_dir("EDFigure2", __file__, create=True)
    print(f"\nOutput directory: {output_dir}")

    df = load_data(test_mode=test_mode)
    onsets = extract_event_onsets(df)

    # Compute rates in ABSOLUTE time bins (mirrors original compute_rates_per_fly)
    print(f"\nComputing interaction rates per fly (bin size = {BIN_SIZE_MIN} min)...")
    rate_df = compute_rates_per_fly(onsets, df, bin_size_min=BIN_SIZE_MIN)
    print(f"  Rate data shape: {rate_df.shape}")

    print("\nComputing ball displacement per event...")
    event_disp = compute_ball_displacement_per_event(df)

    first_push_df = find_first_significant_push(event_disp, focus_hours=FOCUS_HOURS, threshold_px=PUSH_THRESHOLD_PX)

    # Panels (d) and (e): aligned rate curves
    plot_panels_d_e(rate_df, first_push_df, output_dir)

    # Compute pre/post rate change for panels f/g
    print(f"\nComputing pre/post rate change (pre={PRE_WINDOW}, post={POST_WINDOW})...")
    change_df = compute_post_push_rate_change(
        rate_df,
        first_push_df,
        focus_hours=FOCUS_HOURS,
        pre_window_min=PRE_WINDOW,
        post_window_min=POST_WINDOW,
    )

    if not change_df.empty:
        print("\nRunning permutation tests on delta scores...")
        stats = test_post_push_change(change_df, n_permutations=N_PERMUTATIONS)
        plot_panel_f(change_df, stats, output_dir)
        plot_panel_g(change_df, stats, output_dir)
    else:
        print("  WARNING: No change data -- skipping panels f and g")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Data Figure 2 (d-g) -- Interaction rate analysis")
    parser.add_argument("--test", action="store_true", help="Run in test mode with subset of data")
    args = parser.parse_args()
    main(test_mode=args.test)
