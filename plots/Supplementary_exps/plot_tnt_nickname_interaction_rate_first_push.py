#!/usr/bin/env python3
from __future__ import annotations

"""
TNT nickname-vs-control interaction-rate analysis aligned to each fly's first significant push.

Outputs
-------
1) Interaction-rate trajectories (events/min) centered at first significant push,
   plotted separately for the line of interest and its matched control.
2) Pre/post first-push interaction-rate boxplots per group, plus a side panel
   comparing delta = post - pre between groups.

Data assumptions
----------------
- Input is loaded fly-by-fly from experiment folders listed in a YAML file.
- Requires columns to compute first significant push and interaction-rate trajectories.
- Keeps only Light=on rows when a Light column exists.

Usage
-----
python plot_tnt_nickname_interaction_rate_first_push.py --nickname MB247
    [--yaml-path PATH] [--output-dir PATH]
    [--control-nickname NAME] [--split-registry PATH]
    [--bin-size 1] [--threshold-px 5] [--focus-min 0]
    [--window-min 60] [--n-permutations 10000] [--test]
"""

import argparse
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.append(str(Path(__file__).parent.parent))

from ballpushing_utils.filtered_dataset_loader import build_dataset_for_nickname_and_control

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

DEFAULT_YAML_PATH = "/home/matthias/ballpushing_utils/experiments_yaml/TNT_screen.yaml"
DEFAULT_SPLIT_REGISTRY = "/mnt/upramdya_data/MD/Region_map_250908.csv"
DEFAULT_OUTPUT_ROOT = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Interaction_rates"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _safe_path_part(name: str) -> str:
    cleaned = str(name).strip()
    return cleaned if cleaned else "Unknown"


def format_p_value(p_value: float, n_permutations: int) -> str:
    min_p = 1.0 / n_permutations
    if p_value <= min_p:
        return f"p ≤ {min_p:.0e}"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def load_nickname_and_control_data(
    yaml_path: Path,
    nickname: str,
    split_registry_path: Path,
    control_nickname: Optional[str] = None,
    force_control: Optional[str] = None,
    test_mode: bool = False,
    test_max_flies_per_group: int = 6,
    test_max_experiments: Optional[int] = None,
) -> tuple[pd.DataFrame, str, object]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    print(f"Loading filtered coordinates from YAML: {yaml_path}")
    max_experiments = test_max_experiments if test_mode else None
    max_flies = test_max_flies_per_group if test_mode else None

    df, manifest, info = build_dataset_for_nickname_and_control(
        yaml_path=yaml_path,
        dataset_type="coordinates",
        nickname=nickname,
        split_registry_path=split_registry_path,
        control_nickname=control_nickname,
        force_control=force_control,
        max_experiments=max_experiments,
        max_flies_per_group=max_flies,
    )

    if df.empty:
        raise RuntimeError("No coordinates produced for selected nickname/control from YAML experiments")

    focal_canonical = info.nickname
    control = info.control_nickname
    print(f"Resolved mapping: {focal_canonical} ({info.genotype}) vs {control} ({info.control_genotype})")
    print(f"Selected fly directories: {len(manifest)}")

    required = [
        "time",
        "fly",
        "Nickname",
        "interaction_event",
        "interaction_event_onset",
        "x_ball_0",
        "y_ball_0",
    ]
    _require_columns(df, required, "coordinates dataset")

    if "Light" in df.columns:
        before = len(df)
        df = df[df["Light"] == "on"].copy()
        print(f"Filtered Light=on: {before} -> {len(df)} rows")

    df["Nickname"] = df["Nickname"].astype(str)
    focal_candidates = {str(nickname).strip(), str(focal_canonical).strip()}

    if control not in set(df["Nickname"].unique()):
        raise ValueError(f"Resolved control '{control}' is not present in dataset")

    subset = df[df["Nickname"].isin(focal_candidates.union({control}))].copy()

    if subset.empty and "Simplified Nickname" in df.columns:
        simp = df["Simplified Nickname"].astype(str)
        subset = df[(simp == str(nickname).strip()) | (df["Nickname"] == control)].copy()

    if subset.empty:
        raise ValueError(
            f"Nickname '{nickname}' not present in loaded subset (canonical='{focal_canonical}', control='{control}')"
        )

    subset["Group"] = np.where(subset["Nickname"].isin(focal_candidates), nickname, control)

    if test_mode and test_max_flies_per_group is not None:
        keep_flies = []
        for grp in [nickname, control]:
            grp_flies = sorted(subset.loc[subset["Group"] == grp, "fly"].unique())[:test_max_flies_per_group]
            keep_flies.extend(grp_flies)
        subset = subset[subset["fly"].isin(keep_flies)].copy()
        print(
            f"[TEST MODE] Keeping up to {test_max_flies_per_group} flies/group -> "
            f"{subset['fly'].nunique()} flies total"
        )

    print(f"Nickname: {nickname}")
    print(f"Control : {control}")
    print(f"Rows    : {len(subset)}")
    for grp in [nickname, control]:
        grp_df = subset[subset["Group"] == grp]
        print(f"  {grp}: {grp_df['fly'].nunique()} flies, {len(grp_df)} frames")

    return subset, control, info


def extract_event_onsets(df: pd.DataFrame) -> pd.DataFrame:
    non_nan = df[df["interaction_event_onset"].notna()].copy()
    onsets = (
        non_nan.groupby(["fly", "interaction_event_onset"], as_index=False)
        .agg(time=("time", "min"), Group=("Group", "first"))
        .reset_index(drop=True)
    )
    onsets["time_min"] = onsets["time"] / 60.0
    print(f"Unique interaction onsets: {len(onsets)}")
    return onsets


def compute_rates_per_fly(onsets: pd.DataFrame, df_full: pd.DataFrame, bin_size_min: float = 1.0) -> pd.DataFrame:
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "Group"]].set_index("fly")
    rows = []

    for fly in df_full["fly"].unique():
        group = fly_meta.loc[fly, "Group"]
        fly_onsets = onsets[onsets["fly"] == fly]
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            count = ((fly_onsets["time_min"] >= b_start) & (fly_onsets["time_min"] < b_end)).sum()
            rows.append(
                {
                    "fly": fly,
                    "Group": group,
                    "bin_idx": i,
                    "bin_start_min": float(b_start),
                    "bin_center_min": float(bin_centers[i]),
                    "event_count": int(count),
                    "rate_per_min": float(count / bin_size_min),
                }
            )

    return pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)


def compute_ball_displacement_per_event(df: pd.DataFrame) -> pd.DataFrame:
    event_rows = df[df["interaction_event"].notna()].copy()
    group_keys = ["fly", "interaction_event"]

    idx_start = event_rows.groupby(group_keys)["time"].idxmin()
    idx_end = event_rows.groupby(group_keys)["time"].idxmax()

    start_df = event_rows.loc[
        idx_start,
        ["fly", "interaction_event", "time", "x_ball_0", "y_ball_0", "Group"],
    ].rename(
        columns={
            "time": "onset_time_s",
            "x_ball_0": "ball_x_start",
            "y_ball_0": "ball_y_start",
        }
    )
    end_df = event_rows.loc[idx_end, ["fly", "interaction_event", "x_ball_0", "y_ball_0"]].rename(
        columns={"x_ball_0": "ball_x_end", "y_ball_0": "ball_y_end"}
    )

    agg = start_df.merge(end_df, on=["fly", "interaction_event"], how="left")
    agg["displacement_px"] = np.sqrt(
        (agg["ball_x_end"] - agg["ball_x_start"]) ** 2 + (agg["ball_y_end"] - agg["ball_y_start"]) ** 2
    )

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

    out = agg.merge(max_disp, on=["fly", "interaction_event"], how="inner")
    out["onset_time_min"] = out["onset_time_s"] / 60.0
    return out


def compute_first_significant_push(
    event_disp_df: pd.DataFrame,
    threshold_px: float = 5.0,
    focus_min: float = 0.0,
) -> pd.DataFrame:
    eligible = event_disp_df[
        (event_disp_df["onset_time_min"] >= focus_min) & (event_disp_df["max_displacement_px"] > threshold_px)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["fly", "Group", "t_first_push_s", "t_first_push_min", "first_push_event_id"])

    idx_first = eligible.groupby("fly")["onset_time_s"].idxmin()
    first_push = eligible.loc[idx_first].reset_index(drop=True)
    first_push = first_push.rename(
        columns={
            "onset_time_s": "t_first_push_s",
            "onset_time_min": "t_first_push_min",
            "interaction_event": "first_push_event_id",
        }
    )
    print(f"Flies with first significant push (>{threshold_px}px, from {focus_min:.1f} min): {len(first_push)}")
    return first_push[["fly", "Group", "t_first_push_s", "t_first_push_min", "first_push_event_id"]]


def compute_event_index_aligned_rates(
    onsets_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
) -> pd.DataFrame:
    if onsets_df.empty or first_push_df.empty:
        return pd.DataFrame(
            columns=[
                "fly",
                "Group",
                "event_idx",
                "rel_event_idx",
                "time_min",
                "delta_prev_min",
                "delta_next_min",
                "local_rate_per_min",
            ]
        )

    first_push_by_fly = first_push_df.set_index("fly")
    rows = []

    for fly, fly_onsets in onsets_df.groupby("fly"):
        if fly not in first_push_by_fly.index:
            continue

        grp = first_push_by_fly.loc[fly, "Group"]
        t_push = float(first_push_by_fly.loc[fly, "t_first_push_min"])

        s = fly_onsets.sort_values("time_min").reset_index(drop=True).copy()
        if s.empty:
            continue

        s["event_idx"] = np.arange(1, len(s) + 1)
        s["delta_prev_min"] = s["time_min"].diff()
        s["delta_next_min"] = s["time_min"].shift(-1) - s["time_min"]
        interval = s["delta_prev_min"].where(s["delta_prev_min"].notna(), s["delta_next_min"])
        s["local_rate_per_min"] = np.where(interval > 0, 1.0 / interval, np.nan)

        idx_push = int(np.argmin(np.abs(s["time_min"].values - t_push)))
        first_event_idx = int(s.loc[idx_push, "event_idx"])
        s["rel_event_idx"] = s["event_idx"] - first_event_idx
        s["Group"] = grp
        s["fly"] = fly

        rows.append(
            s[
                [
                    "fly",
                    "Group",
                    "event_idx",
                    "rel_event_idx",
                    "time_min",
                    "delta_prev_min",
                    "delta_next_min",
                    "local_rate_per_min",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fly",
                "Group",
                "event_idx",
                "rel_event_idx",
                "time_min",
                "delta_prev_min",
                "delta_next_min",
                "local_rate_per_min",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def plot_rate_aligned_to_first_push_event_index(
    event_rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    group_order: list[str],
    group_colors: dict[str, str],
    sig_coverage: dict[str, dict[str, float]],
    window_events: int,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
) -> None:
    conditions = [c for c in group_order if c in first_push_df["Group"].unique()]
    if not conditions:
        print("No groups with first-push flies. Skipping event-index aligned plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["Group"] == cond]
        flies = set(push_sub["fly"])

        aligned = event_rate_df[
            (event_rate_df["fly"].isin(flies))
            & (event_rate_df["rel_event_idx"].abs() <= window_events)
            & (event_rate_df["local_rate_per_min"].notna())
        ].copy()

        if aligned.empty:
            ax.set_title(f"{cond} (n=0)")
            continue

        bins_sorted = np.sort(aligned["rel_event_idx"].unique())
        means, ci_lo, ci_hi = [], [], []

        for rb in bins_sorted:
            vals = aligned.loc[aligned["rel_event_idx"] == rb, "local_rate_per_min"].values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        color = group_colors.get(cond, "C0")
        ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
        ax.plot(bins_sorted, means, color=color, lw=2.5, label=f"Mean ± 95% CI (n={len(flies)})")
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push event")
        ax.set_xlabel("Event index relative to first significant push")
        cov = sig_coverage.get(cond, {"n_sig": len(flies), "n_total": len(flies), "pct": 100.0})
        pct_txt = f"{cov['pct']:.1f}%" if not np.isnan(cov["pct"]) else "n.d."
        ax.set_title(f"{cond} (n_sig={cov['n_sig']}/{cov['n_total']}, {pct_txt})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Local interaction rate (events/min)")
    fig.suptitle(
        f"Interaction rate aligned to first significant push event index\n"
        f"threshold > {threshold_px:.1f}px, push search from {focus_min:.1f} min",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def compute_significant_coverage(
    df_full: pd.DataFrame,
    first_push_df: pd.DataFrame,
    group_order: list[str],
) -> dict[str, dict[str, float]]:
    total_by_group = df_full.groupby("Group")["fly"].nunique().to_dict()
    sig_by_group = first_push_df.groupby("Group")["fly"].nunique().to_dict() if not first_push_df.empty else {}

    coverage = {}
    for grp in group_order:
        n_total = int(total_by_group.get(grp, 0))
        n_sig = int(sig_by_group.get(grp, 0))
        pct = (100.0 * n_sig / n_total) if n_total > 0 else np.nan
        coverage[grp] = {"n_sig": n_sig, "n_total": n_total, "pct": pct}
    return coverage


def plot_rate_aligned_to_first_push(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    group_order: list[str],
    group_colors: dict[str, str],
    sig_coverage: dict[str, dict[str, float]],
    bin_size_min: float,
    window_min: float,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
) -> None:
    conditions = [c for c in group_order if c in first_push_df["Group"].unique()]
    if not conditions:
        print("No groups with first-push flies. Skipping aligned-rate plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["Group"] == cond]
        flies = set(push_sub["fly"])

        sub = rate_df[rate_df["fly"].isin(flies)].copy()
        sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
        sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
        sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
        aligned = sub[sub["rel_bin"].abs() <= window_min].copy()

        if aligned.empty:
            ax.set_title(f"{cond} (n=0)")
            continue

        bins_sorted = np.sort(aligned["rel_bin"].unique())
        means, ci_lo, ci_hi = [], [], []

        for rb in bins_sorted:
            vals = aligned.loc[aligned["rel_bin"] == rb, "rate_per_min"].values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        color = group_colors.get(cond, "C0")
        ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
        ax.plot(bins_sorted, means, color=color, lw=2.5, label=f"Mean ± 95% CI (n={len(flies)})")
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")
        ax.set_xlabel("Time relative to first push (min)")
        cov = sig_coverage.get(cond, {"n_sig": len(flies), "n_total": len(flies), "pct": 100.0})
        pct_txt = f"{cov['pct']:.1f}%" if not np.isnan(cov["pct"]) else "n.d."
        ax.set_title(f"{cond} (n_sig={cov['n_sig']}/{cov['n_total']}, {pct_txt})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Interaction rate (events/min)")
    fig.suptitle(
        f"Interaction rate aligned to first significant push\n"
        f"threshold > {threshold_px:.1f}px, push search from {focus_min:.1f} min",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def compute_post_push_rate_change(
    rate_df: pd.DataFrame,
    onsets_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    focus_min: float,
) -> pd.DataFrame:
    all_bins = np.sort(rate_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 5.0

    rows = []
    eligibility = {}

    def _ensure_group(group_name: str) -> None:
        if group_name not in eligibility:
            eligibility[group_name] = {
                "n_sig": 0,
                "kept": 0,
                "excluded_no_pre": 0,
                "excluded_no_post": 0,
            }

    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        group = push_row["Group"]
        t_push = push_row["t_first_push_min"]
        _ensure_group(group)
        eligibility[group]["n_sig"] += 1

        fly_data = rate_df[rate_df["fly"] == fly].copy()
        fly_data["rel_time_min"] = fly_data["bin_center_min"] - t_push
        fly_onset_times = onsets_df[onsets_df["fly"] == fly]["time_min"].values

        fly_end = float(fly_data["bin_center_min"].max())
        pre_available = max(0.0, t_push - focus_min)
        post_available = max(0.0, fly_end - t_push)

        pre_far_abs = min(pre_window_min[0], pre_available)
        post_far_abs = min(post_window_min[1], post_available)

        pre_data = fly_data[(fly_data["rel_time_min"] < 0) & (fly_data["rel_time_min"] >= -pre_far_abs)]
        post_data = fly_data[(fly_data["rel_time_min"] > 0) & (fly_data["rel_time_min"] <= post_far_abs)]

        if pre_far_abs <= 0:
            eligibility[group]["excluded_no_pre"] += 1
            continue
        if post_far_abs <= 0:
            eligibility[group]["excluded_no_post"] += 1
            continue

        pre_start = t_push - pre_far_abs
        pre_end = t_push
        post_start = t_push
        post_end = t_push + post_far_abs

        pre_count = int(((fly_onset_times >= pre_start) & (fly_onset_times < pre_end)).sum())
        post_count = int(((fly_onset_times > post_start) & (fly_onset_times <= post_end)).sum())

        pre_mean = float(pre_count / pre_far_abs)
        post_mean = float(post_count / post_far_abs)
        eligibility[group]["kept"] += 1
        rows.append(
            {
                "fly": fly,
                "Group": group,
                "t_first_push_min": float(t_push),
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta": post_mean - pre_mean,
                "n_pre_bins": int(len(pre_data)),
                "n_post_bins": int(len(post_data)),
                "n_pre_events": pre_count,
                "n_post_events": post_count,
                "pre_window_used_min": float(pre_far_abs),
                "post_window_used_min": float(post_far_abs),
            }
        )

    out = pd.DataFrame(rows)
    for grp in sorted(eligibility.keys()):
        info = eligibility[grp]
        if (not out.empty) and (grp in out["Group"].unique()):
            sub = out[out["Group"] == grp]
            mean_delta_txt = f"{sub['delta'].mean():+.3f}"
        else:
            mean_delta_txt = "n.d."
        print(
            f"  {grp}: kept={info['kept']}/{info['n_sig']} first-push flies; "
            f"excluded(no_pre={info['excluded_no_pre']}, no_post={info['excluded_no_post']}), "
            f"mean delta={mean_delta_txt}"
        )
    return out


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    obs = np.abs(np.mean(scores))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(scores))
        if np.abs(np.mean(scores * signs)) >= obs:
            count += 1
    return count / n_permutations


def test_post_push_change(change_df: pd.DataFrame, group_order: list[str], n_permutations: int) -> dict:
    rng = np.random.default_rng(0)
    results = {"within": {}, "between": {}}

    groups = [g for g in group_order if g in change_df["Group"].unique()]

    raw_within_p = []
    for grp in groups:
        vals = change_df[change_df["Group"] == grp]["delta"].dropna().values
        if len(vals) < 3:
            results["within"][grp] = {"p_value": np.nan, "mean_delta": np.nan, "n": len(vals)}
            raw_within_p.append(1.0)
            continue
        p = _one_sample_sign_perm(vals, n_permutations, rng)
        m = float(np.mean(vals))
        results["within"][grp] = {"p_value": p, "mean_delta": m, "n": len(vals)}
        raw_within_p.append(p)

    if raw_within_p:
        _, p_corr, _, _ = multipletests(raw_within_p, alpha=0.05, method="fdr_bh")
        for i, grp in enumerate(groups):
            results["within"][grp]["p_value_fdr"] = float(p_corr[i])

    pairwise = list(combinations(groups, 2))
    raw_between_p = []
    pair_keys = []

    for a, b in pairwise:
        va = change_df[change_df["Group"] == a]["delta"].dropna().values
        vb = change_df[change_df["Group"] == b]["delta"].dropna().values
        key = f"{a}__vs__{b}"

        if len(va) < 3 or len(vb) < 3:
            results["between"][key] = {"a": a, "b": b, "p_value": np.nan, "obs_diff": np.nan}
            raw_between_p.append(1.0)
            pair_keys.append(key)
            continue

        obs_diff = float(np.mean(va) - np.mean(vb))
        combined = np.concatenate([va, vb])
        na = len(va)
        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diffs[i] = np.mean(perm[:na]) - np.mean(perm[na:])
        p = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

        results["between"][key] = {"a": a, "b": b, "p_value": p, "obs_diff": obs_diff}
        raw_between_p.append(p)
        pair_keys.append(key)

    if raw_between_p:
        _, p_corr, _, _ = multipletests(raw_between_p, alpha=0.05, method="fdr_bh")
        for i, key in enumerate(pair_keys):
            results["between"][key]["p_value_fdr"] = float(p_corr[i])

    return results


def plot_post_push_rate_change(
    change_df: pd.DataFrame,
    stats: dict,
    group_order: list[str],
    group_colors: dict[str, str],
    sig_coverage: dict[str, dict[str, float]],
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    n_permutations: int,
    output_path: Path,
) -> None:
    groups = [g for g in group_order if g in change_df["Group"].unique()]
    if not groups:
        print("No groups with post-push change data. Skipping boxplot figure.")
        return

    rng = np.random.default_rng(42)
    n_cols = len(groups) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 6.0))
    if n_cols == 1:
        axes = [axes]

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    all_paired = change_df[change_df["pre_mean"].notna() & change_df["post_mean"].notna()].copy()
    y_min = min(all_paired["pre_mean"].min(), all_paired["post_mean"].min())
    y_max = max(all_paired["pre_mean"].max(), all_paired["post_mean"].max())
    y_rng = y_max - y_min if y_max != y_min else 1.0

    for ax, grp in zip(axes[:-1], groups):
        sub = change_df[change_df["Group"] == grp].copy()
        sub = sub[sub["pre_mean"].notna() & sub["post_mean"].notna()]
        pre_vals = sub["pre_mean"].values
        post_vals = sub["post_mean"].values
        n = len(sub)
        color = group_colors.get(grp, "C0")

        ax.boxplot(
            [pre_vals, post_vals],
            positions=[0, 1],
            patch_artist=True,
            widths=0.35,
            boxprops=dict(facecolor=color, alpha=0.35),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color=color, alpha=0.7),
            capprops=dict(color=color, alpha=0.7),
            flierprops=dict(marker="", linestyle="none"),
        )

        jitter = rng.uniform(-0.07, 0.07, size=n)
        for pv, sv, jit in zip(pre_vals, post_vals, jitter):
            ax.plot([0 + jit, 1 + jit], [pv, sv], color="gray", alpha=0.4, lw=0.8)
        ax.scatter(0 + jitter, pre_vals, s=22, color=color, alpha=0.7, edgecolors="none")
        ax.scatter(1 + jitter, post_vals, s=22, color=color, alpha=0.7, edgecolors="none")

        p = stats["within"].get(grp, {}).get("p_value_fdr", np.nan)
        txt = sig_stars(p) if not np.isnan(p) else "n.d."
        if txt == "ns" and not np.isnan(p):
            txt = f"p={format_p_value(p, n_permutations)}"
        if txt in {"*", "**", "***"}:
            txt = f"{txt}\n{format_p_value(p, n_permutations)}"

        y_annot = y_max + 0.12 * y_rng
        ax.plot([0, 1], [y_annot, y_annot], color="black", lw=1.2)
        ax.text(0.5, y_annot + 0.02 * y_rng, txt, ha="center", va="bottom", fontsize=10, color="red")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([pre_label, post_label], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(y_min - 0.05 * y_rng, y_max + 0.30 * y_rng)
        ax.set_ylabel("Interaction rate (events/min)")
        cov = sig_coverage.get(grp, {"n_sig": np.nan, "n_total": np.nan, "pct": np.nan})
        pct_txt = f"{cov['pct']:.1f}%" if not np.isnan(cov["pct"]) else "n.d."
        ax.set_title(f"{grp}\n(post/pre n={n}, sig={int(cov['n_sig'])}/{int(cov['n_total'])}, {pct_txt})", fontsize=10)

    ax_d = axes[-1]
    positions = np.arange(len(groups))

    for pos, grp in zip(positions, groups):
        vals = change_df[change_df["Group"] == grp]["delta"].dropna().values
        color = group_colors.get(grp, "C0")

        ax_d.boxplot(
            [vals],
            positions=[pos],
            patch_artist=True,
            widths=0.35,
            boxprops=dict(facecolor=color, alpha=0.35),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color=color, alpha=0.7),
            capprops=dict(color=color, alpha=0.7),
            flierprops=dict(marker="", linestyle="none"),
        )
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax_d.scatter(pos + jitter, vals, s=22, color=color, alpha=0.7, edgecolors="none", zorder=3)

    ax_d.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    dvals = change_df["delta"].dropna().values
    if len(dvals):
        d_min, d_max = float(np.min(dvals)), float(np.max(dvals))
    else:
        d_min, d_max = -1.0, 1.0
    d_rng = d_max - d_min if d_max != d_min else 1.0

    pair_items = list(stats.get("between", {}).values())
    valid_pairs = [p for p in pair_items if not np.isnan(p.get("p_value_fdr", np.nan))]
    for row_idx, pair in enumerate(valid_pairs):
        a = pair["a"]
        b = pair["b"]
        if a not in groups or b not in groups:
            continue
        xa = groups.index(a)
        xb = groups.index(b)
        y = d_max + (0.10 + 0.08 * row_idx) * d_rng
        p = pair.get("p_value_fdr", np.nan)
        txt = sig_stars(p) if not np.isnan(p) else "n.d."
        if txt == "ns" and not np.isnan(p):
            txt = f"p={format_p_value(p, n_permutations)}"
        ax_d.plot([xa, xb], [y, y], color="black", lw=1.2)
        ax_d.text((xa + xb) / 2, y + 0.015 * d_rng, txt, ha="center", va="bottom", fontsize=10, color="red")

    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(
        [
            (
                f"{g}\n"
                f"post/pre n={len(change_df[change_df['Group'] == g])}\n"
                f"sig={int(sig_coverage.get(g, {}).get('n_sig', 0))}/{int(sig_coverage.get(g, {}).get('n_total', 0))}"
            )
            for g in groups
        ],
        fontsize=9,
    )
    ax_d.set_xlim(-0.6, len(groups) - 0.4)
    ax_d.set_ylim(d_min - 0.10 * d_rng, d_max + (0.22 + 0.08 * len(valid_pairs)) * d_rng)
    ax_d.set_ylabel("Δ rate (post − pre, events/min)")
    ax_d.set_title("Delta comparison\n(pairwise permutation, FDR)", fontsize=11)

    fig.suptitle("Interaction-rate increase around first significant push", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def save_tables(
    output_dir: Path,
    stem_prefix: str,
    first_push_df: pd.DataFrame,
    event_rate_aligned_df: pd.DataFrame,
    change_df: pd.DataFrame,
    stats: dict,
    group_order: list[str],
    sig_coverage: dict[str, dict[str, float]],
) -> None:
    first_push_path = output_dir / f"{stem_prefix}_first_significant_push_per_fly.csv"
    event_rate_path = output_dir / f"{stem_prefix}_event_index_aligned_rate_per_event.csv"
    change_path = output_dir / f"{stem_prefix}_post_push_rate_change_per_fly.csv"
    stats_path = output_dir / f"{stem_prefix}_post_push_rate_change_stats.txt"

    first_push_df.to_csv(first_push_path, index=False)
    event_rate_aligned_df.to_csv(event_rate_path, index=False)
    change_df.to_csv(change_path, index=False)

    with open(stats_path, "w") as f:
        f.write("POST-PUSH RATE CHANGE STATISTICS (Nickname vs control)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Significant first-push coverage by group:\n")
        for grp in group_order:
            cov = sig_coverage.get(grp, {"n_sig": 0, "n_total": 0, "pct": np.nan})
            pct_txt = f"{cov['pct']:.2f}" if not np.isnan(cov["pct"]) else "nan"
            f.write(f"  {grp:24s} n_sig={int(cov['n_sig']):3d} / n_total={int(cov['n_total']):3d} ({pct_txt}%)\n")

        f.write("\n")

        f.write("Within-group (delta != 0, sign-permutation, FDR-BH):\n")
        for grp in group_order:
            if grp not in stats.get("within", {}):
                continue
            s = stats["within"][grp]
            f.write(
                f"  {grp:24s} n={s.get('n', 0):3d} "
                f"mean_delta={s.get('mean_delta', np.nan):+8.4f} "
                f"p_raw={s.get('p_value', np.nan):.5f} "
                f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
            )

        f.write("\nBetween-group pairwise delta comparisons (permutation, FDR-BH):\n")
        for _, s in stats.get("between", {}).items():
            f.write(
                f"  {s.get('a')} vs {s.get('b')}: "
                f"diff={s.get('obs_diff', np.nan):+8.4f}, "
                f"p_raw={s.get('p_value', np.nan):.5f}, "
                f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
            )

    print(f"  Saved: {first_push_path}")
    print(f"  Saved: {event_rate_path}")
    print(f"  Saved: {change_path}")
    print(f"  Saved: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNT nickname-vs-control interaction-rate analysis aligned to first significant push",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nickname", type=str, required=True, help="Nickname (line of interest)")
    parser.add_argument(
        "--yaml-path",
        type=str,
        default=DEFAULT_YAML_PATH,
        help="YAML file listing experiment folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (overrides auto output path if provided)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output dir for auto path: {output_root}/{Brain_region}/{Simplified_Nickname}",
    )
    parser.add_argument(
        "--split-registry",
        type=str,
        default=DEFAULT_SPLIT_REGISTRY,
        help="Path to split registry CSV (used when Split is missing in dataset)",
    )
    parser.add_argument(
        "--control-nickname",
        type=str,
        default=None,
        help="Override matched control nickname",
    )
    parser.add_argument(
        "--force-control",
        type=str,
        default=None,
        choices=["Empty-Split", "Empty-Gal4", "TNTxPR"],
        help="Force control for split y/n lines (split m still uses TNTxPR)",
    )

    parser.add_argument("--bin-size", type=float, default=1.0, help="Rate bin size in minutes")
    parser.add_argument(
        "--threshold-px", type=float, default=5.0, help="Significant push threshold (max displacement, px)"
    )
    parser.add_argument(
        "--focus-min",
        type=float,
        default=0.0,
        help="Earliest time (min) allowed for first significant push search",
    )
    parser.add_argument("--window-min", type=float, default=60.0, help="Alignment window around first push (± minutes)")
    parser.add_argument(
        "--event-window",
        type=int,
        default=20,
        help="Alignment window around first significant push in event index units (± events)",
    )
    parser.add_argument("--pre-window-far", type=float, default=10.0, help="Pre window far bound (min before push)")
    parser.add_argument("--pre-window-near", type=float, default=1.0, help="Pre window near bound (min before push)")
    parser.add_argument("--post-window-near", type=float, default=1.0, help="Post window near bound (min after push)")
    parser.add_argument("--post-window-far", type=float, default=10.0, help="Post window far bound (min after push)")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count for tests")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap count for CI in aligned plot")

    parser.add_argument("--test", action="store_true", help="Fast test mode: reduce flies and stats iterations")
    parser.add_argument("--test-max-flies", type=int, default=6, help="Maximum flies per group in test mode")
    parser.add_argument(
        "--test-max-experiments",
        type=int,
        default=None,
        help="Maximum experiments to scan in test mode",
    )
    args = parser.parse_args()

    if args.test:
        args.n_permutations = min(args.n_permutations, 1000)
        args.n_bootstrap = min(args.n_bootstrap, 300)

    nickname = str(args.nickname)
    print("=" * 78)
    print("TNT NICKNAME — FIRST SIGNIFICANT PUSH INTERACTION-RATE ANALYSIS")
    print("=" * 78)
    print(f"YAML path        : {args.yaml_path}")
    print(f"Nickname         : {nickname}")
    print(f"Split registry   : {args.split_registry}")
    if args.output_dir:
        print(f"Output dir       : {args.output_dir}")
    else:
        print(f"Output root      : {args.output_root}")
    print(f"Bin size         : {args.bin_size} min")
    print(f"Threshold        : > {args.threshold_px} px")
    print(f"Focus min        : {args.focus_min} min")
    print(f"Window           : ± {args.window_min} min")
    print(f"Event window     : ± {args.event_window} events")
    print(f"Test mode        : {args.test}")

    t0 = time.time()
    df, control, info = load_nickname_and_control_data(
        yaml_path=Path(args.yaml_path),
        nickname=nickname,
        split_registry_path=Path(args.split_registry),
        control_nickname=args.control_nickname,
        force_control=args.force_control,
        test_mode=args.test,
        test_max_flies_per_group=args.test_max_flies,
        test_max_experiments=args.test_max_experiments,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(args.output_root) / _safe_path_part(info.brain_region) / _safe_path_part(info.simplified_nickname)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolved output  : {output_dir}")
    print(f"Data loaded in {time.time() - t0:.1f}s")

    group_order = [nickname, control]
    group_colors = {nickname: "#4C72B0", control: "#DD8452"}

    onsets = extract_event_onsets(df)
    rate_df = compute_rates_per_fly(onsets, df_full=df, bin_size_min=args.bin_size)
    event_disp_df = compute_ball_displacement_per_event(df)

    first_push_df = compute_first_significant_push(
        event_disp_df,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
    )
    if first_push_df.empty:
        raise RuntimeError("No flies found with a qualifying first significant push. Try lowering --threshold-px.")

    sig_coverage = compute_significant_coverage(df_full=df, first_push_df=first_push_df, group_order=group_order)
    for grp in group_order:
        cov = sig_coverage.get(grp, {"n_sig": 0, "n_total": 0, "pct": np.nan})
        pct_txt = f"{cov['pct']:.1f}%" if not np.isnan(cov["pct"]) else "n.d."
        print(f"  Significant first-push coverage — {grp}: {cov['n_sig']}/{cov['n_total']} ({pct_txt})")

    stem = sanitize_filename(f"nickname_{nickname}_vs_{control}")

    plot_rate_aligned_to_first_push(
        rate_df=rate_df,
        first_push_df=first_push_df,
        group_order=group_order,
        group_colors=group_colors,
        sig_coverage=sig_coverage,
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        output_path=output_dir / f"{stem}_rate_aligned_to_first_significant_push",
    )

    event_rate_aligned_df = compute_event_index_aligned_rates(onsets_df=onsets, first_push_df=first_push_df)
    plot_rate_aligned_to_first_push_event_index(
        event_rate_df=event_rate_aligned_df,
        first_push_df=first_push_df,
        group_order=group_order,
        group_colors=group_colors,
        sig_coverage=sig_coverage,
        window_events=args.event_window,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        output_path=output_dir / f"{stem}_rate_aligned_to_first_significant_push_event_index",
    )

    change_df = compute_post_push_rate_change(
        rate_df=rate_df,
        onsets_df=onsets,
        first_push_df=first_push_df,
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        focus_min=args.focus_min,
    )
    if change_df.empty:
        raise RuntimeError("No flies have both valid pre and post windows. Adjust window arguments.")

    stats = test_post_push_change(change_df, group_order=group_order, n_permutations=args.n_permutations)

    plot_post_push_rate_change(
        change_df=change_df,
        stats=stats,
        group_order=group_order,
        group_colors=group_colors,
        sig_coverage=sig_coverage,
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        n_permutations=args.n_permutations,
        output_path=output_dir / f"{stem}_post_push_rate_change",
    )

    save_tables(
        output_dir=output_dir,
        stem_prefix=stem,
        first_push_df=first_push_df,
        event_rate_aligned_df=event_rate_aligned_df,
        change_df=change_df,
        stats=stats,
        group_order=group_order,
        sig_coverage=sig_coverage,
    )

    print("=" * 78)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
