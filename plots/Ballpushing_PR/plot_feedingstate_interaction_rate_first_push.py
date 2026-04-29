#!/usr/bin/env python3
"""
FeedingState interaction-rate analysis aligned to each fly's first significant push.

Outputs
-------
1) Interaction-rate trajectories (events/min) centered at first significant push,
   plotted separately for each FeedingState condition.
2) Event-index aligned interaction-rate trajectories centered at first significant
    push, using relative interaction-event indices (less sparse than time bins).
2) Pre/post first-push interaction-rate boxplots per condition, plus a side panel
   comparing delta = post - pre across conditions.

Data assumptions
----------------
- Uses coordinate files ("*_coordinates.feather").
- Keeps only Light=on rows and excludes files with "dark" in filename.
- Requires event and ball-position columns to compute first significant push.

Usage
-----
python plot_feedingstate_interaction_rate_first_push.py [--coordinates-dir PATH]
    [--output-dir PATH] [--bin-size 5] [--threshold-px 5]
    [--focus-min 0] [--window-min 60] [--n-permutations 10000]
"""

import argparse
import time
from itertools import combinations
from pathlib import Path
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from ballpushing_utils import read_feather

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

FEEDING_ORDER = ["fed", "starved", "starved_noWater"]
FEEDING_LABELS = {
    "fed": "Fed",
    "starved": "Starved",
    "starved_noWater": "Starved (no water)",
}
FEEDING_COLORS = {
    "fed": "#66C2A5",
    "starved": "#FC8D62",
    "starved_noWater": "#8DA0CB",
}


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


def _normalise_feeding_state(series: pd.Series) -> pd.Series:
    rename_map = {
        "Fed": "fed",
        "fed": "fed",
        "starved": "starved",
        "starved_noWater": "starved_noWater",
        "starved_nowater": "starved_noWater",
    }
    out = series.astype(str).str.strip().map(rename_map)
    return out


def load_coordinates_incrementally(coordinates_dir: Union[str, Path], test_mode: bool = False) -> pd.DataFrame:
    coordinates_dir = Path(coordinates_dir)
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    all_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
    if not all_files:
        raise FileNotFoundError(f"No coordinate feather files found in {coordinates_dir}")

    files = [p for p in all_files if "dark" not in p.name.lower()]
    if test_mode:
        files = files[:2]

    print(f"Total coordinate files found: {len(all_files)}")
    print(f"Files used (non-dark): {len(files)}")

    required = [
        "time",
        "fly",
        "Light",
        "FeedingState",
        "interaction_event",
        "interaction_event_onset",
        "x_ball_0",
        "y_ball_0",
    ]

    chunks = []
    for file_path in files:
        try:
            df = read_feather(file_path)
        except Exception as exc:
            print(f"  Error reading {file_path.name}: {exc}")
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  Skipping {file_path.name}: missing columns {missing}")
            continue

        df = df[df["Light"] == "on"].copy()
        if df.empty:
            continue

        df["FeedingState"] = _normalise_feeding_state(df["FeedingState"])
        df = df[df["FeedingState"].isin(FEEDING_ORDER)].copy()
        if df.empty:
            continue

        df["fly"] = file_path.stem + "::" + df["fly"].astype(str)
        chunks.append(df)

    if not chunks:
        raise ValueError("No usable files after filtering (Light, FeedingState, required columns).")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    print(f"Total flies: {combined['fly'].nunique()}")
    for cond in FEEDING_ORDER:
        sub = combined[combined["FeedingState"] == cond]
        if sub.empty:
            continue
        print(f"  {cond}: {sub['fly'].nunique()} flies, {len(sub)} frames")
    return combined


def extract_event_onsets(df: pd.DataFrame) -> pd.DataFrame:
    non_nan = df[df["interaction_event_onset"].notna()].copy()
    onsets = (
        non_nan.groupby(["fly", "interaction_event_onset"], as_index=False)
        .agg(time=("time", "min"), FeedingState=("FeedingState", "first"))
        .reset_index(drop=True)
    )
    onsets = pd.DataFrame(onsets)
    onsets["time_min"] = onsets["time"] / 60.0
    onsets = onsets.sort_values(["fly", "time"]).reset_index(drop=True)
    onsets["event_idx"] = onsets.groupby("fly").cumcount()
    print(f"Unique interaction onsets: {len(onsets)}")
    return onsets


def compute_rates_per_fly(onsets: pd.DataFrame, df_full: pd.DataFrame, bin_size_min: float = 5.0) -> pd.DataFrame:
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "FeedingState"]].set_index("fly")
    rows = []

    for fly in df_full["fly"].unique():
        cond = fly_meta.loc[fly, "FeedingState"]
        fly_onsets = onsets[onsets["fly"] == fly]
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            count = ((fly_onsets["time_min"] >= b_start) & (fly_onsets["time_min"] < b_end)).sum()
            rows.append(
                {
                    "fly": fly,
                    "FeedingState": cond,
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
        ["fly", "interaction_event", "time", "x_ball_0", "y_ball_0", "FeedingState"],
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

    out = agg.merge(max_disp, on=["fly", "interaction_event"])
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
        return pd.DataFrame(
            columns=["fly", "FeedingState", "t_first_push_s", "t_first_push_min", "first_push_event_id"]
        )

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
    return first_push[["fly", "FeedingState", "t_first_push_s", "t_first_push_min", "first_push_event_id"]]


def plot_rate_aligned_to_first_push(
    rate_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    bin_size_min: float,
    window_min: float,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions:
        print("No conditions with first-push flies. Skipping aligned-rate plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        sub = rate_df[rate_df["fly"].isin(flies)].copy()
        sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
        sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
        sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
        aligned = sub[sub["rel_bin"].abs() <= window_min].copy()

        if aligned.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
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

        color = FEEDING_COLORS.get(cond, "C0")
        ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
        ax.plot(
            bins_sorted,
            means,
            color=color,
            lw=2.5,
            label=f"Mean ± 95% CI (n={len(flies)})",
        )
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")
        ax.set_xlabel("Time relative to first push (min)")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(flies)})")
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


def compute_event_index_aligned_rates(
    onsets: pd.DataFrame,
    first_push_df: pd.DataFrame,
    min_rate_dt_s: float = 1e-6,
) -> pd.DataFrame:
    if onsets.empty or first_push_df.empty:
        return pd.DataFrame(columns=["fly", "FeedingState", "rel_event_idx", "event_rate_per_min", "event_time_min"])

    rows = []
    onsets_by_fly = {fly: sub.sort_values("time").reset_index(drop=True) for fly, sub in onsets.groupby("fly")}

    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        cond = push_row["FeedingState"]
        push_event_id = push_row["first_push_event_id"]

        fly_onsets = onsets_by_fly.get(fly)
        if fly_onsets is None or fly_onsets.empty:
            continue

        match_idx = fly_onsets.index[fly_onsets["interaction_event_onset"] == push_event_id].to_numpy()
        if len(match_idx) == 0:
            continue
        push_idx = int(match_idx[0])

        dt_s = fly_onsets["time"].diff().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(dt_s > min_rate_dt_s, 60.0 / dt_s, np.nan)

        rel_idx = (fly_onsets["event_idx"].to_numpy() - push_idx).astype(int)
        time_min = (fly_onsets["time"].to_numpy() / 60.0).astype(float)

        for ri, rate, tmin in zip(rel_idx, rates, time_min):
            rows.append(
                {
                    "fly": fly,
                    "FeedingState": cond,
                    "rel_event_idx": int(ri),
                    "event_rate_per_min": float(rate) if np.isfinite(rate) else np.nan,
                    "event_time_min": float(tmin),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=["event_rate_per_min"]).reset_index(drop=True)


def plot_rate_aligned_to_first_push_event_index(
    event_idx_rate_df: pd.DataFrame,
    event_window: int,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
) -> None:
    if event_idx_rate_df.empty:
        print("No event-index aligned data. Skipping event-index plot.")
        return

    conditions = [c for c in FEEDING_ORDER if c in event_idx_rate_df["FeedingState"].unique()]
    if not conditions:
        print("No conditions with first-push flies. Skipping event-index aligned-rate plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        sub = event_idx_rate_df[event_idx_rate_df["FeedingState"] == cond].copy()
        flies = set(sub["fly"])
        aligned = sub[sub["rel_event_idx"].abs() <= event_window].copy()

        if aligned.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        bins_sorted = np.sort(aligned["rel_event_idx"].unique())
        means, ci_lo, ci_hi = [], [], []
        for rb in bins_sorted:
            vals = aligned.loc[aligned["rel_event_idx"] == rb, "event_rate_per_min"].values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        color = FEEDING_COLORS.get(cond, "C0")
        ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
        ax.plot(
            bins_sorted,
            means,
            color=color,
            lw=2.5,
            label=f"Mean ± 95% CI (n={len(flies)})",
        )
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")
        ax.set_xlabel("Relative interaction-event index")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(flies)})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Interaction rate (events/min)")
    fig.suptitle(
        f"Interaction rate aligned to first significant push (event index)\n"
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
    first_push_df: pd.DataFrame,
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    focus_min: float,
) -> pd.DataFrame:
    all_bins = np.sort(rate_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 5.0

    rows = []
    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        cond = push_row["FeedingState"]
        t_push = push_row["t_first_push_min"]

        fly_data = rate_df[rate_df["fly"] == fly].copy()
        fly_data["rel_bin"] = ((fly_data["bin_center_min"] - t_push) / bin_size_min).round() * bin_size_min

        pre_far = -pre_window_min[0]
        pre_near = -pre_window_min[1]
        pre_data = fly_data[(fly_data["rel_bin"] >= pre_far) & (fly_data["rel_bin"] <= pre_near)]
        pre_data = pre_data[pre_data["bin_center_min"] >= focus_min]

        post_near = post_window_min[0]
        post_far = post_window_min[1]
        post_data = fly_data[(fly_data["rel_bin"] >= post_near) & (fly_data["rel_bin"] <= post_far)]

        if pre_data.empty or post_data.empty:
            continue

        pre_mean = float(pre_data["rate_per_min"].mean())
        post_mean = float(post_data["rate_per_min"].mean())
        rows.append(
            {
                "fly": fly,
                "FeedingState": cond,
                "t_first_push_min": float(t_push),
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta": post_mean - pre_mean,
                "n_pre_bins": int(len(pre_data)),
                "n_post_bins": int(len(post_data)),
            }
        )

    out = pd.DataFrame(rows)
    for cond, sub in out.groupby("FeedingState"):
        print(f"  {cond}: n={len(sub)} flies with pre/post windows, mean delta={sub['delta'].mean():+.3f}")
    return out


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng) -> float:
    obs = np.abs(np.mean(scores))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(scores))
        if np.abs(np.mean(scores * signs)) >= obs:
            count += 1
    return count / n_permutations


def test_post_push_change(change_df: pd.DataFrame, n_permutations: int) -> dict:
    rng = np.random.default_rng(0)
    results = {"within": {}, "between": {}}

    conditions = [c for c in FEEDING_ORDER if c in change_df["FeedingState"].unique()]

    raw_within_p = []
    for cond in conditions:
        vals = change_df[change_df["FeedingState"] == cond]["delta"].dropna().values
        if len(vals) < 3:
            results["within"][cond] = {"p_value": np.nan, "mean_delta": np.nan, "n": len(vals)}
            raw_within_p.append(1.0)
            continue
        p = _one_sample_sign_perm(vals, n_permutations, rng)
        m = float(np.mean(vals))
        results["within"][cond] = {"p_value": p, "mean_delta": m, "n": len(vals)}
        raw_within_p.append(p)

    if raw_within_p:
        _, p_corr, _, _ = multipletests(raw_within_p, alpha=0.05, method="fdr_bh")
        for i, cond in enumerate(conditions):
            results["within"][cond]["p_value_fdr"] = float(p_corr[i])

    pairwise = list(combinations(conditions, 2))
    raw_between_p = []
    pair_keys = []
    for a, b in pairwise:
        va = change_df[change_df["FeedingState"] == a]["delta"].dropna().values
        vb = change_df[change_df["FeedingState"] == b]["delta"].dropna().values
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
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    n_permutations: int,
    output_path: Path,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in change_df["FeedingState"].unique()]
    if not conditions:
        print("No conditions with post-push change data. Skipping boxplot figure.")
        return

    rng = np.random.default_rng(42)
    n_cols = len(conditions) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 6.0))
    if n_cols == 1:
        axes = [axes]

    pre_label = f"Pre\n(-{int(pre_window_min[0])} to -{int(pre_window_min[1])} min)"
    post_label = f"Post\n(+{int(post_window_min[0])} to +{int(post_window_min[1])} min)"

    all_paired = change_df[change_df["pre_mean"].notna() & change_df["post_mean"].notna()].copy()
    y_min = min(all_paired["pre_mean"].min(), all_paired["post_mean"].min())
    y_max = max(all_paired["pre_mean"].max(), all_paired["post_mean"].max())
    y_rng = y_max - y_min if y_max != y_min else 1.0

    for ax, cond in zip(axes[:-1], conditions):
        sub = change_df[change_df["FeedingState"] == cond].copy()
        sub = sub[sub["pre_mean"].notna() & sub["post_mean"].notna()]
        pre_vals = sub["pre_mean"].values
        post_vals = sub["post_mean"].values
        n = len(sub)
        color = FEEDING_COLORS.get(cond, "C0")

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

        p = stats["within"].get(cond, {}).get("p_value_fdr", np.nan)
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
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)}\n(n={n})", fontsize=11)

    ax_d = axes[-1]
    delta_positions = np.arange(len(conditions))
    max_n = 0
    for pos, cond in zip(delta_positions, conditions):
        vals = change_df[change_df["FeedingState"] == cond]["delta"].dropna().values
        max_n = max(max_n, len(vals))
        color = FEEDING_COLORS.get(cond, "C0")

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
        if a not in conditions or b not in conditions:
            continue
        xa = conditions.index(a)
        xb = conditions.index(b)
        y = d_max + (0.10 + 0.08 * row_idx) * d_rng
        p = pair.get("p_value_fdr", np.nan)
        txt = sig_stars(p) if not np.isnan(p) else "n.d."
        if txt == "ns" and not np.isnan(p):
            txt = f"p={format_p_value(p, n_permutations)}"
        ax_d.plot([xa, xb], [y, y], color="black", lw=1.2)
        ax_d.text((xa + xb) / 2, y + 0.015 * d_rng, txt, ha="center", va="bottom", fontsize=10, color="red")

    ax_d.set_xticks(delta_positions)
    ax_d.set_xticklabels(
        [f"{FEEDING_LABELS.get(c, c)}\n(n={len(change_df[change_df['FeedingState'] == c])})" for c in conditions],
        fontsize=9,
    )
    ax_d.set_xlim(-0.6, len(conditions) - 0.4)
    ax_d.set_ylim(d_min - 0.10 * d_rng, d_max + (0.22 + 0.08 * len(valid_pairs)) * d_rng)
    ax_d.set_ylabel("Δ rate (post − pre, events/min)")
    ax_d.set_title("Delta comparison\n(pairwise permutation, FDR)", fontsize=11)

    fig.suptitle(
        "Interaction-rate increase around first significant push",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def save_tables(
    output_dir: Path,
    first_push_df: pd.DataFrame,
    change_df: pd.DataFrame,
    stats: dict,
) -> None:
    first_push_path = output_dir / "feedingstate_first_significant_push_per_fly.csv"
    change_path = output_dir / "feedingstate_post_push_rate_change_per_fly.csv"
    stats_path = output_dir / "feedingstate_post_push_rate_change_stats.txt"

    first_push_df.to_csv(first_push_path, index=False)
    change_df.to_csv(change_path, index=False)

    with open(stats_path, "w") as f:
        f.write("POST-PUSH RATE CHANGE STATISTICS (FeedingState)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Within-condition (delta != 0, sign-permutation, FDR-BH):\n")
        for cond in FEEDING_ORDER:
            if cond not in stats.get("within", {}):
                continue
            s = stats["within"][cond]
            f.write(
                f"  {cond:16s} n={s.get('n', 0):3d} "
                f"mean_delta={s.get('mean_delta', np.nan):+8.4f} "
                f"p_raw={s.get('p_value', np.nan):.5f} "
                f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
            )

        f.write("\nBetween-condition pairwise delta comparisons (permutation, FDR-BH):\n")
        for key, s in stats.get("between", {}).items():
            f.write(
                f"  {s.get('a')} vs {s.get('b')}: "
                f"diff={s.get('obs_diff', np.nan):+8.4f}, "
                f"p_raw={s.get('p_value', np.nan):.5f}, "
                f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
            )

    print(f"  Saved: {first_push_path}")
    print(f"  Saved: {change_path}")
    print(f"  Saved: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FeedingState interaction-rate analysis aligned to first significant push",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates-dir",
        type=str,
        default=(
            "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
            "260220_10_summary_control_folders_Data/coordinates"
        ),
        help="Directory containing *_coordinates.feather files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/" "feedingstate_first_push_interaction_rate"),
        help="Directory for outputs",
    )
    parser.add_argument("--bin-size", type=float, default=5.0, help="Rate bin size in minutes")
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
        default=30,
        help="Alignment window around first push (± event indices) for event-index plot",
    )
    parser.add_argument("--pre-window-far", type=float, default=10.0, help="Pre window far bound (min before push)")
    parser.add_argument("--pre-window-near", type=float, default=1.0, help="Pre window near bound (min before push)")
    parser.add_argument("--post-window-near", type=float, default=1.0, help="Post window near bound (min after push)")
    parser.add_argument("--post-window-far", type=float, default=10.0, help="Post window far bound (min after push)")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count for tests")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap count for CI in aligned plot")
    parser.add_argument("--test", action="store_true", help="Load only first 2 files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEEDINGSTATE — FIRST SIGNIFICANT PUSH INTERACTION-RATE ANALYSIS")
    print("=" * 70)
    print(f"Coordinates dir : {args.coordinates_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Bin size        : {args.bin_size} min")
    print(f"Threshold       : > {args.threshold_px} px")
    print(f"Focus min       : {args.focus_min} min")
    print(f"Window          : ± {args.window_min} min")

    t0 = time.time()
    df = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"Data loaded in {time.time() - t0:.1f}s")

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

    plot_rate_aligned_to_first_push(
        rate_df=rate_df,
        first_push_df=first_push_df,
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        output_path=output_dir / "feedingstate_rate_aligned_to_first_significant_push",
    )

    event_idx_rate_df = compute_event_index_aligned_rates(onsets=onsets, first_push_df=first_push_df)
    plot_rate_aligned_to_first_push_event_index(
        event_idx_rate_df=event_idx_rate_df,
        event_window=args.event_window,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        output_path=output_dir / "feedingstate_rate_aligned_to_first_significant_push_event_index",
    )

    change_df = compute_post_push_rate_change(
        rate_df=rate_df,
        first_push_df=first_push_df,
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        focus_min=args.focus_min,
    )
    if change_df.empty:
        raise RuntimeError("No flies have both valid pre and post windows. Adjust window arguments.")

    stats = test_post_push_change(change_df, n_permutations=args.n_permutations)

    plot_post_push_rate_change(
        change_df=change_df,
        stats=stats,
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        n_permutations=args.n_permutations,
        output_path=output_dir / "feedingstate_post_push_rate_change",
    )

    save_tables(output_dir=output_dir, first_push_df=first_push_df, change_df=change_df, stats=stats)

    print("=" * 70)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
