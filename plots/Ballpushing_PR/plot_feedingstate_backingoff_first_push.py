#!/usr/bin/env python3
"""
FeedingState backing-off analysis aligned to each fly's first significant push.

Goal
----
Quantify continuous retreat bouts (fly moving back toward its own start
position) over the full recording, then test how this evolves around first
significant push across FeedingState conditions.

Outputs
-------
1) Backing-off trajectories (mm/min from continuous retreat bouts) centered at first
   significant push, one panel per FeedingState.
2) Pre/post first-push backing-off boxplots per condition, plus a side panel
   comparing delta = post - pre across conditions.
3) CSV/txt exports for per-bout metrics, per-fly binned metrics, per-fly
    changes, and stats.

Metric definition
-----------------
For each fly:
- Choose reference via `--backoff-reference`:
    - `start`: retreat means moving toward fly start.
    - `ball`: retreat means moving away from ball.
- Compute fly distance to the chosen reference over time.
- Detect contiguous retreat bouts where frame-to-frame reference-distance changes
    in the retreat direction by at least `min_step_px`.
- Bout amplitude = cumulative change in the retreat direction (px).
- Bouts are timestamped by bout start time and binned over absolute time.

Primary per-fly/bin metric used for alignment and pre/post tests:
`backoff_mm_per_min` = total retreat-bout amplitude (mm) / bin duration (min).

Additional normalized metric:
`fractional_backoff_per_bout` = mean over bouts of
(bout_backoff_px / distance_to_start_at_bout_start_px), dimensionless.

Usage
-----
python plot_feedingstate_backingoff_first_push.py [--coordinates-dir PATH]
    [--output-dir PATH] [--bin-size 5] [--threshold-px 5]
    [--focus-min 0] [--window-min 60]
    [--backoff-reference ball|start]
    [--frame-step 3] [--n-permutations 10000]
"""

import argparse
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Union

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

PIXELS_PER_MM = 500 / 30

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
    return series.astype(str).str.strip().map(rename_map)


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
        "x_fly_0",
        "y_fly_0",
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

        df = df.sort_values(["fly", "time"]).copy()
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


def compute_ball_displacement_per_event(df: pd.DataFrame) -> pd.DataFrame:
    event_rows = df[df["interaction_event"].notna()].copy()
    if event_rows.empty:
        return pd.DataFrame(
            columns=[
                "fly",
                "interaction_event",
                "FeedingState",
                "onset_time_s",
                "onset_time_min",
                "max_displacement_px",
            ]
        )

    keys = ["fly", "interaction_event"]
    idx_start = event_rows.groupby(keys)["time"].idxmin()

    start_df = event_rows.loc[
        idx_start,
        ["fly", "interaction_event", "time", "x_ball_0", "y_ball_0", "FeedingState"],
    ].rename(columns={"time": "onset_time_s", "x_ball_0": "ball_x_start", "y_ball_0": "ball_y_start"})

    event_rows = event_rows.merge(
        start_df[["fly", "interaction_event", "ball_x_start", "ball_y_start"]],
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

    out = start_df.merge(max_disp, on=["fly", "interaction_event"], how="left")
    out["onset_time_min"] = out["onset_time_s"] / 60.0
    return out[["fly", "interaction_event", "FeedingState", "onset_time_s", "onset_time_min", "max_displacement_px"]]


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


def _add_reference_distance(df: pd.DataFrame, reference: str = "start") -> pd.DataFrame:
    df = df.sort_values(["fly", "time"]).copy()
    fly_start = df.groupby("fly")[["x_fly_0", "y_fly_0"]].first().reset_index()
    fly_start = fly_start.rename(columns={"x_fly_0": "x_start", "y_fly_0": "y_start"})
    df = df.merge(fly_start, on="fly", how="left")
    df["fly_dist_start_px"] = np.sqrt((df["x_fly_0"] - df["x_start"]) ** 2 + (df["y_fly_0"] - df["y_start"]) ** 2)

    if reference == "start":
        df["reference_dist_px"] = df["fly_dist_start_px"]
    elif reference == "ball":
        df["reference_dist_px"] = np.sqrt((df["x_fly_0"] - df["x_ball_0"]) ** 2 + (df["y_fly_0"] - df["y_ball_0"]) ** 2)
    else:
        raise ValueError(f"Unknown backoff reference: {reference}")
    return df


def _backward_bout_stats_from_dist(
    dist: np.ndarray,
    min_step_px: float,
    moving_away: bool = False,
) -> Tuple[int, float, float, float]:
    if len(dist) < 2:
        return 0, 0.0, 0.0, 0.0

    diffs = np.diff(dist)
    if moving_away:
        back_mask = diffs >= min_step_px
    else:
        back_mask = diffs <= -min_step_px
    if not np.any(back_mask):
        return 0, 0.0, 0.0, 0.0

    amp_steps = np.abs(diffs[back_mask])
    back_idx = np.flatnonzero(back_mask)
    run_starts = np.r_[0, np.flatnonzero(np.diff(back_idx) > 1) + 1]
    run_sums = np.add.reduceat(amp_steps, run_starts)

    n_runs = int(len(run_sums))
    total_backoff_px = float(np.sum(run_sums))
    max_backoff_px = float(np.max(run_sums))
    mean_backoff_px = float(np.mean(run_sums))
    return n_runs, total_backoff_px, max_backoff_px, mean_backoff_px


def compute_backoff_per_event(
    df: pd.DataFrame,
    min_step_px: float = 0.2,
    frame_step: int = 1,
    reference: str = "start",
    chamber_radius_px: float = 50.0,
) -> pd.DataFrame:
    df = _add_reference_distance(df, reference=reference)
    frame_step = max(1, int(frame_step))
    moving_away = reference == "ball"

    fly_frame = df[
        ["fly", "FeedingState", "time", "reference_dist_px", "fly_dist_start_px", "interaction_event"]
    ].copy()
    order = np.lexsort((fly_frame["time"].to_numpy(dtype=float), fly_frame["fly"].astype(str).to_numpy()))
    fly_frame = fly_frame.iloc[order]

    if fly_frame.empty:
        return pd.DataFrame(
            columns=[
                "fly",
                "FeedingState",
                "bout_start_s",
                "bout_end_s",
                "bout_start_min",
                "bout_duration_s",
                "bout_backoff_px",
                "bout_backoff_mm",
                "dist_reference_at_bout_start_px",
                "bout_fractional_retreat",
                "dist_start_at_bout_end_px",
                "ends_in_chamber",
                "start_interaction_event",
                "end_interaction_event",
                "within_same_event",
                "outside_events",
            ]
        )

    out_rows = []
    for fly, sub in tqdm(fly_frame.groupby("fly", sort=False), desc="Computing continuous backoff bouts"):
        time_arr = sub["time"].to_numpy(dtype=float)
        dist_arr = sub["reference_dist_px"].to_numpy(dtype=float)
        dist_start_arr = sub["fly_dist_start_px"].to_numpy(dtype=float)
        event_arr = sub["interaction_event"].to_numpy()
        cond = str(sub["FeedingState"].iloc[0])

        if frame_step > 1:
            time_arr = time_arr[::frame_step]
            dist_arr = dist_arr[::frame_step]
            dist_start_arr = dist_start_arr[::frame_step]
            event_arr = event_arr[::frame_step]

        if len(dist_arr) < 2:
            continue

        diffs = np.diff(dist_arr)
        if moving_away:
            back_mask = diffs >= min_step_px
        else:
            back_mask = diffs <= -min_step_px
        if not np.any(back_mask):
            continue

        back_idx = np.flatnonzero(back_mask)
        run_starts = np.r_[0, np.flatnonzero(np.diff(back_idx) > 1) + 1]
        run_ends = np.r_[run_starts[1:] - 1, len(back_idx) - 1]

        for rs, re in zip(run_starts, run_ends):
            idx_start = int(back_idx[rs])
            idx_end = int(back_idx[re] + 1)

            bout_backoff_px = float(np.sum(np.abs(diffs[idx_start:idx_end])))
            dist_reference_at_bout_start_px = float(dist_arr[idx_start])
            if dist_reference_at_bout_start_px > 1e-9:
                bout_fractional_retreat = float(bout_backoff_px / dist_reference_at_bout_start_px)
            else:
                bout_fractional_retreat = np.nan

            dist_start_at_bout_end_px = float(dist_start_arr[idx_end])
            ends_in_chamber = dist_start_at_bout_end_px <= chamber_radius_px
            start_event = event_arr[idx_start]
            end_event = event_arr[idx_end]
            start_is_nan = pd.isna(start_event)
            end_is_nan = pd.isna(end_event)

            out_rows.append(
                {
                    "fly": fly,
                    "FeedingState": cond,
                    "bout_start_s": float(time_arr[idx_start]),
                    "bout_end_s": float(time_arr[idx_end]),
                    "bout_start_min": float(time_arr[idx_start] / 60.0),
                    "bout_duration_s": float(time_arr[idx_end] - time_arr[idx_start]),
                    "bout_backoff_px": bout_backoff_px,
                    "bout_backoff_mm": bout_backoff_px / PIXELS_PER_MM,
                    "dist_reference_at_bout_start_px": dist_reference_at_bout_start_px,
                    "bout_fractional_retreat": bout_fractional_retreat,
                    "dist_start_at_bout_end_px": dist_start_at_bout_end_px,
                    "ends_in_chamber": bool(ends_in_chamber),
                    "start_interaction_event": start_event,
                    "end_interaction_event": end_event,
                    "within_same_event": (not start_is_nan) and (not end_is_nan) and (start_event == end_event),
                    "outside_events": start_is_nan and end_is_nan,
                }
            )

    out = pd.DataFrame(out_rows)
    print(f"Continuous retreat bouts detected: {len(out)}")
    return out


def bin_backoff_per_fly(
    event_backoff_df: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float = 5.0,
) -> pd.DataFrame:
    time_max_min = df_full["time"].max() / 60.0
    bin_edges = np.arange(0, time_max_min + bin_size_min, bin_size_min)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "FeedingState"]].set_index("fly")
    rows = []

    for fly in df_full["fly"].unique():
        cond = fly_meta.loc[fly, "FeedingState"]
        ev = event_backoff_df[event_backoff_df["fly"] == fly]

        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            in_bin = ev[(ev["bout_start_min"] >= b_start) & (ev["bout_start_min"] < b_end)]
            total_mm = float(in_bin["bout_backoff_mm"].sum()) if len(in_bin) else 0.0

            rows.append(
                {
                    "fly": fly,
                    "FeedingState": cond,
                    "bin_idx": i,
                    "bin_start_min": float(b_start),
                    "bin_center_min": float(bin_centers[i]),
                    "n_bouts": int(len(in_bin)),
                    "total_backoff_mm": total_mm,
                    "backoff_mm_per_min": float(total_mm / bin_size_min),
                    "backoff_mm_per_bout": float(in_bin["bout_backoff_mm"].mean()) if len(in_bin) else np.nan,
                    "fractional_backoff_per_bout": (
                        float(in_bin["bout_fractional_retreat"].mean()) if len(in_bin) else np.nan
                    ),
                    "fractional_backoff_per_min": (
                        float(in_bin["bout_fractional_retreat"].sum() / bin_size_min) if len(in_bin) else 0.0
                    ),
                    "fraction_end_in_chamber": float(in_bin["ends_in_chamber"].mean()) if len(in_bin) else np.nan,
                    "fraction_within_event": float(in_bin["within_same_event"].mean()) if len(in_bin) else np.nan,
                    "bout_duration_s_mean": float(in_bin["bout_duration_s"].mean()) if len(in_bin) else np.nan,
                }
            )

    out = pd.DataFrame(rows).sort_values(["fly", "bin_idx"]).reset_index(drop=True)
    return out


def compute_event_index_aligned_bouts(
    bout_df: pd.DataFrame,
    event_disp_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
) -> pd.DataFrame:
    if bout_df.empty or event_disp_df.empty or first_push_df.empty:
        return pd.DataFrame(
            columns=[
                "fly",
                "FeedingState",
                "rel_event_idx",
                "n_bouts",
                "backoff_mm_per_bout",
                "fractional_backoff_per_bout",
            ]
        )

    event_order = event_disp_df.sort_values(["fly", "onset_time_s"]).copy()
    event_order["event_idx"] = event_order.groupby("fly").cumcount()

    first_idx = first_push_df.merge(
        event_order[["fly", "interaction_event", "event_idx"]],
        left_on=["fly", "first_push_event_id"],
        right_on=["fly", "interaction_event"],
        how="left",
    )[["fly", "event_idx"]].rename(columns={"event_idx": "first_push_event_idx"})

    first_idx_map = dict(zip(first_idx["fly"], first_idx["first_push_event_idx"]))
    onset_map = {fly: sub["onset_time_s"].to_numpy(dtype=float) for fly, sub in event_order.groupby("fly", sort=False)}

    rows = []
    for fly, sub in bout_df.groupby("fly", sort=False):
        if fly not in onset_map:
            continue
        first_event_idx = first_idx_map.get(fly, np.nan)
        if pd.isna(first_event_idx):
            continue

        onset_times = onset_map[fly]
        bout_times = sub["bout_start_s"].to_numpy(dtype=float)
        anchor_idx = np.searchsorted(onset_times, bout_times, side="right") - 1
        rel_idx = anchor_idx - int(first_event_idx)

        fly_sub = sub.copy()
        fly_sub["rel_event_idx"] = rel_idx.astype(int)

        agg = (
            fly_sub.groupby("rel_event_idx", as_index=False)
            .agg(
                n_bouts=("bout_backoff_mm", "size"),
                backoff_mm_per_bout=("bout_backoff_mm", "mean"),
                fractional_backoff_per_bout=("bout_fractional_retreat", "mean"),
            )
            .assign(fly=fly, FeedingState=str(sub["FeedingState"].iloc[0]))
        )
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=[
                "fly",
                "FeedingState",
                "rel_event_idx",
                "n_bouts",
                "backoff_mm_per_bout",
                "fractional_backoff_per_bout",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    return out[
        ["fly", "FeedingState", "rel_event_idx", "n_bouts", "backoff_mm_per_bout", "fractional_backoff_per_bout"]
    ]


def plot_backoff_aligned_to_first_push_event_index(
    event_idx_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    window_events: int,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
    show_raw: bool = False,
    raw_alpha: float = 0.08,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions or event_idx_df.empty:
        print("No event-index aligned data. Skipping event-index plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        sub = event_idx_df[(event_idx_df["FeedingState"] == cond) & (event_idx_df["fly"].isin(flies))].copy()
        sub = sub[(sub["rel_event_idx"] >= -window_events) & (sub["rel_event_idx"] <= window_events)]

        if sub.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        color = FEEDING_COLORS.get(cond, "C0")

        if show_raw:
            for _, fly_sub in sub.groupby("fly"):
                raw_line = (
                    fly_sub[["rel_event_idx", y_col]]
                    .dropna()
                    .groupby("rel_event_idx", as_index=False)[y_col]
                    .mean()
                    .sort_values("rel_event_idx")
                )
                if len(raw_line) >= 2:
                    ax.plot(raw_line["rel_event_idx"], raw_line[y_col], color=color, alpha=raw_alpha, lw=0.8)

        x_sorted = np.sort(sub["rel_event_idx"].unique())
        means, ci_lo, ci_hi = [], [], []
        for rel_idx in x_sorted:
            vals = sub.loc[sub["rel_event_idx"] == rel_idx, y_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            if len(vals) == 1:
                means.append(float(vals[0]))
                ci_lo.append(float(vals[0]))
                ci_hi.append(float(vals[0]))
                continue
            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        if not show_raw:
            ax.fill_between(x_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
            mean_label = f"Mean ± 95% CI (n={len(flies)})"
            mean_lw = 2.5
        else:
            mean_label = f"Mean (n={len(flies)})"
            mean_lw = 3.2
        ax.plot(x_sorted, means, color=color, lw=mean_lw, label=mean_label)
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push event")
        ax.set_xlabel("Event index relative to first push")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(flies)})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel(y_label)
    fig.suptitle(
        "Backing-off aligned to first significant push (event index)\n"
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


def plot_backoff_aligned_to_first_push(
    backoff_bin_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    bin_size_min: float,
    window_min: float,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
    show_raw: bool = False,
    raw_alpha: float = 0.08,
    min_coverage_frac: float = 0.9,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions:
        print("No conditions with first-push flies. Skipping aligned plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        n_cond_flies = len(flies)
        sub = backoff_bin_df[backoff_bin_df["fly"].isin(flies)].copy()
        sub = sub.merge(push_sub[["fly", "t_first_push_min"]], on="fly", how="left")
        sub["rel_time_min"] = sub["bin_center_min"] - sub["t_first_push_min"]
        sub["rel_bin"] = (sub["rel_time_min"] / bin_size_min).round() * bin_size_min
        aligned = sub[sub["rel_bin"].abs() <= window_min].copy()

        if aligned.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        color = FEEDING_COLORS.get(cond, "C0")
        if show_raw:
            for _, fly_sub in aligned.groupby("fly"):
                raw_line = (
                    fly_sub[["rel_bin", y_col]]
                    .dropna()
                    .groupby("rel_bin", as_index=False)[y_col]
                    .mean()
                    .sort_values("rel_bin")
                )
                if len(raw_line) >= 2:
                    ax.plot(raw_line["rel_bin"], raw_line[y_col], color=color, alpha=raw_alpha, lw=0.8)

        bins_sorted = np.sort(aligned["rel_bin"].unique())
        means, ci_lo, ci_hi = [], [], []
        for rb in bins_sorted:
            vals = aligned.loc[aligned["rel_bin"] == rb, y_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue

            if len(vals) == 1:
                means.append(float(vals[0]))
                ci_lo.append(float(vals[0]))
                ci_hi.append(float(vals[0]))
                continue

            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        if not show_raw:
            ax.fill_between(bins_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
            mean_label = f"Mean ± 95% CI (n={len(flies)})"
            mean_lw = 2.5
        else:
            mean_label = f"Mean (n={len(flies)})"
            mean_lw = 3.2
        ax.plot(
            bins_sorted,
            means,
            color=color,
            lw=mean_lw,
            label=mean_label,
        )
        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")

        cov_min_n = max(1, int(np.ceil(min_coverage_frac * max(n_cond_flies, 1))))
        cov = aligned[["fly", "rel_bin", y_col]].dropna().groupby("rel_bin")["fly"].nunique().sort_index()
        dense_bins = cov[cov >= cov_min_n].index.to_numpy(dtype=float)
        if len(dense_bins) > 0:
            x_lo, x_hi = float(np.min(dense_bins)), float(np.max(dense_bins))
            half_bin = max(0.5 * bin_size_min, 1e-6)
            ax.set_xlim(x_lo - half_bin, x_hi + half_bin)

        ax.set_xlabel("Time relative to first push (min)")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(flies)})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel(y_label)
    fig.suptitle(
        f"Backing-off aligned to first significant push\n"
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


def compute_post_push_change(
    metric_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    y_col: str,
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    focus_min: float,
) -> pd.DataFrame:
    all_bins = np.sort(metric_df["bin_center_min"].unique())
    bin_size_min = float(np.median(np.diff(all_bins))) if len(all_bins) >= 2 else 5.0

    rows = []
    for _, push_row in first_push_df.iterrows():
        fly = push_row["fly"]
        cond = push_row["FeedingState"]
        t_push = push_row["t_first_push_min"]

        fly_data = metric_df[metric_df["fly"] == fly].copy()
        fly_data["rel_bin"] = ((fly_data["bin_center_min"] - t_push) / bin_size_min).round() * bin_size_min

        pre_far = -pre_window_min[0]
        pre_near = -pre_window_min[1]
        pre_data = fly_data[(fly_data["rel_bin"] >= pre_far) & (fly_data["rel_bin"] <= pre_near)]
        pre_data = pre_data[pre_data["bin_center_min"] >= focus_min]

        post_near = post_window_min[0]
        post_far = post_window_min[1]
        post_data = fly_data[(fly_data["rel_bin"] >= post_near) & (fly_data["rel_bin"] <= post_far)]

        pre_vals = pre_data[y_col].dropna().values
        post_vals = post_data[y_col].dropna().values

        if len(pre_vals) == 0 or len(post_vals) == 0:
            continue

        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        rows.append(
            {
                "fly": fly,
                "FeedingState": cond,
                "t_first_push_min": float(t_push),
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta": post_mean - pre_mean,
                "n_pre_bins": int(len(pre_vals)),
                "n_post_bins": int(len(post_vals)),
            }
        )

    out = pd.DataFrame(rows)
    for cond, sub in out.groupby("FeedingState"):
        print(f"  {cond}: n={len(sub)} flies with pre/post windows, mean delta={sub['delta'].mean():+.3f}")
    return out


def compute_phase_landmarks(
    df_full: pd.DataFrame,
    first_push_df: pd.DataFrame,
    chamber_radius_px: float,
) -> pd.DataFrame:
    df = df_full[["fly", "FeedingState", "time", "x_fly_0", "y_fly_0", "interaction_event_onset"]].copy()
    order = np.lexsort((df["time"].to_numpy(dtype=float), df["fly"].astype(str).to_numpy()))
    df = df.iloc[order]

    fly_start = df.groupby("fly")[["x_fly_0", "y_fly_0"]].first().reset_index()
    fly_start = fly_start.rename(columns={"x_fly_0": "x_start", "y_fly_0": "y_start"})
    df = df.merge(fly_start, on="fly", how="left")
    df["dist_start_px"] = np.sqrt((df["x_fly_0"] - df["x_start"]) ** 2 + (df["y_fly_0"] - df["y_start"]) ** 2)

    meta = df.groupby("fly", as_index=False).agg(FeedingState=("FeedingState", "first"), t_start_s=("time", "min"))

    exit_df = (
        df[df["dist_start_px"] > chamber_radius_px].groupby("fly", as_index=False).agg(t_chamber_exit_s=("time", "min"))
    )

    interact_df = (
        df[df["interaction_event_onset"].notna()]
        .groupby("fly", as_index=False)
        .agg(t_first_interaction_s=("time", "min"))
    )

    push_df = first_push_df[["fly", "t_first_push_s"]].copy()

    out = (
        meta.merge(exit_df, on="fly", how="left")
        .merge(interact_df, on="fly", how="left")
        .merge(push_df, on="fly", how="left")
    )
    out["has_all_landmarks"] = out[["t_chamber_exit_s", "t_first_interaction_s", "t_first_push_s"]].notna().all(axis=1)
    out["landmark_order_ok"] = (
        (out["t_start_s"] <= out["t_chamber_exit_s"])
        & (out["t_chamber_exit_s"] <= out["t_first_interaction_s"])
        & (out["t_first_interaction_s"] <= out["t_first_push_s"])
    )
    valid = out[out["has_all_landmarks"] & out["landmark_order_ok"]].copy()
    print(f"Phase landmarks valid flies: {len(valid)}/{out['fly'].nunique()}")
    return valid


def _piecewise_phase_map(
    times_s: np.ndarray,
    t_start_s: float,
    t_exit_s: float,
    t_inter_s: float,
    t_push_s: float,
) -> np.ndarray:
    phase = np.full(len(times_s), np.nan, dtype=float)

    if not (t_start_s <= t_exit_s <= t_inter_s <= t_push_s):
        return phase

    eps = 1e-9
    m1 = (times_s >= t_start_s) & (times_s <= t_exit_s)
    m2 = (times_s > t_exit_s) & (times_s <= t_inter_s)
    m3 = (times_s > t_inter_s) & (times_s <= t_push_s)

    d1 = max(t_exit_s - t_start_s, eps)
    d2 = max(t_inter_s - t_exit_s, eps)
    d3 = max(t_push_s - t_inter_s, eps)

    phase[m1] = (times_s[m1] - t_start_s) / d1 * (1.0 / 3.0)
    phase[m2] = (1.0 / 3.0) + (times_s[m2] - t_exit_s) / d2 * (1.0 / 3.0)
    phase[m3] = (2.0 / 3.0) + (times_s[m3] - t_inter_s) / d3 * (1.0 / 3.0)
    return phase


def compute_phase_normalized_metric(
    binned_df: pd.DataFrame,
    landmarks_df: pd.DataFrame,
    y_col: str,
    n_phase_bins: int,
) -> pd.DataFrame:
    if binned_df.empty or landmarks_df.empty:
        return pd.DataFrame(columns=["fly", "FeedingState", "phase_bin", "phase_center", y_col])

    merged = binned_df.merge(
        landmarks_df[
            [
                "fly",
                "FeedingState",
                "t_start_s",
                "t_chamber_exit_s",
                "t_first_interaction_s",
                "t_first_push_s",
            ]
        ],
        on=["fly", "FeedingState"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["fly", "FeedingState", "phase_bin", "phase_center", y_col])

    edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rows = []

    for fly, sub in merged.groupby("fly", sort=False):
        sub = sub.sort_values("bin_center_min")
        t_s = sub["bin_center_min"].to_numpy(dtype=float) * 60.0

        t_start = float(sub["t_start_s"].iloc[0])
        t_exit = float(sub["t_chamber_exit_s"].iloc[0])
        t_inter = float(sub["t_first_interaction_s"].iloc[0])
        t_push = float(sub["t_first_push_s"].iloc[0])
        phase = _piecewise_phase_map(t_s, t_start, t_exit, t_inter, t_push)

        vals = sub[y_col].to_numpy(dtype=float)
        ok = np.isfinite(phase) & np.isfinite(vals) & (phase >= 0.0) & (phase <= 1.0)
        if not np.any(ok):
            continue

        phase_ok = phase[ok]
        vals_ok = vals[ok]
        pbin = np.minimum((phase_ok * n_phase_bins).astype(int), n_phase_bins - 1)

        tmp = pd.DataFrame({"phase_bin": pbin, y_col: vals_ok})
        agg = tmp.groupby("phase_bin", as_index=False)[y_col].mean()
        agg["phase_center"] = centers[agg["phase_bin"].values]
        agg["fly"] = fly
        agg["FeedingState"] = str(sub["FeedingState"].iloc[0])
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["fly", "FeedingState", "phase_bin", "phase_center", y_col])

    out = pd.concat(rows, ignore_index=True)
    return out[["fly", "FeedingState", "phase_bin", "phase_center", y_col]]


def plot_phase_normalized_metric(
    phase_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    n_bootstrap: int,
    output_path: Path,
    show_raw: bool = False,
    raw_alpha: float = 0.08,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions or phase_df.empty:
        print("No phase-normalized data. Skipping phase plot.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        sub = phase_df[(phase_df["FeedingState"] == cond) & (phase_df["fly"].isin(flies))].copy()

        if sub.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        color = FEEDING_COLORS.get(cond, "C0")
        if show_raw:
            for _, fly_sub in sub.groupby("fly"):
                raw_line = (
                    fly_sub[["phase_center", y_col]]
                    .dropna()
                    .groupby("phase_center", as_index=False)[y_col]
                    .mean()
                    .sort_values("phase_center")
                )
                if len(raw_line) >= 2:
                    ax.plot(raw_line["phase_center"], raw_line[y_col], color=color, alpha=raw_alpha, lw=0.8)

        x_sorted = np.sort(sub["phase_center"].unique())
        means, ci_lo, ci_hi = [], [], []
        for x in x_sorted:
            vals = sub.loc[sub["phase_center"] == x, y_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            if len(vals) == 1:
                means.append(float(vals[0]))
                ci_lo.append(float(vals[0]))
                ci_hi.append(float(vals[0]))
                continue
            boot = np.array([bs_rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_bootstrap)])
            means.append(float(np.mean(vals)))
            ci_lo.append(float(np.percentile(boot, 2.5)))
            ci_hi.append(float(np.percentile(boot, 97.5)))

        if not show_raw:
            ax.fill_between(x_sorted, ci_lo, ci_hi, color=color, alpha=0.25)
            mean_label = f"Mean ± 95% CI (n={len(flies)})"
            mean_lw = 2.5
        else:
            mean_label = f"Mean (n={len(flies)})"
            mean_lw = 3.2
        ax.plot(x_sorted, means, color=color, lw=mean_lw, label=mean_label)
        ax.axvline(1.0 / 3.0, color="gray", linestyle="--", lw=1.4, alpha=0.9)
        ax.axvline(2.0 / 3.0, color="gray", linestyle="--", lw=1.4, alpha=0.9)
        ax.axvline(1.0, color="red", linestyle="--", lw=1.8, alpha=0.9)
        ax.set_xlabel("Behavioral phase (0=start, 1/3=exit, 2/3=1st interaction, 1=1st push)")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(flies)})")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel(y_label)
    fig.suptitle("Retreat signal in normalized behavioral phase space", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _one_sample_sign_perm(scores: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
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


def plot_post_push_change(
    change_df: pd.DataFrame,
    stats: dict,
    pre_window_min: Tuple[float, float],
    post_window_min: Tuple[float, float],
    n_permutations: int,
    y_label: str,
    delta_label: str,
    title: str,
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
        ax.set_ylabel(y_label)
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)}\n(n={n})", fontsize=11)

    ax_d = axes[-1]
    delta_positions = np.arange(len(conditions))
    for pos, cond in zip(delta_positions, conditions):
        vals = change_df[change_df["FeedingState"] == cond]["delta"].dropna().values
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
    ax_d.set_ylabel(delta_label)
    ax_d.set_title("Delta comparison\n(pairwise permutation, FDR)", fontsize=11)

    fig.suptitle(
        title,
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
    event_backoff_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    change_df: pd.DataFrame,
    stats: dict,
) -> None:
    first_push_path = output_dir / "feedingstate_first_significant_push_per_fly.csv"
    event_path = output_dir / "feedingstate_continuous_backoff_bouts.csv"
    binned_path = output_dir / "feedingstate_continuous_backoff_binned_per_fly.csv"
    change_path = output_dir / "feedingstate_post_push_backoff_change_per_fly.csv"
    stats_path = output_dir / "feedingstate_post_push_backoff_change_stats.txt"

    first_push_df.to_csv(first_push_path, index=False)
    event_backoff_df.to_csv(event_path, index=False)
    binned_df.to_csv(binned_path, index=False)
    change_df.to_csv(change_path, index=False)

    with open(stats_path, "w") as f:
        f.write("POST-PUSH BACKING-OFF CHANGE STATISTICS (FeedingState)\n")
        f.write("=" * 62 + "\n\n")

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
        for _, s in stats.get("between", {}).items():
            f.write(
                f"  {s.get('a')} vs {s.get('b')}: "
                f"diff={s.get('obs_diff', np.nan):+8.4f}, "
                f"p_raw={s.get('p_value', np.nan):.5f}, "
                f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
            )

    print(f"  Saved: {first_push_path}")
    print(f"  Saved: {event_path}")
    print(f"  Saved: {binned_path}")
    print(f"  Saved: {change_path}")
    print(f"  Saved: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FeedingState backing-off analysis aligned to first significant push",
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
        default=("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/" "feedingstate_first_push_backingoff"),
        help="Directory for outputs",
    )
    parser.add_argument("--bin-size", type=float, default=5.0, help="Time-bin size in minutes")
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
        help="Event-index window around first push event (± number of events)",
    )
    parser.add_argument(
        "--backoff-reference",
        type=str,
        choices=["ball", "start"],
        default="ball",
        help="Reference used to define retreat direction: 'ball' (away from ball) or 'start' (toward fly start)",
    )
    parser.add_argument(
        "--chamber-radius-px",
        type=float,
        default=50.0,
        help="Radius around fly start considered as chamber for bout-end classification",
    )
    parser.add_argument(
        "--min-step-px",
        type=float,
        default=0.2,
        help="Minimum frame-to-frame reference-distance change to count as retreat step",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Use every N-th frame for backoff detection (higher is faster, lower temporal resolution)",
    )
    parser.add_argument("--pre-window-far", type=float, default=10.0, help="Pre window far bound (min before push)")
    parser.add_argument("--pre-window-near", type=float, default=1.0, help="Pre window near bound (min before push)")
    parser.add_argument("--post-window-near", type=float, default=1.0, help="Post window near bound (min after push)")
    parser.add_argument("--post-window-far", type=float, default=10.0, help="Post window far bound (min after push)")
    parser.add_argument("--phase-bins", type=int, default=60, help="Number of bins for phase-normalized analysis")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count for tests")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap count for CI in aligned plot")
    parser.add_argument("--plot-raw", action="store_true", help="Also save aligned plots with raw per-fly overlays")
    parser.add_argument("--raw-alpha", type=float, default=0.08, help="Alpha for raw per-fly overlays")
    parser.add_argument("--test", action="store_true", help="Load only first 2 files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FEEDINGSTATE — BACKING-OFF AROUND FIRST SIGNIFICANT PUSH")
    print("=" * 72)
    print(f"Coordinates dir : {args.coordinates_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Bin size        : {args.bin_size} min")
    print(f"Threshold       : > {args.threshold_px} px")
    print(f"Focus min       : {args.focus_min} min")
    print(f"Align window    : ± {args.window_min} min")
    print(f"Event window    : ± {args.event_window} events")
    print(f"Phase bins      : {args.phase_bins}")
    print(f"Reference       : {args.backoff_reference}")
    print(f"Chamber radius  : {args.chamber_radius_px} px")
    print(f"Plot raw        : {args.plot_raw} (alpha={args.raw_alpha})")
    if args.backoff_reference == "ball":
        print(f"Retreat step    : >= +{args.min_step_px} px / frame (distance to ball)")
    else:
        print(f"Retreat step    : <= -{args.min_step_px} px / frame (distance to start)")
    print(f"Frame step      : {args.frame_step} (1 = full resolution)")

    t0 = time.time()
    df = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"Data loaded in {time.time() - t0:.1f}s")

    event_disp_df = compute_ball_displacement_per_event(df)
    first_push_df = compute_first_significant_push(
        event_disp_df,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
    )
    if first_push_df.empty:
        raise RuntimeError("No flies found with a qualifying first significant push. Try lowering --threshold-px.")

    event_backoff_df = compute_backoff_per_event(
        df,
        min_step_px=args.min_step_px,
        frame_step=args.frame_step,
        reference=args.backoff_reference,
        chamber_radius_px=args.chamber_radius_px,
    )
    if event_backoff_df.empty:
        raise RuntimeError("No continuous retreat bouts detected. Consider lowering --min-step-px or --frame-step.")

    ref_short = "ball" if args.backoff_reference == "ball" else "start"
    abs_y_label = f"Retreat amount (mm / min, relative to {ref_short})"
    frac_y_label = f"Fractional retreat per bout (Δd / d_ref, relative to {ref_short})"

    binned_df = bin_backoff_per_fly(event_backoff_df=event_backoff_df, df_full=df, bin_size_min=args.bin_size)
    event_idx_df = compute_event_index_aligned_bouts(
        bout_df=event_backoff_df,
        event_disp_df=event_disp_df,
        first_push_df=first_push_df,
    )
    landmarks_df = compute_phase_landmarks(df, first_push_df=first_push_df, chamber_radius_px=args.chamber_radius_px)

    if not landmarks_df.empty:
        phase_mm_df = compute_phase_normalized_metric(
            binned_df=binned_df,
            landmarks_df=landmarks_df,
            y_col="backoff_mm_per_min",
            n_phase_bins=args.phase_bins,
        )
        if not phase_mm_df.empty:
            plot_phase_normalized_metric(
                phase_df=phase_mm_df,
                first_push_df=first_push_df,
                y_col="backoff_mm_per_min",
                y_label=abs_y_label,
                n_bootstrap=args.n_bootstrap,
                show_raw=False,
                output_path=output_dir / "feedingstate_phase_normalized_backoff_mm_per_min",
            )
            if args.plot_raw:
                plot_phase_normalized_metric(
                    phase_df=phase_mm_df,
                    first_push_df=first_push_df,
                    y_col="backoff_mm_per_min",
                    y_label=abs_y_label,
                    n_bootstrap=args.n_bootstrap,
                    show_raw=True,
                    raw_alpha=args.raw_alpha,
                    output_path=output_dir / "feedingstate_phase_normalized_backoff_mm_per_min_raw",
                )
            phase_mm_path = output_dir / "feedingstate_phase_normalized_backoff_mm_per_min_per_fly.csv"
            phase_mm_df.to_csv(phase_mm_path, index=False)
            print(f"  Saved: {phase_mm_path}")

        phase_frac_df = compute_phase_normalized_metric(
            binned_df=binned_df,
            landmarks_df=landmarks_df,
            y_col="fractional_backoff_per_bout",
            n_phase_bins=args.phase_bins,
        )
        if not phase_frac_df.empty:
            plot_phase_normalized_metric(
                phase_df=phase_frac_df,
                first_push_df=first_push_df,
                y_col="fractional_backoff_per_bout",
                y_label=frac_y_label,
                n_bootstrap=args.n_bootstrap,
                show_raw=False,
                output_path=output_dir / "feedingstate_phase_normalized_fractional_backoff_per_bout",
            )
            if args.plot_raw:
                plot_phase_normalized_metric(
                    phase_df=phase_frac_df,
                    first_push_df=first_push_df,
                    y_col="fractional_backoff_per_bout",
                    y_label=frac_y_label,
                    n_bootstrap=args.n_bootstrap,
                    show_raw=True,
                    raw_alpha=args.raw_alpha,
                    output_path=output_dir / "feedingstate_phase_normalized_fractional_backoff_per_bout_raw",
                )
            phase_frac_path = output_dir / "feedingstate_phase_normalized_fractional_backoff_per_bout_per_fly.csv"
            phase_frac_df.to_csv(phase_frac_path, index=False)
            print(f"  Saved: {phase_frac_path}")

    plot_backoff_aligned_to_first_push(
        backoff_bin_df=binned_df,
        first_push_df=first_push_df,
        y_col="backoff_mm_per_min",
        y_label=abs_y_label,
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        show_raw=False,
        output_path=output_dir / "feedingstate_backoff_aligned_to_first_significant_push",
    )
    if args.plot_raw:
        plot_backoff_aligned_to_first_push(
            backoff_bin_df=binned_df,
            first_push_df=first_push_df,
            y_col="backoff_mm_per_min",
            y_label=abs_y_label,
            bin_size_min=args.bin_size,
            window_min=args.window_min,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=True,
            raw_alpha=args.raw_alpha,
            output_path=output_dir / "feedingstate_backoff_aligned_to_first_significant_push_raw",
        )

    plot_backoff_aligned_to_first_push(
        backoff_bin_df=binned_df,
        first_push_df=first_push_df,
        y_col="fractional_backoff_per_bout",
        y_label=frac_y_label,
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        show_raw=False,
        output_path=output_dir / "feedingstate_fractional_backoff_aligned_to_first_significant_push",
    )
    if args.plot_raw:
        plot_backoff_aligned_to_first_push(
            backoff_bin_df=binned_df,
            first_push_df=first_push_df,
            y_col="fractional_backoff_per_bout",
            y_label=frac_y_label,
            bin_size_min=args.bin_size,
            window_min=args.window_min,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=True,
            raw_alpha=args.raw_alpha,
            output_path=output_dir / "feedingstate_fractional_backoff_aligned_to_first_significant_push_raw",
        )

    plot_backoff_aligned_to_first_push(
        backoff_bin_df=binned_df,
        first_push_df=first_push_df,
        y_col="fraction_end_in_chamber",
        y_label=f"Fraction of retreat bouts ending in chamber (r ≤ {args.chamber_radius_px:.0f}px)",
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        show_raw=False,
        output_path=output_dir / "feedingstate_fraction_end_in_chamber_aligned_to_first_significant_push",
    )
    if args.plot_raw:
        plot_backoff_aligned_to_first_push(
            backoff_bin_df=binned_df,
            first_push_df=first_push_df,
            y_col="fraction_end_in_chamber",
            y_label=f"Fraction of retreat bouts ending in chamber (r ≤ {args.chamber_radius_px:.0f}px)",
            bin_size_min=args.bin_size,
            window_min=args.window_min,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=True,
            raw_alpha=args.raw_alpha,
            output_path=output_dir / "feedingstate_fraction_end_in_chamber_aligned_to_first_significant_push_raw",
        )

    plot_backoff_aligned_to_first_push(
        backoff_bin_df=binned_df,
        first_push_df=first_push_df,
        y_col="fraction_within_event",
        y_label="Fraction of retreat bouts occurring within events",
        bin_size_min=args.bin_size,
        window_min=args.window_min,
        n_bootstrap=args.n_bootstrap,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
        show_raw=False,
        output_path=output_dir / "feedingstate_fraction_within_event_aligned_to_first_significant_push",
    )
    if args.plot_raw:
        plot_backoff_aligned_to_first_push(
            backoff_bin_df=binned_df,
            first_push_df=first_push_df,
            y_col="fraction_within_event",
            y_label="Fraction of retreat bouts occurring within events",
            bin_size_min=args.bin_size,
            window_min=args.window_min,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=True,
            raw_alpha=args.raw_alpha,
            output_path=output_dir / "feedingstate_fraction_within_event_aligned_to_first_significant_push_raw",
        )

    if not event_idx_df.empty:
        plot_backoff_aligned_to_first_push_event_index(
            event_idx_df=event_idx_df,
            first_push_df=first_push_df,
            y_col="backoff_mm_per_bout",
            y_label=f"Retreat amount (mm / bout, relative to {ref_short})",
            window_events=args.event_window,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=False,
            output_path=output_dir / "feedingstate_backoff_event_index_aligned_to_first_significant_push",
        )
        if args.plot_raw:
            plot_backoff_aligned_to_first_push_event_index(
                event_idx_df=event_idx_df,
                first_push_df=first_push_df,
                y_col="backoff_mm_per_bout",
                y_label=f"Retreat amount (mm / bout, relative to {ref_short})",
                window_events=args.event_window,
                n_bootstrap=args.n_bootstrap,
                threshold_px=args.threshold_px,
                focus_min=args.focus_min,
                show_raw=True,
                raw_alpha=args.raw_alpha,
                output_path=output_dir / "feedingstate_backoff_event_index_aligned_to_first_significant_push_raw",
            )

        plot_backoff_aligned_to_first_push_event_index(
            event_idx_df=event_idx_df,
            first_push_df=first_push_df,
            y_col="fractional_backoff_per_bout",
            y_label=f"Fractional retreat per bout (Δd / d_ref, relative to {ref_short})",
            window_events=args.event_window,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            show_raw=False,
            output_path=output_dir / "feedingstate_fractional_backoff_event_index_aligned_to_first_significant_push",
        )
        if args.plot_raw:
            plot_backoff_aligned_to_first_push_event_index(
                event_idx_df=event_idx_df,
                first_push_df=first_push_df,
                y_col="fractional_backoff_per_bout",
                y_label=f"Fractional retreat per bout (Δd / d_ref, relative to {ref_short})",
                window_events=args.event_window,
                n_bootstrap=args.n_bootstrap,
                threshold_px=args.threshold_px,
                focus_min=args.focus_min,
                show_raw=True,
                raw_alpha=args.raw_alpha,
                output_path=output_dir
                / "feedingstate_fractional_backoff_event_index_aligned_to_first_significant_push_raw",
            )

        event_idx_path = output_dir / "feedingstate_backoff_event_index_aligned_per_fly.csv"
        event_idx_df.to_csv(event_idx_path, index=False)
        print(f"  Saved: {event_idx_path}")

    change_df = compute_post_push_change(
        metric_df=binned_df,
        first_push_df=first_push_df,
        y_col="backoff_mm_per_min",
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        focus_min=args.focus_min,
    )
    if change_df.empty:
        raise RuntimeError("No flies have both valid pre and post windows. Adjust window arguments.")

    stats = test_post_push_change(change_df, n_permutations=args.n_permutations)

    plot_post_push_change(
        change_df=change_df,
        stats=stats,
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        n_permutations=args.n_permutations,
        y_label=abs_y_label,
        delta_label=f"Δ retreat (post − pre, mm/min, relative to {ref_short})",
        title=f"Retreat change around first significant push (relative to {ref_short})",
        output_path=output_dir / "feedingstate_post_push_backoff_change",
    )

    change_df_frac = compute_post_push_change(
        metric_df=binned_df,
        first_push_df=first_push_df,
        y_col="fractional_backoff_per_bout",
        pre_window_min=(args.pre_window_far, args.pre_window_near),
        post_window_min=(args.post_window_near, args.post_window_far),
        focus_min=args.focus_min,
    )
    if not change_df_frac.empty:
        stats_frac = test_post_push_change(change_df_frac, n_permutations=args.n_permutations)
        plot_post_push_change(
            change_df=change_df_frac,
            stats=stats_frac,
            pre_window_min=(args.pre_window_far, args.pre_window_near),
            post_window_min=(args.post_window_near, args.post_window_far),
            n_permutations=args.n_permutations,
            y_label=frac_y_label,
            delta_label=f"Δ fractional retreat (post − pre, relative to {ref_short})",
            title=f"Fractional retreat change around first significant push (relative to {ref_short})",
            output_path=output_dir / "feedingstate_post_push_fractional_backoff_change",
        )
        frac_change_path = output_dir / "feedingstate_post_push_fractional_backoff_change_per_fly.csv"
        frac_stats_path = output_dir / "feedingstate_post_push_fractional_backoff_change_stats.txt"
        change_df_frac.to_csv(frac_change_path, index=False)
        with open(frac_stats_path, "w") as f:
            f.write("POST-PUSH FRACTIONAL BACKOFF CHANGE STATISTICS (FeedingState)\n")
            f.write("=" * 67 + "\n\n")
            f.write("Within-condition (delta != 0, sign-permutation, FDR-BH):\n")
            for cond in FEEDING_ORDER:
                if cond not in stats_frac.get("within", {}):
                    continue
                s = stats_frac["within"][cond]
                f.write(
                    f"  {cond:16s} n={s.get('n', 0):3d} "
                    f"mean_delta={s.get('mean_delta', np.nan):+8.4f} "
                    f"p_raw={s.get('p_value', np.nan):.5f} "
                    f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
                )
            f.write("\nBetween-condition pairwise delta comparisons (permutation, FDR-BH):\n")
            for _, s in stats_frac.get("between", {}).items():
                f.write(
                    f"  {s.get('a')} vs {s.get('b')}: "
                    f"diff={s.get('obs_diff', np.nan):+8.4f}, "
                    f"p_raw={s.get('p_value', np.nan):.5f}, "
                    f"p_fdr={s.get('p_value_fdr', np.nan):.5f}\n"
                )
        print(f"  Saved: {frac_change_path}")
        print(f"  Saved: {frac_stats_path}")

    save_tables(
        output_dir=output_dir,
        first_push_df=first_push_df,
        event_backoff_df=event_backoff_df,
        binned_df=binned_df,
        change_df=change_df,
        stats=stats,
    )

    print("=" * 72)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
