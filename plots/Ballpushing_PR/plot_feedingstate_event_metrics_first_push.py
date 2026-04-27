#!/usr/bin/env python3
"""
FeedingState event-metric trajectories aligned to first significant push.

Computes event-level interaction metrics from raw coordinates, then for each
metric creates a separate aligned plot around each fly's first significant push.
Both time-aligned and event-index-aligned representations are produced.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

METRIC_SPECS: Dict[str, Dict[str, str]] = {
    "iti_prev_s": {
        "label": "Inter-trial interval to previous event (s)",
        "slug": "iti_prev",
    },
    "duration_s": {
        "label": "Interaction event duration (s)",
        "slug": "duration",
    },
    "min_fly_ball_dist_during_event": {
        "label": "Minimum fly-ball distance during event (px)",
        "slug": "min_fly_ball_dist",
    },
    "backoff_max_dist_from_ball_px": {
        "label": "Max fly-ball distance after event before next approach (px)",
        "slug": "backoff_max_dist",
    },
    "backoff_velocity_px_s": {
        "label": "Backoff velocity after event (px/s)",
        "slug": "backoff_velocity",
    },
    "approach_velocity_px_s": {
        "label": "Approach velocity toward next event (px/s)",
        "slug": "approach_velocity",
    },
    "ball_max_displacement_px": {
        "label": "Ball max displacement during event (px)",
        "slug": "ball_max_displacement",
    },
}

DEFAULT_METRICS = [
    "iti_prev_s",
    "duration_s",
    "min_fly_ball_dist_during_event",
    "backoff_max_dist_from_ball_px",
    "backoff_velocity_px_s",
    "approach_velocity_px_s",
    "ball_max_displacement_px",
]


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
        "x_ball_0",
        "y_ball_0",
        "x_fly_0",
        "y_fly_0",
    ]

    chunks = []
    for file_path in files:
        try:
            df = pd.read_feather(file_path)
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


def _build_event_table_per_fly(sub: pd.DataFrame, fps: float) -> pd.DataFrame:
    event_rows = sub[sub["interaction_event"].notna()].copy()
    if event_rows.empty:
        return pd.DataFrame()

    event_rows["fly_ball_dist_px"] = np.sqrt(
        (event_rows["x_fly_0"] - event_rows["x_ball_0"]) ** 2 + (event_rows["y_fly_0"] - event_rows["y_ball_0"]) ** 2
    )

    rows = []
    for event_id, ev in event_rows.groupby("interaction_event", sort=False):
        ev = pd.DataFrame(ev).sort_values(by="time")
        onset_time_s = float(ev["time"].iloc[0])
        end_time_s = float(ev["time"].iloc[-1])
        duration_s = float(len(ev) / fps)

        bx0 = float(ev["x_ball_0"].iloc[0])
        by0 = float(ev["y_ball_0"].iloc[0])
        ball_max_disp_px = float(np.sqrt((ev["x_ball_0"] - bx0) ** 2 + (ev["y_ball_0"] - by0) ** 2).max())

        rows.append(
            {
                "interaction_event": event_id,
                "onset_time_s": onset_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
                "event_end_fly_ball_dist_px": float(ev["fly_ball_dist_px"].iloc[-1]),
                "event_onset_fly_ball_dist_px": float(ev["fly_ball_dist_px"].iloc[0]),
                "min_fly_ball_dist_during_event": float(ev["fly_ball_dist_px"].min()),
                "ball_max_displacement_px": ball_max_disp_px,
            }
        )

    out = pd.DataFrame(rows).sort_values("onset_time_s").reset_index(drop=True)
    out["event_idx"] = np.arange(len(out), dtype=int)
    out["iti_prev_s"] = out["onset_time_s"].diff()
    return out


def _attach_between_event_dynamics(
    sub: pd.DataFrame,
    event_df: pd.DataFrame,
    backward_step_px: float,
    clear_forward_step_px: float,
    clear_forward_min_steps: int,
) -> pd.DataFrame:
    if event_df.empty:
        return event_df

    full = sub.sort_values("time").copy()
    full["fly_ball_dist_px"] = np.sqrt(
        (full["x_fly_0"] - full["x_ball_0"]) ** 2 + (full["y_fly_0"] - full["y_ball_0"]) ** 2
    )

    times = full["time"].to_numpy(dtype=float)
    dists = full["fly_ball_dist_px"].to_numpy(dtype=float)

    event_df = event_df.copy()
    event_df["backoff_max_dist_from_ball_px"] = np.nan
    event_df["backoff_velocity_px_s"] = np.nan
    event_df["approach_velocity_px_s"] = np.nan

    n_events = len(event_df)
    eps = 1e-9
    clear_forward_min_steps = max(1, int(clear_forward_min_steps))

    for idx in range(n_events):
        end_t = float(event_df.at[idx, "end_time_s"])
        end_dist = float(event_df.at[idx, "event_end_fly_ball_dist_px"])

        if idx + 1 >= n_events:
            continue

        next_onset_t = float(event_df.at[idx + 1, "onset_time_s"])
        next_onset_dist = float(event_df.at[idx + 1, "event_onset_fly_ball_dist_px"])

        mid_mask = (times > end_t) & (times < next_onset_t)
        if not np.any(mid_mask):
            continue

        mid_times = times[mid_mask]
        mid_dists = dists[mid_mask]

        stop_idx_exclusive = len(mid_dists)
        if len(mid_dists) >= clear_forward_min_steps + 1:
            d_mid = np.diff(mid_dists)
            forward_mask = d_mid <= -float(clear_forward_step_px)
            run = 0
            for j, is_forward in enumerate(forward_mask):
                if is_forward:
                    run += 1
                    if run >= clear_forward_min_steps:
                        start_forward_idx = j - run + 1
                        stop_idx_exclusive = max(1, start_forward_idx + 1)
                        break
                else:
                    run = 0

        backoff_dists = mid_dists[:stop_idx_exclusive]
        backoff_times = mid_times[:stop_idx_exclusive]
        if len(backoff_dists) == 0:
            continue

        peak_idx = int(np.argmax(backoff_dists))
        peak_dist = float(backoff_dists[peak_idx])
        peak_time = float(backoff_times[peak_idx])
        event_df.at[idx, "backoff_max_dist_from_ball_px"] = peak_dist

        dt_backoff = peak_time - end_t
        if dt_backoff > eps:
            dist_seq = np.r_[end_dist, backoff_dists[: peak_idx + 1]]
            d_back = np.diff(dist_seq)
            backward_only_disp = float(np.sum(d_back[d_back >= float(backward_step_px)]))
            event_df.at[idx, "backoff_velocity_px_s"] = backward_only_disp / dt_backoff

        dt_approach = next_onset_t - peak_time
        if dt_approach > eps:
            event_df.at[idx, "approach_velocity_px_s"] = (peak_dist - next_onset_dist) / dt_approach

    return event_df


def compute_event_feature_table(
    df: pd.DataFrame,
    fps: float,
    backward_step_px: float,
    clear_forward_step_px: float,
    clear_forward_min_steps: int,
) -> pd.DataFrame:
    rows = []
    for fly, sub in df.groupby("fly", sort=False):
        sub = sub.sort_values("time")
        cond = str(sub["FeedingState"].iloc[0])

        event_df = _build_event_table_per_fly(sub, fps=fps)
        if event_df.empty:
            continue

        event_df = _attach_between_event_dynamics(
            sub,
            event_df,
            backward_step_px=backward_step_px,
            clear_forward_step_px=clear_forward_step_px,
            clear_forward_min_steps=clear_forward_min_steps,
        )
        event_df["fly"] = fly
        event_df["FeedingState"] = cond
        event_df["onset_time_min"] = event_df["onset_time_s"] / 60.0
        rows.append(event_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "fly",
            "FeedingState",
            "interaction_event",
            "event_idx",
            "onset_time_s",
            "onset_time_min",
            "iti_prev_s",
            "duration_s",
            "min_fly_ball_dist_during_event",
            "backoff_max_dist_from_ball_px",
            "backoff_velocity_px_s",
            "approach_velocity_px_s",
            "ball_max_displacement_px",
        ]
    ]


def compute_first_significant_push(
    event_df: pd.DataFrame,
    threshold_px: float,
    focus_min: float,
) -> pd.DataFrame:
    eligible = event_df[
        (event_df["onset_time_min"] >= focus_min) & (event_df["ball_max_displacement_px"] > threshold_px)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["fly", "FeedingState", "t_first_push_min", "first_push_event_id"])

    idx_first = eligible.groupby("fly")["onset_time_s"].idxmin()
    out = eligible.loc[idx_first, ["fly", "FeedingState", "onset_time_min", "interaction_event"]].copy()
    out = out.rename(
        columns={
            "onset_time_min": "t_first_push_min",
            "interaction_event": "first_push_event_id",
        }
    )
    print(f"Flies with first significant push (>{threshold_px}px from {focus_min:.1f} min): {len(out)}")
    return out


def build_aligned_metric_bins(
    event_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    metrics: List[str],
    bin_size_min: float,
    window_min: float,
) -> pd.DataFrame:
    rows = []
    push_map = first_push_df.set_index("fly")

    for fly, sub in event_df.groupby("fly", sort=False):
        if fly not in push_map.index:
            continue
        push_t = float(push_map.at[fly, "t_first_push_min"])
        cond = str(push_map.at[fly, "FeedingState"])

        tmp = sub[["onset_time_min", *metrics]].copy()
        tmp["rel_time_min"] = tmp["onset_time_min"] - push_t
        tmp["rel_bin"] = (tmp["rel_time_min"] / bin_size_min).round() * bin_size_min
        tmp = tmp[tmp["rel_bin"].abs() <= window_min]
        if tmp.empty:
            continue

        grouped = tmp.groupby("rel_bin", as_index=False)[metrics].mean(numeric_only=True)
        grouped["fly"] = fly
        grouped["FeedingState"] = cond
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["fly", "FeedingState", "rel_bin", *metrics])

    out = pd.concat(rows, ignore_index=True)
    return out[["fly", "FeedingState", "rel_bin", *metrics]]


def build_aligned_metric_event_index(
    event_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    metrics: List[str],
    event_window: int,
) -> pd.DataFrame:
    rows = []
    push_map = first_push_df.set_index("fly")

    for fly, sub in event_df.groupby("fly", sort=False):
        if fly not in push_map.index:
            continue

        cond = str(push_map.at[fly, "FeedingState"])
        push_event_id = push_map.at[fly, "first_push_event_id"]

        sub = sub.sort_values("event_idx").copy()
        match = sub[sub["interaction_event"] == push_event_id]
        if match.empty:
            continue

        push_idx = int(match["event_idx"].iloc[0])
        sub["rel_event_idx"] = sub["event_idx"].astype(int) - push_idx
        sub = sub[sub["rel_event_idx"].abs() <= int(event_window)]
        if sub.empty:
            continue

        grouped = sub.groupby("rel_event_idx", as_index=False)[metrics].mean(numeric_only=True)
        grouped["fly"] = fly
        grouped["FeedingState"] = cond
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["fly", "FeedingState", "rel_event_idx", *metrics])

    out = pd.concat(rows, ignore_index=True)
    return out[["fly", "FeedingState", "rel_event_idx", *metrics]]


def plot_aligned_metric(
    aligned_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    metric: str,
    y_label: str,
    bin_size_min: float,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    min_coverage_frac: float,
    output_path: Path,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions:
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        n_cond_flies = len(flies)
        sub = aligned_df[aligned_df["FeedingState"] == cond].copy()
        if sub.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        bins_sorted = np.sort(sub["rel_bin"].unique())
        means, ci_lo, ci_hi, n_vals = [], [], [], []

        for rb in bins_sorted:
            vals = sub.loc[sub["rel_bin"] == rb, metric].dropna().to_numpy(dtype=float)
            n_vals.append(len(vals))
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
        ax.plot(bins_sorted, means, color=color, lw=2.5, label=f"Mean ± 95% CI (n={n_cond_flies})")

        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")
        ax.set_xlabel("Time relative to first significant push (min)")
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={n_cond_flies})")

        # Optional coverage-based x-range cropping
        if min_coverage_frac > 0:
            cov_min_n = max(1, int(np.ceil(min_coverage_frac * max(n_cond_flies, 1))))
            dense_bins = bins_sorted[np.asarray(n_vals) >= cov_min_n]
            if len(dense_bins):
                half_bin = 0.5 * bin_size_min
                ax.set_xlim(float(dense_bins.min()) - half_bin, float(dense_bins.max()) + half_bin)

        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel(y_label)
    fig.suptitle(
        f"{y_label} aligned to first significant push\n"
        f"threshold > {threshold_px:.1f}px, push search from {focus_min:.1f} min",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_aligned_metric_event_index(
    aligned_df: pd.DataFrame,
    first_push_df: pd.DataFrame,
    metric: str,
    y_label: str,
    event_window: int,
    n_bootstrap: int,
    threshold_px: float,
    focus_min: float,
    output_path: Path,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in first_push_df["FeedingState"].unique()]
    if not conditions:
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    bs_rng = np.random.default_rng(42)

    for ax, cond in zip(axes, conditions):
        push_sub = first_push_df[first_push_df["FeedingState"] == cond]
        flies = set(push_sub["fly"])
        n_cond_flies = len(flies)
        sub = aligned_df[aligned_df["FeedingState"] == cond].copy()
        if sub.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        bins_sorted = np.sort(sub["rel_event_idx"].unique())
        means, ci_lo, ci_hi = [], [], []

        for rb in bins_sorted:
            vals = sub.loc[sub["rel_event_idx"] == rb, metric].dropna().to_numpy(dtype=float)
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
        ax.plot(bins_sorted, means, color=color, lw=2.5, label=f"Mean ± 95% CI (n={n_cond_flies})")

        ax.axvline(0, color="red", linestyle="--", lw=1.8, alpha=0.9, label="First significant push")
        ax.set_xlabel("Relative interaction-event index")
        ax.set_xlim(-int(event_window) - 0.5, int(event_window) + 0.5)
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={n_cond_flies})")
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel(y_label)
    fig.suptitle(
        f"{y_label} aligned to first significant push (event index)\n"
        f"threshold > {threshold_px:.1f}px, push search from {focus_min:.1f} min",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def parse_metrics(raw: str) -> List[str]:
    keys = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [k for k in keys if k not in METRIC_SPECS]
    if unknown:
        raise ValueError(f"Unknown metric keys: {unknown}. Available: {sorted(METRIC_SPECS.keys())}")
    if not keys:
        raise ValueError("No metrics selected.")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FeedingState event metrics aligned to first significant push",
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
        default=("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/" "feedingstate_event_metrics_first_push"),
        help="Directory for outputs",
    )
    parser.add_argument("--fps", type=float, default=29.0, help="Acquisition frame rate")
    parser.add_argument("--bin-size", type=float, default=1.0, help="Aligned time-bin size (min)")
    parser.add_argument(
        "--bin-size-sec",
        type=float,
        default=None,
        help="Aligned time-bin size in seconds (overrides --bin-size when set)",
    )
    parser.add_argument("--window-min", type=float, default=60.0, help="Aligned time window (±min)")
    parser.add_argument(
        "--event-window",
        type=int,
        default=30,
        help="Aligned event-index window (±events)",
    )
    parser.add_argument("--threshold-px", type=float, default=5.0, help="First significant push threshold (px)")
    parser.add_argument("--focus-min", type=float, default=0.0, help="Earliest time allowed for first push (min)")
    parser.add_argument(
        "--backward-step-px",
        type=float,
        default=0.2,
        help="Minimum positive fly-ball distance step (px/frame) for backoff velocity",
    )
    parser.add_argument(
        "--clear-forward-step-px",
        type=float,
        default=0.2,
        help="Minimum negative fly-ball distance step (px/frame) to stop backoff phase",
    )
    parser.add_argument(
        "--clear-forward-min-steps",
        type=int,
        default=3,
        help="Consecutive clear forward steps required to stop backoff phase",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap count for CI")
    parser.add_argument(
        "--min-coverage-frac",
        type=float,
        default=0.0,
        help="Minimum fraction of flies required in a bin to include x-range; set 0 to disable cutoff",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to compute and plot",
    )
    parser.add_argument("--test", action="store_true", help="Load only first 2 files")
    args = parser.parse_args()

    if args.bin_size_sec is not None:
        if args.bin_size_sec <= 0:
            raise ValueError("--bin-size-sec must be > 0")
        args.bin_size = args.bin_size_sec / 60.0
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    metric_keys = parse_metrics(args.metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FEEDINGSTATE — EVENT METRICS ALIGNED TO FIRST SIGNIFICANT PUSH")
    print("=" * 80)
    print(f"Coordinates dir : {args.coordinates_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Metrics         : {metric_keys}")
    print(f"Bin size        : {args.bin_size:.6f} min ({args.bin_size * 60.0:.3f} s)")
    print(f"Window          : ±{args.window_min} min")
    print(f"Threshold       : >{args.threshold_px} px")
    print(f"Focus min       : {args.focus_min} min")
    print(f"Coverage frac   : {args.min_coverage_frac}")

    t0 = time.time()
    df = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"Data loaded in {time.time() - t0:.1f}s")

    event_df = compute_event_feature_table(
        df=df,
        fps=args.fps,
        backward_step_px=args.backward_step_px,
        clear_forward_step_px=args.clear_forward_step_px,
        clear_forward_min_steps=args.clear_forward_min_steps,
    )
    if event_df.empty:
        raise RuntimeError("No interaction events available.")

    first_push_df = compute_first_significant_push(
        event_df=event_df,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
    )
    if first_push_df.empty:
        raise RuntimeError("No flies with first significant push. Adjust --threshold-px or --focus-min.")

    aligned_df = build_aligned_metric_bins(
        event_df=event_df,
        first_push_df=first_push_df,
        metrics=metric_keys,
        bin_size_min=args.bin_size,
        window_min=args.window_min,
    )
    if aligned_df.empty:
        raise RuntimeError("No aligned bins produced. Check window/threshold settings.")

    aligned_event_idx_df = build_aligned_metric_event_index(
        event_df=event_df,
        first_push_df=first_push_df,
        metrics=metric_keys,
        event_window=args.event_window,
    )
    if aligned_event_idx_df.empty:
        raise RuntimeError("No event-index aligned bins produced. Check event IDs/window settings.")

    event_df.to_csv(output_dir / "feedingstate_event_metrics_per_event.csv", index=False)
    first_push_df.to_csv(output_dir / "feedingstate_event_metrics_first_significant_push_per_fly.csv", index=False)
    aligned_df.to_csv(output_dir / "feedingstate_event_metrics_aligned_binned_per_fly.csv", index=False)
    aligned_event_idx_df.to_csv(output_dir / "feedingstate_event_metrics_aligned_event_index_per_fly.csv", index=False)
    print(f"  Saved: {output_dir / 'feedingstate_event_metrics_per_event.csv'}")
    print(f"  Saved: {output_dir / 'feedingstate_event_metrics_first_significant_push_per_fly.csv'}")
    print(f"  Saved: {output_dir / 'feedingstate_event_metrics_aligned_binned_per_fly.csv'}")
    print(f"  Saved: {output_dir / 'feedingstate_event_metrics_aligned_event_index_per_fly.csv'}")

    for metric in metric_keys:
        spec = METRIC_SPECS[metric]
        plot_aligned_metric(
            aligned_df=aligned_df,
            first_push_df=first_push_df,
            metric=metric,
            y_label=spec["label"],
            bin_size_min=args.bin_size,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            min_coverage_frac=args.min_coverage_frac,
            output_path=output_dir / f"feedingstate_{spec['slug']}_aligned_to_first_significant_push",
        )
        plot_aligned_metric_event_index(
            aligned_df=aligned_event_idx_df,
            first_push_df=first_push_df,
            metric=metric,
            y_label=spec["label"],
            event_window=args.event_window,
            n_bootstrap=args.n_bootstrap,
            threshold_px=args.threshold_px,
            focus_min=args.focus_min,
            output_path=output_dir / f"feedingstate_{spec['slug']}_aligned_to_first_significant_push_event_index",
        )

    print("=" * 80)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
