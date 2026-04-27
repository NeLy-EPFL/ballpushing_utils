#!/usr/bin/env python3
"""
FeedingState interaction-event UMAP analysis.

Builds one feature vector per interaction event, runs a quality-controlled
preprocessing + UMAP pipeline, and generates embeddings colored by either:
- event onset time, or
- event index within fly.

Special markers:
- first significant push event (ball displacement > --significant-threshold-px)
- first major push event (ball displacement > --major-threshold-px)

Notes
-----
All UMAP features are derived in this script from frame-level coordinates and
event IDs (e.g., ITI, event duration, between-event backoff/approach dynamics,
and per-event ball displacement). They are not assumed to exist as precomputed
columns in the input coordinate files.

Usage
-----
python plot_feedingstate_event_umap.py [--coordinates-dir PATH] [--output-dir PATH]
    [--features f1,f2,...] [--focus-min 0]
    [--significant-threshold-px 5] [--major-threshold-px 20]
    [--n-neighbors 30] [--min-dist 0.1] [--random-state 42] [--test]
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

try:
    import umap.umap_ as umap_module
except Exception:
    import umap as umap_module

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


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    column: str
    label: str


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "iti_prev_s": FeatureSpec(
        key="iti_prev_s",
        column="iti_prev_s",
        label="Inter-trial interval to previous event (s)",
    ),
    "duration_s": FeatureSpec(
        key="duration_s",
        column="duration_s",
        label="Interaction event duration (s)",
    ),
    "backoff_max_dist_from_ball_px": FeatureSpec(
        key="backoff_max_dist_from_ball_px",
        column="backoff_max_dist_from_ball_px",
        label="Max fly-ball distance after event, before next approach (px)",
    ),
    "min_fly_ball_dist_during_event": FeatureSpec(
        key="min_fly_ball_dist_during_event",
        column="min_fly_ball_dist_during_event",
        label="Minimum fly-ball distance during event (px)",
    ),
    "backoff_speed_px_s": FeatureSpec(
        key="backoff_speed_px_s",
        column="backoff_speed_px_s",
        label="Backoff speed after event (px/s)",
    ),
    "approach_speed_px_s": FeatureSpec(
        key="approach_speed_px_s",
        column="approach_speed_px_s",
        label="Approach speed toward next event (px/s)",
    ),
    "ball_max_displacement_px": FeatureSpec(
        key="ball_max_displacement_px",
        column="ball_max_displacement_px",
        label="Ball max displacement during event (px)",
    ),
    "event_idx_normalized": FeatureSpec(
        key="event_idx_normalized",
        column="event_idx_normalized",
        label="Event index normalized within fly (0..1)",
    ),
    "cumulative_ball_disp_before": FeatureSpec(
        key="cumulative_ball_disp_before",
        column="cumulative_ball_disp_before",
        label="Cumulative ball displacement before event (px)",
    ),
    "rolling_backoff_mean_5": FeatureSpec(
        key="rolling_backoff_mean_5",
        column="rolling_backoff_mean_5",
        label="Rolling mean backoff distance over last 5 events (px)",
    ),
    "rolling_interaction_rate_5": FeatureSpec(
        key="rolling_interaction_rate_5",
        column="rolling_interaction_rate_5",
        label="Rolling interaction rate from previous 5 events (events/min)",
    ),
    "events_since_last_significant": FeatureSpec(
        key="events_since_last_significant",
        column="events_since_last_significant",
        label="Event count since previous significant push",
    ),
}

DEFAULT_FEATURE_KEYS = [
    "iti_prev_s",
    "duration_s",
    "min_fly_ball_dist_during_event",
    "backoff_max_dist_from_ball_px",
    "backoff_speed_px_s",
    "approach_speed_px_s",
    "event_idx_normalized",
    "cumulative_ball_disp_before",
    "rolling_backoff_mean_5",
    "rolling_interaction_rate_5",
    "events_since_last_significant",
]

STATE_FEATURE_KEYS = [
    "iti_prev_s",
    "duration_s",
    "backoff_max_dist_from_ball_px",
    "backoff_speed_px_s",
    "approach_speed_px_s",
    "rolling_backoff_mean_5",
    "rolling_interaction_rate_5",
    "min_fly_ball_dist_during_event",
]

PROGRESSION_FEATURE_KEYS = [
    "event_idx_normalized",
    "cumulative_ball_disp_before",
    "events_since_last_significant",
    "rolling_interaction_rate_5",
]

SKIP_WITHIN_FLY_NORM = {
    "event_idx_normalized",
    "cumulative_ball_disp_before",
    "events_since_last_significant",
}


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

    grouped = []
    for event_id, ev in event_rows.groupby("interaction_event", sort=False):
        ev = pd.DataFrame(ev).sort_values(by="time")
        onset_time_s = float(ev["time"].iloc[0])
        end_time_s = float(ev["time"].iloc[-1])
        duration_s = float(len(ev) / fps)

        bx0 = float(ev["x_ball_0"].iloc[0])
        by0 = float(ev["y_ball_0"].iloc[0])
        ball_max_disp_px = float(np.sqrt((ev["x_ball_0"] - bx0) ** 2 + (ev["y_ball_0"] - by0) ** 2).max())

        grouped.append(
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

    out = pd.DataFrame(grouped).sort_values("onset_time_s").reset_index(drop=True)
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
    event_df["backoff_speed_px_s"] = np.nan
    event_df["approach_speed_px_s"] = np.nan

    n_events = len(event_df)
    eps = 1e-9

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

        clear_forward_min_steps = max(1, int(clear_forward_min_steps))

        # Backoff spans out-of-event frames after focal event, and is truncated only
        # when a clear new forward movement (toward ball) begins.
        stop_idx_exclusive = len(mid_dists)
        if len(mid_dists) >= clear_forward_min_steps + 1:
            d_mid = np.diff(mid_dists)
            forward_mask = d_mid <= -float(clear_forward_step_px)
            run = 0
            for j, is_forward in enumerate(forward_mask):
                if is_forward:
                    run += 1
                    if run >= clear_forward_min_steps:
                        # j is index in diffs, so forward run starts at distance index (j-run+1)
                        start_forward_idx = j - run + 1
                        stop_idx_exclusive = max(1, start_forward_idx + 1)
                        break
                else:
                    run = 0

        backoff_dists = mid_dists[:stop_idx_exclusive]
        backoff_times = mid_times[:stop_idx_exclusive]
        if len(backoff_dists) == 0:
            continue

        peak_local_idx = int(np.argmax(backoff_dists))
        peak_dist = float(backoff_dists[peak_local_idx])
        peak_time = float(backoff_times[peak_local_idx])

        event_df.at[idx, "backoff_max_dist_from_ball_px"] = peak_dist

        dt_backoff = peak_time - end_t
        if dt_backoff > eps:
            # Backoff speed: only backward (away-from-ball) movement up to max distance.
            dist_seq = np.r_[end_dist, backoff_dists[: peak_local_idx + 1]]
            d_back = np.diff(dist_seq)
            backward_only_disp = float(np.sum(d_back[d_back >= float(backward_step_px)]))
            event_df.at[idx, "backoff_speed_px_s"] = backward_only_disp / dt_backoff

        dt_approach = next_onset_t - peak_time
        if dt_approach > eps:
            event_df.at[idx, "approach_speed_px_s"] = (peak_dist - next_onset_dist) / dt_approach

    return event_df


def compute_event_feature_table(
    df: pd.DataFrame,
    fps: float,
    significant_threshold_px: float,
    major_threshold_px: float,
    focus_min: float,
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

        sig_candidates = event_df[
            (event_df["onset_time_min"] >= focus_min)
            & (event_df["ball_max_displacement_px"] > significant_threshold_px)
        ]
        major_candidates = event_df[
            (event_df["onset_time_min"] >= focus_min) & (event_df["ball_max_displacement_px"] > major_threshold_px)
        ]

        event_df["is_first_significant"] = False
        event_df["is_first_major"] = False

        if not sig_candidates.empty:
            first_idx = int(sig_candidates["onset_time_s"].idxmin())
            event_df.loc[first_idx, "is_first_significant"] = True

        if not major_candidates.empty:
            first_idx = int(major_candidates["onset_time_s"].idxmin())
            event_df.loc[first_idx, "is_first_major"] = True

        max_idx = max(int(event_df["event_idx"].max()), 1)
        event_df["event_idx_normalized"] = event_df["event_idx"] / float(max_idx)

        event_df["cumulative_ball_disp_before"] = event_df["ball_max_displacement_px"].shift(1).fillna(0.0).cumsum()

        event_df["rolling_backoff_mean_5"] = event_df["backoff_max_dist_from_ball_px"].rolling(5, min_periods=1).mean()

        dt5_min = event_df["onset_time_s"].diff(5) / 60.0
        event_df["rolling_interaction_rate_5"] = 5.0 / dt5_min.replace(0.0, np.nan)

        sig_mask = (event_df["onset_time_min"] >= focus_min) & (
            event_df["ball_max_displacement_px"] > significant_threshold_px
        )
        since_vals: List[int] = []
        last_sig_local_idx = None
        for i in range(len(event_df)):
            if last_sig_local_idx is None:
                since_vals.append(i)
            else:
                since_vals.append(i - last_sig_local_idx)
            if bool(sig_mask.iloc[i]):
                last_sig_local_idx = i
        event_df["events_since_last_significant"] = since_vals

        first_sig_events = event_df[event_df["is_first_significant"]]["event_idx"].to_numpy(dtype=float)
        if len(first_sig_events) > 0:
            event_df["event_idx_rel_to_first_sig"] = event_df["event_idx"] - float(first_sig_events[0])
        else:
            event_df["event_idx_rel_to_first_sig"] = np.nan

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
            "end_time_s",
            "iti_prev_s",
            "duration_s",
            "min_fly_ball_dist_during_event",
            "backoff_max_dist_from_ball_px",
            "backoff_speed_px_s",
            "approach_speed_px_s",
            "ball_max_displacement_px",
            "event_idx_normalized",
            "cumulative_ball_disp_before",
            "rolling_backoff_mean_5",
            "rolling_interaction_rate_5",
            "events_since_last_significant",
            "event_idx_rel_to_first_sig",
            "is_first_significant",
            "is_first_major",
        ]
    ]


def parse_feature_keys(raw: str) -> List[str]:
    keys = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [k for k in keys if k not in FEATURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown feature keys: {unknown}. Available: {sorted(FEATURE_SPECS.keys())}")
    if not keys:
        raise ValueError("No feature keys selected.")
    return keys


def prepare_feature_matrix(
    event_df: pd.DataFrame,
    feature_keys: Sequence[str],
    clip_quantile: float,
    use_pca: bool,
    pca_components: int,
    within_fly_normalize: bool,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, float]]:
    feature_cols = [FEATURE_SPECS[k].column for k in feature_keys]
    work = event_df.copy()

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)

    valid_mask = ~work[feature_cols].isna().all(axis=1)
    removed_all_nan = int((~valid_mask).sum())
    work = work[valid_mask].copy()
    if work.empty:
        raise ValueError("All rows have all selected features missing after cleaning.")

    if within_fly_normalize:
        for col in feature_cols:
            if col in SKIP_WITHIN_FLY_NORM:
                continue
            fly_median = work.groupby("fly")[col].transform("median")
            fly_iqr = work.groupby("fly")[col].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
            work[col] = (work[col] - fly_median) / (fly_iqr + 1e-9)

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(work[feature_cols].to_numpy(dtype=float))

    if clip_quantile > 0:
        q_lo = np.quantile(x_imputed, clip_quantile, axis=0)
        q_hi = np.quantile(x_imputed, 1.0 - clip_quantile, axis=0)
        x_imputed = np.clip(x_imputed, q_lo, q_hi)

    keep_idx = []
    dropped_low_var = []
    for idx, col in enumerate(feature_cols):
        std = float(np.std(x_imputed[:, idx]))
        if std <= 1e-12:
            dropped_low_var.append(col)
        else:
            keep_idx.append(idx)

    if not keep_idx:
        raise ValueError("All selected features have near-zero variance after preprocessing.")

    kept_cols = [feature_cols[i] for i in keep_idx]
    x_kept = x_imputed[:, keep_idx]

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_kept)

    if use_pca and x_scaled.shape[1] > 2:
        n_comp = max(2, min(pca_components, x_scaled.shape[1], x_scaled.shape[0] - 1))
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=0)
            x_final = pca.fit_transform(x_scaled)
        else:
            x_final = x_scaled
    else:
        x_final = x_scaled

    prep_stats = {
        "n_input_rows": float(len(event_df)),
        "n_rows_for_umap": float(len(work)),
        "n_removed_all_nan": float(removed_all_nan),
        "n_features_requested": float(len(feature_cols)),
        "n_features_used": float(len(kept_cols)),
    }

    if dropped_low_var:
        print(f"Dropped low-variance features: {dropped_low_var}")

    return work, x_final, kept_cols, prep_stats


def fit_umap_embeddings(
    x: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    random_state: int,
    metric: str,
) -> np.ndarray:
    if x.shape[0] < 5:
        raise ValueError("Need at least 5 events to fit UMAP.")

    n_neighbors_eff = int(min(max(2, n_neighbors), x.shape[0] - 1))
    if n_neighbors_eff != n_neighbors:
        print(f"Adjusted n_neighbors from {n_neighbors} to {n_neighbors_eff} for sample size.")

    model = umap_module.UMAP(
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        metric=metric,
        random_state=random_state,
    )
    emb = model.fit_transform(x)
    return np.asarray(emb, dtype=float)


def _draw_phase_centroid_arrows(ax: Axes, sub: pd.DataFrame) -> None:
    sub = _assign_phase_bins(sub.copy())
    centroids: Dict[str, np.ndarray] = {}
    for phase in ["early", "mid", "late"]:
        pts = sub[sub["phase_bin"] == phase][["UMAP1", "UMAP2"]].to_numpy(dtype=float)
        if len(pts) >= 5:
            centroids[phase] = np.mean(pts, axis=0)

    for (p0, p1), color in zip([("early", "mid"), ("mid", "late")], ["#2196F3", "#FF5722"]):
        if p0 in centroids and p1 in centroids:
            x0, y0 = centroids[p0]
            x1, y1 = centroids[p1]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=20),
                zorder=7,
            )

    for phase, color in zip(["early", "mid", "late"], ["#2196F3", "#9C27B0", "#FF5722"]):
        if phase in centroids:
            cx, cy = centroids[phase]
            ax.scatter(
                cx,
                cy,
                s=120,
                color=color,
                zorder=8,
                label=f"{phase.capitalize()} centroid",
                edgecolors="white",
                linewidths=1.2,
            )


def _plot_one_condition(
    ax: Axes,
    sub: pd.DataFrame,
    color_col: str,
    color_label: str,
    cmap: str,
    cond: str,
    significant_threshold_px: float,
    major_threshold_px: float,
    draw_trajectories: bool,
    draw_average_arrow: bool,
) -> Axes:
    if sub.empty:
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
        ax.set_xlabel("UMAP1")
        return ax

    v = sub[color_col].to_numpy(dtype=float)
    sc = ax.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        c=v,
        cmap=cmap,
        s=16,
        alpha=0.85,
        edgecolors="none",
    )

    if draw_trajectories:
        for _, grp in sub.sort_values(["fly", "event_idx"]).groupby("fly", sort=False):
            if len(grp) < 2:
                continue
            ax.plot(
                grp["UMAP1"].to_numpy(dtype=float),
                grp["UMAP2"].to_numpy(dtype=float),
                color="gray",
                alpha=0.08,
                linewidth=0.4,
                zorder=1,
            )

    if draw_average_arrow:
        _draw_phase_centroid_arrows(ax=ax, sub=sub)

    sig = sub[sub["is_first_significant"]]
    if not sig.empty:
        ax.scatter(
            sig["UMAP1"],
            sig["UMAP2"],
            marker="+",
            c="black",
            s=120,
            linewidths=1.8,
            label=f"First significant (>{significant_threshold_px:g} px)",
            zorder=4,
        )

    major = sub[sub["is_first_major"]]
    if not major.empty:
        ax.scatter(
            major["UMAP1"],
            major["UMAP2"],
            marker="x",
            c="red",
            s=95,
            linewidths=1.8,
            label=f"First major (>{major_threshold_px:g} px)",
            zorder=5,
        )

    ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(sub)})")
    ax.set_xlabel("UMAP1")
    ax.grid(alpha=0.2, linewidth=0.4)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label)
    return ax


def plot_umap_by_condition(
    umap_df: pd.DataFrame,
    color_col: str,
    color_label: str,
    cmap: str,
    title: str,
    output_path: Path,
    significant_threshold_px: float,
    major_threshold_px: float,
    draw_trajectories: bool,
    draw_average_arrow: bool,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in umap_df["FeedingState"].unique()]
    if not conditions:
        print("No FeedingState conditions available for plotting.")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharex=True, sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = pd.DataFrame(umap_df[umap_df["FeedingState"] == cond].copy())
        _plot_one_condition(
            ax=ax,
            sub=sub,
            color_col=color_col,
            color_label=color_label,
            cmap=cmap,
            cond=cond,
            significant_threshold_px=significant_threshold_px,
            major_threshold_px=major_threshold_px,
            draw_trajectories=draw_trajectories,
            draw_average_arrow=draw_average_arrow,
        )

    axes[0].set_ylabel("UMAP2")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _assign_phase_bins(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["phase_bin"] = "mid"

    rel = out["event_idx_rel_to_first_sig"].astype(float)
    valid_rel = rel[rel.notna()]
    if len(valid_rel) >= 9:
        quantiles = np.asarray(np.quantile(valid_rel.to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0]), dtype=float)
        q1 = float(quantiles[0])
        q2 = float(quantiles[1])
        out.loc[rel <= q1, "phase_bin"] = "early"
        out.loc[(rel > q1) & (rel <= q2), "phase_bin"] = "mid"
        out.loc[rel > q2, "phase_bin"] = "late"
        return out

    norm_idx = out["event_idx_normalized"].astype(float)
    out.loc[norm_idx <= 1.0 / 3.0, "phase_bin"] = "early"
    out.loc[(norm_idx > 1.0 / 3.0) & (norm_idx <= 2.0 / 3.0), "phase_bin"] = "mid"
    out.loc[norm_idx > 2.0 / 3.0, "phase_bin"] = "late"
    return out


def plot_umap_phase_density(
    umap_df: pd.DataFrame,
    output_path: Path,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in umap_df["FeedingState"].unique()]
    if not conditions:
        return

    phases = ["early", "mid", "late"]
    phase_title = {"early": "Early", "mid": "Mid", "late": "Late"}

    fig, axes = plt.subplots(
        len(conditions),
        len(phases),
        figsize=(5.0 * len(phases), 4.2 * len(conditions)),
        sharex=True,
        sharey=True,
    )
    if len(conditions) == 1:
        axes = np.array([axes])

    for i, cond in enumerate(conditions):
        sub_cond = _assign_phase_bins(pd.DataFrame(umap_df[umap_df["FeedingState"] == cond].copy()))
        for j, phase in enumerate(phases):
            ax = axes[i, j]
            sub = sub_cond[sub_cond["phase_bin"] == phase]
            if sub.empty:
                ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} · {phase_title[phase]} (n=0)")
                ax.grid(alpha=0.15, linewidth=0.4)
                continue

            hb = ax.hexbin(
                sub["UMAP1"].to_numpy(dtype=float),
                sub["UMAP2"].to_numpy(dtype=float),
                gridsize=45,
                mincnt=1,
                cmap="magma",
                linewidths=0.0,
            )
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} · {phase_title[phase]} (n={len(sub)})")
            ax.grid(alpha=0.15, linewidth=0.4)
            if j == len(phases) - 1:
                cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Local density")

    for j in range(len(phases)):
        axes[-1, j].set_xlabel("UMAP1")
    for i in range(len(conditions)):
        axes[i, 0].set_ylabel("UMAP2")

    fig.suptitle("FeedingState event UMAP density by progression phase", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_kde_phase_overlay(
    umap_df: pd.DataFrame,
    output_path: Path,
    bw_method: float = 0.3,
) -> None:
    conditions = [c for c in FEEDING_ORDER if c in umap_df["FeedingState"].unique()]
    if not conditions:
        return

    phase_colors = {"early": "#4575b4", "mid": "#fee090", "late": "#d73027"}
    phases = ["early", "mid", "late"]

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharex=True, sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = _assign_phase_bins(pd.DataFrame(umap_df[umap_df["FeedingState"] == cond].copy()))
        if sub.empty:
            ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n=0)")
            continue

        x = sub["UMAP1"].to_numpy(dtype=float)
        y = sub["UMAP2"].to_numpy(dtype=float)
        ax.scatter(x, y, c="gray", s=4, alpha=0.10, zorder=1)

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        if x_max > x_min and y_max > y_min:
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            for phase in phases:
                pts = sub[sub["phase_bin"] == phase][["UMAP1", "UMAP2"]].dropna().to_numpy(dtype=float)
                if len(pts) < 20:
                    continue
                try:
                    kde = gaussian_kde(pts.T, bw_method=bw_method)
                    zz = kde(grid).reshape(xx.shape)
                except Exception:
                    continue
                levels = np.quantile(zz, [0.70, 0.82, 0.90, 0.96])
                levels = np.unique(levels)
                if len(levels) == 0:
                    continue
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=levels,
                    colors=[phase_colors[phase]],
                    linewidths=1.5,
                    alpha=0.85,
                    zorder=3,
                )

        handles = [Line2D([0], [0], color=phase_colors[p], lw=2, label=p.capitalize()) for p in phases]
        ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
        ax.set_title(f"{FEEDING_LABELS.get(cond, cond)} (n={len(sub)})")
        ax.set_xlabel("UMAP1")
        ax.grid(alpha=0.2, linewidth=0.4)

    axes[0].set_ylabel("UMAP2")
    fig.suptitle("FeedingState UMAP with early/mid/late KDE contour overlay", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = output_path.with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def run_umap_and_save_outputs(
    event_df: pd.DataFrame,
    output_dir: Path,
    feature_keys: Sequence[str],
    label: str,
    clip_quantile: float,
    use_pca: bool,
    pca_components: int,
    within_fly_normalize: bool,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    random_state: int,
    metric: str,
    significant_threshold_px: float,
    major_threshold_px: float,
    draw_trajectories: bool,
    draw_average_arrow: bool,
    kde_bw: float,
) -> None:
    print("-" * 74)
    print(f"Running UMAP block: {label}")
    print(f"  feature keys: {list(feature_keys)}")

    prep_df, x_final, used_cols, prep_stats = prepare_feature_matrix(
        event_df=event_df,
        feature_keys=feature_keys,
        clip_quantile=clip_quantile,
        use_pca=use_pca,
        pca_components=pca_components,
        within_fly_normalize=within_fly_normalize,
    )

    print(
        "Preprocessing summary: "
        f"rows {int(prep_stats['n_rows_for_umap'])}/{int(prep_stats['n_input_rows'])}, "
        f"features {int(prep_stats['n_features_used'])}/{int(prep_stats['n_features_requested'])}"
    )

    emb = fit_umap_embeddings(
        x=x_final,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        metric=metric,
    )

    prep_df = prep_df.copy()
    prep_df["UMAP1"] = emb[:, 0]
    prep_df["UMAP2"] = emb[:, 1]

    prefix = f"feedingstate_event_umap_{label}"
    csv_emb = output_dir / f"{prefix}_embedding.csv"
    manifest_path = output_dir / f"{prefix}_feature_manifest.txt"
    prep_df.to_csv(csv_emb, index=False)
    save_feature_manifest(manifest_path, selected_keys=feature_keys, used_cols=used_cols)
    print(f"  Saved: {csv_emb}")
    print(f"  Saved: {manifest_path}")

    plot_umap_by_condition(
        umap_df=prep_df,
        color_col="onset_time_min",
        color_label="Event onset time (min)",
        cmap="viridis",
        title=f"FeedingState event UMAP ({label}) colored by event onset time",
        output_path=output_dir / f"{prefix}_by_time",
        significant_threshold_px=significant_threshold_px,
        major_threshold_px=major_threshold_px,
        draw_trajectories=draw_trajectories,
        draw_average_arrow=draw_average_arrow,
    )
    plot_umap_by_condition(
        umap_df=prep_df,
        color_col="event_idx_rel_to_first_sig",
        color_label="Event index relative to first significant push",
        cmap="coolwarm",
        title=f"FeedingState event UMAP ({label}) colored by event index relative to first significant push",
        output_path=output_dir / f"{prefix}_by_event_index_rel_to_first_significant",
        significant_threshold_px=significant_threshold_px,
        major_threshold_px=major_threshold_px,
        draw_trajectories=draw_trajectories,
        draw_average_arrow=draw_average_arrow,
    )
    plot_umap_by_condition(
        umap_df=prep_df,
        color_col="ball_max_displacement_px",
        color_label="Ball max displacement during event (px) [diagnostic only]",
        cmap="cividis",
        title=f"FeedingState event UMAP ({label}) colored by ball displacement (diagnostic)",
        output_path=output_dir / f"{prefix}_by_ball_displacement_diagnostic",
        significant_threshold_px=significant_threshold_px,
        major_threshold_px=major_threshold_px,
        draw_trajectories=draw_trajectories,
        draw_average_arrow=draw_average_arrow,
    )
    plot_umap_phase_density(
        umap_df=prep_df,
        output_path=output_dir / f"{prefix}_density_early_mid_late",
    )
    plot_kde_phase_overlay(
        umap_df=prep_df,
        output_path=output_dir / f"{prefix}_kde_phase_overlay",
        bw_method=kde_bw,
    )


def save_feature_manifest(path: Path, selected_keys: Sequence[str], used_cols: Sequence[str]) -> None:
    with open(path, "w") as f:
        f.write("FEEDINGSTATE EVENT UMAP FEATURE MANIFEST\n")
        f.write("=" * 60 + "\n\n")

        f.write("Feature provenance:\n")
        f.write("  All listed features are derived from raw frame-level coordinates/event IDs in this script.\n")
        f.write("  None are required as precomputed columns in input coordinate files.\n\n")

        f.write("Selected feature keys:\n")
        for key in selected_keys:
            spec = FEATURE_SPECS[key]
            f.write(f"  - {spec.key}: {spec.label} [column={spec.column}]\n")

        f.write("\nColumns used after preprocessing:\n")
        for col in used_cols:
            f.write(f"  - {col}\n")

        f.write("\nAll available features:\n")
        for key in sorted(FEATURE_SPECS):
            spec = FEATURE_SPECS[key]
            f.write(f"  - {spec.key}: {spec.label} [column={spec.column}]\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FeedingState event-level UMAP around interaction dynamics",
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
        default=("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/" "feedingstate_event_umap"),
        help="Directory for outputs",
    )
    parser.add_argument("--fps", type=float, default=29.0, help="Acquisition frame rate")
    parser.add_argument(
        "--backward-step-px",
        type=float,
        default=0.2,
        help="Minimum positive distance-to-ball step (px/frame) counted as backward movement",
    )
    parser.add_argument(
        "--clear-forward-step-px",
        type=float,
        default=0.2,
        help="Minimum negative distance-to-ball step (px/frame) counted as clear forward movement",
    )
    parser.add_argument(
        "--clear-forward-min-steps",
        type=int,
        default=3,
        help="Consecutive clear forward steps required to stop backoff phase",
    )
    parser.add_argument(
        "--focus-min",
        type=float,
        default=0.0,
        help="Earliest time (min) for first significant/major push tagging",
    )
    parser.add_argument(
        "--significant-threshold-px",
        type=float,
        default=5.0,
        help="Significant push threshold (px)",
    )
    parser.add_argument(
        "--major-threshold-px",
        type=float,
        default=20.0,
        help="Major push threshold (px)",
    )
    parser.add_argument(
        "--umap-mode",
        type=str,
        choices=["single", "dual"],
        default="dual",
        help="Run one UMAP with --features (single) or separate state/progression UMAPs (dual)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature keys used for UMAP (single mode only; defaults to DEFAULT_FEATURE_KEYS)",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.01,
        help="Winsorization quantile per feature (0 disables clipping)",
    )
    parser.add_argument("--n-neighbors", type=int, default=75, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.4, help="UMAP min_dist")
    parser.add_argument("--spread", type=float, default=2.0, help="UMAP spread")
    parser.add_argument("--kde-bw", type=float, default=0.3, help="Bandwidth for KDE phase-overlay plot")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP distance metric")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP")
    parser.add_argument("--use-pca", action="store_true", help="Apply PCA before UMAP")
    parser.add_argument("--pca-components", type=int, default=6, help="PCA components when --use-pca")
    parser.add_argument("--list-features", action="store_true", help="List available feature keys and exit")
    parser.add_argument("--test", action="store_true", help="Load only first 2 coordinate files")
    parser.add_argument(
        "--no-within-fly-normalization",
        action="store_true",
        help="Disable within-fly robust normalization before global scaling",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable per-fly trajectory lines on UMAP scatter plots",
    )
    parser.add_argument(
        "--no-average-arrow",
        action="store_true",
        help="Disable average per-condition progression arrow on UMAP scatter plots",
    )
    args = parser.parse_args()

    if args.major_threshold_px <= args.significant_threshold_px:
        raise ValueError(
            "Expected --major-threshold-px to be greater than --significant-threshold-px. "
            f"Got major={args.major_threshold_px}, significant={args.significant_threshold_px}."
        )

    if args.list_features:
        print("Available feature keys:")
        for key in sorted(FEATURE_SPECS):
            spec = FEATURE_SPECS[key]
            print(f"  - {spec.key:28s} {spec.label}")
        return

    selected_feature_keys = parse_feature_keys(args.features) if args.features else list(DEFAULT_FEATURE_KEYS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("FEEDINGSTATE — EVENT-LEVEL UMAP")
    print("=" * 74)
    print(f"Coordinates dir : {args.coordinates_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"UMAP mode       : {args.umap_mode}")
    if args.umap_mode == "single":
        print(f"Features        : {selected_feature_keys}")
    else:
        print(f"State features  : {STATE_FEATURE_KEYS}")
        print(f"Prog features   : {PROGRESSION_FEATURE_KEYS}")
    print(f"Significant px  : > {args.significant_threshold_px}")
    print(f"Major px        : > {args.major_threshold_px}")
    print(f"Backward step   : {args.backward_step_px} px/frame")
    print(f"Clear forward   : {args.clear_forward_step_px} px/frame × {args.clear_forward_min_steps} steps")
    print(f"Focus min       : {args.focus_min}")
    print(f"UMAP neighbors  : {args.n_neighbors}")
    print(f"UMAP min_dist   : {args.min_dist}")
    print(f"UMAP spread     : {args.spread}")
    print(f"KDE bw          : {args.kde_bw}")
    print(f"UMAP metric     : {args.metric}")
    print(f"Use PCA         : {args.use_pca}")
    print(f"Within-fly norm : {not args.no_within_fly_normalization}")
    print(f"Trajectories    : {not args.no_trajectories}")
    print(f"Avg arrow       : {not args.no_average_arrow}")

    t0 = time.time()
    df = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
    print(f"Data loaded in {time.time() - t0:.1f}s")

    event_df = compute_event_feature_table(
        df=df,
        fps=args.fps,
        significant_threshold_px=args.significant_threshold_px,
        major_threshold_px=args.major_threshold_px,
        focus_min=args.focus_min,
        backward_step_px=args.backward_step_px,
        clear_forward_step_px=args.clear_forward_step_px,
        clear_forward_min_steps=args.clear_forward_min_steps,
    )
    if event_df.empty:
        raise RuntimeError("No interaction events available after filtering.")

    print(f"Total events with features: {len(event_df)}")
    for cond in FEEDING_ORDER:
        n_cond = len(event_df[event_df["FeedingState"] == cond])
        if n_cond > 0:
            print(f"  {cond}: {n_cond} events")

    csv_features = output_dir / "feedingstate_event_umap_event_features.csv"
    event_df.to_csv(csv_features, index=False)
    print(f"  Saved: {csv_features}")

    if args.umap_mode == "single":
        run_umap_and_save_outputs(
            event_df=event_df,
            output_dir=output_dir,
            feature_keys=selected_feature_keys,
            label="single",
            clip_quantile=max(0.0, float(args.clip_quantile)),
            use_pca=bool(args.use_pca),
            pca_components=int(args.pca_components),
            within_fly_normalize=not args.no_within_fly_normalization,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            random_state=args.random_state,
            metric=args.metric,
            significant_threshold_px=args.significant_threshold_px,
            major_threshold_px=args.major_threshold_px,
            draw_trajectories=not args.no_trajectories,
            draw_average_arrow=not args.no_average_arrow,
            kde_bw=float(args.kde_bw),
        )
    else:
        run_umap_and_save_outputs(
            event_df=event_df,
            output_dir=output_dir,
            feature_keys=STATE_FEATURE_KEYS,
            label="state",
            clip_quantile=max(0.0, float(args.clip_quantile)),
            use_pca=bool(args.use_pca),
            pca_components=int(args.pca_components),
            within_fly_normalize=not args.no_within_fly_normalization,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            random_state=args.random_state,
            metric=args.metric,
            significant_threshold_px=args.significant_threshold_px,
            major_threshold_px=args.major_threshold_px,
            draw_trajectories=not args.no_trajectories,
            draw_average_arrow=not args.no_average_arrow,
            kde_bw=float(args.kde_bw),
        )
        run_umap_and_save_outputs(
            event_df=event_df,
            output_dir=output_dir,
            feature_keys=PROGRESSION_FEATURE_KEYS,
            label="progression",
            clip_quantile=max(0.0, float(args.clip_quantile)),
            use_pca=bool(args.use_pca),
            pca_components=int(args.pca_components),
            within_fly_normalize=not args.no_within_fly_normalization,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            random_state=args.random_state,
            metric=args.metric,
            significant_threshold_px=args.significant_threshold_px,
            major_threshold_px=args.major_threshold_px,
            draw_trajectories=not args.no_trajectories,
            draw_average_arrow=not args.no_average_arrow,
            kde_bw=float(args.kde_bw),
        )

    print("=" * 74)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 74)


if __name__ == "__main__":
    main()
