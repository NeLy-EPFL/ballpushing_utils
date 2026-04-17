#!/usr/bin/env python3
from __future__ import annotations

"""
TNT nickname-vs-control significant-push rate analysis over absolute time.

Significant pushes are interaction events with max ball displacement > threshold_px.
No alignment to first push is used.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from ballpushing_utils.filtered_dataset_loader import build_dataset_for_nickname_and_control

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

DEFAULT_YAML_PATH = "/home/matthias/ballpushing_utils/experiments_yaml/TNT_screen.yaml"
DEFAULT_SPLIT_REGISTRY = "/mnt/upramdya_data/MD/Region_map_250908.csv"
DEFAULT_OUTPUT_ROOT = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Significant_push_rate"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _safe_path_part(name: str) -> str:
    cleaned = str(name).strip()
    return cleaned if cleaned else "Unknown"


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

    required = ["time", "fly", "Nickname", "interaction_event", "x_ball_0", "y_ball_0"]
    _require_columns(df, required, "coordinates dataset")

    if "Light" in df.columns:
        df = df[df["Light"] == "on"].copy()

    df["Nickname"] = df["Nickname"].astype(str)
    focal_candidates = {str(nickname).strip(), str(focal_canonical).strip()}

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

    print(f"Resolved mapping: {focal_canonical} ({info.genotype}) vs {control} ({info.control_genotype})")
    print(f"Selected fly directories: {len(manifest)}")
    for grp in [nickname, control]:
        grp_df = subset[subset["Group"] == grp]
        print(f"  {grp}: {grp_df['fly'].nunique()} flies, {len(grp_df)} frames")

    return subset, control, info


def compute_significant_push_onsets(df: pd.DataFrame, threshold_px: float) -> pd.DataFrame:
    event_rows = df[df["interaction_event"].notna()].copy()
    if event_rows.empty:
        return pd.DataFrame(columns=["fly", "Group", "interaction_event", "onset_time_min", "max_displacement_px"])

    group_keys = ["fly", "interaction_event"]
    idx_start = event_rows.groupby(group_keys)["time"].idxmin()

    start_df = event_rows.loc[idx_start, ["fly", "interaction_event", "time", "x_ball_0", "y_ball_0", "Group"]].rename(
        columns={"time": "onset_time_s", "x_ball_0": "ball_x_start", "y_ball_0": "ball_y_start"}
    )

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

    out = start_df.merge(max_disp, on=["fly", "interaction_event"], how="inner")
    out["onset_time_min"] = out["onset_time_s"] / 60.0

    sig = out[out["max_displacement_px"] > threshold_px].copy()
    return sig[["fly", "Group", "interaction_event", "onset_time_min", "max_displacement_px"]]


def compute_significant_push_rate_per_fly(
    sig_onsets_df: pd.DataFrame,
    df_full: pd.DataFrame,
    bin_size_min: float,
    max_time_min: Optional[float] = None,
) -> pd.DataFrame:
    if max_time_min is None:
        max_time_min = float(df_full["time"].max() / 60.0)

    bin_edges = np.arange(0.0, max_time_min + bin_size_min, bin_size_min)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, max_time_min + bin_size_min])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fly_meta = df_full.drop_duplicates("fly")[["fly", "Group"]].set_index("fly")
    rows = []

    for fly in df_full["fly"].unique():
        group = fly_meta.loc[fly, "Group"]
        fly_onsets = sig_onsets_df[sig_onsets_df["fly"] == fly]["onset_time_min"].values
        for idx in range(len(bin_edges) - 1):
            b_start = float(bin_edges[idx])
            b_end = float(bin_edges[idx + 1])
            count = int(((fly_onsets >= b_start) & (fly_onsets < b_end)).sum())
            rows.append(
                {
                    "fly": fly,
                    "Group": group,
                    "bin_idx": idx,
                    "bin_start_min": b_start,
                    "bin_center_min": float(bin_centers[idx]),
                    "significant_push_count": count,
                    "significant_push_rate_per_min": float(count / bin_size_min),
                }
            )

    return pd.DataFrame(rows)


def _bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def plot_significant_push_rate_over_time(
    rates_df: pd.DataFrame,
    group_order: list[str],
    group_colors: dict[str, str],
    threshold_px: float,
    bin_size_min: float,
    n_bootstrap: int,
    output_path: Path,
) -> None:
    groups = [g for g in group_order if g in rates_df["Group"].unique()]
    if not groups:
        print("No groups to plot.")
        return

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(11, 6))

    for grp in groups:
        sub = rates_df[rates_df["Group"] == grp].copy()
        bins = np.sort(sub["bin_center_min"].unique())

        means, lo, hi = [], [], []
        for b in bins:
            vals = sub.loc[sub["bin_center_min"] == b, "significant_push_rate_per_min"].values
            m, l, h = _bootstrap_ci_mean(vals, rng, n_bootstrap=n_bootstrap)
            means.append(m)
            lo.append(l)
            hi.append(h)

        color = group_colors.get(grp, "C0")
        n_flies = sub["fly"].nunique()
        ax.fill_between(bins, lo, hi, color=color, alpha=0.22)
        ax.plot(bins, means, lw=2.2, color=color, label=f"{grp} (n={n_flies})")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Significant push rate (events/min)")
    ax.set_title(
        f"Significant push rate over time (threshold > {threshold_px:.1f}px, bin={bin_size_min:.4f} min)",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        out = output_path.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def save_tables(
    output_dir: Path,
    stem_prefix: str,
    sig_onsets_df: pd.DataFrame,
    rates_df: pd.DataFrame,
) -> None:
    onsets_path = output_dir / f"{stem_prefix}_significant_push_onsets_per_event.csv"
    rates_path = output_dir / f"{stem_prefix}_significant_push_rate_per_fly_per_bin.csv"

    sig_onsets_df.to_csv(onsets_path, index=False)
    rates_df.to_csv(rates_path, index=False)
    print(f"  Saved: {onsets_path}")
    print(f"  Saved: {rates_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNT significant push-rate analysis (absolute time, no first-push alignment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nickname", type=str, required=True, help="Nickname (line of interest)")
    parser.add_argument("--yaml-path", type=str, default=DEFAULT_YAML_PATH, help="YAML file listing experiment folders")
    parser.add_argument("--split-registry", type=str, default=DEFAULT_SPLIT_REGISTRY, help="Split registry CSV")
    parser.add_argument("--control-nickname", type=str, default=None, help="Optional control override")
    parser.add_argument(
        "--force-control",
        type=str,
        default=None,
        choices=["Empty-Split", "Empty-Gal4", "TNTxPR"],
        help="Force control for split y/n lines (split m remains TNTxPR)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root for auto path")

    parser.add_argument("--threshold-px", type=float, default=5.0, help="Significant push threshold (px)")
    parser.add_argument("--bin-size", type=float, default=1.0, help="Time bin size in minutes")
    parser.add_argument("--max-time-min", type=float, default=None, help="Optional max time to plot/analyze")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations for CI")

    parser.add_argument("--test", action="store_true", help="Fast test mode")
    parser.add_argument("--test-max-flies", type=int, default=6, help="Max flies per group in test mode")
    parser.add_argument("--test-max-experiments", type=int, default=None, help="Max experiments in test mode")
    args = parser.parse_args()

    nickname = str(args.nickname)

    print("=" * 78)
    print("TNT SIGNIFICANT PUSH RATE ANALYSIS")
    print("=" * 78)
    print(f"YAML path        : {args.yaml_path}")
    print(f"Nickname         : {nickname}")
    print(f"Split registry   : {args.split_registry}")
    print(f"Threshold        : > {args.threshold_px} px")
    print(f"Bin size         : {args.bin_size} min")

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

    sig_onsets_df = compute_significant_push_onsets(df, threshold_px=args.threshold_px)
    if sig_onsets_df.empty:
        raise RuntimeError("No significant pushes found for selected threshold.")

    cov = sig_onsets_df.groupby("Group")["fly"].nunique().to_dict()
    total = df.groupby("Group")["fly"].nunique().to_dict()
    for g in group_order:
        n_sig = int(cov.get(g, 0))
        n_tot = int(total.get(g, 0))
        pct = 100 * n_sig / n_tot if n_tot else np.nan
        pct_txt = f"{pct:.1f}%" if not np.isnan(pct) else "n.d."
        print(f"  Significant push coverage — {g}: {n_sig}/{n_tot} ({pct_txt})")

    rates_df = compute_significant_push_rate_per_fly(
        sig_onsets_df=sig_onsets_df,
        df_full=df,
        bin_size_min=float(args.bin_size),
        max_time_min=args.max_time_min,
    )

    stem = sanitize_filename(f"nickname_{nickname}_vs_{control}")

    plot_significant_push_rate_over_time(
        rates_df=rates_df,
        group_order=group_order,
        group_colors=group_colors,
        threshold_px=float(args.threshold_px),
        bin_size_min=float(args.bin_size),
        n_bootstrap=int(args.n_bootstrap),
        output_path=output_dir / f"{stem}_significant_push_rate_over_time",
    )

    save_tables(
        output_dir=output_dir,
        stem_prefix=stem,
        sig_onsets_df=sig_onsets_df,
        rates_df=rates_df,
    )

    print("=" * 78)
    print("Done.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
