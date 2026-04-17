#!/usr/bin/env python3
from __future__ import annotations

"""
TNT nickname-vs-control CONTACT-rate analysis aligned to each fly's first significant push.

Difference from interaction-rate script:
- Rate is computed from contact onsets (`contact_event_onset`) instead of interaction onsets.
- First significant push is still defined from interaction-event displacement.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from ballpushing_utils.filtered_dataset_loader import build_dataset_for_nickname_and_control
from plot_tnt_nickname_interaction_rate_first_push import (
    DEFAULT_SPLIT_REGISTRY,
    DEFAULT_YAML_PATH,
    _require_columns,
    _safe_path_part,
    compute_ball_displacement_per_event,
    compute_event_index_aligned_rates,
    compute_first_significant_push,
    compute_post_push_rate_change,
    compute_rates_per_fly,
    compute_significant_coverage,
    plot_post_push_rate_change,
    plot_rate_aligned_to_first_push,
    plot_rate_aligned_to_first_push_event_index,
    sanitize_filename,
    save_tables,
    test_post_push_change,
)

DEFAULT_OUTPUT_ROOT = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Contact_rates"


def load_nickname_and_control_data_contacts(
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

    print(f"Loading filtered coordinates with contact annotations from YAML: {yaml_path}")
    max_experiments = test_max_experiments if test_mode else None
    max_flies = test_max_flies_per_group if test_mode else None

    df, manifest, info = build_dataset_for_nickname_and_control(
        yaml_path=yaml_path,
        dataset_type="coordinates",
        nickname=nickname,
        split_registry_path=split_registry_path,
        control_nickname=control_nickname,
        force_control=force_control,
        annotate_contacts=True,
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
        "x_ball_0",
        "y_ball_0",
        "contact_event_onset",
    ]
    _require_columns(df, required, "coordinates dataset with contact annotations")

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


def extract_contact_onsets(df: pd.DataFrame) -> pd.DataFrame:
    non_nan = df[df["contact_event_onset"].notna()].copy()
    onsets = (
        non_nan.groupby(["fly", "contact_event_onset"], as_index=False)
        .agg(time=("time", "min"), Group=("Group", "first"))
        .reset_index(drop=True)
    )
    onsets["time_min"] = onsets["time"] / 60.0
    print(f"Unique contact onsets: {len(onsets)}")
    return onsets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNT nickname-vs-control contact-rate analysis aligned to first significant push",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nickname", type=str, required=True, help="Nickname (line of interest)")
    parser.add_argument("--yaml-path", type=str, default=DEFAULT_YAML_PATH, help="YAML file listing experiments")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output dir for auto path: {output_root}/{Brain_region}/{Simplified_Nickname}",
    )
    parser.add_argument("--split-registry", type=str, default=DEFAULT_SPLIT_REGISTRY)
    parser.add_argument("--control-nickname", type=str, default=None)
    parser.add_argument("--force-control", type=str, default=None, choices=["Empty-Split", "Empty-Gal4", "TNTxPR"])

    parser.add_argument("--bin-size", type=float, default=1.0)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--focus-min", type=float, default=0.0)
    parser.add_argument("--window-min", type=float, default=60.0)
    parser.add_argument("--event-window", type=int, default=20)
    parser.add_argument("--pre-window-far", type=float, default=10.0)
    parser.add_argument("--pre-window-near", type=float, default=1.0)
    parser.add_argument("--post-window-near", type=float, default=1.0)
    parser.add_argument("--post-window-far", type=float, default=10.0)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-max-flies", type=int, default=6)
    parser.add_argument("--test-max-experiments", type=int, default=None)
    args = parser.parse_args()

    if args.test:
        args.n_permutations = min(args.n_permutations, 1000)
        args.n_bootstrap = min(args.n_bootstrap, 300)

    nickname = str(args.nickname)
    print("=" * 78)
    print("TNT NICKNAME — FIRST SIGNIFICANT PUSH CONTACT-RATE ANALYSIS")
    print("=" * 78)
    print(f"YAML path        : {args.yaml_path}")
    print(f"Nickname         : {nickname}")
    print(f"Split registry   : {args.split_registry}")
    print(f"Output dir/root  : {args.output_dir or args.output_root}")
    print(f"Bin size         : {args.bin_size} min")
    print(f"Threshold        : > {args.threshold_px} px")
    print(f"Focus min        : {args.focus_min} min")
    print(f"Window           : ± {args.window_min} min")
    print(f"Event window     : ± {args.event_window} events")

    t0 = time.time()
    df, control, info = load_nickname_and_control_data_contacts(
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

    contact_onsets = extract_contact_onsets(df)
    rate_df = compute_rates_per_fly(contact_onsets, df_full=df, bin_size_min=args.bin_size)

    event_disp_df = compute_ball_displacement_per_event(df)
    first_push_df = compute_first_significant_push(
        event_disp_df,
        threshold_px=args.threshold_px,
        focus_min=args.focus_min,
    )
    if first_push_df.empty:
        raise RuntimeError("No flies found with a qualifying first significant push. Try lowering --threshold-px.")

    sig_coverage = compute_significant_coverage(df_full=df, first_push_df=first_push_df, group_order=group_order)
    stem = sanitize_filename(f"nickname_{nickname}_vs_{control}_contacts")

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

    event_rate_aligned_df = compute_event_index_aligned_rates(onsets_df=contact_onsets, first_push_df=first_push_df)
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
        onsets_df=contact_onsets,
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
