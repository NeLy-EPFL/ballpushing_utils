#!/usr/bin/env python3
"""
Compare permutation test results (mean vs median statistic) for all MagnetBlock metrics.

Runs two permutation tests per metric:
  - median difference (as in the legacy run_permutations_magnetblock.py)
  - mean difference

Outputs a CSV with both p-values and significance symbols side by side.

Usage:
    python compare_mean_vs_median_permutation_magnetblock.py [--test]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

DATASET_PATH = (
    "/mnt/upramdya_data/MD/MagnetBlock/Datasets/"
    "251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"
)
OUTPUT_PATH = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/permutation") / "mean_vs_median_comparison.csv"
N_PERMUTATIONS = 10000
CONTROL_GROUP = "n"
ALPHA = 0.05

PIXELS_PER_MM = 500 / 30


def is_distance_metric(name):
    keywords = [
        "distance",
        "dist",
        "head_ball",
        "fly_distance_moved",
        "max_distance",
        "distance_moved",
        "distance_ratio",
    ]
    return any(k in name.lower() for k in keywords)


def is_time_metric(name):
    keywords = ["time", "duration", "pause", "stop", "freeze", "chamber_exit_time", "time_chamber_beginning"]
    if "ratio" in name.lower():
        return False
    return any(k in name.lower() for k in keywords)


def convert(data, name):
    if is_distance_metric(name):
        return data / PIXELS_PER_MM
    if is_time_metric(name):
        return data / 60.0
    return data


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def permutation_test(g1, g2, statistic="median", n_perm=N_PERMUTATIONS, seed=42):
    """Two-sided permutation test on mean or median difference."""
    fn = np.median if statistic == "median" else np.mean
    n1 = len(g1)
    obs = fn(g2) - fn(g1)
    combined = np.concatenate([g1, g2])
    np.random.seed(seed)
    perm_diffs = np.array([fn((p := np.random.permutation(combined))[n1:]) - fn(p[:n1]) for _ in range(n_perm)])
    return float(obs), float(np.mean(np.abs(perm_diffs) >= np.abs(obs)))


def main(test_mode=False):
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_feather(DATASET_PATH)

    exclude = {"Magnet", "fly", "Corridor"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if test_mode:
        numeric_cols = numeric_cols[:5]
        print(f"Test mode: {len(numeric_cols)} metrics")

    groups = sorted(df["Magnet"].dropna().unique())
    if CONTROL_GROUP not in groups:
        raise ValueError(f"Control group '{CONTROL_GROUP}' not found. Groups: {groups}")
    test_group = [g for g in groups if g != CONTROL_GROUP][0]

    print(f"Groups: control='{CONTROL_GROUP}', test='{test_group}'")
    print(
        f"Running {N_PERMUTATIONS:,} permutations per metric × 2 statistics = {len(numeric_cols) * 2 * N_PERMUTATIONS:,} total permutations\n"
    )

    rows = []
    for i, metric in enumerate(numeric_cols):
        sub = df[["Magnet", metric]].dropna()
        g_ctrl_raw = sub[sub["Magnet"] == CONTROL_GROUP][metric].values
        g_test_raw = sub[sub["Magnet"] == test_group][metric].values

        if len(g_ctrl_raw) < 3 or len(g_test_raw) < 3:
            print(f"  [{i+1}/{len(numeric_cols)}] {metric}: skipped (insufficient data)")
            continue

        g_ctrl = convert(g_ctrl_raw, metric)
        g_test = convert(g_test_raw, metric)

        obs_med, p_med = permutation_test(g_ctrl, g_test, statistic="median")
        obs_mean, p_mean = permutation_test(g_ctrl, g_test, statistic="mean")

        agree = sig_label(p_med) == sig_label(p_mean)

        rows.append(
            {
                "metric": metric,
                "control_n": len(g_ctrl),
                "test_n": len(g_test),
                "control_median": np.median(g_ctrl),
                "test_median": np.median(g_test),
                "obs_diff_median": obs_med,
                "p_median": p_med,
                "sig_median": sig_label(p_med),
                "control_mean": np.mean(g_ctrl),
                "test_mean": np.mean(g_test),
                "obs_diff_mean": obs_mean,
                "p_mean": p_mean,
                "sig_mean": sig_label(p_mean),
                "conclusions_agree": agree,
            }
        )

        status = "" if agree else "  <-- DISAGREE"
        print(
            f"  [{i+1}/{len(numeric_cols)}] {metric}: median p={p_med:.4f} ({sig_label(p_med)})  |  mean p={p_mean:.4f} ({sig_label(p_mean)}){status}"
        )

    result_df = pd.DataFrame(rows)

    n_disagree = (~result_df["conclusions_agree"]).sum()
    print(f"\n{len(result_df)} metrics processed, {n_disagree} have disagreeing significance conclusions.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean vs median permutation comparison — MagnetBlock")
    parser.add_argument("--test", action="store_true", help="Run on first 5 metrics only")
    args = parser.parse_args()
    main(test_mode=args.test)
