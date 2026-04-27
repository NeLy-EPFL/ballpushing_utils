#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 10d: Behavioral metrics showing
no significant differences across nutritional states.

Panels (side by side, 4 subplots):
  - Number of interaction events
  - Pulling ratio
  - Flailing
  - Time to first major push (min)

Three groups: fed (green), starved (orange), starved_noWater/blue (Starved+water-deprived).
All pairwise permutation tests (10,000 permutations), FDR-corrected (α = 0.05, BH).
All brackets shown (including ns).

Expected results (from published figure):
  - All FDR-corrected p ≥ 0.16 (no significant differences)

Sample sizes per group (fed / starved / starved+water-deprived):
  - nb_events, pulling_ratio, flailing: 102 / 101 / 101
  - first_major_event_time: 60 / 69 / 84

Usage:
    python edfigure10_d_metrics_nonsignificant.py [--test] [--n-permutations N]
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from ballpushing_utils import figure_output_dir

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDING_ORDER = ["fed", "starved", "starved_noWater"]
FEEDING_LABELS = {
    "fed": "Fed",
    "starved": "Starved",
    "starved_noWater": "Starved\n(no water)",
}
FEEDING_COLORS = {
    "fed": "#66C2A5",  # green
    "starved": "#FC8D62",  # orange
    "starved_noWater": "#8DA0CB",  # blue
}

# Metrics for this panel — non-significant differences
PANEL_METRICS = [
    {
        "col": "nb_events",
        "label": "Number of events",
        "convert": None,
        "panel": "d1",
    },
    {
        "col": "pulling_ratio",
        "label": "Pulling ratio",
        "convert": None,
        "panel": "d2",
    },
    {
        "col": "flailing",
        "label": "Flailing",
        "convert": None,
        "panel": "d3",
    },
    {
        "col": "first_major_event_time",
        "label": "Time to first major push (min)",
        "convert": lambda x: x / 60.0,  # seconds → minutes
        "panel": "d4",
    },
]

SHORT_EXPERIMENTS = [
    "240718_Afternoon_FeedingState_Videos_Tracked",
    "240718_Afternoon_FeedingState_next_Videos_Tracked",
]

SUMMARY_PATH = (
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets"
    "/260220_10_summary_control_folders_Data/summary/pooled_summary.feather"
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cohens_d(group1, group2):
    """Cohen's d effect size (group2 − group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrapped CI for the mean difference (group2 − group1)."""
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)
    alpha = 100 - ci
    return float(np.percentile(bootstrap_diffs, alpha / 2)), float(np.percentile(bootstrap_diffs, 100 - alpha / 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load summary feather, filter Light=on, normalise FeedingState."""
    print(f"Loading dataset from: {SUMMARY_PATH}")
    try:
        dataset = pd.read_feather(SUMMARY_PATH)
        print(f"Loaded: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {SUMMARY_PATH}")

    # Filter Light=on
    if "Light" not in dataset.columns:
        raise ValueError("'Light' column not found in dataset.")
    dataset = dataset[dataset["Light"] == "on"].copy()
    print(f"After Light=on filter: {dataset.shape}")

    # Exclude short experiments
    if "experiment" in dataset.columns:
        before = len(dataset)
        dataset = dataset[~dataset["experiment"].isin(SHORT_EXPERIMENTS)].copy()
        if before - len(dataset):
            print(f"Excluded {before - len(dataset)} rows from short experiments.")

    # Normalise FeedingState
    if "FeedingState" not in dataset.columns:
        raise ValueError("'FeedingState' column not found in dataset.")
    dataset["FeedingState"] = dataset["FeedingState"].str.strip().str.lower()
    rename_map = {"fed": "fed", "starved": "starved", "starved_nowater": "starved_noWater"}
    dataset["FeedingState"] = dataset["FeedingState"].map(rename_map).fillna(dataset["FeedingState"])
    dataset = dataset[dataset["FeedingState"].isin(FEEDING_ORDER)].copy()
    print(f"FeedingState distribution:\n{dataset['FeedingState'].value_counts()}")

    # Convert bool columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns
    for col in ["insight_effect", "insight_effect_log", "exit_time", "index"]:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])

    # Fill NaN defaults for newer metrics
    new_metric_defaults = {"flailing": 0.0}
    fill_dict = {m: v for m, v in new_metric_defaults.items() if m in dataset.columns and dataset[m].isnull().any()}
    if fill_dict:
        dataset.fillna(fill_dict, inplace=True)

    if test_mode and len(dataset) > test_sample_size:
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"TEST MODE: sampled {test_sample_size} rows.")

    return dataset


# ---------------------------------------------------------------------------
# Permutation tests for one metric
# ---------------------------------------------------------------------------


def run_permutation_tests(data, metric, n_permutations=10000, alpha=0.05):
    """
    All pairwise permutation tests (median difference) with FDR correction.
    Returns list of result dicts.
    """
    present = [c for c in FEEDING_ORDER if c in data["FeedingState"].unique()]
    all_pairs = list(combinations(present, 2))
    pvals = []
    pair_results = []

    for cond_a, cond_b in all_pairs:
        vals_a = data[data["FeedingState"] == cond_a][metric].dropna().values
        vals_b = data[data["FeedingState"] == cond_b][metric].dropna().values
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue

        obs_stat = float(np.median(vals_b) - np.median(vals_a))
        combined = np.concatenate([vals_a, vals_b])
        n_a = len(vals_a)

        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[i] = np.median(combined[n_a:]) - np.median(combined[:n_a])

        pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))
        d = cohens_d(vals_a, vals_b)
        mean_diff = float(np.mean(vals_b) - np.mean(vals_a))
        ci_lower, ci_upper = bootstrap_ci_difference(vals_a, vals_b, n_bootstrap=10000, ci=95, random_state=42)

        pvals.append(pval)
        pair_results.append(
            {
                "ConditionA": cond_a,
                "ConditionB": cond_b,
                "pval_raw": pval,
                "cohens_d": d,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "median_a": float(np.median(vals_a)),
                "median_b": float(np.median(vals_b)),
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_a": len(vals_a),
                "n_b": len(vals_b),
            }
        )

    if not pvals:
        return []

    if len(pvals) > 1:
        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    else:
        pvals_corrected = np.array(pvals)
        rejected = [p < alpha for p in pvals]

    for i, r in enumerate(pair_results):
        r["pval_corrected"] = float(pvals_corrected[i])
        r["significant"] = bool(rejected[i])
        p = pvals_corrected[i]
        if p < 0.001:
            r["sig_level"] = "***"
        elif p < 0.01:
            r["sig_level"] = "**"
        elif p < 0.05:
            r["sig_level"] = "*"
        else:
            r["sig_level"] = "ns"

    return pair_results


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def plot_metric_panel(ax, data, metric, y_label, pair_results):
    """
    Draw a single boxplot + jitter panel for one metric.
    All pairwise brackets shown (including ns).
    """
    present = [c for c in FEEDING_ORDER if c in data["FeedingState"].unique()]
    x_positions = list(range(len(present)))

    font_size_ticks = 9
    font_size_labels = 11
    font_size_ann = 13

    group_vals = {}
    for i, cond in enumerate(present):
        vals = data[data["FeedingState"] == cond][metric].dropna().values
        group_vals[cond] = vals
        color = FEEDING_COLORS.get(cond, "gray")

        ax.boxplot(
            [vals],
            positions=[i],
            widths=0.55,
            patch_artist=False,
            showfliers=False,
            vert=True,
            boxprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
            medianprops=dict(color="black", linewidth=2.0),
        )
        jitter = np.random.normal(i, 0.07, size=len(vals))
        ax.scatter(jitter, vals, s=20, color=color, alpha=0.5, edgecolors="none", zorder=3)

    all_vals_flat = np.concatenate([v for v in group_vals.values() if len(v) > 0])
    y_max = float(np.percentile(all_vals_flat, 99))
    y_min = float(np.min(all_vals_flat))
    y_range = y_max - y_min if y_max > y_min else 1.0

    bracket_base = y_max + 0.08 * y_range
    bracket_step = 0.12 * y_range

    pair_result_map = {(r["ConditionA"], r["ConditionB"]): r for r in pair_results}
    all_pairs = list(combinations(present, 2))
    for bracket_num, (cond_a, cond_b) in enumerate(all_pairs):
        r = pair_result_map.get((cond_a, cond_b))
        if r is None:
            continue
        idx_a = present.index(cond_a)
        idx_b = present.index(cond_b)
        x_left = min(idx_a, idx_b)
        x_right = max(idx_a, idx_b)
        h = bracket_base + bracket_num * bracket_step
        ax.plot([x_left, x_left], [h - 0.01 * y_range, h], "k-", linewidth=1.2)
        ax.plot([x_left, x_right], [h, h], "k-", linewidth=1.2)
        ax.plot([x_right, x_right], [h - 0.01 * y_range, h], "k-", linewidth=1.2)
        sig = r["sig_level"]
        color = "red" if sig != "ns" else "gray"
        ax.text(
            (x_left + x_right) / 2,
            h + 0.01 * y_range,
            sig,
            ha="center",
            va="bottom",
            fontsize=font_size_ann,
            color=color,
            fontweight="bold" if sig != "ns" else "normal",
        )

    n_brackets = len(all_pairs)
    y_top = bracket_base + n_brackets * bracket_step + 0.05 * y_range
    ax.set_ylim(y_min - 0.05 * y_range, y_top)

    counts = [len(group_vals[c]) for c in present]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{FEEDING_LABELS.get(c, c)}\n(n={cnt})" for c, cnt in zip(present, counts)],
        fontsize=font_size_ticks,
    )
    ax.set_ylabel(y_label, fontsize=font_size_labels)
    ax.tick_params(axis="y", labelsize=font_size_ticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_panels(data, output_dir, n_permutations=10000):
    """Generate panel (d): 4 non-significant metrics side by side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = [m for m in PANEL_METRICS if m["col"] in data.columns]
    missing = [m["col"] for m in PANEL_METRICS if m["col"] not in data.columns]
    if missing:
        print(f"Warning: columns not found in dataset: {missing}")
    if not available:
        print("No panel metrics found in dataset — check column names.")
        return pd.DataFrame()

    all_stats = []

    # Run permutation tests
    metric_results = {}
    for m in available:
        col = m["col"]
        plot_df = data.dropna(subset=[col, "FeedingState"]).copy()
        if m["convert"] is not None:
            plot_df[col] = m["convert"](plot_df[col])
        print(f"\nPermutation tests for '{col}':")
        results = run_permutation_tests(plot_df, col, n_permutations=n_permutations)
        metric_results[col] = (plot_df, results)
        for r in results:
            r["Metric"] = col
        all_stats.extend(results)
        for r in results:
            print(f"  {r['ConditionA']} vs {r['ConditionB']}: p_fdr={r['pval_corrected']:.4f} ({r['sig_level']})")

    # Combined figure
    n_panels = len(available)
    panel_width_mm = 90
    panel_height_mm = 120
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(n_panels * panel_width_mm / 25.4, panel_height_mm / 25.4),
    )
    if n_panels == 1:
        axes = [axes]

    for ax, m in zip(axes, available):
        col = m["col"]
        plot_df, results = metric_results[col]
        plot_metric_panel(ax, plot_df, col, m["label"], results)

    plt.tight_layout()
    combined_pdf = output_dir / "edfigure10d_metrics_nonsignificant_combined.pdf"
    fig.savefig(combined_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(combined_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved: {combined_pdf}")
    plt.close(fig)

    # Individual panels
    for m in available:
        col = m["col"]
        plot_df, results = metric_results[col]
        fig_ind, ax_ind = plt.subplots(figsize=(panel_width_mm / 25.4, panel_height_mm / 25.4))
        plot_metric_panel(ax_ind, plot_df, col, m["label"], results)
        plt.tight_layout()
        ind_pdf = output_dir / f"edfigure10d_{col}.pdf"
        fig_ind.savefig(ind_pdf, dpi=300, bbox_inches="tight")
        fig_ind.savefig(ind_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"Saved: {ind_pdf}")
        plt.close(fig_ind)

    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_file = output_dir / "edfigure10d_statistics.csv"
        stats_df.to_csv(stats_file, index=False, float_format="%.6f")
        print(f"Saved: {stats_file}")

    return stats_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 10d: non-significant behavioral metrics by nutritional state",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: small data sample")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else figure_output_dir("EDFigure10", __file__)

    print(f"\n{'='*60}")
    print("Extended Data Figure 10d: Non-significant behavioral metrics")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Permutations:     {args.n_permutations:,}")
    if args.test:
        print("TEST MODE enabled")

    t0 = time.time()
    data = load_and_clean_dataset(test_mode=args.test)
    stats = generate_panels(data, output_dir, n_permutations=args.n_permutations)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s. Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
