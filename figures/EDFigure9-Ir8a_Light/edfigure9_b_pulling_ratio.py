#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 9b: Pulling ratio for IR8a flies
with and without illumination.

This script generates a boxplot comparison showing:
- pulling_ratio for IR8a: Light OFF (black) vs Light ON (orange)
- Permutation test (10,000 permutations, no FDR needed — single comparison)
- Statistical significance annotation

Expected results (from published figure):
  - Animals in illuminated arena had significantly higher pulling ratio (p = 0.018)

Usage:
    python edfigure9_b_pulling_ratio.py [--test] [--n-permutations N]
"""

import argparse
from pathlib import Path


from ballpushing_utils import figure_output_dir, dataset
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMARY_PATH = (
    dataset("TNT_Olfaction_Dark/Datasets/251106_08_summary_TNT_Olfaction_Dark_Data/summary/pooled_summary.feather")
)

# Light OFF = black/gray, Light ON = orange  (matching the legend)
LIGHT_COLORS = {"off": "black", "on": "orange"}
LIGHT_LABELS = {"off": "Light OFF", "on": "Light ON"}

METRIC = "pulling_ratio"
METRIC_LABEL = "Pulling ratio"


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cohens_d(group1, group2):
    """Cohen's d effect size (group2 − group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def permutation_test(group1, group2, n_permutations=10000, random_state=42):
    """
    Two-tailed permutation test on the median difference.

    Parameters
    ----------
    group1 : array-like
        Control group values.
    group2 : array-like
        Test group values.

    Returns
    -------
    float
        Two-tailed p-value.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    if len(group1) == 0 or len(group2) == 0:
        return 1.0

    obs_stat = float(np.median(group2) - np.median(group1))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    rng = np.random.default_rng(random_state)
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_diffs[i] = np.median(shuffled[n1:]) - np.median(shuffled[:n1])

    return float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrapped CI for the mean difference (group2 − group1)."""
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs[i] = np.mean(s2) - np.mean(s1)
    alpha = 100 - ci
    return float(np.percentile(diffs, alpha / 2)), float(np.percentile(diffs, 100 - alpha / 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_clean_dataset(test_mode=False):
    """Load summary dataset, filter to IR8a only."""
    print(f"Loading dataset from: {SUMMARY_PATH}")
    try:
        dataset = pd.read_feather(SUMMARY_PATH)
        print(f"Loaded: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {SUMMARY_PATH}")

    # Convert bool columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns
    for col in ["insight_effect", "insight_effect_log", "exit_time", "index"]:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])

    # Filter to IR8a only
    if "Genotype" in dataset.columns:
        dataset = dataset[dataset["Genotype"] == "TNTxIR8a"].copy()
        print(f"After IR8a filter: {dataset.shape}")

    # Filter to Light on/off
    dataset = dataset[dataset["Light"].isin(["on", "off"])].copy()
    print(f"After Light filter: {dataset.shape}")
    for light in sorted(dataset["Light"].unique()):
        print(f"  Light {light}: {len(dataset[dataset['Light'] == light])} flies")

    if test_mode and len(dataset) > 200:
        dataset = dataset.sample(n=200, random_state=42).reset_index(drop=True)
        print(f"TEST MODE: sampled 200 rows")

    return dataset


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_boxplot(
    data,
    control_condition="off",
    test_condition="on",
    group_col="Light",
    output_path=None,
    n_permutations=10000,
):
    """
    Create EDFigure 9b: boxplot comparing pulling_ratio for Light ON vs OFF.

    The plot follows the same style as other EDFigure panels:
    - Black boxplot outlines
    - Scatter points colored by condition
    - Permutation test annotation
    """
    plot_data = data[data[group_col].isin([control_condition, test_condition])].copy()

    ctrl_vals = plot_data[plot_data[group_col] == control_condition][METRIC].dropna().values
    test_vals = plot_data[plot_data[group_col] == test_condition][METRIC].dropna().values

    if len(ctrl_vals) == 0 or len(test_vals) == 0:
        print("Warning: one of the groups has no data — skipping plot")
        return None

    # ---- Statistical test ----
    p_value = permutation_test(ctrl_vals, test_vals, n_permutations=n_permutations)
    d = cohens_d(ctrl_vals, test_vals)
    ci_lower, ci_upper = bootstrap_ci_difference(ctrl_vals, test_vals, n_bootstrap=10000)

    print(f"\nPermutation test (median difference):")
    print(f"  Light OFF: median={np.median(ctrl_vals):.4f}, n={len(ctrl_vals)}")
    print(f"  Light ON:  median={np.median(test_vals):.4f}, n={len(test_vals)}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {d:.4f}")

    if p_value < 0.001:
        sig_marker = "***"
    elif p_value < 0.01:
        sig_marker = "**"
    elif p_value < 0.05:
        sig_marker = "*"
    else:
        sig_marker = "ns"

    # ---- Figure layout (100 mm × 125 mm, matching other panels) ----
    fig_width_mm = 100
    fig_height_mm = 125
    fig, ax = plt.subplots(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    font_size_ticks = 10
    font_size_labels = 14
    font_size_ann = 16

    # Order: control (off) at position 1, test (on) at position 2
    conditions = [control_condition, test_condition]
    positions = [1, 2]
    box_data = [ctrl_vals, test_vals]
    scatter_colors = [LIGHT_COLORS[control_condition], LIGHT_COLORS[test_condition]]

    # Boxplot with black outlines (scatter shows the color)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    # Jittered scatter points
    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, box_data, scatter_colors):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            s=20,
            color=color,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )

    # Significance bracket
    y_max = max(np.max(ctrl_vals), np.max(test_vals))
    y_min = min(np.min(ctrl_vals), np.min(test_vals))
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_annot = y_max + 0.1 * y_range

    ax.plot([1, 2], [y_annot, y_annot], "k-", linewidth=1.5)
    ax.plot([1, 1], [y_annot - 0.01 * y_range, y_annot], "k-", linewidth=1.5)
    ax.plot([2, 2], [y_annot - 0.01 * y_range, y_annot], "k-", linewidth=1.5)
    color_ann = "red" if sig_marker != "ns" else "gray"
    ax.text(
        1.5,
        y_annot + 0.01 * y_range,
        sig_marker,
        ha="center",
        va="bottom",
        fontsize=font_size_ann,
        color=color_ann,
        fontweight="bold" if sig_marker != "ns" else "normal",
    )

    ax.set_ylim(y_min - 0.05 * y_range, y_annot + 0.2 * y_range)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{LIGHT_LABELS[c]}\n(n={len(v)})" for c, v in zip(conditions, box_data)],
        fontsize=font_size_ticks,
    )
    ax.set_ylabel(METRIC_LABEL, fontsize=font_size_labels)
    ax.tick_params(axis="y", labelsize=font_size_ticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    stats_result = {
        "metric": METRIC,
        "n_light_off": len(ctrl_vals),
        "n_light_on": len(test_vals),
        "median_light_off": float(np.median(ctrl_vals)),
        "median_light_on": float(np.median(test_vals)),
        "mean_light_off": float(np.mean(ctrl_vals)),
        "mean_light_on": float(np.mean(test_vals)),
        "mean_diff_on_minus_off": float(np.mean(test_vals) - np.mean(ctrl_vals)),
        "ci_lower": float(ci_lower) if ci_lower is not None else np.nan,
        "ci_upper": float(ci_upper) if ci_upper is not None else np.nan,
        "cohens_d": float(d) if d is not None else np.nan,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "sig_level": sig_marker,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved boxplot to: {output_path}")

    plt.close(fig)

    return stats_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 9b: IR8a pulling ratio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations (default: 10000)",
    )
    args = parser.parse_args()

    output_dir = figure_output_dir("EDFigure9", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 9b: IR8a Pulling Ratio")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Load data
    data = load_and_clean_dataset(test_mode=args.test)

    if METRIC not in data.columns:
        raise ValueError(f"Column '{METRIC}' not found in dataset. Available: {list(data.columns)}")

    # Generate boxplot
    print(f"\nGenerating boxplot for {METRIC}...")
    plot_file = output_dir / "edfigure9b_pulling_ratio.pdf"

    stats_result = create_boxplot(
        data,
        control_condition="off",
        test_condition="on",
        group_col="Light",
        output_path=plot_file,
        n_permutations=args.n_permutations,
    )

    if stats_result is None:
        print("❌ Could not generate plot — no data")
        return

    # Save statistics
    stats_df = pd.DataFrame([stats_result])
    stats_file = output_dir / "edfigure9b_pulling_ratio_statistics.csv"
    stats_df.to_csv(stats_file, index=False, float_format="%.6f")
    print(f"✅ Statistics saved to: {stats_file}")

    # Summary
    print(f"\n{'='*60}")
    print("Statistical Summary:")
    print(f"{'='*60}")
    print(f"  Light OFF: median={stats_result['median_light_off']:.4f}, n={stats_result['n_light_off']}")
    print(f"  Light ON:  median={stats_result['median_light_on']:.4f}, n={stats_result['n_light_on']}")
    print(f"  p-value: {stats_result['p_value']:.4f}  ({stats_result['sig_level']})")
    print(f"  Cohen's d: {stats_result['cohens_d']:.4f}")

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 9b generated successfully!")
    print(f"{'='*60}")
    print(f"  Plot:  {plot_file}")
    print(f"  Stats: {stats_file}")


if __name__ == "__main__":
    main()
