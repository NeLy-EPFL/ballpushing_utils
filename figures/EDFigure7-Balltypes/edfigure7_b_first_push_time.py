#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 7b: Time to first major push by ball surface treatment.

This script generates a boxplot comparing:
- Time to first major push (>1.2 mm) across ball surface treatments
- Control (untreated stainless steel), rusty, and sandpaper balls
- Permutation tests (10,000 permutations) with FDR correction
- n = 24 per group

Expected results (from published figure):
- Rusty vs control: FDR-corrected p = 0.036 (*)
- Sandpaper vs control: FDR-corrected p = 0.036 (*)

Usage:
    python edfigure7_b_first_push_time.py [--test]

Arguments:
    --test: Test mode - use limited data for quick verification
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Add repository root to path for imports

from ballpushing_utils import figure_output_dir, dataset
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# Set matplotlib parameters for publication quality
# Pixel to mm conversion
PIXELS_PER_MM = 500 / 30

# Fixed color mapping consistent with panel (c) and original scripts
BALLTYPE_COLORS = {
    "control": "#7f7f7f",  # Grey
    "rusty": "#d62728",  # Red
    "sandpaper": "#ff69b4",  # Pink
}

# Dataset path
SUMMARY_PATH = (
    dataset("Ballpushing_Balltypes/Datasets/250815_17_summary_ball_types_Data/summary/pooled_summary.feather")
)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform two-tailed permutation test on median difference.

    Parameters:
    -----------
    group1 : array-like
        Control group values
    group2 : array-like
        Test group values
    n_permutations : int
        Number of permutations

    Returns:
    --------
    float
        Two-tailed p-value
    """
    if len(group1) == 0 or len(group2) == 0:
        return 1.0

    obs_stat = float(np.median(group2) - np.median(group1))

    combined = np.concatenate([group1, group2])
    n_control = len(group1)

    rng = np.random.default_rng(42)
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_control = shuffled[:n_control]
        perm_test = shuffled[n_control:]
        perm_diffs[i] = np.median(perm_test) - np.median(perm_control)

    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))
    return p_value


def load_and_clean_dataset(test_mode=False):
    """
    Load and clean the ball types summary dataset.

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing

    Returns:
    --------
    pd.DataFrame
        Cleaned dataset filtered to control, rusty, and sandpaper groups
    """
    print(f"Loading ball types summary dataset from: {SUMMARY_PATH}")
    try:
        dataset = pd.read_feather(SUMMARY_PATH)
        print(f"Loaded dataset with shape: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {SUMMARY_PATH}")

    # Drop TNT-specific metadata columns that are not needed for ball types analysis
    tnt_metadata_columns = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Genotype",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
    ]
    columns_to_drop = [col for col in tnt_metadata_columns if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)

    # Rename ball types for clarity (matching original scripts)
    if "BallType" in dataset.columns:
        balltype_mapping = {"ctrl": "control", "sand": "sandpaper"}
        dataset["BallType"] = dataset["BallType"].map(balltype_mapping).fillna(dataset["BallType"])
        print(f"Ball types after renaming: {sorted(dataset['BallType'].unique())}")

    # Filter to only control, rusty, sandpaper (the three groups shown in Figure 7)
    groups_of_interest = ["control", "rusty", "sandpaper"]
    dataset = dataset[dataset["BallType"].isin(groups_of_interest)].copy()
    print(f"After filtering to {groups_of_interest}: {dataset.shape}")
    print(f"Sample sizes per ball type:")
    for bt in groups_of_interest:
        n = len(dataset[dataset["BallType"] == bt])
        print(f"  {bt}: n={n}")

    if test_mode:
        print(f"TEST MODE: Sampling 30 rows per ball type for faster processing")
        sampled = []
        for bt in groups_of_interest:
            bt_data = dataset[dataset["BallType"] == bt]
            sampled.append(bt_data.head(min(30, len(bt_data))))
        dataset = pd.concat(sampled, ignore_index=True)
        print(f"Test dataset shape: {dataset.shape}")

    return dataset


def create_first_push_boxplot(data, output_path=None, n_permutations=10000):
    """
    Create a boxplot of time to first major push by ball type with permutation tests.

    Panel (b): Time to first major push (>1.2 mm) grouped by surface treatment.
    Uses the `first_major_event` metric (time at which the first major push
    occurred, already in the appropriate unit in the dataset).
    Groups compared using permutation tests with FDR correction across the
    2 comparisons (rusty vs control, sandpaper vs control).

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with BallType and first_major_event columns
    output_path : Path or str, optional
        Where to save the plot. If None, displays interactively.
    n_permutations : int
        Number of permutations for statistical testing

    Returns:
    --------
    pd.DataFrame
        Statistics table with permutation test results
    """
    metric = "first_major_event"
    control_group = "control"
    test_groups = ["rusty", "sandpaper"]
    group_col = "BallType"

    if metric not in data.columns:
        raise ValueError(f"Required metric '{metric}' not found in dataset. Available: {list(data.columns)}")

    data = data.copy()

    # Get data per group, dropping NaN and sentinel -1 values (no event)
    def clean_vals(series):
        return series.dropna().values[series.dropna().values != -1]

    control_vals = clean_vals(data[data[group_col] == control_group][metric])
    rusty_vals = clean_vals(data[data[group_col] == "rusty"][metric])
    sandpaper_vals = clean_vals(data[data[group_col] == "sandpaper"][metric])

    group_vals = {
        "control": control_vals,
        "rusty": rusty_vals,
        "sandpaper": sandpaper_vals,
    }

    print(f"\nSample sizes:")
    for name, vals in group_vals.items():
        print(f"  {name}: n={len(vals)}, median={np.median(vals):.3f}, mean={np.mean(vals):.3f}")

    # Permutation tests: rusty vs control, sandpaper vs control
    print(f"\nRunning permutation tests ({n_permutations} permutations)...")
    raw_pvals = []
    test_results = []

    for test_group in test_groups:
        test_vals = group_vals[test_group]
        p_raw = permutation_test(control_vals, test_vals, n_permutations=n_permutations)
        d = cohens_d(control_vals, test_vals)
        raw_pvals.append(p_raw)
        test_results.append(
            {
                "comparison": f"{test_group} vs {control_group}",
                "group": test_group,
                "control": control_group,
                "n_test": len(test_vals),
                "n_control": len(control_vals),
                "median_test": float(np.median(test_vals)),
                "median_control": float(np.median(control_vals)),
                "mean_test": float(np.mean(test_vals)),
                "mean_control": float(np.mean(control_vals)),
                "p_value_raw": p_raw,
                "cohens_d": d,
                "n_permutations": n_permutations,
            }
        )
        print(f"  {test_group} vs {control_group}: p_raw={p_raw:.4f}, Cohen's d={d:.3f}")

    # FDR correction across the 2 comparisons
    rejected, pvals_fdr, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    for i, result in enumerate(test_results):
        result["p_value_fdr"] = float(pvals_fdr[i])
        result["significant_fdr"] = bool(rejected[i])
        if pvals_fdr[i] < 0.001:
            result["sig_level"] = "***"
        elif pvals_fdr[i] < 0.01:
            result["sig_level"] = "**"
        elif pvals_fdr[i] < 0.05:
            result["sig_level"] = "*"
        else:
            result["sig_level"] = "ns"
        print(f"  {result['comparison']}: p_fdr={pvals_fdr[i]:.4f} ({result['sig_level']})")

    stats_df = pd.DataFrame(test_results)

    # ---- Create figure ----
    # Layout: 3 conditions (control, rusty, sandpaper) as vertical boxplots
    ordered_groups = ["control", "rusty", "sandpaper"]
    fig, ax = plt.subplots(figsize=(4, 6))

    x_positions = list(range(len(ordered_groups)))
    colors = [BALLTYPE_COLORS[g] for g in ordered_groups]
    box_data = [group_vals[g] for g in ordered_groups]

    # Boxplot with black outlines and colored boxes
    bp = ax.boxplot(
        box_data,
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for i, (g, vals) in enumerate(zip(ordered_groups, box_data)):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=20,
            color=BALLTYPE_COLORS[g],
            alpha=0.5,
            edgecolors="none",
            zorder=3,
        )

    # Significance brackets between control (idx=0) and test groups
    control_idx = ordered_groups.index("control")
    all_vals = np.concatenate([v for v in box_data if len(v) > 0])
    y_max = np.max(all_vals)
    y_min = np.min(all_vals)
    y_range = y_max - y_min

    bracket_base = y_max + 0.08 * y_range
    bracket_step = 0.12 * y_range

    for bracket_num, result in enumerate(test_results):
        test_idx = ordered_groups.index(result["group"])
        sig = result["sig_level"]
        if sig == "ns":
            continue

        current_height = bracket_base + bracket_num * bracket_step
        x1 = min(control_idx, test_idx)
        x2 = max(control_idx, test_idx)

        ax.plot([x1, x1], [current_height - 0.01 * y_range, current_height], "k-", linewidth=1.5)
        ax.plot([x1, x2], [current_height, current_height], "k-", linewidth=1.5)
        ax.plot([x2, x2], [current_height - 0.01 * y_range, current_height], "k-", linewidth=1.5)
        ax.text(
            (x1 + x2) / 2,
            current_height + 0.01 * y_range,
            sig,
            ha="center",
            va="bottom",
            fontsize=12,
            color="red",
            fontweight="bold",
        )

    # Set y-axis limit to accommodate brackets
    n_sig = sum(1 for r in test_results if r["sig_level"] != "ns")
    y_top = bracket_base + max(n_sig, 1) * bracket_step + 0.08 * y_range
    ax.set_ylim(y_min - 0.05 * y_range, y_top)

    # Axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{g}\n(n={len(group_vals[g])})" for g in ordered_groups],
        fontsize=9,
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Time to first major push (>1.2 mm)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # Also save as PNG
        plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved plot to: {output_path}")

    plt.close()
    return stats_df


def main():
    """Main function for Extended Data Figure 7b."""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 7b: Time to first major push by ball surface treatment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Number of permutations (default: 10000)")
    args = parser.parse_args()

    output_dir = figure_output_dir("EDFigure7", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 7b: Time to First Major Push")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Load data
    data = load_and_clean_dataset(test_mode=args.test)

    # Generate plot and statistics
    plot_file = output_dir / "edfigure7b_first_push_time.pdf"
    stats_df = create_first_push_boxplot(
        data,
        output_path=plot_file,
        n_permutations=args.n_permutations,
    )

    # Save statistics
    stats_file = output_dir / "edfigure7b_statistics.csv"
    stats_df.to_csv(stats_file, index=False, float_format="%.6f")
    print(f"✅ Statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Statistical Summary (panel b):")
    print(f"{'='*60}")
    for _, row in stats_df.iterrows():
        print(f"\n{row['comparison']}:")
        print(f"  {row['group']}: median={row['median_test']:.3f}, n={int(row['n_test'])}")
        print(f"  {row['control']}: median={row['median_control']:.3f}, n={int(row['n_control'])}")
        print(f"  p_raw={row['p_value_raw']:.4f}, p_fdr={row['p_value_fdr']:.4f} ({row['sig_level']})")
        print(f"  Cohen's d={row['cohens_d']:.3f}")

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 7b generated successfully!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  - Plot: {plot_file}")
    print(f"  - Plot (PNG): {plot_file.with_suffix('.png')}")
    print(f"  - Statistics: {stats_file}")
    print(f"\nExpected: rusty p_fdr ≈ 0.036 (*), sandpaper p_fdr ≈ 0.036 (*)")
    print(f"  (FDR correction applied across the 2 comparisons: rusty vs control, sandpaper vs control)")


if __name__ == "__main__":
    main()
