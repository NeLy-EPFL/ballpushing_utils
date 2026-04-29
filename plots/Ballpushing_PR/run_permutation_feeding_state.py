#!/usr/bin/env python3
"""
Script to generate permutation-test jitterboxplots for FeedingState experiments.

This script generates comprehensive permutation-test visualizations with:
- Light ON condition only (filtered from dataset)
- FeedingState as grouping variable (fed / starved / starved_noWater)
- All pairwise comparisons (fed vs starved, fed vs starved_noWater, starved vs starved_noWater)
- FDR correction across all 3 comparisons per metric
- Statistical significance annotations (*, **, ***) for significant pairs only

Usage:
    python run_permutation_feeding_state.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: Skip metrics that already have plots.
    --test:         Process only first 3 metrics with a small sample (debugging).
"""
import sys
import time
import argparse
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent))  # Also add current directory
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))  # Add Plotting directory
from statsmodels.stats.multitest import multipletests
from ballpushing_utils import read_feather


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group

    Returns
    -------
    float
        Cohen's d effect size (positive = group2 > group1)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Calculate bootstrapped confidence interval for difference between groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group
    n_bootstrap : int
        Number of bootstrap resamples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI for group2 - group1
    """
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)

    return lower, upper


# ---------------------------------------------------------------------------
# Core plotting + statistics function
# ---------------------------------------------------------------------------


def generate_feeding_state_permutation_plots(
    data,
    metrics,
    y="FeedingState",
    control_condition="fed",
    palette="Set2",
    figsize=(12, 8),
    output_dir="permutation_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
    n_permutations=10000,
):
    """
    Generate jitterboxplots for each metric with permutation tests between each
    FeedingState and the control (fed). Applies FDR correction across comparisons
    for each metric.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset filtered to Light ON condition.
    metrics : list
        List of metric names to plot.
    y : str
        Column name for grouping (default: "FeedingState").
    control_condition : str
        Reference condition used only for ordering (default: "fed").
    palette : str or dict
        Color palette for the plots (default: "Set2").
    figsize : tuple
        Figure size (width, height).
    output_dir : str or Path
        Directory to save plots.
    fdr_method : str
        Method for multiple-testing correction (default: "fdr_bh").
    alpha : float
        Significance level after FDR correction (default: 0.05).
    n_permutations : int
        Number of permutations for permutation tests (default: 10000).

    Returns
    -------
    pd.DataFrame
        Statistics table with permutation test results and FDR-corrected p-values
        for all pairwise comparisons.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_metrics = {
        "max_event_time",
        "final_event_time",
        "first_significant_event_time",
        "first_major_event_time",
        "chamber_exit_time",
        "chamber_time",
        "time_chamber_beginning",
    }

    # Match font and PDF settings for consistency
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # All FeedingState conditions present in data
    all_conditions = sorted(data[y].unique())
    print(f"FeedingState conditions found in data: {all_conditions}")

    # Display labels
    value_to_label = {
        "fed": "Fed",
        "starved": "Starved",
        "starved_noWater": "Starved\n(no water)",
    }

    # Fixed ordering: control first, then test conditions
    fixed_order = ["fed", "starved", "starved_noWater"]
    fixed_order = [c for c in fixed_order if c in all_conditions]
    # Append any unexpected conditions at the end
    for c in all_conditions:
        if c not in fixed_order:
            fixed_order.append(c)
    print(f"Ordering for FeedingState: {fixed_order}")

    # Build colour palette (one colour per condition)
    palette_colors = sns.color_palette(palette, n_colors=len(fixed_order))
    condition_colors = {cond: palette_colors[i] for i, cond in enumerate(fixed_order)}
    print(f"Colour mapping: {condition_colors}")

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"\nGenerating permutation plot for metric {metric_idx + 1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y]).copy()

        # Determine whether metric is binary-like
        all_unique_vals = plot_data[metric].dropna().unique()
        is_binary_like = len(all_unique_vals) == 2 and set(all_unique_vals).issubset({0, 1, 0.0, 1.0})
        if is_binary_like:
            print(f"  Binary-like metric detected; permutation on proportions.")

        # --- All pairwise permutation tests ---
        present_conditions = [c for c in fixed_order if c in plot_data[y].unique()]
        all_pairs = list(combinations(present_conditions, 2))
        pvals = []
        pair_results = []

        for cond_a, cond_b in all_pairs:
            vals_a = plot_data[plot_data[y] == cond_a][metric].dropna()
            vals_b = plot_data[plot_data[y] == cond_b][metric].dropna()
            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            try:
                # Permutation test on median difference
                obs_stat = float(np.median(vals_b) - np.median(vals_a))
                combined = np.concatenate([vals_a.values, vals_b.values])
                n_a = len(vals_a)

                perm_diffs = np.empty(n_permutations)
                for i in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_diffs[i] = np.median(combined[n_a:]) - np.median(combined[:n_a])

                pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))
                direction = "increased" if obs_stat > 0 else ("decreased" if obs_stat < 0 else "none")
                test_name = f"permutation({n_permutations})"

                # Calculate effect size (Cohen's d)
                cohens_d_val = cohens_d(vals_a.values, vals_b.values)

                # Calculate bootstrapped CI for mean difference
                mean_diff = float(np.mean(vals_b) - np.mean(vals_a))
                ci_lower, ci_upper = bootstrap_ci_difference(
                    vals_a.values, vals_b.values, n_bootstrap=10000, ci=95, random_state=42
                )

                # Calculate percentage change
                mean_a = float(np.mean(vals_a))
                mean_b = float(np.mean(vals_b))
                if mean_a != 0:
                    pct_change = (mean_diff / mean_a) * 100
                    pct_ci_lower = (ci_lower / mean_a) * 100
                    pct_ci_upper = (ci_upper / mean_a) * 100
                else:
                    pct_change = np.nan
                    pct_ci_lower = np.nan
                    pct_ci_upper = np.nan

            except Exception as e:
                print(f"  Error in permutation test for {cond_a} vs {cond_b}: {e}")
                pval = 1.0
                obs_stat = np.nan
                direction = "none"
                test_name = f"permutation({n_permutations}) (failed)"
                cohens_d_val = np.nan
                mean_diff = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan
                mean_a = np.nan
                mean_b = np.nan

            pvals.append(pval)
            pair_results.append(
                {
                    "ConditionA": cond_a,
                    "ConditionB": cond_b,
                    "Metric": metric,
                    "pval_raw": pval,
                    "direction": direction,
                    "effect_size_median_diff": obs_stat,
                    "cohens_d": cohens_d_val,
                    "median_a": vals_a.median(),
                    "median_b": vals_b.median(),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "pct_change": pct_change,
                    "pct_ci_lower": pct_ci_lower,
                    "pct_ci_upper": pct_ci_upper,
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                    "test_type": test_name,
                }
            )

        if not pvals:
            print(f"  No valid comparisons for metric {metric}")
            continue

        # FDR correction across all pairwise comparisons
        if len(pvals) > 1:
            rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=fdr_method)
        else:
            pvals_corrected = np.array(pvals)
            rejected = [p < alpha for p in pvals]

        for i, result in enumerate(pair_results):
            result["pval_corrected"] = float(pvals_corrected[i])
            result["significant"] = bool(rejected[i])
            result["correction_applied"] = len(pvals) > 1

            p = pvals_corrected[i]
            if p < 0.001:
                result["sig_level"] = "***"
            elif p < 0.01:
                result["sig_level"] = "**"
            elif p < 0.05:
                result["sig_level"] = "*"
            else:
                result["sig_level"] = "ns"

            if not result["significant"]:
                result["direction"] = "none"

        all_stats.extend(pair_results)

        # --- Build ordered condition list for plotting ---
        sorted_conditions = [c for c in fixed_order if c in plot_data[y].unique()]

        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_conditions, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # --- Figure layout ---
        n_categories = len(sorted_conditions)
        fig_width = 120 / 25.4
        fig_height = 110 / 25.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        font_size_ticks = 9
        font_size_labels = 13
        font_size_annotations = 14

        # Draw boxplots per condition
        for idx, cond in enumerate(sorted_conditions):
            cond_data = plot_data[plot_data[y] == cond][metric].dropna()
            if len(cond_data) == 0:
                continue
            color = condition_colors[cond]
            linestyle = "solid" if cond == control_condition else "dashed"
            ax.boxplot(
                [cond_data],
                positions=[idx],
                widths=0.55,
                patch_artist=False,
                showfliers=False,
                vert=True,
                boxprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                whiskerprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                capprops=dict(color="black", linewidth=2.0, linestyle=linestyle),
                medianprops=dict(color="black", linewidth=2.5, linestyle=linestyle),
            )
            # Jitter overlay
            x_jitter = np.random.normal(idx, 0.07, size=len(cond_data))
            if cond == control_condition:
                ax.scatter(
                    x_jitter,
                    cond_data,
                    s=30,
                    facecolors=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.55,
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_jitter,
                    cond_data,
                    s=30,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.8,
                    alpha=0.65,
                    zorder=3,
                )

        # X-tick labels with n
        counts = [len(plot_data[plot_data[y] == c]) for c in sorted_conditions]
        display_labels = [f"{value_to_label.get(c, c)}\n(n={cnt})" for c, cnt in zip(sorted_conditions, counts)]
        ax.set_xticks(range(len(sorted_conditions)))
        ax.set_xticklabels(display_labels, fontsize=font_size_ticks)

        # Y limits with headroom for all pairwise brackets
        y_max = plot_data[metric].quantile(0.99)
        y_min = plot_data[metric].min()
        data_range = max(y_max - y_min, 1e-6)
        n_pairs = len(pair_results)
        annotation_headroom = 0.10 * data_range * max(1, n_pairs)
        ax.set_ylim(bottom=y_min - 0.05 * data_range, top=y_max + annotation_headroom)

        # --- Pairwise significance brackets (all pairs, skip ns) ---
        ann_y_base = y_max + 0.03 * data_range
        ann_y_step = 0.08 * data_range

        pair_result_map = {(r["ConditionA"], r["ConditionB"]): r for r in pair_results}
        ann_offset = 0
        for cond_a, cond_b in all_pairs:
            result = pair_result_map.get((cond_a, cond_b))
            if result is None:
                continue
            # Skip non-significant pairs — no bracket drawn
            if not result["significant"]:
                ann_offset += 1
                continue

            idx_a = sorted_conditions.index(cond_a) if cond_a in sorted_conditions else None
            idx_b = sorted_conditions.index(cond_b) if cond_b in sorted_conditions else None
            if idx_a is None or idx_b is None:
                continue

            x_left = min(idx_a, idx_b)
            x_right = max(idx_a, idx_b)
            ann_y = ann_y_base + ann_offset * ann_y_step

            ax.annotate(
                "",
                xy=(x_right, ann_y),
                xytext=(x_left, ann_y),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
            )
            ax.text(
                (x_left + x_right) / 2,
                ann_y,
                result["sig_level"],
                fontsize=font_size_annotations,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
            )
            ann_offset += 1

        display_metric = metric.replace("_", " ")
        if metric in time_metrics:
            display_metric = f"{display_metric} (min)"

        ax.set_ylabel(display_metric, fontsize=font_size_labels)
        ax.set_xlabel("Feeding State", fontsize=font_size_labels)
        plt.yticks(fontsize=font_size_ticks)
        ax.grid(False)
        ax.legend_ = None  # remove any legend seaborn/matplotlib may have added

        plt.tight_layout()

        output_file = output_dir / f"{metric}_feedingstate_permutation.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {output_file}")
        n_sig = sum(1 for r in pair_results if r["significant"])
        print(f"  Significant pairs: {n_sig}/{len(pair_results)} (α={alpha})")
        elapsed = time.time() - metric_start_time
        print(f"  Time: {elapsed:.2f}s")

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save detailed statistics CSV with publication-quality formatting
        stats_file = output_dir / "feedingstate_permutation_statistics.csv"

        # Select and reorder columns for better readability
        columns_ordered = [
            "Metric",
            "ConditionA",
            "ConditionB",
            "N_A",
            "N_B",
            "Mean_A",
            "Mean_B",
            "Median_A",
            "Median_B",
            "Mean_Diff",
            "CI_Lower",
            "CI_Upper",
            "Pct_Change",
            "Pct_CI_Lower",
            "Pct_CI_Upper",
            "Cohens_d",
            "Median_Diff",
            "P_Value_Raw",
            "P_Value_FDR",
            "Significant",
            "Direction",
        ]

        # Rename columns in stats_df to match ordered list
        rename_map = {
            "n_a": "N_A",
            "n_b": "N_B",
            "mean_a": "Mean_A",
            "mean_b": "Mean_B",
            "median_a": "Median_A",
            "median_b": "Median_B",
            "mean_diff": "Mean_Diff",
            "ci_lower": "CI_Lower",
            "ci_upper": "CI_Upper",
            "pct_change": "Pct_Change",
            "pct_ci_lower": "Pct_CI_Lower",
            "pct_ci_upper": "Pct_CI_Upper",
            "cohens_d": "Cohens_d",
            "effect_size_median_diff": "Median_Diff",
            "pval_raw": "P_Value_Raw",
            "pval_corrected": "P_Value_FDR",
            "direction": "Direction",
        }
        stats_df_out = stats_df.rename(columns=rename_map)

        # Select only the columns we want and in the right order
        columns_to_save = [c for c in columns_ordered if c in stats_df_out.columns]
        stats_df_out = stats_df_out[columns_to_save]

        # Format numeric columns
        for col in [
            "Mean_A",
            "Mean_B",
            "Median_A",
            "Median_B",
            "Mean_Diff",
            "CI_Lower",
            "CI_Upper",
            "Pct_Change",
            "Pct_CI_Lower",
            "Pct_CI_Upper",
            "Cohens_d",
            "Median_Diff",
            "P_Value_Raw",
            "P_Value_FDR",
        ]:
            if col in stats_df_out.columns:
                stats_df_out[col] = stats_df_out[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "NaN")

        stats_df_out.to_csv(stats_file, index=False)
        print(f"\nDetailed statistics saved: {stats_file}")

        report_file = output_dir / "feedingstate_permutation_report.md"
        generate_text_report(stats_df, data, control_condition, report_file)
        print(f"Report saved: {report_file}")

        n_metrics = stats_df["Metric"].nunique()
        total = len(stats_df)
        n_pairs = total // n_metrics if n_metrics else total
        sig = int(stats_df["significant"].sum())
        print(
            f"\n{n_pairs} pairwise comparisons per metric | {n_metrics} metrics | {total} total rows | Significant: {sig} ({100 * sig / total:.1f}%)"
        )

    return stats_df


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def generate_text_report(stats_df, data, control_condition, report_file):
    """Generate a human-readable markdown report of all pairwise statistical results."""
    lines = [
        "# FeedingState Permutation Test Report",
        "## Statistical Analysis Results (all pairwise comparisons)",
        "",
        "**Dataset filter:** Light ON only",
        "**Comparisons:** all pairwise (FDR-BH corrected per metric)",
        "**Effect sizes:** Cohen's d (standardized mean difference)",
        "**Confidence intervals:** Bootstrapped 95% CI (10,000 resamples)",
        "",
    ]

    for metric in sorted(stats_df["Metric"].unique()):
        metric_stats = stats_df[stats_df["Metric"] == metric]
        lines.append(f"## {metric}")
        lines.append("")

        for _, row in metric_stats.iterrows():
            sig = row.get("sig_level", "ns")
            cond_a = row["ConditionA"]
            cond_b = row["ConditionB"]

            # Build comprehensive statistics line
            line_parts = [
                f"**{cond_a} vs {cond_b}**:",
                f"Mean {cond_a}={row['mean_a']:.3f} (n={row['n_a']})",
                f"Mean {cond_b}={row['mean_b']:.3f} (n={row['n_b']})",
            ]

            # Add effect size and CI if available
            if pd.notna(row.get("mean_diff")):
                line_parts.append(f"ΔMean={row['mean_diff']:+.3f}")
            if pd.notna(row.get("ci_lower")) and pd.notna(row.get("ci_upper")):
                line_parts.append(f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            if pd.notna(row.get("cohens_d")):
                line_parts.append(f"d={row['cohens_d']:.3f}")

            # Add p-values
            p_raw = row.get("pval_raw", np.nan)
            p_fdr = row.get("pval_corrected", np.nan)
            if pd.notna(p_raw):
                line_parts.append(f"p={p_raw:.4f}")
            if pd.notna(p_fdr):
                line_parts.append(f"p_FDR={p_fdr:.4f}")

            line_parts.append(sig)
            lines.append("- " + " | ".join(line_parts))

        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary
    total = len(stats_df)
    sig_total = int(stats_df["significant"].sum())
    n_increased = len(stats_df[(stats_df["significant"]) & (stats_df["direction"] == "increased")])
    n_decreased = len(stats_df[(stats_df["significant"]) & (stats_df["direction"] == "decreased")])

    lines += [
        "## Summary",
        "",
        f"- Total pairwise comparisons: {total}",
        f"- Significant: {sig_total} ({100 * sig_total / total:.1f}%)",
        f"  - ConditionB > ConditionA: {n_increased}",
        f"  - ConditionB < ConditionA: {n_decreased}",
        f"- Non-significant: {total - sig_total} ({100 * (total - sig_total) / total:.1f}%)",
        "",
        f"- Metrics with at least one significant pair: "
        f"{stats_df[stats_df['significant']]['Metric'].nunique()}/{stats_df['Metric'].nunique()}",
        "",
    ]

    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    n_met = stats_df["Metric"].nunique()
    n_pr = total // n_met if n_met else total
    print(f"Generated text report ({n_pr} pairs × {n_met} metrics = {total} total rows)")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Experiments excluded because their recording duration is ~1 h instead of 2 h.
# Identified by checking coordinate files: max(time) < 1.9 h.
SHORT_EXPERIMENTS = [
    "240718_Afternoon_FeedingState_Videos_Tracked",
    "240718_Afternoon_FeedingState_next_Videos_Tracked",
]


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load the new exploration dataset, filter for Light ON, and normalise FeedingState.

    Parameters
    ----------
    test_mode : bool
        If True, sample a subset for faster processing.
    test_sample_size : int
        Number of rows to use in test mode.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset filtered to Light ON condition.
    """
    dataset_path = (
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"
        "260220_10_summary_control_folders_Data/summary/pooled_summary.feather"
    )
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = read_feather(dataset_path)
        print(f"Loaded successfully. Shape: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # --- Step 1: Filter for Light ON only ---
    if "Light" not in dataset.columns:
        raise ValueError("'Light' column not found in dataset.")
    print(f"Light values before filter: {sorted(dataset['Light'].dropna().unique())}")
    dataset = dataset[dataset["Light"] == "on"].copy()
    print(f"After Light=on filter. Shape: {dataset.shape}")

    # --- Step 2: Exclude short experiments (< 2 h recordings) ---
    if "experiment" in dataset.columns:
        before = len(dataset)
        dataset = dataset[~dataset["experiment"].isin(SHORT_EXPERIMENTS)].copy()
        dropped = before - len(dataset)
        if dropped:
            print(
                f"Excluded {dropped} rows from {len(SHORT_EXPERIMENTS)} short experiments (<2 h): {SHORT_EXPERIMENTS}"
            )
        else:
            print("No rows matched the short-experiment exclusion list.")
    else:
        print("Warning: 'experiment' column not found – short-experiment exclusion skipped.")

    # --- Step 3: Normalise FeedingState ---
    if "FeedingState" not in dataset.columns:
        raise ValueError("'FeedingState' column not found in dataset.")
    # Unify capitalisation: "Fed" → "fed"
    dataset["FeedingState"] = dataset["FeedingState"].str.strip().str.lower()
    # Remove rows with empty FeedingState
    dataset = dataset[dataset["FeedingState"] != ""].copy()
    # Rename to canonical labels
    rename_map = {"fed": "fed", "starved": "starved", "starved_nowater": "starved_noWater"}
    dataset["FeedingState"] = dataset["FeedingState"].map(rename_map).fillna(dataset["FeedingState"])
    print(f"FeedingState values after normalisation: {sorted(dataset['FeedingState'].unique())}")
    print(f"Counts:\n{dataset['FeedingState'].value_counts()}")

    # --- Step 3a: Normalise Period (keep column for AM/PM splitting in main) ---
    if "Period" in dataset.columns:
        dataset["Period"] = dataset["Period"].astype(str).str.strip()
        dataset["Period"] = dataset["Period"].apply(
            lambda p: "AM" if "AM" in p.upper() else ("PM" if "PM" in p.upper() else None)
        )
        dataset = dataset[dataset["Period"].isin(["AM", "PM"])].copy()
        print(f"Period values after normalisation: {dict(dataset['Period'].value_counts())}")

    # --- Step 3b: Drop unneeded metadata columns ---
    metadata_columns = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Genotype",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
        # Period intentionally kept for AM/PM subsetting
        "Orientation",
        "Crossing",
        "BallType",
        "Used_to ",
        "Magnet",
        "Peak",
        "Light",  # no longer needed after filter
    ]
    columns_to_drop = [c for c in metadata_columns if c in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"Dropped metadata columns: {columns_to_drop}")

    # --- Step 4: Convert boolean columns to int ---
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # --- Step 5: Drop known problematic columns ---
    drop_cols = [c for c in ["insight_effect", "insight_effect_log", "exit_time", "index"] if c in dataset.columns]
    if drop_cols:
        dataset.drop(columns=drop_cols, inplace=True)

    # --- Step 6: Fill NaN defaults for known new metrics ---
    new_metric_defaults = {
        "has_finished": 0,
        "persistence_at_end": 0.0,
        "fly_distance_moved": 0.0,
        "time_chamber_beginning": 0.0,
        "median_freeze_duration": 0.0,
        "fraction_not_facing_ball": 0.0,
        "flailing": 0.0,
        "head_pushing_ratio": 0.0,
    }
    fill_dict = {m: v for m, v in new_metric_defaults.items() if m in dataset.columns and dataset[m].isnull().any()}
    if fill_dict:
        dataset.fillna(fill_dict, inplace=True)
        print(f"Filled NaN defaults for: {list(fill_dict.keys())}")

    print(f"Dataset cleaning complete. Final shape: {dataset.shape}")

    if test_mode and len(dataset) > test_sample_size:
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"TEST MODE: sampled {test_sample_size} rows.")

    return dataset


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def should_skip_metric(metric, output_dir, overwrite=True):
    """Return True if metric should be skipped (existing plot + no overwrite)."""
    if overwrite:
        return False
    output_dir = Path(output_dir)
    patterns = [
        f"*{metric}*.pdf",
        f"*{metric}*.png",
        f"{metric}_*.pdf",
        f"{metric}_*.png",
    ]
    for pattern in patterns:
        if list(output_dir.glob(pattern)):
            print(f"  Skipping {metric}: existing plot found.")
            return True
    return False


def categorize_metric(col, dataset):
    """Categorise a metric column as 'continuous', 'binary', or other."""
    if col not in dataset.columns:
        return None
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        return "non_numeric"
    non_nan = dataset[col].dropna()
    if len(non_nan) < 3:
        return "insufficient_data"
    unique_vals = set(non_nan.unique())
    if len(unique_vals) == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
        return "binary"
    if len(unique_vals) < 3:
        return "categorical"
    return "continuous"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(overwrite=True, test_mode=False, period_mode="pooled"):
    """Run permutation-test plots for all metrics grouped by FeedingState."""
    print("Starting FeedingState permutation-test analysis (Light ON only)...")
    if not overwrite:
        print("Overwrite disabled: will skip metrics with existing plots.")
    if test_mode:
        print("TEST MODE: only first 3 metrics, small data sample.")

    # Output directory
    base_output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/Summary_metrics/Permutation")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {base_output_dir}")

    # ---- Load data ----
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # ---- Convert time metrics to minutes ----
    time_metrics = [
        "max_event_time",
        "final_event_time",
        "first_significant_event_time",
        "first_major_event_time",
        "chamber_exit_time",
        "chamber_time",
        "time_chamber_beginning",
    ]
    print("\nConverting time metrics from seconds to minutes...")
    for m in time_metrics:
        if m in dataset.columns:
            dataset[m] = dataset[m] / 60.0
            print(f"  Converted: {m}")

    print(f"\nDataset shape: {dataset.shape}")
    print(f"FeedingState distribution:\n{dataset['FeedingState'].value_counts()}")
    print(f"Period distribution:\n{dataset['Period'].value_counts() if 'Period' in dataset.columns else 'N/A'}")

    # ---- Determine run configs based on --period argument ----
    run_configs = []
    if period_mode in ("pooled", "both"):
        run_configs.append(("permutation_feeding_state", dataset))
    if period_mode in ("AM", "PM", "both") and "Period" in dataset.columns:
        periods = ["AM", "PM"] if period_mode == "both" else [period_mode]
        for period in periods:
            period_data = dataset[dataset["Period"] == period].copy()
            if len(period_data) > 0:
                run_configs.append((f"permutation_feeding_state_{period}", period_data))
                print(f"  Period {period}: {len(period_data)} rows")
            else:
                print(f"  Period {period}: no data, skipping")
    if not run_configs:
        print(f"No run configs for period_mode={period_mode!r}. Exiting.")
        return

    # ---- Metric discovery ----
    excluded_patterns = [
        "binned_",
        "r2",
        "slope",
        "_bin_",
        "logistic_",
        "learning_",
        "interaction_rate_bin",
        "binned_auc",
        "binned_slope",
    ]
    base_metrics = [
        "nb_events",
        "max_event",
        "max_event_time",
        "max_distance",
        "final_event",
        "final_event_time",
        "nb_significant_events",
        "significant_ratio",
        "first_significant_event",
        "first_significant_event_time",
        "first_major_event",
        "first_major_event_time",
        "major_event_first",
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_speed",
        "auc",
        "overall_interaction_rate",
        "chamber_time",
        "pushed",
    ]
    additional_patterns = [
        "speed",
        "speed",
        "pause",
        "freeze",
        "persistence",
        "trend",
        "interaction_rate",
        "finished",
        "chamber",
        "facing",
        "flailing",
        "head",
        "leg_visibility",
    ]

    available_metrics = []
    print("\nChecking available metrics...")
    for m in base_metrics:
        if m in dataset.columns and not any(p in m for p in excluded_patterns):
            available_metrics.append(m)
    for pat in additional_patterns:
        for col in dataset.columns:
            if pat in col.lower() and col not in available_metrics and col != "FeedingState":
                if not any(p in col.lower() for p in excluded_patterns):
                    available_metrics.append(col)

    # Categorise
    continuous_metrics = []
    binary_metrics = []
    for m in available_metrics:
        cat = categorize_metric(m, dataset)
        if cat == "continuous":
            continuous_metrics.append(m)
        elif cat == "binary":
            binary_metrics.append(m)
        elif cat == "insufficient_data":
            # Include anyway with a warning
            print(f"  Warning: {m} has few non-NaN values, including as continuous.")
            continuous_metrics.append(m)

    print(f"\nContinuous metrics: {len(continuous_metrics)}")
    print(f"Binary metrics: {len(binary_metrics)}")

    if test_mode:
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2]
        print(f"TEST MODE: reduced to {len(continuous_metrics)} continuous, {len(binary_metrics)} binary metrics.")

    if not continuous_metrics and not binary_metrics:
        print("No metrics to process. Exiting.")
        return

    all_metrics = continuous_metrics + binary_metrics

    # ---- Run permutation analysis for each config (all data, AM, PM) ----
    grand_t0 = time.time()
    for dir_name, subset_data in run_configs:
        subset_dir = base_output_dir / dir_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Drop Period column before analysis (not needed as a metric)
        required_cols = ["FeedingState"] + all_metrics
        required_cols = [c for c in required_cols if c in subset_data.columns]
        analysis_data = subset_data[required_cols].copy()

        label = dir_name.replace("permutation_feeding_state", "").strip("_") or "ALL"
        print(f"\n{'='*60}")
        print(f"Running: {dir_name}  (Period subset: {label})")
        print(f"{'='*60}")
        print(f"Analysis data shape: {analysis_data.shape}")
        print(f"FeedingState conditions: {sorted(analysis_data['FeedingState'].unique())}")
        for cond, cnt in analysis_data["FeedingState"].value_counts().items():
            print(f"  {cond}: {cnt} samples")

        metrics_this_run = all_metrics
        if not overwrite:
            metrics_this_run = [m for m in metrics_this_run if not should_skip_metric(m, subset_dir, overwrite=False)]

        t0 = time.time()
        generate_feeding_state_permutation_plots(
            data=analysis_data,
            metrics=metrics_this_run,
            y="FeedingState",
            control_condition="fed",
            palette="Set2",
            output_dir=str(subset_dir),
            fdr_method="fdr_bh",
            alpha=0.05,
            n_permutations=10000,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s – saved to: {subset_dir}")

    total_elapsed = time.time() - grand_t0
    print(f"\nAll analyses complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Permutation-test jitterboxplots grouped by FeedingState (Light ON only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_permutation_feeding_state.py
  python run_permutation_feeding_state.py --no-overwrite
  python run_permutation_feeding_state.py --test
        """,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip metrics that already have existing plots.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only 3 metrics, 200-row sample.",
    )
    parser.add_argument(
        "--period",
        choices=["pooled", "AM", "PM", "both"],
        default="pooled",
        help=(
            "Which data to analyse: "
            "'pooled' (default) = all periods together in one directory; "
            "'AM' / 'PM' = only that period; "
            "'both' = AM and PM in separate directories."
        ),
    )
    args = parser.parse_args()
    main(overwrite=not args.no_overwrite, test_mode=args.test, period_mode=args.period)
