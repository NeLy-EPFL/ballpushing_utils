#!/usr/bin/env python3
"""
Script to generate permutation-test jitterboxplots for dark olfaction experiments.

This script generates comprehensive permutation-test visualizations with:
- Two grouping factors: Light (on/off) and Genotype (BallTypes are pooled)
- Permutation tests with 10,000 iterations
- No FDR correction needed (only 1 comparison per light condition)
- Side-by-side subplots for Light ON vs OFF conditions
- Statistical significance annotations (*, **, ***) centered between boxplots
- Genotype-dependent scatter colors (purple for IR8a, gray for EmptyGal4)
- Black boxplot outlines (color shown in scatter points)
- No legend (x-axis labels are self-explanatory)
- Consistent styling with other permutation scripts (100mm x 125mm, matching fonts)

Usage:
    python run_permutation_dark_olfaction.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: If specified, skip existing plots. Default is to overwrite.
    --test: Test mode - process only 3 metrics with limited data for debugging.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent))  # Also add current directory

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import scipy.stats as stats_module
import time
import argparse
from ballpushing_utils import read_feather

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def mm_to_inches(mm_value):
    """Convert mm to inches for matplotlib"""
    return mm_value / 25.4


def is_distance_metric(metric_name):
    """Check if a metric represents distance (in pixels)"""
    distance_keywords = [
        "distance",
        "dist",
        "head_ball",
        "fly_distance_moved",
        "max_distance",
        "distance_moved",
        "distance_ratio",
    ]
    return any(keyword in metric_name.lower() for keyword in distance_keywords)


def is_time_metric(metric_name):
    """Check if a metric represents time (in seconds)"""
    time_keywords = ["time", "duration", "pause", "stop", "freeze", "chamber_exit_time", "time_chamber_beginning"]
    # Exclude ratio metrics
    if "ratio" in metric_name.lower():
        return False
    return any(keyword in metric_name.lower() for keyword in time_keywords)


def convert_metric_data(data, metric_name):
    """Convert metric data from pixels to mm or seconds to minutes"""
    if is_distance_metric(metric_name):
        return data / PIXELS_PER_MM
    elif is_time_metric(metric_name):
        return data / 60.0
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "(mm)"
    elif is_time_metric(metric_name):
        return "(min)"
    return ""


def permutation_test(group1_data, group2_data, n_permutations=10000):
    """
    Perform a permutation test to compare two groups.

    Parameters:
    -----------
    group1_data : array-like
        Data from first group (control)
    group2_data : array-like
        Data from second group (test)
    n_permutations : int
        Number of permutations to perform

    Returns:
    --------
    float : Two-tailed p-value
    """
    # Observed difference in means
    obs_diff = np.mean(group2_data) - np.mean(group1_data)

    # Combine the data
    combined = np.concatenate([group1_data, group2_data])
    n_group1 = len(group1_data)
    n_group2 = len(group2_data)

    # Permutation test
    perm_diffs = []
    for _ in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_group1 = combined[:n_group1]
        perm_group2 = combined[n_group1:]
        perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return p_value


def bootstrap_ci_difference(group1_data, group2_data, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrap confidence interval for mean(group2) - mean(group1)."""
    if len(group1_data) == 0 or len(group2_data) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1_data, size=len(group1_data), replace=True)
        sample2 = rng.choice(group2_data, size=len(group2_data), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    ci_lower = float(np.percentile(bootstrap_diffs, alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 - alpha / 2))
    return ci_lower, ci_upper


def compute_detailed_effect_statistics(control_data, test_data, n_bootstrap=10000, random_state=42):
    """Compute detailed raw and percent effect summaries for one comparison."""
    control_n = int(len(control_data))
    test_n = int(len(test_data))

    control_mean = float(np.mean(control_data)) if control_n > 0 else np.nan
    test_mean = float(np.mean(test_data)) if test_n > 0 else np.nan
    control_median = float(np.median(control_data)) if control_n > 0 else np.nan
    test_median = float(np.median(test_data)) if test_n > 0 else np.nan

    mean_diff = test_mean - control_mean if np.isfinite(control_mean) and np.isfinite(test_mean) else np.nan
    median_diff = test_median - control_median if np.isfinite(control_median) and np.isfinite(test_median) else np.nan

    ci_lower, ci_upper = bootstrap_ci_difference(
        control_data,
        test_data,
        n_bootstrap=n_bootstrap,
        ci=95,
        random_state=random_state,
    )

    if np.isfinite(control_mean) and control_mean != 0 and np.isfinite(mean_diff):
        pct_change = (mean_diff / control_mean) * 100.0
        pct_ci_lower = (ci_lower / control_mean) * 100.0 if np.isfinite(ci_lower) else np.nan
        pct_ci_upper = (ci_upper / control_mean) * 100.0 if np.isfinite(ci_upper) else np.nan
    else:
        pct_change = np.nan
        pct_ci_lower = np.nan
        pct_ci_upper = np.nan

    pooled_std = np.sqrt((np.var(control_data, ddof=1) + np.var(test_data, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if np.isfinite(pooled_std) and pooled_std > 0 else np.nan

    return {
        "control_n": control_n,
        "test_n": test_n,
        "control_mean": control_mean,
        "test_mean": test_mean,
        "control_median": control_median,
        "test_median": test_median,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "effect_size_raw": mean_diff,
        "pct_change": pct_change,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pct_ci_lower": pct_ci_lower,
        "pct_ci_upper": pct_ci_upper,
        "cohens_d": cohens_d,
    }


def generate_dark_olfaction_permutation_plots(
    data,
    metrics,
    grouping_cols=["Light", "Genotype"],
    control_group=None,
    genotype_palette=None,
    fig_width_mm=100,
    fig_height_mm=125,
    font_size_ticks=10,
    font_size_labels=14,
    font_size_legend=12,
    font_size_annotations=16,
    output_dir="permutation_plots",
    alpha=0.05,
    n_permutations=10000,
    n_bootstrap=10000,
    split_by_light=True,
):
    """
    Generates jitterboxplots for each metric with permutation tests between groups.
    No FDR correction applied (only 1 comparison per light condition).
    Creates side-by-side subplots for different Light conditions.
    BallTypes are pooled together.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        grouping_cols (list): List of column names for grouping factors. Default is ["Light", "Genotype"].
        control_group (dict): Dictionary specifying control group values for each grouping column.
                             Example: {"Light": "off", "Genotype": "TNTxEmptyGal4"}
        genotype_palette (dict): Color palette mapping genotypes to colors. Default is None (auto-generate).
        fig_width_mm (float): Figure width in mm. Default is 100.
        fig_height_mm (float): Figure height in mm. Default is 125.
        font_size_ticks (float): Font size for tick labels. Default is 10.
        font_size_labels (float): Font size for axis labels. Default is 14.
        font_size_legend (float): Font size for legend. Default is 12.
        font_size_annotations (float): Font size for statistical annotations. Default is 16.
        output_dir: Directory to save the plots. Default is "permutation_plots".
        alpha (float): Significance level for determining significance. Default is 0.05.
        n_permutations (int): Number of permutations for permutation tests. Default is 10000.
        split_by_light (bool): If True, create side-by-side subplots for each Light condition. Default is True.

    Returns:
        pd.DataFrame: Statistics table with permutation test results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Set default control group if not specified
    if control_group is None:
        control_group = {"Light": "off", "Genotype": "TNTxEmptyGal4"}

    # Set default genotype palette if not specified (purple for IR8a, gray for EmptyGal4)
    if genotype_palette is None:
        genotype_palette = {"TNTxEmptyGal4": "#7f7f7f", "TNTxIR8a": "#9467bd"}

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"\nProcessing metric {metric_idx+1}/{len(metrics)}: {metric}")

        # Drop rows with NaN in the metric or grouping columns
        plot_data = data.dropna(subset=[metric] + grouping_cols).copy()

        if len(plot_data) == 0:
            print(f"  ⚠️  No data available for metric '{metric}' after removing NaNs. Skipping.")
            continue

        # Convert metric data to appropriate units (pixels to mm, seconds to minutes)
        plot_data[metric] = convert_metric_data(plot_data[metric], metric)
        metric_unit = get_metric_unit(metric)

        # Create combined grouping column (pooling BallTypes)
        plot_data["Group"] = plot_data.apply(lambda row: f"{row['Light']}_{row['Genotype']}", axis=1)

        # Get all unique groups
        groups = sorted(plot_data["Group"].unique())

        # Perform permutation tests WITHIN each light condition
        # Compare test genotype to control genotype for each light condition separately
        p_values_by_group = {}  # Dictionary mapping group name to p-value
        effect_sizes_by_group = {}
        detailed_stats_by_group = {}
        medians = {}

        control_genotype = control_group["Genotype"]

        # Get unique light conditions
        light_conditions = sorted(plot_data["Light"].unique())

        for light_val in light_conditions:
            # Get control and test groups for this light condition
            control_group_name = f"{light_val}_{control_genotype}"

            if control_group_name not in plot_data["Group"].values:
                print(f"  ⚠️  Control group '{control_group_name}' not found for Light={light_val}. Skipping.")
                continue

            control_data = plot_data[plot_data["Group"] == control_group_name][metric].values
            medians[control_group_name] = np.median(control_data)

            # Find test groups in this light condition (all genotypes except control)
            test_groups_in_light = [g for g in groups if g.startswith(f"{light_val}_") and g != control_group_name]

            for test_idx, test_group in enumerate(test_groups_in_light):
                test_data = plot_data[plot_data["Group"] == test_group][metric].values

                if len(test_data) < 2 or len(control_data) < 2:
                    print(f"  ⚠️  Insufficient data for comparison {test_group} vs {control_group_name}. Skipping.")
                    continue

                # Perform permutation test
                p_value = permutation_test(control_data, test_data, n_permutations=n_permutations)
                p_values_by_group[test_group] = p_value

                detailed_stats = compute_detailed_effect_statistics(
                    control_data,
                    test_data,
                    n_bootstrap=n_bootstrap,
                    random_state=42 + metric_idx * 100 + test_idx,
                )
                effect_sizes_by_group[test_group] = detailed_stats["cohens_d"]

                # Store medians
                medians[test_group] = np.median(test_data)

                # Store detailed stats for output table
                detailed_stats_by_group[test_group] = detailed_stats

        if len(p_values_by_group) == 0:
            print(f"  ⚠️  No valid comparisons for metric '{metric}'. Skipping.")
            continue

        # Store statistics
        for test_group, p_value in p_values_by_group.items():
            # Determine the control group for this test group
            light_val = test_group.split("_")[0]
            control_group_name_for_this = f"{light_val}_{control_genotype}"

            all_stats.append(
                {
                    "Metric": metric,
                    "Control_Group": control_group_name_for_this,
                    "Test_Group": test_group,
                    "Control_n": detailed_stats_by_group[test_group]["control_n"],
                    "Test_n": detailed_stats_by_group[test_group]["test_n"],
                    "Control_mean": detailed_stats_by_group[test_group]["control_mean"],
                    "Test_mean": detailed_stats_by_group[test_group]["test_mean"],
                    "Control_median": detailed_stats_by_group[test_group]["control_median"],
                    "Test_median": detailed_stats_by_group[test_group]["test_median"],
                    "Mean_diff": detailed_stats_by_group[test_group]["mean_diff"],
                    "Median_diff": detailed_stats_by_group[test_group]["median_diff"],
                    "Effect_size_raw": detailed_stats_by_group[test_group]["effect_size_raw"],
                    "Pct_change": detailed_stats_by_group[test_group]["pct_change"],
                    "CI_lower": detailed_stats_by_group[test_group]["ci_lower"],
                    "CI_upper": detailed_stats_by_group[test_group]["ci_upper"],
                    "Pct_CI_lower": detailed_stats_by_group[test_group]["pct_ci_lower"],
                    "Pct_CI_upper": detailed_stats_by_group[test_group]["pct_ci_upper"],
                    "p_value": p_value,
                    "significant": p_value < alpha,
                    "effect_size": effect_sizes_by_group[test_group],
                    "cohens_d": detailed_stats_by_group[test_group]["cohens_d"],
                    "direction": (
                        "increased"
                        if detailed_stats_by_group[test_group]["mean_diff"] > 0
                        else "decreased" if detailed_stats_by_group[test_group]["mean_diff"] < 0 else "neutral"
                    ),
                    "n_permutations": int(n_permutations),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

        # Get all unique groups (including controls)
        all_groups = sorted(plot_data["Group"].unique())

        # Create figure
        if split_by_light:
            # Create subplots for each light condition
            light_values = sorted(plot_data["Light"].unique())
            n_subplots = len(light_values)

            fig, axes = plt.subplots(
                1,
                n_subplots,
                figsize=(mm_to_inches(fig_width_mm * n_subplots), mm_to_inches(fig_height_mm)),
                sharey=True,
            )

            if n_subplots == 1:
                axes = [axes]

            for idx, light_val in enumerate(light_values):
                ax = axes[idx]

                # Filter groups for this light condition
                light_groups = [g for g in all_groups if g.startswith(f"{light_val}_")]

                # Sort: control first (EmptyGal4), then test (by median)
                control_group_this_light = f"{light_val}_{control_genotype}"

                def sort_key(group):
                    is_control = 1 if group == control_group_this_light else 0
                    return (is_control, medians.get(group, 0))

                sorted_light_groups = sorted(light_groups, key=sort_key, reverse=True)
                light_data = plot_data[plot_data["Light"] == light_val]

                # Create positions for boxplots
                positions = np.arange(len(sorted_light_groups))

                # Calculate y-axis limits
                y_min, y_max = light_data[metric].min(), light_data[metric].max()
                y_range = y_max - y_min if y_max > y_min else 1
                y_bottom = y_min - 0.1 * y_range
                y_top = y_max + 0.2 * y_range

                # Plot boxplots
                for pos_idx, group in enumerate(sorted_light_groups):
                    group_data_plot = light_data[light_data["Group"] == group][metric].values

                    # Extract genotype from group name
                    parts = group.split("_")
                    genotype = parts[1]

                    # Get color for scatter points
                    color = genotype_palette.get(genotype, "gray")

                    # Create boxplot with black outline
                    bp = ax.boxplot(
                        [group_data_plot],
                        positions=[positions[pos_idx]],
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,  # Don't show outliers since we have scatter
                        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
                        whiskerprops=dict(color="black", linewidth=1.5),
                        capprops=dict(color="black", linewidth=1.5),
                        medianprops=dict(color="black", linewidth=2),
                        zorder=2,
                    )

                    # Add jitter points with genotype color
                    np.random.seed(42)
                    jitter = np.random.normal(0, 0.04, size=len(group_data_plot))
                    ax.scatter(positions[pos_idx] + jitter, group_data_plot, alpha=0.6, s=20, color=color, zorder=3)

                # Add significance annotation centered between the two boxplots
                if len(sorted_light_groups) == 2:
                    # Find the test group in this light condition (the one that's not the control)
                    control_group_this_light = f"{light_val}_{control_genotype}"
                    test_groups = [g for g in sorted_light_groups if g != control_group_this_light]

                    if test_groups:
                        test_group = test_groups[0]

                        # Look up p-value for this test group
                        if test_group in p_values_by_group:
                            p_val = p_values_by_group[test_group]

                            if p_val < 0.001:
                                sig_text = "***"
                            elif p_val < 0.01:
                                sig_text = "**"
                            elif p_val < 0.05:
                                sig_text = "*"
                            else:
                                sig_text = ""

                            if sig_text:
                                # Position annotation centered between the two groups
                                center_x = (positions[0] + positions[1]) / 2
                                ax.text(
                                    center_x,
                                    y_top - 0.05 * y_range,
                                    sig_text,
                                    ha="center",
                                    va="top",
                                    fontsize=font_size_annotations,
                                    color="red",
                                    fontweight="bold",
                                )

                # Format subplot
                ax.set_xticks(positions)
                ax.set_xticklabels(
                    [g.split("_")[1] for g in sorted_light_groups], rotation=45, ha="right", fontsize=font_size_ticks
                )
                ax.tick_params(axis="y", labelsize=font_size_ticks)
                ax.set_ylim(y_bottom, y_top)

                if idx == 0:
                    ylabel = f"{metric} {metric_unit}".strip()
                    ax.set_ylabel(ylabel, fontsize=font_size_labels)

                ax.set_title(f"Light: {light_val}", fontsize=font_size_labels, pad=10)

        else:  # Single plot with all groups
            fig, ax = plt.subplots(figsize=(mm_to_inches(fig_width_mm), mm_to_inches(fig_height_mm)))

            # Sort all groups (controls first for each light condition, then by median)
            def sort_key_all(group):
                light_val = group.split("_")[0]
                genotype = group.split("_")[1]
                is_control = 1 if genotype == control_genotype else 0
                return (light_val, is_control, medians.get(group, 0))

            sorted_all_groups = sorted(all_groups, key=sort_key_all, reverse=False)
            positions = np.arange(len(sorted_all_groups))

            # Calculate y-axis limits
            y_min, y_max = plot_data[metric].min(), plot_data[metric].max()
            y_range = y_max - y_min if y_max > y_min else 1
            y_bottom = y_min - 0.1 * y_range
            y_top = y_max + 0.2 * y_range

            # Plot boxplots
            for pos_idx, group in enumerate(sorted_all_groups):
                group_data_plot = plot_data[plot_data["Group"] == group][metric].values

                # Extract genotype
                parts = group.split("_")
                light = parts[0]
                genotype = parts[1]

                # Get color for scatter points
                color = genotype_palette.get(genotype, "gray")

                # Create boxplot with black outline
                bp = ax.boxplot(
                    [group_data_plot],
                    positions=[positions[pos_idx]],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,  # Don't show outliers since we have scatter
                    boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    medianprops=dict(color="black", linewidth=2),
                    zorder=2,
                )

                # Add jitter points with genotype color
                np.random.seed(42)
                jitter = np.random.normal(0, 0.04, size=len(group_data_plot))
                ax.scatter(positions[pos_idx] + jitter, group_data_plot, alpha=0.6, s=20, color=color, zorder=3)

            # Note: Significance annotations not implemented for non-split mode
            # (Would require different logic to determine which groups to compare)

            ax.set_xticks(positions)
            ax.set_xticklabels(sorted_all_groups, rotation=45, ha="right", fontsize=font_size_ticks)
            ax.tick_params(axis="y", labelsize=font_size_ticks)
            ax.set_ylim(y_bottom, y_top)
            ylabel = f"{metric} {metric_unit}".strip()
            ax.set_ylabel(ylabel, fontsize=font_size_labels)

        # No legend needed - x-axis labels are self-explanatory
        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"{metric}_permutation_test.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        metric_time = time.time() - metric_start_time
        print(f"  ✅ Saved: {output_path} (took {metric_time:.1f}s)")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        stats_path = output_dir / "dark_olfaction_permutation_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✅ Statistics saved to: {stats_path}")

        # Generate markdown report
        report_file = output_dir / "dark_olfaction_permutation_report.md"
        generate_text_report(stats_df, data, control_group, report_file)

    return stats_df


def generate_genotype_light_permutation_plots(
    data,
    metrics,
    genotypes=None,
    light_col="Light",
    genotype_col="Genotype",
    control_light="off",
    test_light="on",
    fig_width_mm=100,
    fig_height_mm=125,
    font_size_ticks=10,
    font_size_labels=14,
    font_size_annotations=16,
    output_dir="permutation_dark_olfaction_light_comparison_by_genotype",
    alpha=0.05,
    n_permutations=10000,
    n_bootstrap=10000,
    genotype_palette=None,
):
    """
    Generate one-plot-per-metric permutation comparisons (Light on vs off) for each genotype,
    displayed side by side with a shared y-axis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if genotype_palette is None:
        genotype_palette = {"TNTxEmptyGal4": "#7f7f7f", "TNTxIR8a": "#9467bd"}

    if genotypes is None:
        available_genotypes = [g for g in ["TNTxEmptyGal4", "TNTxIR8a"] if g in data[genotype_col].dropna().unique()]
        remaining = sorted([g for g in data[genotype_col].dropna().unique() if g not in available_genotypes])
        genotypes = available_genotypes + remaining

    if len(genotypes) == 0:
        print("⚠️  No genotypes found. Skipping genotype light comparison plots.")
        return pd.DataFrame()

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"\n[Genotype light comparison] Processing metric {metric_idx+1}/{len(metrics)}: {metric}")

        metric_data = data.dropna(subset=[metric, light_col, genotype_col]).copy()
        if metric_data.empty:
            print(f"  ⚠️  No data available for metric '{metric}' after removing NaNs. Skipping.")
            continue

        metric_data[metric] = convert_metric_data(metric_data[metric], metric)
        metric_unit = get_metric_unit(metric)

        genotype_results = []
        for genotype_idx, genotype in enumerate(genotypes):
            genotype_metric_data = metric_data[metric_data[genotype_col] == genotype].copy()
            if genotype_metric_data.empty:
                print(f"  ⚠️  No data for genotype '{genotype}' and metric '{metric}'.")
                continue

            genotype_light_values = sorted(
                genotype_metric_data[light_col].dropna().unique(), key=lambda x: (x != "off", x)
            )
            if control_light not in genotype_light_values or test_light not in genotype_light_values:
                print(
                    f"  ⚠️  Required light conditions '{control_light}' and '{test_light}' not both present for {genotype}. "
                    "Skipping this genotype."
                )
                continue

            control_data = genotype_metric_data[genotype_metric_data[light_col] == control_light][metric].values
            test_data = genotype_metric_data[genotype_metric_data[light_col] == test_light][metric].values

            if len(control_data) < 2 or len(test_data) < 2:
                print(
                    f"  ⚠️  Insufficient data for {genotype}: '{test_light}' vs '{control_light}' "
                    f"(n={len(test_data)} vs n={len(control_data)}). Skipping this genotype."
                )
                continue

            p_value = permutation_test(control_data, test_data, n_permutations=n_permutations)
            detailed_stats = compute_detailed_effect_statistics(
                control_data,
                test_data,
                n_bootstrap=n_bootstrap,
                random_state=4242 + metric_idx * 100 + genotype_idx,
            )

            genotype_results.append(
                {
                    "genotype": genotype,
                    "p_value": p_value,
                    "detailed_stats": detailed_stats,
                    "group_arrays": [control_data, test_data],
                }
            )

            all_stats.append(
                {
                    "Metric": metric,
                    "Genotype": genotype,
                    "Control_Group": f"{control_light}_{genotype}",
                    "Test_Group": f"{test_light}_{genotype}",
                    "Control_n": detailed_stats["control_n"],
                    "Test_n": detailed_stats["test_n"],
                    "Control_mean": detailed_stats["control_mean"],
                    "Test_mean": detailed_stats["test_mean"],
                    "Control_median": detailed_stats["control_median"],
                    "Test_median": detailed_stats["test_median"],
                    "Mean_diff": detailed_stats["mean_diff"],
                    "Median_diff": detailed_stats["median_diff"],
                    "Effect_size_raw": detailed_stats["effect_size_raw"],
                    "Pct_change": detailed_stats["pct_change"],
                    "CI_lower": detailed_stats["ci_lower"],
                    "CI_upper": detailed_stats["ci_upper"],
                    "Pct_CI_lower": detailed_stats["pct_ci_lower"],
                    "Pct_CI_upper": detailed_stats["pct_ci_upper"],
                    "p_value": p_value,
                    "significant": p_value < alpha,
                    "effect_size": detailed_stats["cohens_d"],
                    "cohens_d": detailed_stats["cohens_d"],
                    "direction": (
                        "increased"
                        if detailed_stats["mean_diff"] > 0
                        else "decreased" if detailed_stats["mean_diff"] < 0 else "neutral"
                    ),
                    "n_permutations": int(n_permutations),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

        if len(genotype_results) == 0:
            print(f"  ⚠️  No valid genotype-level light comparisons for metric '{metric}'. Skipping figure.")
            continue

        n_subplots = len(genotype_results)
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=(mm_to_inches(fig_width_mm * n_subplots), mm_to_inches(fig_height_mm)),
            sharey=True,
        )
        if n_subplots == 1:
            axes = [axes]

        y_min = min(np.min(result["group_arrays"][0]) for result in genotype_results)
        y_min = min(y_min, min(np.min(result["group_arrays"][1]) for result in genotype_results))
        y_max = max(np.max(result["group_arrays"][0]) for result in genotype_results)
        y_max = max(y_max, max(np.max(result["group_arrays"][1]) for result in genotype_results))
        y_range = y_max - y_min if y_max > y_min else 1
        y_bottom = y_min - 0.1 * y_range
        y_top = y_max + 0.2 * y_range

        groups = [control_light, test_light]
        positions = np.arange(len(groups))

        for subplot_idx, result in enumerate(genotype_results):
            ax = axes[subplot_idx]
            genotype = result["genotype"]
            group_arrays = result["group_arrays"]
            p_value = result["p_value"]
            genotype_color = genotype_palette.get(genotype, "gray")

            for pos_idx, group_data_plot in enumerate(group_arrays):
                ax.boxplot(
                    [group_data_plot],
                    positions=[positions[pos_idx]],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    medianprops=dict(color="black", linewidth=2),
                    zorder=2,
                )

                np.random.seed(42)
                jitter = np.random.normal(0, 0.04, size=len(group_data_plot))
                ax.scatter(
                    positions[pos_idx] + jitter,
                    group_data_plot,
                    alpha=0.6,
                    s=20,
                    color=genotype_color,
                    zorder=3,
                )

            if p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            elif p_value < 0.05:
                sig_text = "*"
            else:
                sig_text = ""

            if sig_text:
                center_x = (positions[0] + positions[1]) / 2
                ax.text(
                    center_x,
                    y_top - 0.05 * y_range,
                    sig_text,
                    ha="center",
                    va="top",
                    fontsize=font_size_annotations,
                    color="red",
                    fontweight="bold",
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(groups, rotation=0, fontsize=font_size_ticks)
            ax.tick_params(axis="y", labelsize=font_size_ticks)
            ax.set_ylim(y_bottom, y_top)
            ax.set_title(f"{genotype}: Light effect", fontsize=font_size_labels, pad=10)

            if subplot_idx == 0:
                ylabel = f"{metric} {metric_unit}".strip()
                ax.set_ylabel(ylabel, fontsize=font_size_labels)

        plt.tight_layout()
        output_path = output_dir / f"{metric}_genotype_light_permutation_test.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        metric_time = time.time() - metric_start_time
        print(f"  ✅ Saved: {output_path} (took {metric_time:.1f}s)")

    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_path = output_dir / "dark_olfaction_genotype_light_permutation_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✅ Genotype light comparison statistics saved to: {stats_path}")

    return stats_df


def generate_text_report(stats_df, data, control_group, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    control_group : dict
        Dictionary specifying control group values
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Dark Olfaction Permutation Test Report")
    report_lines.append("## FDR-Corrected Statistical Analysis Results")
    report_lines.append("")
    control_str = "_".join([control_group["Light"], control_group["Genotype"]])
    report_lines.append(f"**Control group:** {control_str}")
    report_lines.append("")

    # Group by metric
    metrics = stats_df["Metric"].unique()

    for metric in sorted(metrics):
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"### {metric}")
        report_lines.append("")

        # Significant results
        sig_results = metric_stats[metric_stats["significant"]]
        if len(sig_results) > 0:
            report_lines.append("**Significant comparisons:**")
            report_lines.append("")

            for _, row in sig_results.iterrows():
                direction_symbol = "↑" if row["direction"] == "increased" else "↓"
                report_lines.append(
                    f"- {row['Test_Group']} vs {row['Control_Group']}: "
                    f"p = {row['p_value']:.4f} {direction_symbol} "
                    f"(effect size: {row['effect_size']:.3f})"
                )

            report_lines.append("")

        # Non-significant results
        non_sig = metric_stats[~metric_stats["significant"]]
        if len(non_sig) > 0:
            report_lines.append("**Non-significant comparisons:**")
            report_lines.append("")

            for _, row in non_sig.iterrows():
                report_lines.append(
                    f"- {row['Test_Group']} vs {row['Control_Group']}: " f"p = {row['p_value']:.4f} (n.s.)"
                )

            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Add summary section
    report_lines.append("## Summary")
    report_lines.append("")

    total_comparisons = len(stats_df)
    total_significant = stats_df["significant"].sum()
    total_increased = len(stats_df[(stats_df["significant"]) & (stats_df["direction"] == "increased")])
    total_decreased = len(stats_df[(stats_df["significant"]) & (stats_df["direction"] == "decreased")])

    report_lines.append(f"- **Total comparisons:** {total_comparisons}")
    report_lines.append(
        f"- **Significant results:** {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
    )
    report_lines.append(f"  - Significantly increased: {total_increased}")
    report_lines.append(f"  - Significantly decreased: {total_decreased}")
    report_lines.append(
        f"- **Non-significant results:** {total_comparisons - total_significant} ({100*(total_comparisons - total_significant)/total_comparisons:.1f}%)"
    )
    report_lines.append("")

    # Add metrics summary
    metrics_with_significant = stats_df[stats_df["significant"]]["Metric"].nunique()
    total_metrics = stats_df["Metric"].nunique()

    report_lines.append(
        f"- **Metrics with significant differences:** {metrics_with_significant}/{total_metrics} ({100*metrics_with_significant/total_metrics:.1f}%)"
    )
    report_lines.append("")

    # Write to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Generated text report: {report_file}")


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the dark olfaction dataset

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the dark olfaction dataset
    dataset_path = "/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Datasets/251106_08_summary_TNT_Olfaction_Dark_Data/summary/pooled_summary.feather"

    print(f"Loading dark olfaction dataset from: {dataset_path}")
    try:
        dataset = read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at: {dataset_path}")
        sys.exit(1)

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"Dropped problematic columns: {columns_to_drop}")

    # Handle new metrics with appropriate defaults
    new_metrics_defaults = {
        "has_finished": np.nan,
        "persistence_at_end": np.nan,
        "fly_distance_moved": np.nan,
        "time_chamber_beginning": np.nan,
        "median_freeze_duration": np.nan,
        "fraction_not_facing_ball": np.nan,
        "flailing": np.nan,
        "head_pushing_ratio": np.nan,
        "compute_median_head_ball_distance": np.nan,
        "compute_mean_head_ball_distance": np.nan,
        "median_head_ball_distance": np.nan,
        "mean_head_ball_distance": np.nan,
    }

    # Fill all new metrics at once
    fillna_dict = {}
    for metric, default_value in new_metrics_defaults.items():
        if metric in dataset.columns:
            if dataset[metric].isna().all():
                fillna_dict[metric] = default_value

    if fillna_dict:
        dataset = dataset.fillna(fillna_dict)
        print(f"Filled NaN values for {len(fillna_dict)} metrics with defaults")

    print(f"Dataset cleaning completed successfully")

    # Test mode sampling
    if test_mode and len(dataset) > test_sample_size:
        dataset = dataset.sample(n=test_sample_size, random_state=42)
        print(f"Test mode: Sampled {test_sample_size} rows")

    return dataset


def main(overwrite=True, test_mode=False, n_bootstrap=10000):
    """Main function to run permutation test plots for all metrics.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots (default). If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample data for fast testing.
    """
    print(f"Starting permutation test analysis for dark olfaction experiments...")
    if test_mode:
        print("  🧪 TEST MODE: Processing limited metrics and data")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Plots")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Create condition subdirectory
    condition_subdir = "permutation_dark_olfaction"
    condition_dir = base_output_dir / condition_subdir
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {condition_dir}")

    # Create dedicated genotype light-comparison subdirectory
    genotype_light_subdir = "permutation_dark_olfaction_light_comparison_by_genotype"
    genotype_light_dir = base_output_dir / genotype_light_subdir
    genotype_light_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {genotype_light_dir}")

    # Load the dataset
    print("\nLoading dark olfaction dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Check for required grouping columns
    required_cols = ["Light", "Genotype"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"\n✅ All required grouping columns found")
    print(f"Light conditions: {sorted(dataset['Light'].unique())}")
    print(f"Genotypes: {sorted(dataset['Genotype'].unique())}")
    if "BallType" in dataset.columns:
        print(f"BallTypes (will be pooled): {sorted(dataset['BallType'].unique())}")

    # Define metrics to analyze
    # Test behavior metrics
    all_metrics = [
        # Ball displacement metrics (will be converted from pixels to mm)
        "final_event",
        "max_distance",
        "distance_moved",
        "distance_ratio",
        # Interaction metrics
        "pushed",
        "pulled",
        "pulling_ratio",
        "overall_interaction_rate",
        "interaction_persistence",
        "interaction_proportion",
        # Temporal metrics (will be converted from seconds to minutes)
        "time_to_first_interaction",
        "chamber_exit_time",
        "chamber_time",
        "chamber_ratio",
        # Stop and pause metrics (time will be converted to minutes)
        "nb_stops",
        "total_stop_duration",
        "median_stop_duration",
        "nb_pauses",
        "total_pause_duration",
        "median_pause_duration",
        # Fly movement (will be converted from pixels to mm)
        "fly_distance_moved",
        # Head-ball metrics (will be converted from pixels to mm)
        "median_head_ball_distance",
        "mean_head_ball_distance",
        "head_pushing_ratio",
        "fraction_not_facing_ball",
    ]

    # Filter to metrics that exist in the dataset
    available_metrics = [m for m in all_metrics if m in dataset.columns]
    print(f"\nFound {len(available_metrics)}/{len(all_metrics)} metrics in dataset")

    if test_mode:
        available_metrics = available_metrics[:3]
        print(f"Test mode: Processing only first {len(available_metrics)} metrics")

    # Define control group
    control_group = {"Light": "off", "Genotype": "TNTxEmptyGal4"}

    # Define genotype color palette (purple for IR8a, gray for EmptyGal4)
    genotype_palette = {
        "TNTxEmptyGal4": "#7f7f7f",  # Gray for control
        "TNTxIR8a": "#9467bd",  # Purple for test
    }

    # Generate permutation test plots
    print(f"\nGenerating permutation test plots (pooling BallTypes)...")
    stats_df = generate_dark_olfaction_permutation_plots(
        data=dataset,
        metrics=available_metrics,
        grouping_cols=["Light", "Genotype"],
        control_group=control_group,
        genotype_palette=genotype_palette,
        fig_width_mm=100,
        fig_height_mm=125,
        font_size_ticks=10,
        font_size_labels=14,
        font_size_legend=12,
        font_size_annotations=16,
        output_dir=str(condition_dir),
        alpha=0.05,
        n_permutations=10000,
        n_bootstrap=n_bootstrap,
        split_by_light=True,
    )

    print(f"\nGenerating genotype-wise light comparison permutation plots (EmptyGal4 and IR8a side by side)...")
    genotype_light_stats_df = generate_genotype_light_permutation_plots(
        data=dataset,
        metrics=available_metrics,
        genotypes=["TNTxEmptyGal4", "TNTxIR8a"],
        light_col="Light",
        genotype_col="Genotype",
        control_light="off",
        test_light="on",
        fig_width_mm=100,
        fig_height_mm=125,
        font_size_ticks=10,
        font_size_labels=14,
        font_size_annotations=16,
        output_dir=str(genotype_light_dir),
        alpha=0.05,
        n_permutations=10000,
        n_bootstrap=n_bootstrap,
        genotype_palette=genotype_palette,
    )

    print("\n" + "=" * 80)
    print("PERMUTATION TEST ANALYSIS COMPLETE")
    print("=" * 80)

    if not stats_df.empty:
        total_comparisons = len(stats_df)
        total_significant = stats_df["significant"].sum()
        print(f"\nTotal comparisons: {total_comparisons}")
        print(
            f"Significant results (FDR-corrected): {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
        )

        # Print metrics with significant differences
        metrics_with_sig = stats_df[stats_df["significant"]]["Metric"].unique()
        print(f"\nMetrics with significant differences ({len(metrics_with_sig)}):")
        for metric in sorted(metrics_with_sig):
            metric_sig = stats_df[(stats_df["Metric"] == metric) & (stats_df["significant"])]
            print(f"  - {metric}: {len(metric_sig)} significant comparison(s)")

    print(f"\n✅ Main plots saved to: {condition_dir}")
    print(f"✅ Genotype-wise light-comparison plots saved to: {genotype_light_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate permutation test plots for dark olfaction experiments")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing plots (default: overwrite)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 3 metrics with limited data")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples used for CI and percent-change statistics (default: 10000)",
    )

    args = parser.parse_args()

    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be a positive integer")

    main(overwrite=not args.no_overwrite, test_mode=args.test, n_bootstrap=args.n_bootstrap)
