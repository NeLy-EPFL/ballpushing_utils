#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for MagnetBlock experiments.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Comparison between Magnet and non-Magnet groups
- FDR correction across all comparisons for each metric
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability

Usage:
    python run_mannwhitney_magnetblock.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: If specified, skip metrics that already have plots.
    --test: If specified, only process first 3 metrics for debugging.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent))  # Also add current directory
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))  # Add Plotting directory

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import scipy.stats as stats_module
from statsmodels.stats.multitest import multipletests
import time


def generate_magnet_mannwhitney_plots(
    data,
    metrics,
    y="Magnet",
    control_group="y",
    hue=None,
    palette="Set2",
    figsize=(12, 8),
    output_dir="mann_whitney_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between Magnet and non-Magnet groups.
    Applies FDR correction across all comparisons for each metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (Magnet). Default is "Magnet".
        control_group (str): Name of the control group. Default is "non-Magnet".
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette: Color palette for the plots (str or dict). Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (12, 8).
        output_dir: Directory to save the plots. Default is "mann_whitney_plots".
        fdr_method (str): Method for FDR correction. Default is "fdr_bh" (Benjamini-Hochberg).
        alpha (float): Significance level after FDR correction. Default is 0.05.

    Returns:
        pd.DataFrame: Statistics table with Mann-Whitney U test results and FDR correction.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating Mann-Whitney jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y])

        # Get all groups
        groups = sorted(plot_data[y].unique())

        if len(groups) != 2:
            print(f"Warning: Expected 2 groups, found {len(groups)} for metric {metric}. Skipping.")
            continue

        # Check if control exists
        if control_group not in plot_data[y].unique():
            print(f"Warning: Control group '{control_group}' not found for metric {metric}")
            continue

        # Get test group (the non-control group)
        test_group = [g for g in groups if g != control_group][0]

        # Perform Mann-Whitney U test
        control_data = plot_data[plot_data[y] == control_group][metric].dropna()
        test_data = plot_data[plot_data[y] == test_group][metric].dropna()

        if len(control_data) < 3 or len(test_data) < 3:
            print(f"  ⚠️  Insufficient data for {metric} (control: {len(control_data)}, test: {len(test_data)})")
            continue

        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(test_data, control_data, alternative="two-sided")

        # Calculate medians
        control_median = control_data.median()
        test_median = test_data.median()
        median_diff = test_median - control_median

        # Calculate effect size (rank-biserial correlation)
        n1 = len(control_data)
        n2 = len(test_data)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # Store results
        stat_result = {
            "Metric": metric,
            "Control": control_group,
            "Test": test_group,
            "Control_n": n1,
            "Test_n": n2,
            "Control_median": control_median,
            "Test_median": test_median,
            "Median_diff": median_diff,
            "U_statistic": u_stat,
            "p_value": p_value,
            "effect_size": effect_size,
        }
        all_stats.append(stat_result)

        # Create plot with smaller figure size for publication-quality layout
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 3.5))

        # Map to intuitive labels: "n" -> "Control", "y" -> "Magnet block"
        if control_group == "n" and test_group == "y":
            control_label = "Control"
            test_label = "Magnet block"
        else:
            control_label = control_group
            test_label = test_group

        # Create color palette (matching F1 style)
        colors = {control_group: "#ff7f0e", test_group: "#1f77b4"}  # Orange for control, blue for test

        # Determine significance level
        if p_value < 0.001:
            sig_symbol = "***"
        elif p_value < 0.01:
            sig_symbol = "**"
        elif p_value < 0.05:
            sig_symbol = "*"
        else:
            sig_symbol = "ns"

        # Create thinner boxplot with transparency (matching F1 style)
        box_props = dict(linewidth=1.5, edgecolor="black")  # Black outline
        whisker_props = dict(linewidth=1.5, color="black")
        cap_props = dict(linewidth=1.5, color="black")
        median_props = dict(linewidth=2, color="black")

        positions = [0, 1]
        box_data = [control_data, test_data]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,  # Thinner boxes (50% of spacing)
            patch_artist=True,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            showfliers=False,  # Don't show outliers since we're plotting all points
        )

        # Color the boxes consistently with black outlines
        bp["boxes"][0].set_facecolor(colors[control_group])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][0].set_linewidth(1.5)
        bp["boxes"][1].set_facecolor(colors[test_group])
        bp["boxes"][1].set_alpha(0.7)
        bp["boxes"][1].set_edgecolor("black")
        bp["boxes"][1].set_linewidth(1.5)

        # Add individual data points as larger filled circles matching box colors
        x_jitter = 0.08  # Amount of horizontal jitter
        for i, (group, color) in enumerate([(control_group, colors[control_group]), (test_group, colors[test_group])]):
            y_data = box_data[i]
            # Add random jitter to x positions
            x_positions = np.random.normal(positions[i], x_jitter, size=len(y_data))
            ax.scatter(
                x_positions,
                y_data,
                s=25,  # Smaller point size for cleaner look
                c=color,
                alpha=0.6,  # Translucent
                edgecolors="none",  # No edge
                zorder=3,
            )  # Draw on top

        # Calculate y-axis range for annotations
        y_max = max(control_data.max(), test_data.max())
        y_min = min(control_data.min(), test_data.min())
        y_range = y_max - y_min

        # Draw significance bar and asterisks (only if significant)
        if sig_symbol != "ns":
            bar_height = y_max + 0.08 * y_range
            ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
            ax.text(
                0.5, bar_height + 0.02 * y_range, sig_symbol, ha="center", va="bottom", fontsize=12, fontname="Arial"
            )

        # Add p-value text in top-left corner of plot
        if p_value < 0.001:
            p_text = f"p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.3f}"

        ax.text(
            0.02,
            0.98,
            p_text,
            transform=ax.transAxes,
            fontsize=10,
            fontname="Arial",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
        )

        # Clean formatting - no grid, white background, with tick marks
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(False)

        # Add tick marks pointing outward
        ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)

        # Set x-axis labels with intuitive labels
        ax.set_xticks(positions)
        ax.set_xticklabels([control_label, test_label], fontsize=10, fontname="Arial")

        # Set y-axis label
        ax.set_ylabel(metric, fontsize=11, fontname="Arial")

        # Format y-axis tick labels
        ax.tick_params(axis="y", labelsize=9)
        for label in ax.get_yticklabels():
            label.set_fontname("Arial")

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        plt.tight_layout()

        # Save plot
        safe_metric_name = metric.replace("/", "_").replace(" ", "_")
        plot_path = output_dir / f"{safe_metric_name}_magnetblock.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        png_path = output_dir / f"{safe_metric_name}_magnetblock.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

        metric_elapsed = time.time() - metric_start_time
        print(f"  ✅ Saved: {plot_path} (took {metric_elapsed:.1f}s)")

    # Convert to DataFrame
    if not all_stats:
        print("⚠️  No statistics computed - no valid data found")
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_stats)

    # Apply FDR correction
    if len(stats_df) > 0:
        _, p_corrected, _, _ = multipletests(stats_df["p_value"], alpha=alpha, method=fdr_method)
        stats_df["p_corrected"] = p_corrected
        stats_df["significant"] = p_corrected < alpha

        # Sort by significance and median difference
        stats_df = stats_df.sort_values(["significant", "Median_diff"], ascending=[False, False])

    # Save statistics
    stats_path = output_dir / "magnetblock_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics saved to: {stats_path}")

    # Generate text report
    report_file = output_dir / "magnetblock_statistics_report.txt"
    generate_text_report(stats_df, data, report_file)
    print(f"✅ Text report saved to: {report_file}")

    return stats_df


def generate_text_report(stats_df, data, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        Statistics dataframe with test results
    data : pd.DataFrame
        Original dataset
    report_file : Path
        Path to save the report
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("MAGNETBLOCK MANN-WHITNEY U TEST RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 80)

    total_comparisons = len(stats_df)
    total_significant = stats_df["significant"].sum()
    total_metrics = stats_df["Metric"].nunique()

    report_lines.append(f"- **Total metrics analyzed:** {total_metrics}")
    report_lines.append(f"- **Total comparisons:** {total_comparisons}")
    report_lines.append(
        f"- **Significant results:** {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
    )
    report_lines.append("")

    # Significant results
    if total_significant > 0:
        report_lines.append("SIGNIFICANT RESULTS (p_corrected < 0.05)")
        report_lines.append("-" * 80)

        sig_df = stats_df[stats_df["significant"]].copy()

        for _, row in sig_df.iterrows():
            report_lines.append(f"\n**{row['Metric']}**")
            report_lines.append(
                f"  Control ({row['Control']}): median = {row['Control_median']:.3f}, n = {row['Control_n']}"
            )
            report_lines.append(f"  Test ({row['Test']}): median = {row['Test_median']:.3f}, n = {row['Test_n']}")
            report_lines.append(f"  Median difference: {row['Median_diff']:.3f}")
            report_lines.append(f"  U statistic: {row['U_statistic']:.1f}")
            report_lines.append(f"  P-value: {row['p_value']:.6f}")
            report_lines.append(f"  Corrected p-value: {row['p_corrected']:.6f}")
            report_lines.append(f"  Effect size: {row['effect_size']:.3f}")

            if row["p_corrected"] < 0.001:
                sig_level = "***"
            elif row["p_corrected"] < 0.01:
                sig_level = "**"
            elif row["p_corrected"] < 0.05:
                sig_level = "*"
            else:
                sig_level = ""

            direction = "increased" if row["Median_diff"] > 0 else "decreased"
            report_lines.append(f"  → {row['Test']} {direction} compared to {row['Control']} {sig_level}")

        report_lines.append("")

    # Non-significant results
    non_sig_count = total_comparisons - total_significant
    if non_sig_count > 0:
        report_lines.append(f"\nNON-SIGNIFICANT RESULTS: {non_sig_count} metrics")
        report_lines.append("-" * 80)

        non_sig_df = stats_df[~stats_df["significant"]].copy()
        for _, row in non_sig_df.iterrows():
            report_lines.append(f"  {row['Metric']}: p_corrected = {row['p_corrected']:.4f}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the MagnetBlock dataset

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the MagnetBlock dataset
    dataset_path = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"

    print(f"Loading MagnetBlock dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Check for Magnet column
    if "Magnet" not in dataset.columns:
        raise ValueError(f"'Magnet' column not found in dataset. Available columns: {list(dataset.columns)}")

    print(f"Magnet groups: {sorted(dataset['Magnet'].unique())}")

    # Drop metadata columns that might cause warnings
    metadata_columns = [
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Driver",
        "Experiment",
        "Date",
        "Arena",
        "BallType",
        "Dissected",
        "Genotype",
    ]
    columns_to_drop = [col for col in metadata_columns if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"Dropped metadata columns: {columns_to_drop}")
        print(f"Dataset shape after dropping metadata: {dataset.shape}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Fill NA values with appropriate defaults (batch operation for performance)
    fillna_dict = {}
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

    for metric, default_value in new_metrics_defaults.items():
        if metric in dataset.columns:
            nan_count = dataset[metric].isnull().sum()
            if nan_count > 0:
                fillna_dict[metric] = default_value

    if fillna_dict:
        dataset.fillna(fillna_dict, inplace=True)
        print(f"Batch filled NaN values for {len(fillna_dict)} metrics")

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    print(f"Dataset cleaning completed successfully")

    # Add test mode sampling for faster debugging
    if test_mode and len(dataset) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} random rows from {len(dataset)} total rows")
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: Dataset reduced to {len(dataset)} rows")
        if "Magnet" in dataset.columns:
            print(f"🧪 TEST MODE: Magnet groups in sample: {sorted(dataset['Magnet'].unique())}")

    return dataset


def should_skip_metric(metric, output_dir, overwrite=True):
    """
    Check if a metric should be skipped based on existing plots and overwrite setting.

    Parameters:
    -----------
    metric : str
        Name of the metric
    output_dir : Path
        Directory where plots would be saved
    overwrite : bool
        If True, always generate plots. If False, skip if plots already exist.

    Returns:
    --------
    bool
        True if metric should be skipped, False if it should be processed
    """
    if overwrite:
        return False

    output_dir = Path(output_dir)

    # Check for existing plot files for this metric
    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    plot_patterns = [f"{safe_metric_name}_magnetblock.pdf", f"{safe_metric_name}_magnetblock.png"]

    for pattern in plot_patterns:
        existing_plot = output_dir / pattern
        if existing_plot.exists():
            print(f"  📄 Skipping {metric}: Found existing plot {existing_plot.name}")
            return True

    return False


def analyze_binary_metrics(
    data, binary_metrics, y="Magnet", output_dir=None, overwrite=True, control_group="non-Magnet"
):
    """
    Analyze binary metrics using Fisher's exact test or Chi-square test.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (Magnet)
    output_dir : Path or str
        Directory to save analysis results
    overwrite : bool
        Whether to overwrite existing files
    control_group : str
        Name of the control group

    Returns:
    --------
    pd.DataFrame
        Statistics table with test results
    """
    if output_dir is None:
        output_dir = Path("mann_whitney_plots/binary_analysis")
    else:
        output_dir = Path(output_dir) / "binary_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    print(f"\n{'='*60}")
    print("ANALYZING BINARY METRICS")
    print(f"{'='*60}")

    for metric_idx, metric in enumerate(binary_metrics):
        print(f"Analyzing binary metric {metric_idx+1}/{len(binary_metrics)}: {metric}")

        if not overwrite and should_skip_metric(metric, output_dir, overwrite):
            continue

        plot_data = data.dropna(subset=[metric, y])

        # Get groups
        groups = sorted(plot_data[y].unique())

        if len(groups) != 2:
            print(f"  ⚠️  Expected 2 groups, found {len(groups)} for {metric}. Skipping.")
            continue

        if control_group not in groups:
            print(f"  ⚠️  Control group '{control_group}' not found for {metric}. Skipping.")
            continue

        test_group = [g for g in groups if g != control_group][0]

        # Create contingency table
        contingency = pd.crosstab(plot_data[y], plot_data[metric])

        if contingency.shape != (2, 2):
            print(f"  ⚠️  Invalid contingency table shape for {metric}: {contingency.shape}. Skipping.")
            continue

        # Perform Fisher's exact test (better for small sample sizes)
        try:
            odds_ratio, p_value = fisher_exact(contingency)
        except Exception as e:
            print(f"  ⚠️  Error in Fisher's exact test for {metric}: {e}. Trying Chi-square...")
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                odds_ratio = np.nan
            except Exception as e2:
                print(f"  ⚠️  Error in Chi-square test for {metric}: {e2}. Skipping.")
                continue

        # Calculate proportions
        control_data = plot_data[plot_data[y] == control_group]
        test_data = plot_data[plot_data[y] == test_group]

        control_n = len(control_data)
        test_n = len(test_data)

        control_success = control_data[metric].sum()
        test_success = test_data[metric].sum()

        control_prop = control_success / control_n if control_n > 0 else 0
        test_prop = test_success / test_n if test_n > 0 else 0

        prop_diff = test_prop - control_prop

        # Store results
        stat_result = {
            "Metric": metric,
            "Control": control_group,
            "Test": test_group,
            "Control_n": control_n,
            "Test_n": test_n,
            "Control_successes": control_success,
            "Test_successes": test_success,
            "Control_proportion": control_prop,
            "Test_proportion": test_prop,
            "Proportion_diff": prop_diff,
            "Odds_ratio": odds_ratio,
            "p_value": p_value,
        }
        all_stats.append(stat_result)

        # Create plot
        create_binary_metric_plot(data, metric, y, output_dir, control_group)

    if not all_stats:
        print("⚠️  No binary statistics computed")
        return pd.DataFrame()

    # Convert to DataFrame
    stats_df = pd.DataFrame(all_stats)

    # Apply FDR correction
    if len(stats_df) > 0:
        _, p_corrected, _, _ = multipletests(stats_df["p_value"], alpha=0.05, method="fdr_bh")
        stats_df["p_corrected"] = p_corrected
        stats_df["significant"] = p_corrected < 0.05

    # Save statistics
    stats_path = output_dir / "binary_metrics_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Binary statistics saved to: {stats_path}")

    return stats_df


def create_binary_metric_plot(data, metric, y, output_dir, control_group=None):
    """
    Create a bar plot for a binary metric with confidence intervals.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    metric : str
        Binary metric name
    y : str
        Grouping column (Magnet)
    output_dir : Path
        Output directory
    control_group : str
        Name of control group
    """
    plot_data = data.dropna(subset=[metric, y])

    def wilson_ci(successes, total, confidence=0.95):
        """Calculate Wilson score confidence interval"""
        if total == 0:
            return 0, 0
        p = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return max(0, center - margin), min(1, center + margin)

    # Calculate proportions and CIs
    groups = sorted(plot_data[y].unique())
    proportions = []
    cis = []
    labels = []
    n_totals = []
    n_successes = []

    for group in groups:
        group_data = plot_data[plot_data[y] == group]
        n_total = len(group_data)
        n_success = group_data[metric].sum()
        prop = n_success / n_total if n_total > 0 else 0

        ci_low, ci_high = wilson_ci(n_success, n_total)

        proportions.append(prop)
        cis.append((prop - ci_low, ci_high - prop))
        labels.append(group)
        n_totals.append(n_total)
        n_successes.append(n_success)

    # Perform Fisher's exact test to get p-value
    if len(groups) == 2:
        contingency = pd.crosstab(plot_data[y], plot_data[metric])
        try:
            odds_ratio, p_value = fisher_exact(contingency)
        except Exception:
            p_value = 1.0

        # Determine significance
        if p_value < 0.001:
            sig_symbol = "***"
        elif p_value < 0.01:
            sig_symbol = "**"
        elif p_value < 0.05:
            sig_symbol = "*"
        else:
            sig_symbol = "ns"
    else:
        p_value = 1.0
        sig_symbol = "ns"

    # Create plot with smaller figure size matching F1 style
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 3.5))

    # Map to intuitive labels: "n" -> "Control", "y" -> "Magnet block"
    display_labels = []
    for group in groups:
        if group == "n":
            display_labels.append("Control")
        elif group == "y":
            display_labels.append("Magnet block")
        else:
            display_labels.append(group)

    # Use F1-style colors
    colors = {control_group: "#ff7f0e"}  # Orange for control
    if len(groups) == 2:
        test_group = [g for g in groups if g != control_group][0]
        colors[test_group] = "#1f77b4"  # Blue for test

    bar_colors = [colors.get(g, "#95a5a6") for g in groups]

    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, proportions, width=0.5, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add error bars
    ax.errorbar(
        x_pos, proportions, yerr=np.array(cis).T, fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5
    )

    # Add sample sizes as text on bars
    for i, (prop, bar, n_total, n_success) in enumerate(zip(proportions, bars, n_totals, n_successes)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            prop / 2,
            f"{n_success}/{n_total}",
            ha="center",
            va="center",
            fontsize=9,
            fontname="Arial",
            fontweight="bold",
            color="white" if prop > 0.3 else "black",
        )

    # Add significance annotation if there are exactly 2 groups and result is significant
    if len(groups) == 2 and sig_symbol != "ns":
        y_max = max(proportions) + max([ci[1] for ci in cis])
        bar_height = y_max + 0.05
        ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
        ax.text(0.5, bar_height + 0.02, sig_symbol, ha="center", va="bottom", fontsize=12, fontname="Arial")

    # Add p-value text in top-left corner
    if p_value < 0.001:
        p_text = f"p < 0.001"
    elif p_value < 0.01:
        p_text = f"p = {p_value:.3f}"
    else:
        p_text = f"p = {p_value:.3f}"

    ax.text(
        0.02,
        0.98,
        p_text,
        transform=ax.transAxes,
        fontsize=10,
        fontname="Arial",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
    )

    # Clean formatting matching F1 style
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    # Add tick marks pointing outward
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontsize=10, fontname="Arial")
    ax.set_ylabel(f"Proportion\n{metric}", fontsize=11, fontname="Arial")
    ax.set_ylim(0, 1.0 if sig_symbol == "ns" else 1.15)

    # Format y-axis tick labels
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_yticklabels():
        label.set_fontname("Arial")

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()

    # Save plot
    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    plot_path = output_dir / f"{safe_metric_name}_binary_analysis.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    png_path = output_dir / f"{safe_metric_name}_binary_analysis.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved: {plot_path}")


def main(overwrite=True, test_mode=False):
    """Main function to run Mann-Whitney analysis for MagnetBlock experiments"""

    print(f"\n{'='*80}")
    print("MAGNETBLOCK MANN-WHITNEY U TEST ANALYSIS")
    print(f"{'='*80}\n")

    # Load dataset
    dataset = load_and_clean_dataset(test_mode=test_mode)

    # Define output directory
    output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/mann_whitney")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")

    # Get all numeric columns (excluding Magnet and fly identifier)
    exclude_cols = ["Magnet", "fly", "Corridor"]
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [col for col in numeric_cols if col not in exclude_cols]

    print(f"\nFound {len(metrics)} numeric metrics to analyze")

    # In test mode, only process first 3 metrics
    if test_mode:
        metrics = metrics[:3]
        print(f"🧪 TEST MODE: Analyzing only first {len(metrics)} metrics")

    # Filter out metrics that should be skipped
    if not overwrite:
        metrics_to_process = [m for m in metrics if not should_skip_metric(m, output_dir, overwrite)]
        print(f"📄 Skipping {len(metrics) - len(metrics_to_process)} metrics with existing plots")
        metrics = metrics_to_process

    if len(metrics) == 0:
        print("⚠️  No metrics to process!")
        return

    # Generate Mann-Whitney plots
    print(f"\n{'='*60}")
    print(f"GENERATING MANN-WHITNEY PLOTS FOR {len(metrics)} METRICS")
    print(f"{'='*60}\n")

    start_time = time.time()

    stats_df = generate_magnet_mannwhitney_plots(
        data=dataset,
        metrics=metrics,
        y="Magnet",
        control_group="n",
        output_dir=output_dir,
        figsize=(12, 8),
        alpha=0.05,
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Analysis complete! Generated {len(metrics)} plots in {elapsed:.1f} seconds")

    # Analyze binary metrics
    binary_metrics = [
        col
        for col in dataset.columns
        if dataset[col].dtype in [bool, int] and dataset[col].nunique() == 2 and col not in exclude_cols
    ]

    if binary_metrics:
        print(f"\nFound {len(binary_metrics)} binary metrics")
        if test_mode:
            binary_metrics = binary_metrics[:2]
            print(f"🧪 TEST MODE: Analyzing only first {len(binary_metrics)} binary metrics")

        binary_stats = analyze_binary_metrics(
            data=dataset,
            binary_metrics=binary_metrics,
            y="Magnet",
            output_dir=output_dir,
            overwrite=overwrite,
            control_group="non-Magnet",
        )

    print(f"\n{'='*80}")
    print("✅ MAGNETBLOCK MANN-WHITNEY ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test plots for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip metrics that already have existing plots (default: overwrite existing plots)",
    )

    parser.add_argument("--test", action="store_true", help="Test mode: only process first 3 metrics for debugging")

    args = parser.parse_args()

    main(overwrite=not args.no_overwrite, test_mode=args.test)
