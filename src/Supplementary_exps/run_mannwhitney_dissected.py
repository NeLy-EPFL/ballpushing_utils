#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for dissected vs non-dissected experiments.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Simple two-group comparison (dissected vs non-dissected)
- No FDR correction needed (only one comparison per metric)
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability

Usage:
    python run_mannwhitney_dissected.py [--no-overwrite] [--test]

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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import scipy.stats as stats_module
import time


def generate_dissected_mannwhitney_plots(
    data,
    metrics,
    y="Dissected",
    control_value="non-dissected",
    hue=None,
    palette="Set2",
    figsize=(15, 8),
    output_dir="mann_whitney_plots",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between dissected groups.
    No FDR correction needed since there's only one comparison per metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (dissected status). Default is "Dissected".
        control_value (str): Name of the control group. Default is "non-dissected".
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette: Color palette for the plots (str or dict). Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (15, 8).
        output_dir: Directory to save the plots. Default is "mann_whitney_plots".
        alpha (float): Significance level. Default is 0.05.

    Returns:
        pd.DataFrame: Statistics table with Mann-Whitney U test results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating Mann-Whitney jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y])

        # Get unique groups
        groups = sorted(plot_data[y].unique())

        if len(groups) != 2:
            print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
            continue

        # Determine control and test groups
        if control_value in groups:
            test_value = [g for g in groups if g != control_value][0]
        else:
            print(f"Warning: Control value '{control_value}' not found. Using {groups[0]} as control.")
            control_value = groups[0]
            test_value = groups[1]

        control_vals = plot_data[plot_data[y] == control_value][metric].dropna()
        test_vals = plot_data[plot_data[y] == test_value][metric].dropna()

        if len(control_vals) == 0 or len(test_vals) == 0:
            print(f"Warning: Insufficient data for metric {metric}")
            continue

        # Perform Mann-Whitney U test
        try:
            stat, pval = mannwhitneyu(test_vals, control_vals, alternative="two-sided")

            # Determine direction of effect
            test_median = test_vals.median()
            control_median = control_vals.median()
            direction = "increased" if test_median > control_median else "decreased"
            effect_size = test_median - control_median

            # Determine significance level
            if pval < 0.001:
                sig_level = "***"
                significant = True
            elif pval < 0.01:
                sig_level = "**"
                significant = True
            elif pval < 0.05:
                sig_level = "*"
                significant = True
            else:
                sig_level = "ns"
                significant = False
                direction = "none"

        except Exception as e:
            print(f"Error in Mann-Whitney test: {e}")
            pval = 1.0
            direction = "none"
            effect_size = 0.0
            stat = np.nan
            sig_level = "ns"
            significant = False
            test_median = np.nan
            control_median = np.nan

        # Store results
        result = {
            "Metric": metric,
            "Control": control_value,
            "Test": test_value,
            "pval": pval,
            "direction": direction,
            "effect_size": effect_size,
            "test_statistic": stat,
            "test_median": test_median,
            "control_median": control_median,
            "test_n": len(test_vals),
            "control_n": len(control_vals),
            "significant": significant,
            "sig_level": sig_level,
        }
        all_stats.append(result)

        # Sort groups for plotting: control first, then test
        sorted_groups = [control_value, test_value]

        # Update plot data with sorted categories
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_groups, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # Create the plot with fixed size for 2 groups
        fig_height = 6  # Fixed height for 2 groups
        fig_width = figsize[0]

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set up colors for groups
        colors = ["#FF6B6B", "#4ECDC4"]  # Red for control, teal for test

        # Add colored backgrounds for significant results
        y_positions = range(len(sorted_groups))
        for i, group in enumerate(sorted_groups):
            if group == control_value:
                bg_color = "lightblue"
                alpha_bg = 0.1
            elif significant and direction == "increased":
                bg_color = "lightgreen"
                alpha_bg = 0.15
            elif significant and direction == "decreased":
                bg_color = "lightcoral"
                alpha_bg = 0.15
            else:
                # Non-significant gets no background
                continue

            # Add background rectangle
            ax.axhspan(i - 0.4, i + 0.4, color=bg_color, alpha=alpha_bg, zorder=0)

        # Draw boxplots with unfilled styling
        for group in sorted_groups:
            group_data = plot_data[plot_data[y] == group]
            if group_data.empty:
                continue

            is_control = group == control_value

            if is_control:
                # Control group styling - red dashed boxes
                sns.boxplot(
                    data=group_data,
                    x=metric,
                    y=y,
                    showfliers=False,
                    width=0.5,
                    ax=ax,
                    boxprops=dict(facecolor="none", edgecolor="red", linewidth=2, linestyle="--"),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(color="red", linewidth=1.5, linestyle="--"),
                    capprops=dict(color="red", linewidth=1.5),
                )
            else:
                # Test group styling - black solid boxes
                sns.boxplot(
                    data=group_data,
                    x=metric,
                    y=y,
                    showfliers=False,
                    width=0.5,
                    ax=ax,
                    boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1),
                    capprops=dict(color="black", linewidth=1),
                )

        # Overlay stripplot for jitter with group colors
        sns.stripplot(
            data=plot_data,
            x=metric,
            y=y,
            dodge=False,
            alpha=0.7,
            jitter=True,
            palette=colors,
            size=6,
            ax=ax,
        )

        # Add significance annotation
        if sig_level != "ns":
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            yticklocs = ax.get_yticks()
            # Add annotation next to test group (index 1)
            ax.text(
                x=ax.get_xlim()[1] + 0.01 * x_range,
                y=float(yticklocs[1]),
                s=sig_level,
                color="red",
                fontsize=14,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )

        # Set xlim to accommodate annotations
        x_max = plot_data[metric].quantile(0.99)
        ax.set_xlim(left=None, right=x_max * 1.2)

        # Formatting
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel("Dissection Status", fontsize=14)

        # Add p-value to title
        title = f"Mann-Whitney U Test: {metric} by Dissection Status\n"
        title += f"p = {pval:.4f} {sig_level}"
        plt.title(title, fontsize=16)
        ax.grid(axis="x", alpha=0.3)

        # Create custom legend
        legend_elements = [
            Patch(facecolor="none", edgecolor="red", linestyle="--", linewidth=2, label=f"Control ({control_value})"),
            Patch(facecolor="none", edgecolor="black", linewidth=1, label=f"Test ({test_value})"),
            Patch(facecolor="lightgreen", alpha=0.15, label="Significantly Increased"),
            Patch(facecolor="lightblue", alpha=0.1, label="Control"),
            Patch(facecolor="lightcoral", alpha=0.15, label="Significantly Decreased"),
        ]

        ax.legend(
            legend_elements,
            [str(elem.get_label()) for elem in legend_elements],
            fontsize=10,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Use tight_layout with padding to accommodate external legend
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric}_dissected_mannwhitney.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {output_file}")
        print(f"  p-value: {pval:.4f} {sig_level}")
        print(f"  Effect: {direction} (median difference: {effect_size:.4f})")

        # Print timing for this metric
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save statistics
        stats_file = output_dir / "dissected_mannwhitney_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "dissected_mannwhitney_report.md"
        generate_text_report(stats_df, data, control_value, report_file)
        print(f"Text report saved to: {report_file}")

        # Print overall summary
        total_tests = len(stats_df)
        total_significant = stats_df["significant"].sum()

        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Significant results (p<0.05): {total_significant} ({100*total_significant/total_tests:.1f}%)")

    return stats_df


def generate_text_report(stats_df, data, control_value, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    control_value : str
        Name of the control group
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Dissected vs Non-Dissected Mann-Whitney U Test Report")
    report_lines.append("## Statistical Analysis Results")
    report_lines.append("")
    report_lines.append(f"**Control group:** {control_value}")
    report_lines.append("")

    # Analyze results
    significant_increased = []
    significant_decreased = []
    non_significant = []

    for _, row in stats_df.iterrows():
        metric = row["Metric"]
        p_val = row["pval"]
        sig_level = row["sig_level"]
        test_median = row["test_median"]
        control_median = row["control_median"]
        effect_size = row["effect_size"]
        test_n = row["test_n"]
        control_n = row["control_n"]

        # Calculate percent change
        if control_median != 0:
            percent_change = (effect_size / control_median) * 100
        else:
            percent_change = 0

        description = f'"{metric}" (test median = {test_median:.3f}, n={test_n}; control median = {control_median:.3f}, n={control_n})'

        if row["significant"]:
            if row["direction"] == "increased":
                significant_increased.append(
                    {
                        "metric": metric,
                        "description": description,
                        "effect_size": effect_size,
                        "percent_change": percent_change,
                        "p_val": p_val,
                        "sig_level": sig_level,
                    }
                )
            elif row["direction"] == "decreased":
                significant_decreased.append(
                    {
                        "metric": metric,
                        "description": description,
                        "effect_size": effect_size,
                        "percent_change": percent_change,
                        "p_val": p_val,
                        "sig_level": sig_level,
                    }
                )
        else:
            non_significant.append(
                {
                    "metric": metric,
                    "description": description,
                    "effect_size": effect_size,
                    "percent_change": percent_change,
                    "p_val": p_val,
                }
            )

    # Write results
    if significant_increased:
        # Sort by effect size (largest increase first)
        significant_increased.sort(key=lambda x: x["effect_size"], reverse=True)
        report_lines.append("## Significantly Higher in Dissected Group")
        report_lines.append("")
        for item in significant_increased:
            report_lines.append(
                f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_val']:.4f}{item['sig_level']})"
            )
        report_lines.append("")

    if significant_decreased:
        # Sort by effect size (largest decrease first)
        significant_decreased.sort(key=lambda x: x["effect_size"])
        report_lines.append("## Significantly Lower in Dissected Group")
        report_lines.append("")
        for item in significant_decreased:
            report_lines.append(
                f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_val']:.4f}{item['sig_level']})"
            )
        report_lines.append("")

    if non_significant:
        # Sort by absolute effect size for consistency
        non_significant.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
        report_lines.append("## No Significant Difference")
        report_lines.append("")
        for item in non_significant:
            report_lines.append(
                f"- {item['description']} showed {item['percent_change']:+.1f}% change (p={item['p_val']:.4f}, ns)"
            )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Add summary section
    report_lines.append("## Summary")
    report_lines.append("")

    total_comparisons = len(stats_df)
    total_significant = stats_df["significant"].sum()
    total_increased = len(significant_increased)
    total_decreased = len(significant_decreased)

    report_lines.append(f"- **Total metrics analyzed:** {total_comparisons}")
    report_lines.append(
        f"- **Significant results:** {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
    )
    report_lines.append(f"  - Significantly increased: {total_increased}")
    report_lines.append(f"  - Significantly decreased: {total_decreased}")
    report_lines.append(
        f"- **Non-significant results:** {total_comparisons - total_significant} ({100*(total_comparisons - total_significant)/total_comparisons:.1f}%)"
    )
    report_lines.append("")

    # Write to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Generated text report with {total_comparisons} metrics")


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the dissected experiments dataset

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the dissected experiments dataset
    dataset_path = "/mnt/upramdya_data/MD/Antennae_dissection/Datasets/251205_13_summary_antennae_cutting_Data/summary/pooled_summary.feather"

    print(f"Loading dissected experiments dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Drop TNT-specific metadata columns that cause pandas warnings
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
        "BallType",  # Not needed for dissection analysis
    ]
    columns_to_drop = [col for col in tnt_metadata_columns if col in dataset.columns]
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

    # Normalize Dissected values to expected labels
    if "Dissected" in dataset.columns:
        print("Normalizing 'Dissected' values to ['dissected', 'non-dissected']")
        # Create a copy to avoid SettingWithCopy warnings
        dissected_series = dataset["Dissected"].copy()

        # Standardize to strings for mapping
        def _normalize_dissection_value(v):
            if pd.isna(v):
                return np.nan
            # Convert to lowercase string for comparison
            s = str(v).strip().lower()
            if s in {"y", "yes", "true", "1"}:
                return "dissected"
            if s in {"n", "no", "false", "0"}:
                return "non-dissected"
            # Already in expected form or other labels
            if s in {"dissected", "non-dissected"}:
                return s
            # Fallback: keep original to avoid unintended relabeling
            return s

        dataset["Dissected"] = dissected_series.apply(_normalize_dissection_value)

        # Report resulting groups
        try:
            groups_after = sorted([g for g in dataset["Dissected"].unique() if pd.notna(g)])
            print(f"'Dissected' groups after normalization: {groups_after}")
        except Exception:
            pass

    # Add test mode sampling for faster debugging
    if test_mode and len(dataset) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} random rows from {len(dataset)} total rows")
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: Dataset reduced to {len(dataset)} rows")
        if "Dissected" in dataset.columns:
            print(f"🧪 TEST MODE: Dissection groups in sample: {sorted(dataset['Dissected'].unique())}")

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
    plot_patterns = [f"*{metric}*.pdf", f"*{metric}*.png", f"{metric}_*.pdf", f"{metric}_*.png"]

    for pattern in plot_patterns:
        existing_plots = list(output_dir.glob(pattern))
        if existing_plots:
            print(f"  📄 Skipping {metric}: Found existing plot(s) {[p.name for p in existing_plots]}")
            return True

    # Also check in binary_analysis subdirectory
    binary_subdir = output_dir / "binary_analysis"
    if binary_subdir.exists():
        binary_plot_patterns = [f"{metric}_binary_analysis.pdf", f"{metric}_binary_analysis.png"]
        for pattern in binary_plot_patterns:
            existing_plots = list(binary_subdir.glob(pattern))
            if existing_plots:
                print(f"  📄 Skipping {metric}: Found existing binary plot(s) {[p.name for p in existing_plots]}")
                return True

    return False


def analyze_binary_metrics(data, binary_metrics, y="Dissected", output_dir=None, overwrite=True):
    """
    Analyze binary metrics using Fisher's exact test for dissection status.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (dissection status)
    output_dir : str or Path
        Directory to save plots
    overwrite : bool
        If True, overwrite existing plots. If False, skip existing plots.

    Returns:
    --------
    pd.DataFrame
        Statistics results for binary metrics
    """
    if not binary_metrics:
        print("No binary metrics to analyze")
        return pd.DataFrame()

    print(f"\n{'='*50}")
    print(f"BINARY METRICS ANALYSIS")
    print(f"{'='*50}")

    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique groups
    groups = sorted(data[y].unique())
    if len(groups) != 2:
        print(f"Warning: Expected 2 groups for dissection analysis, found {len(groups)}: {groups}")
        return pd.DataFrame()

    control_value = groups[0]  # Assume first group is control (alphabetically)
    test_value = groups[1]

    print(f"Using control group: {control_value}")
    print(f"Using test group: {test_value}")

    for metric in binary_metrics:
        print(f"\nAnalyzing binary metric: {metric}")

        # Check if we should skip this metric
        if output_dir and should_skip_metric(metric, output_dir, overwrite):
            continue

        # Create contingency table
        contingency = pd.crosstab(data[y], data[metric], margins=True)
        print(f"Contingency table for {metric}:")
        print(contingency)

        # Get data for this comparison
        test_data = data[data[y] == test_value][metric].dropna()
        control_data_subset = data[data[y] == control_value][metric].dropna()

        if len(test_data) == 0 or len(control_data_subset) == 0:
            print(f"  No data for {metric}")
            continue

        # Create 2x2 contingency table for Fisher's exact test
        test_success = test_data.sum()
        test_total = len(test_data)
        control_success = control_data_subset.sum()
        control_total = len(control_data_subset)

        # Fisher's exact test
        try:
            table = [[test_success, test_total - test_success], [control_success, control_total - control_success]]
            fisher_result = fisher_exact(table)
            odds_ratio = fisher_result[0]  # type: ignore
            p_value = fisher_result[1]  # type: ignore

            # Effect size (difference in proportions)
            test_prop = test_success / test_total if test_total > 0 else 0
            control_prop = control_success / control_total if control_total > 0 else 0
            effect_size = test_prop - control_prop

            # Determine significance level
            if p_value < 0.001:  # type: ignore
                sig_level = "***"
                significant = True
            elif p_value < 0.01:  # type: ignore
                sig_level = "**"
                significant = True
            elif p_value < 0.05:  # type: ignore
                sig_level = "*"
                significant = True
            else:
                sig_level = "ns"
                significant = False

            results.append(
                {
                    "metric": metric,
                    "control": control_value,
                    "test": test_value,
                    "test_success": test_success,
                    "test_total": test_total,
                    "test_proportion": test_prop,
                    "control_success": control_success,
                    "control_total": control_total,
                    "control_proportion": control_prop,
                    "effect_size": effect_size,
                    "odds_ratio": odds_ratio,
                    "p_value": p_value,
                    "significant": significant,
                    "sig_level": sig_level,
                    "test_type": "Fisher exact",
                }
            )

            print(f"  {test_value} vs {control_value}: OR={odds_ratio:.3f}, p={p_value:.4f} {sig_level}")

        except Exception as e:
            print(f"  Error testing {metric}: {e}")

        # Create visualization
        if output_dir:
            create_binary_metric_plot(data, metric, y, output_dir, control_value)

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    return results_df


def create_binary_metric_plot(data, metric, y, output_dir, control_value=None):
    """Create a visualization for binary metrics showing proportions"""

    # Calculate proportions
    prop_data = data.groupby(y)[metric].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_success"] = prop_data["sum"]
    prop_data["n_total"] = prop_data["count"]

    # Sort by group name (control first if specified)
    if control_value:
        prop_data["sort_key"] = prop_data[y].apply(lambda x: 0 if x == control_value else 1)
        prop_data = prop_data.sort_values("sort_key").reset_index(drop=True)
        prop_data = prop_data.drop(columns=["sort_key"])

    # Use simple colors for 2 groups
    colors = ["#FF6B6B", "#4ECDC4"]  # Red for control, teal for test

    # Calculate confidence intervals (Wilson score interval)
    def wilson_ci(successes, total, confidence=0.95):
        if total == 0:
            return 0, 0
        z = stats.norm.ppf((1 + confidence) / 2)
        p = successes / total
        denominator = 1 + z**2 / total
        centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        return max(0, lower_bound), min(1, upper_bound)

    ci_data = []
    for _, row in prop_data.iterrows():
        lower, upper = wilson_ci(row["n_success"], row["n_total"])
        ci_data.append({"lower": lower, "upper": upper})

    ci_df = pd.DataFrame(ci_data)
    prop_data = pd.concat([prop_data, ci_df], axis=1)

    # Create the plot - fixed size for 2 groups
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Proportion with confidence intervals (horizontal bars)
    y_positions = range(len(prop_data))
    bars = ax1.barh(y_positions, prop_data["proportion"], color=colors, alpha=0.7)

    # Add error bars
    yerr_lower = np.maximum(0, prop_data["proportion"] - prop_data["lower"])
    yerr_upper = np.maximum(0, prop_data["upper"] - prop_data["proportion"])
    xerr = [yerr_lower, yerr_upper]
    ax1.errorbar(prop_data["proportion"], y_positions, xerr=xerr, fmt="none", color="black", capsize=5)

    # Add sample sizes on bars
    for i, (bar, row) in enumerate(zip(bars, prop_data.itertuples())):
        width = bar.get_width()
        ax1.text(
            width + 0.01, bar.get_y() + bar.get_height() / 2.0, f"n={row.n_total}", ha="left", va="center", fontsize=10
        )

    ax1.set_ylabel("Dissection Status")
    ax1.set_xlabel(f"Proportion of {metric}")
    ax1.set_title(f"{metric} - Proportions with 95% CI")
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(prop_data[y])
    ax1.set_xlim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stacked bar chart showing counts (horizontal bars)
    ax2.barh(y_positions, prop_data["n_success"], label=f"{metric}=1", color="lightgreen", alpha=0.8)
    ax2.barh(
        y_positions,
        prop_data["n_total"] - prop_data["n_success"],
        left=prop_data["n_success"],
        label=f"{metric}=0",
        color="lightcoral",
        alpha=0.8,
    )

    ax2.set_ylabel("Dissection Status")
    ax2.set_xlabel("Count")
    ax2.set_title(f"{metric} - Raw Counts")
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(prop_data[y])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"{metric}_binary_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Binary plot saved: {output_file}")


def main(overwrite=True, test_mode=False):
    """Main function to run Mann-Whitney plots for dissected experiments.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample 200 rows for fast testing.
    """
    print(f"Starting Mann-Whitney U test analysis for dissected experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directory
    base_output_dir = Path("/mnt/upramdya_data/MD/Antennae_dissection/Plots/Summary_metrics/Dissected_Mannwhitney")

    # Ensure the output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # Load the dataset
    print("Loading dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Check for Dissected column
    if "Dissected" not in dataset.columns:
        print("❌ Dissected column not found in dataset!")
        print("Available columns that might contain dissection info:")
        potential_cols = [
            col for col in dataset.columns if any(word in col.lower() for word in ["dissect", "cut", "antenna"])
        ]
        print(potential_cols)
        return

    print(f"✅ Dissected column found")
    print(f"Dissection groups in dataset: {sorted(dataset['Dissected'].unique())}")

    # Define core metrics to analyze
    print(f"🎯 Using predefined core metrics for analysis...")

    # Core base metrics
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
        "major_event",
        "major_event_time",
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_velocity",
        "auc",
        "overall_interaction_rate",
    ]

    # Additional pattern-based metrics
    additional_patterns = [
        "velocity",
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
        "median_",
        "mean_",
    ]

    excluded_patterns = [
        "binned_",
        "r2",
        "slope",
        "_bin_",
        "logistic_",
    ]

    # Find metrics that exist in the dataset
    print(f"📊 Checking which metrics exist in the dataset...")
    available_metrics = []

    # Check base metrics
    for metric in base_metrics:
        if metric in dataset.columns:
            should_exclude = any(pattern in metric.lower() for pattern in excluded_patterns)
            if should_exclude:
                print(f"  ❌ {metric} (excluded due to pattern)")
                continue
            available_metrics.append(metric)
            print(f"  ✓ {metric}")
        else:
            print(f"  ❌ {metric} (not found)")

    # Check pattern-based metrics
    print(f"📊 Searching for pattern-based metrics...")
    for pattern in additional_patterns:
        pattern_cols = [col for col in dataset.columns if pattern in col.lower()]
        for col in pattern_cols:
            if col not in available_metrics and col != "Dissected":
                should_exclude = any(excl_pattern in col.lower() for excl_pattern in excluded_patterns)
                if should_exclude:
                    print(f"  ❌ {col} (excluded due to pattern)")
                    continue
                available_metrics.append(col)
                print(f"  ✓ {col} (matches pattern '{pattern}')")

    print(f"📊 Found {len(available_metrics)} metrics after filtering")

    # Categorize metrics
    def categorize_metrics(col):
        """Categorize metrics for appropriate analysis"""
        if col not in dataset.columns:
            return None

        if not pd.api.types.is_numeric_dtype(dataset[col]):
            return "non_numeric"

        non_nan_values = dataset[col].dropna()

        if len(non_nan_values) < 3:
            return "insufficient_data"

        unique_vals = non_nan_values.unique()

        # Check for binary metrics
        if len(unique_vals) == 2:
            vals_set = set(unique_vals)
            if vals_set == {0, 1} or vals_set == {0.0, 1.0} or vals_set == {0, 1.0} or vals_set == {0.0, 1}:
                return "binary"

        if len(unique_vals) < 3:
            return "categorical"

        return "continuous"

    print(f"\n📋 Categorizing {len(available_metrics)} available metrics...")
    continuous_metrics = []
    binary_metrics = []
    excluded_metrics = []

    for col in available_metrics:
        category = categorize_metrics(col)

        if category == "continuous":
            continuous_metrics.append(col)
        elif category == "binary":
            binary_metrics.append(col)
        elif category == "insufficient_data":
            non_nan_count = dataset[col].count()
            total_count = len(dataset)
            print(f"  ⚠️  {col}: Only {non_nan_count}/{total_count} non-NaN values, including anyway")
            non_nan_values = dataset[col].dropna().unique()
            if len(non_nan_values) >= 2 and set(non_nan_values) == {0, 1}:
                binary_metrics.append(col)
            else:
                continuous_metrics.append(col)
        else:
            excluded_metrics.append((col, category))

    print(f"\n📊 METRICS ANALYSIS SUMMARY:")
    print(f"=" * 60)
    print(f"Found {len(continuous_metrics)} continuous metrics for Mann-Whitney analysis:")
    for i, metric in enumerate(continuous_metrics, 1):
        print(f"  {i:2d}. {metric}")

    print(f"\nFound {len(binary_metrics)} binary metrics for Fisher's exact test:")
    for i, metric in enumerate(binary_metrics, 1):
        print(f"  {i:2d}. {metric}")

    if excluded_metrics:
        print(f"\nExcluded {len(excluded_metrics)} metrics:")
        for metric, reason in excluded_metrics:
            if metric in dataset.columns:
                unique_count = len(dataset[metric].dropna().unique())
                dtype = dataset[metric].dtype
                print(f"  - {metric} ({reason}, {dtype}, {unique_count} unique values)")

    # Test mode filtering
    if test_mode:
        original_count = len(continuous_metrics)
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2] if binary_metrics else []
        print(f"🧪 TEST MODE: Reduced from {original_count} to {len(continuous_metrics)} continuous metrics")
        print(f"🧪 TEST MODE: Limited to {len(binary_metrics)} binary metrics")
        print(f"🧪 TEST MODE: Processing metrics: {continuous_metrics}")

    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique dissection groups: {len(dataset['Dissected'].unique())}")

    # Validate required columns
    required_cols = ["Dissected"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Process continuous metrics
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests) ---")
        continuous_data = dataset[["Dissected"] + continuous_metrics].copy()

        print(f"Generating Mann-Whitney plots for dissected experiments...")
        print(f"Continuous metrics dataset shape: {continuous_data.shape}")

        # Filter out metrics that should be skipped
        if not overwrite:
            print(f"📄 Checking for existing plots...")
            metrics_to_process = []
            for metric in continuous_metrics:
                if not should_skip_metric(metric, base_output_dir, overwrite):
                    metrics_to_process.append(metric)

            if len(metrics_to_process) < len(continuous_metrics):
                skipped_count = len(continuous_metrics) - len(metrics_to_process)
                print(f"📄 Skipped {skipped_count} metrics with existing plots")
                print(f"📊 Processing {len(metrics_to_process)} metrics")

            if not metrics_to_process:
                print(f"📄 All continuous metrics already have plots, skipping...")
            else:
                continuous_data = continuous_data[["Dissected"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            print(f"🚀 Starting Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics...")

            import time

            start_time = time.time()

            # Determine control and test groups
            groups = sorted(continuous_data["Dissected"].unique())
            if len(groups) != 2:
                print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
                return

            # Assume alphabetically first is control (e.g., "non-dissected" before "dissected")
            control_value = groups[0]
            test_value = groups[1]

            print(f"📊 Control group: {control_value}")
            print(f"📊 Test group: {test_value}")

            stats_df = generate_dissected_mannwhitney_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="Dissected",
                control_value=control_value,
                hue="Dissected",
                palette="Set2",
                output_dir=str(base_output_dir),
                alpha=0.05,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"✅ Mann-Whitney analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Process binary metrics
    if binary_metrics:
        print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
        binary_data = dataset[["Dissected"] + binary_metrics].copy()

        print(f"Analyzing binary metrics for dissected experiments...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        binary_output_dir = base_output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            y="Dissected",
            output_dir=binary_output_dir,
            overwrite=overwrite,
        )

    print(f"\n{'='*60}")
    print("✅ Dissected experiments analysis complete! Check output directory for results.")
    print(f"📁 Output directory: {base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test plots for dissected vs non-dissected experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_dissected.py                    # Overwrite existing plots
  python run_mannwhitney_dissected.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_dissected.py --test             # Test mode with 3 metrics
        """,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip metrics that already have existing plots (default: overwrite existing plots)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process first 3 metrics for debugging (default: process all metrics)",
    )

    args = parser.parse_args()

    # Run main function with overwrite parameter (invert --no-overwrite)
    main(overwrite=not args.no_overwrite, test_mode=args.test)
