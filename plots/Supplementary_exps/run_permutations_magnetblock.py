#!/usr/bin/env python3
"""
Script to generate permutation-based jitterboxplots for MagnetBlock experiments.


    parser.add_argument(
        "--metrics-file",
        type=str,
        default="src/PCA/metrics_lists/final_metrics_for_pca_alt.txt",
        help=(
            "Path to a file listing metrics (one per line) to analyze (reduces computation)."
            " Defaults to src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"
        ),
    )
This script generates comprehensive permutation-test visualizations with:
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
from ballpushing_utils import read_feather

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def mm_to_inches(mm_value):
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
        return data / 60  # Convert seconds to minutes
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "(mm)"
    elif is_time_metric(metric_name):
        return "(min)"
    return ""


def format_p_value(p_value, n_permutations=10000):
    """Format p-value for display with appropriate precision

    Uses scientific notation for very small p-values, with minimum precision
    based on permutation test resolution (1/n_permutations).

    Parameters:
    -----------
    p_value : float
        The p-value to format
    n_permutations : int
        Number of permutations used (determines precision floor)
    """
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return "n/a"

    # Minimum detectable p-value from permutation test
    min_p = 1 / n_permutations

    # If p-value equals the minimum from permutation test precision, show it directly
    if p_value <= min_p:
        return f"p ≤ 1e-{len(str(int(1/min_p)))-1}"

    # Use scientific notation for very small p-values
    if p_value < 1e-4:
        return f"{p_value:.2e}"

    # For larger p-values, use fixed notation with 6 decimal places
    return f"{p_value:.6f}"


def calculate_cohens_d(group1_data, group2_data):
    """Calculate Cohen's d effect size between two groups

    Parameters:
    -----------
    group1_data : array-like
        Data from first group
    group2_data : array-like
        Data from second group

    Returns:
    --------
    float : Cohen's d effect size
    """
    n1 = len(group1_data)
    n2 = len(group2_data)

    mean1 = np.mean(group1_data)
    mean2 = np.mean(group2_data)

    var1 = np.var(group1_data, ddof=1)
    var2 = np.var(group2_data, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean2 - mean1) / pooled_std


def bootstrap_ci_difference(group1_data, group2_data, n_bootstrap=10000, ci=95):
    """Calculate bootstrapped confidence interval for difference between groups

    Parameters:
    -----------
    group1_data : array-like
        Data from first group
    group2_data : array-like
        Data from second group
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)

    Returns:
    --------
    tuple : (lower_bound, upper_bound) of the CI
    """
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(group1_data, size=len(group1_data), replace=True)
        sample2 = np.random.choice(group2_data, size=len(group2_data), replace=True)

        # Calculate difference in means
        diff = np.mean(sample2) - np.mean(sample1)
        bootstrap_diffs.append(diff)

    # Calculate CI
    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)

    return lower, upper


def get_elegant_metric_name(metric_name):
    """Convert metric code names to elegant display names"""
    metric_map = {
        # Event metrics
        "nb_events": "Number of events",
        "has_major": "Has major event",
        "first_major_event": "First major event",
        "first_major_event_time": "First major event time",
        "max_event": "Max event",
        "max_event_time": "Max event time",
        "final_event": "Final event",
        "final_event_time": "Final event time",
        "has_significant": "Has significant event",
        "nb_significant_events": "Number of significant events",
        "significant_ratio": "Significant event ratio",
        "first_significant_event": "First significant event",
        "first_significant_event_time": "First significant event time",
        # Derived timing metrics
        "time_significant_to_major": "Time: first significant → first major event",
        "time_significant_to_max": "Time: first significant → max event",
        # Distance metrics
        "max_distance": "Max ball distance",
        "distance_moved": "Total ball distance moved",
        "distance_ratio": "Distance ratio",
        "fly_distance_moved": "Fly distance moved",
        # Interaction metrics
        "overall_interaction_rate": "Interaction rate",
        "interaction_persistence": "Interaction persistence",
        "interaction_proportion": "Interaction proportion",
        "time_to_first_interaction": "Time to first interaction",
        # Push/pull metrics
        "pushed": "Number of pushes",
        "pulled": "Number of pulls",
        "pulling_ratio": "Pulling ratio",
        "head_pushing_ratio": "Head pushing ratio",
        # Speed metrics
        "normalized_speed": "Normalized speed",
        "speed_during_interactions": "Speed during interactions",
        "speed_trend": "Speed trend",
        # Pause/stop metrics
        "has_long_pauses": "Has long pauses",
        "nb_stops": "Number of stops",
        "total_stop_duration": "Total stop duration",
        "median_stop_duration": "Median stop duration",
        "nb_pauses": "Number of pauses",
        "total_pause_duration": "Total pause duration",
        "median_pause_duration": "Median pause duration",
        "nb_long_pauses": "Number of long pauses",
        "total_long_pause_duration": "Total long pause duration",
        "median_long_pause_duration": "Median long pause duration",
        "pauses_persistence": "Pauses persistence",
        "cumulated_breaks_duration": "Total break duration",
        # Freeze metrics
        "nb_freeze": "Number of freezes",
        "median_freeze_duration": "Median freeze duration",
        # Chamber metrics
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "chamber_exit_time": "Chamber exit time",
        "time_chamber_beginning": "Time in chamber (early)",
        # Task completion
        "has_finished": "Task completed",
        "persistence_at_end": "Persistence at end",
        # Other behavioral metrics
        "fraction_not_facing_ball": "Fraction not facing ball",
        "leg_visibility_ratio": "Leg visibility ratio",
        "flailing": "Leg flailing",
        "median_head_ball_distance": "Median head-ball distance",
        "mean_head_ball_distance": "Mean head-ball distance",
    }

    return metric_map.get(metric_name, metric_name)


def format_metric_label(metric_name):
    """Format metric name with elegant display name and appropriate unit"""
    elegant_name = get_elegant_metric_name(metric_name)
    unit = get_metric_unit(metric_name)
    if unit:
        return f"{elegant_name} {unit}"
    return elegant_name


def generate_magnet_permutation_plots(
    data,
    metrics,
    y="Magnet",
    control_group="y",
    hue=None,
    palette="Set2",
    fig_width_mm=64,
    fig_height_mm=89,
    font_size_ticks=9,
    font_size_labels=11,
    font_size_legend=10,
    font_size_annotations=12,
    output_dir="permutation_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
    n_permutations=10000,
):
    """
    Generates jitterboxplots for each metric with permutation tests between Magnet and non-Magnet groups.
    Applies FDR correction across all comparisons for each metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (Magnet). Default is "Magnet".
        control_group (str): Name of the control group. Default is "non-Magnet".
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette: Color palette for the plots (str or dict). Default is "Set2".
        fig_width_mm (float, optional): Figure width in mm. Default is 64.
        fig_height_mm (float, optional): Figure height in mm. Default is 89.
        font_size_ticks (float, optional): Font size for tick labels. Default is 9.
        font_size_labels (float, optional): Font size for axis labels. Default is 11.
        font_size_legend (float, optional): Font size for legend. Default is 10.
        font_size_annotations (float, optional): Font size for statistical annotations. Default is 12.
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
        print(f"Generating permutation jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

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

        # Get raw data and convert if needed
        control_data_raw = plot_data[plot_data[y] == control_group][metric].dropna()
        test_data_raw = plot_data[plot_data[y] == test_group][metric].dropna()

        if len(control_data_raw) < 3 or len(test_data_raw) < 3:
            print(f"  ⚠️  Insufficient data for {metric} (control: {len(control_data_raw)}, test: {len(test_data_raw)})")
            continue

        # Convert data for plotting (pixels to mm if needed)
        control_data = convert_metric_data(control_data_raw, metric)
        test_data = convert_metric_data(test_data_raw, metric)

        # Perform permutation test (on CONVERTED data) using median difference as statistic
        control_median = control_data.median()
        test_median = test_data.median()
        median_diff = test_median - control_median

        n1 = len(control_data)
        n2 = len(test_data)

        obs_stat = float(median_diff)

        combined = np.concatenate([control_data.values, test_data.values])
        perm_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_control = combined[:n1]
            perm_test = combined[n1:]
            perm_diffs[i] = np.median(perm_test) - np.median(perm_control)

        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))

        # Calculate effect size (Cohen's d using converted data with means)
        cohens_d = calculate_cohens_d(control_data.values, test_data.values)

        # Calculate bootstrapped confidence interval for mean difference
        ci_lower, ci_upper = bootstrap_ci_difference(control_data.values, test_data.values, n_bootstrap=10000, ci=95)

        # Calculate percentage change and CI as percentages (relative to control)
        control_mean = np.mean(control_data.values)
        test_mean = np.mean(test_data.values)
        mean_diff = test_mean - control_mean

        if control_mean != 0:
            pct_change = (mean_diff / control_mean) * 100
            pct_ci_lower = (ci_lower / control_mean) * 100
            pct_ci_upper = (ci_upper / control_mean) * 100
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan

        # Also compute median-based effect size for reference
        s1 = control_data.std(ddof=1)
        s2 = test_data.std(ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(1, (n1 + n2 - 2)))
        effect_size = float((test_median - control_median) / pooled_sd) if pooled_sd > 0 else np.nan

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
            "Mean_diff": mean_diff,
            "Pct_change": pct_change,
            "U_statistic": np.nan,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pct_ci_lower": pct_ci_lower,
            "pct_ci_upper": pct_ci_upper,
            "effect_size": effect_size,
            "n_permutations": n_permutations,
        }
        all_stats.append(stat_result)

        # Create plot with publication-quality layout
        fig, ax = plt.subplots(1, 1, figsize=(mm_to_inches(fig_width_mm), mm_to_inches(fig_height_mm)))

        # Map to intuitive labels with sample sizes: "n" -> "No access to ball", "y" -> "Access to immobile ball"
        if control_group == "n" and test_group == "y":
            control_label = f"No access\nto ball\n(n={n1})"
            test_label = f"Access to\nimmobile ball\n(n={n2})"
        else:
            control_label = f"{control_group}\n(n={n1})"
            test_label = f"{test_group}\n(n={n2})"

        # Create color palette (matching F1 style)
        colors = {control_group: "#faa41a", test_group: "#3953A4"}  # Orange for control, blue for test

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

        # Style the boxes with no fill and black outlines
        bp["boxes"][0].set_facecolor("none")
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][0].set_linewidth(1.5)
        bp["boxes"][1].set_facecolor("none")
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

        # Draw significance bar and asterisks (always show, including ns)
        bar_height = y_max + 0.08 * y_range
        ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
        ax.text(
            0.5,
            bar_height + 0.02 * y_range,
            sig_symbol,
            ha="center",
            va="bottom",
            fontsize=font_size_annotations,
            fontname="Arial",
        )

        # Always add p-value text in top-left corner (for both significant and non-significant)
        p_text = f"p = {format_p_value(p_value, n_permutations)}"

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
        ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5, labelsize=font_size_ticks)

        # Set x-axis labels with intuitive labels and sample sizes
        ax.set_xticks(positions)
        ax.set_xticklabels([control_label, test_label], fontsize=font_size_ticks, fontname="Arial")

        # Set y-axis label with units
        ylabel = format_metric_label(metric)
        ax.set_ylabel(ylabel, fontsize=font_size_labels, fontname="Arial")

        # Format y-axis tick labels
        ax.tick_params(axis="y", labelsize=font_size_ticks)
        for label in ax.get_yticklabels():
            label.set_fontname("Arial")

        legend = ax.get_legend()
        if legend is not None:
            legend.set_fontsize(font_size_legend)

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

    if len(stats_df) > 0:
        stats_df["significant"] = stats_df["p_value"] < alpha

        # Sort by significance and median difference
        stats_df = stats_df.sort_values(["significant", "Median_diff"], ascending=[False, False])

    # Save statistics
    stats_path = output_dir / "magnetblock_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics saved to: {stats_path}")

    # Generate markdown report
    report_file = output_dir / "magnetblock_statistics_report.md"
    generate_text_report(stats_df, data, report_file)
    print(f"✅ Markdown report saved to: {report_file}")

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

    report_lines.append("# MAGNETBLOCK PERMUTATION TEST RESULTS")
    report_lines.append("")

    # Overall summary
    report_lines.append("## OVERALL SUMMARY")
    report_lines.append("")

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
        report_lines.append("## SIGNIFICANT RESULTS (p < 0.05)")
        report_lines.append("")
        report_lines.append(
            "|Metric|Difference|% Change|95% CI Lower|95% CI Upper|95% CI % Lower|95% CI % Upper|Cohen's D|P-value|"
        )
        report_lines.append("|---|---|---|---|---|---|---|---|---|")

        sig_df = stats_df[stats_df["significant"]].copy()

        for _, row in sig_df.iterrows():
            p_text = format_p_value(row["p_value"], n_permutations=row.get("n_permutations", 10000))
            cohens_d_text = f"{row['cohens_d']:.3f}" if not np.isnan(row.get("cohens_d", np.nan)) else "n/a"
            ci_lower_text = f"{row['ci_lower']:.3f}" if not np.isnan(row.get("ci_lower", np.nan)) else "n/a"
            ci_upper_text = f"{row['ci_upper']:.3f}" if not np.isnan(row.get("ci_upper", np.nan)) else "n/a"
            pct_change_text = f"{row['Pct_change']:.1f}%" if not np.isnan(row.get("Pct_change", np.nan)) else "n/a"
            pct_ci_lower_text = (
                f"{row['pct_ci_lower']:.1f}%" if not np.isnan(row.get("pct_ci_lower", np.nan)) else "n/a"
            )
            pct_ci_upper_text = (
                f"{row['pct_ci_upper']:.1f}%" if not np.isnan(row.get("pct_ci_upper", np.nan)) else "n/a"
            )

            # Use Mean_diff instead of Median_diff for consistency with CI
            diff_text = f"{row['Mean_diff']:.3f}" if not np.isnan(row.get("Mean_diff", np.nan)) else "n/a"

            report_lines.append(
                f"|{row['Metric']}|{diff_text}|{pct_change_text}|{ci_lower_text}|{ci_upper_text}|{pct_ci_lower_text}|{pct_ci_upper_text}|{cohens_d_text}|{p_text}|"
            )

        report_lines.append("")

    # Non-significant results
    non_sig_count = total_comparisons - total_significant
    if non_sig_count > 0:
        report_lines.append(f"## NON-SIGNIFICANT RESULTS ({non_sig_count} metrics)")
        report_lines.append("")
        report_lines.append("|Metric|Difference|% Change|Cohen's D|P-value|")
        report_lines.append("|---|---|---|---|---|")

        non_sig_df = stats_df[~stats_df["significant"]].copy()
        for _, row in non_sig_df.iterrows():
            p_text = format_p_value(row["p_value"], n_permutations=row.get("n_permutations", 10000))
            cohens_d_text = f"{row['cohens_d']:.3f}" if not np.isnan(row.get("cohens_d", np.nan)) else "n/a"
            diff_text = f"{row['Mean_diff']:.3f}" if not np.isnan(row.get("Mean_diff", np.nan)) else "n/a"
            pct_change_text = f"{row['Pct_change']:.1f}%" if not np.isnan(row.get("Pct_change", np.nan)) else "n/a"
            report_lines.append(f"|{row['Metric']}|{diff_text}|{pct_change_text}|{cohens_d_text}|{p_text}|")

    report_lines.append("")
    report_lines.append("-" * 80)

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
        dataset = read_feather(dataset_path)
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

    # ---------------------------------------------------------------------------
    # Derived metrics
    # ---------------------------------------------------------------------------
    # Time between first significant event and first major event
    if "first_significant_event_time" in dataset.columns and "first_major_event_time" in dataset.columns:
        dataset["time_significant_to_major"] = (
            dataset["first_major_event_time"] - dataset["first_significant_event_time"]
        )
        print("  Derived: time_significant_to_major")

    # Time between first significant event and time of max event
    if "first_significant_event_time" in dataset.columns and "max_event_time" in dataset.columns:
        dataset["time_significant_to_max"] = dataset["max_event_time"] - dataset["first_significant_event_time"]
        print("  Derived: time_significant_to_max")

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
    data,
    binary_metrics,
    y="Magnet",
    output_dir=None,
    overwrite=True,
    control_group="non-Magnet",
    fig_width_mm=64,
    fig_height_mm=89,
    font_size_ticks=9,
    font_size_labels=11,
    font_size_legend=10,
    font_size_annotations=12,
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
        create_binary_metric_plot(
            data,
            metric,
            y,
            output_dir,
            control_group,
            fig_width_mm=fig_width_mm,
            fig_height_mm=fig_height_mm,
            font_size_ticks=font_size_ticks,
            font_size_labels=font_size_labels,
            font_size_legend=font_size_legend,
            font_size_annotations=font_size_annotations,
        )

    if not all_stats:
        print("⚠️  No binary statistics computed")
        return pd.DataFrame()

    # Convert to DataFrame
    stats_df = pd.DataFrame(all_stats)

    if len(stats_df) > 0:
        stats_df["significant"] = stats_df["p_value"] < 0.05

    # Save statistics
    stats_path = output_dir / "binary_metrics_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Binary statistics saved to: {stats_path}")

    return stats_df


def create_binary_metric_plot(
    data,
    metric,
    y,
    output_dir,
    control_group=None,
    fig_width_mm=64,
    fig_height_mm=89,
    font_size_ticks=9,
    font_size_labels=11,
    font_size_legend=10,
    font_size_annotations=12,
):
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

    # Create plot with publication-quality layout
    fig, ax = plt.subplots(1, 1, figsize=(mm_to_inches(fig_width_mm), mm_to_inches(fig_height_mm)))

    # Map to intuitive labels with sample sizes: "n" -> "No access to ball", "y" -> "Access to immobile ball"
    display_labels = []
    for i, group in enumerate(groups):
        n = n_totals[i]
        if group == "n":
            display_labels.append(f"No access\nto ball\n(n={n})")
        elif group == "y":
            display_labels.append(f"Access to\nimmobile ball\n(n={n})")
        else:
            display_labels.append(f"{group}\n(n={n})")

    # Use F1-style colors
    colors = {control_group: "#faa41a"}  # Orange for control
    if len(groups) == 2:
        test_group = [g for g in groups if g != control_group][0]
        colors[test_group] = "#3953A4"  # Blue for test

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

    # Add significance annotation if there are exactly 2 groups (always show, including ns)
    if len(groups) == 2:
        y_max = max(proportions) + max([ci[1] for ci in cis])
        bar_height = y_max + 0.05
        ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
        ax.text(
            0.5,
            bar_height + 0.02,
            sig_symbol,
            ha="center",
            va="bottom",
            fontsize=font_size_annotations,
            fontname="Arial",
        )

    # Always add p-value text in top-left corner (for both significant and non-significant)
    p_text = f"p = {format_p_value(p_value)}"

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
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5, labelsize=font_size_ticks)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontsize=font_size_ticks, fontname="Arial")
    ax.set_ylabel(f"Proportion\n{metric}", fontsize=font_size_labels, fontname="Arial")
    ax.set_ylim(0, 1.0 if sig_symbol == "ns" else 1.15)

    # Format y-axis tick labels
    ax.tick_params(axis="y", labelsize=font_size_ticks)
    for label in ax.get_yticklabels():
        label.set_fontname("Arial")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_fontsize(font_size_legend)

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


def main(overwrite=True, test_mode=False, metrics_file=None):
    """Main function to run permutation analysis for MagnetBlock experiments

    Parameters
    ----------
    overwrite : bool
        Whether to overwrite existing plots
    test_mode : bool
        If True, run a small subset for testing
    metrics_file : str or None
        Path to a file containing metric names (one per line) to restrict analysis
    """

    print(f"\n{'='*80}")
    print("MAGNETBLOCK PERMUTATION TEST ANALYSIS")
    print(f"{'='*80}\n")

    # Load dataset
    dataset = load_and_clean_dataset(test_mode=test_mode)

    # Define output directory
    output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/permutation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")

    # Get all numeric columns (excluding Magnet and fly identifier)
    exclude_cols = ["Magnet", "fly", "Corridor"]
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [col for col in numeric_cols if col not in exclude_cols]

    # If a metrics file is provided, read it and intersect with available metrics
    if metrics_file is not None:
        metrics_path = Path(metrics_file)
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                requested = [line.strip() for line in f if line.strip()]
            # Keep only metrics that exist in the dataset and are numeric
            requested_present = [m for m in requested if m in metrics]
            missing = [m for m in requested if m not in metrics]
            if missing:
                print(f"Warning: {len(missing)} requested metrics not found or non-numeric: {missing}")
            if requested_present:
                metrics = requested_present
                print(f"Using {len(metrics)} metrics from {metrics_path}")
            else:
                print(f"No requested metrics found in dataset; falling back to automatic metric selection")
        else:
            print(f"Warning: metrics file not found: {metrics_path}; using automatic selection")

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

    fig_width_mm = 72
    fig_height_mm = 104
    font_size_ticks = 10
    font_size_labels = 14
    font_size_legend = 12
    font_size_annotations = 11

    stats_df = generate_magnet_permutation_plots(
        data=dataset,
        metrics=metrics,
        y="Magnet",
        control_group="n",
        output_dir=output_dir,
        fig_width_mm=fig_width_mm,
        fig_height_mm=fig_height_mm,
        font_size_ticks=font_size_ticks,
        font_size_labels=font_size_labels,
        font_size_legend=font_size_legend,
        font_size_annotations=font_size_annotations,
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
            fig_width_mm=fig_width_mm,
            fig_height_mm=fig_height_mm,
            font_size_ticks=font_size_ticks,
            font_size_labels=font_size_labels,
            font_size_legend=font_size_legend,
            font_size_annotations=font_size_annotations,
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
