#!/usr/bin/env python3
"""
Script to generate permutation test jitterboxplots for ball scents experiments.

This script generates comprehensive permutation test visualizations with:
- Permutation tests (10,000 permutations) instead of Mann-Whitney U tests
- Outlier clipping at 99th percentile for better visualization
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Statistical significance annotations (*, **, ***) with gray p-values
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_permutation_ballscents.py [--overwrite]

Arguments:
    --overwrite: If specified, overwrite existing plots. If not specified, skip metrics that already have plots.
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
import matplotlib

import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import scipy.stats as stats_module
from statsmodels.stats.multitest import multipletests
import time
import argparse


# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def mm_to_inches(mm_value):
    return mm_value / 25.4


def bootstrap_ci_difference(group1_data, group2_data, n_bootstrap=10000, ci=95):
    """Calculate bootstrapped CI for mean(group2) - mean(group1)."""
    if len(group1_data) == 0 or len(group2_data) == 0:
        return np.nan, np.nan

    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample1 = np.random.choice(group1_data, size=len(group1_data), replace=True)
        sample2 = np.random.choice(group2_data, size=len(group2_data), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)
    return float(lower), float(upper)


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
        return data / PIXELS_PER_MM  # Convert pixels to mm
    if is_time_metric(metric_name):
        return data / 60.0  # Convert seconds to minutes
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "(mm)"
    if is_time_metric(metric_name):
        return "(min)"
    return ""


def format_metric_label(metric_name):
    """Format metric name with appropriate unit"""
    unit = get_metric_unit(metric_name)
    if unit:
        return f"{metric_name} {unit}"
    return metric_name


# Fixed color mapping for ball scents - ensures consistent colors across all plots
# Colors chosen to distinguish different experimental conditions
BALLSCENT_COLORS = {
    "New": "#7f7f7f",  # Grey — control (new clean ball)
    "Pre-exposed": "#1f77b4",  # Blue  (CtrlScent: pre-exposed to fly odors)
    "Washed": "#2ca02c",  # Green
    "Washed + Pre-exposed": "#ff7f0e",  # Orange
}


def get_ballscent_color(ballscent):
    """Get the fixed color for a ball scent condition.

    Parameters:
    -----------
    ballscent : str
        Name of the ball scent condition

    Returns:
    --------
    str
        Hex color code for the ball scent condition
    """
    # Return the color, or a fallback grey if not found
    return BALLSCENT_COLORS.get(ballscent, "#7f7f7f")


def normalize_ball_scent_labels(df, group_col="BallScent"):
    """Normalize/alias BallScent values to canonical factorial labels.

    Maps existing values to one of: Ctrl, CtrlScent, Washed, Scented, New, NewScent
    using substring and fuzzy matching when needed.
    """
    if group_col not in df.columns:
        return df

    design_keys = ["Ctrl", "CtrlScent", "Washed", "Scented", "New", "NewScent"]
    available = pd.Series(df[group_col].dropna().unique()).astype(str).tolist()
    from difflib import get_close_matches

    mapping = {}
    for val in available:
        if val in design_keys:
            mapping[val] = val
            continue
        vs = str(val).strip().lower()
        found = None
        for k in design_keys:
            ks = k.lower()
            if vs == ks or ks in vs or vs in ks:
                found = k
                break
        if not found:
            candidates = get_close_matches(vs, [k.lower() for k in design_keys], n=1, cutoff=0.6)
            if candidates:
                for k in design_keys:
                    if k.lower() == candidates[0]:
                        found = k
                        break

        mapping[val] = found if found is not None else val

    # For any entries that were not mapped, try simple substring heuristics
    for val in list(mapping.keys()):
        if mapping[val] == val:
            vs = str(val).strip().lower()
            if "new" in vs and "scent" in vs:
                mapping[val] = "NewScent"
            elif "new" in vs:
                mapping[val] = "New"
            elif "wash" in vs or "washed" in vs:
                if "scent" in vs:
                    mapping[val] = "Scented"
                else:
                    mapping[val] = "Washed"
            elif "ctrl" in vs and "scent" in vs:
                mapping[val] = "CtrlScent"
            elif "scent" in vs:
                # Prefer mapping to Scented/NewScent when 'scent' appears alone
                if "new" in vs:
                    mapping[val] = "NewScent"
                elif "wash" in vs or "washed" in vs or vs == "scented":
                    mapping[val] = "Scented"
                else:
                    mapping[val] = "CtrlScent" if "ctrl" in vs else "Scented"
            elif "pre" in vs or "exposed" in vs:
                mapping[val] = "CtrlScent"
            else:
                mapping[val] = val

    remapped = {k: v for k, v in mapping.items() if k != v}
    print(f"Normalizing {group_col} labels (total variants: {len(mapping)}):")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")

    df = df.copy()
    df[group_col] = df[group_col].map(lambda x: mapping.get(str(x), x))
    # Preserve canonical mapping in a separate column for downstream matching
    canonical_col = f"{group_col}_canonical"
    df[canonical_col] = df[group_col]
    # Map canonical keys to descriptive factorial labels for plots/reports
    display_map = {
        "Ctrl": "Ctrl",
        "CtrlScent": "Pre-exposed",
        "Washed": "Washed",
        "Scented": "Washed + Pre-exposed",
        "New": "New",
        "NewScent": "New + Pre-exposed",
    }

    df[group_col] = df[group_col].map(lambda x: display_map.get(x, x))
    return df


def generate_BallScent_permutation_plots(
    data,
    metrics,
    y="BallScent",
    control_BallScent="New",  # New ball as control
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
    Generates jitterboxplots for each metric with permutation tests between each ball scent and control.
    Applies FDR correction across all comparisons for each metric.
    Clips outliers at 99th percentile for better visualization.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (ball scents). Default is "BallScent".
        control_BallScent (str): Name of the control ball scent. Default is "New".
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette: Color palette for the plots (str or dict). Default is "Set2".
        fig_width_mm (float): Figure width in mm. Default is 64.
        fig_height_mm (float): Figure height in mm. Default is 89.
        font_size_ticks (float): Font size for tick labels. Default is 9.
        font_size_labels (float): Font size for axis labels. Default is 11.
        font_size_legend (float): Font size for legend. Default is 10.
        font_size_annotations (float): Font size for statistical annotations. Default is 12.
        output_dir: Directory to save the plots. Default is "permutation_plots".
        fdr_method (str): Method for FDR correction. Default is "fdr_bh" (Benjamini-Hochberg).
        alpha (float): Significance level after FDR correction. Default is 0.05.
        n_permutations (int): Number of permutations for permutation tests. Default is 10000.

    Returns:
        pd.DataFrame: Statistics table with permutation test results and FDR correction.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"]

    # Fixed ordering for ball scents: New (ctrl), Pre-exposed, Washed, Washed+Pre-exposed
    # Data is pre-filtered to these four conditions in load_and_clean_dataset.
    fixed_order = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]
    # Keep only conditions present in this data slice
    fixed_order = [cond for cond in fixed_order if cond in data[y].unique()]
    # Filter data to the fixed conditions (drops any residual Ctrl rows)
    data = data[data[y].isin(fixed_order)].copy()
    print(f"Conditions: {fixed_order}")

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating permutation test jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y])
        plot_data = plot_data.copy()
        plot_data[metric] = convert_metric_data(plot_data[metric], metric)

        # Get all ball scents except control
        ball_scents = [bs for bs in plot_data[y].unique() if bs != control_BallScent]

        # Check if control exists
        if control_BallScent not in plot_data[y].unique():
            print(f"Warning: Control ball scent '{control_BallScent}' not found for metric {metric}")
            continue

        control_vals = plot_data[plot_data[y] == control_BallScent][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue

        control_median = control_vals.median()  # Calculate control median once
        control_mean = control_vals.mean()

        # Perform permutation tests for all ball scents vs control
        pvals = []
        test_results = []

        for ball_scent in ball_scents:
            test_vals = plot_data[plot_data[y] == ball_scent][metric].dropna()

            if len(test_vals) == 0:
                continue

            try:
                # Permutation test using median difference as statistic
                test_median = test_vals.median()
                test_mean = test_vals.mean()
                obs_diff = test_median - control_median

                n1 = len(control_vals)
                n2 = len(test_vals)

                # Combine data for permutation
                combined = np.concatenate([control_vals.values, test_vals.values])
                perm_diffs = np.empty(n_permutations)

                for i in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_control = combined[:n1]
                    perm_test = combined[n1:]
                    perm_diffs[i] = np.median(perm_test) - np.median(perm_control)

                # Two-sided p-value
                pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

                # Determine direction of effect
                direction = "increased" if test_median > control_median else "decreased"
                effect_size = test_median - control_median

                # Bootstrap CI for mean difference (test - control)
                ci_lower, ci_upper = bootstrap_ci_difference(
                    control_vals.values,
                    test_vals.values,
                    n_bootstrap=n_permutations,
                    ci=95,
                )
                mean_diff = test_mean - control_mean

                if control_mean != 0:
                    pct_change = (mean_diff / control_mean) * 100
                    pct_ci_lower = (ci_lower / control_mean) * 100
                    pct_ci_upper = (ci_upper / control_mean) * 100
                else:
                    pct_change = np.nan
                    pct_ci_lower = np.nan
                    pct_ci_upper = np.nan

            except Exception as e:
                print(f"Error in permutation test for {ball_scent} vs {control_BallScent}: {e}")
                pval = 1.0
                direction = "none"
                effect_size = 0.0
                test_mean = test_vals.mean() if len(test_vals) > 0 else np.nan
                mean_diff = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan

            pvals.append(pval)
            test_results.append(
                {
                    "BallScent": ball_scent,
                    "Metric": metric,
                    "Control": control_BallScent,
                    "pval_raw": pval,
                    "direction": direction,
                    "effect_size": effect_size,
                    "test_median": test_vals.median() if len(test_vals) > 0 else np.nan,
                    "control_median": control_median,
                    "test_mean": test_mean,
                    "control_mean": control_mean,
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "pct_change": pct_change,
                    "pct_ci_lower": pct_ci_lower,
                    "pct_ci_upper": pct_ci_upper,
                    "test_n": len(test_vals),
                    "control_n": len(control_vals),
                    "n_permutations": n_permutations,
                    "n_bootstrap": n_permutations,
                }
            )

        if not pvals:
            print(f"No valid comparisons for metric {metric}")
            continue

        # Apply FDR correction
        rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(pvals, alpha=alpha, method=fdr_method)

        # Update test results with FDR correction
        for i, result in enumerate(test_results):
            result["pval_fdr"] = pvals_corrected[i]
            result["significant_fdr"] = rejected[i]

            # Determine significance level based on FDR-corrected p-values
            if pvals_corrected[i] < 0.001:
                result["sig_level"] = "***"
            elif pvals_corrected[i] < 0.01:
                result["sig_level"] = "**"
            elif pvals_corrected[i] < 0.05:
                result["sig_level"] = "*"
            else:
                result["sig_level"] = "ns"

            # Override direction if not significant after FDR
            if not rejected[i]:
                result["direction"] = "none"

        # Add control to results for sorting
        control_result = {
            "BallScent": control_BallScent,
            "Metric": metric,
            "Control": control_BallScent,
            "pval_raw": 1.0,
            "pval_fdr": 1.0,
            "significant_fdr": False,
            "sig_level": "control",
            "direction": "control",
            "effect_size": 0.0,
            "test_statistic": np.nan,
            "test_median": control_median,
            "control_median": control_median,
            "test_n": len(control_vals),
            "control_n": len(control_vals),
        }

        all_results = test_results + [control_result]

        # Use fixed ordering instead of significance-based sorting
        # This ensures consistent ordering across all plots for side-by-side comparison
        sorted_ball_scents = [bs for bs in fixed_order if bs in [r["BallScent"] for r in all_results]]

        # Update plot data with sorted categories
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_ball_scents, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # Clip outliers at 99th percentile for better visualization
        percentile_99 = plot_data[metric].quantile(0.99)
        plot_data_clipped = plot_data.copy()
        plot_data_clipped[metric] = plot_data_clipped[metric].clip(upper=percentile_99)

        # Store stats for output
        for result in test_results:  # Exclude control from final stats
            all_stats.append(result)

        # Calculate figure size based on mm parameters
        n_categories = len(sorted_ball_scents)
        fig_width_inches = mm_to_inches(fig_width_mm)
        fig_height_inches = mm_to_inches(fig_height_mm)

        # Create the plot with specified size (vertical orientation)
        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

        # Set up colors matching trajectory plots using fixed mapping
        # This ensures the same ball scent always gets the same color
        color_mapping = {}
        for bs in sorted_ball_scents:
            color_mapping[bs] = get_ballscent_color(bs)

        colors = [color_mapping[bs] for bs in sorted_ball_scents]

        x_positions = range(len(sorted_ball_scents))

        # Draw vertical boxplots with black outlines and no fill
        bp = ax.boxplot(
            [plot_data_clipped[plot_data_clipped[y] == bs][metric].dropna().values for bs in sorted_ball_scents],
            positions=x_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(linewidth=1.5, facecolor="none", edgecolor="black"),
            whiskerprops=dict(linewidth=1.5, color="black"),
            capprops=dict(linewidth=1.5, color="black"),
            medianprops=dict(linewidth=2, color="black"),
        )

        # Overlay stripplot for individual points with matching colors
        for i, ball_scent in enumerate(sorted_ball_scents):
            ball_scent_data = plot_data_clipped[plot_data_clipped[y] == ball_scent][metric].dropna()
            if len(ball_scent_data) == 0:
                continue

            # Add jitter to x positions
            x_jitter = np.random.normal(i, 0.08, size=len(ball_scent_data))
            ax.scatter(
                x_jitter,
                ball_scent_data.values,
                s=20,
                c=color_mapping[ball_scent],
                alpha=0.5,
                edgecolors="none",
                zorder=3,
            )

        # Add significance brackets and annotations
        y_max = plot_data_clipped[metric].max()
        y_min = plot_data_clipped[metric].min()
        y_range = y_max - y_min

        # Find control position
        control_idx = sorted_ball_scents.index(control_BallScent)

        # Add brackets for significant comparisons
        bracket_height = y_max + 0.05 * y_range
        bracket_offset = 0.12 * y_range  # Vertical spacing between brackets

        significant_comparisons = []
        for i, ball_scent in enumerate(sorted_ball_scents):
            if ball_scent == control_BallScent:
                continue

            result = next((r for r in all_results if r["BallScent"] == ball_scent), None)
            if result and result.get("significant_fdr", False):
                significant_comparisons.append((i, result))

        # Draw brackets for significant comparisons
        for bracket_idx, (test_idx, result) in enumerate(significant_comparisons):
            # Calculate bracket height (stack multiple brackets)
            current_height = bracket_height + bracket_idx * bracket_offset

            # Draw horizontal line connecting control to test
            x1, x2 = control_idx, test_idx
            if x1 > x2:
                x1, x2 = x2, x1

            # Draw bracket
            ax.plot([x1, x1], [current_height - 0.01 * y_range, current_height], "k-", linewidth=1.5)
            ax.plot([x1, x2], [current_height, current_height], "k-", linewidth=1.5)
            ax.plot([x2, x2], [current_height - 0.01 * y_range, current_height], "k-", linewidth=1.5)

            # Add significance stars in red
            mid_x = (x1 + x2) / 2
            ax.text(
                mid_x,
                current_height + 0.015 * y_range,
                result["sig_level"],
                ha="center",
                va="bottom",
                fontsize=font_size_annotations,
                color="red",
                fontweight="bold",
            )

        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sorted_ball_scents, rotation=45, ha="right", fontsize=font_size_ticks)
        ax.set_ylabel(format_metric_label(metric), fontsize=font_size_labels)
        ax.tick_params(axis="both", labelsize=font_size_ticks)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        # Use tight_layout with padding
        plt.tight_layout()

        # Adjust y-axis limits to accommodate brackets (AFTER formatting and tight_layout)
        if significant_comparisons:
            max_bracket_height = bracket_height + len(significant_comparisons) * bracket_offset + 0.08 * y_range
            ax.set_ylim(bottom=y_min - 0.05 * y_range, top=max_bracket_height)
        else:
            ax.set_ylim(bottom=y_min - 0.05 * y_range, top=y_max + 0.1 * y_range)

        # Save plot
        output_file = output_dir / f"{metric}_BallScent_permutation.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {output_file}")

        # Print statistical summary
        n_significant = sum(1 for r in test_results if r["significant_fdr"])
        print(f"  Significant: {n_significant}/{len(test_results)} (α={alpha}, FDR corrected)")

        # Print timing for this metric
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save statistics
        stats_file = output_dir / "BallScent_permutation_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "BallScent_permutation_report.md"
        generate_text_report(stats_df, data, control_BallScent, report_file)
        print(f"Text report saved to: {report_file}")

        # Print overall summary
        total_tests = len(stats_df)
        total_significant_raw = (stats_df["pval_raw"] < 0.05).sum()
        total_significant = stats_df["significant_fdr"].sum()

        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Raw significant (p<0.05): {total_significant_raw} ({100*total_significant_raw/total_tests:.1f}%)")
        print(f"FDR-corrected significant (p<{alpha}): {total_significant} ({100*total_significant/total_tests:.1f}%)")

    # Persist the compiled statistics to the canonical summaries folder so
    # downstream plotting scripts can find it at a known location.
    try:
        summaries_dir = Path("/mnt/upramdya_data/MD/Ball_scents/Plots/summaries/Permutation_BallScent")
        summaries_dir.mkdir(parents=True, exist_ok=True)
        stats_out = summaries_dir / "ballscent_permutation_statistics.csv"
        stats_df.to_csv(stats_out, index=False)
        print(f"Saved permutation statistics to: {stats_out}")
    except Exception as e:
        print(f"Warning: could not save stats CSV to canonical location: {e}")

    return stats_df


def generate_text_report(stats_df, data, control_BallScent, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    control_BallScent : str
        Name of the control ball scent
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Ball Scents Permutation Test Report")
    report_lines.append("## Statistical Analysis Results (Permutation Tests with FDR Correction)")
    report_lines.append("")
    report_lines.append(f"**Control group:** {control_BallScent}")
    report_lines.append("")

    # Group by metric
    metrics = stats_df["Metric"].unique()

    for metric in sorted(metrics):
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"## {metric}")
        report_lines.append("")

        # Get control median for reference
        control_median = 0  # Initialize default value
        control_data = data[data["BallScent"] == control_BallScent][metric].dropna()
        if len(control_data) > 0:
            control_median = control_data.median()
            control_mean = control_data.mean()
            control_std = control_data.std()
            control_n = len(control_data)

            report_lines.append(
                f"**Control ({control_BallScent}):** median = {control_median:.3f}, mean = {control_mean:.3f} ± {control_std:.3f} (n={control_n})"
            )
            report_lines.append("")
        else:
            report_lines.append(f"**Control ({control_BallScent}):** No data available")
            report_lines.append("")

        # Analyze each ball type vs control
        significant_increased = []
        significant_decreased = []
        non_significant = []

        for _, row in metric_stats.iterrows():
            ball_type = row["BallScent"]
            p_corrected = row["pval_fdr"]
            sig_level = row["sig_level"]
            test_median = row["test_median"]
            effect_size = row["effect_size"]
            test_n = row["test_n"]

            # Calculate percent change
            if control_median != 0:
                percent_change = (effect_size / control_median) * 100
            else:
                percent_change = 0

            description = f'"{ball_type}" (median = {test_median:.3f}, n={test_n})'

            # Get pval_fdr for all cases
            pval_corrected = row["pval_fdr"]

            if row["significant_fdr"]:
                sig_level = row["sig_level"]
                direction = row["direction"]
                effect_size = row["effect_size"]

                if row["direction"] == "increased":
                    significant_increased.append(
                        {
                            "ball_type": ball_type,
                            "description": description,
                            "effect_size": effect_size,
                            "percent_change": percent_change,
                            "p_corrected": pval_corrected,
                            "sig_level": sig_level,
                        }
                    )
                elif row["direction"] == "decreased":
                    significant_decreased.append(
                        {
                            "ball_type": ball_type,
                            "description": description,
                            "effect_size": effect_size,
                            "percent_change": percent_change,
                            "p_corrected": pval_corrected,
                            "sig_level": sig_level,
                        }
                    )
            else:
                non_significant.append(
                    {
                        "ball_type": ball_type,
                        "description": description,
                        "effect_size": effect_size,
                        "percent_change": percent_change,
                        "p_corrected": pval_corrected,
                    }
                )

        # Write results
        if significant_increased:
            # Sort by effect size (largest increase first)
            significant_increased.sort(key=lambda x: x["effect_size"], reverse=True)
            report_lines.append("**Significantly higher values than control:**")
            for item in significant_increased:
                report_lines.append(
                    f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_corrected']:.4f}{item['sig_level']})"
                )
            report_lines.append("")

        if significant_decreased:
            # Sort by effect size (largest decrease first, so most negative first)
            significant_decreased.sort(key=lambda x: x["effect_size"])
            report_lines.append("**Significantly lower values than control:**")
            for item in significant_decreased:
                report_lines.append(
                    f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_corrected']:.4f}{item['sig_level']})"
                )
            report_lines.append("")

        if non_significant:
            # Sort by effect size for consistency
            non_significant.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
            report_lines.append("**No significant difference from control:**")
            for item in non_significant:
                report_lines.append(
                    f"- {item['description']} showed {item['percent_change']:+.1f}% change (p={item['p_corrected']:.4f}, ns)"
                )
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Add summary section
    report_lines.append("## Summary")
    report_lines.append("")

    total_comparisons = len(stats_df)
    total_significant = stats_df["significant_fdr"].sum()
    total_increased = len(stats_df[(stats_df["significant_fdr"]) & (stats_df["direction"] == "increased")])
    total_decreased = len(stats_df[(stats_df["significant_fdr"]) & (stats_df["direction"] == "decreased")])

    report_lines.append(f"- **Total comparisons:** {total_comparisons}")
    report_lines.append(
        f"- **Significant results (FDR corrected):** {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
    )
    report_lines.append(f"  - Significantly increased: {total_increased}")
    report_lines.append(f"  - Significantly decreased: {total_decreased}")
    report_lines.append(
        f"- **Non-significant results:** {total_comparisons - total_significant} ({100*(total_comparisons - total_significant)/total_comparisons:.1f}%)"
    )
    report_lines.append("")

    # Add metrics summary
    metrics_with_significant = stats_df[stats_df["significant_fdr"]]["Metric"].nunique()
    total_metrics = stats_df["Metric"].nunique()

    report_lines.append(
        f"- **Metrics with significant differences:** {metrics_with_significant}/{total_metrics} ({100*metrics_with_significant/total_metrics:.1f}%)"
    )
    report_lines.append("")

    # Write to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Generated text report with {len(metrics)} metrics and {total_comparisons} comparisons")


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the ball types dataset

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the ball types dataset
    dataset_path = (
        "/mnt/upramdya_data/MD/Ball_scents/Datasets/251103_10_summary_ballscents_Data/summary/pooled_summary.feather"
    )

    print(f"Loading ball types dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Ball types dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Ball types dataset not found at {dataset_path}")

    # Drop TNT-specific metadata columns that cause pandas warnings
    # These columns are not needed for ball types analysis
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
        print(f"Dropped TNT metadata columns to avoid warnings: {columns_to_drop}")
        print(f"Dataset shape after dropping metadata: {dataset.shape}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Clean NA values following the same logic as All_metrics.py
    # Use safe column access to handle missing columns - FIXED for performance
    def safe_fillna(column_name, fill_value):
        if column_name in dataset.columns:
            dataset[column_name] = dataset[column_name].fillna(fill_value)
            print(f"Filled NaN values in {column_name} with {fill_value}")
        else:
            print(f"Column {column_name} not found in dataset, skipping...")

    # safe_fillna("max_event", -1)
    # safe_fillna("first_significant_event", -1)
    # safe_fillna("first_major_event", -1)
    # safe_fillna("final_event", -1)

    # Fill time columns with 3600
    # safe_fillna("max_event_time", 3600)
    # safe_fillna("first_significant_event_time", 3600)
    # safe_fillna("first_major_event_time", 3600)
    # safe_fillna("final_event_time", 3600)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    # Fill remaining NA values with appropriate defaults (only if columns exist)
    # safe_fillna("pulling_ratio", 0)
    # safe_fillna("avg_displacement_after_success", 0)
    # safe_fillna("avg_displacement_after_failure", 0)
    # safe_fillna("influence_ratio", 0)

    # Handle new metrics with appropriate defaults - FIXED for performance
    new_metrics_defaults = {
        "has_finished": np.nan,  # Binary metric, 0 if no finish detected
        "persistence_at_end": np.nan,  # Time fraction, 0 if no persistence
        "fly_distance_moved": np.nan,  # Distance, 0 if no movement detected
        "time_chamber_beginning": np.nan,  # Time, 0 if no time in beginning chamber
        "median_freeze_duration": np.nan,  # Duration, 0 if no freezes detected
        "fraction_not_facing_ball": np.nan,  # Fraction, 0 if always facing ball or no data
        "flailing": np.nan,  # Motion energy, 0 if no leg movement detected
        "head_pushing_ratio": np.nan,  # Ratio, 0.5 if no contact data (neutral)
        "compute_median_head_ball_distance": np.nan,  # Distance, 0 if no contact data
        "compute_mean_head_ball_distance": np.nan,  # Distance, 0 if no contact data
        "median_head_ball_distance": np.nan,  # Distance, 0 if no contact data
        "mean_head_ball_distance": np.nan,  # Distance, 0 if no contact data
    }

    # Fill all new metrics at once for better performance
    fillna_dict = {}
    for metric, default_value in new_metrics_defaults.items():
        if metric in dataset.columns:
            nan_count = dataset[metric].isnull().sum()
            if nan_count > 0:
                fillna_dict[metric] = default_value
                print(f"Will fill {nan_count} NaN values in {metric} with {default_value}")
            else:
                print(f"No NaN values found in {metric}")
        else:
            print(f"New metric {metric} not found in dataset")

    # Perform batch fillna for better performance
    if fillna_dict:
        dataset.fillna(fillna_dict, inplace=True)
        print(f"Batch filled NaN values for {len(fillna_dict)} metrics")

    print(f"Dataset cleaning completed successfully")

    # Load additional historical Ctrl flies (summary-level) and append to dataset
    print("\nChecking for additional historical Ctrl cohorts to include (summary-level)...")
    ctrl_summary_paths = [
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250815_18_summary_control_folders_Data/summary/230704_FeedingState_1_AM_Videos_Tracked_summary.feather",
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250815_18_summary_control_folders_Data/summary/230705_FeedingState_2_AM_Videos_Tracked_summary.feather",
    ]

    ctrl_summary_dfs = []
    for summ_path in ctrl_summary_paths:
        try:
            print(f"  Loading Ctrl summary from: {summ_path}")
            ctrl_df = pd.read_feather(summ_path)

            # Filter to FeedingState == 'starved_noWater' if available
            if "FeedingState" in ctrl_df.columns:
                before = ctrl_df.shape
                ctrl_df = ctrl_df[ctrl_df["FeedingState"] == "starved_noWater"].copy()
                print(f"    Filtered starved_noWater: {before} -> {ctrl_df.shape}")
                if len(ctrl_df) == 0:
                    print("    ⚠️  No rows after FeedingState filter; skipping this file")
                    continue
            else:
                print("    ⚠️  FeedingState column not found in summary; keeping all rows")

            # Ensure BallScent column set to 'Ctrl'
            ctrl_df["BallScent"] = "Ctrl"

            ctrl_summary_dfs.append(ctrl_df)
            print(f"    ✅ Loaded Ctrl summary: shape={ctrl_df.shape}")
        except FileNotFoundError:
            print(f"    ⚠️  Not found: {summ_path}")
            continue
        except Exception as e:
            print(f"    ⚠️  Error loading {summ_path}: {e}")
            continue

    if ctrl_summary_dfs:
        print(f"\n📊 Appending {len(ctrl_summary_dfs)} Ctrl summary dataset(s) to main dataset...")
        # Align columns via outer concat (union); missing columns will be NaN
        combined = pd.concat([dataset] + ctrl_summary_dfs, ignore_index=True, sort=False)

        # Drop TNT-specific metadata again post-merge to keep columns clean
        columns_to_drop = [col for col in tnt_metadata_columns if col in combined.columns]
        if columns_to_drop:
            combined = combined.drop(columns=columns_to_drop)
            print(f"Dropped TNT metadata columns after merge: {columns_to_drop}")

        # Normalize dtypes for booleans possibly introduced by appended data
        for col in combined.columns:
            if combined[col].dtype == "bool":
                combined[col] = combined[col].astype(int)

        dataset = combined

        # Quick overview
        if "BallScent" in dataset.columns:
            counts = dataset["BallScent"].value_counts(dropna=False).to_dict()
            print(f"  BallScent counts (including Ctrl): {counts}")
        print(f"  Combined dataset shape: {dataset.shape}")
    else:
        print("No additional Ctrl summary datasets found; proceeding with main dataset only.")

    # Add test mode sampling for faster debugging
    if test_mode and len(dataset) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} random rows from {len(dataset)} total rows")
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: Dataset reduced to {len(dataset)} rows")
        print(f"🧪 TEST MODE: Ball types in sample: {sorted(dataset['BallScent'].unique())}")

    # Normalize BallScent labels to canonical factorial names
    dataset = normalize_ball_scent_labels(dataset, group_col="BallScent")

    # Filter to exactly the four conditions shown in the figure.
    # Pre-exposed = CtrlScent (ball pre-exposed to fly odors); NewScent → "New + Pre-exposed" is excluded.
    allowed = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]
    initial_shape = dataset.shape
    dataset = dataset[dataset["BallScent"].isin(allowed)].copy()
    print(f"\n📊 Filtered to allowed ball scents: {allowed}")
    print(f"   Shape: {initial_shape} -> {dataset.shape}")
    print(f"   Remaining conditions: {sorted(dataset['BallScent'].unique())}")

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

    # Check for existing plot files for this metric
    output_dir = Path(output_dir)

    # Check for continuous metric plots (Mann-Whitney style)
    continuous_plot_patterns = [f"*{metric}*.pdf", f"*{metric}*.png", f"{metric}_*.pdf", f"{metric}_*.png"]

    # Check for binary metric plots
    binary_plot_patterns = [f"{metric}_binary_analysis.pdf", f"{metric}_binary_analysis.png"]

    all_patterns = continuous_plot_patterns + binary_plot_patterns

    for pattern in all_patterns:
        existing_plots = list(output_dir.glob(pattern))
        if existing_plots:
            print(f"  📄 Skipping {metric}: Found existing plot(s) {[p.name for p in existing_plots]}")
            return True

    # Also check in binary_analysis subdirectory
    binary_subdir = output_dir / "binary_analysis"
    if binary_subdir.exists():
        for pattern in binary_plot_patterns:
            existing_plots = list(binary_subdir.glob(pattern))
            if existing_plots:
                print(f"  📄 Skipping {metric}: Found existing binary plot(s) {[p.name for p in existing_plots]}")
                return True

    return False


def get_nan_annotations(data, metrics, y_col="BallScent"):
    """
    Generate annotations for ball types that have NaN values in specific metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the metrics
    metrics : list
        List of metric column names to check
    y_col : str
        Column name for grouping (e.g., ball type)

    Returns:
    --------
    dict
        Dictionary mapping metric names to dictionaries of BallScent: nan_info
    """
    nan_annotations = {}

    for metric in metrics:
        if metric not in data.columns:
            continue

        metric_annotations = {}

        for BallScent in data[y_col].unique():
            BallScent_data = data[data[y_col] == BallScent][metric]
            total_count = len(BallScent_data)
            nan_count = BallScent_data.isnull().sum()

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                metric_annotations[BallScent] = {
                    "nan_count": nan_count,
                    "total_count": total_count,
                    "nan_percentage": nan_percentage,
                    "annotation": f"({nan_count}/{total_count} NaN)",
                }

        if metric_annotations:
            nan_annotations[metric] = metric_annotations

    return nan_annotations


def analyze_binary_metrics(data, binary_metrics, y="BallScent", output_dir=None, overwrite=True):
    """
    Analyze binary metrics using appropriate statistical tests and visualizations.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (e.g., ball type)
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

    # Determine control group (New)
    control_BallScent = "New"
    if control_BallScent not in data[y].unique():
        print(f"Warning: Control ball type '{control_BallScent}' not found in data")
        # Use the most common ball type as control
        control_BallScent = data[y].value_counts().index[0]

    print(f"Using control group: {control_BallScent}")

    for metric in binary_metrics:
        print(f"\nAnalyzing binary metric: {metric}")

        # Check if we should skip this metric
        if output_dir and should_skip_metric(metric, output_dir, overwrite):
            continue

        # Create contingency table
        contingency = pd.crosstab(data[y], data[metric], margins=True)
        print(f"Contingency table for {metric}:")
        print(contingency)

        # Calculate proportions for each ball type
        proportions = data.groupby(y)[metric].agg(["count", "sum", "mean"]).round(4)
        proportions["proportion"] = proportions["mean"]
        proportions["n_success"] = proportions["sum"]
        proportions["n_total"] = proportions["count"]
        print(f"\nProportions for {metric}:")
        print(proportions[["n_success", "n_total", "proportion"]])

        # Statistical tests for each ball type vs control
        pvals = []
        temp_results = []

        for BallScent in data[y].unique():
            if BallScent == control_BallScent:
                continue

            # Get data for this comparison
            test_data = data[data[y] == BallScent][metric].dropna()
            control_data_subset = data[data[y] == control_BallScent][metric].dropna()

            if len(test_data) == 0 or len(control_data_subset) == 0:
                continue

            # Create 2x2 contingency table for Fisher's exact test
            test_success = test_data.sum()
            test_total = len(test_data)
            control_success = control_data_subset.sum()
            control_total = len(control_data_subset)

            # Fisher's exact test
            try:
                # Create contingency table for Fisher's exact test
                table = [[test_success, test_total - test_success], [control_success, control_total - control_success]]

                # Run Fisher's exact test
                fisher_result = fisher_exact(table)

                # Extract results - scipy returns (odds_ratio, p_value)
                odds_ratio = fisher_result[0]  # type: ignore
                p_value = fisher_result[1]  # type: ignore

                # Effect size (difference in proportions)
                test_prop = test_success / test_total if test_total > 0 else 0
                control_prop = control_success / control_total if control_total > 0 else 0
                effect_size = test_prop - control_prop

                pvals.append(p_value)
                temp_results.append(
                    {
                        "metric": metric,
                        "BallScent": BallScent,
                        "control": control_BallScent,
                        "test_success": test_success,
                        "test_total": test_total,
                        "test_proportion": test_prop,
                        "control_success": control_success,
                        "control_total": control_total,
                        "control_proportion": control_prop,
                        "effect_size": effect_size,
                        "odds_ratio": odds_ratio,
                        "p_value_raw": p_value,
                        "test_type": "Fisher exact",
                    }
                )

                print(f"  {BallScent} vs {control_BallScent}: OR={odds_ratio:.3f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error testing {BallScent} vs {control_BallScent}: {e}")

        if pvals:
            # Apply FDR correction
            rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(pvals, alpha=0.05, method="fdr_bh")

            # Update results with FDR correction
            for i, result in enumerate(temp_results):
                result["p_value"] = pvals_corrected[i]  # FDR-corrected p-value
                result["significant"] = rejected[i]

                # Determine significance level based on FDR-corrected p-values
                if pvals_corrected[i] < 0.001:
                    sig_level = "***"
                elif pvals_corrected[i] < 0.01:
                    sig_level = "**"
                elif pvals_corrected[i] < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                result["sig_level"] = sig_level
                results.append(result)

            # Print summary
            n_significant = sum(rejected)
            print(f"  Significant: {n_significant}/{len(pvals)}")
        else:
            print(f"  No valid comparisons for {metric}")

        # Create visualization
        if output_dir:
            create_binary_metric_plot(data, metric, y, output_dir, control_BallScent)

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    return results_df


def create_binary_metric_plot(data, metric, y, output_dir, control_BallScent=None):
    """Create a visualization for binary metrics showing proportions"""

    # Calculate proportions
    prop_data = data.groupby(y)[metric].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_success"] = prop_data["sum"]
    prop_data["n_total"] = prop_data["count"]

    # Sort by proportion (descending) - higher proportions first
    prop_data = prop_data.sort_values("proportion", ascending=False).reset_index(drop=True)

    # Use a categorical color palette for ball types
    n_ball_types = len(prop_data)
    palette = sns.color_palette("Set1", n_colors=n_ball_types)
    colors = palette[:n_ball_types]

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

    # Create the plot - make it taller like the jitterboxplots
    n_ball_types = len(prop_data)
    fig_height = max(8, 0.6 * n_ball_types + 4)  # Dynamic height based on number of ball types
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))

    # Plot 1: Proportion with confidence intervals (horizontal bars)
    y_positions = range(len(prop_data))
    bars = ax1.barh(y_positions, prop_data["proportion"], color=colors, alpha=0.7)

    # Add error bars (ensure no negative values)
    yerr_lower = np.maximum(0, prop_data["proportion"] - prop_data["lower"])
    yerr_upper = np.maximum(0, prop_data["upper"] - prop_data["proportion"])
    xerr = [yerr_lower, yerr_upper]
    ax1.errorbar(prop_data["proportion"], y_positions, xerr=xerr, fmt="none", color="black", capsize=5)

    # Add sample sizes on bars
    for i, (bar, row) in enumerate(zip(bars, prop_data.itertuples())):
        width = bar.get_width()
        ax1.text(
            width + 0.01, bar.get_y() + bar.get_height() / 2.0, f"n={row.n_total}", ha="left", va="center", fontsize=8
        )

    ax1.set_ylabel("Ball Type")
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

    ax2.set_ylabel("Ball Type")
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
    print(f"  Sorted by proportion (descending): {prop_data['proportion'].tolist()}")


def main(overwrite=True, test_mode=False):
    """Main function to run permutation test plots for all metrics.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample 200 rows for fast testing.

    Performance improvements:
    - Fixed pandas chained assignment warnings for faster DataFrame operations
    - Added test mode for faster debugging with limited data and metrics
    - Added timing information for each metric to identify bottlenecks
    - Batch fillna operations for better performance
    - FDR correction for proper multiple testing correction
    - Permutation tests (10,000 permutations) for robust statistical inference
    """
    print(f"Starting permutation test analysis for ball scents experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/Ball_scents/Plots")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Clean up any potential old outputs to avoid confusion
    print("Ensuring clean output directory structure...")
    condition_subdir = "permutation_BallScents"
    condition_dir = base_output_dir / condition_subdir
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {condition_dir}")

    # Also create the main directory for the combined statistics
    print(f"All outputs will be saved under: {base_output_dir}")

    # Load the dataset
    print("Loading ball scents dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Check what columns we have
    print(f"\nDataset columns: {list(dataset.columns)}")

    # Check for BallScent column
    if "BallScent" not in dataset.columns:
        print("❌ BallScent column not found in dataset!")
        print("Available columns that might contain ball type info:")
        potential_cols = [
            col for col in dataset.columns if any(word in col.lower() for word in ["ball", "type", "condition"])
        ]
        print(potential_cols)
        return

    print(f"✅ BallScent column found")
    print(f"Ball types in dataset: {sorted(dataset['BallScent'].unique())}")
    print("Sample sizes per BallScent:")
    ball_scent_counts = dataset["BallScent"].value_counts()
    for ball_scent, count in ball_scent_counts.items():
        print(f"  {ball_scent}: {count} samples")

    # Define core metrics to analyze (simplified approach)
    print(f"🎯 Using predefined core metrics for simplified analysis...")

    # Core base metrics from correlation analysis
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
    print(f"📊 Checking which core metrics exist in the dataset...")
    available_metrics = []

    # Check base metrics (exclude unwanted ones immediately)
    for metric in base_metrics:
        if metric in dataset.columns:
            # Check if metric should be excluded
            should_exclude = any(pattern in metric.lower() for pattern in excluded_patterns)
            if should_exclude:
                print(f"  ❌ {metric} (excluded due to pattern)")
                continue
            available_metrics.append(metric)
            print(f"  ✓ {metric}")
        else:
            print(f"  ❌ {metric} (not found)")

    # Check pattern-based metrics (exclude unwanted ones immediately)
    print(f"📊 Searching for pattern-based metrics...")
    for pattern in additional_patterns:
        pattern_cols = [col for col in dataset.columns if pattern in col.lower()]
        for col in pattern_cols:
            if col not in available_metrics and col != "BallScent":  # Avoid duplicates and metadata
                # Check if metric should be excluded
                should_exclude = any(excl_pattern in col.lower() for excl_pattern in excluded_patterns)
                if should_exclude:
                    print(f"  ❌ {col} (excluded due to excluded pattern)")
                    continue
                available_metrics.append(col)
                print(f"  ✓ {col} (matches pattern '{pattern}')")

    print(f"📊 Found {len(available_metrics)} metrics after exclusion filtering")

    # Categorize found metrics (now working only with pre-filtered metrics)
    def categorize_metrics(col):
        """Categorize metrics for appropriate analysis"""
        if col not in dataset.columns:
            return None

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(dataset[col]):
            return "non_numeric"

        # Get non-NaN values for analysis
        non_nan_values = dataset[col].dropna()

        # If too few non-NaN values, still categorize but flag it
        if len(non_nan_values) < 3:
            return "insufficient_data"

        unique_vals = non_nan_values.unique()

        # Check for binary metrics (only 0s and 1s, allowing for float representation)
        if len(unique_vals) == 2:
            vals_set = set(unique_vals)
            # Handle both int and float representations of 0 and 1
            if vals_set == {0, 1} or vals_set == {0.0, 1.0} or vals_set == {0, 1.0} or vals_set == {0.0, 1}:
                return "binary"

        # Check for columns with very few unique values (might be categorical)
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
            # Still include these but with a warning
            non_nan_count = dataset[col].count()
            total_count = len(dataset)
            print(f"  ⚠️  {col}: Only {non_nan_count}/{total_count} non-NaN values, including anyway")
            # Determine if it should be binary or continuous based on available data
            non_nan_values = dataset[col].dropna().unique()
            if len(non_nan_values) >= 2 and set(non_nan_values) == {0, 1}:
                binary_metrics.append(col)
            else:
                continuous_metrics.append(col)
        else:
            excluded_metrics.append((col, category))

    print(f"\n📊 EFFICIENT METRICS ANALYSIS SUMMARY:")
    print(f"=" * 60)
    print(f"✅ Metrics filtering applied early for efficiency - excluded patterns filtered before categorization")
    print(f"Found {len(continuous_metrics)} continuous metrics for Mann-Whitney analysis:")
    for i, metric in enumerate(continuous_metrics, 1):
        print(f"  {i:2d}. {metric}")

    print(f"\nFound {len(binary_metrics)} binary metrics for proportion analysis:")
    for i, metric in enumerate(binary_metrics, 1):
        print(f"  {i:2d}. {metric}")

    if excluded_metrics:
        print(f"\nAdditionally excluded {len(excluded_metrics)} metrics during categorization:")
        for metric, reason in excluded_metrics:
            if metric in dataset.columns:
                unique_count = len(dataset[metric].dropna().unique())
                dtype = dataset[metric].dtype
                print(f"  - {metric} ({reason}, {dtype}, {unique_count} unique values)")

    # Use continuous metrics for the Mann-Whitney analysis
    metric_cols = continuous_metrics

    # If in test mode, limit to first 3 metrics and show more timing info
    if test_mode:
        original_count = len(continuous_metrics)
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2] if binary_metrics else []
        print(f"🧪 TEST MODE: Reduced from {original_count} to {len(continuous_metrics)} continuous metrics")
        print(f"🧪 TEST MODE: Limited to {len(binary_metrics)} binary metrics")
        print(f"🧪 TEST MODE: Processing metrics: {continuous_metrics}")
        print(f"🧪 TEST MODE: With dataset sample size: {len(dataset)} rows")
        print(f"🧪 TEST MODE: This should complete much faster for debugging")

    # Clean the dataset (already done in load_and_clean_dataset)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique ball types: {len(dataset['BallScent'].unique())}")

    # Validate required columns
    required_cols = ["BallScent"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Generate plots for ball types analysis
    print(f"\n{'='*60}")
    print(f"Processing Ball Types Analysis...")
    print("=" * 60)

    # Use all dataset for ball types
    filtered_data = dataset
    print(f"Dataset shape: {filtered_data.shape}")

    if len(filtered_data) == 0:
        print(f"No data for ball types analysis. Skipping...")
        return

    # Create output directory
    output_dir = base_output_dir / condition_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ball types in dataset: {sorted(filtered_data['BallScent'].unique())}")
    print(f"Using simplified color palette for visualization")

    # 1. Process continuous metrics with Mann-Whitney U tests
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests) ---")
        # Filter to only continuous metrics + required columns
        continuous_data = filtered_data[["BallScent"] + continuous_metrics].copy()

        print(f"Generating Mann-Whitney plots for ball types...")
        print(f"Continuous metrics dataset shape: {continuous_data.shape}")
        print(f"Output directory: {output_dir}")

        # Filter out metrics that should be skipped
        if not overwrite:
            print(f"📄 Checking for existing plots...")
            metrics_to_process = []
            for metric in continuous_metrics:
                if not should_skip_metric(metric, output_dir, overwrite):
                    metrics_to_process.append(metric)

            if len(metrics_to_process) < len(continuous_metrics):
                skipped_count = len(continuous_metrics) - len(metrics_to_process)
                print(f"📄 Skipped {skipped_count} metrics with existing plots")
                print(f"📊 Processing {len(metrics_to_process)} metrics: {metrics_to_process}")

            if not metrics_to_process:
                print(f"📄 All continuous metrics already have plots, skipping...")
            else:
                continuous_data = continuous_data[["BallScent"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            # Check for NaN annotations
            print(f"📊 Checking for NaN values in {len(metrics_to_process)} metrics...")
            nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "BallScent")
            if nan_annotations:
                print(f"📋 NaN values detected in continuous metrics:")
                for metric, BallScent_info in nan_annotations.items():
                    print(f"  {metric}: {len(BallScent_info)} ball types have NaN values")
                    for BallScent, info in BallScent_info.items():
                        print(f"    - {BallScent}: {info['annotation']}")

            print(f"🚀 Starting Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics...")
            print(f"📊 Dataset shape for analysis: {continuous_data.shape}")
            print(f"📊 Ball types: {sorted(continuous_data['BallScent'].unique())}")
            print(f"📊 Sample sizes per ball type:")
            ball_type_counts = continuous_data["BallScent"].value_counts()
            for ball_type, count in ball_type_counts.items():
                print(f"    {ball_type}: {count} samples")

            print(f"⏳ This may take several minutes depending on the number of metrics and samples...")

            import time

            start_time = time.time()

            # Determine control ball type - use "New" as control
            control_BallScent = "New"

            print(f"🚀 Starting permutation test analysis for {len(metrics_to_process)} continuous metrics...")
            print(f"📊 Control group: {control_BallScent}")
            print(f"📊 Test groups: {[bs for bs in continuous_data['BallScent'].unique() if bs != control_BallScent]}")

            # Use the permutation test function with FDR correction
            stats_df = generate_BallScent_permutation_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="BallScent",
                control_BallScent=control_BallScent,
                hue="BallScent",
                palette="Set2",  # Palette not used (we use fixed colors)
                fig_width_mm=64,
                fig_height_mm=89,
                font_size_ticks=9,
                font_size_labels=11,
                font_size_legend=10,
                font_size_annotations=12,
                output_dir=str(output_dir),
                alpha=0.05,  # Significance level after FDR correction
                n_permutations=10000,  # Number of permutations
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"✅ Permutation test analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
            )

    # 2. Process binary metrics with Fisher's exact tests
    if binary_metrics:
        print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
        # Filter to only binary metrics + required columns
        binary_data = filtered_data[["BallScent"] + binary_metrics].copy()

        print(f"Analyzing binary metrics for ball types...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        # Check for NaN annotations in binary metrics
        nan_annotations = get_nan_annotations(binary_data, binary_metrics, "BallScent")
        if nan_annotations:
            print(f"📋 NaN values detected in binary metrics:")
            for metric, BallScent_info in nan_annotations.items():
                print(f"  {metric}: {len(BallScent_info)} ball types have NaN values")
                for BallScent, info in BallScent_info.items():
                    print(f"    - {BallScent}: {info['annotation']}")

        binary_output_dir = output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            y="BallScent",
            output_dir=binary_output_dir,
            overwrite=overwrite,
        )

    print(f"Unique ball types: {len(filtered_data['BallScent'].unique())}")

    print(f"\n{'='*60}")
    print("✅ Ball types analysis complete! Check output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate permutation test jitterboxplots for ball scents experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_permutation_ballscents.py                    # Overwrite existing plots
  python run_permutation_ballscents.py --no-overwrite     # Skip existing plots
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
