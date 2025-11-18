#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for ball types experiments.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_mannwhitney_balltypes.py [--overwrite]

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
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import scipy.stats as stats_module
from statsmodels.stats.multitest import multipletests
import time
import argparse


def generate_balltype_mannwhitney_plots(
    data,
    metrics,
    y="BallType",
    control_balltype="ctrl",
    hue=None,
    palette="Set2",
    figsize=(15, 10),
    output_dir="mann_whitney_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between each ball type and control.
    Applies FDR correction across all comparisons for each metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (ball types). Default is "BallType".
        control_balltype (str): Name of the control ball type. Default is "ctrl".
        hue (str, optional): The name of the column for color grouping. Default is None.
        palette: Color palette for the plots (str or dict). Default is "Set2".
        figsize (tuple, optional): Size of each figure. Default is (15, 10).
        output_dir: Directory to save the plots. Default is "mann_whitney_plots".
        fdr_method (str): Method for FDR correction. Default is "fdr_bh" (Benjamini-Hochberg).
        alpha (float): Significance level after FDR correction. Default is 0.05.

    Returns:
        pd.DataFrame: Statistics table with Mann-Whitney U test results and FDR correction.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating Mann-Whitney jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y])

        # Get all ball types except control
        ball_types = [bt for bt in plot_data[y].unique() if bt != control_balltype]

        # Check if control exists
        if control_balltype not in plot_data[y].unique():
            print(f"Warning: Control ball type '{control_balltype}' not found for metric {metric}")
            continue

        control_vals = plot_data[plot_data[y] == control_balltype][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue

        control_median = control_vals.median()  # Calculate control median once

        # Perform Mann-Whitney U tests for all ball types vs control
        pvals = []
        test_results = []

        for ball_type in ball_types:
            test_vals = plot_data[plot_data[y] == ball_type][metric].dropna()

            if len(test_vals) == 0:
                continue

            try:
                stat, pval = mannwhitneyu(test_vals, control_vals, alternative="two-sided")
                # Determine direction of effect
                test_median = test_vals.median()
                control_median = control_vals.median()
                direction = "increased" if test_median > control_median else "decreased"
                effect_size = test_median - control_median
            except Exception as e:
                print(f"Error in Mann-Whitney test for {ball_type} vs {control_balltype}: {e}")
                pval = 1.0
                direction = "none"
                effect_size = 0.0
                stat = np.nan

            pvals.append(pval)
            test_results.append(
                {
                    "BallType": ball_type,
                    "Metric": metric,
                    "Control": control_balltype,
                    "pval_raw": pval,
                    "direction": direction,
                    "effect_size": effect_size,
                    "test_statistic": stat,
                    "test_median": test_vals.median() if len(test_vals) > 0 else np.nan,
                    "control_median": control_median,
                    "test_n": len(test_vals),
                    "control_n": len(control_vals),
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
            "BallType": control_balltype,
            "Metric": metric,
            "Control": control_balltype,
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

        # Create sorting key: significance groups (increased > none > decreased), then by median
        def sort_key(result):
            # Primary sort: significance and direction
            if result["sig_level"] == "control":
                priority = 2  # Controls in middle
            elif result["significant_fdr"] and result["direction"] == "increased":
                priority = 1  # Significant increases at top
            elif result["significant_fdr"] and result["direction"] == "decreased":
                priority = 3  # Significant decreases at bottom
            else:
                priority = 2  # Non-significant in middle

            # Secondary sort: by median (descending for increased/none, ascending for decreased)
            if priority == 3:  # For decreased group, sort by median ascending (most decrease first)
                median_sort = result["test_median"]
            else:  # For increased and none groups, sort by median descending (highest first)
                median_sort = -result["test_median"]

            return (priority, median_sort)

        # Sort ball types using the custom sorting key
        sorted_results = sorted(all_results, key=sort_key)
        sorted_ball_types = [r["BallType"] for r in sorted_results]

        # Update plot data with sorted categories
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_ball_types, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # Store stats for output
        for result in test_results:  # Exclude control from final stats
            all_stats.append(result)

        # Calculate appropriate figure height based on number of categories
        n_categories = len(sorted_ball_types)
        # For ball types (fewer categories), use different sizing than nicknames
        # Minimum 1.0 inches per category for better readability with fewer items
        fig_height = max(8, n_categories * 1.0 + 4)  # +4 for margins and title
        # Increase width to accommodate legend outside
        fig_width = figsize[0] + 4  # Extra width for external legend

        # Create the plot with adjusted size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set up colors for ball types
        if isinstance(palette, dict):
            colors = [palette.get(bt, "gray") for bt in sorted_ball_types]
        elif isinstance(palette, str):
            colors = sns.color_palette(palette, n_colors=len(sorted_ball_types))
        else:
            colors = palette  # Assume it's already a list of colors

        # Add colored backgrounds only for significant results
        y_positions = range(len(sorted_ball_types))
        for i, ball_type in enumerate(sorted_ball_types):
            result = next((r for r in all_results if r["BallType"] == ball_type), None)
            if not result:
                continue

            # Add background color only for significant results (like TNT version)
            if result["sig_level"] == "control":
                bg_color = "lightblue"
                alpha_bg = 0.1
            elif result["significant_fdr"] and result["direction"] == "increased":
                bg_color = "lightgreen"
                alpha_bg = 0.15
            elif result["significant_fdr"] and result["direction"] == "decreased":
                bg_color = "lightcoral"
                alpha_bg = 0.15
            else:
                # Non-significant gets no background (white/transparent)
                continue

            # Add background rectangle
            ax.axhspan(i - 0.4, i + 0.4, color=bg_color, alpha=alpha_bg, zorder=0)

        # Draw boxplots with unfilled styling (like TNT version)
        for ball_type in sorted_ball_types:
            ball_type_data = plot_data[plot_data[y] == ball_type]
            if ball_type_data.empty:
                continue

            is_control = ball_type == control_balltype

            if is_control:
                # Control group styling - red dashed boxes (like TNT version)
                sns.boxplot(
                    data=ball_type_data,
                    x=metric,
                    y=y,
                    showfliers=False,
                    width=0.5,  # Wider for better visibility with fewer categories
                    ax=ax,
                    boxprops=dict(facecolor="none", edgecolor="red", linewidth=2, linestyle="--"),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(color="red", linewidth=1.5, linestyle="--"),
                    capprops=dict(color="red", linewidth=1.5),
                )
            else:
                # Test group styling - black solid boxes (like TNT version)
                sns.boxplot(
                    data=ball_type_data,
                    x=metric,
                    y=y,
                    showfliers=False,
                    width=0.5,  # Wider for better visibility with fewer categories
                    ax=ax,
                    boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1),
                    capprops=dict(color="black", linewidth=1),
                )

        # Overlay stripplot for jitter with ball type colors (like TNT version)
        sns.stripplot(
            data=plot_data,
            x=metric,
            y=y,
            dodge=False,
            alpha=0.7,
            jitter=True,
            palette=colors,
            size=6,  # Larger dots for better visibility
            ax=ax,
        )

        # Add significance annotations
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        for i, ball_type in enumerate(sorted_ball_types):
            result = next((r for r in all_results if r["BallType"] == ball_type), None)
            if not result:
                continue

            # Add significance annotation (like TNT version)
            if result["sig_level"] not in ["control", "ns"]:
                yticklocs = ax.get_yticks()
                ax.text(
                    x=ax.get_xlim()[1] + 0.01 * x_range,
                    y=float(yticklocs[i]),
                    s=result["sig_level"],
                    color="red",
                    fontsize=14,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        # Set xlim to accommodate annotations
        x_max = plot_data[metric].quantile(0.99)
        ax.set_xlim(left=None, right=x_max * 1.2)  # Extra space for annotations

        # Formatting (like TNT version)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel("Ball Type", fontsize=14)
        plt.title(f"Mann-Whitney U Test: {metric} by Ball Type (FDR corrected)", fontsize=16)
        ax.grid(axis="x", alpha=0.3)

        # Create custom legend (like TNT version)
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="none", edgecolor="red", linestyle="--", linewidth=2, label="Control Group"),
            Patch(facecolor="none", edgecolor="black", linewidth=1, label="Test Groups"),
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
        output_file = output_dir / f"{metric}_balltype_mannwhitney_fdr.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {output_file}")

        # Print FDR correction summary
        n_significant_raw = sum(1 for r in test_results if r["pval_raw"] < 0.05)
        n_significant_fdr = sum(1 for r in test_results if r["significant_fdr"])
        print(f"  Raw significant: {n_significant_raw}/{len(test_results)}")
        print(f"  FDR significant: {n_significant_fdr}/{len(test_results)} (α={alpha})")

        # Print timing for this metric
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save statistics
        stats_file = output_dir / "balltype_mannwhitney_fdr_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "balltype_mannwhitney_fdr_report.md"
        generate_text_report(stats_df, data, control_balltype, report_file)
        print(f"Text report saved to: {report_file}")

        # Print overall summary
        total_tests = len(stats_df)
        total_significant_raw = (stats_df["pval_raw"] < 0.05).sum()
        total_significant_fdr = stats_df["significant_fdr"].sum()

        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Raw significant (p<0.05): {total_significant_raw} ({100*total_significant_raw/total_tests:.1f}%)")
        print(f"FDR significant (q<{alpha}): {total_significant_fdr} ({100*total_significant_fdr/total_tests:.1f}%)")

    return stats_df


def generate_text_report(stats_df, data, control_balltype, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    control_balltype : str
        Name of the control ball type
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Ball Types Mann-Whitney U Test Report")
    report_lines.append("## FDR-Corrected Statistical Analysis Results")
    report_lines.append("")
    report_lines.append(f"**Control group:** {control_balltype}")
    report_lines.append("")

    # Group by metric
    metrics = stats_df["Metric"].unique()

    for metric in sorted(metrics):
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"## {metric}")
        report_lines.append("")

        # Get control median for reference
        control_median = 0  # Initialize default value
        control_data = data[data["BallType"] == control_balltype][metric].dropna()
        if len(control_data) > 0:
            control_median = control_data.median()
            control_mean = control_data.mean()
            control_std = control_data.std()
            control_n = len(control_data)

            report_lines.append(
                f"**Control ({control_balltype}):** median = {control_median:.3f}, mean = {control_mean:.3f} ± {control_std:.3f} (n={control_n})"
            )
            report_lines.append("")
        else:
            report_lines.append(f"**Control ({control_balltype}):** No data available")
            report_lines.append("")

        # Analyze each ball type vs control
        significant_increased = []
        significant_decreased = []
        non_significant = []

        for _, row in metric_stats.iterrows():
            ball_type = row["BallType"]
            p_fdr = row["pval_fdr"]
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

            if row["significant_fdr"]:
                if row["direction"] == "increased":
                    significant_increased.append(
                        {
                            "ball_type": ball_type,
                            "description": description,
                            "effect_size": effect_size,
                            "percent_change": percent_change,
                            "p_fdr": p_fdr,
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
                            "p_fdr": p_fdr,
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
                        "p_fdr": p_fdr,
                    }
                )

        # Write results
        if significant_increased:
            # Sort by effect size (largest increase first)
            significant_increased.sort(key=lambda x: x["effect_size"], reverse=True)
            report_lines.append("**Significantly higher values than control:**")
            for item in significant_increased:
                report_lines.append(
                    f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_fdr']:.4f}{item['sig_level']})"
                )
            report_lines.append("")

        if significant_decreased:
            # Sort by effect size (largest decrease first, so most negative first)
            significant_decreased.sort(key=lambda x: x["effect_size"])
            report_lines.append("**Significantly lower values than control:**")
            for item in significant_decreased:
                report_lines.append(
                    f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_fdr']:.4f}{item['sig_level']})"
                )
            report_lines.append("")

        if non_significant:
            # Sort by effect size for consistency
            non_significant.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
            report_lines.append("**No significant difference from control:**")
            for item in non_significant:
                report_lines.append(
                    f"- {item['description']} showed {item['percent_change']:+.1f}% change (p={item['p_fdr']:.4f}, ns)"
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
        f"- **Significant results:** {total_significant} ({100*total_significant/total_comparisons:.1f}%)"
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
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_Balltypes/Datasets/250815_17_summary_ball_types_Data/summary/pooled_summary.feather"

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

    # Add test mode sampling for faster debugging
    if test_mode and len(dataset) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} random rows from {len(dataset)} total rows")
        dataset = dataset.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: Dataset reduced to {len(dataset)} rows")
        print(f"🧪 TEST MODE: Ball types in sample: {sorted(dataset['BallType'].unique())}")

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


def get_nan_annotations(data, metrics, y_col="BallType"):
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
        Dictionary mapping metric names to dictionaries of balltype: nan_info
    """
    nan_annotations = {}

    for metric in metrics:
        if metric not in data.columns:
            continue

        metric_annotations = {}

        for balltype in data[y_col].unique():
            balltype_data = data[data[y_col] == balltype][metric]
            total_count = len(balltype_data)
            nan_count = balltype_data.isnull().sum()

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                metric_annotations[balltype] = {
                    "nan_count": nan_count,
                    "total_count": total_count,
                    "nan_percentage": nan_percentage,
                    "annotation": f"({nan_count}/{total_count} NaN)",
                }

        if metric_annotations:
            nan_annotations[metric] = metric_annotations

    return nan_annotations


def analyze_binary_metrics(data, binary_metrics, y="BallType", output_dir=None, overwrite=True):
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

    # Determine control group (ctrl)
    control_balltype = "ctrl"
    if control_balltype not in data[y].unique():
        print(f"Warning: Control ball type '{control_balltype}' not found in data")
        # Use the most common ball type as control
        control_balltype = data[y].value_counts().index[0]

    print(f"Using control group: {control_balltype}")

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

        for balltype in data[y].unique():
            if balltype == control_balltype:
                continue

            # Get data for this comparison
            test_data = data[data[y] == balltype][metric].dropna()
            control_data_subset = data[data[y] == control_balltype][metric].dropna()

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
                        "balltype": balltype,
                        "control": control_balltype,
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

                print(f"  {balltype} vs {control_balltype}: OR={odds_ratio:.3f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error testing {balltype} vs {control_balltype}: {e}")

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

            # Print FDR correction summary
            n_significant_raw = sum(1 for p in pvals if p < 0.05)
            n_significant_fdr = sum(rejected)
            print(f"  Raw significant: {n_significant_raw}/{len(pvals)}")
            print(f"  FDR significant: {n_significant_fdr}/{len(pvals)}")
        else:
            print(f"  No valid comparisons for {metric}")

        # Create visualization
        if output_dir:
            create_binary_metric_plot(data, metric, y, output_dir, control_balltype)

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    return results_df


def create_binary_metric_plot(data, metric, y, output_dir, control_balltype=None):
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
    """Main function to run Mann-Whitney plots for all metrics.

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
    """
    print(f"Starting Mann-Whitney U test analysis for ball types experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Balltypes/Plots/Summary_metrics/BallTypes_Mannwhitney")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Clean up any potential old outputs to avoid confusion
    print("Ensuring clean output directory structure...")
    condition_subdir = "mannwhitney_balltypes"
    condition_dir = base_output_dir / condition_subdir
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {condition_dir}")

    # Also create the main directory for the combined statistics
    print(f"All outputs will be saved under: {base_output_dir}")

    # Load the dataset
    print("Loading ball types dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Check what columns we have
    print(f"\nDataset columns: {list(dataset.columns)}")

    # Check for BallType column
    if "BallType" not in dataset.columns:
        print("❌ BallType column not found in dataset!")
        print("Available columns that might contain ball type info:")
        potential_cols = [
            col for col in dataset.columns if any(word in col.lower() for word in ["ball", "type", "condition"])
        ]
        print(potential_cols)
        return

    print(f"✅ BallType column found")
    print(f"Ball types in dataset: {sorted(dataset['BallType'].unique())}")

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
            if col not in available_metrics and col != "BallType":  # Avoid duplicates and metadata
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
    print(f"Unique ball types: {len(dataset['BallType'].unique())}")

    # Validate required columns
    required_cols = ["BallType"]
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

    print(f"Ball types in dataset: {sorted(filtered_data['BallType'].unique())}")
    print(f"Using simplified color palette for visualization")

    # 1. Process continuous metrics with Mann-Whitney U tests
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests) ---")
        # Filter to only continuous metrics + required columns
        continuous_data = filtered_data[["BallType"] + continuous_metrics].copy()

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
                continuous_data = continuous_data[["BallType"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            # Check for NaN annotations
            print(f"📊 Checking for NaN values in {len(metrics_to_process)} metrics...")
            nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "BallType")
            if nan_annotations:
                print(f"📋 NaN values detected in continuous metrics:")
                for metric, balltype_info in nan_annotations.items():
                    print(f"  {metric}: {len(balltype_info)} ball types have NaN values")
                    for balltype, info in balltype_info.items():
                        print(f"    - {balltype}: {info['annotation']}")

            print(f"🚀 Starting Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics...")
            print(f"📊 Dataset shape for analysis: {continuous_data.shape}")
            print(f"📊 Ball types: {sorted(continuous_data['BallType'].unique())}")
            print(f"📊 Sample sizes per ball type:")
            ball_type_counts = continuous_data["BallType"].value_counts()
            for ball_type, count in ball_type_counts.items():
                print(f"    {ball_type}: {count} samples")

            print(f"⏳ This may take several minutes depending on the number of metrics and samples...")

            import time

            start_time = time.time()

            # Determine control ball type
            control_balltype = "ctrl"
            available_balltypes = continuous_data["BallType"].unique()
            if control_balltype not in available_balltypes:
                print(f"Warning: Control ball type '{control_balltype}' not found in data")
                # Use the most common ball type as control
                control_balltype = str(continuous_data["BallType"].value_counts().index[0])
                print(f"Using most common ball type as control: {control_balltype}")
            else:
                control_balltype = str(control_balltype)  # Ensure it's a string

            print(
                f"🚀 Starting FDR-corrected Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics..."
            )
            print(f"📊 Control group: {control_balltype}")
            print(f"📊 Test groups: {[bt for bt in continuous_data['BallType'].unique() if bt != control_balltype]}")

            # Use the new FDR-corrected function
            stats_df = generate_balltype_mannwhitney_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="BallType",
                control_balltype=control_balltype,
                hue="BallType",
                palette="Set1",  # Use string palette instead of dict for compatibility
                output_dir=str(output_dir),
                fdr_method="fdr_bh",  # Benjamini-Hochberg FDR correction
                alpha=0.05,  # FDR-corrected significance level
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"✅ FDR-corrected Mann-Whitney analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
            )

    # 2. Process binary metrics with Fisher's exact tests
    if binary_metrics:
        print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
        # Filter to only binary metrics + required columns
        binary_data = filtered_data[["BallType"] + binary_metrics].copy()

        print(f"Analyzing binary metrics for ball types...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        # Check for NaN annotations in binary metrics
        nan_annotations = get_nan_annotations(binary_data, binary_metrics, "BallType")
        if nan_annotations:
            print(f"📋 NaN values detected in binary metrics:")
            for metric, balltype_info in nan_annotations.items():
                print(f"  {metric}: {len(balltype_info)} ball types have NaN values")
                for balltype, info in balltype_info.items():
                    print(f"    - {balltype}: {info['annotation']}")

        binary_output_dir = output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            y="BallType",
            output_dir=binary_output_dir,
            overwrite=overwrite,
        )

    print(f"Unique ball types: {len(filtered_data['BallType'].unique())}")

    print(f"\n{'='*60}")
    print("✅ Ball types analysis complete! Check output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test jitterboxplots for ball types experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_balltypes.py                    # Overwrite existing plots
  python run_mannwhitney_balltypes.py --no-overwrite     # Skip existing plots
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
