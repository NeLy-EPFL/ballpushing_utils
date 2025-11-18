#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for dark olfaction experiments.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Three grouping factors: Light (on/off), Genotype, and BallType
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- Genotype-dependent outline colors
- BallType-dependent line styles (sand=dotted, ctrl=solid)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_mannwhitney_dark_olfaction.py [--overwrite]

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


def generate_dark_olfaction_mannwhitney_plots(
    data,
    metrics,
    grouping_cols=["Light", "Genotype", "BallType"],
    control_group=None,
    genotype_palette=None,
    figsize=(15, 10),
    output_dir="mann_whitney_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
    split_by_light=True,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between groups.
    Applies FDR correction across all comparisons for each metric.
    Creates side-by-side subplots for different Light conditions.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        grouping_cols (list): List of column names for grouping factors. Default is ["Light", "Genotype", "BallType"].
        control_group (dict): Dictionary specifying control group values for each grouping column.
                             Example: {"Light": "off", "Genotype": "TNTxEmptyGal4", "BallType": "ctrl"}
        genotype_palette (dict): Color palette mapping genotypes to colors. Default is None (auto-generate).
        figsize (tuple, optional): Size of each figure. Default is (15, 10).
        output_dir: Directory to save the plots. Default is "mann_whitney_plots".
        fdr_method (str): Method for FDR correction. Default is "fdr_bh" (Benjamini-Hochberg).
        alpha (float): Significance level after FDR correction. Default is 0.05.
        split_by_light (bool): If True, create side-by-side subplots for each Light condition. Default is True.

    Returns:
        pd.DataFrame: Statistics table with Mann-Whitney U test results and FDR correction.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default control group if not specified
    if control_group is None:
        control_group = {"Light": "off", "Genotype": "TNTxEmptyGal4", "BallType": "ctrl"}

    # Set default genotype palette if not specified
    if genotype_palette is None:
        unique_genotypes = data["Genotype"].unique()
        genotype_palette = dict(zip(unique_genotypes, sns.color_palette("Set1", n_colors=len(unique_genotypes))))

    # BallType line styles
    balltype_linestyles = {"ctrl": "-", "sand": ":"}  # solid for ctrl, dotted for sand

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating Mann-Whitney jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric] + grouping_cols)

        # Create combined grouping column for easier analysis
        plot_data["Group"] = plot_data.apply(lambda row: f"{row['Light']}_{row['Genotype']}_{row['BallType']}", axis=1)

        # Get all groups except control
        control_group_name = f"{control_group['Light']}_{control_group['Genotype']}_{control_group['BallType']}"
        all_groups = [g for g in plot_data["Group"].unique() if g != control_group_name]

        # Check if control exists
        if control_group_name not in plot_data["Group"].unique():
            print(f"Warning: Control group '{control_group_name}' not found for metric {metric}")
            continue

        control_vals = plot_data[plot_data["Group"] == control_group_name][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue

        control_median = control_vals.median()

        # Perform Mann-Whitney U tests for all groups vs control
        pvals = []
        test_results = []

        for group in all_groups:
            test_vals = plot_data[plot_data["Group"] == group][metric].dropna()

            if len(test_vals) == 0:
                continue

            try:
                stat, pval = mannwhitneyu(test_vals, control_vals, alternative="two-sided")
                # Determine direction of effect
                test_median = test_vals.median()
                direction = "increased" if test_median > control_median else "decreased"
                effect_size = test_median - control_median
            except Exception as e:
                print(f"Error in Mann-Whitney test for {group} vs {control_group_name}: {e}")
                pval = 1.0
                direction = "none"
                effect_size = 0.0
                stat = np.nan

            # Parse group name
            light, genotype, balltype = group.split("_", 2)

            pvals.append(pval)
            test_results.append(
                {
                    "Group": group,
                    "Light": light,
                    "Genotype": genotype,
                    "BallType": balltype,
                    "Metric": metric,
                    "Control": control_group_name,
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
            "Group": control_group_name,
            "Light": control_group["Light"],
            "Genotype": control_group["Genotype"],
            "BallType": control_group["BallType"],
            "Metric": metric,
            "Control": control_group_name,
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

        # Sort groups using the custom sorting key
        sorted_results = sorted(all_results, key=sort_key)
        sorted_groups = [r["Group"] for r in sorted_results]

        # Update plot data with sorted categories
        plot_data["Group"] = pd.Categorical(plot_data["Group"], categories=sorted_groups, ordered=True)
        plot_data = plot_data.sort_values(by=["Group"])

        # Store stats for output
        for result in test_results:  # Exclude control from final stats
            all_stats.append(result)

        # Get unique Light conditions and create side-by-side subplots if split_by_light=True
        if split_by_light:
            light_conditions = sorted(plot_data["Light"].unique())
            n_light = len(light_conditions)

            # Calculate appropriate figure height based on number of categories per Light condition
            # We'll estimate by dividing total groups by number of light conditions
            n_categories_per_light = len(sorted_groups) // max(n_light, 1)
            fig_height = max(8, n_categories_per_light * 1.0 + 4)  # +4 for margins and title
            fig_width = (figsize[0] + 4) * n_light / 2  # Scale width for multiple subplots, extra for legends

            # Create subplots: one per Light condition
            fig, axes = plt.subplots(1, n_light, figsize=(fig_width, fig_height))
            if n_light == 1:
                axes = [axes]  # Make iterable if only one subplot
        else:
            # Original single-plot behavior
            n_categories = len(sorted_groups)
            fig_height = max(8, n_categories * 1.0 + 4)  # +4 for margins and title
            fig_width = figsize[0] + 4  # Extra width for external legend

            # Create the plot with adjusted size
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            axes = [ax]  # Make iterable for consistent loop handling
            light_conditions = [None]  # Dummy for loop

        # Loop over Light conditions (or just once if not splitting)
        for light_idx, light_condition in enumerate(light_conditions):
            ax = axes[light_idx]

            # Filter data for this Light condition if splitting
            if split_by_light:
                # Filter plot_data and sorted_groups to only include this light condition
                light_plot_data = plot_data[plot_data["Light"] == light_condition].copy()
                light_sorted_groups = [g for g in sorted_groups if g.startswith(f"{light_condition}_")]

                # Update categorical with only the groups for this light condition
                light_plot_data["Group"] = pd.Categorical(
                    light_plot_data["Group"], categories=light_sorted_groups, ordered=True
                )
                light_plot_data = light_plot_data.sort_values(by=["Group"])
            else:
                # Use all data
                light_plot_data = plot_data
                light_sorted_groups = sorted_groups

            # Add colored backgrounds only for significant results
            y_positions = range(len(light_sorted_groups))
            for i, group in enumerate(light_sorted_groups):
                result = next((r for r in all_results if r["Group"] == group), None)
                if not result:
                    continue

                # Add background color only for significant results
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

            # Draw boxplots with genotype-colored outlines and balltype-dependent line styles
            for group in light_sorted_groups:
                group_data = light_plot_data[light_plot_data["Group"] == group]
                if group_data.empty:
                    continue

                is_control = group == control_group_name

                # Parse group to get genotype and balltype
                light, genotype, balltype = group.split("_", 2)

                # Get color and linestyle
                edge_color = genotype_palette.get(genotype, "black")
                linestyle = balltype_linestyles.get(balltype, "-")

                if is_control:
                    # Control group styling - red color but preserve balltype linestyle
                    sns.boxplot(
                        data=group_data,
                        x=metric,
                        y="Group",
                        showfliers=False,
                        width=0.5,
                        ax=ax,
                        boxprops=dict(facecolor="none", edgecolor="red", linewidth=2, linestyle=linestyle),
                        medianprops=dict(color="red", linewidth=2),
                        whiskerprops=dict(color="red", linewidth=1.5, linestyle=linestyle),
                        capprops=dict(color="red", linewidth=1.5),
                    )
                else:
                    # Test group styling - genotype-colored outlines with balltype linestyles
                    sns.boxplot(
                        data=group_data,
                        x=metric,
                        y="Group",
                        showfliers=False,
                        width=0.5,
                        ax=ax,
                        boxprops=dict(facecolor="none", edgecolor=edge_color, linewidth=1, linestyle=linestyle),
                        medianprops=dict(color=edge_color, linewidth=1.5),
                        whiskerprops=dict(color=edge_color, linewidth=1, linestyle=linestyle),
                        capprops=dict(color=edge_color, linewidth=1),
                    )

            # Overlay stripplot for jitter with genotype colors
            # Create color mapping for each row based on genotype
            strip_colors = light_plot_data["Genotype"].map(genotype_palette)

            sns.stripplot(
                data=light_plot_data,
                x=metric,
                y="Group",
                dodge=False,
                alpha=0.7,
                jitter=True,
                palette=strip_colors.unique(),
                hue="Genotype",
                size=6,
                ax=ax,
                legend=False,  # We'll create custom legend
            )

            # Add significance annotations
            yticklabels = [label.get_text() for label in ax.get_yticklabels()]
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

            for i, group in enumerate(light_sorted_groups):
                result = next((r for r in all_results if r["Group"] == group), None)
                if not result:
                    continue

                # Add significance annotation
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
            x_max = light_plot_data[metric].quantile(0.99)
            ax.set_xlim(left=None, right=x_max * 1.2)  # Extra space for annotations

            # Formatting
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.tick_params(axis="y", which="major", labelsize=10)
            ax.set_xlabel(metric, fontsize=14)
            ax.set_ylabel("Group (Light_Genotype_BallType)", fontsize=14)

            # Set subplot title if splitting by light
            if split_by_light:
                ax.set_title(f"Light: {light_condition}", fontsize=14)
            else:
                ax.set_title(
                    f"Mann-Whitney U Test: {metric} by Light × Genotype × BallType (FDR corrected)", fontsize=16
                )

            ax.grid(axis="x", alpha=0.3)

            # Create custom legend
            legend_elements = [
                Patch(facecolor="none", edgecolor="red", linestyle="-", linewidth=2, label="Control Group"),
            ]

            # Add genotype colors
            for genotype, color in genotype_palette.items():
                legend_elements.append(Patch(facecolor="none", edgecolor=color, linewidth=1, label=f"{genotype}"))

            # Add balltype linestyles
            legend_elements.append(
                Patch(facecolor="none", edgecolor="black", linestyle="-", linewidth=1, label="BallType: ctrl (solid)")
            )
            legend_elements.append(
                Patch(facecolor="none", edgecolor="black", linestyle=":", linewidth=1, label="BallType: sand (dotted)")
            )

            # Add significance backgrounds
            legend_elements.extend(
                [
                    Patch(facecolor="lightgreen", alpha=0.15, label="Significantly Increased"),
                    Patch(facecolor="lightblue", alpha=0.1, label="Control"),
                    Patch(facecolor="lightcoral", alpha=0.15, label="Significantly Decreased"),
                ]
            )

            ax.legend(
                legend_elements,
                [str(elem.get_label()) for elem in legend_elements],
                fontsize=10,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

        # Add overall title if splitting by light
        if split_by_light:
            fig.suptitle(f"Mann-Whitney U Test: {metric} by Genotype × BallType (FDR corrected)", fontsize=16, y=1.02)

        # Use tight_layout with padding to accommodate external legend
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric}_dark_olfaction_mannwhitney_fdr.pdf"
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
        stats_file = output_dir / "dark_olfaction_mannwhitney_fdr_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "dark_olfaction_mannwhitney_fdr_report.md"
        generate_text_report(stats_df, data, control_group, report_file)
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
    report_lines.append("# Dark Olfaction Mann-Whitney U Test Report")
    report_lines.append("## FDR-Corrected Statistical Analysis Results")
    report_lines.append("")
    control_str = "_".join([control_group["Light"], control_group["Genotype"], control_group["BallType"]])
    report_lines.append(f"**Control group:** {control_str}")
    report_lines.append("")

    # Group by metric
    metrics = stats_df["Metric"].unique()

    for metric in sorted(metrics):
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"## {metric}")
        report_lines.append("")

        # Get control median for reference
        control_median = 0  # Initialize default value
        # Create control group filter
        control_filter = (
            (data["Light"] == control_group["Light"])
            & (data["Genotype"] == control_group["Genotype"])
            & (data["BallType"] == control_group["BallType"])
        )
        control_data = data[control_filter][metric].dropna()

        if len(control_data) > 0:
            control_median = control_data.median()
            control_mean = control_data.mean()
            control_std = control_data.std()
            control_n = len(control_data)

            report_lines.append(
                f"**Control ({control_str}):** median = {control_median:.3f}, mean = {control_mean:.3f} ± {control_std:.3f} (n={control_n})"
            )
            report_lines.append("")
        else:
            report_lines.append(f"**Control ({control_str}):** No data available")
            report_lines.append("")

        # Analyze each group vs control
        significant_increased = []
        significant_decreased = []
        non_significant = []

        for _, row in metric_stats.iterrows():
            group = row["Group"]
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

            description = f'"{group}" (median = {test_median:.3f}, n={test_n})'

            if row["significant_fdr"]:
                if row["direction"] == "increased":
                    significant_increased.append(
                        {
                            "group": group,
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
                            "group": group,
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
                        "group": group,
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
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dark olfaction dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dark olfaction dataset not found at {dataset_path}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    # Handle new metrics with appropriate defaults - batch fillna for better performance
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

    # Fill all new metrics at once for better performance
    fillna_dict = {}
    for metric, default_value in new_metrics_defaults.items():
        if metric in dataset.columns:
            nan_count = dataset[metric].isnull().sum()
            if nan_count > 0:
                fillna_dict[metric] = default_value
                print(f"Will fill {nan_count} NaN values in {metric} with {default_value}")

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

    # Check for continuous metric plots
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


def analyze_binary_metrics(
    data,
    binary_metrics,
    grouping_cols=["Light", "Genotype", "BallType"],
    control_group=None,
    output_dir=None,
    overwrite=True,
):
    """
    Analyze binary metrics using appropriate statistical tests and visualizations.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    grouping_cols : list
        Column names for grouping factors
    control_group : dict
        Dictionary specifying control group values
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

    # Set default control group if not specified
    if control_group is None:
        control_group = {"Light": "off", "Genotype": "TNTxEmptyGal4", "BallType": "ctrl"}

    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined grouping column
    data = data.copy()
    data["Group"] = data.apply(lambda row: f"{row['Light']}_{row['Genotype']}_{row['BallType']}", axis=1)

    control_group_name = f"{control_group['Light']}_{control_group['Genotype']}_{control_group['BallType']}"
    print(f"Using control group: {control_group_name}")

    for metric in binary_metrics:
        print(f"\nAnalyzing binary metric: {metric}")

        # Check if we should skip this metric
        if output_dir and should_skip_metric(metric, output_dir, overwrite):
            continue

        # Calculate proportions for each group
        proportions = data.groupby("Group")[metric].agg(["count", "sum", "mean"]).round(4)
        proportions["proportion"] = proportions["mean"]
        proportions["n_success"] = proportions["sum"]
        proportions["n_total"] = proportions["count"]
        print(f"\nProportions for {metric}:")
        print(proportions[["n_success", "n_total", "proportion"]])

        # Statistical tests for each group vs control
        pvals = []
        temp_results = []

        for group in data["Group"].unique():
            if group == control_group_name:
                continue

            # Get data for this comparison
            test_data = data[data["Group"] == group][metric].dropna()
            control_data_subset = data[data["Group"] == control_group_name][metric].dropna()

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

                # Extract results
                odds_ratio = fisher_result[0]  # type: ignore
                p_value = fisher_result[1]  # type: ignore

                # Effect size (difference in proportions)
                test_prop = test_success / test_total if test_total > 0 else 0
                control_prop = control_success / control_total if control_total > 0 else 0
                effect_size = test_prop - control_prop

                # Parse group name
                light, genotype, balltype = group.split("_", 2)

                pvals.append(p_value)
                temp_results.append(
                    {
                        "metric": metric,
                        "Group": group,
                        "Light": light,
                        "Genotype": genotype,
                        "BallType": balltype,
                        "control": control_group_name,
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

                print(f"  {group} vs {control_group_name}: OR={odds_ratio:.3f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error testing {group} vs {control_group_name}: {e}")

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

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    return results_df


def main(overwrite=True, test_mode=False):
    """Main function to run Mann-Whitney plots for all metrics.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample data for fast testing.
    """
    print(f"Starting Mann-Whitney U test analysis for dark olfaction experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Plots")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Create condition subdirectory
    condition_subdir = "mannwhitney_dark_olfaction"
    condition_dir = base_output_dir / condition_subdir
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {condition_dir}")

    # Load the dataset
    print("Loading dark olfaction dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Check what columns we have
    print(f"\nDataset columns: {list(dataset.columns)}")

    # Check for required grouping columns
    required_cols = ["Light", "Genotype", "BallType"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        print(f"❌ Required columns not found in dataset: {missing_cols}")
        return

    print(f"✅ All required grouping columns found")
    print(f"Light values: {sorted(dataset['Light'].unique())}")
    print(f"Genotype values: {sorted(dataset['Genotype'].unique())}")
    print(f"BallType values: {sorted(dataset['BallType'].unique())}")

    # Define core metrics to analyze
    print(f"🎯 Using predefined core metrics for simplified analysis...")

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
    print(f"📊 Checking which core metrics exist in the dataset...")
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
            if col not in available_metrics and col not in required_cols:
                should_exclude = any(excl_pattern in col.lower() for excl_pattern in excluded_patterns)
                if should_exclude:
                    print(f"  ❌ {col} (excluded due to excluded pattern)")
                    continue
                available_metrics.append(col)
                print(f"  ✓ {col} (matches pattern '{pattern}')")

    print(f"📊 Found {len(available_metrics)} metrics after exclusion filtering")

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

    print(f"\nFound {len(binary_metrics)} binary metrics for proportion analysis:")
    for i, metric in enumerate(binary_metrics, 1):
        print(f"  {i:2d}. {metric}")

    # If in test mode, limit to first 3 metrics
    if test_mode:
        original_count = len(continuous_metrics)
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2] if binary_metrics else []
        print(f"🧪 TEST MODE: Reduced from {original_count} to {len(continuous_metrics)} continuous metrics")
        print(f"🧪 TEST MODE: Limited to {len(binary_metrics)} binary metrics")

    # Clean the dataset
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique groups: {len(dataset.groupby(['Light', 'Genotype', 'BallType']))}")

    # Generate plots
    print(f"\n{'='*60}")
    print(f"Processing Dark Olfaction Analysis...")
    print("=" * 60)

    output_dir = base_output_dir / condition_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set control group
    control_group = {"Light": "off", "Genotype": "TNTxEmptyGal4", "BallType": "ctrl"}

    # Set genotype color palette
    genotype_palette = {
        "TNTxEmptyGal4": "blue",
        "TNTxIR8a": "orange",
    }

    # 1. Process continuous metrics with Mann-Whitney U tests
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests) ---")

        # Filter to only continuous metrics + required columns
        continuous_data = dataset[required_cols + continuous_metrics].copy()

        print(f"Generating Mann-Whitney plots for dark olfaction...")
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

            if not metrics_to_process:
                print(f"📄 All continuous metrics already have plots, skipping...")
            else:
                continuous_data = continuous_data[required_cols + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            print(
                f"🚀 Starting FDR-corrected Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics..."
            )

            import time

            start_time = time.time()

            # Use the FDR-corrected function
            stats_df = generate_dark_olfaction_mannwhitney_plots(
                continuous_data,
                metrics=metrics_to_process,
                grouping_cols=required_cols,
                control_group=control_group,
                genotype_palette=genotype_palette,
                output_dir=str(output_dir),
                fdr_method="fdr_bh",
                alpha=0.05,
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
        binary_data = dataset[required_cols + binary_metrics].copy()

        print(f"Analyzing binary metrics for dark olfaction...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        binary_output_dir = output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            grouping_cols=required_cols,
            control_group=control_group,
            output_dir=binary_output_dir,
            overwrite=overwrite,
        )

    print(f"\n{'='*60}")
    print("✅ Dark olfaction analysis complete! Check output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test jitterboxplots for dark olfaction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_dark_olfaction.py                    # Overwrite existing plots
  python run_mannwhitney_dark_olfaction.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_dark_olfaction.py --test             # Test mode (first 3 metrics)
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
