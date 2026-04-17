#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for Date effect experiments.

This script is adapted from run_mannwhitney_light_conditions.py, but tests the effect of Date (session/date) rather than Light condition.

Usage:
    python run_mannwhitney_date_effect.py [--overwrite]

Arguments:
    --overwrite: If specified, overwrite existing plots. If not specified, skip metrics that already have plots.
"""
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent))  # Also add current directory
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))  # Add Plotting directory
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind, levene
from statsmodels.stats.multitest import multipletests

# --- All helper functions (generate_text_report, load_and_clean_exploration_dataset, should_skip_metric, get_nan_annotations) are copied as in the original script ---
# ...existing code for helper functions...

# --- Main analysis function, adapted to use 'Date' instead of 'Light' ---
def generate_date_effect_mannwhitney_plots(
    data,
    metrics,
    y="Date",
    control_condition=None,
    hue=None,
    palette="Set2",
    figsize=(15, 10),
    output_dir="mann_whitney_date_effect_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between each Date and control (if specified).
    Applies FDR correction across all comparisons for each metric.
    If no control_condition is specified, uses the first sorted Date as control.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dates = sorted(data[y].unique())
    def generate_text_report(stats_df, data, control_condition, report_file):
        """
        Generate a human-readable text report of the statistical results for Date effect.
        """
        report_lines = []
        report_lines.append("# Date Effect Mann-Whitney U Test Report")
        report_lines.append("## FDR-Corrected Statistical Analysis Results")
        report_lines.append("")
        report_lines.append(f"**Control group:** {control_condition}")
        report_lines.append("")

        # Group by metric
        metrics = stats_df["Metric"].unique()

        for metric in sorted(metrics):
            metric_stats = stats_df[stats_df["Metric"] == metric]
            report_lines.append(f"## {metric}")
            report_lines.append("")
            # Get control median for reference
            control_median = 0
            control_data = data[data["Date"] == control_condition][metric].dropna()
            if len(control_data) > 0:
                control_median = control_data.median()
                control_mean = control_data.mean()
                control_std = control_data.std()
                control_n = len(control_data)
                report_lines.append(
                    f"**Control ({control_condition}):** median = {control_median:.3f}, mean = {control_mean:.3f} ± {control_std:.3f} (n={control_n})"
                )
                report_lines.append("")
            else:
                report_lines.append(f"**Control ({control_condition}):** No data available")
                report_lines.append("")
            significant_increased = []
            significant_decreased = []
            non_significant = []
            for _, row in metric_stats.iterrows():
                date_condition = row["Date"]
                p_fdr = row["pval_corrected"]
                sig_level = row["sig_level"]
                test_median = row["test_median"]
                effect_size = row["effect_size"]
                test_n = row["test_n"]
                if control_median != 0:
                    percent_change = (effect_size / control_median) * 100
                else:
                    percent_change = 0
                description = f'"{date_condition}" (median = {test_median:.3f}, n={test_n})'
                if row["significant"]:
                    if row["direction"] == "increased":
                        significant_increased.append({
                            "date_condition": date_condition,
                            "description": description,
                            "effect_size": effect_size,
                            "percent_change": percent_change,
                            "p_fdr": p_fdr,
                            "sig_level": sig_level,
                        })
                    elif row["direction"] == "decreased":
                        significant_decreased.append({
                            "date_condition": date_condition,
                            "description": description,
                            "effect_size": effect_size,
                            "percent_change": percent_change,
                            "p_fdr": p_fdr,
                            "sig_level": sig_level,
                        })
                else:
                    non_significant.append({
                        "date_condition": date_condition,
                        "description": description,
                        "effect_size": effect_size,
                        "percent_change": percent_change,
                        "p_fdr": p_fdr,
                    })
            if significant_increased:
                significant_increased.sort(key=lambda x: x["effect_size"], reverse=True)
                report_lines.append("**Significantly higher values than control:**")
                for item in significant_increased:
                    report_lines.append(
                        f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_fdr']:.4f}{item['sig_level']})"
                    )
                report_lines.append("")
            if significant_decreased:
                significant_decreased.sort(key=lambda x: x["effect_size"])
                report_lines.append("**Significantly lower values than control:**")
                for item in significant_decreased:
                    report_lines.append(
                        f"- {item['description']} showed **{item['percent_change']:+.1f}%** change (p={item['p_fdr']:.4f}{item['sig_level']})"
                    )
                report_lines.append("")
            if non_significant:
                non_significant.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
                report_lines.append("**No significant difference from control:**")
                for item in non_significant:
                    report_lines.append(
                        f"- {item['description']} showed {item['percent_change']:+.1f}% change (p={item['p_fdr']:.4f}, ns)"
                    )
                report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
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
        metrics_with_significant = stats_df[stats_df["significant"]]["Metric"].nunique()
        total_metrics = stats_df["Metric"].nunique()
        report_lines.append(
            f"- **Metrics with significant differences:** {metrics_with_significant}/{total_metrics} ({100*metrics_with_significant/total_metrics:.1f}%)"
        )
        report_lines.append("")
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Generated text report with {len(metrics)} metrics and {total_comparisons} comparisons")
    if control_condition is None:
        control_condition = all_dates[0]
    print(f"Fixed color mapping for dates: using Set1 palette")
    date_colors = {date: sns.color_palette("Set1", n_colors=len(all_dates))[i] for i, date in enumerate(all_dates)}

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating Mann-Whitney jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")
        plot_data = data.dropna(subset=[metric, y])
        date_conditions = [d for d in plot_data[y].unique() if d != control_condition]
        if control_condition not in plot_data[y].unique():
            print(f"Warning: Control date '{control_condition}' not found for metric {metric}")
            continue
        control_vals = plot_data[plot_data[y] == control_condition][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue
        control_median = control_vals.median()
        use_parametric = True
        parametric_reason = []
        central_tendency = "median"
        all_unique_vals = plot_data[metric].dropna().unique()
        is_binary_like = len(all_unique_vals) == 2 and set(all_unique_vals).issubset({0, 1, 0.0, 1.0})
        if is_binary_like:
            use_parametric = False
            parametric_reason.append("binary-like data")
        else:
            central_tendency = "mean" if use_parametric else "median"
        min_sample_size = len(control_vals)
        for d in date_conditions:
            d_vals = plot_data[plot_data[y] == d][metric].dropna()
            min_sample_size = min(min_sample_size, len(d_vals))
        if min_sample_size < 30:
            use_parametric = False
            parametric_reason.append(f"minimum sample size {min_sample_size} < 30")
        n_comparisons = len(date_conditions)
        need_correction = n_comparisons > 1
        test_type = "t-test" if use_parametric else "Mann-Whitney U"
        correction_note = "with FDR correction" if need_correction else "no correction needed"
        if use_parametric:
            print(f"  Using parametric tests ({test_type}) - sample sizes ≥30, continuous data")
        else:
            print(f"  Using non-parametric tests ({test_type}) - {', '.join(parametric_reason)}")
        if need_correction:
            print(f"  Multiple comparisons detected ({n_comparisons}), applying FDR correction")
        else:
            print(f"  Single comparison, no multiple testing correction needed")
        pvals = []
        test_results = []
        for date_condition in date_conditions:
            test_vals = plot_data[plot_data[y] == date_condition][metric].dropna()
            if len(test_vals) == 0:
                continue
            try:
                if use_parametric:
                    levene_stat, levene_p = levene(test_vals, control_vals)
                    equal_var = levene_p > 0.05
                    stat, pval = ttest_ind(test_vals, control_vals, equal_var=equal_var)
                    test_name = f"t-test ({'equal' if equal_var else 'unequal'} var)"
                    central_tendency = "mean"
                else:
                    stat, pval = mannwhitneyu(test_vals, control_vals, alternative="two-sided")
                    test_name = "Mann-Whitney U"
                    central_tendency = "median"
                test_median = test_vals.median()
                test_mean = test_vals.mean()
                control_median = control_vals.median()
                control_mean = control_vals.mean()
                if use_parametric:
                    direction = "increased" if test_mean > control_mean else "decreased"
                    effect_size = test_mean - control_mean
                else:
                    direction = "increased" if test_median > control_median else "decreased"
                    effect_size = test_median - control_median
            except Exception as e:
                print(f"Error in {test_type} for {date_condition} vs {control_condition}: {e}")
                pval = 1.0
                direction = "none"
                effect_size = 0.0
                stat = np.nan
                test_name = f"{test_type} (failed)"
                central_tendency = "median"
            pvals.append(pval)
            test_results.append({
                "Date": date_condition,
                "Metric": metric,
                "Control": control_condition,
                "pval_raw": pval,
                "direction": direction,
                "effect_size": effect_size,
                "test_statistic": stat,
                "test_median": test_vals.median() if len(test_vals) > 0 else np.nan,
                "test_mean": test_vals.mean() if len(test_vals) > 0 else np.nan,
                "control_median": control_median,
                "control_mean": control_vals.mean(),
                "test_n": len(test_vals),
                "control_n": len(control_vals),
                "test_type": test_name,
                "parametric": use_parametric,
                "central_tendency": central_tendency,
            })
        if not pvals:
            print(f"No valid comparisons for metric {metric}")
            continue
        if need_correction:
            rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(pvals, alpha=alpha, method=fdr_method)
            correction_applied = True
        else:
            pvals_corrected = pvals
            rejected = [p < alpha for p in pvals]
            correction_applied = False
        for i, result in enumerate(test_results):
            result["pval_corrected"] = pvals_corrected[i]
            result["significant"] = rejected[i]
            result["correction_applied"] = correction_applied
            if pvals_corrected[i] < 0.001:
                result["sig_level"] = "***"
            elif pvals_corrected[i] < 0.01:
                result["sig_level"] = "**"
            elif pvals_corrected[i] < 0.05:
                result["sig_level"] = "*"
            else:
                result["sig_level"] = "ns"
            if not rejected[i]:
                result["direction"] = "none"
        control_result = {
            "Date": control_condition,
            "Metric": metric,
            "Control": control_condition,
            "pval_raw": 1.0,
            "pval_corrected": 1.0,
            "significant": False,
            "sig_level": "control",
            "direction": "control",
            "effect_size": 0.0,
            "test_statistic": np.nan,
            "test_median": control_median,
            "test_mean": control_vals.mean(),
            "control_median": control_median,
            "control_mean": control_vals.mean(),
            "test_n": len(control_vals),
            "control_n": len(control_vals),
            "test_type": "control",
            "parametric": use_parametric,
            "central_tendency": central_tendency if "central_tendency" in locals() else "median",
            "correction_applied": correction_applied,
        }
        all_results = test_results + [control_result]
        def sort_key(result):
            if result["sig_level"] == "control":
                priority = 2
            elif result["significant"] and result["direction"] == "increased":
                priority = 1
            elif result["significant"] and result["direction"] == "decreased":
                priority = 3
            else:
                priority = 2
            if priority == 3:
                median_sort = result["test_median"]
            else:
                median_sort = -result["test_median"]
            return (priority, median_sort)
        sorted_results = sorted(all_results, key=sort_key)
        sorted_dates = [r["Date"] for r in sorted_results]
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_dates, ordered=True)
        plot_data = plot_data.sort_values(by=[y])
        for result in test_results:
            all_stats.append(result)
        n_categories = len(sorted_dates)
        fig_height = max(8, n_categories * 1.0 + 4)
        fig_width = figsize[0] + 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        colors = [date_colors[d] for d in sorted_dates]
        y_positions = range(len(sorted_dates))
        for i, date_condition in enumerate(sorted_dates):
            result = next((r for r in all_results if r["Date"] == date_condition), None)
            if not result:
                continue
            if result["sig_level"] == "control":
                bg_color = "lightblue"
                alpha_bg = 0.1
            elif result["significant"] and result["direction"] == "increased":
                bg_color = "lightgreen"
                alpha_bg = 0.15
            elif result["significant"] and result["direction"] == "decreased":
                bg_color = "lightcoral"
                alpha_bg = 0.15
            else:
                continue
            ax.axhspan(i - 0.4, i + 0.4, color=bg_color, alpha=alpha_bg, zorder=0)
        for date_condition in sorted_dates:
            condition_data = plot_data[plot_data[y] == date_condition]
            if condition_data.empty:
                continue
            is_control = date_condition == control_condition
            if is_control:
                sns.boxplot(
                    data=condition_data,
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
                sns.boxplot(
                    data=condition_data,
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
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        for i, date_condition in enumerate(sorted_dates):
            result = next((r for r in all_results if r["Date"] == date_condition), None)
            if not result:
                continue
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
        x_max = plot_data[metric].quantile(0.99)
        ax.set_xlim(left=None, right=x_max * 1.2)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel("Date", fontsize=14)
        plt.title(f"Mann-Whitney U Test: {metric} by Date (FDR corrected)", fontsize=16)
        ax.grid(axis="x", alpha=0.3)
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
        plt.tight_layout()
        output_file = output_dir / f"{metric}_date_effect_mannwhitney_fdr.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {output_file}")
        n_significant_raw = sum(1 for r in test_results if r["pval_raw"] < 0.05)
        n_significant_fdr = sum(1 for r in test_results if r["significant"])
        print(f"  Raw significant: {n_significant_raw}/{len(test_results)}")
        print(f"  FDR significant: {n_significant_fdr}/{len(test_results)} (α={alpha})")
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")
    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_file = output_dir / "date_effect_mannwhitney_fdr_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")
        report_file = output_dir / "date_effect_mannwhitney_fdr_report.md"
        generate_text_report(stats_df, data, control_condition, report_file)
        print(f"Text report saved to: {report_file}")
        total_tests = len(stats_df)
        total_significant_raw = (stats_df["pval_raw"] < 0.05).sum()
        total_significant_fdr = stats_df["significant"].sum()
        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Raw significant (p<0.05): {total_significant_raw} ({100*total_significant_raw/total_tests:.1f}%)")
        print(f"FDR significant (q<{alpha}): {total_significant_fdr} ({100*total_significant_fdr/total_tests:.1f}%)")
    return stats_df

# ...existing code for main(), CLI, etc. adapted to use Date instead of Light...
