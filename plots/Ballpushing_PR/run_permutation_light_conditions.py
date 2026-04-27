#!/usr/bin/env python3
"""
-Script to generate permutation-test jitterboxplots for light condition experiments.

This script generates comprehensive permutation-test visualizations with:
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_permutation_light_conditions.py [--overwrite]

Arguments:
    --overwrite: If specified, overwrite existing plots. If not specified, skip metrics that already have plots.
"""
import sys
import time
import argparse
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
from matplotlib.patches import Rectangle, Patch
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, levene
from statsmodels.stats.multitest import multipletests


def generate_light_condition_permutation_plots(
    data,
    metrics,
    y="Light",
    control_condition="on",  # Light ON (control)
    hue=None,
    palette="Set2",
    figsize=(12, 8),
    output_dir="permutation_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
    n_permutations=10000,
):
    """
    Generate jitterboxplots for each metric with permutation tests between each
    light condition and the control. Applies FDR correction across comparisons
    for each metric.

    Parameters:
        data (pd.DataFrame): Dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): Column name for grouping (default: "Light").
        control_condition (str): Control light condition name (default: "off").
        hue (str, optional): Column name for color grouping (default: None).
        palette (str or dict): Color palette for the plots (default: "Set2").
        figsize (tuple): Figure size (width, height).
        output_dir (str or Path): Directory to save plots.
        fdr_method (str): Method for multiple-testing correction (default: "fdr_bh").
        alpha (float): Significance level after FDR correction (default: 0.05).
        n_permutations (int): Number of permutations for permutation tests
            (default: 10000).

    Returns:
        pd.DataFrame: Statistics table with permutation test results and FDR-corrected p-values.
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

    # Match font and PDF settings to magnetblock script for consistency
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Get actual light conditions from data
    all_light_conditions = sorted(data[y].unique())
    print(f"Light conditions found in data: {all_light_conditions}")

    # Map data values to display labels (capitalized)
    value_to_label = {
        "on": "ON",
        "off": "OFF",
    }

    # Fixed ordering: control (on) first, then test (off)
    fixed_order = ["on", "off"]
    print(f"Fixed ordering for light conditions: {fixed_order}")

    # Style mapping: on (control) = solid, off (test) = dashed
    light_condition_styles = {
        "on": "solid",  # Control: solid black lines
        "off": "dashed",  # Test: dashed black lines
    }

    print(f"Style mapping: {light_condition_styles}")

    all_stats = []

    for metric_idx, metric in enumerate(metrics):
        metric_start_time = time.time()
        print(f"Generating permutation jitterboxplot for metric {metric_idx+1}/{len(metrics)}: {metric}")

        plot_data = data.dropna(subset=[metric, y])

        # Get all light conditions except control
        light_conditions = [lc for lc in plot_data[y].unique() if lc != control_condition]

        # Check if control exists
        if control_condition not in plot_data[y].unique():
            print(f"Warning: Control light condition '{control_condition}' not found for metric {metric}")
            continue

        control_vals = plot_data[plot_data[y] == control_condition][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue

        control_median = control_vals.median()  # Calculate control median once

        # Decision heuristic retained (parametric vs non-parametric), but tests below use permutation
        # Criteria: 1) Both groups have >30 samples, 2) Data is continuous (not binary-like)
        use_parametric = True
        parametric_reason = []
        central_tendency = "median"  # Default

        # Check if metric is binary-like (only 0s and 1s)
        all_unique_vals = plot_data[metric].dropna().unique()
        is_binary_like = len(all_unique_vals) == 2 and set(all_unique_vals).issubset({0, 1, 0.0, 1.0})

        if is_binary_like:
            use_parametric = False
            parametric_reason.append("binary-like data")
        else:
            central_tendency = "mean" if use_parametric else "median"

        # Check sample sizes for all conditions
        min_sample_size = len(control_vals)
        for lc in light_conditions:
            lc_vals = plot_data[plot_data[y] == lc][metric].dropna()
            min_sample_size = min(min_sample_size, len(lc_vals))

        if min_sample_size < 30:
            use_parametric = False
            parametric_reason.append(f"minimum sample size {min_sample_size} < 30")

        # No multiple comparison correction needed for 2 conditions
        n_comparisons = len(light_conditions)
        need_correction = False  # Only 2 conditions (on vs off)

        test_type = "permutation"
        correction_note = "no correction needed (only 2 conditions)"

        if use_parametric:
            print(f"  Using parametric tests ({test_type}) - sample sizes ≥30, continuous data")
        else:
            print(f"  Using non-parametric tests ({test_type}) - {', '.join(parametric_reason)}")

        print(f"  Single comparison (on vs off), no multiple testing correction needed")

        # Perform statistical tests for all light conditions vs control
        pvals = []
        test_results = []

        for light_condition in light_conditions:
            test_vals = plot_data[plot_data[y] == light_condition][metric].dropna()

            if len(test_vals) == 0:
                continue

            try:
                # Use permutation test on median difference (robust to non-normality)
                obs_stat = float(np.median(test_vals) - np.median(control_vals))
                combined = np.concatenate([control_vals.values, test_vals.values])
                n_control = len(control_vals)

                perm_diffs = np.empty(n_permutations)
                for i in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_control = combined[:n_control]
                    perm_test = combined[n_control:]
                    perm_diffs[i] = np.median(perm_test) - np.median(perm_control)

                pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))
                stat = obs_stat
                test_name = f"permutation({n_permutations})"
                test_median = test_vals.median()
                test_mean = test_vals.mean()
                control_median = control_vals.median()
                control_mean = control_vals.mean()
                direction = "increased" if obs_stat > 0 else ("decreased" if obs_stat < 0 else "none")
                effect_size = obs_stat
                central_tendency = "median"
            except Exception as e:
                print(f"Error in permutation test for {light_condition} vs {control_condition}: {e}")
                pval = 1.0
                direction = "none"
                effect_size = 0.0
                stat = np.nan
                test_name = f"permutation({n_permutations}) (failed)"
                test_median = np.nan
                test_mean = np.nan
                central_tendency = "median"

            pvals.append(pval)
            test_results.append(
                {
                    "LightCondition": light_condition,
                    "Metric": metric,
                    "Control": control_condition,
                    "pval_raw": pval,
                    "direction": direction,
                    "effect_size": effect_size,
                    "test_statistic": stat,
                    "test_median": test_median,
                    "test_mean": test_mean,
                    "control_median": control_median,
                    "control_mean": control_vals.mean(),
                    "test_n": len(test_vals),
                    "control_n": len(control_vals),
                    "test_type": test_name,
                    "parametric": False,
                    "central_tendency": central_tendency,
                }
            )

        if not pvals:
            print(f"No valid comparisons for metric {metric}")
            continue

        # No FDR correction needed for 2 conditions
        if need_correction:
            rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(pvals, alpha=alpha, method=fdr_method)
            correction_applied = True
        else:
            # No correction needed for single comparison
            pvals_corrected = pvals  # Use raw p-values
            rejected = [p < alpha for p in pvals]  # Simple alpha threshold
            correction_applied = False

        # Update test results with corrected p-values (or raw if no correction)
        for i, result in enumerate(test_results):
            result["pval_corrected"] = pvals_corrected[i]
            result["significant"] = rejected[i]
            result["correction_applied"] = correction_applied

            # Determine significance level based on corrected p-values (or raw if no correction)
            if pvals_corrected[i] < 0.001:
                result["sig_level"] = "***"
            elif pvals_corrected[i] < 0.01:
                result["sig_level"] = "**"
            elif pvals_corrected[i] < 0.05:
                result["sig_level"] = "*"
            else:
                result["sig_level"] = "ns"

            # Override direction if not significant
            if not rejected[i]:
                result["direction"] = "none"

        # Add control to results for sorting
        control_result = {
            "LightCondition": control_condition,
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

        # Use fixed ordering instead of significance-based sorting
        # This ensures consistent ordering across all plots for side-by-side comparison
        sorted_light_conditions = [lc for lc in fixed_order if lc in [r["LightCondition"] for r in all_results]]

        # Update plot data with sorted categories
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_light_conditions, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # Create display labels for plotting (ON/OFF instead of 1/0)
        plot_data[y + "_display"] = plot_data[y].map(value_to_label)
        # Ensure categorical ordering is preserved in labels
        label_order = [value_to_label.get(lc, str(lc)) for lc in sorted_light_conditions]
        plot_data[y + "_display"] = pd.Categorical(plot_data[y + "_display"], categories=label_order, ordered=True)

        # Store stats for output
        for result in test_results:  # Exclude control from final stats
            all_stats.append(result)

        # Calculate appropriate figure width based on number of categories
        n_categories = len(sorted_light_conditions)
        # For light conditions (few categories), use minimum width but allow expansion
        fig_width = max(8, n_categories * 1.0 + 4)  # +4 for margins and title
        fig_height = figsize[1]  # Use requested height (matches magnetblock style)
        fig_width = 100 / 25.4
        fig_height = 125 / 25.4

        # Create the plot with adjusted size (vertical orientation)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Define font sizes for consistency
        font_size_ticks = 10
        font_size_labels = 14
        font_size_legend = 12
        font_size_annotations = 16

        # No background shading for significance (clean styling per request)

        # Draw vertical boxplots with black styling: solid for 'on', dashed for 'off'
        for light_condition in sorted_light_conditions:
            condition_data = plot_data[plot_data[y] == light_condition][metric].dropna()
            if len(condition_data) == 0:
                continue

            position = sorted_light_conditions.index(light_condition)

            # Determine line style based on condition
            linestyle = light_condition_styles.get(light_condition, "solid")
            linewidth = 2.5

            # Create vertical boxplot with black lines
            bp = ax.boxplot(
                [condition_data],
                positions=[position],
                widths=0.6,
                patch_artist=False,
                showfliers=False,
                vert=True,
                boxprops=dict(color="black", linewidth=linewidth, linestyle=linestyle),
                whiskerprops=dict(color="black", linewidth=linewidth, linestyle=linestyle),
                capprops=dict(color="black", linewidth=linewidth, linestyle=linestyle),
                medianprops=dict(color="black", linewidth=linewidth + 0.5, linestyle=linestyle),
            )
        # Overlay jittered scatter per condition with requested styling (vertical)
        for idx, lc in enumerate(sorted_light_conditions):
            vals = plot_data[plot_data[y] == lc][metric].dropna().values
            if len(vals) == 0:
                continue
            x_jitter = np.random.normal(idx, 0.08, size=len(vals))
            # Light OFF: no fill, black outline; Light ON: filled black
            if lc == "off":
                ax.scatter(
                    x_jitter,
                    vals,
                    s=40,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.6,
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_jitter,
                    vals,
                    s=40,
                    facecolors="black",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.6,
                    zorder=3,
                )

        # Explicitly set x-tick labels to show ON/OFF and sample sizes (n)
        counts = [len(plot_data[plot_data[y] == lc]) for lc in sorted_light_conditions]
        display_labels_ordered = [
            f"{value_to_label.get(lc, str(lc))} (n={cnt})" for lc, cnt in zip(sorted_light_conditions, counts)
        ]
        ax.set_xticks(range(len(sorted_light_conditions)))
        ax.set_xticklabels(display_labels_ordered)

        # Set ylim BEFORE adding annotations to ensure proper positioning
        y_max = plot_data[metric].quantile(0.99)
        y_min = plot_data[metric].min()
        # Add space for annotations above
        data_range = y_max - y_min
        ax.set_ylim(bottom=y_min - 0.05 * data_range, top=y_max + 0.15 * data_range)

        # Add a single between-group significance annotation when exactly two conditions
        if len(sorted_light_conditions) == 2:
            vals0 = plot_data[plot_data[y] == sorted_light_conditions[0]][metric].dropna().values
            vals1 = plot_data[plot_data[y] == sorted_light_conditions[1]][metric].dropna().values
            if len(vals0) > 0 and len(vals1) > 0:
                obs_stat = float(np.median(vals1) - np.median(vals0))
                combined = np.concatenate([vals0, vals1])
                n0 = len(vals0)

                perm_diffs = np.empty(n_permutations)
                for i in range(n_permutations):
                    np.random.shuffle(combined)
                    perm0 = combined[:n0]
                    perm1 = combined[n0:]
                    perm_diffs[i] = np.median(perm1) - np.median(perm0)

                pval_between = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_stat)))

                if pval_between < 0.001:
                    sig_between = "***"
                elif pval_between < 0.01:
                    sig_between = "**"
                elif pval_between < 0.05:
                    sig_between = "*"
                else:
                    sig_between = "ns"

                x_pos = 0.5 * (0 + 1)
                y_pos = y_max + 0.05 * data_range
                ann_text = f"{sig_between}"
                ax.text(
                    x_pos,
                    y_pos,
                    ann_text,
                    color="red",
                    fontsize=font_size_annotations,
                    fontweight="bold",
                    va="bottom",
                    ha="center",
                    clip_on=False,
                )

        display_metric = metric.replace("_", " ")
        if metric in time_metrics:
            display_metric = f"{display_metric} (min)"

        # Formatting
        plt.xticks(fontsize=font_size_ticks)
        plt.yticks(fontsize=font_size_ticks)
        plt.xlabel("Light Condition", fontsize=font_size_labels)
        plt.ylabel(display_metric, fontsize=font_size_labels)
        ax.grid(axis="x", alpha=0.3)

        # Use tight_layout with padding to accommodate external legend
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric}_light_condition_permutation.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {output_file}")

        # Print statistical summary
        n_significant = sum(1 for r in test_results if r["significant"])
        print(f"  Significant: {n_significant}/{len(test_results)} (α={alpha})")

        # Print timing for this metric
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save statistics
        stats_file = output_dir / "light_condition_permutation_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "light_condition_permutation_report.md"
        generate_text_report(stats_df, data, control_condition, report_file)
        print(f"Text report saved to: {report_file}")

        # Print overall summary
        total_tests = len(stats_df)
        total_significant = stats_df["significant"].sum()

        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Significant (p<{alpha}): {total_significant} ({100*total_significant/total_tests:.1f}%)")

    return stats_df


def generate_text_report(stats_df, data, control_condition, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    control_condition : str
        Name of the control light condition
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Light Condition Permutation Test Report")
    report_lines.append("## Statistical Analysis Results")
    report_lines.append("")
    report_lines.append(f"**Control group:** Light ON ('{control_condition}')")
    report_lines.append(f"**Test group:** Light OFF ('off')")
    report_lines.append("")

    # Group by metric
    metrics = stats_df["Metric"].unique()

    for metric in sorted(metrics):
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"## {metric}")
        report_lines.append("")

        # Get control median for reference
        control_median = 0  # Initialize default value
        control_data = data[data["Light"] == control_condition][metric].dropna()
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

        # Analyze each light condition vs control
        significant_increased = []
        significant_decreased = []
        non_significant = []

        for _, row in metric_stats.iterrows():
            light_condition = row["LightCondition"]
            p_fdr = row["pval_corrected"]
            sig_level = row["sig_level"]
            test_median = row["test_median"]
            effect_size = row["effect_size"]
            test_n = row["test_n"]

            # Calculate percent change
            if control_median != 0:
                percent_change = (effect_size / control_median) * 100
            else:
                percent_change = 0

            description = f'"{light_condition}" (median = {test_median:.3f}, n={test_n})'

            if row["significant"]:
                if row["direction"] == "increased":
                    significant_increased.append(
                        {
                            "light_condition": light_condition,
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
                            "light_condition": light_condition,
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
                        "light_condition": light_condition,
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

    print(f"Generated text report with {len(metrics)} metrics and {total_comparisons} comparisons")


def load_and_clean_exploration_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the exploration dataset with light conditions

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the exploration dataset
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/summary/pooled_summary.feather"

    print(f"Loading exploration dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Exploration dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Exploration dataset not found at {dataset_path}")

    # Drop TNT-specific metadata columns that cause pandas warnings
    # These columns are not needed for light condition analysis
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
        "Period",
        # "FeedingState",
        "Orientation",
        "Crossing",
        "BallType",
        "Used_to",
        "Magnet",
        "Peak",
    ]
    columns_to_drop = [col for col in tnt_metadata_columns if col in dataset.columns]
    if columns_to_drop:
        dataset = dataset.drop(columns=columns_to_drop)
        print(f"Dropped metadata columns to avoid warnings: {columns_to_drop}")
        print(f"Dataset shape after dropping metadata: {dataset.shape}")

    # Clean up Light column - remove empty strings and convert to proper values
    print(f"Original Light column values: {sorted(dataset['Light'].unique())}")

    # Remove rows with empty Light values
    dataset = dataset[dataset["Light"] != ""].copy()
    print(f"After removing empty Light values, shape: {dataset.shape}")
    print(f"Light column values after cleanup: {sorted(dataset['Light'].unique())}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time", "index"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    # Handle new metrics with appropriate defaults - FIXED for performance
    new_metrics_defaults = {
        "has_finished": 0,  # Binary metric, 0 if no finish detected
        "persistence_at_end": 0.0,  # Time fraction, 0 if no persistence
        "fly_distance_moved": 0.0,  # Distance, 0 if no movement detected
        "time_chamber_beginning": 0.0,  # Time, 0 if no time in beginning chamber
        "median_freeze_duration": 0.0,  # Duration, 0 if no freezes detected
        "fraction_not_facing_ball": 0.0,  # Fraction, 0 if always facing ball or no data
        "flailing": 0.0,  # Motion energy, 0 if no leg movement detected
        "head_pushing_ratio": 0.0,  # Ratio, 0.5 if no contact data (neutral)
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
        print(f"🧪 TEST MODE: Light conditions in sample: {sorted(dataset['Light'].unique())}")

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

    # Check for continuous metric plots (permutation style)
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


def get_nan_annotations(data, metrics, y_col="Light"):
    """
    Generate annotations for light conditions that have NaN values in specific metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the metrics
    metrics : list
        List of metric column names to check
    y_col : str
        Column name for grouping (e.g., light condition)

    Returns:
    --------
    dict
        Dictionary mapping metric names to dictionaries of condition: nan_info
    """
    nan_annotations = {}

    for metric in metrics:
        if metric not in data.columns:
            continue

        metric_annotations = {}

        for condition in data[y_col].unique():
            condition_data = data[data[y_col] == condition][metric]
            total_count = len(condition_data)
            nan_count = condition_data.isnull().sum()

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                metric_annotations[condition] = {
                    "nan_count": nan_count,
                    "total_count": total_count,
                    "nan_percentage": nan_percentage,
                    "annotation": f"({nan_count}/{total_count} NaN)",
                }

        if metric_annotations:
            nan_annotations[metric] = metric_annotations

    return nan_annotations


def main(overwrite=True, test_mode=False, metrics_file="src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"):
    """Main function to run permutation-test plots for all metrics.

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
    print(f"Starting permutation-test analysis for light condition experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/Summary_metrics/Permutation")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Clean up any potential old outputs to avoid confusion
    print("Ensuring clean output directory structure...")
    condition_subdir = "permutation_light_conditions"
    condition_dir = base_output_dir / condition_subdir
    condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created/verified: {condition_dir}")

    # Also create the main directory for the combined statistics
    print(f"All outputs will be saved under: {base_output_dir}")

    # Load the dataset
    print("Loading exploration dataset...")
    dataset = load_and_clean_exploration_dataset(test_mode=test_mode, test_sample_size=200)

    # Convert time-related metrics from seconds to minutes
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
    for metric in time_metrics:
        if metric in dataset.columns:
            dataset[metric] = dataset[metric] / 60.0
            print(f"  Converted {metric} to minutes")
        else:
            print(f"  Skipping {metric} (not in dataset)")

    # Get only the data where FeedingState is "starved_noWater"

    dataset = dataset[dataset["FeedingState"] == "starved_noWater"]

    # Check what columns we have
    print(f"\nDataset columns: {list(dataset.columns)}")

    # Check for Light column
    if "Light" not in dataset.columns:
        print("❌ Light column not found in dataset!")
        print("Available columns that might contain light condition info:")
        potential_cols = [col for col in dataset.columns if any(word in col.lower() for word in ["light", "condition"])]
        print(potential_cols)
        return

    print(f"✅ Light column found")
    print(f"Light conditions in dataset: {sorted(dataset['Light'].unique())}")

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

    # Additional pattern-based metrics
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
            if col not in available_metrics and col != "Light":  # Avoid duplicates and metadata
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
    print(f"Found {len(continuous_metrics)} continuous metrics for permutation analysis:")
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

    # If in test mode, limit to first 3 metrics and show more timing info
    # If a metrics file is provided, restrict available metrics to that list (default: PCA list)
    if metrics_file is not None:
        metrics_path = Path(metrics_file)
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                requested = [line.strip() for line in f if line.strip()]
            # Keep only requested metrics that were classified as continuous
            requested_present = [m for m in requested if m in continuous_metrics]
            missing = [m for m in requested if m not in available_metrics]
            if missing:
                print(f"Warning: {len(missing)} requested metrics not found or non-numeric: {missing}")
            if requested_present:
                continuous_metrics = requested_present
                print(f"Using {len(continuous_metrics)} metrics from {metrics_path}")
            else:
                print(f"No requested continuous metrics found in dataset; proceeding with automatic selection")
        else:
            print(f"Warning: metrics file not found: {metrics_path}; using automatic selection")

    if test_mode:
        original_count = len(continuous_metrics)
        continuous_metrics = continuous_metrics[:3]
        binary_metrics = binary_metrics[:2] if binary_metrics else []
        print(f"🧪 TEST MODE: Reduced from {original_count} to {len(continuous_metrics)} continuous metrics")
        print(f"🧪 TEST MODE: Limited to {len(binary_metrics)} binary metrics")
        print(f"🧪 TEST MODE: Processing metrics: {continuous_metrics}")
        print(f"🧪 TEST MODE: With dataset sample size: {len(dataset)} rows")
        print(f"🧪 TEST MODE: This should complete much faster for debugging")

    # Clean the dataset (already done in load_and_clean_exploration_dataset)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique light conditions: {len(dataset['Light'].unique())}")

    # Validate required columns
    required_cols = ["Light"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Generate plots for light condition analysis
    print(f"\n{'='*60}")
    print(f"Processing Light Condition Analysis...")
    print("=" * 60)

    # Use all dataset for light condition analysis
    filtered_data = dataset
    print(f"Dataset shape: {filtered_data.shape}")

    if len(filtered_data) == 0:
        print(f"No data for light condition analysis. Skipping...")
        return

    # Create output directory
    output_dir = base_output_dir / condition_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Light conditions in dataset: {sorted(filtered_data['Light'].unique())}")
    print(f"Using simplified color palette for visualization")

    # 1. Process continuous metrics with permutation tests
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (permutation tests) ---")
        # Filter to only continuous metrics + required columns
        continuous_data = filtered_data[["Light"] + continuous_metrics].copy()

        print(f"Generating permutation plots for light conditions...")
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
                print(f"📄 Skipping {len(continuous_metrics) - len(metrics_to_process)} metrics with existing plots")

            if not metrics_to_process:
                print("📄 All metrics already have plots. Use --overwrite to regenerate.")
            else:
                print(f"📄 Processing {len(metrics_to_process)} metrics without existing plots")
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            # Check for NaN annotations
            print(f"📊 Checking for NaN values in {len(metrics_to_process)} metrics...")
            nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "Light")
            if nan_annotations:
                print(f"📋 NaN values detected in continuous metrics:")
                for metric, condition_info in nan_annotations.items():
                    print(f"  {metric}: {len(condition_info)} light conditions have NaN values")
                    for condition, info in condition_info.items():
                        print(f"    - {condition}: {info['annotation']}")

            print(f"🚀 Starting permutation analysis for {len(metrics_to_process)} continuous metrics...")
            print(f"📊 Dataset shape for analysis: {continuous_data.shape}")
            print(f"📊 Light conditions: {sorted(continuous_data['Light'].unique())}")
            print(f"📊 Sample sizes per light condition:")
            condition_counts = continuous_data["Light"].value_counts()
            for condition, count in condition_counts.items():
                print(f"    {condition}: {count} samples")

            # Determine control condition
            control_condition = "off"
            print(f"📊 Control condition: {control_condition}")
            print(f"📊 Test conditions: {[lc for lc in continuous_data['Light'].unique() if lc != control_condition]}")

            # Use the new FDR-corrected function
            start_time = time.time()
            stats_df = generate_light_condition_permutation_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="Light",
                control_condition=control_condition,
                hue="Light",
                palette="Set1",  # Use string palette instead of dict for compatibility
                output_dir=str(output_dir),
                fdr_method="fdr_bh",  # Benjamini-Hochberg FDR correction
                alpha=0.05,  # FDR-corrected significance level
                n_permutations=10000,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"✅ FDR-corrected permutation analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
            )

    print(f"Unique light conditions: {len(filtered_data['Light'].unique())}")

    print(f"\n{'='*60}")
    print("✅ Light condition analysis complete! Check output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate permutation-test jitterboxplots for light condition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_permutation_light_conditions.py                    # Overwrite existing plots
  python run_permutation_light_conditions.py --no-overwrite     # Skip existing plots
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
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="src/PCA/metrics_lists/final_metrics_for_pca_alt.txt",
        help=(
            "Path to a file listing metrics (one per line) to analyze (reduces computation)."
            " Defaults to src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"
        ),
    )

    args = parser.parse_args()

    # Run main function with overwrite parameter (invert --no-overwrite)
    main(overwrite=not args.no_overwrite, test_mode=args.test, metrics_file=args.metrics_file)
