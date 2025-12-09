#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for learning mutants experiments.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Comparison across Genotype groups
- FDR correction across all comparisons for each metric
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability

Usage:
    python run_mannwhitney_learning_mutants.py [--no-overwrite] [--test]

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
from statsmodels.stats.multitest import multipletests
import time


def _normalize_genotype_string(s):
    """Normalize a single genotype string to avoid duplicates due to case/spacing.

    - Strips leading/trailing whitespace
    - Collapses internal whitespace to single spaces
    - Preserves original case by default, but returns a compare key (lowercased)
    """
    if pd.isna(s):
        return s, s
    original = str(s)
    cleaned = " ".join(original.strip().split())  # trim + collapse spaces
    compare_key = cleaned.lower()
    return cleaned, compare_key


def normalize_genotypes_column(df, col_name="Genotype"):
    """Normalize the Genotype column and merge duplicates that differ only by case/spacing.

    Returns a tuple: (df_copy, merge_report)
    - df_copy: DataFrame with normalized `Genotype`
    - merge_report: dict mapping compare_key -> list of original variants observed
    """
    if col_name not in df.columns:
        return df.copy(), {}

    df = df.copy()
    originals = df[col_name].astype(str).tolist()
    normalized = []
    keys = []
    for s in originals:
        cleaned, key = _normalize_genotype_string(s)
        normalized.append(cleaned)
        keys.append(key)

    df[col_name] = normalized

    # Build report of variants per key
    report = {}
    for orig, key in zip(originals, keys):
        report.setdefault(key, set()).add(orig)

    # If multiple variants map to the same key, choose a canonical label:
    # Prefer the most frequent cleaned label among those variants
    variant_counts = df.groupby(col_name).size().to_dict()
    key_to_canonical = {}
    for key, variants in report.items():
        # Map variants to their cleaned form and count
        cleaned_variants = {
            " ".join(str(v).strip().split()): variant_counts.get(" ".join(str(v).strip().split()), 0) for v in variants
        }
        # Pick the cleaned variant with max count; tie-breaker: shortest string
        canonical = (
            sorted(cleaned_variants.items(), key=lambda x: (-x[1], len(x[0])))[0][0] if cleaned_variants else key
        )
        key_to_canonical[key] = canonical

    # Apply canonical mapping by compare key
    df[col_name] = [key_to_canonical[k] for k in keys]

    # Convert to categorical to lock consistent ordering
    ordered = sorted(df[col_name].unique())
    df[col_name] = pd.Categorical(df[col_name], categories=ordered, ordered=True)

    # Convert sets to sorted lists for printing
    report = {k: sorted(list(v)) for k, v in report.items()}
    return df, report


def print_genotype_individual_counts(df, group_col="Genotype", subject_candidates=None):
    """Print a summary of how many individuals are present per genotype.

    Prefers a subject identifier column (e.g., 'fly'), falling back to row counts.
    """
    if subject_candidates is None:
        subject_candidates = ["fly", "Fly", "subject", "Subject", "id", "ID"]

    subject_col = next((c for c in subject_candidates if c in df.columns), None)

    print("\nPer-genotype individual counts:")
    if subject_col:
        for genotype in sorted(pd.Series(df[group_col]).astype(str).unique()):
            n = df[df[group_col] == genotype][subject_col].nunique()
            print(f"  {genotype}: {n} individuals (unique {subject_col})")
    else:
        for genotype in sorted(pd.Series(df[group_col]).astype(str).unique()):
            n = len(df[df[group_col] == genotype])
            print(f"  {genotype}: {n} rows (no subject id found)")


def generate_genotype_mannwhitney_plots(
    data,
    metrics,
    y="Genotype",
    control_genotype=None,
    hue=None,
    palette="Set2",
    figsize=(15, 10),
    output_dir="mann_whitney_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between each genotype and control.
    Applies FDR correction across all comparisons for each metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (genotypes). Default is "Genotype".
        control_genotype (str): Name of the control genotype. If None, uses alphabetically first genotype.
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

        # Check data availability before filtering
        original_data_counts = data.groupby(y).size()

        plot_data = data.dropna(subset=[metric, y])

        # Report data loss due to NaN values
        filtered_data_counts = plot_data.groupby(y).size()
        total_original = len(data)
        total_filtered = len(plot_data)

        if total_filtered < total_original:
            pct_kept = 100 * total_filtered / total_original
            print(
                f"  ⚠️  Data filtering: kept {total_filtered}/{total_original} rows ({pct_kept:.1f}%) after removing NaN values"
            )

            # Show per-genotype NaN statistics
            for genotype in sorted(data[y].unique()):
                orig_count = original_data_counts.get(genotype, 0)
                filt_count = filtered_data_counts.get(genotype, 0)
                if orig_count != filt_count:
                    nan_count = orig_count - filt_count
                    print(f"    {genotype}: {filt_count}/{orig_count} samples ({nan_count} NaN values for {metric})")

        # Get all genotypes
        genotypes = sorted(plot_data[y].unique())

        # Determine control genotype
        if control_genotype is None:
            control_genotype = genotypes[0]  # Use first genotype alphabetically
            print(f"  Using {control_genotype} as control genotype")

        # Get all test genotypes (excluding control)
        test_genotypes = [g for g in genotypes if g != control_genotype]

        # Check if control exists
        if control_genotype not in plot_data[y].unique():
            print(f"Warning: Control genotype '{control_genotype}' not found for metric {metric}")
            continue

        control_vals = plot_data[plot_data[y] == control_genotype][metric].dropna()
        if len(control_vals) == 0:
            print(f"Warning: No control data for metric {metric}")
            continue

        control_median = control_vals.median()

        # Perform Mann-Whitney U tests for all genotypes vs control
        stats_list = []
        p_values = []
        genotype_order = []

        for genotype in test_genotypes:
            genotype_vals = plot_data[plot_data[y] == genotype][metric].dropna()

            if len(genotype_vals) == 0:
                print(f"  Warning: No data for genotype {genotype}")
                continue

            # Perform Mann-Whitney U test
            try:
                stat, pval = mannwhitneyu(genotype_vals, control_vals, alternative="two-sided")

                # Determine direction of effect
                genotype_median = genotype_vals.median()
                direction = "increased" if genotype_median > control_median else "decreased"
                effect_size = genotype_median - control_median

                stats_list.append(
                    {
                        "genotype": genotype,
                        "pval_raw": pval,
                        "direction": direction,
                        "effect_size": effect_size,
                        "test_statistic": stat,
                        "genotype_median": genotype_median,
                        "genotype_n": len(genotype_vals),
                    }
                )
                p_values.append(pval)
                genotype_order.append(genotype)

            except Exception as e:
                print(f"  Error testing {genotype}: {e}")
                continue

        if len(p_values) == 0:
            print(f"  No valid comparisons for metric {metric}")
            continue

        # Apply FDR correction across all comparisons for this metric
        p_values = np.array(p_values)
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)

        # Add corrected p-values and significance to stats_list
        for i, stat_dict in enumerate(stats_list):
            stat_dict["pval_corrected"] = pvals_corrected[i]
            stat_dict["significant"] = rejected[i]

            # Determine significance level based on corrected p-value
            if pvals_corrected[i] < 0.001:
                stat_dict["sig_level"] = "***"
            elif pvals_corrected[i] < 0.01:
                stat_dict["sig_level"] = "**"
            elif pvals_corrected[i] < 0.05:
                stat_dict["sig_level"] = "*"
            else:
                stat_dict["sig_level"] = "ns"

            # If not significant, set direction to "none"
            if not rejected[i]:
                stat_dict["direction"] = "none"

        # Sort genotypes by significance and effect
        # Group 1: Significantly increased (sorted by median descending)
        # Group 2: Non-significant (sorted by median descending)
        # Group 3: Significantly decreased (sorted by median ascending)

        increased = [s for s in stats_list if s["significant"] and s["direction"] == "increased"]
        decreased = [s for s in stats_list if s["significant"] and s["direction"] == "decreased"]
        neutral = [s for s in stats_list if not s["significant"]]

        # Sort within each group
        increased.sort(key=lambda x: x["genotype_median"], reverse=True)
        neutral.sort(key=lambda x: x["genotype_median"], reverse=True)
        decreased.sort(key=lambda x: x["genotype_median"])

        # Combine all groups
        sorted_stats = increased + neutral + decreased
        sorted_genotypes = [control_genotype] + [s["genotype"] for s in sorted_stats]

        # Store results for output
        for stat_dict in stats_list:
            all_stats.append(
                {
                    "Metric": metric,
                    "Control": control_genotype,
                    "Test": stat_dict["genotype"],
                    "pval_raw": stat_dict["pval_raw"],
                    "pval_corrected": stat_dict["pval_corrected"],
                    "direction": stat_dict["direction"],
                    "effect_size": stat_dict["effect_size"],
                    "test_statistic": stat_dict["test_statistic"],
                    "test_median": stat_dict["genotype_median"],
                    "control_median": control_median,
                    "test_n": stat_dict["genotype_n"],
                    "control_n": len(control_vals),
                    "significant": stat_dict["significant"],
                    "sig_level": stat_dict["sig_level"],
                }
            )

        # Update plot data with sorted categories
        plot_data[y] = pd.Categorical(plot_data[y], categories=sorted_genotypes, ordered=True)
        plot_data = plot_data.sort_values(by=[y])

        # Calculate dynamic figure height based on number of genotypes
        n_genotypes = len(sorted_genotypes)
        fig_height = max(8, n_genotypes * 0.8)
        fig_width = figsize[0]

        # Create the plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set up colors for genotypes
        control_color = "#FF6B6B"  # Red for control
        test_colors = sns.color_palette("husl", len(test_genotypes))
        color_map = {control_genotype: control_color}
        for i, genotype in enumerate([s["genotype"] for s in sorted_stats]):
            color_map[genotype] = test_colors[i % len(test_colors)]

        colors = [color_map[g] for g in sorted_genotypes]

        # Add colored backgrounds for significant results
        y_positions = range(len(sorted_genotypes))
        for i, genotype in enumerate(sorted_genotypes):
            if genotype == control_genotype:
                # Control gets light blue background
                bg_color = "lightblue"
                alpha_bg = 0.1
            else:
                # Find this genotype in sorted_stats
                genotype_stat = next((s for s in sorted_stats if s["genotype"] == genotype), None)
                if genotype_stat is None:
                    continue

                if genotype_stat["significant"]:
                    if genotype_stat["direction"] == "increased":
                        bg_color = "lightgreen"
                        alpha_bg = 0.15
                    else:  # decreased
                        bg_color = "lightcoral"
                        alpha_bg = 0.15
                else:
                    # Non-significant gets no background
                    continue

            # Add background rectangle
            ax.axhspan(i - 0.4, i + 0.4, color=bg_color, alpha=alpha_bg, zorder=0)

        # Draw boxplots with unfilled styling
        for genotype in sorted_genotypes:
            genotype_data = plot_data[plot_data[y] == genotype]
            if genotype_data.empty:
                continue

            is_control = genotype == control_genotype

            if is_control:
                # Control genotype styling - red dashed boxes
                sns.boxplot(
                    data=genotype_data,
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
                # Test genotype styling - black solid boxes
                sns.boxplot(
                    data=genotype_data,
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

        # Overlay stripplot for jitter with genotype colors
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

        # Add significance annotations
        stat_dict_map = {s["genotype"]: s for s in sorted_stats}
        for i, genotype in enumerate(sorted_genotypes):
            if genotype == control_genotype:
                continue

            genotype_stat = stat_dict_map.get(genotype)
            if genotype_stat is None or genotype_stat["sig_level"] == "ns":
                continue

            # Add significance marker
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            yticklocs = ax.get_yticks()
            ax.text(
                x=ax.get_xlim()[1] + 0.01 * x_range,
                y=float(yticklocs[i]),
                s=genotype_stat["sig_level"],
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
        plt.ylabel("Genotype", fontsize=14)

        # Add title with FDR correction info
        n_significant = sum(1 for s in sorted_stats if s["significant"])
        title = f"Mann-Whitney U Test: {metric} by Genotype\n"
        title += f"(FDR-corrected, {n_significant}/{len(test_genotypes)} significant comparisons)"
        plt.title(title, fontsize=16)
        ax.grid(axis="x", alpha=0.3)

        # Create custom legend
        legend_elements = [
            Patch(
                facecolor="none", edgecolor="red", linestyle="--", linewidth=2, label=f"Control ({control_genotype})"
            ),
            Patch(facecolor="none", edgecolor="black", linewidth=1, label="Test Genotypes"),
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
        output_file = output_dir / f"{metric}_genotype_mannwhitney.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {output_file}")
        print(f"  Significant comparisons: {n_significant}/{len(test_genotypes)}")

        # Print timing for this metric
        metric_end_time = time.time()
        metric_elapsed = metric_end_time - metric_start_time
        print(f"  ⏱️  Metric {metric} completed in {metric_elapsed:.2f} seconds")

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    if not stats_df.empty:
        # Save statistics
        stats_file = output_dir / "genotype_mannwhitney_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")

        # Generate text report
        report_file = output_dir / "genotype_mannwhitney_report.md"
        generate_text_report(stats_df, data, report_file)
        print(f"Text report saved to: {report_file}")

        # Print overall summary
        total_tests = len(stats_df)
        total_significant = stats_df["significant"].sum()

        print(f"\nOverall Summary:")
        print(f"Total comparisons: {total_tests}")
        print(f"Significant results (FDR-corrected): {total_significant} ({100*total_significant/total_tests:.1f}%)")

    return stats_df


def generate_text_report(stats_df, data, report_file):
    """
    Generate a human-readable text report of the statistical results.

    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistical results
    data : pd.DataFrame
        Original data used for analysis
    report_file : Path
        Path where to save the report
    """
    report_lines = []
    report_lines.append("# Learning Mutants Genotype Mann-Whitney U Test Report")
    report_lines.append("## Statistical Analysis Results (FDR-corrected)")
    report_lines.append("")

    # Get control genotype
    control_genotype = stats_df["Control"].iloc[0] if len(stats_df) > 0 else "Unknown"
    report_lines.append(f"**Control genotype:** {control_genotype}")
    report_lines.append("")

    # Group by metric
    for metric in stats_df["Metric"].unique():
        metric_stats = stats_df[stats_df["Metric"] == metric]

        report_lines.append(f"## {metric}")
        report_lines.append("")

        # Separate significant and non-significant results
        significant_increased = metric_stats[
            (metric_stats["significant"]) & (metric_stats["direction"] == "increased")
        ].sort_values("effect_size", ascending=False)

        significant_decreased = metric_stats[
            (metric_stats["significant"]) & (metric_stats["direction"] == "decreased")
        ].sort_values("effect_size")

        non_significant = metric_stats[~metric_stats["significant"]].sort_values(
            "effect_size", key=lambda x: abs(x), ascending=False
        )

        if len(significant_increased) > 0:
            report_lines.append("### Significantly Higher")
            report_lines.append("")
            for _, row in significant_increased.iterrows():
                percent_change = (row["effect_size"] / row["control_median"] * 100) if row["control_median"] != 0 else 0
                report_lines.append(
                    f"- **{row['Test']}**: median = {row['test_median']:.3f} (n={row['test_n']}), "
                    f"control median = {row['control_median']:.3f} (n={row['control_n']}), "
                    f"change = {percent_change:+.1f}%, p = {row['pval_corrected']:.4f} {row['sig_level']}"
                )
            report_lines.append("")

        if len(significant_decreased) > 0:
            report_lines.append("### Significantly Lower")
            report_lines.append("")
            for _, row in significant_decreased.iterrows():
                percent_change = (row["effect_size"] / row["control_median"] * 100) if row["control_median"] != 0 else 0
                report_lines.append(
                    f"- **{row['Test']}**: median = {row['test_median']:.3f} (n={row['test_n']}), "
                    f"control median = {row['control_median']:.3f} (n={row['control_n']}), "
                    f"change = {percent_change:+.1f}%, p = {row['pval_corrected']:.4f} {row['sig_level']}"
                )
            report_lines.append("")

        if len(non_significant) > 0:
            report_lines.append("### No Significant Difference")
            report_lines.append("")
            for _, row in non_significant.iterrows():
                percent_change = (row["effect_size"] / row["control_median"] * 100) if row["control_median"] != 0 else 0
                report_lines.append(
                    f"- **{row['Test']}**: median = {row['test_median']:.3f} (n={row['test_n']}), "
                    f"control median = {row['control_median']:.3f} (n={row['control_n']}), "
                    f"change = {percent_change:+.1f}%, p = {row['pval_corrected']:.4f} (ns)"
                )
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Add summary section
    report_lines.append("## Overall Summary")
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

    # Write to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """Load and clean the learning mutants dataset

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the learning mutants dataset
    dataset_path = "/mnt/upramdya_data/MD/Learning_mutants/Datasets/251205_12_summary_learning_mutants_Data/summary/pooled_summary.feather"

    print(f"Loading learning mutants dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

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
        if "Genotype" in dataset.columns:
            print(f"🧪 TEST MODE: Genotypes in sample: {sorted(dataset['Genotype'].unique())}")

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


def analyze_binary_metrics(data, binary_metrics, y="Genotype", output_dir=None, overwrite=True, control_genotype=None):
    """
    Analyze binary metrics using Fisher's exact test for genotype comparisons.

    For 2 groups: Uses Fisher's exact test
    For 3+ groups: Uses overall chi-square test followed by pairwise Fisher's tests with FDR correction

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (genotype)
    output_dir : str or Path
        Directory to save plots
    overwrite : bool
        If True, overwrite existing plots. If False, skip existing plots.
    control_genotype : str
        Name of control genotype for comparisons

    Returns:
    --------
    tuple
        (pairwise_results_df, overall_results_df) - DataFrames with pairwise and overall statistics
    """
    if not binary_metrics:
        print("No binary metrics to analyze")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\n{'='*50}")
    print(f"BINARY METRICS ANALYSIS")
    print(f"{'='*50}")

    results = []
    overall_results = []  # Store overall test results

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique groups
    genotypes = sorted(data[y].unique())
    n_genotypes = len(genotypes)

    # Determine control genotype
    if control_genotype is None:
        control_genotype = genotypes[0]
        print(f"Using {control_genotype} as control genotype")

    test_genotypes = [g for g in genotypes if g != control_genotype]

    print(f"Control genotype: {control_genotype}")
    print(f"Test genotypes: {test_genotypes}")
    print(f"Number of genotypes: {n_genotypes}")

    for metric in binary_metrics:
        print(f"\nAnalyzing binary metric: {metric}")

        # Check if we should skip this metric
        if output_dir and should_skip_metric(metric, output_dir, overwrite):
            continue

        # Create contingency table
        contingency = pd.crosstab(data[y], data[metric], margins=True)
        print(f"Contingency table for {metric}:")
        print(contingency)

        # Calculate proportions for each genotype
        proportions = data.groupby(y)[metric].agg(["count", "sum", "mean"]).round(4)
        proportions["proportion"] = proportions["mean"]
        proportions["n_success"] = proportions["sum"]
        proportions["n_total"] = proportions["count"]
        print(f"\nProportions for {metric}:")
        print(proportions[["n_success", "n_total", "proportion"]])

        # For 3+ groups, perform overall chi-square test first
        overall_p = None
        overall_chi2 = None
        if n_genotypes > 2:
            try:
                # Overall chi-square test
                ct_values = pd.crosstab(data[y], data[metric])
                chi2, overall_p, dof, expected = chi2_contingency(ct_values)
                overall_chi2 = chi2
                min_expected = expected.min()

                print(f"\n  Overall Chi-square test: χ²={chi2:.3f}, p={overall_p:.4f}, df={dof}")
                print(f"  Min expected frequency: {min_expected:.1f}")

                if min_expected < 5:
                    print(f"  ⚠️  Warning: Some expected frequencies < 5, chi-square may not be valid")

                # Store overall result
                overall_results.append(
                    {
                        "metric": metric,
                        "grouping": y,
                        "n_groups": n_genotypes,
                        "chi2": chi2,
                        "p_value": overall_p,
                        "dof": dof,
                        "min_expected": min_expected,
                        "significant": overall_p < 0.05,
                    }
                )

            except Exception as e:
                print(f"  Error in overall chi-square test: {e}")

        # Pairwise comparisons: each genotype vs control
        pairwise_results_for_metric = []

        for test_genotype in test_genotypes:
            # Get data for this comparison
            test_data = data[data[y] == test_genotype][metric].dropna()
            control_data_subset = data[data[y] == control_genotype][metric].dropna()

            if len(test_data) == 0 or len(control_data_subset) == 0:
                print(f"  No data for {test_genotype} vs {control_genotype}")
                continue

            # Create 2x2 contingency table for Fisher's exact test
            test_success = int(test_data.sum())
            test_total = len(test_data)
            control_success = int(control_data_subset.sum())
            control_total = len(control_data_subset)

            # Fisher's exact test
            try:
                table = [[test_success, test_total - test_success], [control_success, control_total - control_success]]

                # Run Fisher's exact test
                fisher_result = fisher_exact(table)

                # Extract results (fisher_exact returns (oddsratio, pvalue))
                odds_ratio = float(fisher_result[0])
                p_value = float(fisher_result[1])

                # Effect size (difference in proportions)
                test_prop = test_success / test_total if test_total > 0 else 0
                control_prop = control_success / control_total if control_total > 0 else 0
                effect_size = test_prop - control_prop

                pairwise_results_for_metric.append(
                    {
                        "metric": metric,
                        "control": control_genotype,
                        "test": test_genotype,
                        "test_success": test_success,
                        "test_total": test_total,
                        "test_proportion": test_prop,
                        "control_success": control_success,
                        "control_total": control_total,
                        "control_proportion": control_prop,
                        "effect_size": effect_size,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                        "p_value_uncorrected": p_value,  # Store uncorrected p-value
                        "test_type": "Fisher exact",
                        "overall_chi2": overall_chi2,
                        "overall_p": overall_p,
                    }
                )

                print(f"  {test_genotype} vs {control_genotype}: OR={odds_ratio:.3f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error testing {test_genotype} vs {control_genotype}: {e}")

        # Apply FDR correction if we have multiple comparisons for this metric
        if len(pairwise_results_for_metric) > 1:
            p_values = [r["p_value"] for r in pairwise_results_for_metric]
            reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

            print(f"\n  FDR correction (Benjamini-Hochberg) applied to {len(p_values)} comparisons:")
            for i, result in enumerate(pairwise_results_for_metric):
                result["p_value_fdr"] = p_corrected[i]
                result["significant_fdr"] = reject[i]

                # Update significance level based on FDR-corrected p-value
                if p_corrected[i] < 0.001:
                    sig_level = "***"
                elif p_corrected[i] < 0.01:
                    sig_level = "**"
                elif p_corrected[i] < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                result["sig_level_fdr"] = sig_level
                print(
                    f"    {result['test']} vs {result['control']}: p_uncorrected={result['p_value']:.4f}, p_fdr={p_corrected[i]:.4f} {sig_level}"
                )
        else:
            # No FDR correction needed for single comparison
            for result in pairwise_results_for_metric:
                result["p_value_fdr"] = result["p_value"]
                result["significant_fdr"] = result["p_value"] < 0.05

                # Determine significance level
                if result["p_value"] < 0.001:
                    sig_level = "***"
                elif result["p_value"] < 0.01:
                    sig_level = "**"
                elif result["p_value"] < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                result["sig_level_fdr"] = sig_level

        # Add all pairwise results for this metric to the overall results list
        results.extend(pairwise_results_for_metric)

        # Create visualization
        if output_dir:
            create_binary_metric_plot(data, metric, y, output_dir, control_genotype, n_genotypes)

    results_df = pd.DataFrame(results)
    overall_df = pd.DataFrame(overall_results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    if output_dir and not overall_df.empty:
        overall_file = output_dir / "binary_metrics_overall_tests.csv"
        overall_df.to_csv(overall_file, index=False)
        print(f"Overall chi-square tests saved to: {overall_file}")

    return results_df, overall_df


def create_binary_metric_plot(data, metric, y, output_dir, control_genotype=None, n_genotypes=2):
    """Create a visualization for binary metrics showing proportions with significance annotations"""

    import matplotlib

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Calculate proportions
    prop_data = data.groupby(y)[metric].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_success"] = prop_data["sum"].astype(int)
    prop_data["n_total"] = prop_data["count"].astype(int)

    # Sort by genotype (control first if specified)
    if control_genotype:
        prop_data["sort_key"] = prop_data[y].apply(lambda x: 0 if x == control_genotype else 1)
        prop_data = prop_data.sort_values(["sort_key", y]).reset_index(drop=True)
        prop_data = prop_data.drop(columns=["sort_key"])

    # Calculate statistical significance for each comparison vs control
    significance_results = {}
    if control_genotype and n_genotypes > 1:
        control_data_subset = data[data[y] == control_genotype][metric].dropna()
        control_success = int(control_data_subset.sum())
        control_total = len(control_data_subset)

        for _, row in prop_data.iterrows():
            genotype = row[y]
            if genotype == control_genotype:
                continue

            test_data = data[data[y] == genotype][metric].dropna()
            test_success = int(test_data.sum())
            test_total = len(test_data)

            if test_total > 0 and control_total > 0:
                try:
                    table = [
                        [test_success, test_total - test_success],
                        [control_success, control_total - control_success],
                    ]
                    fisher_result = fisher_exact(table)

                    # Extract p-value (fisher_exact returns (oddsratio, pvalue))
                    p_value = float(fisher_result[1])

                    # Determine significance level
                    if p_value < 0.001:
                        sig_level = "***"
                    elif p_value < 0.01:
                        sig_level = "**"
                    elif p_value < 0.05:
                        sig_level = "*"
                    else:
                        sig_level = "ns"

                    significance_results[genotype] = {"p_value": p_value, "sig_level": sig_level}
                except Exception:
                    pass

    # Use color palette
    n_groups = len(prop_data)
    colors = sns.color_palette("husl", n_groups)

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

    # Create the plot with two subplots
    fig_height = max(6, n_groups * 0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))

    # Plot 1: Proportion with confidence intervals (horizontal bars)
    y_positions = range(len(prop_data))
    bars = ax1.barh(y_positions, prop_data["proportion"], color=colors, alpha=0.7)

    # Add error bars
    yerr_lower = np.maximum(0, prop_data["proportion"] - prop_data["lower"])
    yerr_upper = np.maximum(0, prop_data["upper"] - prop_data["proportion"])
    xerr = [yerr_lower, yerr_upper]
    ax1.errorbar(prop_data["proportion"], y_positions, xerr=xerr, fmt="none", color="black", capsize=5)

    # Add sample sizes and significance annotations on bars
    for i, (bar, row) in enumerate(zip(bars, prop_data.itertuples())):
        width = bar.get_width()
        genotype = getattr(row, y)

        # Add sample size
        text_x = width + 0.01
        text = f"n={row.n_total}"

        # Add significance if available
        if genotype in significance_results:
            sig_level = significance_results[genotype]["sig_level"]
            if sig_level != "ns":
                text += f" {sig_level}"

        ax1.text(text_x, bar.get_y() + bar.get_height() / 2.0, text, ha="left", va="center", fontsize=10)

    ax1.set_ylabel("Genotype", fontsize=11)
    ax1.set_xlabel(f"Proportion of {metric}", fontsize=11)
    ax1.set_title(f"{metric} - Proportions with 95% CI", fontsize=12)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(prop_data[y])
    ax1.set_xlim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("white")
    fig.patch.set_facecolor("white")

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

    ax2.set_ylabel("Genotype", fontsize=11)
    ax2.set_xlabel("Count", fontsize=11)
    ax2.set_title(f"{metric} - Raw Counts", fontsize=12)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(prop_data[y])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("white")

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"{metric}_binary_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"  Binary plot saved: {output_file}")


def main(overwrite=True, test_mode=False):
    """Main function to run Mann-Whitney plots for learning mutants experiments.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample 200 rows for fast testing.
    """
    print(f"Starting Mann-Whitney U test analysis for learning mutants experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Define output directory
    base_output_dir = Path("/mnt/upramdya_data/MD/Learning_mutants/Plots/Summary_metrics/Genotype_Mannwhitney")

    # Ensure the output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # Load the dataset
    print("Loading dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Normalize Genotype values to avoid duplicates due to case/spacing
    if "Genotype" in dataset.columns:
        before = sorted(dataset["Genotype"].astype(str).unique())
        dataset, merge_report = normalize_genotypes_column(dataset, col_name="Genotype")
        after = sorted([str(x) for x in pd.Series(dataset["Genotype"]).astype(str).unique()])
        print(f"🔤 Genotype normalization: {len(before)} unique -> {len(after)} unique")
        # Print mergers where multiple variants map to same key
        merged = {k: v for k, v in merge_report.items() if len(v) > 1}
        if merged:
            print("  Merged variants (case/spacing differences):")
            for key, variants in merged.items():
                canonical = " ".join(str(variants[0]).strip().split()).lower()
                print(f"   - Key '{key}': {variants}")
        else:
            print("  No case/spacing duplicate genotypes detected.")

        # Print per-genotype individual counts after normalization
        print_genotype_individual_counts(dataset, group_col="Genotype")

    # Check for Genotype column
    if "Genotype" not in dataset.columns:
        print("❌ Genotype column not found in dataset!")
        print("Available columns that might contain genotype info:")
        potential_cols = [
            col for col in dataset.columns if any(word in col.lower() for word in ["genotype", "strain", "mutant"])
        ]
        print(potential_cols)
        return

    print(f"✅ Genotype column found")
    print(f"Genotypes in dataset: {sorted(dataset['Genotype'].unique())}")

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
            if col not in available_metrics and col != "Genotype":
                should_exclude = any(excl_pattern in col.lower() for excl_pattern in excluded_patterns)
                if should_exclude:
                    print(f"  ❌ {col} (excluded due to pattern)")
                    continue
                available_metrics.append(col)
                print(f"  ✓ {col} (matches pattern '{pattern}')")

    print(f"📊 Found {len(available_metrics)} metrics after filtering")

    # Define explicit binary metrics to look for
    all_possible_binary_metrics = ["has_finished", "has_major", "has_significant", "pulled", "pushed"]
    predefined_binary_metrics = [metric for metric in all_possible_binary_metrics if metric in available_metrics]

    print(f"Checking for binary metrics: {all_possible_binary_metrics}")
    print(f"Found predefined binary metrics in dataset: {predefined_binary_metrics}")

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
    binary_metrics = list(predefined_binary_metrics)  # Start with predefined binary metrics
    excluded_metrics = []

    for col in available_metrics:
        # Skip if already in predefined binary metrics
        if col in predefined_binary_metrics:
            continue

        category = categorize_metrics(col)

        if category == "continuous":
            continuous_metrics.append(col)
        elif category == "binary":
            if col not in binary_metrics:
                binary_metrics.append(col)
        elif category == "insufficient_data":
            non_nan_count = dataset[col].count()
            total_count = len(dataset)
            print(f"  ⚠️  {col}: Only {non_nan_count}/{total_count} non-NaN values, including anyway")
            non_nan_values = dataset[col].dropna().unique()
            if len(non_nan_values) >= 2 and set(non_nan_values) == {0, 1}:
                if col not in binary_metrics:
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
    print(f"Unique genotypes: {len(dataset['Genotype'].unique())}")

    # Validate required columns
    required_cols = ["Genotype"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Determine control genotype (alphabetically first or could be specified)
    genotypes = sorted(dataset["Genotype"].unique())
    # control_genotype = genotypes[0]
    control_genotype = "PR" if "PR" in genotypes else genotypes[0]
    print(f"\n📊 Using {control_genotype} as control genotype")

    # Process continuous metrics
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests with FDR correction) ---")
        continuous_data = dataset[["Genotype"] + continuous_metrics].copy()

        print(f"Generating Mann-Whitney plots for learning mutants experiments...")
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
                continuous_data = continuous_data[["Genotype"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            print(f"🚀 Starting Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics...")

            start_time = time.time()

            stats_df = generate_genotype_mannwhitney_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="Genotype",
                control_genotype=control_genotype,
                hue="Genotype",
                palette="Set2",
                output_dir=str(base_output_dir),
                fdr_method="fdr_bh",
                alpha=0.05,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"✅ Mann-Whitney analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Process binary metrics
    if binary_metrics:
        print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests with FDR correction) ---")
        binary_data = dataset[["Genotype"] + binary_metrics].copy()

        print(f"Analyzing binary metrics for learning mutants experiments...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        binary_output_dir = base_output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results, overall_binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            y="Genotype",
            output_dir=binary_output_dir,
            overwrite=overwrite,
            control_genotype=control_genotype,
        )

        print(f"\nBinary metrics analysis completed:")
        if not binary_results.empty:
            print(f"  Pairwise comparisons: {len(binary_results)}")
        if not overall_binary_results.empty:
            print(f"  Overall tests (chi-square): {len(overall_binary_results)}")

    print(f"\n{'='*60}")
    print("✅ Learning mutants genotype analysis complete! Check output directory for results.")
    print(f"📁 Output directory: {base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test plots for learning mutants experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_learning_mutants.py                    # Overwrite existing plots
  python run_mannwhitney_learning_mutants.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_learning_mutants.py --test             # Test mode with 3 metrics
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
