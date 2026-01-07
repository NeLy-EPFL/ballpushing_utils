#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for Gtacr OR67d ATR control experiments.

This script compares GtacrxOR67d WITH ATR (ATR='yes', the control/focal group) to:
1. GtacrxOR67d WITHOUT ATR (ATR='no')
2. GtacrxEmptyGal4 WITH ATR (ATR='yes')

The control group is GtacrxOR67d WITH ATR (ATR='yes').

This script generates comprehensive Mann-Whitney U test visualizations with:
- Comparison across Genotype+ATR groups
- FDR correction across all comparisons for each metric
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability

Usage:
    python run_mannwhitney_gtacr_atr_control.py [--no-overwrite] [--test]

Arguments:
    --no-overwrite: If specified, skip metrics that already have plots.
    --test: If specified, only process first 3 metrics for debugging.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))

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


def generate_genotype_mannwhitney_plots(
    data,
    metrics,
    y="Genotype_ATR",
    control_group=None,
    hue=None,
    figsize=(15, 10),
    output_dir="mann_whitney_plots",
    fdr_method="fdr_bh",
    alpha=0.05,
):
    """
    Generates jitterboxplots for each metric with Mann-Whitney U tests between each group and control.
    Applies FDR correction across all comparisons for each metric.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        metrics (list): List of metric names to plot.
        y (str): The name of the column for the y-axis (genotype+ATR groups). Default is "Genotype_ATR".
        control_group (str): Name of the control group. If None, uses alphabetically first group.
        hue (str, optional): The name of the column for color grouping. Default is None.
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

        # Get all groups
        groups = sorted(plot_data[y].unique())

        # Determine control group
        if control_group is None:
            control_group = groups[0]
            print(f"  Using {control_group} as control group")

        # Get all test groups (excluding control)
        test_groups = [g for g in groups if g != control_group]

        # Check if control exists
        if control_group not in plot_data[y].unique():
            print(f"Warning: Control group '{control_group}' not found for metric {metric}")
            continue

        # Get control data
        control_data = plot_data[plot_data[y] == control_group][metric].values

        if len(control_data) < 2:
            print(f"Warning: Control group '{control_group}' has insufficient data for metric {metric}")
            continue

        # Compute Mann-Whitney U tests for all test groups
        stats_results = []
        p_values = []

        for test_group in test_groups:
            test_data = plot_data[plot_data[y] == test_group][metric].values

            if len(test_data) < 2:
                print(f"  Warning: Group '{test_group}' has insufficient data, skipping")
                stats_results.append(
                    {
                        "Metric": metric,
                        "Control": control_group,
                        "Test": test_group,
                        "control_median": np.nan,
                        "test_median": np.nan,
                        "effect_size": np.nan,
                        "U_statistic": np.nan,
                        "pval": np.nan,
                        "n_control": len(control_data),
                        "n_test": len(test_data),
                    }
                )
                p_values.append(1.0)
                continue

            # Mann-Whitney U test
            try:
                u_stat, p_val = mannwhitneyu(control_data, test_data, alternative="two-sided")
            except Exception as e:
                print(f"  Error computing Mann-Whitney U for {test_group}: {e}")
                stats_results.append(
                    {
                        "Metric": metric,
                        "Control": control_group,
                        "Test": test_group,
                        "control_median": np.median(control_data),
                        "test_median": np.median(test_data),
                        "effect_size": np.median(test_data) - np.median(control_data),
                        "U_statistic": np.nan,
                        "pval": np.nan,
                        "n_control": len(control_data),
                        "n_test": len(test_data),
                    }
                )
                p_values.append(1.0)
                continue

            # Store results
            control_median = np.median(control_data)
            test_median = np.median(test_data)
            effect_size = test_median - control_median

            stats_results.append(
                {
                    "Metric": metric,
                    "Control": control_group,
                    "Test": test_group,
                    "control_median": control_median,
                    "test_median": test_median,
                    "effect_size": effect_size,
                    "U_statistic": u_stat,
                    "pval": p_val,
                    "n_control": len(control_data),
                    "n_test": len(test_data),
                }
            )
            p_values.append(p_val)

        # Apply FDR correction
        if len(p_values) > 0:
            p_values_array = np.array(p_values)
            rejected, p_corrected, _, _ = multipletests(p_values_array, alpha=alpha, method=fdr_method)

            for i, result in enumerate(stats_results):
                result["pval_corrected"] = p_corrected[i]
                result["significant"] = rejected[i]

                # Determine direction
                if rejected[i]:
                    if result["effect_size"] > 0:
                        result["direction"] = "increased"
                    else:
                        result["direction"] = "decreased"
                else:
                    result["direction"] = "none"

        all_stats.extend(stats_results)

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Fixed order: GtacrxOR67d (ATR=no), GtacrxOR67d (ATR=yes), GtacrxEmptyGal4 (ATR=yes)
        group_order = ["GtacrxOR67d (ATR=no)", "GtacrxOR67d (ATR=yes)", "GtacrxEmptyGal4 (ATR=yes)"]

        # Filter to only include groups that exist in the data
        group_order = [g for g in group_order if g in groups]

        # Reorder plot data
        plot_data[y] = pd.Categorical(plot_data[y], categories=group_order, ordered=True)
        plot_data = plot_data.sort_values(y)

        # Plot boxplot with no fill, black lines (dashed for ATR=no, solid for ATR=yes)
        # Horizontal orientation
        # Extract ATR status from group names to determine line style
        box_styles = []
        for group in group_order:
            if "ATR=no" in group:
                box_styles.append('--')  # Dashed for ATR=no
            else:
                box_styles.append('-')   # Solid for ATR=yes (or NaN, treated as yes)

        bp = ax.boxplot(
            [plot_data[plot_data[y] == group][metric].values for group in group_order],
            positions=range(len(group_order)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            vert=False,  # Horizontal orientation
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color='black')
        )

        # Style boxes: no fill, black edges, dashed/solid based on ATR
        for patch, linestyle in zip(bp['boxes'], box_styles):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
            patch.set_linestyle(linestyle)

        # Style whiskers and caps with matching line styles
        for i, linestyle in enumerate(box_styles):
            bp['whiskers'][i*2].set_linestyle(linestyle)
            bp['whiskers'][i*2+1].set_linestyle(linestyle)
            bp['caps'][i*2].set_linestyle(linestyle)
            bp['caps'][i*2+1].set_linestyle(linestyle)

        # Add scatter points: purple for OR67d genotypes, gray otherwise
        for i, group in enumerate(group_order):
            group_data = plot_data[plot_data[y] == group][metric].values
            if len(group_data) > 0:
                # Determine color based on genotype
                if "OR67d" in group:
                    color = '#9467bd'  # Purple
                else:
                    color = '#7f7f7f'  # Gray

                # Add jitter
                np.random.seed(42)
                jitter = np.random.normal(0, 0.08, size=len(group_data))
                ax.scatter(group_data, i + jitter, alpha=0.5, s=50, color=color,
                          edgecolors='black', linewidths=0.5, zorder=3)

        # Add color-coded backgrounds based on significance (horizontal orientation)
        # Create a lookup for significance
        sig_lookup = {}
        for result in stats_results:
            test_group = result["Test"]
            if result["significant"]:
                if result["direction"] == "increased":
                    sig_lookup[test_group] = "increased"
                else:
                    sig_lookup[test_group] = "decreased"
            else:
                sig_lookup[test_group] = "neutral"

        # Get x-axis limits for rectangles
        x_min, x_max = ax.get_xlim()

        for i, group in enumerate(group_order):
            if group == control_group:
                color = "lightgray"
            elif group in sig_lookup:
                if sig_lookup[group] == "increased":
                    color = "lightgreen"  # Green for increased
                elif sig_lookup[group] == "decreased":
                    color = "lightcoral"  # Red for decreased
                else:
                    color = "lightyellow"
            else:
                color = "lightyellow"

            # Rectangle spans horizontally for horizontal boxplots
            rect = Rectangle(
                (x_min, i - 0.4),
                x_max - x_min,
                0.8,
                facecolor=color,
                alpha=0.3,
                zorder=0,
            )
            ax.add_patch(rect)

        # Add significance annotations with p-values (horizontal orientation)
        x_max = plot_data[metric].max()
        x_range = plot_data[metric].max() - plot_data[metric].min()
        annotation_x = x_max + 0.05 * x_range
        pvalue_x = x_max + 0.12 * x_range

        for i, group in enumerate(group_order):
            if group == control_group:
                continue

            result = next((r for r in stats_results if r["Test"] == group), None)
            if result:
                p_val = result["pval_corrected"]
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "ns"

                # Add significance marker
                ax.text(annotation_x, i, sig_marker, ha='left', va='center',
                       fontsize=12, fontweight='bold')

                # Add p-value text
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else f"p<0.001"
                ax.text(pvalue_x, i, p_text, ha='left', va='center',
                       fontsize=9, style='italic')

        # Formatting
        ax.set_ylabel("Group (Genotype + ATR)", fontsize=12, fontweight="bold")
        ax.set_xlabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(
            f"Mann-Whitney U Test: {metric}\n(Control: {control_group}, FDR-corrected)", fontsize=14, fontweight="bold"
        )
        ax.set_yticks(range(len(group_order)))
        ax.set_yticklabels(group_order)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="lightgray", alpha=0.3, label="Control"),
            Patch(facecolor="lightgreen", alpha=0.3, label="Significantly increased"),
            Patch(facecolor="lightyellow", alpha=0.3, label="Not significant"),
            Patch(facecolor="lightcoral", alpha=0.3, label="Significantly decreased"),
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='ATR=yes'),
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='ATR=no'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=8, label='OR67d genotype'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', markersize=8, label='Other genotype'),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric}_mannwhitney.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"{metric}_mannwhitney.pdf", bbox_inches="tight")
        plt.close()

        metric_elapsed = time.time() - metric_start_time
        print(f"  Saved: {output_file} ({metric_elapsed:.2f}s)")

    # Create statistics dataframe
    stats_df = pd.DataFrame(all_stats)

    # Save statistics
    stats_file = output_dir / "genotype_atr_mannwhitney_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✅ Statistics saved to: {stats_file}")

    return stats_df


def load_and_clean_dataset(test_mode=False, test_sample_size=200):
    """
    Load and clean the Gtacr OR67d dataset.

    Parameters:
    -----------
    test_mode : bool
        If True, sample the dataset for quick testing
    test_sample_size : int
        Number of rows to sample in test mode
    """
    dataset_path = (
        "/mnt/upramdya_data/MD/Gtacr/Datasets/251219_10_summary_OR67d_Gtacr_Data/summary/pooled_summary.feather"
    )

    print(f"Loading Gtacr OR67d dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Filter out experiments with Date == 230719
    dataset = dataset[dataset["Date"] != 230719]
    print(f"Dataset shape after filtering Date == 230719: {dataset.shape}")

    # Handle ATR column - DON'T fill NaN, keep them as-is for proper group separation
    if "ATR" in dataset.columns:
        # Convert categorical to object type if needed
        if pd.api.types.is_categorical_dtype(dataset["ATR"]):
            dataset["ATR"] = dataset["ATR"].astype(str)

        # Replace 'nan' string with actual NaN for proper handling
        dataset.loc[dataset["ATR"] == "nan", "ATR"] = pd.NA

        atr_values = dataset["ATR"].value_counts(dropna=False)
        print(f"\nATR values (including NaN):")
        print(atr_values)

        # For display purposes, we'll treat NaN as "yes" (implicit ATR presence)
        # but keep them separate in the data
        dataset["ATR_display"] = dataset["ATR"].fillna("yes")
    else:
        print("\nATR column not found - cannot perform ATR control analysis")
        raise ValueError("ATR column required for this analysis")

    # Harmonize genotype naming (synonyms)
    if "Genotype" in dataset.columns:
        genotype_mapping = {
            "OGS32xOR67d": "GtacrxOR67d",
            "OGS32xEmptyGal4": "GtacrxEmptyGal4",
        }

        # Apply mapping
        original_genotypes = dataset["Genotype"].value_counts()
        print(f"\nGenotypes before harmonization:")
        print(original_genotypes)

        dataset["Genotype"] = dataset["Genotype"].replace(genotype_mapping)

        harmonized_genotypes = dataset["Genotype"].value_counts()
        print(f"\nGenotypes after harmonization:")
        print(harmonized_genotypes)

    # Create combined Genotype_ATR column using display version
    # NaN ATR will show as "yes" in the label
    dataset["Genotype_ATR"] = dataset["Genotype"] + " (ATR=" + dataset["ATR_display"] + ")"

    print(f"\nGenotype_ATR groups created:")
    print(dataset["Genotype_ATR"].value_counts())

    # Also show actual ATR breakdown
    print(f"\nActual ATR distribution by Genotype:")
    for genotype in sorted(dataset["Genotype"].unique()):
        print(f"  {genotype}:")
        atr_counts = dataset[dataset["Genotype"] == genotype]["ATR"].value_counts(dropna=False)
        for atr_val, count in atr_counts.items():
            if pd.isna(atr_val):
                print(f"    NaN (will be treated as 'yes'): {count}")
            else:
                print(f"    {atr_val}: {count}")  # Drop metadata columns
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
        print(f"\nDropped metadata columns: {columns_to_drop}")

    # Test mode sampling
    if test_mode:
        original_shape = dataset.shape
        dataset = dataset.sample(n=min(test_sample_size, len(dataset)), random_state=42)
        print(f"\n🧪 TEST MODE: Sampled {dataset.shape[0]} rows from {original_shape[0]}")

    return dataset


def should_skip_metric(metric, output_dir, overwrite):
    """Check if a metric should be skipped based on existing files"""
    if overwrite:
        return False

    output_file = output_dir / f"{metric}_mannwhitney.png"
    return output_file.exists()


def analyze_binary_metrics(data, binary_metrics, y, output_dir, overwrite=True, control_group=None):
    """Analyze binary metrics using Fisher's exact test"""
    print(f"\n📊 Analyzing {len(binary_metrics)} binary metrics...")

    all_results = []

    for metric in binary_metrics:
        if not overwrite and should_skip_metric(metric, output_dir, overwrite):
            print(f"  Skipping {metric} (already exists)")
            continue

        print(f"\n  Analyzing binary metric: {metric}")

        # Get data
        metric_data = data[[y, metric]].dropna()

        # Get all groups
        groups = sorted(metric_data[y].unique())

        # Determine control
        if control_group is None:
            control_group = groups[0]

        test_groups = [g for g in groups if g != control_group]

        # Get control data
        control_vals = metric_data[metric_data[y] == control_group][metric].values

        # Run Fisher's exact test for each test group
        for test_group in test_groups:
            test_vals = metric_data[metric_data[y] == test_group][metric].values

            # Create contingency table
            control_pos = np.sum(control_vals == 1)
            control_neg = np.sum(control_vals == 0)
            test_pos = np.sum(test_vals == 1)
            test_neg = np.sum(test_vals == 0)

            contingency = [[control_pos, control_neg], [test_pos, test_neg]]

            # Fisher's exact test
            try:
                odds_ratio, p_value = fisher_exact(contingency)
            except Exception as e:
                print(f"    Error computing Fisher's exact test for {test_group}: {e}")
                continue

            result = {
                "Metric": metric,
                "Control": control_group,
                "Test": test_group,
                "control_positive": control_pos,
                "control_negative": control_neg,
                "test_positive": test_pos,
                "test_negative": test_neg,
                "odds_ratio": odds_ratio,
                "pval": p_value,
            }

            all_results.append(result)

    # Apply FDR correction
    if all_results:
        results_df = pd.DataFrame(all_results)
        p_values = results_df["pval"].values
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        results_df["pval_corrected"] = p_corrected
        results_df["significant"] = rejected

        # Save results
        results_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Binary metrics statistics saved to: {results_file}")

        return results_df

    return None


def main(overwrite=True, test_mode=False):
    """Main function to run Mann-Whitney plots for Gtacr OR67d ATR control experiments.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 3 metrics and sample 200 rows for fast testing.
    """
    print(f"Starting Mann-Whitney U test analysis for Gtacr OR67d ATR control experiments...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Only processing first 3 metrics for debugging")

    # Load the dataset
    print("Loading dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=200)

    # Filter to only include the groups we want:
    # - GtacrxOR67d (ATR=yes) - CONTROL (the focal group WITH ATR)
    # - GtacrxOR67d (ATR=no) - Test group (same genotype WITHOUT ATR)
    # - GtacrxEmptyGal4 (ATR=yes) - Test group (genetic control WITH ATR)

    control_group_name = "GtacrxOR67d (ATR=yes)"
    test_groups = ["GtacrxOR67d (ATR=no)", "GtacrxEmptyGal4 (ATR=yes)"]

    groups_to_include = [control_group_name] + test_groups

    print(f"\n📊 Filtering to ATR control comparison groups:")
    print(f"  Control: {control_group_name}")
    print(f"  Test groups: {test_groups}")

    dataset = dataset[dataset["Genotype_ATR"].isin(groups_to_include)].copy()
    print(f"\nDataset shape after filtering: {dataset.shape}")

    # Check if we have all required groups
    available_groups = dataset["Genotype_ATR"].unique()
    print(f"\nAvailable groups: {sorted(available_groups)}")

    missing_groups = set(groups_to_include) - set(available_groups)
    if missing_groups:
        print(f"⚠️  Warning: Missing groups: {missing_groups}")

    if control_group_name not in available_groups:
        print(f"❌ Control group '{control_group_name}' not found in dataset!")
        return

    # Check sample sizes
    if "fly" in dataset.columns:
        print(f"\nSample sizes per group:")
        for group in sorted(dataset["Genotype_ATR"].unique()):
            n_flies = dataset[dataset["Genotype_ATR"] == group]["fly"].nunique()
            print(f"  {group}: {n_flies} flies")

    # Define output directory
    base_output_dir = Path("/mnt/upramdya_data/MD/Gtacr/Plots/summaries/Genotype_ATR_Control")

    # Ensure the output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

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
        "time_to_first_interaction",
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
            if col not in available_metrics and col != "Genotype_ATR":
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
    print(f"Unique groups: {len(dataset['Genotype_ATR'].unique())}")

    # Validate required columns
    required_cols = ["Genotype_ATR"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    print(f"\n📊 Using {control_group_name} as control group")

    # Process continuous metrics
    if continuous_metrics:
        print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests with FDR correction) ---")
        continuous_data = dataset[["Genotype_ATR"] + continuous_metrics].copy()

        print(f"Generating Mann-Whitney plots for ATR control experiments...")
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
                continuous_data = continuous_data[["Genotype_ATR"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            print(f"🚀 Starting Mann-Whitney analysis for {len(metrics_to_process)} continuous metrics...")

            start_time = time.time()

            stats_df = generate_genotype_mannwhitney_plots(
                continuous_data,
                metrics=metrics_to_process,
                y="Genotype_ATR",
                control_group=control_group_name,
                hue="Genotype_ATR",
                output_dir=str(base_output_dir),
                fdr_method="fdr_bh",
                alpha=0.05,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"✅ Mann-Whitney analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Process binary metrics
    if binary_metrics:
        print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
        binary_data = dataset[["Genotype_ATR"] + binary_metrics].copy()

        print(f"Analyzing binary metrics for ATR control experiments...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        binary_output_dir = base_output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        binary_results = analyze_binary_metrics(
            binary_data,
            binary_metrics,
            y="Genotype_ATR",
            output_dir=binary_output_dir,
            overwrite=overwrite,
            control_group=control_group_name,
        )

    print(f"\n{'='*60}")
    print("✅ Gtacr OR67d ATR control analysis complete! Check output directory for results.")
    print(f"📁 Output directory: {base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test plots for Gtacr OR67d ATR control experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_gtacr_atr_control.py                    # Overwrite existing plots
  python run_mannwhitney_gtacr_atr_control.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_gtacr_atr_control.py --test             # Test mode with 3 metrics
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
