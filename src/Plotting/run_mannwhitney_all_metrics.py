#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for all metrics.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_mannwhitney_all_metrics.py [--overwrite] [--test]

Arguments:
    --overwrite: If specified, overwrite existing plots. If not specified, skip metrics that already have plots.
    --test: Enable test mode for faster debugging (limits metrics and samples)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent))  # Also add current directory

try:
    from All_metrics import generate_jitterboxplots_with_mannwhitney

    print("✅ Successfully imported All_metrics module")
except Exception as e:
    print(f"❌ Error importing All_metrics: {e}")
    print("This error occurs during All_metrics module initialization")
    print("Check the All_metrics.py file for issues in the module-level code")
    sys.exit(1)

from Config import color_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import scipy.stats as stats_module


def load_nickname_mapping():
    """Load the simplified nickname mapping for visualization"""
    region_map_path = "/mnt/upramdya_data/MD/Region_map_250908.csv"
    print(f"📋 Loading nickname mapping from {region_map_path}")

    try:
        region_map = pd.read_csv(region_map_path)
        # Create mapping from Nickname to Simplified Nickname
        nickname_mapping = dict(zip(region_map["Nickname"], region_map["Simplified Nickname"]))
        print(f"📋 Loaded {len(nickname_mapping)} nickname mappings")

        # Also create brain region mapping for simplified nicknames
        simplified_to_region = dict(zip(region_map["Simplified Nickname"], region_map["Simplified region"]))

        return nickname_mapping, simplified_to_region
    except Exception as e:
        print(f"⚠️  Could not load region mapping: {e}")
        return {}, {}


def load_metrics_list():
    """Load the final metrics list for PCA analysis"""
    metrics_file = "/home/matthias/ballpushing_utils/src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"
    print(f"📋 Loading metrics from {metrics_file}")

    try:
        with open(metrics_file, "r") as f:
            metrics = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(metrics)} metrics from file")
        return metrics
    except Exception as e:
        print(f"⚠️  Could not load metrics file: {e}")
        return []


def load_and_clean_dataset(test_mode=False, test_sample_size=500):
    """Load the dataset with NO DEFAULT IMPUTATION - keep real data only

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the dataset - updated to use the latest dataset with all metrics
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"

    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        print("Falling back to older dataset...")
        dataset = pd.read_feather(
            "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250803_20_summary_TNT_screen_Data/summary/pooled_summary.feather"
        )
        print(f"✅ Fallback dataset loaded! Shape: {dataset.shape}")
        print("⚠️  Using fallback dataset - some metrics may be missing")

    # Exclude problematic nicknames BEFORE any other processing (EARLY FILTERING)
    exclude_nicknames = ["PR", "CS", "TNTxCS", "Ple-Gal4.F a.k.a TH-Gal4"]
    initial_count = len(dataset)
    dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]
    excluded_count = initial_count - len(dataset)

    if excluded_count > 0:
        print(f"🚫 Excluded {excluded_count} flies from problematic nicknames: {exclude_nicknames}")
        print(f"   Remaining dataset shape: {dataset.shape}")
    else:
        print(f"ℹ️  No flies found with problematic nicknames to exclude")

    # Test mode sampling for faster debugging - EARLY SAMPLING
    if test_mode and len(dataset) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} rows from {len(dataset)} for faster processing")
        dataset = dataset.sample(n=test_sample_size, random_state=42).copy()
        print(f"   Sampled dataset shape: {dataset.shape}")

    # BATCH BOOLEAN CONVERSION for better performance
    print("🔍 Checking for boolean columns...")
    boolean_cols = [col for col in dataset.columns if dataset[col].dtype == "bool"]
    if boolean_cols:
        print(f"🔄 Converting {len(boolean_cols)} boolean columns to int...")
        for i, col in enumerate(boolean_cols, 1):
            print(f"   Converting {i}/{len(boolean_cols)}: {col}")
            dataset[col] = dataset[col].astype(int)  # NO DEFAULT IMPUTATION - Keep real data only
        print("✅ Boolean conversion completed")
    else:
        print("✅ No boolean columns found to convert")
    print(f"⚠️  NO DEFAULT IMPUTATION APPLIED - Working with real data only")
    print(f"NaN values will be preserved and handled appropriately during analysis")

    # Only drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    # Add Simplified Nickname column
    print("📋 Adding simplified nickname mappings...")
    nickname_mapping, simplified_to_region = load_nickname_mapping()
    if nickname_mapping:
        print("📋 Applying nickname mappings to dataset...")
        dataset["Simplified Nickname"] = dataset["Nickname"].map(nickname_mapping)
        print("📋 Applying brain region mappings...")
        dataset["Simplified region"] = dataset["Simplified Nickname"].map(simplified_to_region)

        # Report mapping success
        print("📋 Calculating mapping statistics...")
        mapped_count = dataset["Simplified Nickname"].notna().sum()
        total_count = len(dataset)
        print(
            f"📋 Mapped {mapped_count}/{total_count} flies to simplified nicknames ({mapped_count/total_count*100:.1f}%)"
        )

        if mapped_count < total_count:
            unmapped_nicknames = dataset[dataset["Simplified Nickname"].isna()]["Nickname"].unique()
            print(f"⚠️  Unmapped nicknames: {list(unmapped_nicknames)}")
    else:
        print(f"⚠️  Could not load nickname mapping, using original nicknames")
        dataset["Simplified Nickname"] = dataset["Nickname"]
        dataset["Simplified region"] = dataset["Brain region"]

    print(f"Dataset loading completed - NO DEFAULT IMPUTATION APPLIED")

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


def get_nan_annotations(data, metrics, y_col="Simplified Nickname"):
    """
    Generate annotations for genotypes that have NaN values in specific metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the metrics
    metrics : list
        List of metric column names to check
    y_col : str
        Column name for grouping (e.g., genotype)

    Returns:
    --------
    dict
        Dictionary mapping metric names to dictionaries of genotype: nan_info
    """
    nan_annotations = {}

    for metric in metrics:
        if metric not in data.columns:
            continue

        metric_annotations = {}

        for genotype in data[y_col].unique():
            genotype_data = data[data[y_col] == genotype][metric]
            total_count = len(genotype_data)
            nan_count = genotype_data.isnull().sum()

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                metric_annotations[genotype] = {
                    "nan_count": nan_count,
                    "total_count": total_count,
                    "nan_percentage": nan_percentage,
                    "annotation": f"({nan_count}/{total_count} NaN)",
                }

        if metric_annotations:
            nan_annotations[metric] = metric_annotations

    return nan_annotations


def analyze_binary_metrics(
    data, binary_metrics, y="Simplified Nickname", split_col="Split", output_dir=None, overwrite=True
):
    """
    Analyze binary metrics using appropriate statistical tests and visualizations.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (e.g., genotype)
    split_col : str
        Column name for split information
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

    # Determine control group
    control_nickname = None
    if split_col in data.columns:
        split_counts = data[split_col].value_counts()
        if "y" in split_counts.index:  # Changed from "Split" to "y"
            control_data = data[data[split_col] == "y"]
            if len(control_data[y].unique()) > 0:
                control_nickname = control_data[y].value_counts().index[0]  # Most common in Split group

    print(f"Using control group: {control_nickname}")

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

        # Statistical tests for each genotype vs control
        for nickname in data[y].unique():
            if nickname == control_nickname:
                continue

            # Get data for this comparison
            test_data = data[data[y] == nickname][metric].dropna()
            control_data_subset = data[data[y] == control_nickname][metric].dropna()

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

                # Significance level
                if p_value < 0.001:  # type: ignore
                    sig_level = "***"
                elif p_value < 0.01:  # type: ignore
                    sig_level = "**"
                elif p_value < 0.05:  # type: ignore
                    sig_level = "*"
                else:
                    sig_level = "ns"

                results.append(
                    {
                        "metric": metric,
                        "nickname": nickname,
                        "control": control_nickname,
                        "test_success": test_success,
                        "test_total": test_total,
                        "test_proportion": test_prop,
                        "control_success": control_success,
                        "control_total": control_total,
                        "control_proportion": control_prop,
                        "effect_size": effect_size,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                        "significant": p_value < 0.05,  # type: ignore
                        "sig_level": sig_level,
                        "test_type": "Fisher exact",
                    }
                )

                print(f"  {nickname} vs {control_nickname}: OR={odds_ratio:.3f}, p={p_value:.4f} {sig_level}")

            except Exception as e:
                print(f"  Error testing {nickname} vs {control_nickname}: {e}")

        # Create visualization
        if output_dir:
            create_binary_metric_plot(data, metric, y, output_dir, control_nickname)

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nBinary metrics statistics saved to: {stats_file}")

    return results_df


def create_binary_metric_plot(data, metric, y, output_dir, control_nickname=None):
    """Create a visualization for binary metrics showing proportions"""

    # Calculate proportions
    prop_data = data.groupby(y)[metric].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_success"] = prop_data["sum"]
    prop_data["n_total"] = prop_data["count"]

    # Sort by proportion (descending) - higher proportions first
    prop_data = prop_data.sort_values("proportion", ascending=False).reset_index(drop=True)

    # Add brain region information for coloring
    if "Simplified region" in data.columns:
        # Get brain region for each genotype (take the first occurrence)
        brain_region_map = data.groupby(y)["Simplified region"].first().to_dict()
        prop_data["Simplified region"] = prop_data[y].map(brain_region_map)
    elif "Brain region" in data.columns:
        # Fallback to original brain region column
        brain_region_map = data.groupby(y)["Brain region"].first().to_dict()
        prop_data["Simplified region"] = prop_data[y].map(brain_region_map)
    else:
        # Fallback if neither column exists
        prop_data["Simplified region"] = "Unknown"

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
    n_genotypes = len(prop_data)
    fig_height = max(10, 0.8 * n_genotypes + 4)  # Dynamic height based on number of genotypes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))

    # Use brain region colors from color_dict, similar to jitterboxplots
    colors = []
    for _, row in prop_data.iterrows():
        brain_region = str(row["Simplified region"])  # Convert to string
        genotype = str(row[y])

        # Get color from color_dict if available, otherwise use default colors
        if brain_region in color_dict:
            color = color_dict[brain_region]
        else:
            # Fallback color scheme
            color = "red" if genotype == control_nickname else "blue"
        colors.append(color)

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

    ax1.set_ylabel("Genotype")
    ax1.set_xlabel(f"Proportion of {metric}")
    ax1.set_title(f"{metric} - Proportions with 95% CI")
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(prop_data[y])
    ax1.set_xlim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Color y-axis labels by brain region (similar to jitterboxplots)
    for i, (tick, brain_region) in enumerate(zip(ax1.get_yticklabels(), prop_data["Simplified region"])):
        if brain_region in color_dict:
            tick.set_color(color_dict[brain_region])

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

    ax2.set_ylabel("Genotype")
    ax2.set_xlabel("Count")
    ax2.set_title(f"{metric} - Raw Counts")
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(prop_data[y])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Color y-axis labels by brain region for second plot too
    for i, (tick, brain_region) in enumerate(zip(ax2.get_yticklabels(), prop_data["Simplified region"])):
        if brain_region in color_dict:
            tick.set_color(color_dict[brain_region])

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
        If True, only process the first 5 metrics and sample data for fast testing.
    """
    # Record overall start time
    overall_start_time = time.time()

    print(f"Starting Mann-Whitney U test analysis for all metrics...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Processing limited metrics and sample data for faster debugging")

    # Define output directories
    base_output_dir = Path(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics/All_Metrics_bysplit_Mannwhitney"
    )

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Clean up any potential old outputs to avoid confusion
    print("Ensuring clean output directory structure...")
    for condition_subdir in [
        "mannwhitney_split",
        "mannwhitney_nosplit",
        "mannwhitney_mutant",
        "mannwhitney_all_pooled",
    ]:
        condition_dir = base_output_dir / condition_subdir
        condition_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {condition_dir}")

    # Also create the main directory for the combined statistics
    print(f"All outputs will be saved under: {base_output_dir}")

    # Load the dataset
    print("Loading dataset...")
    dataset = load_and_clean_dataset(test_mode=test_mode, test_sample_size=500)

    # Load the specific metrics from file - OPTIMIZED APPROACH
    all_metrics = load_metrics_list()

    if not all_metrics:
        print("⚠️  No metrics loaded from file, using simplified core metrics approach")
        # Use core metrics approach like in the other scripts for better performance
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
            "pulled",
            "pulling_ratio",
            "interaction_persistence",
            "distance_moved",
            "distance_ratio",
            "chamber_exit_time",
            "normalized_velocity",
            "has_finished",
            "has_major",
            "has_significant",
        ]
        metric_cols = [col for col in base_metrics if col in dataset.columns]
        print(f"📊 Using {len(metric_cols)} core metrics for simplified analysis")
    else:
        # Use metrics from file, but only those present in dataset - BATCH CHECK
        available_metrics = [col for col in all_metrics if col in dataset.columns]
        missing_metrics = [col for col in all_metrics if col not in dataset.columns]

        print(f"📊 Found {len(available_metrics)}/{len(all_metrics)} metrics in dataset")
        if missing_metrics:
            print(f"⚠️  Missing metrics: {missing_metrics}")

        metric_cols = available_metrics

    # If in test mode, limit to first 5 metrics for faster debugging
    # EARLY EXCLUSION of problematic patterns for better performance
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

    # Filter out problematic metrics early
    print(f"📊 EARLY FILTERING: Checking {len(metric_cols)} metrics for problematic patterns...")
    filtered_metrics = []
    excluded_reasons = []

    for i, metric in enumerate(metric_cols, 1):
        if i % 10 == 0:  # Progress indicator for large lists
            print(f"   Processed {i}/{len(metric_cols)} metrics...")

        # Skip if it matches any excluded pattern
        excluded_pattern = None
        for pattern in excluded_patterns:
            if pattern in metric:
                excluded_pattern = pattern
                break

        if excluded_pattern:
            excluded_reasons.append(f"{metric} (pattern: {excluded_pattern})")
            continue

        # Skip if it's non-numeric immediately
        if not pd.api.types.is_numeric_dtype(dataset[metric]):
            excluded_reasons.append(f"{metric} (non-numeric)")
            continue

        filtered_metrics.append(metric)

    excluded_count = len(metric_cols) - len(filtered_metrics)
    print(f"✅ EARLY FILTERING completed: Excluded {excluded_count} metrics, {len(filtered_metrics)} remain")
    if excluded_count > 0 and excluded_count <= 10:
        print(f"   Excluded: {excluded_reasons}")
    elif excluded_count > 10:
        print(f"   Excluded {excluded_count} metrics (first 5: {excluded_reasons[:5]})")

    metric_cols = filtered_metrics

    # If in test mode, limit to first 5 metrics for faster debugging
    if test_mode:
        original_count = len(metric_cols)
        metric_cols = metric_cols[:5]
        print(f"🧪 TEST MODE: Limited to {len(metric_cols)}/{original_count} metrics for faster processing")

    # Define binary metrics explicitly (only those that exist in dataset)
    all_possible_binary_metrics = ["has_finished", "has_major", "has_significant"]
    predefined_binary_metrics = [metric for metric in all_possible_binary_metrics if metric in metric_cols]

    print(f"Checking for binary metrics: {all_possible_binary_metrics}")
    print(f"Found binary metrics in dataset: {predefined_binary_metrics}")
    if len(predefined_binary_metrics) < len(all_possible_binary_metrics):
        missing = [m for m in all_possible_binary_metrics if m not in predefined_binary_metrics]
        print(f"⚠️  Missing binary metrics: {missing}")

    # Filter out non-numeric columns and separate binary metrics
    print(f"🔍 Categorizing {len(metric_cols)} metrics for analysis...")

    def categorize_metrics(col):
        """Categorize metrics for appropriate analysis"""
        if col not in dataset.columns:
            return None

        # Check if it's a predefined binary metric
        if col in predefined_binary_metrics:
            return "binary"

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

    print(f"📊 Starting detailed categorization of {len(metric_cols)} metrics...")
    continuous_metrics = []
    binary_metrics = []
    excluded_metrics = []
    insufficient_data_metrics = []

    for i, col in enumerate(metric_cols, 1):
        if i % 5 == 0:  # Progress indicator every 5 metrics
            print(f"   Categorizing {i}/{len(metric_cols)}: {col}")
        elif len(metric_cols) <= 10:  # Show all for small lists
            print(f"   Categorizing {i}/{len(metric_cols)}: {col}")

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
            insufficient_data_metrics.append(col)
            # Determine if it should be binary or continuous based on available data
            if col in predefined_binary_metrics:
                binary_metrics.append(col)
            else:
                non_nan_values = dataset[col].dropna().unique()
                if len(non_nan_values) >= 2 and set(non_nan_values) == {0, 1}:
                    binary_metrics.append(col)
                else:
                    continuous_metrics.append(col)
        else:
            excluded_metrics.append((col, category))

    print(f"Found {len(continuous_metrics)} continuous metrics for Mann-Whitney analysis:")
    for i, metric in enumerate(continuous_metrics, 1):
        print(f"  {i:2d}. {metric}")

    print(f"\nFound {len(binary_metrics)} binary metrics for proportion analysis:")
    for i, metric in enumerate(binary_metrics, 1):
        print(f"  {i:2d}. {metric}")

    if excluded_metrics:
        print(f"\nExcluded {len(excluded_metrics)} metrics:")
        for metric, reason in excluded_metrics:
            if metric in dataset.columns:
                unique_count = len(dataset[metric].dropna().unique())
                dtype = dataset[metric].dtype
                print(f"  - {metric} ({reason}, {dtype}, {unique_count} unique values)")

    # Use continuous metrics for the Mann-Whitney analysis
    metric_cols = continuous_metrics

    # Report metrics loaded from file
    print(f"\n📊 Metrics loaded from file:")
    print(f"  Continuous metrics: {len(continuous_metrics)}")
    print(f"  Binary metrics: {len(binary_metrics)}")
    print(f"  Total metrics for analysis: {len(continuous_metrics) + len(binary_metrics)}")

    if excluded_metrics:
        print(f"  Excluded metrics: {len(excluded_metrics)}")
        for metric, reason in excluded_metrics[:5]:  # Show first 5
            print(f"    - {metric} ({reason})")
        if len(excluded_metrics) > 5:
            print(f"    ... and {len(excluded_metrics) - 5} more")

    # Clean the dataset (already done in load_and_clean_dataset)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique genotypes: {len(dataset['Nickname'].unique())}")

    # Validate required columns
    required_cols = ["Nickname", "Simplified Nickname", "Simplified region", "Split"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Check what values exist in the Split column to debug filtering
    print(f"\n🔍 Split column values in dataset:")
    split_counts = dataset["Split"].value_counts()
    print(split_counts)
    print(f"Unique Split values: {sorted(dataset['Split'].unique())}")

    # Generate plots for each split condition + summary plot with all data
    conditions = [
        ("Split lines", lambda df: df[df["Split"] == "y"], "mannwhitney_split"),
        ("No Split lines", lambda df: df[df["Split"] == "n"], "mannwhitney_nosplit"),
        ("Mutants", lambda df: df[df["Split"] == "m"], "mannwhitney_mutant"),
        ("All Data (Pooled)", lambda df: df, "mannwhitney_all_pooled"),  # Add summary plot with all data
    ]

    total_stats = []

    for condition_name, filter_func, output_subdir in conditions:
        print(f"\n{'='*60}")
        print(f"Processing {condition_name}...")
        print("=" * 60)

        # Filter dataset for this condition
        filtered_data = filter_func(dataset)
        print(f"Filtered dataset shape: {filtered_data.shape}")

        if len(filtered_data) == 0:
            print(f"No data for {condition_name}. Skipping...")
            continue

        # Create output directory
        output_dir = base_output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Process continuous metrics with Mann-Whitney U tests
        if continuous_metrics:
            print(f"\n--- CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests) ---")
            # Filter to only continuous metrics + required columns
            continuous_data = filtered_data[
                ["Simplified Nickname", "Split", "Simplified region"] + continuous_metrics
            ].copy()

            print(f"Generating Mann-Whitney plots for {condition_name}...")
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
                    continuous_data = continuous_data[
                        ["Simplified Nickname", "Split", "Simplified region"] + metrics_to_process
                    ]
                    continuous_metrics = metrics_to_process
            else:
                metrics_to_process = continuous_metrics

            if metrics_to_process:
                condition_start_time = time.time()
                print(f"⏱️  Processing {len(metrics_to_process)} continuous metrics for {condition_name}...")

                # Check for NaN annotations (simplified for performance)
                nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "Simplified Nickname")
                if nan_annotations:
                    print(f"📋 NaN values detected in {len(nan_annotations)} metrics")

                try:
                    generate_jitterboxplots_with_mannwhitney(
                        continuous_data,
                        metrics=metrics_to_process,
                        output_dir=str(output_dir),
                        y="Simplified Nickname",
                        hue="Simplified region",
                        palette="Set2",
                        color_y_labels_by_brain_region=True,
                    )
                    condition_time = time.time() - condition_start_time
                    print(f"⏱️  {condition_name} continuous metrics completed in {condition_time:.1f}s")
                except Exception as e:
                    print(f"❌ Error processing continuous metrics for {condition_name}: {e}")
                    if test_mode:
                        print("🧪 TEST MODE: Continuing despite error...")

        # 2. Process binary metrics with Fisher's exact tests
        if binary_metrics:
            binary_start_time = time.time()
            print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
            # Filter to only binary metrics + required columns
            binary_data = filtered_data[["Simplified Nickname", "Split", "Simplified region"] + binary_metrics].copy()

            print(f"⏱️  Analyzing {len(binary_metrics)} binary metrics for {condition_name}...")
            print(f"Binary metrics dataset shape: {binary_data.shape}")

            # Check for NaN annotations in binary metrics (simplified)
            nan_annotations = get_nan_annotations(binary_data, binary_metrics, "Simplified Nickname")
            if nan_annotations:
                print(f"📋 NaN values detected in {len(nan_annotations)} binary metrics")

            binary_output_dir = output_dir / "binary_analysis"
            binary_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                binary_results = analyze_binary_metrics(
                    binary_data,
                    binary_metrics,
                    y="Simplified Nickname",
                    split_col="Split",
                    output_dir=binary_output_dir,
                    overwrite=overwrite,
                )
                binary_time = time.time() - binary_start_time
                print(f"⏱️  {condition_name} binary metrics completed in {binary_time:.1f}s")
            except Exception as e:
                print(f"❌ Error processing binary metrics for {condition_name}: {e}")
                if test_mode:
                    print("🧪 TEST MODE: Continuing despite error...")

        print(f"Unique genotypes in {condition_name}: {len(filtered_data['Simplified Nickname'].unique())}")

        # Check for data quality in predefined binary metrics for this condition
        print(f"\nData quality check for binary metrics in {condition_name}:")
        for metric in predefined_binary_metrics:
            if metric in filtered_data.columns:
                valid_count = filtered_data[metric].count()
                total_count = len(filtered_data)
                print(f"  {metric}: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")

                if valid_count < 3:
                    print(f"    ⚠️  WARNING: Too few valid values for statistical testing!")
                elif filtered_data[metric].dtype in ["int64", "float64"]:
                    print(f"    Range: [{filtered_data[metric].min():.3f}, {filtered_data[metric].max():.3f}]")

        # Memory cleanup - clear large DataFrames for this condition
        del filtered_data

        print(f"✅ Completed analysis for {condition_name}")

    # Overall timing summary
    total_time = time.time() - overall_start_time
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete! Total runtime: {total_time:.1f}s")
    print("Check output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test jitterboxplots for all metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_all_metrics.py                    # Overwrite existing plots
  python run_mannwhitney_all_metrics.py --no-overwrite     # Skip existing plots
  python run_mannwhitney_all_metrics.py --test            # Run in test mode (faster debugging)
  python run_mannwhitney_all_metrics.py --test --no-overwrite  # Test mode without overwriting
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
        help="Run in test mode with limited metrics and samples for faster debugging",
    )

    args = parser.parse_args()

    # Run main function with overwrite and test_mode parameters
    main(overwrite=not args.no_overwrite, test_mode=args.test)
