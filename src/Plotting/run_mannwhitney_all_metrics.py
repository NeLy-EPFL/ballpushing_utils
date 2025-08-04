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
    python run_mannwhitney_all_metrics.py [--overwrite]

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

from All_metrics import generate_jitterboxplots_with_mannwhitney
from Config import color_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import scipy.stats as stats_module


def load_and_clean_dataset():
    """Load and clean the dataset following the same process as All_metrics.py"""
    # Load the dataset - updated to use the newer dataset with new metrics
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250803_20_summary_TNT_screen_Data/summary/pooled_summary.feather"

    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        print("Falling back to older dataset...")
        dataset = pd.read_feather(
            "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
        )
        print(f"✅ Fallback dataset loaded! Shape: {dataset.shape}")

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Clean NA values following the same logic as All_metrics.py
    # Use safe column access to handle missing columns
    def safe_fillna(column_name, fill_value):
        if column_name in dataset.columns:
            dataset[column_name].fillna(fill_value, inplace=True)
            print(f"Filled NaN values in {column_name} with {fill_value}")
        else:
            print(f"Column {column_name} not found in dataset, skipping...")

    safe_fillna("max_event", -1)
    safe_fillna("first_significant_event", -1)
    safe_fillna("first_major_event", -1)
    safe_fillna("final_event", -1)

    # Fill time columns with 3600
    safe_fillna("max_event_time", 3600)
    safe_fillna("first_significant_event_time", 3600)
    safe_fillna("first_major_event_time", 3600)
    safe_fillna("final_event_time", 3600)

    # Drop problematic columns if they exist
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

    # Fill remaining NA values with appropriate defaults (only if columns exist)
    safe_fillna("pulling_ratio", 0)
    safe_fillna("avg_displacement_after_success", 0)
    safe_fillna("avg_displacement_after_failure", 0)
    safe_fillna("influence_ratio", 0)

    # Handle new metrics with appropriate defaults
    new_metrics_defaults = {
        "has_finished": 0,  # Binary metric, 0 if no finish detected
        "persistence_at_end": 0.0,  # Time fraction, 0 if no persistence
        "fly_distance_moved": 0.0,  # Distance, 0 if no movement detected
        "time_chamber_beginning": 0.0,  # Time, 0 if no time in beginning chamber
        "median_freeze_duration": 0.0,  # Duration, 0 if no freezes detected
        "fraction_not_facing_ball": 0.0,  # Fraction, 0 if always facing ball or no data
        "flailing": 0.0,  # Motion energy, 0 if no leg movement detected
        "head_pushing_ratio": 0.0,  # Ratio, 0.5 if no contact data (neutral)
        "compute_median_head_ball_distance": 0.0,  # Distance, 0 if no contact data
        "compute_mean_head_ball_distance": 0.0,  # Distance, 0 if no contact data
        "median_head_ball_distance": 0.0,  # Distance, 0 if no contact data
        "mean_head_ball_distance": 0.0,  # Distance, 0 if no contact data
    }

    for metric, default_value in new_metrics_defaults.items():
        if metric in dataset.columns:
            nan_count = dataset[metric].isnull().sum()
            if nan_count > 0:
                dataset[metric].fillna(default_value, inplace=True)
                print(f"Filled {nan_count} NaN values in {metric} with {default_value}")
            else:
                print(f"No NaN values found in {metric}")
        else:
            print(f"New metric {metric} not found in dataset")

    print(f"Dataset cleaning completed successfully")

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


def get_nan_annotations(data, metrics, y_col="Nickname"):
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


def analyze_binary_metrics(data, binary_metrics, y="Nickname", split_col="Split", output_dir=None, overwrite=True):
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
    if "Brain region" in data.columns:
        # Get brain region for each genotype (take the first occurrence)
        brain_region_map = data.groupby(y)["Brain region"].first().to_dict()
        prop_data["Brain region"] = prop_data[y].map(brain_region_map)
    else:
        # Fallback if Brain region column doesn't exist
        prop_data["Brain region"] = "Unknown"

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
        brain_region = row["Brain region"]
        genotype = row[y]

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
    for i, (tick, brain_region) in enumerate(zip(ax1.get_yticklabels(), prop_data["Brain region"])):
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
    for i, (tick, brain_region) in enumerate(zip(ax2.get_yticklabels(), prop_data["Brain region"])):
        if brain_region in color_dict:
            tick.set_color(color_dict[brain_region])

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"{metric}_binary_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Binary plot saved: {output_file}")
    print(f"  Sorted by proportion (descending): {prop_data['proportion'].tolist()}")


def main(overwrite=True):
    """Main function to run Mann-Whitney plots for all metrics.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    """
    print(f"Starting Mann-Whitney U test analysis for all metrics...")
    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")

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
    dataset = load_and_clean_dataset()

    # Get all metric columns (excluding metadata)
    metadata_cols = [
        "index",
        "fly",
        "flypath",
        "experiment",
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Date",
        "Genotype",
        "Period",
        "FeedingState",
        "Orientation",
        "Light",
        "Crossing",
    ]

    all_cols = dataset.columns.tolist()
    metric_cols = [col for col in all_cols if col not in metadata_cols]

    # Filter out non-numeric columns and separate binary metrics
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

    print(f"Categorizing metrics for analysis...")
    continuous_metrics = []
    binary_metrics = []
    excluded_metrics = []
    insufficient_data_metrics = []

    for col in metric_cols:
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

    # Check for new metrics specifically
    new_metrics = [
        "has_finished",
        "persistence_at_end",
        "fly_distance_moved",
        "time_chamber_beginning",
        "median_freeze_duration",
        "fraction_not_facing_ball",
        "flailing",
        "head_pushing_ratio",
        "median_head_ball_distance",
        "mean_head_ball_distance",
        "compute_median_head_ball_distance",
        "compute_mean_head_ball_distance",
    ]

    print(f"\nChecking for new metrics:")
    found_new_metrics = []
    suitable_new_metrics = []
    for metric in new_metrics:
        if metric in dataset.columns:
            print(f"  ✅ {metric}: FOUND in dataset")
            found_new_metrics.append(metric)

            # Detailed diagnostic information
            non_nan_values = dataset[metric].dropna()
            unique_vals = non_nan_values.unique()
            category = categorize_metrics(metric)

            print(
                f"    📊 Non-NaN count: {len(non_nan_values)}/{len(dataset)} ({len(non_nan_values)/len(dataset)*100:.1f}%)"
            )
            print(f"    📊 Unique values: {len(unique_vals)}")
            print(f"    📊 Category: {category}")
            print(f"    📊 Data type: {dataset[metric].dtype}")

            if len(unique_vals) <= 10:  # Show actual values for small sets
                print(f"    📊 Unique values: {sorted(unique_vals)}")
            else:
                print(f"    📊 Value range: [{non_nan_values.min():.3f}, {non_nan_values.max():.3f}]")

            if metric in continuous_metrics:
                print(f"    ✅ {metric}: INCLUDED in continuous metrics")
                suitable_new_metrics.append(metric)
            elif metric in binary_metrics:
                print(f"    ✅ {metric}: INCLUDED in binary metrics")
                suitable_new_metrics.append(metric)
            else:
                print(f"    ❌ {metric}: EXCLUDED from analysis (category: {category})")
        else:
            print(f"  ❌ {metric}: NOT FOUND in dataset")

    if found_new_metrics:
        print(f"\n🎉 Found {len(found_new_metrics)} new metrics in the dataset!")
        if suitable_new_metrics:
            print(f"📊 {len(suitable_new_metrics)} new metrics suitable for plotting: {suitable_new_metrics}")

        # Show some basic stats for new metrics that are suitable for plotting
        if suitable_new_metrics:
            print(f"\nNew metrics statistics (suitable for plotting):")
            for metric in suitable_new_metrics:
                non_null = dataset[metric].count()
                total = len(dataset)
                print(f"  {metric}: {non_null}/{total} non-null ({non_null/total*100:.1f}%)")
                if dataset[metric].dtype in ["int64", "float64"]:
                    print(f"    Range: [{dataset[metric].min():.3f}, {dataset[metric].max():.3f}]")
    else:
        print(f"\n⚠️  No new metrics found! Check if dataset generation completed successfully.")

    # Update found_new_metrics to only include suitable ones for the rest of the script
    found_new_metrics = suitable_new_metrics

    # Clean the dataset (already done in load_and_clean_dataset)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique genotypes: {len(dataset['Nickname'].unique())}")

    # Validate required columns
    required_cols = ["Nickname", "Split", "Brain region"]
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
            continuous_data = filtered_data[["Nickname", "Split", "Brain region"] + continuous_metrics].copy()

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
                    continuous_data = continuous_data[["Nickname", "Split", "Brain region"] + metrics_to_process]
                    continuous_metrics = metrics_to_process
            else:
                metrics_to_process = continuous_metrics

            if metrics_to_process:
                # Check for NaN annotations
                nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "Nickname")
                if nan_annotations:
                    print(f"📋 NaN values detected in continuous metrics:")
                    for metric, genotype_info in nan_annotations.items():
                        print(f"  {metric}: {len(genotype_info)} genotypes have NaN values")
                        for genotype, info in genotype_info.items():
                            print(f"    - {genotype}: {info['annotation']}")

                generate_jitterboxplots_with_mannwhitney(
                    continuous_data,
                    metrics=metrics_to_process,
                    output_dir=str(output_dir),
                    y="Nickname",
                    hue="Brain region",
                    palette=color_dict,
                    color_y_labels_by_brain_region=True,
                )

        # 2. Process binary metrics with Fisher's exact tests
        if binary_metrics:
            print(f"\n--- BINARY METRICS ANALYSIS (Fisher's exact tests) ---")
            # Filter to only binary metrics + required columns
            binary_data = filtered_data[["Nickname", "Split", "Brain region"] + binary_metrics].copy()

            print(f"Analyzing binary metrics for {condition_name}...")
            print(f"Binary metrics dataset shape: {binary_data.shape}")

            # Check for NaN annotations in binary metrics
            nan_annotations = get_nan_annotations(binary_data, binary_metrics, "Nickname")
            if nan_annotations:
                print(f"📋 NaN values detected in binary metrics:")
                for metric, genotype_info in nan_annotations.items():
                    print(f"  {metric}: {len(genotype_info)} genotypes have NaN values")
                    for genotype, info in genotype_info.items():
                        print(f"    - {genotype}: {info['annotation']}")

            binary_output_dir = output_dir / "binary_analysis"
            binary_output_dir.mkdir(parents=True, exist_ok=True)

            binary_results = analyze_binary_metrics(
                binary_data,
                binary_metrics,
                y="Nickname",
                split_col="Split",
                output_dir=binary_output_dir,
                overwrite=overwrite,
            )
        print(f"Unique genotypes in {condition_name}: {len(filtered_data['Nickname'].unique())}")

        # Check for data quality in new metrics for this condition
        if found_new_metrics:
            print(f"\nData quality check for new metrics in {condition_name}:")
            for metric in found_new_metrics:
                if metric in filtered_data.columns:
                    valid_count = filtered_data[metric].count()
                    total_count = len(filtered_data)
                    print(f"  {metric}: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")

                    if valid_count < 3:
                        print(f"    ⚠️  WARNING: Too few valid values for statistical testing!")
                    elif filtered_data[metric].dtype in ["int64", "float64"]:
                        print(f"    Range: [{filtered_data[metric].min():.3f}, {filtered_data[metric].max():.3f}]")
    print(f"\n{'='*60}")
    print("✅ Analysis complete! Check output directories for results.")
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
        """,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip metrics that already have existing plots (default: overwrite existing plots)",
    )

    args = parser.parse_args()

    # Run main function with overwrite parameter (invert --no-overwrite)
    main(overwrite=not args.no_overwrite)
