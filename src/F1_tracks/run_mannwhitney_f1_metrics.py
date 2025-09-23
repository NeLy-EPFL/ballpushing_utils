#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for F1 experiments.

This script is adapted for F1 experiments with the following key differences:
1) Focuses on test ball data specifically (subsets by ball_identity == 'test')
2) Compares pretraining conditions instead of genotypes (Simplified Nickname)
3) Uses F1-specific dataset location

The script generates comprehensive Mann-Whitney U test visualizations with:
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_mannwhitney_f1_metrics.py [--no-overwrite] [--test] [--metrics METRIC1 METRIC2 ...]

Arguments:
    --no-overwrite: Skip metrics that already have existing plots (default: overwrite existing plots)
    --test: Enable test mode for faster debugging (limits metrics and samples)
    --metrics: Process only specific metrics (space-separated list)
    --ball-identity: Specify which ball identity to analyze ('test', 'training', or 'both') (default: test)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))  # Go up to src directory
sys.path.append(str(Path(__file__).parent.parent / "Plotting"))  # Add Plotting directory

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


def load_f1_dataset(test_mode=False, test_sample_size=500):
    """
    Load the F1 dataset with NO DEFAULT IMPUTATION - keep real data only

    Parameters:
    -----------
    test_mode : bool
        If True, sample a subset of data for faster processing
    test_sample_size : int
        Number of samples to use in test mode
    """
    # Load the F1 dataset
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/250918_15_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    print(f"Loading F1 dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ F1 dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ F1 dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Could not find F1 dataset at {dataset_path}")

    print(f"Original F1 dataset shape: {dataset.shape}")

    # Check ball identity column
    if "ball_identity" not in dataset.columns:
        print("❌ ball_identity column not found in F1 dataset!")
        print(f"Available columns: {list(dataset.columns)}")
        raise ValueError("F1 dataset must contain ball_identity column")

    # Check ball identity values
    ball_identities = dataset["ball_identity"].value_counts()
    print(f"Ball identity distribution:")
    for identity, count in ball_identities.items():
        print(f"  {identity}: {count} rows")

    # Check pretraining column
    pretraining_col = None
    possible_pretraining_cols = ["Pretraining", "pretraining", "Pre-training", "pre_training"]
    for col in possible_pretraining_cols:
        if col in dataset.columns:
            pretraining_col = col
            break

    if pretraining_col is None:
        print(f"❌ Pretraining column not found in F1 dataset!")
        print(f"Available columns: {list(dataset.columns)}")
        print(f"Looked for: {possible_pretraining_cols}")
        raise ValueError("F1 dataset must contain a pretraining condition column")

    print(f"Using pretraining column: {pretraining_col}")

    # Check pretraining values
    pretraining_values = dataset[pretraining_col].value_counts()
    print(f"Pretraining condition distribution:")
    for condition, count in pretraining_values.items():
        print(f"  {condition}: {count} rows")

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

    # Rename the pretraining column to a standard name for easier processing
    if pretraining_col != "Pretraining":
        dataset = dataset.rename(columns={pretraining_col: "Pretraining"})
        print(f"Renamed {pretraining_col} to Pretraining for standardization")

    print(f"F1 dataset loading completed - NO DEFAULT IMPUTATION APPLIED")
    return dataset


def subset_dataset_by_ball_identity(dataset, ball_identity="test"):
    """
    Subset the dataset by ball identity

    Parameters:
    -----------
    dataset : pd.DataFrame
        F1 dataset with ball_identity column
    ball_identity : str
        Ball identity to filter by ('test', 'training', or 'both')

    Returns:
    --------
    pd.DataFrame
        Filtered dataset
    """
    if ball_identity == "both":
        print(f"🎾 Using all ball identities for analysis")
        return dataset.copy()

    print(f"🎾 Filtering dataset to ball_identity == '{ball_identity}'")

    # Check available ball identities
    available_identities = dataset["ball_identity"].unique()
    print(f"Available ball identities: {list(available_identities)}")

    if ball_identity not in available_identities:
        print(f"❌ Requested ball_identity '{ball_identity}' not found in dataset!")
        print(f"Available options: {list(available_identities)}")
        raise ValueError(f"Ball identity '{ball_identity}' not found in dataset")

    # Filter the dataset
    filtered_dataset = dataset[dataset["ball_identity"] == ball_identity].copy()

    print(f"Filtered dataset shape: {filtered_dataset.shape}")
    print(f"Original dataset shape: {dataset.shape}")
    print(f"Kept {len(filtered_dataset)}/{len(dataset)} rows ({len(filtered_dataset)/len(dataset)*100:.1f}%)")

    # Check pretraining distribution after filtering
    pretraining_dist = filtered_dataset["Pretraining"].value_counts()
    print(f"Pretraining distribution after filtering:")
    for condition, count in pretraining_dist.items():
        print(f"  {condition}: {count} flies")

    return filtered_dataset


def create_pretraining_colors():
    """
    Create color mapping for pretraining conditions

    Returns:
    --------
    dict
        Color mapping for pretraining conditions
    """
    # Define colors for common pretraining conditions
    pretraining_colors = {
        "y": "#1f77b4",  # Blue for yes/pretrained
        "n": "#ff7f0e",  # Orange for no/untrained
        "yes": "#1f77b4",  # Blue
        "no": "#ff7f0e",  # Orange
        "pretrained": "#1f77b4",
        "untrained": "#ff7f0e",
        "control": "#2ca02c",  # Green for control
        "naive": "#d62728",  # Red for naive
    }

    return pretraining_colors


def load_f1_metrics_list():
    """Load metrics list suitable for F1 analysis"""

    # First try to load from a specific F1 metrics file if it exists
    f1_metrics_file = "/home/matthias/ballpushing_utils/src/F1_tracks/f1_metrics_list.txt"

    if Path(f1_metrics_file).exists():
        print(f"📋 Loading F1-specific metrics from {f1_metrics_file}")
        try:
            with open(f1_metrics_file, "r") as f:
                metrics = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(metrics)} F1-specific metrics from file")
            return metrics
        except Exception as e:
            print(f"⚠️  Could not load F1 metrics file: {e}")

    # Fallback to general metrics file
    general_metrics_file = "/home/matthias/ballpushing_utils/src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"

    if Path(general_metrics_file).exists():
        print(f"📋 Loading general metrics from {general_metrics_file}")
        try:
            with open(general_metrics_file, "r") as f:
                metrics = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(metrics)} general metrics from file")
            return metrics
        except Exception as e:
            print(f"⚠️  Could not load general metrics file: {e}")

    # Final fallback to core F1 metrics
    print("📋 Using core F1 metrics as fallback")
    core_f1_metrics = [
        # Core behavioral metrics
        "nb_events",
        "max_event",
        "max_event_time",
        "final_event",
        "final_event_time",
        "max_distance",
        "distance_moved",
        "distance_ratio",
        "chamber_time",
        "chamber_ratio",
        "chamber_exit_time",
        # Success metrics
        "has_finished",
        "has_major",
        "has_significant",
        "nb_significant_events",
        "significant_ratio",
        # Timing metrics
        "first_significant_event",
        "first_significant_event_time",
        "first_major_event",
        "first_major_event_time",
        # Movement patterns
        "pulled",
        "pushed",
        "pulling_ratio",
        "interaction_persistence",
        "normalized_velocity",
        "velocity_during_interactions",
        # Pause behavior
        "nb_long_pauses",
        "median_long_pause_duration",
        "total_pause_duration",
        "pauses_persistence",
        "nb_freeze",
        "median_freeze_duration",
        # Advanced metrics
        "interaction_proportion",
        "cumulated_breaks_duration",
        "fly_distance_moved",
        "persistence_at_end",
        "time_chamber_beginning",
        # Orientation metrics (if available)
        "fraction_not_facing_ball",
        "head_pushing_ratio",
        "leg_visibility_ratio",
        "flailing",
    ]

    print(f"📋 Using {len(core_f1_metrics)} core F1 metrics")
    return core_f1_metrics


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


def get_nan_annotations(data, metrics, y_col="Pretraining"):
    """
    Generate annotations for pretraining conditions that have NaN values in specific metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the metrics
    metrics : list
        List of metric column names to check
    y_col : str
        Column name for grouping (pretraining conditions)

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


def analyze_binary_metrics_f1(data, binary_metrics, y="Pretraining", output_dir=None, overwrite=True):
    """
    Analyze binary metrics for F1 experiments using appropriate statistical tests.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (pretraining conditions)
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
    print(f"F1 BINARY METRICS ANALYSIS")
    print(f"{'='*50}")

    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine control group (typically 'n' for no pretraining or 'control')
    control_condition = None
    pretraining_values = data[y].value_counts()
    print(f"Pretraining condition distribution: {dict(pretraining_values)}")

    # Prioritize common control conditions
    control_candidates = ["n", "no", "control", "untrained", "naive"]
    for candidate in control_candidates:
        if candidate in pretraining_values.index:
            control_condition = candidate
            break

    if control_condition is None:
        # Use the most common condition as control
        control_condition = pretraining_values.index[0]

    print(f"Using control condition: {control_condition}")

    for metric in binary_metrics:
        print(f"\nAnalyzing F1 binary metric: {metric}")

        # Check if we should skip this metric
        if output_dir and should_skip_metric(metric, output_dir, overwrite):
            continue

        # Create contingency table
        contingency = pd.crosstab(data[y], data[metric], margins=True)
        print(f"Contingency table for {metric}:")
        print(contingency)

        # Calculate proportions for each condition
        proportions = data.groupby(y)[metric].agg(["count", "sum", "mean"]).round(4)
        proportions["proportion"] = proportions["mean"]
        proportions["n_success"] = proportions["sum"]
        proportions["n_total"] = proportions["count"]
        print(f"\nProportions for {metric}:")
        print(proportions[["n_success", "n_total", "proportion"]])

        # Statistical tests for each condition vs control
        for condition in data[y].unique():
            if condition == control_condition:
                continue

            # Get data for this comparison
            test_data = data[data[y] == condition][metric].dropna()
            control_data_subset = data[data[y] == control_condition][metric].dropna()

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
                odds_ratio = fisher_result[0]
                p_value = fisher_result[1]

                # Effect size (difference in proportions)
                test_prop = test_success / test_total if test_total > 0 else 0
                control_prop = control_success / control_total if control_total > 0 else 0
                effect_size = test_prop - control_prop

                # Significance level
                if p_value < 0.001:
                    sig_level = "***"
                elif p_value < 0.01:
                    sig_level = "**"
                elif p_value < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                results.append(
                    {
                        "metric": metric,
                        "condition": condition,
                        "control": control_condition,
                        "test_success": test_success,
                        "test_total": test_total,
                        "test_proportion": test_prop,
                        "control_success": control_success,
                        "control_total": control_total,
                        "control_proportion": control_prop,
                        "effect_size": effect_size,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "sig_level": sig_level,
                        "test_type": "Fisher exact",
                    }
                )

                print(f"  {condition} vs {control_condition}: OR={odds_ratio:.3f}, p={p_value:.4f} {sig_level}")

            except Exception as e:
                print(f"  Error testing {condition} vs {control_condition}: {e}")

        # Create visualization
        if output_dir:
            create_binary_metric_plot_f1(data, metric, y, output_dir, control_condition)

    results_df = pd.DataFrame(results)

    if output_dir and not results_df.empty:
        stats_file = output_dir / "f1_binary_metrics_statistics.csv"
        results_df.to_csv(stats_file, index=False)
        print(f"\nF1 binary metrics statistics saved to: {stats_file}")

    return results_df


def create_f1_pretraining_plots(data, metrics, output_dir, y_col="Pretraining"):
    """
    Create simple Mann-Whitney U comparison plots for F1 experiments.

    This is a simplified plotting function that directly compares pretraining conditions
    without the complexity of split-based analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with metrics and pretraining conditions
    metrics : list
        List of metric names to plot
    output_dir : Path
        Directory to save plots
    y_col : str
        Column name for pretraining conditions
    """
    from scipy.stats import mannwhitneyu
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pretraining colors
    pretraining_colors = create_pretraining_colors()

    # Determine control group (typically 'n' for no pretraining)
    pretraining_values = data[y_col].unique()
    control_condition = "n" if "n" in pretraining_values else pretraining_values[0]
    experimental_condition = [c for c in pretraining_values if c != control_condition][0]

    print(f"Control condition: {control_condition}")
    print(f"Experimental condition: {experimental_condition}")

    # Process each metric
    for metric in metrics:
        print(f"  Creating F1 plot for: {metric}")

        # Get data for this metric
        metric_data = data.dropna(subset=[metric, y_col])

        if len(metric_data) == 0:
            print(f"    ⚠️  No data available for {metric}")
            continue

        # Get control and experimental data
        control_data = metric_data[metric_data[y_col] == control_condition][metric]
        experimental_data = metric_data[metric_data[y_col] == experimental_condition][metric]

        if len(control_data) == 0 or len(experimental_data) == 0:
            print(f"    ⚠️  Insufficient data for comparison in {metric}")
            continue

        # Perform Mann-Whitney U test
        try:
            statistic, p_value = mannwhitneyu(experimental_data, control_data, alternative="two-sided")

            # Determine significance level
            if p_value < 0.001:
                sig_level = "***"
            elif p_value < 0.01:
                sig_level = "**"
            elif p_value < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"

        except Exception as e:
            print(f"    ⚠️  Error in statistical test for {metric}: {e}")
            p_value = 1.0
            sig_level = "ns"

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create boxplot with jitter
        sns.boxplot(
            data=metric_data,
            x=y_col,
            y=metric,
            hue=y_col,
            palette=[
                pretraining_colors.get(control_condition, "#ff7f0e"),
                pretraining_colors.get(experimental_condition, "#1f77b4"),
            ],
            legend=False,
            ax=ax,
        )

        # Add jitter points
        sns.stripplot(data=metric_data, x=y_col, y=metric, color="black", alpha=0.6, size=3, ax=ax)

        # Add statistical annotation
        y_max = metric_data[metric].max()
        y_min = metric_data[metric].min()
        y_range = y_max - y_min
        annotation_y = y_max + 0.05 * y_range

        ax.text(
            0.5,
            annotation_y,
            f"Mann-Whitney U: p={p_value:.4f} {sig_level}",
            ha="center",
            va="bottom",
            transform=ax.transData,
            fontsize=10,
            weight="bold" if sig_level != "ns" else "normal",
        )

        # Add sample sizes
        for i, condition in enumerate([control_condition, experimental_condition]):
            n = len(metric_data[metric_data[y_col] == condition])
            ax.text(i, y_min - 0.05 * y_range, f"n={n}", ha="center", va="top", fontsize=9)

        # Formatting
        ax.set_title(f"F1 Analysis: {metric}", fontsize=14, weight="bold")
        ax.set_xlabel("Pretraining Condition", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set background color based on significance
        if sig_level != "ns":
            control_median = control_data.median()
            experimental_median = experimental_data.median()

            if experimental_median > control_median:
                # Experimental higher than control - light green background
                ax.set_facecolor("#e8f5e8")
            else:
                # Experimental lower than control - light coral background
                ax.set_facecolor("#ffeaea")

        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"f1_{metric}_pretraining_comparison.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"    ✅ Saved: {output_file.name}")


def create_binary_metric_plot_f1(data, metric, y, output_dir, control_condition=None):
    """Create a visualization for F1 binary metrics showing proportions by pretraining condition"""

    # Calculate proportions
    prop_data = data.groupby(y)[metric].agg(["count", "sum", "mean"]).reset_index()
    prop_data["proportion"] = prop_data["mean"]
    prop_data["n_success"] = prop_data["sum"]
    prop_data["n_total"] = prop_data["count"]

    # Sort by proportion (descending) - higher proportions first
    prop_data = prop_data.sort_values("proportion", ascending=False).reset_index(drop=True)

    # Get pretraining colors
    pretraining_colors = create_pretraining_colors()

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

    # Create the plot
    n_conditions = len(prop_data)
    fig_height = max(8, 0.8 * n_conditions + 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))

    # Use pretraining condition colors
    colors = []
    for _, row in prop_data.iterrows():
        condition = str(row[y])
        if condition in pretraining_colors:
            color = pretraining_colors[condition]
        else:
            # Default colors
            color = "red" if condition == control_condition else "blue"
        colors.append(color)

    # Plot 1: Proportion with confidence intervals (horizontal bars)
    y_positions = range(len(prop_data))
    bars = ax1.barh(y_positions, prop_data["proportion"], color=colors, alpha=0.7)

    # Add error bars
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

    ax1.set_ylabel("Pretraining Condition")
    ax1.set_xlabel(f"Proportion of {metric}")
    ax1.set_title(f"F1 {metric} - Proportions by Pretraining with 95% CI")
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

    ax2.set_ylabel("Pretraining Condition")
    ax2.set_xlabel("Count")
    ax2.set_title(f"F1 {metric} - Raw Counts by Pretraining")
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(prop_data[y])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"f1_{metric}_binary_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  F1 binary plot saved: {output_file}")


def main(overwrite=True, test_mode=False, specific_metrics=None, ball_identity="test"):
    """
    Main function to run Mann-Whitney plots for F1 experiments.

    Parameters:
    -----------
    overwrite : bool
        If True, overwrite existing plots. If False, skip metrics with existing plots.
    test_mode : bool
        If True, only process the first 5 metrics and sample data for fast testing.
    specific_metrics : list or None
        If provided, only process these specific metrics. If None, process all metrics.
    ball_identity : str
        Which ball identity to analyze ('test', 'training', or 'both')
    """
    # Record overall start time
    overall_start_time = time.time()

    if specific_metrics:
        print(f"Starting F1 Mann-Whitney U test analysis for specific metrics: {specific_metrics}")
    else:
        print(f"Starting F1 Mann-Whitney U test analysis for all metrics...")

    print(f"Ball identity focus: {ball_identity}")

    if not overwrite:
        print("📄 Overwrite disabled: Will skip metrics with existing plots")
    if test_mode:
        print("🧪 TEST MODE: Processing limited metrics and sample data for faster debugging")

    # Define output directories
    base_output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/Summary_metrics/F1_Metrics_byPretraining_Mannwhitney")

    # Ensure the base output directory exists
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output base directory: {base_output_dir}")

    # Create subdirectories for different ball identities
    ball_output_dir = base_output_dir / f"ball_{ball_identity}"
    ball_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ball-specific output directory: {ball_output_dir}")

    # Load the F1 dataset
    print("Loading F1 dataset...")
    dataset = load_f1_dataset(test_mode=test_mode, test_sample_size=500)

    # Subset by ball identity
    dataset = subset_dataset_by_ball_identity(dataset, ball_identity)

    # Load F1-specific metrics
    all_metrics = load_f1_metrics_list()

    # Filter to only metrics present in dataset
    available_metrics = [col for col in all_metrics if col in dataset.columns]
    missing_metrics = [col for col in all_metrics if col not in dataset.columns]

    print(f"📊 Found {len(available_metrics)}/{len(all_metrics)} metrics in F1 dataset")
    if missing_metrics:
        print(f"⚠️  Missing metrics: {missing_metrics[:10]}{'...' if len(missing_metrics) > 10 else ''}")

    metric_cols = available_metrics

    # Early exclusion of problematic patterns
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

    for metric in metric_cols:
        # Skip if it matches any excluded pattern
        excluded_pattern = None
        for pattern in excluded_patterns:
            if pattern in metric:
                excluded_pattern = pattern
                break

        if excluded_pattern:
            excluded_reasons.append(f"{metric} (pattern: {excluded_pattern})")
            continue

        # Skip if it's non-numeric
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

    # Apply specific metrics filtering if provided
    if specific_metrics:
        print(f"\n📊 SPECIFIC METRICS FILTERING: Requested {len(specific_metrics)} specific metrics")

        available_specific = []
        missing_specific = []

        for metric in specific_metrics:
            if metric in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[metric]):
                    # Check for excluded patterns
                    excluded_pattern = None
                    for pattern in excluded_patterns:
                        if pattern in metric:
                            excluded_pattern = pattern
                            break

                    if excluded_pattern:
                        print(f"  ⚠️  Skipping {metric}: matches excluded pattern '{excluded_pattern}'")
                        missing_specific.append(f"{metric} (excluded pattern: {excluded_pattern})")
                    else:
                        available_specific.append(metric)
                else:
                    print(f"  ⚠️  Skipping {metric}: not numeric (dtype: {dataset[metric].dtype})")
                    missing_specific.append(f"{metric} (non-numeric)")
            else:
                print(f"  ⚠️  Skipping {metric}: not found in dataset")
                missing_specific.append(f"{metric} (not in dataset)")

        if available_specific:
            metric_cols = available_specific
            print(f"✅ Found {len(available_specific)} requested metrics in dataset: {available_specific}")
        else:
            print(f"❌ None of the requested metrics are valid!")

        if missing_specific:
            print(f"⚠️  Issues with requested metrics: {missing_specific}")

        if not available_specific:
            print(f"⚠️  No valid metrics to process. Exiting...")
            return

    # Test mode limitation
    if test_mode and not specific_metrics:
        original_count = len(metric_cols)
        metric_cols = metric_cols[:5]
        print(f"🧪 TEST MODE: Limited to {len(metric_cols)}/{original_count} metrics for faster processing")

    # Define binary metrics
    all_possible_binary_metrics = ["has_finished", "has_major", "has_significant"]
    predefined_binary_metrics = [metric for metric in all_possible_binary_metrics if metric in metric_cols]

    print(f"Checking for binary metrics: {all_possible_binary_metrics}")
    print(f"Found binary metrics in dataset: {predefined_binary_metrics}")

    # Categorize metrics
    print(f"🔍 Categorizing {len(metric_cols)} F1 metrics for analysis...")

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

        if len(non_nan_values) < 3:
            return "insufficient_data"

        unique_vals = non_nan_values.unique()

        # Check for binary metrics (only 0s and 1s)
        if len(unique_vals) == 2:
            vals_set = set(unique_vals)
            if vals_set == {0, 1} or vals_set == {0.0, 1.0} or vals_set == {0, 1.0} or vals_set == {0.0, 1}:
                return "binary"

        # Check for columns with very few unique values
        if len(unique_vals) < 3:
            return "categorical"

        return "continuous"

    continuous_metrics = []
    binary_metrics = []
    excluded_metrics = []

    for col in metric_cols:
        category = categorize_metrics(col)

        if category == "continuous":
            continuous_metrics.append(col)
        elif category == "binary":
            binary_metrics.append(col)
        elif category == "insufficient_data":
            # Include anyway but with warning
            non_nan_count = dataset[col].count()
            total_count = len(dataset)
            print(f"  ⚠️  {col}: Only {non_nan_count}/{total_count} non-NaN values")

            if col in predefined_binary_metrics:
                binary_metrics.append(col)
            else:
                continuous_metrics.append(col)
        else:
            excluded_metrics.append((col, category))

    print(f"\nFound {len(continuous_metrics)} continuous metrics for Mann-Whitney analysis:")
    for i, metric in enumerate(continuous_metrics, 1):
        print(f"  {i:2d}. {metric}")

    print(f"\nFound {len(binary_metrics)} binary metrics for proportion analysis:")
    for i, metric in enumerate(binary_metrics, 1):
        print(f"  {i:2d}. {metric}")

    if excluded_metrics:
        print(f"\nExcluded {len(excluded_metrics)} metrics:")
        for metric, reason in excluded_metrics:
            print(f"  - {metric} ({reason})")

    # Validate required columns
    required_cols = ["Pretraining"]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present")

    # Check pretraining distribution
    print(f"\n🔍 Pretraining condition distribution:")
    pretraining_counts = dataset["Pretraining"].value_counts()
    print(pretraining_counts)

    # Process continuous metrics with Mann-Whitney U tests
    if continuous_metrics:
        print(f"\n{'='*60}")
        print(f"F1 CONTINUOUS METRICS ANALYSIS (Mann-Whitney U tests)")
        print(f"Ball identity: {ball_identity}")
        print("=" * 60)

        # Filter to only continuous metrics + required columns
        continuous_data = dataset[["Pretraining"] + continuous_metrics].copy()

        print(f"Generating F1 Mann-Whitney plots...")
        print(f"Continuous metrics dataset shape: {continuous_data.shape}")
        print(f"Output directory: {ball_output_dir}")

        # Filter out metrics that should be skipped
        if not overwrite:
            print(f"📄 Checking for existing plots...")
            metrics_to_process = []
            for metric in continuous_metrics:
                if not should_skip_metric(metric, ball_output_dir, overwrite):
                    metrics_to_process.append(metric)

            if len(metrics_to_process) < len(continuous_metrics):
                skipped_count = len(continuous_metrics) - len(metrics_to_process)
                print(f"📄 Skipped {skipped_count} metrics with existing plots")

            if not metrics_to_process:
                print(f"📄 All continuous metrics already have plots, skipping...")
            else:
                continuous_data = continuous_data[["Pretraining"] + metrics_to_process]
                continuous_metrics = metrics_to_process
        else:
            metrics_to_process = continuous_metrics

        if metrics_to_process:
            condition_start_time = time.time()
            print(f"⏱️  Processing {len(metrics_to_process)} continuous metrics for F1 analysis...")

            # Check for NaN annotations
            nan_annotations = get_nan_annotations(continuous_data, metrics_to_process, "Pretraining")
            if nan_annotations:
                print(f"📋 NaN values detected in {len(nan_annotations)} metrics")

            try:
                # Use simple F1-specific plotting instead of complex split-based function
                print(f"📊 Creating simple F1 pretraining comparison plots...")
                create_f1_pretraining_plots(
                    continuous_data,
                    metrics=metrics_to_process,
                    output_dir=ball_output_dir,
                    y_col="Pretraining",
                )
                condition_time = time.time() - condition_start_time
                print(f"⏱️  F1 continuous metrics completed in {condition_time:.1f}s")
            except Exception as e:
                print(f"❌ Error processing F1 continuous metrics: {e}")
                if test_mode:
                    print("🧪 TEST MODE: Continuing despite error...")

    # Process binary metrics with Fisher's exact tests
    if binary_metrics:
        binary_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"F1 BINARY METRICS ANALYSIS (Fisher's exact tests)")
        print(f"Ball identity: {ball_identity}")
        print("=" * 60)

        # Filter to only binary metrics + required columns
        binary_data = dataset[["Pretraining"] + binary_metrics].copy()

        print(f"⏱️  Analyzing {len(binary_metrics)} binary metrics for F1...")
        print(f"Binary metrics dataset shape: {binary_data.shape}")

        # Check for NaN annotations in binary metrics
        nan_annotations = get_nan_annotations(binary_data, binary_metrics, "Pretraining")
        if nan_annotations:
            print(f"📋 NaN values detected in {len(nan_annotations)} binary metrics")

        binary_output_dir = ball_output_dir / "binary_analysis"
        binary_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            binary_results = analyze_binary_metrics_f1(
                binary_data,
                binary_metrics,
                y="Pretraining",
                output_dir=binary_output_dir,
                overwrite=overwrite,
            )
            binary_time = time.time() - binary_start_time
            print(f"⏱️  F1 binary metrics completed in {binary_time:.1f}s")
        except Exception as e:
            print(f"❌ Error processing F1 binary metrics: {e}")
            if test_mode:
                print("🧪 TEST MODE: Continuing despite error...")

    print(f"Unique pretraining conditions: {len(dataset['Pretraining'].unique())}")

    # Overall timing summary
    total_time = time.time() - overall_start_time
    print(f"\n{'='*60}")
    print(f"✅ F1 Analysis complete! Total runtime: {total_time:.1f}s")
    print(f"Ball identity analyzed: {ball_identity}")
    print(f"Check output directory for results: {ball_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Mann-Whitney U test jitterboxplots for F1 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mannwhitney_f1_metrics.py                                  # Process all metrics for test ball, overwrite existing
  python run_mannwhitney_f1_metrics.py --ball-identity training         # Analyze training ball instead
  python run_mannwhitney_f1_metrics.py --ball-identity both             # Analyze both balls together
  python run_mannwhitney_f1_metrics.py --no-overwrite                  # Skip existing plots
  python run_mannwhitney_f1_metrics.py --test                          # Run in test mode (faster debugging)
  python run_mannwhitney_f1_metrics.py --metrics nb_events max_event   # Process only specific metrics
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        help="Process only specific metrics (space-separated list). Example: --metrics nb_events max_event has_finished",
    )
    parser.add_argument(
        "--ball-identity",
        type=str,
        choices=["test", "training", "both"],
        default="test",
        help="Which ball identity to analyze (default: test)",
    )

    args = parser.parse_args()

    # Run main function with parameters
    main(
        overwrite=not args.no_overwrite,
        test_mode=args.test,
        specific_metrics=args.metrics,
        ball_identity=args.ball_identity,
    )
