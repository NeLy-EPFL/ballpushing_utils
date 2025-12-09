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

# NOTE: All_metrics import commented out - not needed for F1 plotting
# try:
#     from All_metrics import generate_jitterboxplots_with_mannwhitney
#     print("✅ Successfully imported All_metrics module")
# except Exception as e:
#     print(f"❌ Error importing All_metrics: {e}")
#     print("This error occurs during All_metrics module initialization")
#     print("Check the All_metrics.py file for issues in the module-level code")
#     sys.exit(1)

from Config import color_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats_module
import matplotlib

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


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
        return data / PIXELS_PER_MM
    elif is_time_metric(metric_name):
        return data / 60  # Convert seconds to minutes
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "(mm)"
    elif is_time_metric(metric_name):
        return "(min)"
    return ""


def get_elegant_metric_name(metric_name):
    """Convert metric code names to elegant display names"""
    metric_map = {
        # Event metrics
        "nb_events": "Number of events",
        "has_major": "Has major event",
        "first_major_event": "First major event",
        "first_major_event_time": "First major event time",
        "max_event": "Max event",
        "max_event_time": "Max event time",
        "final_event": "Final event",
        "final_event_time": "Final event time",
        "has_significant": "Has significant event",
        "nb_significant_events": "Number of significant events",
        "significant_ratio": "Significant event ratio",
        "first_significant_event": "First significant event",
        "first_significant_event_time": "First significant event time",
        # Distance metrics
        "max_distance": "Max ball distance",
        "distance_moved": "Total ball distance moved",
        "distance_ratio": "Distance ratio",
        "fly_distance_moved": "Fly distance moved",
        # Interaction metrics
        "overall_interaction_rate": "Interaction rate",
        "interaction_persistence": "Interaction persistence",
        "interaction_proportion": "Interaction proportion",
        "time_to_first_interaction": "Time to first interaction",
        # Push/pull metrics
        "pushed": "Number of pushes",
        "pulled": "Number of pulls",
        "pulling_ratio": "Pulling ratio",
        "head_pushing_ratio": "Head pushing ratio",
        # Velocity metrics
        "normalized_velocity": "Normalized velocity",
        "velocity_during_interactions": "Velocity during interactions",
        "velocity_trend": "Velocity trend",
        # Pause/stop metrics
        "has_long_pauses": "Has long pauses",
        "nb_stops": "Number of stops",
        "total_stop_duration": "Total stop duration",
        "median_stop_duration": "Median stop duration",
        "nb_pauses": "Number of pauses",
        "total_pause_duration": "Total pause duration",
        "median_pause_duration": "Median pause duration",
        "nb_long_pauses": "Number of long pauses",
        "total_long_pause_duration": "Total long pause duration",
        "median_long_pause_duration": "Median long pause duration",
        "pauses_persistence": "Pauses persistence",
        "cumulated_breaks_duration": "Total break duration",
        # Freeze metrics
        "nb_freeze": "Number of freezes",
        "median_freeze_duration": "Median freeze duration",
        # Chamber metrics
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "chamber_exit_time": "Chamber exit time",
        "time_chamber_beginning": "Time in chamber (early)",
        # Task completion
        "has_finished": "Task completed",
        "persistence_at_end": "Persistence at end",
        # Other behavioral metrics
        "fraction_not_facing_ball": "Fraction not facing ball",
        "leg_visibility_ratio": "Leg visibility ratio",
        "flailing": "Leg flailing",
        "median_head_ball_distance": "Median head-ball distance",
        "mean_head_ball_distance": "Mean head-ball distance",
    }
    return metric_map.get(metric_name, metric_name)


def format_metric_label(metric_name):
    """Format metric name with elegant display name and appropriate unit"""
    elegant_name = get_elegant_metric_name(metric_name)
    unit = get_metric_unit(metric_name)
    if unit:
        return f"{elegant_name} {unit}"
    return elegant_name


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
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251119_10_summary_F1_New_Data/summary/pooled_summary.feather"
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
        "Pretrained": "#1f77b4",  # Blue
        "Ctrl": "#ff7f0e",  # Orange
    }

    return pretraining_colors


def map_condition_labels(condition):
    """
    Map short condition codes to informative display labels

    Parameters:
    -----------
    condition : str
        Original condition label

    Returns:
    --------
    str
        Mapped informative label
    """
    label_map = {
        "y": "Pretrained",
        "n": "Ctrl",
        "yes": "Pretrained",
        "no": "Ctrl",
    }
    return label_map.get(condition, condition)


def map_metric_name(metric):
    """
    Map metric codes to informative display names based on README documentation

    Parameters:
    -----------
    metric : str
        Original metric name

    Returns:
    --------
    str
        Mapped informative display name
    """
    metric_map = {
        # Task Performance Metrics
        "has_finished": "Task completed",
        "max_distance": "Max ball distance",
        "has_major": "Has major event",
        "first_major_event": "First major event",
        "first_major_event_time": "First major event time",
        "max_event": "Max event",
        "max_event_time": "Max event time",
        # Interaction Behavior Metrics
        "nb_events": "Number of events",
        "has_significant": "Has significant event",
        "nb_significant_events": "Number of significant events",
        "significant_ratio": "Significant event ratio",
        "overall_interaction_rate": "Interaction rate",
        "distance_moved": "Total ball distance moved",
        "distance_ratio": "Distance ratio",
        "pushed": "Number of pushes",
        "pulled": "Number of pulls",
        "pulling_ratio": "Pulling ratio",
        "flailing": "Leg flailing",
        "head_pushing_ratio": "Head pushing ratio",
        # Locomotor Activity Metrics
        "normalized_velocity": "Normalized velocity",
        "velocity_during_interactions": "Velocity during interactions",
        "velocity_trend": "Velocity trend",
        "has_long_pauses": "Has long pauses",
        "nb_stops": "Number of stops",
        "total_stop_duration": "Total stop duration",
        "median_stop_duration": "Median stop duration",
        "nb_pauses": "Number of pauses",
        "total_pause_duration": "Total pause duration",
        "median_pause_duration": "Median pause duration",
        "nb_long_pauses": "Number of long pauses",
        "total_long_pause_duration": "Total long pause duration",
        "median_long_pause_duration": "Median long pause duration",
        # Spatial Behavior Metrics
        "time_chamber_beginning": "Time in chamber (early)",
        "persistence_at_end": "Persistence at end",
        # Additional metrics
        "final_event": "Final event",
        "final_event_time": "Final event time",
        "first_significant_event": "First significant event",
        "first_significant_event_time": "First significant event time",
        "interaction_persistence": "Interaction persistence",
        "interaction_proportion": "Interaction proportion",
        "cumulated_breaks_duration": "Total break duration",
        "fly_distance_moved": "Fly distance moved",
        "fraction_not_facing_ball": "Fraction not facing ball",
        "leg_visibility_ratio": "Leg visibility ratio",
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "chamber_exit_time": "Chamber exit time",
    }
    return metric_map.get(metric, metric)


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
        "time_to_first_interaction",
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

    For 2 groups: Uses Fisher's exact test
    For 3+ groups: Uses overall chi-square test followed by pairwise Fisher's tests with FDR correction

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the binary metrics
    binary_metrics : list
        List of binary metric column names
    y : str
        Column name for grouping (pretraining conditions or F1_condition)
    output_dir : str or Path
        Directory to save plots
    overwrite : bool
        If True, overwrite existing plots. If False, skip existing plots.

    Returns:
    --------
    tuple
        (pairwise_results_df, overall_results_df) - DataFrames with pairwise and overall statistics
    """
    if not binary_metrics:
        print("No binary metrics to analyze")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\n{'='*50}")
    print(f"F1 BINARY METRICS ANALYSIS - Grouping by {y}")
    print(f"{'='*50}")

    results = []
    overall_results = []  # Store overall test results

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine control group (typically 'n' for no pretraining or 'control')
    control_condition = None
    condition_values = data[y].value_counts()
    n_conditions = len(condition_values)
    print(f"{y} distribution: {dict(condition_values)}")
    print(f"Number of groups: {n_conditions}")

    # Prioritize common control conditions
    control_candidates = ["n", "no", "control", "untrained", "naive"]
    for candidate in control_candidates:
        if candidate in condition_values.index:
            control_condition = candidate
            break

    if control_condition is None:
        # Use the first condition alphabetically as control
        control_condition = sorted(condition_values.index)[0]

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

        # For 3+ groups, perform overall chi-square test first
        overall_p = None
        overall_chi2 = None
        if n_conditions > 2:
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
                        "n_groups": n_conditions,
                        "chi2": chi2,
                        "p_value": overall_p,
                        "dof": dof,
                        "min_expected": min_expected,
                        "significant": overall_p < 0.05,
                    }
                )

            except Exception as e:
                print(f"  Error in overall chi-square test: {e}")

        # Pairwise comparisons: each condition vs control
        pairwise_results_for_metric = []

        for condition in data[y].unique():
            if condition == control_condition:
                continue

            # Get data for this comparison
            test_data = data[data[y] == condition][metric].dropna()
            control_data_subset = data[data[y] == control_condition][metric].dropna()

            if len(test_data) == 0 or len(control_data_subset) == 0:
                continue

            # Create 2x2 contingency table for Fisher's exact test
            test_success = int(test_data.sum())
            test_total = len(test_data)
            control_success = int(control_data_subset.sum())
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

                pairwise_results_for_metric.append(
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
                        "p_value_uncorrected": p_value,  # Store uncorrected p-value
                        "test_type": "Fisher exact",
                        "overall_chi2": overall_chi2,
                        "overall_p": overall_p,
                    }
                )

                print(f"  {condition} vs {control_condition}: OR={odds_ratio:.3f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error testing {condition} vs {control_condition}: {e}")

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
                result["significant"] = reject[i]  # Use FDR-corrected significance
                result["sig_level"] = sig_level  # Use FDR-corrected significance level

                print(
                    f"    {result['condition']}: p_uncorr={result['p_value_uncorrected']:.4f} → p_FDR={p_corrected[i]:.4f} {sig_level}"
                )
        else:
            # Single comparison, no correction needed
            for result in pairwise_results_for_metric:
                result["p_value_fdr"] = result["p_value"]
                result["significant_fdr"] = result["p_value"] < 0.05

                # Significance level based on uncorrected p-value
                if result["p_value"] < 0.001:
                    sig_level = "***"
                elif result["p_value"] < 0.01:
                    sig_level = "**"
                elif result["p_value"] < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                result["sig_level_fdr"] = sig_level
                result["sig_level"] = sig_level
                result["significant"] = result["significant_fdr"]

        # Add all pairwise results to main results list
        results.extend(pairwise_results_for_metric)

        # Create visualization
        if output_dir:
            create_binary_metric_plot_f1(data, metric, y, output_dir, control_condition, n_conditions)

    results_df = pd.DataFrame(results)
    overall_df = pd.DataFrame(overall_results)

    if output_dir:
        if not results_df.empty:
            stats_file = output_dir / f"f1_binary_metrics_pairwise_{y}.csv"
            results_df.to_csv(stats_file, index=False)
            print(f"\nPairwise comparison statistics saved to: {stats_file}")

        if not overall_df.empty:
            overall_file = output_dir / f"f1_binary_metrics_overall_{y}.csv"
            overall_df.to_csv(overall_file, index=False)
            print(f"Overall test statistics saved to: {overall_file}")

    return results_df, overall_df


def permutation_test_means(group1, group2, n_permutations=10000, random_state=42):
    """
    Perform permutation test on difference of means.

    Parameters:
    -----------
    group1 : array-like
        First group of observations
    group2 : array-like
        Second group of observations
    n_permutations : int
        Number of permutations to perform (default: 10000)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (observed_difference, p_value)
        - observed_difference: actual difference in means (group2 - group1)
        - p_value: two-sided p-value
    """
    np.random.seed(random_state)

    # Convert to numpy arrays
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Calculate observed difference in means
    observed_diff = np.mean(group2) - np.mean(group1)

    # Combine all data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Perform permutations
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined)
        # Split into two groups
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        # Calculate difference in means
        perm_diffs[i] = np.mean(perm_group2) - np.mean(perm_group1)

    # Calculate two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value


def create_f1_pretraining_plots(data, metrics, output_dir, y_col="Pretraining", test_type="mannwhitney"):
    """
    Create simple statistical comparison plots for F1 experiments.

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
    test_type : str
        Type of statistical test: 'mannwhitney' or 'permutation' (default: 'mannwhitney')

    Returns:
    --------
    list
        List of dictionaries containing statistical results for each metric
    """
    from scipy.stats import mannwhitneyu
    import matplotlib.pyplot as plt
    import matplotlib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Arial font globally for editable text in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for editable text in PDF
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Store statistical results for summary
    stats_results = []

    # Get pretraining colors
    pretraining_colors = create_pretraining_colors()

    # Determine control group (typically 'n' for no pretraining)
    pretraining_values = data[y_col].unique()
    control_condition = "n" if "n" in pretraining_values else pretraining_values[0]
    experimental_condition = [c for c in pretraining_values if c != control_condition][0]

    print(f"Control condition: {control_condition}")
    print(f"Experimental condition: {experimental_condition}")

    # Map conditions to display labels
    control_label = map_condition_labels(control_condition)
    experimental_label = map_condition_labels(experimental_condition)

    # Process each metric
    for metric in metrics:
        print(f"  Creating F1 plot for: {metric}")

        # Get data for this metric
        metric_data = data.dropna(subset=[metric, y_col])

        if len(metric_data) == 0:
            print(f"    ⚠️  No data available for {metric}")
            continue

        # Get raw data and convert if needed
        control_data_raw = metric_data[metric_data[y_col] == control_condition][metric]
        experimental_data_raw = metric_data[metric_data[y_col] == experimental_condition][metric]

        if len(control_data_raw) == 0 or len(experimental_data_raw) == 0:
            print(f"    ⚠️  Insufficient data for comparison in {metric}")
            continue

        # Store sample sizes
        n_control = len(control_data_raw)
        n_experimental = len(experimental_data_raw)

        # Perform statistical test on RAW data (before conversion)
        try:
            if test_type == "permutation":
                # Permutation test on RAW means
                mean_diff, p_value = permutation_test_means(control_data_raw, experimental_data_raw)
                statistic = mean_diff  # Store mean difference as statistic
                test_name = "Permutation test (means)"
            else:
                # Mann-Whitney U test on RAW data (default)
                statistic, p_value = mannwhitneyu(experimental_data_raw, control_data_raw, alternative="two-sided")
                test_name = "Mann-Whitney U"

            # Convert data for plotting (pixels to mm, seconds to minutes if needed)
            control_data = convert_metric_data(control_data_raw, metric)
            experimental_data = convert_metric_data(experimental_data_raw, metric)

            # Determine significance level
            if p_value < 0.001:
                sig_level = "***"
            elif p_value < 0.01:
                sig_level = "**"
            elif p_value < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"

            # Calculate effect sizes on CONVERTED data for display
            control_median = control_data.median()
            experimental_median = experimental_data.median()
            median_diff = experimental_median - control_median

            control_mean = control_data.mean()
            experimental_mean = experimental_data.mean()
            mean_diff_calc = experimental_mean - control_mean

            if control_median != 0:
                percent_change_median = (median_diff / control_median) * 100
            else:
                percent_change_median = np.nan

            if control_mean != 0:
                percent_change_mean = (mean_diff_calc / control_mean) * 100
            else:
                percent_change_mean = np.nan

            # Store results for summary (use sample sizes from raw data)
            result_dict = {
                "metric": metric,
                "metric_display_name": get_elegant_metric_name(metric),
                "control_condition": control_condition,
                "experimental_condition": experimental_condition,
                "control_n": n_control,
                "experimental_n": n_experimental,
                "control_median": control_median,
                "control_mean": control_mean,
                "control_std": control_data.std(),
                "experimental_median": experimental_median,
                "experimental_mean": experimental_mean,
                "experimental_std": experimental_data.std(),
                "median_difference": median_diff,
                "mean_difference": mean_diff_calc,
                "percent_change_median": percent_change_median,
                "percent_change_mean": percent_change_mean,
                "test_statistic": statistic,
                "p_value": p_value,
                "significance": sig_level,
                "significant": p_value < 0.05,
                "test_type": test_name,
            }

            # Add test-specific fields for backward compatibility
            if test_type == "permutation":
                result_dict["permutation_mean_diff"] = statistic
            else:
                result_dict["mann_whitney_u"] = statistic
                result_dict["percent_change"] = percent_change_median  # Legacy field

            stats_results.append(result_dict)

        except Exception as e:
            print(f"    ⚠️  Error in statistical test for {metric}: {e}")
            p_value = 1.0
            sig_level = "ns"
            statistic = np.nan
            # Still convert data for plotting even if test fails
            control_data = convert_metric_data(control_data_raw, metric)
            experimental_data = convert_metric_data(experimental_data_raw, metric)

        # Create the plot with smaller figure size for publication-quality layout
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 3.5))

        # Define colors consistently - ALWAYS use the same color for the same condition
        # This ensures 'n' is always orange and 'y' is always blue across all plots
        control_color = pretraining_colors.get(control_condition, "#808080")  # Gray fallback
        experimental_color = pretraining_colors.get(experimental_condition, "#808080")  # Gray fallback

        # Create thinner boxplot with transparency
        box_props = dict(linewidth=1.5, edgecolor="black")  # Black outline
        whisker_props = dict(linewidth=1.5, color="black")
        cap_props = dict(linewidth=1.5, color="black")
        median_props = dict(linewidth=2, color="black")

        bp = ax.boxplot(
            [control_data, experimental_data],
            positions=[0, 1],
            widths=0.5,  # Thinner boxes (50% of spacing)
            patch_artist=True,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            medianprops=median_props,
            showfliers=False,  # Don't show outliers since we're plotting all points
        )

        # Color the boxes consistently with black outlines - position 0 is control, position 1 is experimental
        bp["boxes"][0].set_facecolor(control_color)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][0].set_linewidth(1.5)
        bp["boxes"][1].set_facecolor(experimental_color)
        bp["boxes"][1].set_alpha(0.7)
        bp["boxes"][1].set_edgecolor("black")
        bp["boxes"][1].set_linewidth(1.5)

        # Add individual data points as larger filled circles matching box colors (using CONVERTED data)
        x_jitter = 0.08  # Amount of horizontal jitter
        for i, (condition_data_to_plot, color) in enumerate(
            [(control_data, control_color), (experimental_data, experimental_color)]
        ):
            # Add random jitter to x positions
            x_positions = np.random.normal(i, x_jitter, size=len(condition_data_to_plot))
            ax.scatter(
                x_positions,
                condition_data_to_plot,
                s=25,  # Smaller point size for cleaner look
                c=color,
                alpha=0.6,  # Translucent
                edgecolors="none",  # No edge
                zorder=3,
            )  # Draw on top

        # Add significance annotation (using CONVERTED data for y-axis limits)
        y_max = max(control_data.max(), experimental_data.max())
        y_min = min(control_data.min(), experimental_data.min())
        y_range = y_max - y_min

        # Draw significance bar and asterisks
        if sig_level != "ns":
            bar_height = y_max + 0.08 * y_range
            ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
            ax.text(
                0.5, bar_height + 0.02 * y_range, sig_level, ha="center", va="bottom", fontsize=12, fontname="Arial"
            )

        # Always add p-value text in top-left corner (for both significant and non-significant)
        if p_value < 0.001:
            p_text = f"p < 0.001"
        else:
            p_text = f"p = {p_value:.3f}"

        ax.text(
            0.02,
            0.98,
            p_text,
            transform=ax.transAxes,
            fontsize=10,
            fontname="Arial",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
        )

        # Clean formatting - no grid, white background, with tick marks
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(False)

        # Add tick marks pointing outward
        ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)

        # Set x-axis labels with sample sizes
        ax.set_xticks([0, 1])
        control_label_with_n = f"{control_label}\n(n={n_control})"
        experimental_label_with_n = f"{experimental_label}\n(n={n_experimental})"
        ax.set_xticklabels([control_label_with_n, experimental_label_with_n], fontsize=10, fontname="Arial")

        # Set y-axis label with elegant metric name and units
        ylabel = format_metric_label(metric)
        ax.set_ylabel(ylabel, fontsize=11, fontname="Arial")

        # Format y-axis tick labels
        ax.tick_params(axis="y", labelsize=9)
        for label in ax.get_yticklabels():
            label.set_fontname("Arial")

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        plt.tight_layout()

        # Save plot with editable text - filename reflects test type
        test_suffix = "permutation" if test_type == "permutation" else "mannwhitney"
        output_file = output_dir / f"f1_{metric}_{test_suffix}_comparison.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", format="pdf")
        plt.close()

        print(f"    ✅ Saved: {output_file.name}")

    return stats_results


def generate_markdown_summary(
    stats_results, binary_results, overall_binary_results, output_dir, ball_identity="test", test_type="mannwhitney"
):
    """
    Generate a comprehensive Markdown summary file with all statistical test results.

    Parameters:
    -----------
    stats_results : list
        List of dictionaries with continuous metric statistics
    binary_results : pd.DataFrame or None
        DataFrame with binary metric statistics
    output_dir : Path
        Directory to save the summary file
    ball_identity : str
        Ball identity analyzed ('test', 'training', or 'both')
    test_type : str
        Type of statistical test used ('mannwhitney' or 'permutation')
    """
    output_dir = Path(output_dir)
    test_label = "Permutation" if test_type == "permutation" else "Mann_Whitney"
    output_file = output_dir / f"F1_Statistical_Summary_{test_label}_ball_{ball_identity}.md"

    with open(output_file, "w") as f:
        # Header
        test_name = "Permutation Test (Means)" if test_type == "permutation" else "Mann-Whitney U Test"
        f.write(f"# F1 Pretraining Statistical Analysis Summary\n\n")
        f.write(f"**Statistical Test:** {test_name}\n\n")
        f.write(f"**Ball Identity:** {ball_identity}\n\n")
        f.write(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Continuous Metrics Section
        if stats_results:
            f.write(f"## Continuous Metrics ({test_name})\n\n")

            if test_type == "permutation":
                f.write(
                    "Statistical comparison of pretraining conditions using permutation test on mean differences.\n\n"
                )
            else:
                f.write("Statistical comparison of pretraining conditions using two-sided Mann-Whitney U test.\n\n")

            # Create control and experimental labels from first result
            if stats_results:
                ctrl = map_condition_labels(stats_results[0]["control_condition"])
                exp = map_condition_labels(stats_results[0]["experimental_condition"])
                f.write(f"**Conditions:** {ctrl} (control) vs {exp} (experimental)\n\n")

            # Significance summary
            sig_count = sum(1 for r in stats_results if r["significant"])
            total_count = len(stats_results)
            f.write(f"**Summary:** {sig_count}/{total_count} metrics showed significant differences (p < 0.05)\n\n")

            # Separate by significance
            significant_metrics = [r for r in stats_results if r["significant"]]
            non_significant_metrics = [r for r in stats_results if not r["significant"]]

            # Significant results
            if significant_metrics:
                f.write(f"### Significant Results (n={len(significant_metrics)})\n\n")

                if test_type == "permutation":
                    f.write(
                        "| Metric | Control<br>Mean (n) | Experimental<br>Mean (n) | Mean Diff | % Change | p-value | Sig |\n"
                    )
                    f.write("|--------|-----------|--------------|-----------|----------|---------|-----|\n")

                    # Sort by p-value (ascending)
                    for result in sorted(significant_metrics, key=lambda x: x["p_value"]):
                        metric_name = result["metric_display_name"]
                        ctrl_mean = f"{result['control_mean']:.3f} (n={result['control_n']})"
                        exp_mean = f"{result['experimental_mean']:.3f} (n={result['experimental_n']})"
                        diff = f"{result['mean_difference']:+.3f}"
                        pct = (
                            f"{result['percent_change_mean']:+.1f}%"
                            if not np.isnan(result["percent_change_mean"])
                            else "N/A"
                        )
                        pval = f"{result['p_value']:.4f}" if result["p_value"] >= 0.0001 else "< 0.0001"
                        sig = result["significance"]

                        f.write(f"| {metric_name} | {ctrl_mean} | {exp_mean} | {diff} | {pct} | {pval} | {sig} |\n")
                else:
                    f.write(
                        "| Metric | Control<br>Median (n) | Experimental<br>Median (n) | Difference | % Change | p-value | Sig |\n"
                    )
                    f.write("|--------|-----------|--------------|------------|----------|---------|-----|\n")

                    # Sort by p-value (ascending)
                    for result in sorted(significant_metrics, key=lambda x: x["p_value"]):
                        metric_name = result["metric_display_name"]
                        ctrl_med = f"{result['control_median']:.3f} (n={result['control_n']})"
                        exp_med = f"{result['experimental_median']:.3f} (n={result['experimental_n']})"
                        diff = f"{result['median_difference']:+.3f}"
                        pct = (
                            f"{result['percent_change_median']:+.1f}%"
                            if not np.isnan(result["percent_change_median"])
                            else "N/A"
                        )
                        pval = f"{result['p_value']:.4f}" if result["p_value"] >= 0.0001 else "< 0.0001"
                        sig = result["significance"]

                        f.write(f"| {metric_name} | {ctrl_med} | {exp_med} | {diff} | {pct} | {pval} | {sig} |\n")

                f.write("\n")

            # Non-significant results
            if non_significant_metrics:
                f.write(f"### Non-Significant Results (n={len(non_significant_metrics)})\n\n")

                if test_type == "permutation":
                    f.write("| Metric | Control<br>Mean (n) | Experimental<br>Mean (n) | Mean Diff | p-value |\n")
                    f.write("|--------|-----------|--------------|-----------|--------|\n")

                    for result in sorted(non_significant_metrics, key=lambda x: x["p_value"]):
                        metric_name = result["metric_display_name"]
                        ctrl_mean = f"{result['control_mean']:.3f} (n={result['control_n']})"
                        exp_mean = f"{result['experimental_mean']:.3f} (n={result['experimental_n']})"
                        diff = f"{result['mean_difference']:+.3f}"
                        pval = f"{result['p_value']:.4f}"

                        f.write(f"| {metric_name} | {ctrl_mean} | {exp_mean} | {diff} | {pval} |\n")
                else:
                    f.write("| Metric | Control<br>Median (n) | Experimental<br>Median (n) | Difference | p-value |\n")
                    f.write("|--------|-----------|--------------|------------|--------|\n")

                    for result in sorted(non_significant_metrics, key=lambda x: x["p_value"]):
                        metric_name = result["metric_display_name"]
                        ctrl_med = f"{result['control_median']:.3f} (n={result['control_n']})"
                        exp_med = f"{result['experimental_median']:.3f} (n={result['experimental_n']})"
                        diff = f"{result['median_difference']:+.3f}"
                        pval = f"{result['p_value']:.4f}"

                        f.write(f"| {metric_name} | {ctrl_med} | {exp_med} | {diff} | {pval} |\n")

                f.write("\n")

            # Detailed statistics table
            f.write("### Detailed Statistics\n\n")

            if test_type == "permutation":
                f.write("| Metric | Condition | n | Mean ± SD | Median | Mean Diff | p-value |\n")
                f.write("|--------|-----------|---|-----------|--------|-----------|----------|\n")

                for result in sorted(stats_results, key=lambda x: x["p_value"]):
                    metric_name = result["metric_display_name"]

                    # Control row
                    ctrl_stats = f"{result['control_mean']:.3f} ± {result['control_std']:.3f}"
                    f.write(
                        f"| {metric_name} | Control | {result['control_n']} | {ctrl_stats} | {result['control_median']:.3f} | {result['mean_difference']:+.3f} | {result['p_value']:.4f} |\n"
                    )

                    # Experimental row
                    exp_stats = f"{result['experimental_mean']:.3f} ± {result['experimental_std']:.3f}"
                    f.write(
                        f"| | Experimental | {result['experimental_n']} | {exp_stats} | {result['experimental_median']:.3f} | | |\n"
                    )
            else:
                f.write("| Metric | Condition | n | Mean ± SD | Median | U Statistic | p-value |\n")
                f.write("|--------|-----------|---|-----------|--------|-------------|----------|\n")

                for result in sorted(stats_results, key=lambda x: x["p_value"]):
                    metric_name = result["metric_display_name"]

                    # Control row
                    ctrl_stats = f"{result['control_mean']:.3f} ± {result['control_std']:.3f}"
                    u_stat = result.get("mann_whitney_u", result.get("test_statistic", np.nan))
                    f.write(
                        f"| {metric_name} | Control | {result['control_n']} | {ctrl_stats} | {result['control_median']:.3f} | {u_stat:.1f} | {result['p_value']:.4f} |\n"
                    )

                    # Experimental row
                    exp_stats = f"{result['experimental_mean']:.3f} ± {result['experimental_std']:.3f}"
                    f.write(
                        f"| | Experimental | {result['experimental_n']} | {exp_stats} | {result['experimental_median']:.3f} | | |\n"
                    )

            f.write("\n")

        # Binary Metrics Section
        if binary_results is not None and not binary_results.empty:
            f.write("## Binary Metrics\\n\\n")

            # Overall tests (if available)
            if overall_binary_results is not None and not overall_binary_results.empty:
                f.write("### Overall Tests\\n\\n")
                f.write("Chi-square tests for overall differences across all groups:\\n\\n")
                f.write("| Metric | Groups | χ² | df | p-value | Sig |\\n")
                f.write("|--------|--------|-----|-----|---------|-----|\\n")

                for _, row in overall_binary_results.iterrows():
                    metric_name = map_metric_name(row["metric"])
                    n_groups = int(row["n_groups"])
                    chi2_val = f"{row['chi2']:.3f}"
                    dof = int(row["dof"])
                    p_val = f"{row['p_value']:.4f}"
                    sig = (
                        "***"
                        if row["p_value"] < 0.001
                        else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
                    )

                    f.write(f"| {metric_name} | {n_groups} | {chi2_val} | {dof} | {p_val} | {sig} |\\n")

                f.write("\\n")

            # Pairwise comparisons
            f.write("### Pairwise Comparisons (Fisher's Exact Test with FDR Correction)\\n\\n")
            f.write("Statistical comparison of proportions using Fisher's exact test.\\n")

            # Check if FDR correction was applied
            has_fdr = "p_value_fdr" in binary_results.columns
            if has_fdr:
                n_comparisons = len(binary_results)
                f.write(f"FDR correction (Benjamini-Hochberg) applied for {n_comparisons} pairwise comparisons.\\n\\n")
            else:
                f.write("\\n")

            sig_binary = sum(binary_results["significant"])
            total_binary = len(binary_results)
            f.write(
                f"**Summary:** {sig_binary}/{total_binary} pairwise comparisons showed significant differences (FDR-corrected p < 0.05)\\n\\n"
            )

            if has_fdr:
                f.write(
                    "| Metric | Comparison | Control<br>Prop (n) | Test<br>Prop (n) | Odds Ratio | p-value | p-FDR | Sig |\\n"
                )
                f.write("|--------|------------|----------|----------|------------|---------|-------|-----|\\n")

                for _, row in binary_results.sort_values("p_value_fdr").iterrows():
                    metric_name = map_metric_name(row["metric"])
                    comparison = f"{map_condition_labels(row['condition'])} vs {map_condition_labels(row['control'])}"
                    ctrl_prop = (
                        f"{row['control_proportion']:.3f} ({int(row['control_success'])}/{int(row['control_total'])})"
                    )
                    test_prop = f"{row['test_proportion']:.3f} ({int(row['test_success'])}/{int(row['test_total'])})"
                    odds_ratio = f"{row['odds_ratio']:.3f}"
                    p_val = f"{row['p_value_uncorrected']:.4f}" if row["p_value_uncorrected"] >= 0.0001 else "< 0.0001"
                    p_fdr = f"{row['p_value_fdr']:.4f}" if row["p_value_fdr"] >= 0.0001 else "< 0.0001"
                    sig = row["sig_level"]

                    f.write(
                        f"| {metric_name} | {comparison} | {ctrl_prop} | {test_prop} | {odds_ratio} | {p_val} | {p_fdr} | {sig} |\\n"
                    )
            else:
                f.write(
                    "| Metric | Comparison | Control<br>Prop (n) | Test<br>Prop (n) | Odds Ratio | p-value | Sig |\\n"
                )
                f.write("|--------|------------|----------|----------|------------|---------|-----|\\n")

                for _, row in binary_results.sort_values("p_value").iterrows():
                    metric_name = map_metric_name(row["metric"])
                    comparison = f"{map_condition_labels(row['condition'])} vs {map_condition_labels(row['control'])}"
                    ctrl_prop = (
                        f"{row['control_proportion']:.3f} ({int(row['control_success'])}/{int(row['control_total'])})"
                    )
                    test_prop = f"{row['test_proportion']:.3f} ({int(row['test_success'])}/{int(row['test_total'])})"
                    odds_ratio = f"{row['odds_ratio']:.3f}"
                    p_val = f"{row['p_value']:.4f}" if row["p_value"] >= 0.0001 else "< 0.0001"
                    sig = row["sig_level"]

                    f.write(
                        f"| {metric_name} | {comparison} | {ctrl_prop} | {test_prop} | {odds_ratio} | {p_val} | {sig} |\\n"
                    )

            f.write("\\n")

        # Footnotes
        f.write("---\n\n")
        f.write("## Notes\n\n")
        f.write("**Significance levels:**\n")
        f.write("- `***` p < 0.001 (highly significant)\n")
        f.write("- `**` p < 0.01 (very significant)\n")
        f.write("- `*` p < 0.05 (significant)\n")
        f.write("- `ns` p ≥ 0.05 (not significant)\n\n")
        f.write("**Statistical tests:**\n")

        if test_type == "permutation":
            f.write("- Continuous metrics: Permutation test on mean differences (10,000 permutations)\\n")
            f.write("  - Null hypothesis: No difference in means between groups\\n")
            f.write("  - Test statistic: Difference in means (Experimental - Control)\\n")
        else:
            f.write("- Continuous metrics: Two-sided Mann-Whitney U test (non-parametric)\\n")
            f.write("  - Null hypothesis: Distributions are the same\\n")
            f.write("  - Test statistic: U statistic\\n")

        f.write("- Binary metrics:\\n")
        f.write("  - Overall: Chi-square test (for 3+ groups)\\n")
        f.write("  - Pairwise: Fisher's exact test (for 2×2 contingency tables)\\n")
        f.write("  - Multiple comparison correction: FDR (Benjamini-Hochberg)\\n\\n")
        f.write("**Effect size:**\n")

        if test_type == "permutation":
            f.write("- Mean Diff: Experimental mean - Control mean\n")
            f.write("- % Change: ((Experimental mean - Control mean) / Control mean) × 100\n")
        else:
            f.write("- Difference: Experimental median - Control median\n")
            f.write("- % Change: ((Experimental - Control) / Control) × 100\n")

        f.write("- Odds Ratio: Odds of event in experimental / Odds of event in control\n\n")

    print(f"\n📊 Statistical summary saved to: {output_file}")
    return output_file


def create_binary_metric_plot_f1(data, metric, y, output_dir, control_condition=None, n_conditions=2):
    """Create a visualization for F1 binary metrics matching boxplot style with significance annotations"""

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

    # Sort: control first, then others alphabetically
    if control_condition and control_condition in prop_data[y].values:
        control_row = prop_data[prop_data[y] == control_condition]
        other_rows = prop_data[prop_data[y] != control_condition].sort_values(y)
        prop_data = pd.concat([control_row, other_rows]).reset_index(drop=True)
    else:
        prop_data = prop_data.sort_values(y).reset_index(drop=True)

    # Get pretraining colors matching boxplot style
    pretraining_colors = create_pretraining_colors()

    # Calculate statistical significance for each comparison vs control
    significance_results = {}
    if control_condition and n_conditions > 1:
        control_data = data[data[y] == control_condition][metric].dropna()
        control_success = int(control_data.sum())
        control_total = len(control_data)

        for condition in prop_data[y].unique():
            if condition == control_condition:
                continue

            test_data = data[data[y] == condition][metric].dropna()
            test_success = int(test_data.sum())
            test_total = len(test_data)

            # Fisher's exact test
            table = [[test_success, test_total - test_success], [control_success, control_total - control_success]]
            odds_ratio, p_value = fisher_exact(table)

            # Determine significance level
            if p_value < 0.001:
                sig_level = "***"
            elif p_value < 0.01:
                sig_level = "**"
            elif p_value < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"

            significance_results[condition] = {"p_value": p_value, "sig_level": sig_level, "odds_ratio": odds_ratio}

    # Create figure with size matching boxplot style
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))

    # Map conditions to colors (matching boxplot style)
    colors = []
    for _, row in prop_data.iterrows():
        condition = str(row[y])
        color = pretraining_colors.get(condition, "#808080")  # Gray fallback
        colors.append(color)

    # Create bar plot
    x_positions = range(len(prop_data))
    bars = ax.bar(
        x_positions, prop_data["proportion"], width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add sample sizes as text on bars
    for i, (bar, row) in enumerate(zip(bars, prop_data.itertuples())):
        height = bar.get_height()
        # Display success/total
        label_text = f"{row.n_success}/{row.n_total}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontname="Arial",
        )

    # Add significance annotations (pairwise comparisons vs control)
    if control_condition and len(significance_results) > 0:
        y_max = prop_data["proportion"].max()

        # For each non-control condition, add significance bar and annotation
        control_idx = prop_data[prop_data[y] == control_condition].index[0]

        for i, row in prop_data.iterrows():
            condition = row[y]
            if condition == control_condition:
                continue

            if condition in significance_results:
                sig_info = significance_results[condition]
                sig_level = sig_info["sig_level"]

                if sig_level != "ns":  # Only show significant results
                    # Draw significance bar between control and this condition
                    bar_height = y_max + 0.08 + (i - control_idx - 1) * 0.08  # Stagger if multiple comparisons

                    # Draw horizontal line
                    ax.plot([control_idx, i], [bar_height, bar_height], "k-", linewidth=1.5)

                    # Draw vertical ticks at ends
                    tick_height = 0.02
                    ax.plot(
                        [control_idx, control_idx],
                        [bar_height - tick_height, bar_height + tick_height],
                        "k-",
                        linewidth=1.5,
                    )
                    ax.plot([i, i], [bar_height - tick_height, bar_height + tick_height], "k-", linewidth=1.5)

                    # Add significance stars
                    ax.text(
                        (control_idx + i) / 2,
                        bar_height + 0.02,
                        sig_level,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontname="Arial",
                    )

        # Add p-value text box in top-left corner (for 2-group comparison)
        if n_conditions == 2:
            # Get the p-value from the single comparison
            non_control_conditions = [c for c in significance_results.keys()]
            if len(non_control_conditions) == 1:
                p_value = significance_results[non_control_conditions[0]]["p_value"]

                if p_value < 0.001:
                    p_text = f"p < 0.001"
                elif p_value < 0.01:
                    p_text = f"p = {p_value:.3f}"
                else:
                    p_text = f"p = {p_value:.3f}"

                ax.text(
                    0.02,
                    0.98,
                    p_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontname="Arial",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8, linewidth=1),
                )

    # Set x-axis labels with sample sizes
    ax.set_xticks(x_positions)
    x_labels = []
    for i, row in prop_data.iterrows():
        cond = row[y]
        n_total = row["n_total"]
        label = map_condition_labels(str(cond)) if map_condition_labels(str(cond)) else str(cond)
        x_labels.append(f"{label}\n(n={n_total})")
    ax.set_xticklabels(x_labels, fontsize=10, fontname="Arial")

    # Set y-axis label with elegant metric name
    metric_display_name = get_elegant_metric_name(metric)
    ax.set_ylabel(f"Proportion\n{metric_display_name}", fontsize=11, fontname="Arial")

    # Adjust y-axis limit to accommodate significance bars
    if (
        control_condition
        and len(significance_results) > 0
        and any(s["sig_level"] != "ns" for s in significance_results.values())
    ):
        n_sig = sum(1 for s in significance_results.values() if s["sig_level"] != "ns")
        ax.set_ylim(0, 1.0 + 0.15 + n_sig * 0.08)
    else:
        ax.set_ylim(0, 1.1)

    # Format y-axis tick labels
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_yticklabels():
        label.set_fontname("Arial")

    # Clean formatting - no grid, white background, with tick marks
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    # Add tick marks pointing outward
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.5)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()

    # Save the plot with editable text
    output_file = output_dir / f"f1_{metric}_binary_{y}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"  Binary plot saved: {output_file.name}")


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

    # Initialize results storage
    continuous_stats_results = []
    binary_results = None
    overall_binary_results = None  # Initialize here

    # Initialize results storage for both test types
    mannwhitney_stats_results = []
    permutation_stats_results = []

    # Process continuous metrics with BOTH Mann-Whitney U tests AND Permutation tests
    if continuous_metrics:
        # Filter to only continuous metrics + required columns
        continuous_data = dataset[["Pretraining"] + continuous_metrics].copy()

        # Loop through both test types
        for test_type in ["mannwhitney", "permutation"]:
            # Create test-specific output directory
            if test_type == "mannwhitney":
                test_output_dir = (
                    ball_output_dir.parent.parent / "F1_Metrics_byPretraining_Mannwhitney" / f"ball_{ball_identity}"
                )
                test_label = "Mann-Whitney U"
            else:
                test_output_dir = (
                    ball_output_dir.parent.parent / "F1_Metrics_byPretraining_Permutation" / f"ball_{ball_identity}"
                )
                test_label = "Permutation Test"

            test_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"F1 CONTINUOUS METRICS ANALYSIS ({test_label})")
            print(f"Ball identity: {ball_identity}")
            print(f"Test type: {test_type}")
            print("=" * 60)

            print(f"Generating F1 {test_type} plots...")
            print(f"Continuous metrics dataset shape: {continuous_data.shape}")
            print(f"Output directory: {test_output_dir}")

            # Filter out metrics that should be skipped
            metrics_to_process = continuous_metrics  # Initialize with all metrics
            continuous_data_subset = continuous_data  # Initialize with full data

            if not overwrite:
                print(f"📄 Checking for existing plots...")
                metrics_to_process = []
                for metric in continuous_metrics:
                    if not should_skip_metric(metric, test_output_dir, overwrite):
                        metrics_to_process.append(metric)

                if len(metrics_to_process) < len(continuous_metrics):
                    skipped_count = len(continuous_metrics) - len(metrics_to_process)
                    print(f"📄 Skipped {skipped_count} metrics with existing plots")

                if not metrics_to_process:
                    print(f"📄 All continuous metrics already have plots, skipping...")
                else:
                    continuous_data_subset = continuous_data[["Pretraining"] + metrics_to_process]

            if metrics_to_process:
                condition_start_time = time.time()
                print(f"⏱️  Processing {len(metrics_to_process)} continuous metrics for F1 analysis...")

                # Check for NaN annotations
                nan_annotations = get_nan_annotations(continuous_data_subset, metrics_to_process, "Pretraining")
                if nan_annotations:
                    print(f"📋 NaN values detected in {len(nan_annotations)} metrics")

                try:
                    # Use simple F1-specific plotting with test type
                    print(f"📊 Creating simple F1 pretraining comparison plots ({test_type})...")
                    current_stats_results = create_f1_pretraining_plots(
                        continuous_data_subset,
                        metrics=metrics_to_process,
                        output_dir=test_output_dir,
                        y_col="Pretraining",
                        test_type=test_type,
                    )

                    # Store results in appropriate variable
                    if test_type == "mannwhitney":
                        mannwhitney_stats_results = current_stats_results
                    else:
                        permutation_stats_results = current_stats_results

                    condition_time = time.time() - condition_start_time
                    print(f"⏱️  F1 continuous metrics ({test_type}) completed in {condition_time:.1f}s")
                except Exception as e:
                    print(f"❌ Error processing F1 continuous metrics ({test_type}): {e}")
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
            binary_results, overall_binary_results = analyze_binary_metrics_f1(
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

    # Generate comprehensive Markdown summaries for both test types
    if (mannwhitney_stats_results or permutation_stats_results) or binary_results is not None:
        print(f"\n{'='*60}")
        print(f"GENERATING STATISTICAL SUMMARIES")
        print("=" * 60)

        # Generate Mann-Whitney summary
        if mannwhitney_stats_results or binary_results is not None:
            try:
                mannwhitney_output_dir = (
                    ball_output_dir.parent.parent / "F1_Metrics_byPretraining_Mannwhitney" / f"ball_{ball_identity}"
                )
                summary_file = generate_markdown_summary(
                    mannwhitney_stats_results,
                    binary_results,
                    overall_binary_results,
                    mannwhitney_output_dir,
                    ball_identity=ball_identity,
                    test_type="mannwhitney",
                )
                print(f"✅ Mann-Whitney statistical summary generated successfully")
            except Exception as e:
                print(f"❌ Error generating Mann-Whitney statistical summary: {e}")
                if test_mode:
                    print("🧪 TEST MODE: Continuing despite error...")

        # Generate Permutation test summary
        if permutation_stats_results:
            try:
                permutation_output_dir = (
                    ball_output_dir.parent.parent / "F1_Metrics_byPretraining_Permutation" / f"ball_{ball_identity}"
                )
                summary_file = generate_markdown_summary(
                    permutation_stats_results,
                    None,  # Binary tests only in Mann-Whitney
                    None,  # No overall binary for permutation
                    permutation_output_dir,
                    ball_identity=ball_identity,
                    test_type="permutation",
                )
                print(f"✅ Permutation test statistical summary generated successfully")
            except Exception as e:
                print(f"❌ Error generating Permutation statistical summary: {e}")
                if test_mode:
                    print("🧪 TEST MODE: Continuing despite error...")

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
