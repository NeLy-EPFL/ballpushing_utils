#!/usr/bin/env python3
"""
F1 Control Condition Comparison Script

This script loads the F1 control dataset and allows comparison of behavioral metrics
across F1_conditions (control, pretrained, pretrained_unlocked) or Pretraining (naive vs pretrained).

Performs comprehensive statistical analysis and creates visualization plots.

Usage:
    python f1_control_condition_comparison.py
    python f1_control_condition_comparison.py --show  # Display plots instead of just saving
    python f1_control_condition_comparison.py --analysis f1_condition  # Compare by F1_condition
    python f1_control_condition_comparison.py --analysis pretraining   # Compare by pretraining
    python f1_control_condition_comparison.py --metrics interaction_rate,interaction_duration
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

# matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
from itertools import combinations
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# DATASET PATHS - Update these if datasets change
# ============================================================================
DATASET_PATH = (
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260121_16_F1_coordinates_F1_New_Data/summary/pooled_summary.feather"
)
OUTPUT_DIR = "/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New/PR"

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# F1 conditions to analyze
SELECTED_F1_CONDITIONS = [
    "control",
    "pretrained",
    "pretrained_unlocked",
]

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Binary metrics to analyze
BINARY_METRICS = ["has_major", "has_finished"]

# Continuous metrics - matching PCA analysis
# This list is used by default; set via --metrics to override
DEFAULT_CONTINUOUS_METRICS = [
    "ball_proximity_proportion_200px",  # Time spent near ball
    "distance_moved",  # Total ball distance moved
    "max_distance",  # Maximum ball distance
    "nb_events",  # Number of interaction events
    "has_major",  # Binary - achieved major event
    "has_finished",  # Binary - completed task
    "significant_ratio",  # Proportion of significant events
]

# Columns to exclude from auto-detection
EXCLUDED_COLUMNS = {
    "fly",
    "Date",
    "date",
    "filename",
    "video",
    "path",
    "folder",
    "F1_condition",
    "f1_condition",
    "Pretraining",
    "pretraining",
    "ball_condition",
    "ball_identity",
    "arena",
    "corridor",
    "time",
    "frame",
    "index",
}

# Ball condition filtering
FILTER_FOR_TEST_BALL = True
TEST_BALL_VALUES = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]

# ============================================================================
# PLOT STYLING - Edit these to customize colors and styles
# ============================================================================

# F1_condition color palette
F1_CONDITION_COLORS = {
    "control": "#ff7f0e",  # Orange
    "pretrained": "#4682B4",  # Steelblue
    "pretrained_unlocked": "#2ca02c",  # Green
}

# Pretraining styles
PRETRAINING_COLORS = {
    "n": "#ff7f0e",  # Orange (Naive)
    "y": "#4682B4",  # Steelblue (Pretrained)
}

# Plot configuration
FIGURE_SIZE = (6, 8)  # Width, Height for individual metric plots
DPI = 300
PLOT_FORMATS = ["png", "pdf", "svg"]  # Save in multiple formats
JITTER_AMOUNT = 0.08  # Amount of jitter for scatter points (aligned with boxplot width of 0.3)
SCATTER_SIZE = 60  # Increased for better visibility
SCATTER_ALPHA = 0.7

# Y-axis limits for handling outliers
YLIM_PERCENTILE_LOW = 0  # Lower percentile for y-axis (0 = minimum)
YLIM_PERCENTILE_HIGH = 100  # Upper percentile for y-axis (100 = include all data)
YLIM_MARGIN = 0.05  # Add 5% margin above/below the percentile limits

# Statistical testing
ALPHA = 0.05
MIN_SAMPLE_SIZE = 2


def significance_label(p_value):
    """Return significance stars or ns for a p-value."""
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# Statistical method selection
USE_LMM = True  # Use Linear Mixed-Effects Models for effect sizes and diagnostics
USE_RESIDUAL_PERMUTATION = False  # Permutation tests on residuals
USE_MANN_WHITNEY = True  # Enabled: use for significance annotations (simpler, no blocking assumptions)

# Diagnostic options
SAVE_DIAGNOSTIC_PLOTS = True  # Save diagnostic plots for model validation
TEST_DATE_EFFECT = True  # Test whether Date has a significant effect on metrics

# Permutation test settings
N_PERMUTATIONS = 10000  # Number of permutations for permutation tests


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


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
    if "ratio" in metric_name.lower():
        return False
    return any(keyword in metric_name.lower() for keyword in time_keywords)


def convert_metric_data(data, metric_name):
    """Convert metric data from pixels to mm or seconds to minutes"""
    if is_distance_metric(metric_name):
        return data / PIXELS_PER_MM
    elif is_time_metric(metric_name):
        return data / 60.0
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "mm"
    elif is_time_metric(metric_name):
        return "min"
    return ""


def get_elegant_metric_name(metric_name):
    """Convert metric code names to elegant display names"""
    metric_name_map = {
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
        "max_distance": "Max ball distance",
        "distance_moved": "Total ball distance moved",
        "distance_ratio": "Distance ratio",
        "fly_distance_moved": "Fly distance moved",
        "max_distance_from_ball": "Maximum distance from ball",
        "overall_interaction_rate": "Interaction rate",
        "interaction_rate": "Interaction rate",
        "interaction_persistence": "Interaction persistence",
        "interaction_proportion": "Interaction proportion",
        "time_to_first_interaction": "Time to first interaction",
        "time_until_first_interaction": "Time until first interaction",
        "interaction_duration": "Mean interaction duration",
        "total_interaction_time": "Total interaction time",
        "mean_interaction_duration": "Mean interaction duration",
        "head_ball_distance_during_interaction": "Head-ball distance during interaction",
        "head_ball_distance_outside_interaction": "Head-ball distance outside interaction",
        "pushed": "Number of pushes",
        "pulled": "Number of pulls",
        "pulling_ratio": "Pulling ratio",
        "head_pushing_ratio": "Head pushing ratio",
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
        "pauses_persistence": "Pauses persistence",
        "cumulated_breaks_duration": "Total break duration",
        "nb_freeze": "Number of freezes",
        "median_freeze_duration": "Median freeze duration",
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "chamber_exit_time": "Chamber exit time",
        "time_chamber_beginning": "Time in chamber (early)",
        "has_finished": "Task completed",
        "persistence_at_end": "Persistence at end",
        "fraction_not_facing_ball": "Fraction not facing ball",
        "leg_visibility_ratio": "Leg visibility ratio",
        "flailing": "Leg flailing",
        "median_head_ball_distance": "Median head-ball distance",
        "mean_head_ball_distance": "Mean head-ball distance",
        "ball_proximity_proportion_70px": "Ball proximity (<70px)",
        "ball_proximity_proportion_140px": "Ball proximity (<140px)",
        "ball_proximity_proportion_200px": "Ball proximity (<200px)",
    }
    return metric_name_map.get(metric_name, metric_name.replace("_", " ").title())


def format_metric_label(metric_name):
    """Format metric name with units for plot labels"""
    elegant_name = get_elegant_metric_name(metric_name)
    unit = get_metric_unit(metric_name)
    if unit:
        return f"{elegant_name} ({unit})"
    return elegant_name


def save_model_diagnostic_plots(model, metric, output_dir):
    """
    Save diagnostic plots for OLS model validation.
    Includes: Q-Q plot, residuals vs fitted, histogram, and scale-location plot.
    """
    try:
        from statsmodels.graphics.gofplots import ProbPlot

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Q-Q plot
        pp = ProbPlot(model.resid)
        pp.qqplot(ax=axes[0, 0], line="45")
        axes[0, 0].set_title("Q-Q Plot", fontweight="bold")
        axes[0, 0].grid(alpha=0.3)

        # Residuals vs Fitted
        fitted = model.fittedvalues
        residuals = model.resid
        axes[0, 1].scatter(fitted, residuals, alpha=0.5, s=30)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Fitted values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Fitted", fontweight="bold")
        axes[0, 1].grid(alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Histogram of Residuals", fontweight="bold")
        axes[1, 0].grid(alpha=0.3, axis="y")

        # Scale-location plot
        standardized_resid = residuals / np.std(residuals)
        axes[1, 1].scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.5, s=30)
        axes[1, 1].set_xlabel("Fitted values")
        axes[1, 1].set_ylabel("√|Standardized residuals|")
        axes[1, 1].set_title("Scale-Location Plot", fontweight="bold")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        output_path = Path(output_dir) / f"diagnostic_{metric}.{PLOT_FORMAT}"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  Warning: Could not save diagnostic plots: {e}")


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def auto_detect_continuous_metrics(df):
    """Auto-detect continuous (numeric) metrics from the dataset"""
    continuous_metrics = []

    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            continuous_metrics.append(col)

    print(f"\n📊 Auto-detected {len(continuous_metrics)} continuous metrics:")
    if len(continuous_metrics) > 0:
        print(f"  {', '.join(continuous_metrics[:5])}...")

    return continuous_metrics if continuous_metrics else DEFAULT_CONTINUOUS_METRICS


def load_dataset():
    """Load dataset from feather file"""
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_feather(DATASET_PATH)
    print(f"  Loaded {len(df)} rows")
    return df


def detect_columns(df):
    """Auto-detect f1_condition and pretraining columns"""
    f1_condition_col = None
    pretraining_col = None

    for col in df.columns:
        if "f1" in col.lower() and "condition" in col.lower() and f1_condition_col is None:
            f1_condition_col = col
        if "pretrain" in col.lower() and pretraining_col is None:
            pretraining_col = col

    if f1_condition_col is None:
        raise ValueError("Could not find F1_condition column")
    if pretraining_col is None:
        raise ValueError("Could not find pretraining column")

    print(f"  F1 Condition column: {f1_condition_col}")
    print(f"  Pretraining column: {pretraining_col}")

    return f1_condition_col, pretraining_col


def filter_dataset(df, f1_condition_col):
    """Filter dataset for selected conditions and test ball"""
    # Filter for selected F1 conditions
    if SELECTED_F1_CONDITIONS:
        df = df[df[f1_condition_col].isin(SELECTED_F1_CONDITIONS)].copy()
        print(f"  Filtered for {len(SELECTED_F1_CONDITIONS)} F1 conditions: {SELECTED_F1_CONDITIONS}")

    # Filter for test ball
    if FILTER_FOR_TEST_BALL:
        ball_col = "ball_identity"
        if ball_col not in df.columns:
            print(f"  Warning: {ball_col} column not found, skipping ball filter")
        else:
            initial_size = len(df)
            df = df[df[ball_col].isin(TEST_BALL_VALUES)].copy()
            print(f"  Filtered for test ball: {len(df)} rows (removed {initial_size - len(df)})")

    print(f"  Final dataset size: {len(df)} rows")
    return df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def perform_binary_analysis(df, condition_col, condition_values):
    """Perform statistical analysis on binary metrics"""
    results = {}

    for metric in BINARY_METRICS:
        if metric not in df.columns:
            continue

        results[metric] = {}

        # Chi-square or Fisher exact test
        contingency_table = pd.crosstab(df[condition_col], df[metric])

        if contingency_table.shape[0] < 2:
            continue

        try:
            if (contingency_table < 5).sum().sum() > 0:
                # Use Fisher exact for small samples (not applicable for >2x2, so use chi-square)
                if contingency_table.shape == (2, 2):
                    oddsratio, p_value = fisher_exact(contingency_table)
                    test_name = "Fisher exact"
                else:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    test_name = "Chi-square"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                test_name = "Chi-square"

            results[metric]["p_value"] = p_value
            results[metric]["test"] = test_name
            results[metric]["significant"] = p_value < ALPHA

            # Calculate proportions per condition
            for cond in condition_values:
                if cond in df[condition_col].values:
                    subset = df[df[condition_col] == cond]
                    prop = subset[metric].mean()
                    results[metric][f"{cond}_prop"] = prop

        except Exception as e:
            print(f"  Warning: Could not analyze {metric}: {e}")

    return results


def perform_lmm_continuous_analysis(df, condition_col, condition_values, metrics=None):
    """
    Perform Linear Mixed-Effects Model analysis on continuous metrics.
    """
    if metrics is None:
        metrics = auto_detect_continuous_metrics(df)

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        try:
            # Prepare data for LMM
            data = df[[condition_col, "Date", metric]].copy()
            data = data.dropna()

            if len(data) < MIN_SAMPLE_SIZE:
                continue

            # Fit LMM: Metric ~ Condition + Date
            formula = f"{metric} ~ C({condition_col}) + C(Date)"
            model = smf.ols(formula, data=data).fit()

            results[metric] = {
                "model": model,
                "rsquared": model.rsquared,
                "rsquared_adj": model.rsquared_adj,
            }

            # Test condition effect
            # Create contrast matrix to test if all condition coefficients are zero
            n_conditions = len(condition_values)
            if n_conditions > 1:
                condition_terms = [f"C({condition_col})[T.{cond}]" for cond in condition_values[1:]]
                if condition_terms:
                    contrast_matrix = np.zeros((len(condition_terms), len(model.params)))
                    for i, term in enumerate(condition_terms):
                        if term in model.params.index:
                            contrast_matrix[i, model.params.index.get_loc(term)] = 1

                    f_stat = model.fvalue
                    p_value_f = model.f_pvalue

                    results[metric]["f_stat"] = f_stat
                    results[metric]["p_value_condition"] = p_value_f
                    results[metric]["significant"] = p_value_f < ALPHA

                    # Effect sizes and estimates per condition
                    for cond in condition_values:
                        subset = data[data[condition_col] == cond]
                        mean = subset[metric].mean()
                        std = subset[metric].std()
                        n = len(subset)
                        results[metric][f"{cond}_mean"] = mean
                        results[metric][f"{cond}_std"] = std
                        results[metric][f"{cond}_n"] = n

        except Exception as e:
            print(f"  Warning: Could not analyze {metric} with LMM: {e}")

    return results


def perform_mann_whitney_continuous_analysis(df, condition_col, condition_values, metrics=None):
    """
    Perform Mann-Whitney U test analysis on continuous metrics.
    Simple non-parametric test for pairwise comparisons.
    """
    if metrics is None:
        metrics = auto_detect_continuous_metrics(df)

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        try:
            data = df[[condition_col, metric]].copy()
            data = data.dropna()

            if len(data) < MIN_SAMPLE_SIZE:
                continue

            results[metric] = {"pairwise": {}}

            # Pairwise comparisons
            for cond1, cond2 in combinations(condition_values, 2):
                group1 = data[data[condition_col] == cond1][metric].values
                group2 = data[data[condition_col] == cond2][metric].values

                if len(group1) < 2 or len(group2) < 2:
                    continue

                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")

                results[metric]["pairwise"][f"{cond1}_vs_{cond2}"] = {
                    "p_value": p_value,
                    "significant": p_value < ALPHA,
                    "mean_diff": group1.mean() - group2.mean(),
                }

        except Exception as e:
            print(f"  Warning: Could not analyze {metric} with Mann-Whitney: {e}")

    return results
    """
    Perform permutation test analysis on continuous metrics.
    """
    if metrics is None:
        metrics = auto_detect_continuous_metrics(df)

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        try:
            data = df[[condition_col, metric]].copy()
            data = data.dropna()

            if len(data) < MIN_SAMPLE_SIZE:
                continue

            results[metric] = {"pairwise": {}}

            # Pairwise comparisons
            for cond1, cond2 in combinations(condition_values, 2):
                group1 = data[data[condition_col] == cond1][metric].values
                group2 = data[data[condition_col] == cond2][metric].values

                if len(group1) < 2 or len(group2) < 2:
                    continue

                # Observed difference in means
                obs_diff = np.abs(group1.mean() - group2.mean())

                # Permutation test
                combined = np.concatenate([group1, group2])
                perm_diffs = []
                np.random.seed(42)
                for _ in range(n_permutations):
                    perm_combined = np.random.permutation(combined)
                    perm_g1 = perm_combined[: len(group1)]
                    perm_g2 = perm_combined[len(group1) :]
                    perm_diffs.append(np.abs(perm_g1.mean() - perm_g2.mean()))

                p_value = np.mean(np.array(perm_diffs) >= obs_diff)

                # Cohen's d effect size
                pooled_std = np.sqrt(
                    ((len(group1) - 1) * group1.std() ** 2 + (len(group2) - 1) * group2.std() ** 2)
                    / (len(group1) + len(group2) - 2)
                )
                cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

                results[metric]["pairwise"][f"{cond1}_vs_{cond2}"] = {
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "mean_diff": group1.mean() - group2.mean(),
                    "significant": p_value < ALPHA,
                }

        except Exception as e:
            print(f"  Warning: Could not analyze {metric} with permutation test: {e}")

    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def get_condition_color(condition, analysis_type):
    """Get color for a condition"""
    if analysis_type == "f1_condition":
        return F1_CONDITION_COLORS.get(condition, "#808080")
    elif analysis_type == "pretraining":
        return PRETRAINING_COLORS.get(str(condition).lower(), "#808080")
    return "#808080"


def plot_binary_metric_single(
    df, condition_col, condition_values, analysis_type, metric, output_dir, stats_results=None
):
    """Create a single bar plot for one binary metric"""
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Calculate proportions and counts
    data_to_plot = []
    labels_with_n = []

    for cond in condition_values:
        subset = df[df[condition_col] == cond]
        prop = subset[metric].mean()
        n = len(subset)
        data_to_plot.append(prop * 100)
        labels_with_n.append(f"{cond}\n(n={n})")

    # Plot bars
    positions = np.arange(len(condition_values))
    colors = [get_condition_color(cond, analysis_type) for cond in condition_values]

    bars = ax.bar(positions, data_to_plot, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, data_to_plot):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_with_n)
    ax.set_ylabel("Proportion (%)", fontsize=12, fontweight="bold")
    ax.set_title(get_elegant_metric_name(metric), fontsize=14, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance annotation if available
    if stats_results and metric in stats_results:
        p_val = stats_results[metric].get("p_value")
        if p_val is not None:
            sig_str = f"{significance_label(p_val)} (p={p_val:.3g})"
            ax.text(
                0.5,
                0.98,
                sig_str,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
            )

    plt.tight_layout()

    # Save in multiple formats
    for fmt in PLOT_FORMATS:
        output_path = Path(output_dir) / f"binary_{metric}.{fmt}"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_binary_metrics(df, condition_col, condition_values, analysis_type, stats_results, output_dir):
    """Plot all binary metrics"""
    print(f"\n📊 Plotting binary metrics...")

    for metric in BINARY_METRICS:
        if metric in df.columns:
            plot_binary_metric_single(
                df, condition_col, condition_values, analysis_type, metric, output_dir, stats_results
            )


def plot_continuous_metric_single(
    df, condition_col, condition_values, analysis_type, metric, output_dir, lmm_results=None, mw_results=None
):
    """Create a single boxplot for one continuous metric"""
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Prepare data
    data_list = []
    labels_with_n = []
    colors = []

    for cond in condition_values:
        subset = df[df[condition_col] == cond][metric].copy()
        subset = convert_metric_data(subset, metric)
        data_list.append(subset.dropna().values)
        labels_with_n.append(f"{cond}\n(n={len(subset.dropna())})")
        colors.append(get_condition_color(cond, analysis_type))

    # Check if we have any data
    all_data = np.concatenate(data_list) if data_list and any(len(d) > 0 for d in data_list) else np.array([])
    if len(all_data) == 0:
        print(f"  Skipping {metric}: no valid data")
        plt.close()
        return

    # Set y-axis limits based on percentiles
    y_min = np.percentile(all_data, YLIM_PERCENTILE_LOW)
    y_max = np.percentile(all_data, YLIM_PERCENTILE_HIGH)
    y_range = max(y_max - y_min, 1e-6)

    # Add extra space for significance annotations (scale with number of pairwise comparisons)
    n_comparisons = len(list(combinations(condition_values, 2)))
    annotation_space = y_range * max(0.20, 0.08 * n_comparisons)
    y_min -= y_range * YLIM_MARGIN
    y_max += annotation_space

    # Create boxplots
    positions = np.arange(len(condition_values))
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Style boxes - no fill, black outline only
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    # Add jittered scatter points
    for data, pos, color in zip(data_list, positions, colors):
        if len(data) > 0:
            jitter = np.random.normal(0, JITTER_AMOUNT, len(data))
            ax.scatter(
                pos + jitter,
                data,
                alpha=SCATTER_ALPHA,
                s=SCATTER_SIZE,
                color=color,
                edgecolors=color,
                linewidth=0.5,
                zorder=3,
            )

    # Add significance annotations
    if mw_results and metric in mw_results:
        # Start annotations just above the data
        y_position = y_max - (y_range * 0.05)  # Start 5% below the top
        y_step = y_range * 0.06  # Step size for each annotation

        for comparison, result in mw_results[metric].get("pairwise", {}).items():
            parts = comparison.split("_vs_")
            if len(parts) != 2:
                continue

            cond1, cond2 = parts[0], parts[1]
            if cond1 in condition_values and cond2 in condition_values:
                idx1 = condition_values.index(cond1)
                idx2 = condition_values.index(cond2)
                label = significance_label(result.get("p_value"))
                ax.plot([idx1, idx2], [y_position, y_position], "k-", linewidth=1)
                ax.text((idx1 + idx2) / 2, y_position, label, ha="center", va="bottom", fontsize=12, fontweight="bold")
                y_position -= y_step  # Move down for next annotation

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_with_n)
    ax.set_ylabel(format_metric_label(metric), fontsize=12, fontweight="bold")
    ax.set_title(get_elegant_metric_name(metric), fontsize=14, fontweight="bold")
    ax.set_ylim([y_min, y_max])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in PLOT_FORMATS:
        output_path = Path(output_dir) / f"continuous_{metric}.{fmt}"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_continuous_metrics(
    df,
    condition_col,
    condition_values,
    analysis_type,
    metrics=None,
    output_dir=None,
    lmm_results=None,
    mw_results=None,
):
    """Plot all continuous metrics"""
    if metrics is None:
        metrics = auto_detect_continuous_metrics(df)

    print(f"\n📊 Plotting continuous metrics...")

    for metric in metrics:
        if metric in df.columns:
            plot_continuous_metric_single(
                df, condition_col, condition_values, analysis_type, metric, output_dir, lmm_results, mw_results
            )


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="F1 Control Condition Comparison")
    parser.add_argument(
        "--analysis",
        type=str,
        default=None,
        choices=["f1_condition", "pretraining"],
        help="Analysis to perform (default: run both f1_condition and pretraining)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to analyze (default: PCA metrics)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print("=" * 80)
    print("F1 CONTROL CONDITION COMPARISON")
    print("=" * 80)

    # Determine which analyses to run
    if args.analysis:
        analyses_to_run = [args.analysis]
    else:
        analyses_to_run = ["f1_condition", "pretraining"]

    # Load and prepare data once
    df = load_dataset()
    f1_condition_col, pretraining_col = detect_columns(df)
    df = filter_dataset(df, f1_condition_col)

    # Parse metrics if specified
    if args.metrics:
        metrics = args.metrics.split(",")
    else:
        metrics = DEFAULT_CONTINUOUS_METRICS

    # Run each analysis
    for analysis_type in analyses_to_run:
        print(f"\n\n{'='*80}")
        print(f"ANALYSIS: {analysis_type.upper()}")
        print(f"{'='*80}")

        # Select condition column and values
        if analysis_type == "f1_condition":
            condition_col = f1_condition_col
            condition_values = SELECTED_F1_CONDITIONS
        else:  # pretraining
            condition_col = pretraining_col
            condition_values = ["n", "y"]

        # Create output subdirectory
        output_dir = Path(OUTPUT_DIR) / analysis_type
        output_dir.mkdir(parents=True, exist_ok=True)

        # ====================================================================
        # BINARY METRICS ANALYSIS
        # ====================================================================
        print(f"\n{'='*80}")
        print("BINARY METRICS ANALYSIS")
        print(f"{'='*80}")

        binary_stats = perform_binary_analysis(df, condition_col, condition_values)
        plot_binary_metrics(df, condition_col, condition_values, analysis_type, binary_stats, output_dir)

        # ====================================================================
        # CONTINUOUS METRICS ANALYSIS
        # ====================================================================
        print(f"\n{'='*80}")
        print("CONTINUOUS METRICS ANALYSIS")
        print(f"{'='*80}")

        lmm_results = perform_lmm_continuous_analysis(df, condition_col, condition_values, metrics)
        mw_results = perform_mann_whitney_continuous_analysis(df, condition_col, condition_values, metrics)

        print(f"✓ LMM analysis complete on {len(lmm_results)} metrics")
        print(f"✓ Mann-Whitney U tests complete on {len(mw_results)} metrics")

        # ====================================================================
        # PLOTTING
        # ====================================================================
        print(f"\n{'='*80}")
        print("GENERATING PLOTS")
        print(f"{'='*80}")

        plot_continuous_metrics(
            df, condition_col, condition_values, analysis_type, metrics, output_dir, lmm_results, mw_results
        )

        print(f"\n✓ All plots saved to: {output_dir}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'='*80}")
    print(f"Plots saved to: {OUTPUT_DIR}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
