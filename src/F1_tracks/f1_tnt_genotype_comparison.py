#!/usr/bin/env python3
"""
F1 TNT Genotype Comparison Script

This script loads the full F1 TNT dataset and allows selection of specific genotypes
for comparison and plotting. Genotypes can be included/excluded by commenting out
lines in the SELECTED_GENOTYPES list below.

Usage:
    python f1_tnt_genotype_comparison.py
    python f1_tnt_genotype_comparison.py --show  # Display plots instead of just saving
    python f1_tnt_genotype_comparison.py --metrics interaction_rate,interaction_duration
    python f1_tnt_genotype_comparison.py --no-stats  # Generate plots without statistical annotations
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
from itertools import combinations
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# DATASET PATHS - Update these if datasets change
# ============================================================================
DATASET_PATHS = {
    "summary": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather",
    "coordinates": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather",
}

OUTPUT_DIR = "/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/MB"

# ============================================================================
# GENOTYPE SELECTION - Comment out genotypes to exclude from analysis
# ============================================================================
# NOTE: This list will be auto-populated when you run the script for the first time.
# After that, you can comment out genotypes you want to exclude.

SELECTED_GENOTYPES = [
    # === CONTROLS ===
    # "TNTxEmptyGal4",
    "TNTxEmptySplit",
    #'TNTxPR',
    # === EXPERIMENTAL GENOTYPES (paired with internal controls) ===
    # "TNTxDDC",
    # "TNTxTH",
    #'PRxTH',
    # "TNTxTRH",
    # "TNTxMB247",
    #'PRxMB247',
    "TNTxLC10-2",
    "TNTxLC16-1",
    #'PRxLC16-1',
]

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Binary metrics to analyze (matching PCA analysis)
BINARY_METRICS = ["has_major", "has_finished"]

# Continuous metrics - will be auto-detected from dataset
# NOTE: These metrics match those used in PCA analysis (f1_pca_analysis.py)
# This list is used as a fallback if auto-detection fails
DEFAULT_CONTINUOUS_METRICS = [
    "ball_proximity_proportion_70px",  # Time spent near ball (small threshold),
    "ball_proximity_proportion_140px",  # Time spent near ball (medium threshold),
    "ball_proximity_proportion_200px",  # Time spent near ball (large threshold)
    "distance_moved",  # Total ball distance moved
    "max_distance",  # Maximum ball distance
    "nb_events",  # Number of interaction events
    "significant_ratio",  # Proportion of significant events
    "raw_distance_moved",  # Total ball distance moved (raw)
    "raw_max_distance",  # Maximum ball distance (raw)
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
    "Genotype",
    "genotype",
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

# Genotype color palette - Edit these colors as needed
GENOTYPE_COLORS = {
    # Controls
    "TNTxEmptyGal4": "#7f7f7f",  # Gray
    "TNTxEmptySplit": "#7f7f7f",  # Yellow-green
    "TNTxPR": "#7f7f7f",  # Cyan
    # Experimental genotypes
    "TNTxDDC": "#8B4513",  # Brown (MB extrinsic)
    "TNTxTH": "#8B4513",  # Brown (MB extrinsic)
    "PRxTH": "#8B4513",  # Brown (MB extrinsic)
    "TNTxTRH": "#8B4513",  # Brown (MB extrinsic)
    "TNTxMB247": "#1f77b4",  # Blue (MB)
    "PRxMB247": "#1f77b4",  # Blue (MB)
    "TNTxLC10-2": "#ff7f0e",  # Orange (Vision)
    "TNTxLC16-1": "#ff7f0e",  # Orange (Vision)
    "PRxLC16-1": "#ff7f0e",  # Orange (Vision)
}

# Pretraining styles - Controls visual distinction for pretrained vs naive
PRETRAINING_STYLES = {
    "n": {"alpha": 0.6, "edgecolor": "black", "linewidth": 1.5, "label_suffix": " (Naive)"},  # Naive (no pretraining)
    "y": {"alpha": 1.0, "edgecolor": "black", "linewidth": 2.5, "label_suffix": " (Pretrained)"},  # Pretrained
}

# Plot configuration - Base values (will be adapted based on number of genotypes)
FIGURE_SIZE = (14, 8)  # Width, Height for individual metric plots
DPI = 300
PLOT_FORMATS = ["png", "pdf", "svg"]  # Save plots in multiple formats
JITTER_AMOUNT = 0.15  # Amount of jitter for scatter points (baseline - adapted per analysis)
SCATTER_SIZE = 30  # Baseline scatter point size (adapted per analysis)
SCATTER_ALPHA = 0.5

# Y-axis limits for handling outliers
YLIM_PERCENTILE_LOW = 0  # Lower percentile for y-axis (0 = minimum)
YLIM_PERCENTILE_HIGH = 100  # Upper percentile for y-axis (100 = maximum)
YLIM_MARGIN = 0.05  # Add 5% margin above/below the percentile limits

# Statistical testing
ALPHA = 0.05
MIN_SAMPLE_SIZE = 2

# Statistical method selection
USE_LMM = True  # Use Linear Mixed-Effects Models for effect sizes and diagnostics
USE_RESIDUAL_PERMUTATION = (
    True  # Primary method: permutation tests on residuals (distribution-free, accounts for blocking)
)
USE_MANN_WHITNEY = False  # Disabled: doesn't account for blocking factors (Date, Arena, Side)
FORCE_REGULAR_PERMUTATION = False  # Can be overridden by command-line flag --force-regular-permutation

# Diagnostic options
SAVE_DIAGNOSTIC_PLOTS = True  # Save diagnostic plots (Q-Q, residuals, etc.) for model validation
TEST_DATE_EFFECT = True  # Test whether Date has a significant effect on metrics

# Permutation test settings
N_PERMUTATIONS = 10000  # Number of permutations for permutation tests

# ============================================================================
# ADAPTIVE STYLING AND ANALYSIS METHOD SELECTION
# ============================================================================


def get_adaptive_styling_params(n_genotypes, n_pretraining=2):
    """
    Calculate adaptive styling parameters based on number of genotypes and pretraining conditions.

    With fewer genotypes, we want:
    - Smaller jitter to avoid overlapping boxes
    - Potentially larger figure width to give more space
    - Larger scatter points for better visibility

    Args:
        n_genotypes: Number of genotypes being analyzed
        n_pretraining: Number of pretraining conditions (default 2)

    Returns:
        dict with adaptive parameters: jitter_amount, figure_size, scatter_size
    """
    # Adaptive jitter: inversely proportional to number of genotypes
    # 3 genotypes -> 0.08, 5 -> 0.10, 10 -> 0.12, 15+ -> 0.15
    if n_genotypes <= 3:
        jitter = 0.08
    elif n_genotypes <= 5:
        jitter = 0.10
    elif n_genotypes <= 10:
        jitter = 0.12
    else:
        jitter = 0.15

    # Adaptive figure width based on number of genotypes
    # 3 genotypes -> (10, 8), 5 -> (12, 8), 10+ -> (14, 8), 15+ -> (16, 8)
    if n_genotypes <= 3:
        fig_width = 10
    elif n_genotypes <= 5:
        fig_width = 12
    elif n_genotypes <= 10:
        fig_width = 14
    else:
        fig_width = 16

    # Adaptive scatter size: inversely proportional to number of genotypes
    # More genotypes = smaller points (less visual clutter)
    if n_genotypes <= 3:
        scatter_size = 50  # Larger for focused analysis
    elif n_genotypes <= 5:
        scatter_size = 40
    elif n_genotypes <= 10:
        scatter_size = 30
    else:
        scatter_size = 20

    return {
        "jitter_amount": jitter,
        "figure_size": (fig_width, 8),
        "scatter_size": scatter_size,
    }


def check_lmm_adequacy(n_samples, n_params):
    """
    Check if sample size is adequate for reliable LMM inference.

    Returns:
        tuple: (is_adequate: bool, ratio: float, warning_msg: str or None)
    """
    ratio = n_samples / n_params if n_params > 0 else 0

    if ratio < 5:
        return False, ratio, f"Only {ratio:.1f} samples per parameter (WARNING: severely underpowered)"
    elif ratio < 10:
        return False, ratio, f"{ratio:.1f} samples per parameter (CAUTION: underpowered, prefer >10)"
    else:
        return True, ratio, None


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
        return data / 60
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
        "max_distance_from_ball": "Maximum distance from ball",
        # Interaction metrics
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
        # Ball proximity metrics
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


def format_pretraining_label(value):
    """Format pretraining labels for display"""
    if pd.isna(value):
        return "Unknown"
    value_str = str(value).lower()
    if value_str in ["y", "yes", "true", "1"]:
        return "Pretrained"
    elif value_str in ["n", "no", "false", "0"]:
        return "Naive"
    return str(value)


def save_model_diagnostic_plots(model, metric, output_dir):
    """
    Save diagnostic plots for OLS model validation.
    Includes: Q-Q plot, residuals vs fitted, histogram, and scale-location plot.
    """
    try:
        from scipy import stats as sp_stats

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Q-Q plot
        sp_stats.probplot(model.resid, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title("Q-Q Plot", fontweight="bold")
        axes[0, 0].grid(alpha=0.3)

        # Residuals vs Fitted
        axes[0, 1].scatter(model.fittedvalues, model.resid, alpha=0.5, s=30)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("Fitted values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Fitted Values", fontweight="bold")
        axes[0, 1].grid(alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(model.resid, bins=25, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Histogram of Residuals", fontweight="bold")
        axes[1, 0].grid(alpha=0.3, axis="y")

        # Scale-Location plot (sqrt of standardized residuals)
        standardized_resid = model.resid / model.resid.std()
        axes[1, 1].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.5, s=30)
        axes[1, 1].set_xlabel("Fitted values")
        axes[1, 1].set_ylabel("√|Standardized Residuals|")
        axes[1, 1].set_title("Scale-Location Plot", fontweight="bold")
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle(f"Model Diagnostics: {get_elegant_metric_name(metric)}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = Path(output_dir) / f"{metric}_diagnostics.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"   Diagnostic plots saved: {output_path.name}")

    except Exception as e:
        print(f"   ⚠️ Could not generate diagnostic plots: {e}")


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def auto_detect_continuous_metrics(df):
    """Auto-detect continuous (numeric) metrics from the dataset"""
    continuous_metrics = []

    for col in df.columns:
        # Skip excluded columns
        if col in EXCLUDED_COLUMNS:
            continue

        # Skip columns that are in binary metrics list
        if col in BINARY_METRICS:
            continue

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Skip boolean columns (binary metrics)
            if pd.api.types.is_bool_dtype(df[col]):
                continue

            # Skip columns that only have 0 and 1 values (likely binary)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                continue

            # Additional check: skip if it looks like an ID or categorical
            if not any(keyword in col.lower() for keyword in ["id", "index", "frame"]):
                continuous_metrics.append(col)

    print(f"\n📊 Auto-detected {len(continuous_metrics)} continuous metrics:")
    if len(continuous_metrics) > 0:
        print(f"   {', '.join(continuous_metrics[:10])}")
        if len(continuous_metrics) > 10:
            print(f"   ... and {len(continuous_metrics) - 10} more")

    return continuous_metrics if continuous_metrics else DEFAULT_CONTINUOUS_METRICS


def load_dataset(dataset_type="summary"):
    """Load dataset from feather file"""
    path = DATASET_PATHS.get(dataset_type)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"Loading {dataset_type} dataset from: {path}")
    df = pd.read_feather(path)
    print(f"  Loaded {len(df)} rows")
    return df


def detect_genotypes(df):
    """Auto-detect all genotypes in the dataset"""
    genotype_col = None
    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
            break

    if genotype_col is None:
        raise ValueError("Could not find genotype column in dataset")

    genotypes = df[genotype_col].dropna().unique().tolist()
    genotypes = sorted(genotypes)

    # Separate controls and experimental genotypes
    controls = ["Empty", "EmptySplit", "TNTxPR"]
    control_genotypes = [g for g in controls if g in genotypes]
    experimental_genotypes = [g for g in genotypes if g not in controls]

    return control_genotypes + experimental_genotypes, genotype_col


def filter_dataset(df, genotype_col):
    """Filter dataset for selected genotypes and ball condition"""
    # Filter for selected genotypes
    if SELECTED_GENOTYPES:
        df = df[df[genotype_col].isin(SELECTED_GENOTYPES)].copy()
        print(f"Filtered for {len(SELECTED_GENOTYPES)} selected genotypes: {SELECTED_GENOTYPES}")

    # Filter for ball condition - use ball_identity column (not ball_condition)
    if FILTER_FOR_TEST_BALL:
        # Use ball_identity column which has the correct ball type information
        ball_col = "ball_identity"

        if ball_col in df.columns:
            # Check available ball identity values
            print(f"Available {ball_col} values: {df[ball_col].unique()}")

            # Try different possible values for test ball
            test_ball_data = pd.DataFrame()
            for test_val in TEST_BALL_VALUES:
                test_subset = df[df[ball_col] == test_val]
                if not test_subset.empty:
                    test_ball_data = test_subset
                    print(f"Found test ball data using value: '{test_val}'")
                    break

            if test_ball_data.empty:
                print("WARNING: No specific 'test' ball data found. Using all data.")
                print(f"Available ball identities: {df[ball_col].value_counts()}")
            else:
                df = test_ball_data
        else:
            print(f"WARNING: Column '{ball_col}' not found in dataset. Available columns: {df.columns.tolist()}")

    print(f"Final dataset size: {len(df)} rows")
    return df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def perform_binary_analysis(df, genotype_col, pretraining_col):
    """Perform statistical analysis on binary metrics"""
    results = {}

    for metric in BINARY_METRICS:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\nAnalyzing binary metric: {metric}")

        # Create contingency tables for each pretraining condition
        for pretrain_val in df[pretraining_col].dropna().unique():
            subset = df[df[pretraining_col] == pretrain_val]

            # Create contingency table
            contingency = pd.crosstab(subset[genotype_col], subset[metric])

            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            # Perform chi-square or Fisher's exact test
            if contingency.min().min() < 5:
                # Use Fisher's exact for small samples
                try:
                    if contingency.shape == (2, 2):
                        _, p_value = fisher_exact(contingency)
                    else:
                        _, p_value, _, _ = chi2_contingency(contingency)
                except:
                    p_value = np.nan
            else:
                _, p_value, _, _ = chi2_contingency(contingency)

            key = f"{metric}_{pretrain_val}"
            results[key] = {
                "metric": metric,
                "pretraining": pretrain_val,
                "p_value": p_value,
                "contingency": contingency,
            }

    return results


def perform_binary_pretraining_effect(df, genotype_col, pretraining_col):
    """Compare pretraining effect within each genotype for binary metrics using Fisher/chi-square."""

    results = {}

    for metric in BINARY_METRICS:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"Binary Pretraining Effect: {get_elegant_metric_name(metric)}")
        print(f"{'='*60}")

        test_results = []

        for genotype in sorted(df[genotype_col].unique()):
            df_geno = df[df[genotype_col] == genotype]

            # Counts: rows = pretraining (naive, pretrained), cols = outcome (0,1)
            counts = pd.crosstab(df_geno[pretraining_col], df_geno[metric])

            # Ensure both pretraining levels present
            if counts.shape[0] != 2:
                print(f"  {genotype}: skipped (missing pretraining level)")
                continue

            # Ensure both outcome levels present; fill missing with zeros
            for val in [0, 1]:
                if val not in counts.columns:
                    counts[val] = 0
            counts = counts[[0, 1]]  # order columns

            table = counts.values

            # Choose test: Fisher for small counts, else chi-square
            use_fisher = table.min() < 5 or table.shape == (2, 2)
            try:
                if use_fisher:
                    odds, p_value = fisher_exact(table)
                    stat_label = "odds_ratio"
                    stat_value = odds
                    method = "Fisher"
                else:
                    chi2, p_value, _, _ = chi2_contingency(table)
                    stat_label = "chi2"
                    stat_value = chi2
                    method = "Chi-square"
            except Exception as e:
                print(f"  {genotype}: test failed ({e})")
                continue

            test_results.append(
                {
                    "genotype": genotype,
                    "p_value": p_value,
                    stat_label: stat_value,
                    "method": method,
                    "counts": counts,
                }
            )

        # FDR correction across genotypes being tested
        if test_results:
            p_values = [r["p_value"] for r in test_results]

            if len(p_values) > 1:
                # Apply FDR correction across all genotypes in this analysis
                _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
                print(f"\n  FDR correction applied across {len(test_results)} genotypes")
            else:
                # No correction needed for single genotype
                p_corrected = p_values
                print(f"\n  No FDR correction (single genotype)")

            for i, r in enumerate(test_results):
                r["p_corrected"] = p_corrected[i]
                p_value = r["p_value"]
                sig = (
                    "***"
                    if p_corrected[i] < 0.001
                    else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else "ns"
                )
                stat_label = "odds_ratio" if "odds_ratio" in r else "chi2"
                print(
                    f"  {r['genotype']}: {stat_label}={r[stat_label]:.3f}, p={p_value:.4f}, p_FDR={p_corrected[i]:.4f} {sig}"
                )

            results[metric] = {"test_results": test_results}

    return results


def check_blocking_factor_balance(df, genotype_col, pretraining_col, blocking_col):
    """
    Test if blocking factor (Date) is balanced across pretraining conditions.
    If confounded, residual permutation approach is critical.
    """
    try:
        # Create contingency table
        contingency = pd.crosstab(df[blocking_col], df[pretraining_col])

        # Chi-square test for independence
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        is_confounded = p_value < 0.05

        print(f"\n🔍 Blocking Factor Balance Test ({blocking_col} × {pretraining_col}):")
        print(f"   χ² = {chi2:.2f}, p-value = {p_value:.4f}")

        if is_confounded:
            print(f"   ⚠️ WARNING: {blocking_col} is CONFOUNDED with {pretraining_col}")
            print(f"      → Residual permutation approach is CRITICAL")
        else:
            print(f"   ✓ {blocking_col} is balanced across {pretraining_col}")
            print(f"      → Both LMM and residual permutation are valid")

        return is_confounded
    except Exception as e:
        print(f"   Could not test balance: {e}")
        return None


def perform_residual_permutation_analysis(df, genotype_col, pretraining_col, metrics=None, n_permutations=10000):
    """
    Permutation test on residuals after removing blocking factors (Date, Arena, Side).

    This approach:
    1. Fits a nuisance-only model with Date, Arena, Side (no treatment effects)
    2. Extracts residuals (blocking-corrected metric)
    3. Performs permutation test on residuals for each genotype
    4. Returns p-values that are:
       - Distribution-free (no normality assumption)
       - Account for blocking factors (Date, Arena, Side)
       - Conservative if blocking is confounded with treatment
    """
    if metrics is None:
        metrics = CONTINUOUS_METRICS

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"Residual Permutation Test for: {get_elegant_metric_name(metric)}")
        print(f"{'='*60}")

        # Prepare data
        fixed_effect_cols = []
        for col in ["Date", "date", "arena", "corridor"]:
            if col in df.columns:
                fixed_effect_cols.append(col)

        columns_needed = [genotype_col, pretraining_col, metric] + fixed_effect_cols
        if "fly" in df.columns:
            columns_needed.append("fly")

        df_clean = df[columns_needed].copy()
        df_clean = df_clean.dropna()

        # Parse arena information
        arena_col = None
        side_col = None
        if "fly" in df_clean.columns:

            def parse_fly_name(fly_name):
                parts = str(fly_name).split("_")
                arena_num = None
                side = None
                for i, part in enumerate(parts):
                    if "arena" in part.lower():
                        arena_num = part.lower().replace("arena", "")
                    if part in ["Left", "Right"]:
                        side = part
                return arena_num, side

            df_clean[["arena_number", "arena_side"]] = df_clean["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))
            if df_clean["arena_number"].notna().any():
                arena_col = "arena_number"
            if df_clean["arena_side"].notna().any():
                side_col = "arena_side"

        # Convert pretraining to numeric
        df_clean["pretraining_numeric"] = df_clean[pretraining_col].apply(
            lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
        )

        # Step 1: Fit nuisance-only model (blocking factors only, no treatment effects)
        print(f"\n📋 Step 1: Fitting nuisance model (blocking factors only)...")

        nuisance_formula_parts = []
        date_col = None
        if "Date" in df_clean.columns:
            date_col = "Date"
        elif "date" in df_clean.columns:
            date_col = "date"

        if date_col:
            nuisance_formula_parts.append(f"C({date_col})")

        if arena_col and side_col:
            nuisance_formula_parts.append(f"C({arena_col}) * C({side_col})")
        elif arena_col:
            nuisance_formula_parts.append(f"C({arena_col})")
        elif side_col:
            nuisance_formula_parts.append(f"C({side_col})")

        if nuisance_formula_parts:
            nuisance_formula = f"{metric} ~ " + " + ".join(nuisance_formula_parts)

            try:
                nuisance_model = smf.ols(nuisance_formula, data=df_clean).fit()
                df_clean["residual"] = nuisance_model.resid
                print(f"   ✓ Nuisance model fitted (R² = {nuisance_model.rsquared:.3f})")
                r2 = nuisance_model.rsquared
                print(f"   Blocking factors explain {r2*100:.1f}% of variance in {metric}")
                # Add interpretation guidance
                if r2 > 0.2:
                    print(f"      → Strong blocking effects - residual correction is important")
                elif r2 > 0.1:
                    print(f"      → Moderate blocking effects - residual correction helpful")
                else:
                    print(f"      → Weak blocking effects - results likely similar to naive permutation")
            except Exception as e:
                print(f"   ⚠️ Could not fit nuisance model: {e}")
                print(f"   Falling back to RAW METRIC (no blocking correction)")
                print(f"   ⚠️ WARNING: P-values may not account for blocking factors!")
                df_clean["residual"] = df_clean[metric]
                df_clean["nuisance_failed"] = True
        else:
            print(f"   No blocking factors available, using raw metric")
            df_clean["residual"] = df_clean[metric]

        # Identify pretraining labels (needed for raw-scale permutation)
        pretrain_vals = sorted(df_clean[pretraining_col].dropna().unique())
        if len(pretrain_vals) != 2:
            print(f"  Skipping metric {metric}: need exactly 2 pretraining levels, found {len(pretrain_vals)}")
            continue
        pretrain_naive, pretrain_trained = pretrain_vals

        # Step 2: Permutation test on residuals for each genotype
        print(f"\n📊 Step 2: Permutation test on residuals ({n_permutations} permutations)...")

        genotype_effects = {}
        test_results = []

        for genotype in sorted(df_clean[genotype_col].unique()):
            df_geno = df_clean[df_clean[genotype_col] == genotype]

            # Get residuals for each pretraining condition
            resid_naive = df_geno[df_geno["pretraining_numeric"] == 0]["residual"].values
            resid_pretrained = df_geno[df_geno["pretraining_numeric"] == 1]["residual"].values

            if len(resid_naive) < 2 or len(resid_pretrained) < 2:
                print(f"  {genotype}: Skipped (insufficient samples)")
                continue

            # Observed difference in RESIDUALS (blocking-corrected)
            observed_diff_resid = np.mean(resid_pretrained) - np.mean(resid_naive)

            # Also calculate on original scale for interpretability
            orig_naive = df_geno[df_geno["pretraining_numeric"] == 0][metric].mean()
            orig_pretrained = df_geno[df_geno["pretraining_numeric"] == 1][metric].mean()
            observed_diff_orig = orig_pretrained - orig_naive

            # Calculate Cohen's d on residuals for effect size (Issue 4)
            pooled_std_resid = np.sqrt(
                (
                    (len(resid_naive) - 1) * np.var(resid_naive, ddof=1)
                    + (len(resid_pretrained) - 1) * np.var(resid_pretrained, ddof=1)
                )
                / (len(resid_naive) + len(resid_pretrained) - 2)
            )
            cohens_d_resid = observed_diff_resid / pooled_std_resid if pooled_std_resid > 0 else 0

            # Calculate Cohen's d on original scale for reference
            pooled_std_orig = np.sqrt(
                (
                    (len(resid_naive) - 1) * np.var(df_geno[df_geno["pretraining_numeric"] == 0][metric].values, ddof=1)
                    + (len(resid_pretrained) - 1)
                    * np.var(df_geno[df_geno["pretraining_numeric"] == 1][metric].values, ddof=1)
                )
                / (len(resid_naive) + len(resid_pretrained) - 2)
            )
            cohens_d_orig = observed_diff_orig / pooled_std_orig if pooled_std_orig > 0 else 0

            # Permutation test on residuals
            combined = np.concatenate([resid_naive, resid_pretrained])
            n1 = len(resid_naive)
            n_total = len(combined)

            perm_diffs = []
            np.random.seed(42 + hash(genotype) % 1000)

            for _ in range(n_permutations):
                perm_idx = np.random.permutation(n_total)
                perm_group1 = combined[perm_idx[:n1]]
                perm_group2 = combined[perm_idx[n1:]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)

            # Two-tailed p-value on residuals
            p_value_resid = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff_resid)) / n_permutations

            # Permutation test directly on original data (for comparison)
            data_orig_naive = df_geno[df_geno[pretraining_col] == pretrain_naive][metric].values
            data_orig_pretrained = df_geno[df_geno[pretraining_col] == pretrain_trained][metric].values
            combined_orig = np.concatenate([data_orig_naive, data_orig_pretrained])
            n1_orig = len(data_orig_naive)
            n_total_orig = len(combined_orig)

            perm_diffs_orig = []
            np.random.seed(99 + hash(genotype) % 1000)
            for _ in range(n_permutations):
                perm_idx = np.random.permutation(n_total_orig)
                perm_group1 = combined_orig[perm_idx[:n1_orig]]
                perm_group2 = combined_orig[perm_idx[n1_orig:]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                perm_diffs_orig.append(perm_diff)

            perm_diffs_orig = np.array(perm_diffs_orig)
            p_value_orig = np.sum(np.abs(perm_diffs_orig) >= np.abs(observed_diff_orig)) / n_permutations

            test_results.append(
                {
                    "genotype": genotype,
                    "observed_diff_residual": observed_diff_resid,
                    "observed_diff_original": observed_diff_orig,
                    "cohens_d_residual": cohens_d_resid,
                    "cohens_d_original": cohens_d_orig,
                    "p_value_residual": p_value_resid,
                    "p_value_original": p_value_orig,
                    "n_naive": len(resid_naive),
                    "n_pretrained": len(resid_pretrained),
                }
            )

        # FDR correction across genotypes being tested (e.g., 3 comparisons if testing 3 genotypes)
        if test_results:
            # Apply FDR correction across all genotypes in this analysis
            p_values = [r["p_value_residual"] for r in test_results]
            if len(p_values) > 1:
                _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
                print(f"\n  FDR correction applied across {len(test_results)} genotypes")
            else:
                p_corrected = p_values  # No correction needed for single genotype
                print(f"\n  No FDR correction (single genotype)")

            for i, result in enumerate(test_results):
                result["p_value_corrected"] = p_corrected[i]
                genotype_effects[result["genotype"]] = result

                sig_resid = (
                    "***"
                    if result["p_value_residual"] < 0.001
                    else (
                        "**"
                        if result["p_value_residual"] < 0.01
                        else "*" if result["p_value_residual"] < 0.05 else "ns"
                    )
                )
                sig_corr = (
                    "***"
                    if result["p_value_corrected"] < 0.001
                    else (
                        "**"
                        if result["p_value_corrected"] < 0.01
                        else "*" if result["p_value_corrected"] < 0.05 else "ns"
                    )
                )
                sig_orig = (
                    "***"
                    if result["p_value_original"] < 0.001
                    else (
                        "**"
                        if result["p_value_original"] < 0.01
                        else "*" if result["p_value_original"] < 0.05 else "ns"
                    )
                )
                print(
                    f"  {result['genotype']}: Δ(resid)={result['observed_diff_residual']:7.3f}, "
                    f"d_resid={result['cohens_d_residual']:6.3f}, p={result['p_value_residual']:.4f} {sig_resid}, "
                    f"p_FDR={result['p_value_corrected']:.4f} {sig_corr} | "
                    f"Δ(orig)={result['observed_diff_original']:7.3f}, d_orig={result['cohens_d_original']:6.3f}, p_orig={result['p_value_original']:.4f} {sig_orig}"
                )

        results[metric] = {
            "test_results": test_results,
            "genotype_effects": genotype_effects,
            "n_permutations": n_permutations,
        }

    return results


def perform_lmm_continuous_analysis(df, genotype_col, pretraining_col, metrics=None):
    """
    Perform Linear Mixed-Effects Model analysis on continuous metrics.

    This approach:
    1. Tests global pretraining effect across all genotypes
    2. Tests interaction: does pretraining effect differ by genotype?
    3. Estimates pretraining effect for each genotype separately
    4. Provides effect sizes and confidence intervals

    Model formula: Metric ~ Genotype * Pretraining + Date + Arena * Side
    Fixed effects: Genotype (reference-coded), Pretraining, interaction, Date, Arena, Side
    Each fly is measured once, so no random effects needed.
    """
    if metrics is None:
        metrics = CONTINUOUS_METRICS

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"LMM Analysis for: {get_elegant_metric_name(metric)}")
        print(f"{'='*60}")

        # Prepare data - collect columns needed for fixed effects
        fixed_effect_cols = []
        for col in ["Date", "date", "arena", "corridor"]:
            if col in df.columns:
                fixed_effect_cols.append(col)

        columns_needed = [genotype_col, pretraining_col, metric] + fixed_effect_cols
        if "fly" in df.columns:
            columns_needed.append("fly")

        df_metric = df[columns_needed].copy()
        df_metric = df_metric.dropna()

        # Parse fly name to extract arena and side information
        # Format: 260114_TNT_F1_Videos_Checked_arena1_Left
        arena_col = None
        side_col = None

        if "fly" in df_metric.columns:

            def parse_fly_name(fly_name):
                """Extract arena number and side from fly name"""
                parts = str(fly_name).split("_")
                arena_num = None
                side = None

                for i, part in enumerate(parts):
                    if "arena" in part.lower():
                        # Extract arena number (e.g., "arena1" -> "1")
                        arena_num = part.lower().replace("arena", "")
                    if part in ["Left", "Right"]:
                        side = part

                return arena_num, side

            df_metric[["arena_number", "arena_side"]] = df_metric["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))

            # Use parsed arena info if available
            if df_metric["arena_number"].notna().any():
                arena_col = "arena_number"
            if df_metric["arena_side"].notna().any():
                side_col = "arena_side"

        # Convert pretraining to numeric (0 = naive, 1 = pretrained)
        df_metric["pretraining_numeric"] = df_metric[pretraining_col].apply(
            lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
        )

        # Create reference-coded genotype (Empty as reference)
        control_genotype = (
            "TNTxEmptySplit"
            if "TNTxEmptySplit" in df_metric[genotype_col].unique()
            else df_metric[genotype_col].iloc[0]
        )
        df_metric["genotype_cat"] = pd.Categorical(
            df_metric[genotype_col],
            categories=[control_genotype]
            + [g for g in sorted(df_metric[genotype_col].unique()) if g != control_genotype],
        )

        try:
            # ============================================================================
            # DATE EFFECT TESTING
            # ============================================================================
            date_has_effect = False
            date_col = None

            if TEST_DATE_EFFECT:
                # Check which date column exists
                if "Date" in df_metric.columns:
                    date_col = "Date"
                elif "date" in df_metric.columns:
                    date_col = "date"

                if date_col:
                    # Test if Date has a significant effect
                    try:
                        date_test_model = smf.ols(f"{metric} ~ C({date_col})", data=df_metric).fit()
                        date_pval = date_test_model.f_pvalue
                        date_has_effect = date_pval < 0.1  # Use 0.1 as threshold for "interesting" effect

                        print(f"\n🗓️ Date Effect Test:")
                        print(f"   F-test p-value: {date_pval:.4f}", end="")
                        if date_has_effect:
                            print(f" ⚠️ Date has effect - consider including")
                        else:
                            print(f" (✓ Date has minimal effect - safe to exclude)")
                    except Exception as e:
                        print(f"\n⚠️ Could not test Date effect: {e}")

            # Build formula with fixed effects: Genotype * Pretraining + Arena * Side
            formula_parts = [f"{metric} ~ C(genotype_cat) * pretraining_numeric"]

            # Optionally add Date if it has significant effect
            if date_has_effect and date_col:
                # Use continuous numeric date instead of categorical to reduce parameters
                df_metric["date_numeric"] = pd.factorize(df_metric[date_col])[0]
                formula_parts.append("date_numeric")
                date_note = "(included as linear trend)"
            else:
                date_note = "(excluded to avoid multicollinearity)"

            # Add Arena * Side interaction to control for experimental setup
            if arena_col and side_col:
                formula_parts.append(f"C({arena_col}) * C({side_col})")
            elif arena_col:
                formula_parts.append(f"C({arena_col})")
            elif side_col:
                formula_parts.append(f"C({side_col})")

            formula = " + ".join(formula_parts)
            print(f"\n📐 Model Formula: {formula}")
            print(f"📊 Fixed Effects: Genotype, Pretraining, Arena × Side")
            print(f"   Date: {date_note}")
            print(f"   (Each fly measured once - no random effects needed)")

            # Fit model using OLS with robust standard errors
            model = smf.ols(formula, data=df_metric).fit(cov_type="HC3")

            # ============================================================================
            # MODEL DIAGNOSTICS
            # ============================================================================
            print(f"\n📊 Model Diagnostics:")

            # Check convergence (OLS always converges, but check if warnings)
            print(f"   ✓ Convergence: OLS (always converges)")

            # Model fit
            print(f"   R²: {model.rsquared:.3f}")
            print(f"   Adj. R²: {model.rsquared_adj:.3f}")
            print(f"   F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4e})")
            print(f"   AIC: {model.aic:.1f}")
            print(f"   BIC: {model.bic:.1f}")
            print(f"   N: {len(df_metric)}")
            print(f"   DoF: {model.df_model} (residual: {model.df_resid})")

            # Check sample size adequacy
            n_params = model.df_model
            n_samples = len(df_metric)
            lmm_adequate, ratio, warning_msg = check_lmm_adequacy(n_samples, n_params)
            print(f"\n   Sample Size Check:")
            if not lmm_adequate:
                print(f"   ⚠️ {warning_msg}")
                if ratio < 5:
                    print(f"      Model may be overfitted. Consider:")
                    print(f"      - Removing Arena×Side interaction if not critical")
                    print(f"      - Pooling small Arena/Side categories if possible")
                    print(f"      - Collecting more data")
                    print(f"      → Using PERMUTATION TEST results for significance annotations")
                else:
                    print(f"      Model may be slightly unstable")
            else:
                print(f"   ✓ Adequate: {ratio:.1f} samples per parameter ({n_params} params, {n_samples} samples)")

            # Store adequacy flag for downstream use
            results[metric]["lmm_adequate"] = lmm_adequate

            # Check for multicollinearity (condition number)
            print(f"\n   Multicollinearity Check:")
            try:
                cond_num = np.linalg.cond(model.model.exog)
                print(f"   Condition Number: {cond_num:.2e}", end="")
                if cond_num > 1e10:
                    print(" (⚠️ SEVERE - too many categorical levels)")
                elif cond_num > 1000:
                    print(" (⚠️ High - model may be unstable)")
                elif cond_num > 100:
                    print(" (⚠️ Moderate)")
                else:
                    print(" (✓ Low)")

            except:
                print(f"   Condition Number: Unable to compute")

            # Calculate Variance Inflation Factors (VIF) for detailed multicollinearity check
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                vif_data = pd.DataFrame()
                vif_data["feature"] = model.model.exog_names
                vif_data["VIF"] = [
                    variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])
                ]

                # Filter out Intercept from VIF check (harmless constant)
                vif_data_check = vif_data[vif_data["feature"] != "Intercept"]
                high_vif = vif_data_check[vif_data_check["VIF"] > 10]
                if not high_vif.empty:
                    print(f"\n   ⚠️ High VIF (>10) detected:")
                    for _, row in high_vif.iterrows():
                        print(f"      {row['feature']}: VIF={row['VIF']:.2f}")
                else:
                    print(f"   ✓ VIF check: All predictors have VIF < 10 (no severe multicollinearity)")
            except Exception as e:
                pass  # VIF calculation optional

            # Check residuals for normality (Jarque-Bera test)
            jb_stat, jb_pval = stats.jarque_bera(model.resid)
            print(f"\n   Normality Check:")
            print(f"   Jarque-Bera test: p={jb_pval:.4f}", end="")
            if jb_pval < 0.05:
                print(" (⚠️ Residuals may not be normally distributed)")
            else:
                print(" (✓ Residuals approximately normal)")

            # Check for heteroscedasticity (residual median test - binomial)
            try:
                from scipy.stats import binomtest

                bp_result = binomtest(sum(model.resid > 0), len(model.resid), 0.5, alternative="two-sided")
                bp_test = bp_result.pvalue
                print(f"   Residual median test: p={bp_test:.4f}")
            except (AttributeError, ImportError):
                # Fallback for older scipy versions
                try:
                    from scipy.stats import binom_test

                    bp_test = binom_test(sum(model.resid > 0), len(model.resid), 0.5, alternative="two-sided")
                    print(f"   Residual median test: p={bp_test:.4f}")
                except:
                    print(f"   Residual median test: Unable to compute (scipy version issue)")

            # Save diagnostic plots if enabled
            if SAVE_DIAGNOSTIC_PLOTS:
                print(f"\n📊 Saving diagnostic plots...")
                save_model_diagnostic_plots(model, metric, OUTPUT_DIR)

            # Extract key results
            # Main effect of pretraining (for reference genotype)
            pretrain_coef = model.params.get("pretraining_numeric", np.nan)
            pretrain_pval = model.pvalues.get("pretraining_numeric", np.nan)
            pretrain_stderr = model.bse.get("pretraining_numeric", np.nan)

            print(f"\n🎯 Pretraining Effect (reference: {control_genotype}):")
            print(f"   Coefficient: {pretrain_coef:.3f} ± {pretrain_stderr:.3f}")
            print(
                f"   p-value: {pretrain_pval:.4f} {'***' if pretrain_pval < 0.001 else '**' if pretrain_pval < 0.01 else '*' if pretrain_pval < 0.05 else 'ns'}"
            )

            # Test for interaction: do genotypes differ in pretraining effect?
            interaction_terms = [k for k in model.params.index if "pretraining_numeric" in k and ":" in k]
            if interaction_terms:
                print(f"\n🔄 Genotype × Pretraining Interactions:")
                for term in interaction_terms:
                    genotype_name = term.split("[T.")[1].split("]")[0] if "[T." in term else term
                    coef = model.params[term]
                    pval = model.pvalues[term]
                    print(
                        f"   {genotype_name}: β={coef:.3f}, p={pval:.4f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}"
                    )

            # Perform separate analysis for each genotype to get individual pretraining effects
            genotype_effects = {}
            print(f"\n📈 Pretraining Effect by Genotype:")

            for genotype in sorted(df_metric[genotype_col].unique()):
                df_geno = df_metric[df_metric[genotype_col] == genotype]

                if len(df_geno) < MIN_SAMPLE_SIZE * 2:  # Need samples in both conditions
                    continue

                # Check if both pretraining conditions exist
                pretrain_vals = df_geno["pretraining_numeric"].unique()
                if len(pretrain_vals) < 2:
                    continue

                # Fit simple model for this genotype
                try:
                    geno_model = smf.ols(f"{metric} ~ pretraining_numeric", data=df_geno).fit(cov_type="HC3")

                    effect_size = geno_model.params["pretraining_numeric"]
                    effect_pval = geno_model.pvalues["pretraining_numeric"]
                    effect_stderr = geno_model.bse["pretraining_numeric"]
                    effect_ci = geno_model.conf_int().loc["pretraining_numeric"].values

                    # Calculate means for naive vs pretrained
                    mean_naive = df_geno[df_geno["pretraining_numeric"] == 0][metric].mean()
                    mean_pretrained = df_geno[df_geno["pretraining_numeric"] == 1][metric].mean()
                    n_naive = len(df_geno[df_geno["pretraining_numeric"] == 0])
                    n_pretrained = len(df_geno[df_geno["pretraining_numeric"] == 1])

                    genotype_effects[genotype] = {
                        "effect_size": effect_size,
                        "stderr": effect_stderr,
                        "pvalue": effect_pval,
                        "ci_lower": effect_ci[0],
                        "ci_upper": effect_ci[1],
                        "mean_naive": mean_naive,
                        "mean_pretrained": mean_pretrained,
                        "n_naive": n_naive,
                        "n_pretrained": n_pretrained,
                        "percent_change": (
                            ((mean_pretrained - mean_naive) / mean_naive * 100) if mean_naive != 0 else np.nan
                        ),
                    }

                    # Format output with control highlighting
                    is_control = genotype in ["TNTxEmptyGal4", "TNTxEmptySplit"]
                    marker = "✓" if is_control else " "
                    sig = (
                        "***"
                        if effect_pval < 0.001
                        else "**" if effect_pval < 0.01 else "*" if effect_pval < 0.05 else "ns"
                    )

                    print(
                        f"   {marker} {genotype:20s}: β={effect_size:7.3f} (SE={effect_stderr:.3f}), p={effect_pval:.4f} {sig:3s}, "
                    )
                    print(
                        f"      {'':23s}  Naive={mean_naive:.2f} (n={n_naive}), Pretrained={mean_pretrained:.2f} (n={n_pretrained}), Δ={genotype_effects[genotype]['percent_change']:+.1f}%"
                    )

                except Exception as e:
                    print(f"   ⚠ {genotype}: Failed to fit model ({str(e)})")

            results[metric] = {
                "global_model": model,
                "pretraining_effect": pretrain_coef,
                "pretraining_pvalue": pretrain_pval,
                "genotype_effects": genotype_effects,
                "interaction_terms": {
                    term: {"coef": model.params[term], "pval": model.pvalues[term]} for term in interaction_terms
                },
            }

        except Exception as e:
            print(f"\n⚠️ Error fitting model for {metric}: {str(e)}")
            results[metric] = {"error": str(e)}

    return results


def perform_permutation_continuous_analysis(df, genotype_col, pretraining_col, metrics=None, n_permutations=10000):
    """
    Perform permutation test analysis on continuous metrics.

    This approach:
    1. Tests difference in means between pretrained vs naive for each genotype
    2. Uses permutation testing (non-parametric, no normality assumption)
    3. Calculates Cohen's d effect size
    4. Applies FDR correction across genotypes for each metric
    """
    if metrics is None:
        metrics = CONTINUOUS_METRICS

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"Permutation Test for: {get_elegant_metric_name(metric)}")
        print(f"{'='*60}")

        # Prepare data
        df_metric = df[[genotype_col, pretraining_col, metric]].copy()
        df_metric = df_metric.dropna()

        # Get unique pretraining values
        pretrain_vals = sorted(df_metric[pretraining_col].unique())

        if len(pretrain_vals) != 2:
            print(f"Warning: Permutation test requires exactly 2 pretraining conditions, found {len(pretrain_vals)}")
            continue

        pretrain_naive = pretrain_vals[0]
        pretrain_trained = pretrain_vals[1]

        print(f"Comparing: {format_pretraining_label(pretrain_naive)} vs {format_pretraining_label(pretrain_trained)}")
        print(f"Number of permutations: {n_permutations}")

        # Per-genotype analysis
        genotype_effects = {}
        test_results = []

        for genotype in sorted(df_metric[genotype_col].unique()):
            df_genotype = df_metric[df_metric[genotype_col] == genotype]

            # Get data for each pretraining condition
            data_naive = df_genotype[df_genotype[pretraining_col] == pretrain_naive][metric].values
            data_pretrained = df_genotype[df_genotype[pretraining_col] == pretrain_trained][metric].values

            if len(data_naive) < MIN_SAMPLE_SIZE or len(data_pretrained) < MIN_SAMPLE_SIZE:
                print(
                    f"  {genotype}: Skipped (insufficient samples: n_naive={len(data_naive)}, n_pretrained={len(data_pretrained)})"
                )
                continue

            # Calculate observed difference in means
            observed_diff = np.mean(data_pretrained) - np.mean(data_naive)

            # Permutation test
            combined_data = np.concatenate([data_naive, data_pretrained])
            n1 = len(data_naive)
            n_total = len(combined_data)

            permuted_diffs = []
            np.random.seed(42 + hash(genotype) % 1000)  # Different seed per genotype

            for i in range(n_permutations):
                perm_indices = np.random.permutation(n_total)
                perm_group1 = combined_data[perm_indices[:n1]]
                perm_group2 = combined_data[perm_indices[n1:]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                permuted_diffs.append(perm_diff)

            permuted_diffs = np.array(permuted_diffs)

            # Two-tailed p-value
            p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(data_naive) - 1) * np.var(data_naive, ddof=1)
                    + (len(data_pretrained) - 1) * np.var(data_pretrained, ddof=1)
                )
                / (len(data_naive) + len(data_pretrained) - 2)
            )
            cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

            test_results.append(
                {
                    "genotype": genotype,
                    "observed_diff": observed_diff,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "mean_naive": np.mean(data_naive),
                    "mean_pretrained": np.mean(data_pretrained),
                    "n_naive": len(data_naive),
                    "n_pretrained": len(data_pretrained),
                }
            )

        # FDR correction across genotypes being tested
        if test_results:
            p_values = [r["p_value"] for r in test_results]

            if len(p_values) > 1:
                # Apply FDR correction across all genotypes in this analysis
                _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
                print(f"\n  FDR correction applied across {len(test_results)} genotypes")
            else:
                # No correction needed for single genotype
                p_corrected = p_values
                print(f"\n  No FDR correction (single genotype)")

            for i, result in enumerate(test_results):
                result["p_corrected"] = p_corrected[i]
                genotype_effects[result["genotype"]] = result

                sig = (
                    "***"
                    if p_corrected[i] < 0.001
                    else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else "ns"
                )
                print(
                    f"  {result['genotype']}: Δ={result['observed_diff']:.3f}, "
                    f"p={result['p_value']:.4f}, p_FDR={p_corrected[i]:.4f} {sig}, d={result['cohens_d']:.3f}"
                )

            # Store results with genotype_effects structure (matching residual permutation format)
            # Use p_value_corrected instead of p_corrected to match expected format
            for genotype, effects in genotype_effects.items():
                effects["p_value_corrected"] = effects["p_corrected"]

            results[metric] = {"genotype_effects": genotype_effects}

    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def get_genotype_color(genotype):
    """Get color for a genotype, with fallback to default"""
    return GENOTYPE_COLORS.get(genotype, "#808080")  # Gray as default


def create_grouped_positions(n_genotypes, n_pretraining):
    """Create x-positions for grouped boxplots"""
    group_width = 0.8
    box_width = group_width / n_pretraining

    positions = []
    for i in range(n_genotypes):
        for j in range(n_pretraining):
            pos = i + (j - (n_pretraining - 1) / 2) * box_width
            positions.append(pos)

    return positions, box_width


def plot_binary_metric_single(
    df, genotype_col, pretraining_col, metric, output_dir, pretrain_stats=None, show_stats=True
):
    """Create a single bar plot for one binary metric with genotype grouping and pretraining hue."""
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in dataset")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get unique pretraining values
    pretrain_values = sorted(df[pretraining_col].dropna().unique())
    n_pretrain = len(pretrain_values)

    # Calculate positions
    positions, bar_width = create_grouped_positions(len(SELECTED_GENOTYPES), n_pretrain)

    # Plot bars for each genotype and pretraining combination
    pos_idx = 0
    xtick_positions = []
    xtick_labels = []

    for i, genotype in enumerate(SELECTED_GENOTYPES):
        xtick_positions.append(i)
        xtick_labels.append(genotype)

        for j, pretrain_val in enumerate(pretrain_values):
            subset = df[(df[genotype_col] == genotype) & (df[pretraining_col] == pretrain_val)]

            if len(subset) == 0:
                pos_idx += 1
                continue

            # Calculate proportion
            prop = subset[metric].sum() / len(subset)
            n = len(subset)

            # Get color and style
            color = get_genotype_color(genotype)
            pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])

            # Plot bar
            bar = ax.bar(
                positions[pos_idx],
                prop,
                width=bar_width * 0.9,
                color=color,
                alpha=pretrain_style["alpha"],
                edgecolor=pretrain_style["edgecolor"],
                linewidth=pretrain_style["linewidth"],
            )

            # Add sample size on top
            ax.text(positions[pos_idx], prop, f"n={n}", ha="center", va="bottom", fontsize=7)

            pos_idx += 1

    # Set x-axis labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.set_title(get_elegant_metric_name(metric), fontsize=14, fontweight="bold")

    # Add second x-axis label for pretraining
    ax.set_xlabel("Genotype", fontsize=12)

    # Add legend
    legend_elements = []
    for pretrain_val in pretrain_values:
        pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])
        label = format_pretraining_label(pretrain_val)
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                alpha=pretrain_style["alpha"],
                edgecolor=pretrain_style["edgecolor"],
                linewidth=pretrain_style["linewidth"],
                label=label,
            )
        )
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    # Add significance annotations
    if show_stats and pretrain_stats and "test_results" in pretrain_stats:
        annotation_height = 1.05  # Just above the y-axis max (which is 1.1)

        for i, genotype in enumerate(SELECTED_GENOTYPES):
            # Find corresponding result
            result = next((r for r in pretrain_stats["test_results"] if r["genotype"] == genotype), None)
            if result and "p_value" in result:
                p_val = result["p_value"]

                # Determine significance marker
                if p_val < 0.001:
                    sig_text = "***"
                elif p_val < 0.01:
                    sig_text = "**"
                elif p_val < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"

                # Add annotation above the genotype group
                ax.text(
                    i,
                    annotation_height,
                    sig_text,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="red" if p_val < 0.05 else "gray",
                )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save plot in multiple formats
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for fmt in PLOT_FORMATS:
        output_path = Path(output_dir) / f"{metric}_binary.{fmt}"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight", format=fmt)
    print(f"Saved: {metric}_binary ({', '.join(PLOT_FORMATS)})")

    return fig


def plot_binary_metrics(
    df, genotype_col, pretraining_col, stats_results, output_dir, pretrain_stats=None, show_stats=True
):
    """Create individual bar plots for each binary metric"""
    print("\nGenerating binary metric plots...")

    for metric in BINARY_METRICS:
        metric_stats = pretrain_stats.get(metric) if pretrain_stats else None
        plot_binary_metric_single(
            df, genotype_col, pretraining_col, metric, output_dir, pretrain_stats=metric_stats, show_stats=show_stats
        )
        plt.close()

    print(f"Completed {len(BINARY_METRICS)} binary metric plots")


def plot_continuous_metric_single(
    df,
    genotype_col,
    pretraining_col,
    metric,
    output_dir,
    lmm_results=None,
    permutation_results=None,
    regular_permutation_results=None,
    show_stats=True,
):
    """Create a single boxplot with jitter for one continuous metric"""
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in dataset")
        return None

    # Convert metric data
    df_plot = df.copy()
    df_plot[f"{metric}_converted"] = convert_metric_data(df_plot[metric], metric)

    # Get adaptive styling parameters based on number of genotypes
    adaptive_params = get_adaptive_styling_params(len(SELECTED_GENOTYPES), n_pretraining=2)
    jitter_amount_adaptive = adaptive_params["jitter_amount"]
    figure_size_adaptive = adaptive_params["figure_size"]
    scatter_size_adaptive = adaptive_params["scatter_size"]

    fig, ax = plt.subplots(figsize=figure_size_adaptive)

    # Get unique pretraining values
    pretrain_values = sorted(df_plot[pretraining_col].dropna().unique())
    n_pretrain = len(pretrain_values)

    # Calculate positions
    positions, box_width = create_grouped_positions(len(SELECTED_GENOTYPES), n_pretrain)

    # Prepare data for plotting and statistical testing
    all_data = []
    all_positions = []
    all_colors = []
    all_styles = []
    genotype_pretrain_data = {}  # Store data for statistical testing

    pos_idx = 0
    xtick_positions = []
    xtick_labels = []

    for i, genotype in enumerate(SELECTED_GENOTYPES):
        xtick_positions.append(i)
        xtick_labels.append(genotype)
        genotype_pretrain_data[genotype] = {}

        for j, pretrain_val in enumerate(pretrain_values):
            subset = df_plot[(df_plot[genotype_col] == genotype) & (df_plot[pretraining_col] == pretrain_val)]

            data = subset[f"{metric}_converted"].dropna()
            genotype_pretrain_data[genotype][pretrain_val] = data.values

            if len(data) > 0:
                all_data.append(data.values)
                all_positions.append(positions[pos_idx])
                all_colors.append(get_genotype_color(genotype))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"]))
            else:
                all_data.append([])
                all_positions.append(positions[pos_idx])
                all_colors.append(get_genotype_color(genotype))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"]))

            pos_idx += 1

    # Perform statistical tests for each genotype (pretrained vs naive)
    lmm_genotype_results = {}
    perm_genotype_results = {}
    regular_perm_genotype_results = {}

    # Get LMM results if available (for effect sizes and diagnostics)
    if lmm_results and metric in lmm_results and "genotype_effects" in lmm_results[metric]:
        lmm_genotype_results = lmm_results[metric]["genotype_effects"]

    # Get residual permutation test results if available (PRIMARY inference method when LMM adequate)
    if permutation_results and metric in permutation_results and "genotype_effects" in permutation_results[metric]:
        perm_genotype_results = permutation_results[metric]["genotype_effects"]

    # Get regular permutation test results (FALLBACK when LMM inadequate)
    if (
        regular_permutation_results
        and metric in regular_permutation_results
        and "genotype_effects" in regular_permutation_results[metric]
    ):
        regular_perm_genotype_results = regular_permutation_results[metric]["genotype_effects"]

    # Determine which statistical method to use for significance annotations
    stat_results = []

    # Check if LMM is adequate for this metric
    # If LMM results are missing or LMM was not run, treat as inadequate so we fall back to regular permutation
    lmm_adequate = True
    if lmm_results is None or metric not in lmm_results:
        # LMM not run or metric missing in LMM output
        lmm_adequate = False
    else:
        lmm_adequate = lmm_results[metric].get("lmm_adequate", True)  # Default to True if not specified

    # Log which path we intend to use (helps debugging)
    # Note: kept minimal to avoid cluttering output

    # Decision tree:
    # 1) If FORCE_REGULAR_PERMUTATION is True, always use regular permutation
    # 2) Prefer regular permutation when LMM is inadequate OR when we are in the small-genotype mode (<=3 genotypes)
    # 3) Otherwise, if LMM is adequate and residual permutation exists, use residual permutation (blocking-aware)
    # 4) Otherwise fall back to whatever permutation results exist

    small_genotype_mode = len(SELECTED_GENOTYPES) <= 3

    if FORCE_REGULAR_PERMUTATION and regular_perm_genotype_results:
        # Forced regular permutation mode (user requested via --force-regular-permutation)
        for genotype in SELECTED_GENOTYPES:
            if genotype in regular_perm_genotype_results:
                stat_results.append(
                    {
                        "genotype": genotype,
                        "p_value_corrected": regular_perm_genotype_results[genotype]["p_value_corrected"],
                        "method": "Regular Permutation",
                    }
                )
    elif (not lmm_adequate or small_genotype_mode) and regular_perm_genotype_results:
        # LMM underpowered or intentionally focusing on few genotypes -> use simpler regular permutation
        for genotype in SELECTED_GENOTYPES:
            if genotype in regular_perm_genotype_results:
                stat_results.append(
                    {
                        "genotype": genotype,
                        "p_value_corrected": regular_perm_genotype_results[genotype]["p_value_corrected"],
                        "method": "Regular Permutation",
                    }
                )
    elif lmm_adequate and perm_genotype_results:
        # LMM is adequate and residual permutation available (accounts for blocking)
        for genotype in SELECTED_GENOTYPES:
            if genotype in perm_genotype_results:
                stat_results.append(
                    {
                        "genotype": genotype,
                        "p_value_corrected": perm_genotype_results[genotype]["p_value_corrected"],
                        "method": "Residual Permutation",
                    }
                )
    elif perm_genotype_results:
        # Fallback to residual permutation if LMM adequacy unknown and regular permutation not available
        for genotype in SELECTED_GENOTYPES:
            if genotype in perm_genotype_results:
                stat_results.append(
                    {
                        "genotype": genotype,
                        "p_value_corrected": perm_genotype_results[genotype]["p_value_corrected"],
                        "method": "Residual Permutation",
                    }
                )
    elif regular_perm_genotype_results:
        # Final fallback to regular permutation if nothing else is available
        for genotype in SELECTED_GENOTYPES:
            if genotype in regular_perm_genotype_results:
                stat_results.append(
                    {
                        "genotype": genotype,
                        "p_value_corrected": regular_perm_genotype_results[genotype]["p_value_corrected"],
                        "method": "Regular Permutation",
                    }
                )

    # Create boxplots
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=box_width * 0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Style boxes with genotype colors and pretraining styles
    for patch, color, style in zip(bp["boxes"], all_colors, all_styles):
        patch.set_facecolor(color)
        patch.set_alpha(style["alpha"])
        patch.set_edgecolor(style["edgecolor"])
        patch.set_linewidth(style["linewidth"])

    # Add jittered scatter points
    for data, pos, color in zip(all_data, all_positions, all_colors):
        if len(data) > 0:
            # Add jitter using adaptive jitter amount
            x_jitter = np.random.normal(pos, jitter_amount_adaptive * box_width, size=len(data))
            ax.scatter(
                x_jitter,
                data,
                alpha=SCATTER_ALPHA,
                s=scatter_size_adaptive,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    # Calculate y-axis limits based on percentiles to handle outliers
    all_values = np.concatenate([d for d in all_data if len(d) > 0])
    if len(all_values) > 0:
        y_low = np.percentile(all_values, YLIM_PERCENTILE_LOW)
        y_high = np.percentile(all_values, YLIM_PERCENTILE_HIGH)
        y_range = y_high - y_low

        # Add margin (extra space at top for significance annotations)
        y_min = y_low - y_range * YLIM_MARGIN
        y_max = y_high + y_range * 0.15  # Increased top margin for annotations

        # Don't go below 0 for metrics that shouldn't be negative
        if y_low >= 0 and y_min < 0:
            y_min = 0

        ax.set_ylim(y_min, y_max)

    # Add significance annotations
    if show_stats and "stat_results" in locals() and stat_results:
        y_max_curr = ax.get_ylim()[1]
        y_range_curr = ax.get_ylim()[1] - ax.get_ylim()[0]
        annotation_height = y_max_curr + y_range_curr * 0.02

        for i, genotype in enumerate(SELECTED_GENOTYPES):
            # Find corresponding result
            result = next((r for r in stat_results if r["genotype"] == genotype), None)
            if result and "p_value_corrected" in result:
                p_val = result["p_value_corrected"]

                # Determine significance marker
                if p_val < 0.001:
                    sig_text = "***"
                elif p_val < 0.01:
                    sig_text = "**"
                elif p_val < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"

                # Add annotation above the genotype group
                ax.text(
                    i,
                    annotation_height,
                    sig_text,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="red" if p_val < 0.05 else "gray",
                )

    # Set x-axis labels with sample sizes
    ax.set_xticks(xtick_positions)

    # Create labels with sample sizes
    labels_with_n = []
    for i, genotype in enumerate(SELECTED_GENOTYPES):
        # Get sample sizes for this genotype
        n_values = []
        for pretrain_val in pretrain_values:
            n = len(genotype_pretrain_data[genotype].get(pretrain_val, []))
            n_values.append(str(n))

        # Format: Genotype\n(n = x, y)
        if len(n_values) == 2:
            label = f"{genotype}\n(n = {n_values[0]}, {n_values[1]})"
        else:
            label = f"{genotype}\n(n = {', '.join(n_values)})"
        labels_with_n.append(label)

    ax.set_xticklabels(labels_with_n, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(format_metric_label(metric), fontsize=12)
    ax.set_xlabel("Genotype", fontsize=12)

    # Add legend for pretraining only
    legend_elements = []
    for pretrain_val in pretrain_values:
        pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])
        label = format_pretraining_label(pretrain_val)
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                alpha=pretrain_style["alpha"],
                edgecolor=pretrain_style["edgecolor"],
                linewidth=pretrain_style["linewidth"],
                label=label,
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout to prevent title overlap with significance annotations
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave 4% space at top for title

    # Save plot in multiple formats
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for fmt in PLOT_FORMATS:
        output_path = Path(output_dir) / f"{metric}_boxplot.{fmt}"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight", format=fmt)
    print(f"Saved: {metric}_boxplot ({', '.join(PLOT_FORMATS)})")

    # Print statistics summary
    if "stat_results" in locals() and stat_results:
        print(f"  Statistics for {metric}:")
        for result in stat_results:
            if "p_value_corrected" in result:
                method = result.get("method", "Unknown")
                sig_marker = ""
                if result["p_value_corrected"] < 0.001:
                    sig_marker = "***"
                elif result["p_value_corrected"] < 0.01:
                    sig_marker = "**"
                elif result["p_value_corrected"] < 0.05:
                    sig_marker = "*"
                print(f"    {result['genotype']}: p={result['p_value_corrected']:.4f} {sig_marker} ({method})")

    return fig


def plot_continuous_metrics(
    df,
    genotype_col,
    pretraining_col,
    stats_results,
    metrics=None,
    output_dir=None,
    lmm_results=None,
    permutation_results=None,
    regular_permutation_results=None,
    show_stats=True,
):
    """Create individual boxplots for each continuous metric"""
    if metrics is None:
        metrics = DEFAULT_CONTINUOUS_METRICS

    # Filter to metrics that exist
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("No valid metrics to plot")
        return

    print(f"\nGenerating {len(metrics)} continuous metric plots...")

    for metric in metrics:
        plot_continuous_metric_single(
            df,
            genotype_col,
            pretraining_col,
            metric,
            output_dir,
            lmm_results=lmm_results,
            permutation_results=permutation_results,
            regular_permutation_results=regular_permutation_results,
            show_stats=show_stats,
        )
        plt.close()

    print(f"Completed {len(metrics)} continuous metric plots")


# ============================================================================
# VALIDATION FUNCTIONS: Residual vs. Original Analysis and Summary Tables
# ============================================================================


def plot_residual_vs_original_effects(lmm_results, residual_perm_results, output_dir):
    """
    Compare effect sizes from raw metric vs. residual-based analysis.
    Should be similar if blocking is balanced.

    This validation plot helps verify that residual permutation approach
    is not artifacts-driven and shows when blocking correction matters.
    """
    if not lmm_results or not residual_perm_results:
        print("⚠️ Insufficient results for method comparison plot")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    n_points = 0
    for metric in lmm_results.keys():
        if metric not in residual_perm_results:
            continue

        # Skip if genotype_effects is not available (happens when LMM is underpowered)
        if "genotype_effects" not in lmm_results[metric] or "genotype_effects" not in residual_perm_results[metric]:
            continue

        for genotype in lmm_results[metric]["genotype_effects"].keys():
            if genotype not in residual_perm_results[metric]["genotype_effects"]:
                continue

            # LMM effect (original scale)
            lmm_effect = lmm_results[metric]["genotype_effects"][genotype]["effect_size"]

            # Residual permutation (original scale for comparison)
            resid_effect = residual_perm_results[metric]["genotype_effects"][genotype]["observed_diff_original"]

            # Color by metric for visibility
            ax.scatter(lmm_effect, resid_effect, alpha=0.7, s=100, label=metric if n_points == 0 else "")
            n_points += 1

    if n_points == 0:
        print("⚠️ No matching results for method comparison plot")
        return

    # Add diagonal line (perfect agreement)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, linewidth=2, label="Perfect agreement")

    ax.set_xlabel("LMM Effect Size (with blocking)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residual Permutation Effect Size (original scale)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Agreement Between Methods\n(Points should cluster near diagonal if blocking is balanced)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    try:
        plt.tight_layout()
        for fmt in PLOT_FORMATS:
            output_path = Path(output_dir) / f"method_comparison.{fmt}"
            plt.savefig(output_path, dpi=150, bbox_inches="tight", format=fmt)
        print(f"✓ Saved method comparison plot ({', '.join(PLOT_FORMATS)})")
    except Exception as e:
        print(f"⚠️ Could not save method comparison plot: {e}")


def generate_summary_table(lmm_results, residual_perm_results, output_file):
    """
    Generate publication-ready summary table with effect sizes and p-values.

    Output format:
    - Metric: Analyzed continuous variable
    - Genotype: Genotype name
    - Effect_Size: Δ from LMM (units of metric)
    - SE: Standard error
    - Cohen_d: Standardized effect size
    - p_value: FDR-corrected p-value from residual permutation
    - n_naive: Sample size (naive/control)
    - n_pretrained: Sample size (pretrained/treatment)
    """
    rows = []

    for metric in lmm_results.keys():
        if metric not in residual_perm_results:
            continue

        # Skip if genotype_effects is not available (happens when LMM is underpowered)
        if "genotype_effects" not in lmm_results[metric] or "genotype_effects" not in residual_perm_results[metric]:
            continue

        for genotype, lmm_effects in lmm_results[metric]["genotype_effects"].items():
            if genotype not in residual_perm_results[metric]["genotype_effects"]:
                continue

            perm = residual_perm_results[metric]["genotype_effects"][genotype]

            row = {
                "Metric": get_elegant_metric_name(metric),
                "Genotype": genotype,
                "Effect_Size": lmm_effects.get("effect_size", np.nan),
                "SE": lmm_effects.get("stderr", np.nan),
                "Cohen_d": perm.get("cohens_d_residual", np.nan),
                "p_value": perm.get("p_value_corrected", np.nan),
                "n_naive": perm.get("n_naive", 0),
                "n_pretrained": perm.get("n_pretrained", 0),
            }
            rows.append(row)

    if not rows:
        print("⚠️ No summary data available for table")
        return

    df_summary = pd.DataFrame(rows)

    try:
        output_file = Path(output_file)
        df_summary.to_csv(output_file, index=False)
        print(f"✓ Saved summary table: {output_file}")
        print(f"  Summary includes {len(df_summary)} genotype×metric combinations")
    except Exception as e:
        print(f"⚠️ Could not save summary table: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="F1 TNT Genotype Comparison Analysis")
    parser.add_argument("--show", action="store_true", help="Display plots instead of just saving")
    parser.add_argument("--metrics", type=str, help="Comma-separated list of specific metrics to plot")
    parser.add_argument(
        "--all-metrics", action="store_true", help="Analyze all auto-detected metrics instead of predefined list"
    )
    parser.add_argument("--detect-genotypes", action="store_true", help="Print all genotypes found in dataset and exit")
    parser.add_argument("--no-stats", action="store_true", help="Generate plots without statistical annotations")
    parser.add_argument(
        "--force-regular-permutation",
        action="store_true",
        help="Force regular permutation tests (skip LMM and residual permutation)",
    )
    args = parser.parse_args()

    # Update global configuration if force-regular-permutation flag is set
    global FORCE_REGULAR_PERMUTATION
    if args.force_regular_permutation:
        FORCE_REGULAR_PERMUTATION = True
        print("\n⚠️  FORCING REGULAR PERMUTATION MODE")
        print("   - LMM analysis will be SKIPPED")
        print("   - Residual permutation will be SKIPPED")
        print("   - Only regular permutation tests will be performed\n")

    # Load dataset
    print("=" * 80)
    print("F1 TNT GENOTYPE COMPARISON ANALYSIS")
    print("=" * 80)

    df = load_dataset("summary")

    # Detect genotype column
    all_genotypes, genotype_col = detect_genotypes(df)
    print(f"\nGenotype column: {genotype_col}")
    print(f"Found {len(all_genotypes)} genotypes: {all_genotypes}")

    if args.detect_genotypes:
        print("\n" + "=" * 80)
        print("DETECTED GENOTYPES (copy to SELECTED_GENOTYPES in script):")
        print("=" * 80)
        print("\nSELECTED_GENOTYPES = [")
        print("    # === CONTROLS ===")
        for g in all_genotypes:
            if g in ["Empty", "EmptySplit", "TNTxPR"]:
                print(f"    '{g}',")
        print("\n    # === EXPERIMENTAL GENOTYPES ===")
        for g in all_genotypes:
            if g not in ["Empty", "EmptySplit", "TNTxPR"]:
                print(f"    '{g}',")
        print("]")
        return

    # Check if genotypes are selected
    if not SELECTED_GENOTYPES or SELECTED_GENOTYPES == ["Empty", "EmptySplit", "TNTxPR"]:
        print("\n" + "!" * 80)
        print("WARNING: SELECTED_GENOTYPES is empty or only contains placeholders!")
        print("!" * 80)
        print("\nRun with --detect-genotypes to see all available genotypes:")
        print("  python f1_tnt_genotype_comparison.py --detect-genotypes")
        binary_pretrain_stats = perform_binary_pretraining_effect(df_filtered, genotype_col, pretraining_col)

        print("\nThen update SELECTED_GENOTYPES in the script.")
        return

    # Detect pretraining column
    pretraining_col = None
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    if pretraining_col is None:
        raise ValueError("Could not find pretraining column in dataset")

    print(f"Pretraining column: {pretraining_col}")

    # Filter dataset
    df_filtered = filter_dataset(df, genotype_col)

    # Auto-detect continuous metrics
    all_continuous_metrics = auto_detect_continuous_metrics(df_filtered)

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"\nGenotypes included: {SELECTED_GENOTYPES}")
    print(f"\nSample sizes by genotype and pretraining:")
    summary = df_filtered.groupby([genotype_col, pretraining_col]).size().unstack(fill_value=0)
    print(summary)

    # Perform statistical analyses
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Check if blocking factors are confounded with pretraining
    print("\n🔍 Checking for confounding between blocking factors and pretraining:")
    for col in ["Date", "date"]:
        if col in df_filtered.columns:
            check_blocking_factor_balance(df_filtered, genotype_col, pretraining_col, col)

    binary_stats = perform_binary_analysis(df_filtered, genotype_col, pretraining_col)
    binary_pretrain_stats = perform_binary_pretraining_effect(df_filtered, genotype_col, pretraining_col)

    # Determine which metrics to analyze
    if args.metrics:
        # User specified specific metrics
        metrics_to_analyze = [m.strip() for m in args.metrics.split(",")]
        print(f"\n📌 Analyzing user-specified metrics: {metrics_to_analyze}")
    elif args.all_metrics:
        # Use all auto-detected metrics
        metrics_to_analyze = all_continuous_metrics
        print(f"\n📌 Analyzing ALL {len(metrics_to_analyze)} auto-detected metrics")
        print(f"   First 10: {', '.join(metrics_to_analyze[:10])}")
        if len(metrics_to_analyze) > 10:
            print(f"   ... and {len(metrics_to_analyze) - 10} more")
    else:
        # Use predefined metrics (matching PCA analysis)
        metrics_to_analyze = DEFAULT_CONTINUOUS_METRICS
        print(f"\n📌 Analyzing {len(metrics_to_analyze)} predefined metrics (matching PCA analysis)")
        print(f"   Metrics: {', '.join(metrics_to_analyze)}")

    # Perform LMM analysis if enabled (for effect sizes and diagnostics)
    lmm_stats = None
    if USE_LMM and not FORCE_REGULAR_PERMUTATION:
        print("\n" + "=" * 80)
        print("LINEAR MIXED-EFFECTS MODEL ANALYSIS")
        print("=" * 80)
        lmm_stats = perform_lmm_continuous_analysis(
            df_filtered, genotype_col, pretraining_col, metrics=metrics_to_analyze
        )

    # Check blocking factor balance early in analysis
    print("\n" + "=" * 80)
    print("BLOCKING FACTOR BALANCE ANALYSIS")
    print("=" * 80)
    for date_col in df_filtered.columns:
        if "date" in date_col.lower() or date_col == "Date":
            check_blocking_factor_balance(df_filtered, genotype_col, pretraining_col, date_col)

    # Perform residual permutation test analysis if enabled (PRIMARY METHOD)
    permutation_stats = None
    if USE_RESIDUAL_PERMUTATION and not FORCE_REGULAR_PERMUTATION:
        print("\n" + "=" * 80)
        print("RESIDUAL PERMUTATION TEST ANALYSIS (PRIMARY METHOD)")
        print("Distribution-free inference accounting for blocking factors (Date, Arena, Side)")
        print("=" * 80)
        permutation_stats = perform_residual_permutation_analysis(
            df_filtered, genotype_col, pretraining_col, metrics=metrics_to_analyze, n_permutations=N_PERMUTATIONS
        )

    # Perform regular permutation test analysis (fallback when LMM is inadequate)
    regular_permutation_stats = None
    print("\n" + "=" * 80)
    print("REGULAR PERMUTATION TEST ANALYSIS (FALLBACK METHOD)")
    print("Distribution-free inference on original data (no blocking correction)")
    print("=" * 80)
    regular_permutation_stats = perform_permutation_continuous_analysis(
        df_filtered, genotype_col, pretraining_col, metrics=metrics_to_analyze, n_permutations=N_PERMUTATIONS
    )

    # Create plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_binary_metrics(
        df_filtered,
        genotype_col,
        pretraining_col,
        binary_stats,
        output_dir,
        pretrain_stats=binary_pretrain_stats,
        show_stats=not args.no_stats,
    )
    plot_continuous_metrics(
        df_filtered,
        genotype_col,
        pretraining_col,
        None,  # No longer using Mann-Whitney (continuous_stats)
        metrics=metrics_to_analyze,
        output_dir=output_dir,
        lmm_results=lmm_stats,
        permutation_results=permutation_stats,
        regular_permutation_results=regular_permutation_stats,
        show_stats=not args.no_stats,
    )

    if args.show:
        plt.show()
    else:
        plt.close("all")

    # Generate validation plots and summary tables
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION OUTPUTS")
    print("=" * 80)

    # Plot comparison between methods
    if lmm_stats and permutation_stats:
        plot_residual_vs_original_effects(lmm_stats, permutation_stats, output_dir)

    # Generate summary statistics table for publication
    if lmm_stats and permutation_stats:
        summary_output = output_dir / "summary_statistics.csv"
        generate_summary_table(lmm_stats, permutation_stats, summary_output)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
