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
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle
from itertools import combinations
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# DATASET PATHS - Update these if datasets change
# ============================================================================
DATASET_PATHS = {
    'summary': '/mnt/upramdya_data/MD/F1_Tracks/Datasets/260102_20_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather',
    'coordinates': '/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Datasets/260102_14_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather',
    'fly_positions': '/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Datasets/260102_14_F1_coordinates_F1_TNT_Full_Data/fly_positions/pooled_fly_positions.feather',
}

OUTPUT_DIR = '/mnt/upramdya_data/MD/F1_Tracks/F1_TNT'

# ============================================================================
# GENOTYPE SELECTION - Comment out genotypes to exclude from analysis
# ============================================================================
# NOTE: This list will be auto-populated when you run the script for the first time.
# After that, you can comment out genotypes you want to exclude.

SELECTED_GENOTYPES = [
    # === CONTROLS ===
    'TNTxEmptyGal4',
    'TNTxEmptySplit',
    #'TNTxPR',

    # === EXPERIMENTAL GENOTYPES (paired with internal controls) ===
    'TNTxDDC',
    'TNTxTH',
    #'PRxTH',
    'TNTxMB247',
    #'PRxMB247',
    'TNTxLC10-2',
    'TNTxLC16-1',
    #'PRxLC16-1',
]

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Binary metrics to analyze
BINARY_METRICS = ["has_finished", "has_long_pauses", "has_major", "has_significant"]

# Continuous metrics - will be auto-detected from dataset
# NOTE: Ball proximity metrics (ball_proximity_proportion_XXpx) are now available
#       These measure the proportion of time flies spend near the ball at different distances
# This list is used as a fallback if auto-detection fails
DEFAULT_CONTINUOUS_METRICS = [
    "interaction_rate",
    "interaction_duration",
    "total_interaction_time",
    "mean_interaction_duration",
    "head_ball_distance_during_interaction",
    "head_ball_distance_outside_interaction",
    "max_distance_from_ball",
    "fly_distance_moved",
    "time_until_first_interaction",
    "chamber_exit_time",
    # Ball proximity metrics (proportion of time spent near ball after initial movement)
    "ball_proximity_proportion_70px",
    "ball_proximity_proportion_140px",
    "ball_proximity_proportion_200px",
]

# Columns to exclude from auto-detection
EXCLUDED_COLUMNS = {
    'fly', 'Date', 'date', 'filename', 'video', 'path', 'folder',
    'Genotype', 'genotype', 'Pretraining', 'pretraining',
    'ball_condition', 'ball_identity', 'arena', 'corridor',
    'time', 'frame', 'index'
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
    'TNTxEmptyGal4': '#7f7f7f',      # Gray
    'TNTxEmptySplit': '#7f7f7f',     # Yellow-green
    'TNTxPR': '#7f7f7f',             # Cyan

    # Experimental genotypes
    'TNTxDDC': '#8B4513',            # Brown (MB extrinsic)
    'TNTxTH': '#8B4513',             # Brown (MB extrinsic)
    'PRxTH': '#8B4513',              # Brown (MB extrinsic)
    'TNTxMB247': '#1f77b4',          # Blue (MB)
    'PRxMB247': '#1f77b4',           # Blue (MB)
    'TNTxLC10-2': '#ff7f0e',         # Orange (Vision)
    'TNTxLC16-1': '#ff7f0e',         # Orange (Vision)
    'PRxLC16-1': '#ff7f0e',          # Orange (Vision)
}

# Pretraining styles - Controls visual distinction for pretrained vs naive
PRETRAINING_STYLES = {
    'n': {  # Naive (no pretraining)
        'alpha': 0.6,
        'edgecolor': 'black',
        'linewidth': 1.5,
        'label_suffix': ' (Naive)'
    },
    'y': {  # Pretrained
        'alpha': 1.0,
        'edgecolor': 'black',
        'linewidth': 2.5,
        'label_suffix': ' (Pretrained)'
    }
}

# Plot configuration
FIGURE_SIZE = (14, 8)  # Width, Height for individual metric plots
DPI = 300
PLOT_FORMAT = "png"
JITTER_AMOUNT = 0.15  # Amount of jitter for scatter points
SCATTER_SIZE = 30
SCATTER_ALPHA = 0.5

# Y-axis limits for handling outliers
YLIM_PERCENTILE_LOW = 0    # Lower percentile for y-axis (0 = minimum)
YLIM_PERCENTILE_HIGH = 99  # Upper percentile for y-axis (99 = 99th percentile)
YLIM_MARGIN = 0.05         # Add 5% margin above/below the percentile limits

# Statistical testing
ALPHA = 0.05
MIN_SAMPLE_SIZE = 2


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_distance_metric(metric_name):
    """Check if a metric represents distance (in pixels)"""
    distance_keywords = [
        "distance", "dist", "head_ball", "fly_distance_moved",
        "max_distance", "distance_moved", "distance_ratio",
    ]
    return any(keyword in metric_name.lower() for keyword in distance_keywords)


def is_time_metric(metric_name):
    """Check if a metric represents time (in seconds)"""
    time_keywords = ["time", "duration", "pause", "stop", "freeze",
                     "chamber_exit_time", "time_chamber_beginning"]
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
            if not any(keyword in col.lower() for keyword in ['id', 'index', 'frame']):
                continuous_metrics.append(col)

    print(f"\n📊 Auto-detected {len(continuous_metrics)} continuous metrics:")
    if len(continuous_metrics) > 0:
        print(f"   {', '.join(continuous_metrics[:10])}")
        if len(continuous_metrics) > 10:
            print(f"   ... and {len(continuous_metrics) - 10} more")

    return continuous_metrics if continuous_metrics else DEFAULT_CONTINUOUS_METRICS


def load_dataset(dataset_type='summary'):
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
        if 'genotype' in col.lower():
            genotype_col = col
            break

    if genotype_col is None:
        raise ValueError("Could not find genotype column in dataset")

    genotypes = df[genotype_col].dropna().unique().tolist()
    genotypes = sorted(genotypes)

    # Separate controls and experimental genotypes
    controls = ['Empty', 'EmptySplit', 'TNTxPR']
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
        ball_col = 'ball_identity'

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
                'metric': metric,
                'pretraining': pretrain_val,
                'p_value': p_value,
                'contingency': contingency
            }

    return results


def perform_continuous_analysis(df, genotype_col, pretraining_col, metrics=None):
    """Perform statistical analysis on continuous metrics"""
    if metrics is None:
        metrics = CONTINUOUS_METRICS

    results = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataset")
            continue

        print(f"\nAnalyzing continuous metric: {metric}")

        # Analyze for each pretraining condition
        for pretrain_val in df[pretraining_col].dropna().unique():
            subset = df[df[pretraining_col] == pretrain_val].copy()

            # Convert metric data
            subset[f'{metric}_converted'] = convert_metric_data(subset[metric], metric)

            # Get data by genotype
            genotype_data = {}
            for genotype in SELECTED_GENOTYPES:
                data = subset[subset[genotype_col] == genotype][f'{metric}_converted'].dropna()
                if len(data) >= MIN_SAMPLE_SIZE:
                    genotype_data[genotype] = data

            if len(genotype_data) < 2:
                continue

            # Perform Kruskal-Wallis test
            groups = list(genotype_data.values())
            if len(groups) >= 2:
                try:
                    stat, p_value = kruskal(*groups)
                except:
                    p_value = np.nan
                    stat = np.nan
            else:
                p_value = np.nan
                stat = np.nan

            # Perform pairwise comparisons
            pairwise_results = []
            genotype_list = list(genotype_data.keys())
            for g1, g2 in combinations(genotype_list, 2):
                try:
                    _, p = mannwhitneyu(genotype_data[g1], genotype_data[g2], alternative='two-sided')
                    pairwise_results.append({
                        'genotype1': g1,
                        'genotype2': g2,
                        'p_value': p
                    })
                except:
                    pass

            # Apply multiple testing correction
            if pairwise_results:
                p_values = [r['p_value'] for r in pairwise_results]
                _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method='fdr_bh')
                for i, result in enumerate(pairwise_results):
                    result['p_corrected'] = p_corrected[i]

            key = f"{metric}_{pretrain_val}"
            results[key] = {
                'metric': metric,
                'pretraining': pretrain_val,
                'kruskal_p': p_value,
                'kruskal_stat': stat,
                'genotype_data': genotype_data,
                'pairwise': pairwise_results
            }

    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def get_genotype_color(genotype):
    """Get color for a genotype, with fallback to default"""
    return GENOTYPE_COLORS.get(genotype, '#808080')  # Gray as default


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


def plot_binary_metric_single(df, genotype_col, pretraining_col, metric, output_dir):
    """Create a single bar plot for one binary metric with genotype grouping and pretraining hue"""
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
            pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])

            # Plot bar
            bar = ax.bar(positions[pos_idx], prop, width=bar_width * 0.9,
                        color=color, alpha=pretrain_style['alpha'],
                        edgecolor=pretrain_style['edgecolor'],
                        linewidth=pretrain_style['linewidth'])

            # Add sample size on top
            ax.text(positions[pos_idx], prop, f'n={n}',
                   ha='center', va='bottom', fontsize=7)

            pos_idx += 1

    # Set x-axis labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.set_title(get_elegant_metric_name(metric), fontsize=14, fontweight='bold')

    # Add second x-axis label for pretraining
    ax.set_xlabel('Genotype', fontsize=12)

    # Add legend
    legend_elements = []
    for pretrain_val in pretrain_values:
        pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])
        label = format_pretraining_label(pretrain_val)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                            facecolor='gray', alpha=pretrain_style['alpha'],
                                            edgecolor=pretrain_style['edgecolor'],
                                            linewidth=pretrain_style['linewidth'],
                                            label=label))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / f'{metric}_binary.{PLOT_FORMAT}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path.name}")

    return fig


def plot_binary_metrics(df, genotype_col, pretraining_col, stats_results, output_dir):
    """Create individual bar plots for each binary metric"""
    print("\nGenerating binary metric plots...")

    for metric in BINARY_METRICS:
        plot_binary_metric_single(df, genotype_col, pretraining_col, metric, output_dir)
        plt.close()

    print(f"Completed {len(BINARY_METRICS)} binary metric plots")


def plot_continuous_metric_single(df, genotype_col, pretraining_col, metric, output_dir):
    """Create a single boxplot with jitter for one continuous metric"""
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in dataset")
        return None

    # Convert metric data
    df_plot = df.copy()
    df_plot[f'{metric}_converted'] = convert_metric_data(df_plot[metric], metric)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

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
            subset = df_plot[(df_plot[genotype_col] == genotype) &
                            (df_plot[pretraining_col] == pretrain_val)]

            data = subset[f'{metric}_converted'].dropna()
            genotype_pretrain_data[genotype][pretrain_val] = data.values

            if len(data) > 0:
                all_data.append(data.values)
                all_positions.append(positions[pos_idx])
                all_colors.append(get_genotype_color(genotype))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n']))
            else:
                all_data.append([])
                all_positions.append(positions[pos_idx])
                all_colors.append(get_genotype_color(genotype))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n']))

            pos_idx += 1

    # Perform Mann-Whitney tests for each genotype (pretrained vs naive)
    mw_results = []
    for genotype in SELECTED_GENOTYPES:
        if len(pretrain_values) == 2:
            data_group1 = genotype_pretrain_data[genotype].get(pretrain_values[0], [])
            data_group2 = genotype_pretrain_data[genotype].get(pretrain_values[1], [])

            if len(data_group1) >= MIN_SAMPLE_SIZE and len(data_group2) >= MIN_SAMPLE_SIZE:
                try:
                    stat, p_val = mannwhitneyu(data_group1, data_group2, alternative='two-sided')
                    mw_results.append({
                        'genotype': genotype,
                        'p_value': p_val,
                        'n1': len(data_group1),
                        'n2': len(data_group2)
                    })
                except:
                    pass

    # Apply FDR correction to p-values
    if mw_results:
        p_values = [r['p_value'] for r in mw_results]
        _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method='fdr_bh')
        for i, result in enumerate(mw_results):
            result['p_corrected'] = p_corrected[i]

    # Create boxplots
    bp = ax.boxplot(all_data, positions=all_positions, widths=box_width * 0.6,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))

    # Style boxes with genotype colors and pretraining styles
    for patch, color, style in zip(bp['boxes'], all_colors, all_styles):
        patch.set_facecolor(color)
        patch.set_alpha(style['alpha'])
        patch.set_edgecolor(style['edgecolor'])
        patch.set_linewidth(style['linewidth'])

    # Add jittered scatter points
    for data, pos, color in zip(all_data, all_positions, all_colors):
        if len(data) > 0:
            # Add jitter
            x_jitter = np.random.normal(pos, JITTER_AMOUNT * box_width, size=len(data))
            ax.scatter(x_jitter, data, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                      color=color, edgecolors='black', linewidths=0.5, zorder=3)

    # Calculate y-axis limits based on percentiles to handle outliers
    all_values = np.concatenate([d for d in all_data if len(d) > 0])
    if len(all_values) > 0:
        y_low = np.percentile(all_values, YLIM_PERCENTILE_LOW)
        y_high = np.percentile(all_values, YLIM_PERCENTILE_HIGH)
        y_range = y_high - y_low

        # Add margin
        y_min = y_low - y_range * YLIM_MARGIN
        y_max = y_high + y_range * YLIM_MARGIN

        # Don't go below 0 for metrics that shouldn't be negative
        if y_low >= 0 and y_min < 0:
            y_min = 0

        ax.set_ylim(y_min, y_max)

        # Count outliers
        n_outliers = np.sum((all_values < y_min) | (all_values > y_max))
        if n_outliers > 0:
            ax.text(0.02, 0.98, f'{n_outliers} outliers not shown',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add significance annotations
    if mw_results:
        y_max_curr = ax.get_ylim()[1]
        y_range_curr = ax.get_ylim()[1] - ax.get_ylim()[0]
        annotation_height = y_max_curr + y_range_curr * 0.02

        for i, genotype in enumerate(SELECTED_GENOTYPES):
            # Find the corresponding result
            result = next((r for r in mw_results if r['genotype'] == genotype), None)
            if result:
                p_corr = result['p_corrected']

                # Determine significance level
                if p_corr < 0.001:
                    sig_text = '***'
                elif p_corr < 0.01:
                    sig_text = '**'
                elif p_corr < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'

                # Add text above the genotype
                ax.text(i, annotation_height, sig_text,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

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

    ax.set_xticklabels(labels_with_n, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(format_metric_label(metric), fontsize=12)
    ax.set_title(get_elegant_metric_name(metric), fontsize=14, fontweight='bold')
    ax.set_xlabel('Genotype', fontsize=12)

    # Add legend for pretraining
    legend_elements = []
    for pretrain_val in pretrain_values:
        pretrain_style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])
        label = format_pretraining_label(pretrain_val)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                            facecolor='gray', alpha=pretrain_style['alpha'],
                                            edgecolor=pretrain_style['edgecolor'],
                                            linewidth=pretrain_style['linewidth'],
                                            label=label))

    # Add significance legend
    legend_elements.append(plt.Line2D([0], [0], color='none', label=''))  # Spacer
    legend_elements.append(plt.Line2D([0], [0], color='none', marker='$***$',
                                     markersize=10, label='p < 0.001 (FDR)'))
    legend_elements.append(plt.Line2D([0], [0], color='none', marker='$**$',
                                     markersize=10, label='p < 0.01 (FDR)'))
    legend_elements.append(plt.Line2D([0], [0], color='none', marker='$*$',
                                     markersize=10, label='p < 0.05 (FDR)'))
    legend_elements.append(plt.Line2D([0], [0], color='none', marker='$ns$',
                                     markersize=10, label='not significant'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / f'{metric}_boxplot.{PLOT_FORMAT}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path.name}")

    # Print statistics summary
    if mw_results:
        print(f"  Statistics for {metric}:")
        for result in mw_results:
            sig_marker = ''
            if result['p_corrected'] < 0.001:
                sig_marker = '***'
            elif result['p_corrected'] < 0.01:
                sig_marker = '**'
            elif result['p_corrected'] < 0.05:
                sig_marker = '*'
            print(f"    {result['genotype']}: p={result['p_corrected']:.4f} {sig_marker} "
                  f"(n={result['n1']}, {result['n2']})")

    return fig


def plot_continuous_metrics(df, genotype_col, pretraining_col, stats_results,
                           metrics=None, output_dir=None):
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
        plot_continuous_metric_single(df, genotype_col, pretraining_col, metric, output_dir)
        plt.close()

    print(f"Completed {len(metrics)} continuous metric plots")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='F1 TNT Genotype Comparison Analysis')
    parser.add_argument('--show', action='store_true', help='Display plots instead of just saving')
    parser.add_argument('--metrics', type=str, help='Comma-separated list of specific metrics to plot')
    parser.add_argument('--detect-genotypes', action='store_true',
                       help='Print all genotypes found in dataset and exit')
    args = parser.parse_args()

    # Load dataset
    print("="*80)
    print("F1 TNT GENOTYPE COMPARISON ANALYSIS")
    print("="*80)

    df = load_dataset('summary')

    # Detect genotype column
    all_genotypes, genotype_col = detect_genotypes(df)
    print(f"\nGenotype column: {genotype_col}")
    print(f"Found {len(all_genotypes)} genotypes: {all_genotypes}")

    if args.detect_genotypes:
        print("\n" + "="*80)
        print("DETECTED GENOTYPES (copy to SELECTED_GENOTYPES in script):")
        print("="*80)
        print("\nSELECTED_GENOTYPES = [")
        print("    # === CONTROLS ===")
        for g in all_genotypes:
            if g in ['Empty', 'EmptySplit', 'TNTxPR']:
                print(f"    '{g}',")
        print("\n    # === EXPERIMENTAL GENOTYPES ===")
        for g in all_genotypes:
            if g not in ['Empty', 'EmptySplit', 'TNTxPR']:
                print(f"    '{g}',")
        print("]")
        return

    # Check if genotypes are selected
    if not SELECTED_GENOTYPES or SELECTED_GENOTYPES == ['Empty', 'EmptySplit', 'TNTxPR']:
        print("\n" + "!"*80)
        print("WARNING: SELECTED_GENOTYPES is empty or only contains placeholders!")
        print("!"*80)
        print("\nRun with --detect-genotypes to see all available genotypes:")
        print("  python f1_tnt_genotype_comparison.py --detect-genotypes")
        print("\nThen update SELECTED_GENOTYPES in the script.")
        return

    # Detect pretraining column
    pretraining_col = None
    for col in df.columns:
        if 'pretrain' in col.lower():
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
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"\nGenotypes included: {SELECTED_GENOTYPES}")
    print(f"\nSample sizes by genotype and pretraining:")
    summary = df_filtered.groupby([genotype_col, pretraining_col]).size().unstack(fill_value=0)
    print(summary)

    # Perform statistical analyses
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    binary_stats = perform_binary_analysis(df_filtered, genotype_col, pretraining_col)

    # Determine which metrics to analyze
    if args.metrics:
        # User specified specific metrics
        metrics_to_analyze = [m.strip() for m in args.metrics.split(',')]
        print(f"\n📌 Analyzing user-specified metrics: {metrics_to_analyze}")
    else:
        # Use all auto-detected metrics
        metrics_to_analyze = all_continuous_metrics
        print(f"\n📌 Analyzing all {len(metrics_to_analyze)} auto-detected metrics")

    continuous_stats = perform_continuous_analysis(df_filtered, genotype_col, pretraining_col,
                                                   metrics=metrics_to_analyze)

    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_binary_metrics(df_filtered, genotype_col, pretraining_col, binary_stats, output_dir)
    plot_continuous_metrics(df_filtered, genotype_col, pretraining_col, continuous_stats,
                          metrics=metrics_to_analyze, output_dir=output_dir)

    if args.show:
        plt.show()
    else:
        plt.close('all')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
