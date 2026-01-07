#!/usr/bin/env python3
"""
F1 PCA Analysis Script

Performs Principal Component Analysis on selected behavioral metrics from F1 experiments
to identify underlying behavioral patterns and detect pretraining effects.

Selected metrics for PCA:
- ball_proximity_proportion_200px: Time spent near ball
- distance_moved: Total ball distance moved
- max_distance: Maximum ball distance achieved
- nb_events: Number of interaction events
- has_major: Binary - achieved major event
- has_finished: Binary - completed task
- significant_ratio: Proportion of significant events

Usage:
    python f1_pca_analysis.py
    python f1_pca_analysis.py --show
    python f1_pca_analysis.py --n-components 3
    python f1_pca_analysis.py --detect-genotypes  # Show available genotypes
    python f1_pca_analysis.py --metrics "ball_proximity_proportion_200px,distance_moved"
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis mode - 'control' or 'tnt_full'
ANALYSIS_MODE = 'tnt_full'  # Change to 'control' for F1 New control experiment

# Dataset paths by mode
DATASET_PATHS = {
    'control': '/mnt/upramdya_data/MD/F1_Tracks/Datasets/260104_12_F1_coordinates_F1_New_Data/summary/pooled_summary.feather',
    'tnt_full': '/mnt/upramdya_data/MD/F1_Tracks/Datasets/260102_20_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather',
}

OUTPUT_DIRS = {
    'control': '/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New/PCA',
    'tnt_full': '/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/PCA/Simplified',
}

# Create output directory if it doesn't exist
Path(OUTPUT_DIRS[ANALYSIS_MODE]).mkdir(parents=True, exist_ok=True)

# Mode-specific grouping configuration
GROUPING_CONFIG = {
    'control': {
        'secondary_var': 'f1_condition',
        'secondary_label': 'F1 Condition',
        'selected_values': 'SELECTED_F1_CONDITIONS',
    },
    'tnt_full': {
        'secondary_var': 'genotype',
        'secondary_label': 'Genotype',
        'selected_values': 'SELECTED_GENOTYPES',
    },
}

# ============================================================================
# GENOTYPE SELECTION - Comment out genotypes to exclude from analysis
# For 'control' mode: No genotype filtering (single genotype)
# For 'tnt_full' mode: Select which TNT genotypes to include
# ============================================================================

SELECTED_GENOTYPES = [
    # === CONTROLS ===
    'TNTxEmptyGal4',
    'TNTxEmptySplit',
    #'TNTxPR',

    # === EXPERIMENTAL GENOTYPES ===
    'TNTxDDC',
    'TNTxTH',
    #'PRxTH',
    'TNTxMB247',
    #'PRxMB247',
    'TNTxLC10-2',
    'TNTxLC16-1',
    #'PRxLC16-1',
]

# F1 conditions for control mode
SELECTED_F1_CONDITIONS = [
    'control',
    'pretrained',
    #'pretrained_unlocked',  # Uncomment to include
]

# Metrics for PCA - these should be present in the summary dataset
PCA_METRICS = [
    'ball_proximity_proportion_200px',  # Time spent near ball
    'distance_moved',                    # Total ball distance moved
    'max_distance',                      # Maximum ball distance
    'nb_events',                         # Number of interaction events
    'has_major',                         # Binary: achieved major event
    'has_finished',                      # Binary: completed task
    'significant_ratio',                 # Proportion of significant events
]

# Elegant names for display
METRIC_DISPLAY_NAMES = {
    'ball_proximity_proportion_200px': 'Ball Proximity',
    'distance_moved': 'Distance Moved',
    'max_distance': 'Max Distance',
    'nb_events': 'N Events',
    'has_major': 'Has Major',
    'has_finished': 'Finished Task',
    'significant_ratio': 'Significant Ratio',
}

# Pixel to mm conversion
PIXELS_PER_MM = 500 / 30

# Filter for test ball only
FILTER_TEST_BALL = True
TEST_BALL_VALUES = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]

# Plot styling
FIGURE_SIZE_SINGLE = (14, 8)  # Increased width for better boxplot visibility
FIGURE_SIZE_MULTI = (14, 10)
FIGURE_SIZE_LOADING = (12, 8)
DPI = 300
PLOT_FORMAT = "png"
JITTER_AMOUNT = 0.15
SCATTER_SIZE = 40
SCATTER_ALPHA = 0.6

# Colors for genotypes (from f1_tnt_genotype_comparison.py)
GENOTYPE_COLORS = {
    'TNTxEmptyGal4': '#7f7f7f',
    'TNTxEmptySplit': '#7f7f7f',
    'TNTxPR': '#7f7f7f',
    'TNTxDDC': '#8B4513',
    'TNTxTH': '#8B4513',
    'PRxTH': '#8B4513',
    'TNTxMB247': '#1f77b4',
    'PRxMB247': '#1f77b4',
    'TNTxLC10-2': '#ff7f0e',
    'TNTxLC16-1': '#ff7f0e',
    'PRxLC16-1': '#ff7f0e',
}

# Colors for F1 conditions (control mode)
F1_CONDITION_COLORS = {
    'control': 'steelblue',
    'pretrained': 'orange',
    'pretrained_unlocked': 'lightgreen',
}

# Pretraining styles
PRETRAINING_STYLES = {
    'n': {
        'alpha': 0.6,
        'edgecolor': 'black',
        'linewidth': 1.5,
        'label': 'Naive',
        'marker': 'o',
    },
    'y': {
        'alpha': 1.0,
        'edgecolor': 'black',
        'linewidth': 2.5,
        'label': 'Pretrained',
        'marker': 's',
    }
}

ALPHA = 0.05


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_dataset(mode):
    """Load dataset from feather file based on mode"""
    dataset_path = DATASET_PATHS.get(mode)
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found for mode '{mode}': {dataset_path}")

    print(f"Loading dataset for mode '{mode}' from: {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"  Loaded {len(df)} rows")
    return df


def detect_columns(df, mode):
    """Detect genotype/f1_condition and pretraining columns based on mode"""
    secondary_col = None
    pretraining_col = None

    # Get expected secondary variable from config
    secondary_var = GROUPING_CONFIG[mode]['secondary_var']

    for col in df.columns:
        # Look for secondary grouping variable (genotype or f1_condition)
        if secondary_var == 'genotype' and 'genotype' in col.lower() and secondary_col is None:
            secondary_col = col
        elif secondary_var == 'f1_condition' and 'f1' in col.lower() and 'condition' in col.lower() and secondary_col is None:
            secondary_col = col

        # Look for pretraining column
        if 'pretrain' in col.lower() and pretraining_col is None:
            pretraining_col = col

    if secondary_col is None:
        raise ValueError(f"Could not find {secondary_var} column for mode '{mode}'")
    if pretraining_col is None:
        raise ValueError("Could not find pretraining column")

    secondary_label = GROUPING_CONFIG[mode]['secondary_label']
    print(f"  {secondary_label} column: {secondary_col}")
    print(f"  Pretraining column: {pretraining_col}")

    return secondary_col, pretraining_col


def filter_dataset(df, secondary_col, mode):
    """Filter dataset for selected genotypes/conditions and test ball"""
    secondary_var = GROUPING_CONFIG[mode]['secondary_var']

    # Filter based on mode
    if mode == 'tnt_full':
        # Filter for selected genotypes
        if SELECTED_GENOTYPES:
            df = df[df[secondary_col].isin(SELECTED_GENOTYPES)].copy()
            print(f"  Filtered for {len(SELECTED_GENOTYPES)} selected genotypes")
            print(f"  Genotypes: {SELECTED_GENOTYPES}")
    elif mode == 'control':
        # Filter for selected F1 conditions
        if SELECTED_F1_CONDITIONS:
            df = df[df[secondary_col].isin(SELECTED_F1_CONDITIONS)].copy()
            print(f"  Filtered for {len(SELECTED_F1_CONDITIONS)} selected F1 conditions")
            print(f"  F1 Conditions: {SELECTED_F1_CONDITIONS}")

    # Filter for test ball
    if FILTER_TEST_BALL:
        ball_col = 'ball_identity'
        if ball_col not in df.columns:
            print(f"  Warning: '{ball_col}' column not found, skipping ball filter")
        else:
            print(f"  Available ball identities: {df[ball_col].unique()}")

            test_ball_data = pd.DataFrame()
            for test_val in TEST_BALL_VALUES:
                subset = df[df[ball_col] == test_val]
                if len(subset) > 0:
                    test_ball_data = pd.concat([test_ball_data, subset])

            if test_ball_data.empty:
                print(f"  Warning: No test ball data found")
            else:
                df = test_ball_data
                print(f"  Filtered to test ball: {len(df)} rows")

    print(f"  Final dataset size: {len(df)} rows")
    return df


def prepare_pca_data(df, metrics=None):
    """
    Prepare data for PCA analysis.

    Returns:
        X: Standardized feature matrix
        scaler: The fitted StandardScaler
        valid_metrics: List of metrics actually used (some may be missing)
        metadata_df: DataFrame with metadata (genotype, pretraining, etc.)
    """
    if metrics is None:
        metrics = PCA_METRICS

    print(f"\nPreparing data for PCA...")
    print(f"  Requested metrics: {metrics}")

    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    missing_metrics = [m for m in metrics if m not in df.columns]

    if missing_metrics:
        print(f"  ⚠️  Missing metrics: {missing_metrics}")

    if len(available_metrics) < 2:
        raise ValueError(f"Need at least 2 metrics for PCA, only found: {available_metrics}")

    print(f"  Using {len(available_metrics)} metrics: {available_metrics}")

    # Extract feature matrix
    X = df[available_metrics].copy()

    # Store metadata before filtering
    metadata_cols = ['fly', 'Genotype', 'Pretraining']
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata_df = df[metadata_cols].copy()

    # Handle missing values - drop rows with any NaN in the selected metrics
    n_before = len(X)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    metadata_df = metadata_df[valid_mask]
    n_after = len(X)

    if n_after < n_before:
        print(f"  Dropped {n_before - n_after} rows with missing values ({n_after} remaining)")

    if len(X) < 10:
        raise ValueError(f"Too few valid samples for PCA: {len(X)}")

    # Convert distance metrics from pixels to mm
    for metric in available_metrics:
        if 'distance' in metric.lower() and 'proximity' not in metric.lower():
            X[metric] = X[metric] / PIXELS_PER_MM
            print(f"  Converted {metric} from pixels to mm")

    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Final data shape: {X_scaled.shape}")
    print(f"  Features standardized (mean=0, std=1)")

    return X_scaled, scaler, available_metrics, metadata_df


# ============================================================================
# PCA ANALYSIS
# ============================================================================

def perform_pca(X, n_components=3):
    """
    Perform PCA analysis.

    Returns:
        pca: Fitted PCA object
        X_pca: Transformed data (principal components)
    """
    print(f"\nPerforming PCA with {n_components} components...")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(f"  Explained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var*100:.2f}%")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    return pca, X_pca


def create_pca_dataframe(X_pca, metadata_df, pca, metrics):
    """Create a DataFrame with PCA results and metadata"""
    n_components = X_pca.shape[1]

    # Create PC columns
    pc_cols = {f'PC{i+1}': X_pca[:, i] for i in range(n_components)}
    pca_df = pd.DataFrame(pc_cols)

    # Add metadata
    for col in metadata_df.columns:
        pca_df[col] = metadata_df[col].values

    # Add explained variance as attributes (for reference)
    pca_df.attrs['explained_variance_ratio'] = pca.explained_variance_ratio_
    pca_df.attrs['components'] = pca.components_
    pca_df.attrs['feature_names'] = metrics

    return pca_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_explained_variance(pca, output_dir):
    """Plot explained variance by component"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)

    # Individual explained variance
    ax1.bar(components, pca.explained_variance_ratio_ * 100,
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
    ax1.set_xticks(components)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on bars
    for i, (comp, var) in enumerate(zip(components, pca.explained_variance_ratio_)):
        ax1.text(comp, var * 100 + 1, f'{var*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(components, cumsum, 'o-', color='steelblue',
            linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.set_xticks(components)
    ax2.set_ylim([0, 105])
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')

    # Add cumulative percentage labels
    for i, (comp, var) in enumerate(zip(components, cumsum)):
        ax2.text(comp, var + 2, f'{var:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / f'pca_explained_variance.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")

    return fig


def plot_loadings_heatmap(pca, metrics, output_dir):
    """Plot heatmap of PCA loadings (component weights)"""
    n_components = pca.components_.shape[0]

    # Create DataFrame of loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=[METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LOADING)

    # Create heatmap
    sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'}, linewidths=0.5, linecolor='gray',
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('PCA Loadings: Metric Contributions to Principal Components',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    # Add explained variance to column labels
    col_labels = [f'PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)'
                  for i in range(n_components)]
    ax.set_xticklabels(col_labels, rotation=0)

    plt.tight_layout()

    output_path = Path(output_dir) / f'pca_loadings_heatmap.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")

    # Print top contributors for each PC
    print("\n  Top contributors to each PC:")
    for i in range(n_components):
        pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(f"    PC{i+1}: {pc_loadings.head(3).to_dict()}")

    return fig


def get_secondary_color(value, mode):
    """Get color for secondary grouping variable (genotype or F1 condition)"""
    if mode == 'tnt_full':
        return GENOTYPE_COLORS.get(value, '#808080')
    elif mode == 'control':
        return F1_CONDITION_COLORS.get(value, '#808080')
    return '#808080'


def plot_pc_boxplot(pca_df, pc_name, secondary_col, pretraining_col, output_dir, mode):
    """Create boxplot with jitter for a single principal component"""
    if pc_name not in pca_df.columns:
        print(f"  Warning: {pc_name} not found in data")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == 'tnt_full':
        # Use order from SELECTED_GENOTYPES (controls first, then by brain region)
        all_values = pca_df[secondary_col].unique()
        secondary_values = [g for g in SELECTED_GENOTYPES if g in all_values]
    else:
        # Use order from SELECTED_F1_CONDITIONS
        all_values = pca_df[secondary_col].unique()
        secondary_values = [c for c in SELECTED_F1_CONDITIONS if c in all_values]

    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())
    n_pretrain = len(pretraining_vals)

    # Create positions for grouped boxplots with more spacing
    # Increase spacing between groups for better separation
    group_spacing = 1.5  # Increased from 1.0 (default)
    group_width = 1.0    # Width available for boxes within each group
    box_width = group_width / n_pretrain

    all_data = []
    all_positions = []
    all_colors = []
    all_styles = []

    pos_idx = 0
    xtick_positions = []
    xtick_labels = []

    # Store data for statistical testing
    secondary_pretrain_data = {}

    for i, sec_val in enumerate(secondary_values):
        group_center = i * group_spacing
        xtick_positions.append(group_center)
        xtick_labels.append(sec_val)
        secondary_pretrain_data[sec_val] = {}

        for j, pretrain_val in enumerate(pretraining_vals):
            data = pca_df[(pca_df[secondary_col] == sec_val) &
                         (pca_df[pretraining_col] == pretrain_val)][pc_name].values

            if len(data) > 0:
                all_data.append(data)
                pos = group_center + (j - (n_pretrain - 1) / 2) * box_width
                all_positions.append(pos)
                all_colors.append(get_secondary_color(sec_val, mode))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(),
                                                         PRETRAINING_STYLES['n']))
                secondary_pretrain_data[sec_val][pretrain_val] = data
            else:
                all_data.append([])
                all_positions.append(group_center + (j - (n_pretrain - 1) / 2) * box_width)
                all_colors.append(get_secondary_color(sec_val, mode))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(),
                                                         PRETRAINING_STYLES['n']))

    # Perform Mann-Whitney tests for pretraining effect
    mw_results = []
    if len(pretraining_vals) == 2:
        for sec_val in secondary_values:
            data_dict = secondary_pretrain_data[sec_val]
            if len(data_dict) == 2:
                groups = list(data_dict.values())
                if len(groups[0]) >= 2 and len(groups[1]) >= 2:
                    stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    mw_results.append({
                        'group': sec_val,
                        'p_value': p_value,
                        'n1': len(groups[0]),
                        'n2': len(groups[1]),
                    })

    # FDR correction
    if mw_results:
        p_values = [r['p_value'] for r in mw_results]
        _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method='fdr_bh')
        for i, result in enumerate(mw_results):
            result['p_corrected'] = p_corrected[i]

    # Create boxplots - wider boxes for better visibility
    bp = ax.boxplot(all_data, positions=all_positions, widths=box_width * 0.75,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))

    # Style boxes
    for patch, color, style in zip(bp['boxes'], all_colors, all_styles):
        patch.set_facecolor(color)
        patch.set_alpha(style['alpha'])
        patch.set_edgecolor(style['edgecolor'])
        patch.set_linewidth(style['linewidth'])

    # Add jittered scatter points with reduced jitter to avoid overlap
    for data, pos, color in zip(all_data, all_positions, all_colors):
        if len(data) > 0:
            np.random.seed(42)
            jitter = np.random.normal(0, box_width * 0.15, size=len(data))  # Jitter proportional to box width
            ax.scatter(pos + jitter, data, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                      color=color, edgecolors='black', linewidths=0.5, zorder=3)

    # Add significance annotations
    if mw_results:
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        annotation_height = y_max + y_range * 0.02

        for i, sec_val in enumerate(secondary_values):
            result = next((r for r in mw_results if r['group'] == sec_val), None)
            if result:
                p_corr = result['p_corrected']
                if p_corr < 0.001:
                    sig_marker = '***'
                elif p_corr < 0.01:
                    sig_marker = '**'
                elif p_corr < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'

                ax.text(i * group_spacing, annotation_height, sig_marker, ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

        # Expand y-axis to accommodate annotations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + y_range * 0.08)

    # Set labels
    ax.set_xticks(xtick_positions)

    # Create labels with sample sizes
    labels_with_n = []
    for sec_val in secondary_values:
        n_values = []
        for pretrain_val in pretraining_vals:
            data = secondary_pretrain_data[sec_val].get(pretrain_val, [])
            n_values.append(len(data))

        if len(n_values) == 2:
            label = f"{sec_val}\n(n = {n_values[0]}, {n_values[1]})"
        else:
            label = f"{sec_val}\n(n = {sum(n_values)})"
        labels_with_n.append(label)

    ax.set_xticklabels(labels_with_n, rotation=45, ha='right', fontsize=9)

    # Get explained variance for this PC
    pc_idx = int(pc_name[2:]) - 1
    var_explained = pca_df.attrs['explained_variance_ratio'][pc_idx] * 100

    ax.set_ylabel(f'{pc_name} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{pc_name} ({var_explained:.1f}% variance explained)',
                fontsize=14, fontweight='bold')

    # Use appropriate xlabel based on mode
    secondary_label = GROUPING_CONFIG[mode]['secondary_label']
    ax.set_xlabel(secondary_label, fontsize=12, fontweight='bold')

    # Add legend
    legend_elements = []
    for pretrain_val in pretraining_vals:
        style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])
        legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                            facecolor='gray', alpha=style['alpha'],
                                            edgecolor=style['edgecolor'],
                                            linewidth=style['linewidth'],
                                            label=style['label']))

    # Add significance legend
    if mw_results:
        legend_elements.append(plt.Line2D([0], [0], color='none', label=''))
        legend_elements.append(plt.Line2D([0], [0], color='none', marker='$***$',
                                         markersize=10, label='p < 0.001 (FDR)'))
        legend_elements.append(plt.Line2D([0], [0], color='none', marker='$**$',
                                         markersize=10, label='p < 0.01 (FDR)'))
        legend_elements.append(plt.Line2D([0], [0], color='none', marker='$*$',
                                         markersize=10, label='p < 0.05 (FDR)'))
        legend_elements.append(plt.Line2D([0], [0], color='none', marker='$ns$',
                                         markersize=10, label='not significant'))

    # Place legend outside plot area to avoid hiding data
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = Path(output_dir) / f'{pc_name}_boxplot.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")

    # Print statistics
    if mw_results:
        print(f"  Statistics for {pc_name}:")
        for result in mw_results:
            print(f"    {result['group']}: p = {result['p_corrected']:.4f} "
                  f"(n = {result['n1']}, {result['n2']})")

    return fig


def plot_pca_scatter(pca_df, pc1, pc2, secondary_col, pretraining_col, output_dir, mode, suffix=''):
    """Create scatter plot of two principal components"""
    if pc1 not in pca_df.columns or pc2 not in pca_df.columns:
        print(f"  Warning: {pc1} or {pc2} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == 'tnt_full':
        # Use order from SELECTED_GENOTYPES (controls first, then by brain region)
        all_values = pca_df[secondary_col].unique()
        secondary_values = [g for g in SELECTED_GENOTYPES if g in all_values]
    else:
        # Use order from SELECTED_F1_CONDITIONS
        all_values = pca_df[secondary_col].unique()
        secondary_values = [c for c in SELECTED_F1_CONDITIONS if c in all_values]

    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

    # Plot each genotype-pretraining combination
    for sec_val in secondary_values:
        for pretrain_val in pretraining_vals:
            mask = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretrain_val)
            subset = pca_df[mask]

            if len(subset) == 0:
                continue

            color = get_secondary_color(sec_val, mode)
            style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])

            label = f"{sec_val} - {style['label']}"

            ax.scatter(subset[pc1], subset[pc2],
                      color=color, alpha=style['alpha'],
                      s=SCATTER_SIZE*2, marker=style['marker'],
                      edgecolors=style['edgecolor'], linewidths=style['linewidth'],
                      label=label, zorder=3)

    # Get explained variance
    pc1_idx = int(pc1[2:]) - 1
    pc2_idx = int(pc2[2:]) - 1
    var1 = pca_df.attrs['explained_variance_ratio'][pc1_idx] * 100
    var2 = pca_df.attrs['explained_variance_ratio'][pc2_idx] * 100

    ax.set_xlabel(f'{pc1} ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{pc2} ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_title(f'PCA: {pc1} vs {pc2}', fontsize=14, fontweight='bold')

    # Add origin lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    ax.grid(alpha=0.3, linestyle='--', zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir) / f'pca_scatter_{pc1}_vs_{pc2}{suffix}.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")

    return fig


def plot_3d_pca(pca_df, secondary_col, pretraining_col, output_dir, mode):
    """Create 3D scatter plot of first three PCs"""
    if 'PC1' not in pca_df.columns or 'PC2' not in pca_df.columns or 'PC3' not in pca_df.columns:
        print("  Warning: Not enough PCs for 3D plot")
        return None

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == 'tnt_full':
        # Use order from SELECTED_GENOTYPES (controls first, then by brain region)
        all_values = pca_df[secondary_col].unique()
        secondary_values = [g for g in SELECTED_GENOTYPES if g in all_values]
    else:
        # Use order from SELECTED_F1_CONDITIONS
        all_values = pca_df[secondary_col].unique()
        secondary_values = [c for c in SELECTED_F1_CONDITIONS if c in all_values]

    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

    # Plot each combination
    for sec_val in secondary_values:
        for pretrain_val in pretraining_vals:
            mask = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretrain_val)
            subset = pca_df[mask]

            if len(subset) == 0:
                continue

            color = get_secondary_color(sec_val, mode)
            style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES['n'])

            label = f"{sec_val} - {style['label']}"

            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                      color=color, alpha=style['alpha'],
                      s=SCATTER_SIZE*2, marker=style['marker'],
                      edgecolors=style['edgecolor'], linewidths=style['linewidth'],
                      label=label)

    # Get explained variance
    var1 = pca_df.attrs['explained_variance_ratio'][0] * 100
    var2 = pca_df.attrs['explained_variance_ratio'][1] * 100
    var3 = pca_df.attrs['explained_variance_ratio'][2] * 100

    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'PC3 ({var3:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_title('PCA: 3D View of First Three Components', fontsize=14, fontweight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir) / f'pca_3d_scatter.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PCA Analysis for F1 Behavioral Metrics')
    parser.add_argument('--mode', type=str, default='tnt_full',
                       choices=['control', 'tnt_full'],
                       help='Analysis mode: control (F1_condition) or tnt_full (genotypes)')
    parser.add_argument('--n-components', type=int, default=3,
                       help='Number of principal components')
    parser.add_argument('--show', action='store_true',
                       help='Display plots instead of just saving')
    parser.add_argument('--metrics', type=str,
                       help='Comma-separated list of metrics (default: predefined list)')
    parser.add_argument('--detect-secondary', action='store_true',
                       help='Print all secondary grouping values found in dataset and exit')

    args = parser.parse_args()

    print("=" * 80)
    print(f"PCA ANALYSIS FOR F1 BEHAVIORAL METRICS ({args.mode.upper()} MODE)")
    print("=" * 80)

    # Load dataset for selected mode
    df = load_dataset(args.mode)

    # Detect columns
    secondary_col, pretraining_col = detect_columns(df, args.mode)

    # Detect secondary values if requested
    if args.detect_secondary:
        all_values = sorted(df[secondary_col].dropna().unique().tolist())
        print(f"\nFound {len(all_values)} {secondary_col} values: {all_values}")
        print("\n" + "=" * 80)
        if args.mode == 'tnt_full':
            print("Copy to SELECTED_GENOTYPES in script:")
            print("=" * 80)
            print("\nSELECTED_GENOTYPES = [")
        else:
            print("Copy to SELECTED_F1_CONDITIONS in script:")
            print("=" * 80)
            print("\nSELECTED_F1_CONDITIONS = [")
        for v in all_values:
            print(f"    '{v}',")
        print("]")
        return

    # Filter dataset
    df = filter_dataset(df, secondary_col, args.mode)

    # Determine metrics to use
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
        print(f"\nUsing user-specified metrics: {metrics}")
    else:
        metrics = PCA_METRICS
        print(f"\nUsing default metrics: {metrics}")

    # Prepare data
    X, scaler, valid_metrics, metadata_df = prepare_pca_data(df, metrics)

    # Perform PCA
    pca, X_pca = perform_pca(X, n_components=args.n_components)

    # Create results DataFrame
    pca_df = create_pca_dataframe(X_pca, metadata_df, pca, valid_metrics)

    # Set up output directory
    output_dir = Path(OUTPUT_DIRS[args.mode])
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # 1. Explained variance
    print("\n1. Plotting explained variance...")
    plot_explained_variance(pca, output_dir)

    # 2. Loadings heatmap
    print("\n2. Plotting loadings heatmap...")
    plot_loadings_heatmap(pca, valid_metrics, output_dir)

    # 3. PC boxplots
    print("\n3. Plotting PC boxplots...")
    for i in range(args.n_components):
        pc_name = f'PC{i+1}'
        plot_pc_boxplot(pca_df, pc_name, secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()

    # 4. 2D scatter plots
    print("\n4. Plotting 2D scatter plots...")
    plot_pca_scatter(pca_df, 'PC1', 'PC2', secondary_col, pretraining_col, output_dir, args.mode)
    plt.close()

    if args.n_components >= 3:
        plot_pca_scatter(pca_df, 'PC1', 'PC3', secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()
        plot_pca_scatter(pca_df, 'PC2', 'PC3', secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()

    # 5. 3D scatter plot
    if args.n_components >= 3:
        print("\n5. Plotting 3D scatter plot...")
        plot_3d_pca(pca_df, secondary_col, pretraining_col, output_dir, args.mode)
    output_csv = output_dir / 'pca_results.csv'
    pca_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}")

    # Save loadings to CSV
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(args.n_components)],
        index=valid_metrics
    )
    loadings_csv = output_dir / 'pca_loadings.csv'
    loadings_df.to_csv(loadings_csv)
    print(f"  Saved: {loadings_csv.name}")

    if args.show:
        plt.show()
    else:
        plt.close('all')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  - Total variance explained by {args.n_components} PCs: "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  - Top contributors to PC1: "
          f"{loadings_df['PC1'].abs().nlargest(3).to_dict()}")
    if args.n_components >= 2:
        print(f"  - Top contributors to PC2: "
              f"{loadings_df['PC2'].abs().nlargest(3).to_dict()}")


if __name__ == "__main__":
    main()
