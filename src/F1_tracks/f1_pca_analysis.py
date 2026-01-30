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
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from itertools import combinations

# Import multivariate statistical analysis functions
from pca_multivariate_stats import perform_multivariate_statistical_analysis

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis mode - 'control' or 'tnt_full'
ANALYSIS_MODE = "tnt_full"  # Change to 'control' for F1 New control experiment

# Dataset paths by mode
DATASET_PATHS = {
    "control": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260104_12_F1_coordinates_F1_New_Data/summary/pooled_summary.feather",
    "tnt_full": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather",
}

OUTPUT_DIRS = {
    "control": "/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New/PCA",
    "tnt_full": "/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/PCA/Simplified",
}

# Create output directory if it doesn't exist
Path(OUTPUT_DIRS[ANALYSIS_MODE]).mkdir(parents=True, exist_ok=True)

# Mode-specific grouping configuration
GROUPING_CONFIG = {
    "control": {
        "secondary_var": "f1_condition",
        "secondary_label": "F1 Condition",
        "selected_values": "SELECTED_F1_CONDITIONS",
    },
    "tnt_full": {
        "secondary_var": "genotype",
        "secondary_label": "Genotype",
        "selected_values": "SELECTED_GENOTYPES",
    },
}

# ============================================================================
# GENOTYPE SELECTION - Comment out genotypes to exclude from analysis
# For 'control' mode: No genotype filtering (single genotype)
# For 'tnt_full' mode: Select which TNT genotypes to include
# ============================================================================

SELECTED_GENOTYPES = [
    # === CONTROLS ===
    # "TNTxEmptyGal4",
    "TNTxEmptySplit",
    #'TNTxPR',
    # === EXPERIMENTAL GENOTYPES ===
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    #'PRxTH',
    "TNTxMB247",
    #'PRxMB247',
    "TNTxLC10-2",
    "TNTxLC16-1",
    #'PRxLC16-1',
]

# F1 conditions for control mode
SELECTED_F1_CONDITIONS = [
    "control",
    "pretrained",
    #'pretrained_unlocked',  # Uncomment to include
]

# Metrics for PCA - these should be present in the summary dataset
PCA_METRICS = [
    "ball_proximity_proportion_200px",  # Time spent near ball
    "distance_moved",  # Total ball distance moved
    "max_distance",  # Maximum ball distance
    "nb_events",  # Number of interaction events
    "has_major",  # Binary: achieved major event
    "has_finished",  # Binary: completed task
    "significant_ratio",  # Proportion of significant events
]

# Elegant names for display
METRIC_DISPLAY_NAMES = {
    "ball_proximity_proportion_200px": "Ball Proximity",
    "distance_moved": "Distance Moved",
    "max_distance": "Max Distance",
    "nb_events": "N Events",
    "has_major": "Has Major",
    "has_finished": "Finished Task",
    "significant_ratio": "Significant Ratio",
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
PLOT_FORMAT = ["png", "pdf", "svg"]
JITTER_AMOUNT = 0.15
SCATTER_SIZE = 40
SCATTER_ALPHA = 0.6


def save_figure_multiple_formats(fig, output_dir, filename_base, dpi=300, formats=None):
    """Save figure in multiple formats (png, pdf, svg)"""
    if formats is None:
        formats = PLOT_FORMAT

    output_dir = Path(output_dir)
    saved_files = []

    for fmt in formats:
        output_path = output_dir / f"{filename_base}.{fmt}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", format=fmt)
        saved_files.append(output_path.name)

    return saved_files


# Colors for genotypes (from f1_tnt_genotype_comparison.py)
GENOTYPE_COLORS = {
    "TNTxEmptyGal4": "#7f7f7f",
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxPR": "#7f7f7f",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "PRxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "PRxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
    "PRxLC16-1": "#ff7f0e",
}

# Colors for F1 conditions (control mode)
F1_CONDITION_COLORS = {
    "control": "steelblue",
    "pretrained": "orange",
    "pretrained_unlocked": "lightgreen",
}

# Pretraining styles
PRETRAINING_STYLES = {
    "n": {
        "alpha": 0.6,
        "edgecolor": "black",
        "linewidth": 1.5,
        "label": "Naive",
        "marker": "o",
    },
    "y": {
        "alpha": 1.0,
        "edgecolor": "black",
        "linewidth": 2.5,
        "label": "Pretrained",
        "marker": "s",
    },
}

ALPHA = 0.05

# Statistical method configuration
USE_PERMANOVA = True  # PERMANOVA for global multivariate test
USE_PAIRWISE_PERMUTATION = True  # Pairwise permutation tests per genotype
USE_LMM_ON_PCS = True  # Linear mixed-effects models on individual PC scores (for effect sizes)
USE_MAHALANOBIS = True  # Mahalanobis distances between centroids
USE_MANN_WHITNEY_ON_PCS = False  # DISABLED: Mann-Whitney (doesn't account for blocking)
USE_GENOTYPE_EFFECT_ANNOTATION = False  # Show genotype main effect background shading (can clutter plot)
USE_RESIDUAL_PERMUTATION_ON_PCS = True  # PRIMARY: Residual permutation tests on PCs (distribution-free, blocking-aware)
USE_PERMUTATION_ON_PCS = False  # DISABLED: Standard permutation (doesn't account for blocking)
N_PERMUTATIONS = 10000  # Number of permutations for permutation tests
N_BOOTSTRAP = 1000  # Number of bootstrap samples for confidence ellipses
SAVE_DIAGNOSTIC_PLOTS = True  # Save diagnostic residual plots for each PC
TEST_DATE_EFFECT = True  # Test if date has effect before including as blocking factor


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
    secondary_var = GROUPING_CONFIG[mode]["secondary_var"]

    for col in df.columns:
        # Look for secondary grouping variable (genotype or f1_condition)
        if secondary_var == "genotype" and "genotype" in col.lower() and secondary_col is None:
            secondary_col = col
        elif (
            secondary_var == "f1_condition"
            and "f1" in col.lower()
            and "condition" in col.lower()
            and secondary_col is None
        ):
            secondary_col = col

        # Look for pretraining column
        if "pretrain" in col.lower() and pretraining_col is None:
            pretraining_col = col

    if secondary_col is None:
        raise ValueError(f"Could not find {secondary_var} column for mode '{mode}'")
    if pretraining_col is None:
        raise ValueError("Could not find pretraining column")

    secondary_label = GROUPING_CONFIG[mode]["secondary_label"]
    print(f"  {secondary_label} column: {secondary_col}")
    print(f"  Pretraining column: {pretraining_col}")

    return secondary_col, pretraining_col


def filter_dataset(df, secondary_col, mode):
    """Filter dataset for selected genotypes/conditions and test ball"""
    secondary_var = GROUPING_CONFIG[mode]["secondary_var"]

    # Filter based on mode
    if mode == "tnt_full":
        # Filter for selected genotypes
        if SELECTED_GENOTYPES:
            df = df[df[secondary_col].isin(SELECTED_GENOTYPES)].copy()
            print(f"  Filtered for {len(SELECTED_GENOTYPES)} selected genotypes")
            print(f"  Genotypes: {SELECTED_GENOTYPES}")
    elif mode == "control":
        # Filter for selected F1 conditions
        if SELECTED_F1_CONDITIONS:
            df = df[df[secondary_col].isin(SELECTED_F1_CONDITIONS)].copy()
            print(f"  Filtered for {len(SELECTED_F1_CONDITIONS)} selected F1 conditions")
            print(f"  F1 Conditions: {SELECTED_F1_CONDITIONS}")

    # Filter for test ball
    if FILTER_TEST_BALL:
        ball_col = "ball_identity"
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
    metadata_cols = ["fly", "Genotype", "Pretraining"]
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
        if "distance" in metric.lower() and "proximity" not in metric.lower():
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
    pc_cols = {f"PC{i+1}": X_pca[:, i] for i in range(n_components)}
    pca_df = pd.DataFrame(pc_cols)

    # Add metadata
    for col in metadata_df.columns:
        pca_df[col] = metadata_df[col].values

    # Add explained variance as attributes (for reference)
    pca_df.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    pca_df.attrs["components"] = pca.components_
    pca_df.attrs["feature_names"] = metrics

    return pca_df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def perform_lmm_on_pc(pca_df, pc_name, secondary_col, pretraining_col, mode):
    """
    Perform LMM analysis on a single PC, testing pretraining effect for each genotype/group.

    Uses the same model structure as f1_tnt_genotype_comparison.py:
    - Random effects: fly (grouping), date, arena_number, arena_side
    - Fixed effects: genotype * pretraining interaction
    - Tests pretraining effect within each genotype

    Returns:
        dict: Results with genotype-specific p-values and effect sizes
    """
    if pc_name not in pca_df.columns:
        return {}

    print(f"\n{'='*60}")
    print(f"LMM Analysis for {pc_name}")
    print(f"{'='*60}")

    # Prepare data - include random effect columns
    random_effect_cols = []
    for col in ["fly", "Date", "date", "arena", "corridor"]:
        if col in pca_df.columns:
            random_effect_cols.append(col)

    columns_needed = [secondary_col, pretraining_col, pc_name] + random_effect_cols
    df_metric = pca_df[columns_needed].copy()
    df_metric = df_metric.dropna()

    # Parse fly name to extract arena and side information
    if "fly" in df_metric.columns:

        def parse_fly_name(fly_name):
            """Extract arena number and side from fly name"""
            parts = str(fly_name).split("_")
            arena_num = None
            side = None

            for i, part in enumerate(parts):
                if "arena" in part.lower():
                    arena_num = part.lower().replace("arena", "")
                if part in ["Left", "Right"]:
                    side = part

            return arena_num, side

        df_metric[["arena_number", "arena_side"]] = df_metric["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))

        # Add to random effects if successfully parsed
        if df_metric["arena_number"].notna().any():
            random_effect_cols.append("arena_number")
        if df_metric["arena_side"].notna().any():
            random_effect_cols.append("arena_side")

    # Convert pretraining to numeric (0 = naive, 1 = pretrained)
    df_metric["pretraining_numeric"] = df_metric[pretraining_col].apply(
        lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
    )

    # Create reference-coded secondary variable
    secondary_values = df_metric[secondary_col].unique()
    control_value = secondary_values[0]  # Use first as reference
    df_metric["secondary_cat"] = pd.Categorical(
        df_metric[secondary_col],
        categories=[control_value] + [g for g in sorted(secondary_values) if g != control_value],
    )

    genotype_effects = {}

    try:
        # Print random effects structure
        if "fly" in random_effect_cols:
            groups = df_metric["fly"]
            print(f"\n🔀 Random Effects: fly (n={df_metric['fly'].nunique()} groups)")

            if "Date" in random_effect_cols or "date" in random_effect_cols:
                date_col = "Date" if "Date" in random_effect_cols else "date"
                print(f"   + {date_col} (n={df_metric[date_col].nunique()} levels)")
            if "arena" in random_effect_cols:
                print(f"   + arena (n={df_metric['arena'].nunique()} levels)")
            if "arena_number" in random_effect_cols:
                print(f"   + arena_number (n={df_metric['arena_number'].nunique()} levels)")
            if "arena_side" in random_effect_cols:
                print(f"   + arena_side (n={df_metric['arena_side'].nunique()} levels)")

        # For each genotype/group, fit separate LMM to test pretraining effect
        for sec_value in secondary_values:
            df_geno = df_metric[df_metric[secondary_col] == sec_value].copy()

            if len(df_geno) < 10:  # Need sufficient data
                continue

            if df_geno["pretraining_numeric"].nunique() < 2:  # Need both conditions
                continue

            try:
                # Fit LMM for this genotype
                formula = f"{pc_name} ~ pretraining_numeric"

                if "fly" in df_geno.columns and df_geno["fly"].nunique() > 1:
                    model = MixedLM.from_formula(formula, data=df_geno, groups=df_geno["fly"], re_formula="1").fit(
                        method="lbfgs"
                    )
                else:
                    # Fall back to OLS if not enough groups
                    model = smf.ols(formula, data=df_geno).fit(cov_type="HC3")

                # Extract results
                effect_size = model.params.get("pretraining_numeric", np.nan)
                effect_stderr = model.bse.get("pretraining_numeric", np.nan)
                effect_pval = model.pvalues.get("pretraining_numeric", np.nan)

                # Calculate means
                mean_naive = df_geno[df_geno["pretraining_numeric"] == 0][pc_name].mean()
                mean_pretrained = df_geno[df_geno["pretraining_numeric"] == 1][pc_name].mean()
                n_naive = len(df_geno[df_geno["pretraining_numeric"] == 0])
                n_pretrained = len(df_geno[df_geno["pretraining_numeric"] == 1])

                genotype_effects[sec_value] = {
                    "effect_size": effect_size,
                    "stderr": effect_stderr,
                    "pvalue": effect_pval,
                    "mean_naive": mean_naive,
                    "mean_pretrained": mean_pretrained,
                    "n_naive": n_naive,
                    "n_pretrained": n_pretrained,
                }

                # Print result
                sig = (
                    "***"
                    if effect_pval < 0.001
                    else "**" if effect_pval < 0.01 else "*" if effect_pval < 0.05 else "ns"
                )
                print(f"  {sec_value:20s}: β={effect_size:7.3f} (SE={effect_stderr:.3f}), p={effect_pval:.4f} {sig}")

            except Exception as e:
                print(f"  {sec_value}: Failed to fit model ({str(e)})")

    except Exception as e:
        print(f"\n⚠️ Error in LMM analysis for {pc_name}: {str(e)}")

    return genotype_effects


def perform_lmm_genotype_effect_on_pc(pca_df, pc_name, secondary_col, pretraining_col, mode):
    """
    Test for main genotype effect on PC (controlling for pretraining).

    This reveals inherent genotype-specific differences in ball pushing efficiency
    independent of pretraining effects.

    Returns:
        dict: Global model results and per-genotype means
    """
    print(f"\n{'='*60}")
    print(f"Genotype Main Effect on {pc_name}")
    print(f"{'='*60}")

    # Prepare data with random effects
    random_effect_cols = []
    for col in ["fly", "Date", "date", "arena", "corridor"]:
        if col in pca_df.columns:
            random_effect_cols.append(col)

    columns_needed = [secondary_col, pretraining_col, pc_name] + random_effect_cols
    df_metric = pca_df[columns_needed].copy()
    df_metric = df_metric.dropna()

    # Parse fly name for arena info if needed
    if "fly" in df_metric.columns:

        def parse_fly_name(fly_name):
            parts = str(fly_name).split("_")
            arena_num = None
            side = None
            for part in parts:
                if "arena" in part.lower():
                    arena_num = part.lower().replace("arena", "")
                if part in ["Left", "Right"]:
                    side = part
            return arena_num, side

        df_metric[["arena_number", "arena_side"]] = df_metric["fly"].apply(lambda x: pd.Series(parse_fly_name(x)))
        if df_metric["arena_number"].notna().any():
            random_effect_cols.append("arena_number")
        if df_metric["arena_side"].notna().any():
            random_effect_cols.append("arena_side")

    # Convert pretraining to numeric
    df_metric["pretraining_numeric"] = df_metric[pretraining_col].apply(
        lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
    )

    # Create reference-coded secondary variable
    secondary_values = sorted(df_metric[secondary_col].unique())
    control_value = secondary_values[0]
    df_metric["secondary_cat"] = pd.Categorical(
        df_metric[secondary_col],
        categories=[control_value] + [g for g in secondary_values if g != control_value],
    )

    try:
        # Fit global LMM testing genotype effect while controlling for pretraining
        formula = f"{pc_name} ~ C(secondary_cat) + pretraining_numeric"

        if "fly" in df_metric.columns and df_metric["fly"].nunique() > 1:
            model = MixedLM.from_formula(formula, data=df_metric, groups=df_metric["fly"], re_formula="1").fit(
                method="lbfgs"
            )
        else:
            model = smf.ols(formula, data=df_metric).fit(cov_type="HC3")

        # Get overall mean and per-genotype means
        overall_mean = df_metric[pc_name].mean()

        genotype_stats = {}
        for sec_val in secondary_values:
            df_geno = df_metric[df_metric[secondary_col] == sec_val]
            mean_val = df_geno[pc_name].mean()
            n_val = len(df_geno)
            genotype_stats[sec_val] = {
                "mean": mean_val,
                "n": n_val,
                "deviation": mean_val - overall_mean,
            }

        # Print genotype means (sorted by deviation)
        print(f"\n📊 Per-genotype PC{pc_name.replace('PC', '')} means (controlling for pretraining):")
        print(f"   Overall mean: {overall_mean:.3f}")

        sorted_stats = sorted(genotype_stats.items(), key=lambda x: x[1]["deviation"], reverse=True)
        for sec_val, stats_dict in sorted_stats:
            dev = stats_dict["deviation"]
            sign = "+" if dev > 0 else ""
            print(f"   {sec_val:20s}: mean={stats_dict['mean']:7.3f} ({sign}{dev:6.3f})  n={stats_dict['n']}")

        return genotype_stats, model

    except Exception as e:
        print(f"⚠️ Error in genotype effect analysis: {str(e)}")
        return {}, None


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_explained_variance(pca, output_dir):
    """Plot explained variance by component"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)

    # Individual explained variance
    ax1.bar(
        components, pca.explained_variance_ratio_ * 100, color="steelblue", alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Explained Variance by Component", fontsize=14, fontweight="bold")
    ax1.set_xticks(components)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add percentage labels on bars
    for i, (comp, var) in enumerate(zip(components, pca.explained_variance_ratio_)):
        ax1.text(comp, var * 100 + 1, f"{var*100:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(
        components,
        cumsum,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax2.axhline(y=80, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="80% threshold")
    ax2.axhline(y=90, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="90% threshold")
    ax2.set_xlabel("Number of Components", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Explained Variance", fontsize=14, fontweight="bold")
    ax2.set_xticks(components)
    ax2.set_ylim([0, 105])
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(loc="lower right")

    # Add cumulative percentage labels
    for i, (comp, var) in enumerate(zip(components, cumsum)):
        ax2.text(comp, var + 2, f"{var:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, "pca_explained_variance")
    print(f"  Saved: {', '.join(saved_files)}")

    return fig


def plot_loadings_heatmap(pca, metrics, output_dir):
    """Plot heatmap of PCA loadings (component weights)"""
    n_components = pca.components_.shape[0]

    # Create DataFrame of loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=[METRIC_DISPLAY_NAMES.get(m, m) for m in metrics],
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LOADING)

    # Create heatmap
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Loading"},
        linewidths=0.5,
        linecolor="gray",
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("PCA Loadings: Metric Contributions to Principal Components", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")

    # Add explained variance to column labels
    col_labels = [f"PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_components)]
    ax.set_xticklabels(col_labels, rotation=0)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, "pca_loadings_heatmap")
    print(f"  Saved: {', '.join(saved_files)}")

    # Print top contributors for each PC
    print("\n  Top contributors to each PC:")
    for i in range(n_components):
        pc_loadings = loadings[f"PC{i+1}"].abs().sort_values(ascending=False)
        print(f"    PC{i+1}: {pc_loadings.head(3).to_dict()}")

    return fig


def get_secondary_color(value, mode):
    """Get color for secondary grouping variable (genotype or F1 condition)"""
    if mode == "tnt_full":
        return GENOTYPE_COLORS.get(value, "#808080")
    elif mode == "control":
        return F1_CONDITION_COLORS.get(value, "#808080")
    return "#808080"


def plot_pc_boxplot(pca_df, pc_name, secondary_col, pretraining_col, output_dir, mode, stat_results=None):
    """Create boxplot with jitter for a single principal component"""
    if pc_name not in pca_df.columns:
        print(f"  Warning: {pc_name} not found in data")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == "tnt_full":
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
    group_width = 1.0  # Width available for boxes within each group
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
            data = pca_df[(pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretrain_val)][
                pc_name
            ].values

            if len(data) > 0:
                all_data.append(data)
                pos = group_center + (j - (n_pretrain - 1) / 2) * box_width
                all_positions.append(pos)
                all_colors.append(get_secondary_color(sec_val, mode))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"]))
                secondary_pretrain_data[sec_val][pretrain_val] = data
            else:
                all_data.append([])
                all_positions.append(group_center + (j - (n_pretrain - 1) / 2) * box_width)
                all_colors.append(get_secondary_color(sec_val, mode))
                all_styles.append(PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"]))

    # Extract residual permutation test results for this PC
    # These are pretraining effects within each genotype (no FDR correction)
    perm_results = []
    if stat_results and "residual_permutation_on_pcs" in stat_results:
        pc_results = stat_results["residual_permutation_on_pcs"].get(pc_name, [])
        for result in pc_results:
            perm_results.append(
                {
                    "group": result["group"],
                    "p_corrected": result["p_value_corrected"],  # No FDR - only 2 comparisons per genotype
                    "method": "Residual Permutation",
                }
            )

    # Also perform genotype main effect analysis (for PC1 and PC2 only to avoid clutter)
    pc_idx = int(pc_name.replace("PC", ""))
    genotype_effect_stats = None
    if pc_idx <= 2:  # Only for PC1 and PC2
        genotype_effect_stats, _ = perform_lmm_genotype_effect_on_pc(
            pca_df, pc_name, secondary_col, pretraining_col, mode
        )

    # Create boxplots - wider boxes for better visibility
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=box_width * 0.75,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Style boxes
    for patch, color, style in zip(bp["boxes"], all_colors, all_styles):
        patch.set_facecolor(color)
        patch.set_alpha(style["alpha"])
        patch.set_edgecolor(style["edgecolor"])
        patch.set_linewidth(style["linewidth"])

    # Add jittered scatter points with reduced jitter to avoid overlap
    for data, pos, color in zip(all_data, all_positions, all_colors):
        if len(data) > 0:
            np.random.seed(42)
            jitter = np.random.normal(0, box_width * 0.15, size=len(data))  # Jitter proportional to box width
            ax.scatter(
                pos + jitter,
                data,
                alpha=SCATTER_ALPHA,
                s=SCATTER_SIZE,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    # Add significance annotations (pretraining effect within each genotype)
    if perm_results:
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        annotation_height = y_max + y_range * 0.02

        for i, sec_val in enumerate(secondary_values):
            result = next((r for r in perm_results if r["group"] == sec_val), None)
            if result:
                p_corr = result["p_corrected"]
                if p_corr < 0.001:
                    sig_marker = "***"
                elif p_corr < 0.01:
                    sig_marker = "**"
                elif p_corr < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "ns"

                ax.text(
                    i * group_spacing,
                    annotation_height,
                    sig_marker,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="red" if p_corr < 0.05 else "gray",
                )

        # Expand y-axis to accommodate annotations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + y_range * 0.08)

    # Add genotype main effect annotation (if available and enabled)
    if USE_GENOTYPE_EFFECT_ANNOTATION and genotype_effect_stats:
        y_min, y_max = ax.get_ylim()
        overall_mean = np.mean([stats_dict["mean"] for stats_dict in genotype_effect_stats.values()])

        # Add horizontal line at overall mean
        ax.axhline(y=overall_mean, color="gray", linestyle=":", linewidth=1.5, alpha=0.6, zorder=1)

        # Highlight genotypes with significant deviations (>0.3 SD from mean)
        # Calculate standard deviation across all data
        all_pc_values = np.concatenate([d for d in all_data if len(d) > 0])
        pc_std = np.std(all_pc_values)
        sig_threshold = 0.3 * pc_std  # Significant if deviation > 0.3 SD

        for i, sec_val in enumerate(secondary_values):
            if sec_val in genotype_effect_stats:
                deviation = genotype_effect_stats[sec_val]["deviation"]
                # Highlight with subtle background color if deviation is substantial
                if abs(deviation) > sig_threshold:
                    color = "green" if deviation > 0 else "red"
                    alpha = min(0.15, abs(deviation) / (2 * pc_std))  # Intensity based on magnitude
                    # Add vertical band
                    x_pos = i * group_spacing
                    ax.axvspan(x_pos - group_width / 2, x_pos + group_width / 2, color=color, alpha=alpha, zorder=0)

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

    ax.set_xticklabels(labels_with_n, rotation=45, ha="right", fontsize=9)

    # Get explained variance for this PC
    pc_idx = int(pc_name[2:]) - 1
    var_explained = pca_df.attrs["explained_variance_ratio"][pc_idx] * 100

    ax.set_ylabel(f"{pc_name} Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{pc_name} ({var_explained:.1f}% variance explained)", fontsize=14, fontweight="bold")

    # Use appropriate xlabel based on mode
    secondary_label = GROUPING_CONFIG[mode]["secondary_label"]
    ax.set_xlabel(secondary_label, fontsize=12, fontweight="bold")

    # Add legend
    legend_elements = []
    for pretrain_val in pretraining_vals:
        style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                alpha=style["alpha"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                label=style["label"],
            )
        )

    # Add significance legend (simplified - no FDR since LMM tests each genotype separately)
    if stat_results:
        legend_elements.append(plt.Line2D([0], [0], color="none", label=""))
        legend_elements.append(
            plt.Line2D([0], [0], color="none", marker="$***$", markersize=10, label="p < 0.001 (LMM)")
        )
        legend_elements.append(plt.Line2D([0], [0], color="none", marker="$**$", markersize=10, label="p < 0.01 (LMM)"))
        legend_elements.append(plt.Line2D([0], [0], color="none", marker="$*$", markersize=10, label="p < 0.05 (LMM)"))
        legend_elements.append(
            plt.Line2D([0], [0], color="none", marker="$ns$", markersize=10, label="not significant")
        )

    # Add genotype main effect legend (only if enabled)
    if USE_GENOTYPE_EFFECT_ANNOTATION and genotype_effect_stats:
        legend_elements.append(plt.Line2D([0], [0], color="none", label=""))
        legend_elements.append(plt.Line2D([0], [0], linestyle=":", color="gray", linewidth=1.5, label="Overall mean"))
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="green", alpha=0.15, label="Above mean (genotype effect)")
        )
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.15, label="Below mean (genotype effect)")
        )

    # Place legend outside plot area to avoid hiding data
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, f"{pc_name}_boxplot")
    print(f"  Saved: {', '.join(saved_files)}")

    # Print statistics
    if perm_results:
        print(f"\n  Residual Permutation Statistics for {pc_name}:")
        for result in perm_results:
            group = result["group"]
            p_val = result["p_corrected"]
            # Find the full result from stat_results to get sample sizes
            if stat_results and "residual_permutation_on_pcs" in stat_results:
                pc_test_results = stat_results["residual_permutation_on_pcs"].get(pc_name, [])
                full_result = next((r for r in pc_test_results if r["group"] == group), None)
                if full_result:
                    n_naive = full_result.get("n_naive", "?")
                    n_pretrained = full_result.get("n_pretrained", "?")
                    print(f"    {group}: p = {p_val:.4f} (n = {n_naive}, {n_pretrained})")
                else:
                    print(f"    {group}: p = {p_val:.4f}")
            else:
                print(f"    {group}: p = {p_val:.4f}")

    return fig


def plot_pca_scatter(pca_df, pc1, pc2, secondary_col, pretraining_col, output_dir, mode, suffix=""):
    """Create scatter plot of two principal components"""
    if pc1 not in pca_df.columns or pc2 not in pca_df.columns:
        print(f"  Warning: {pc1} or {pc2} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == "tnt_full":
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
            style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])

            label = f"{sec_val} - {style['label']}"

            ax.scatter(
                subset[pc1],
                subset[pc2],
                color=color,
                alpha=style["alpha"],
                s=SCATTER_SIZE * 2,
                marker=style["marker"],
                edgecolors=style["edgecolor"],
                linewidths=style["linewidth"],
                label=label,
                zorder=3,
            )

    # Get explained variance
    pc1_idx = int(pc1[2:]) - 1
    pc2_idx = int(pc2[2:]) - 1
    var1 = pca_df.attrs["explained_variance_ratio"][pc1_idx] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][pc2_idx] * 100

    ax.set_xlabel(f"{pc1} ({var1:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({var2:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_title(f"PCA: {pc1} vs {pc2}", fontsize=14, fontweight="bold")

    # Add origin lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    ax.grid(alpha=0.3, linestyle="--", zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, f"pca_scatter_{pc1}_vs_{pc2}{suffix}")
    print(f"  Saved: {', '.join(saved_files)}")

    return fig


def plot_bivariate_confidence_ellipses(
    pca_df, pc1, pc2, secondary_col, pretraining_col, output_dir, mode, bivariate_results=None
):
    """
    Plot two PCs with 95% confidence ellipses for each genotype-pretraining combination.
    Ellipses visualize the bivariate distribution and separation between groups.
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if pc1 not in pca_df.columns or pc2 not in pca_df.columns:
        print(f"  Warning: {pc1} or {pc2} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    def confidence_ellipse(x, y, ax, n_std=1.96, facecolor="none", **kwargs):
        """Draw 95% confidence ellipse (1.96 std)"""
        if len(x) < 2 or len(y) < 2:
            return None

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D().scale(scale_x, scale_y).translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    # Get unique values - preserve order
    if mode == "tnt_full":
        all_values = pca_df[secondary_col].unique()
        secondary_values = [g for g in SELECTED_GENOTYPES if g in all_values]
    else:
        all_values = pca_df[secondary_col].unique()
        secondary_values = [c for c in SELECTED_F1_CONDITIONS if c in all_values]

    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

    # Plot each genotype-pretraining combination
    for sec_val in secondary_values:
        color = get_secondary_color(sec_val, mode)

        for pretrain_val in pretraining_vals:
            mask = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretrain_val)
            data = pca_df[mask]

            if len(data) < 2:
                continue

            style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])
            label = f"{sec_val} - {style['label']}"

            # Scatter points
            ax.scatter(
                data[pc1],
                data[pc2],
                color=color,
                alpha=style["alpha"],
                s=60,
                marker=style["marker"],
                edgecolors=style["edgecolor"],
                linewidths=style["linewidth"],
                label=label,
                zorder=3,
            )

            # 95% confidence ellipse
            linestyle = "--" if str(pretrain_val).lower() == "n" else "-"
            confidence_ellipse(
                data[pc1].values,
                data[pc2].values,
                ax,
                edgecolor=color,
                linewidth=2,
                linestyle=linestyle,
                alpha=0.3,
            )

            # Mark centroid
            ax.plot(
                data[pc1].mean(),
                data[pc2].mean(),
                marker="X",
                markersize=12,
                color=color,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=4,
            )

    # Annotate with bivariate test results if available
    if bivariate_results:
        y_min, y_max = ax.get_ylim()
        y_text = y_max - (y_max - y_min) * 0.05

        for sec_val in secondary_values:
            if sec_val in bivariate_results:
                result = bivariate_results[sec_val]
                p_val = result["p_value"]

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                # Get x position from genotype centroid
                geno_data = pca_df[pca_df[secondary_col] == sec_val]
                x_pos = geno_data[pc1].mean()

                ax.text(
                    x_pos,
                    y_text,
                    sig,
                    ha="center",
                    va="top",
                    fontsize=12,
                    fontweight="bold",
                    color="red" if p_val < 0.05 else "gray",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
                    zorder=5,
                )

    # Get explained variance
    pc1_idx = int(pc1[2:]) - 1
    pc2_idx = int(pc2[2:]) - 1
    var1 = pca_df.attrs["explained_variance_ratio"][pc1_idx] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][pc2_idx] * 100

    ax.set_xlabel(f"{pc1} ({var1:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({var2:.1f}% variance)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Bivariate Analysis: {pc1} vs {pc2}\n95% Confidence Ellipses (Distribution-Free Test)",
        fontsize=14,
        fontweight="bold",
    )

    # Add origin lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    ax.grid(alpha=0.3, linestyle="--", zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, f"bivariate_{pc1}_vs_{pc2}_ellipses")
    print(f"  Saved: {', '.join(saved_files)}")

    return fig


def plot_bivariate_faceted(pcadf, pc1, pc2, genotype_col, pretraining_col, output_dir, bivariate_results=None):
    """Faceted bivariate plot: one subplot per genotype with ellipses and shift arrow."""
    from matplotlib.patches import Ellipse, FancyArrowPatch
    import matplotlib.transforms as transforms

    genotypes = [g for g in SELECTED_GENOTYPES if g in pcadf[genotype_col].unique()]
    n_genotypes = len(genotypes)

    n_cols = 3
    n_rows = int(np.ceil(n_genotypes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten() if n_genotypes > 1 else [axes]

    def draw_ellipse(ax, x, y, color, linestyle, alpha):
        if len(x) < 3:
            return None, None

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        scale_x = np.sqrt(cov[0, 0]) * 1.96
        scale_y = np.sqrt(cov[1, 1]) * 1.96
        mean_x, mean_y = np.mean(x), np.mean(y)

        ellipse = Ellipse(
            (mean_x, mean_y),
            width=scale_x * 2 * ell_radius_x,
            height=scale_y * 2 * ell_radius_y,
            facecolor="none",
            edgecolor=color,
            linewidth=2.5,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_patch(ellipse)
        return mean_x, mean_y

    all_pc1 = pcadf[pc1].values
    all_pc3 = pcadf[pc2].values
    pc1_range = all_pc1.max() - all_pc1.min()
    pc3_range = all_pc3.max() - all_pc3.min()
    xlim = [all_pc1.min() - 0.1 * pc1_range, all_pc1.max() + 0.1 * pc1_range]
    ylim = [all_pc3.min() - 0.1 * pc3_range, all_pc3.max() + 0.1 * pc3_range]

    for idx, genotype in enumerate(genotypes):
        ax = axes[idx]
        df_geno = pcadf[pcadf[genotype_col] == genotype]

        naive_data = df_geno[df_geno[pretraining_col] == "n"]
        pretrained_data = df_geno[df_geno[pretraining_col] == "y"]

        if len(naive_data) < 2 or len(pretrained_data) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(genotype, fontweight="bold")
            continue

        color_naive = "steelblue"
        color_pretrained = "darkorange"

        ax.scatter(
            naive_data[pc1],
            naive_data[pc2],
            color=color_naive,
            alpha=0.5,
            s=50,
            marker="o",
            edgecolors="black",
            linewidths=1,
            label="Naive",
            zorder=3,
        )

        ax.scatter(
            pretrained_data[pc1],
            pretrained_data[pc2],
            color=color_pretrained,
            alpha=0.7,
            s=50,
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            label="Pretrained",
            zorder=3,
        )

        naive_x, naive_y = draw_ellipse(ax, naive_data[pc1].values, naive_data[pc2].values, color_naive, "--", 0.6)
        pretrained_x, pretrained_y = draw_ellipse(
            ax, pretrained_data[pc1].values, pretrained_data[pc2].values, color_pretrained, "-", 0.8
        )

        if naive_x is not None and pretrained_x is not None:
            arrow = FancyArrowPatch(
                (naive_x, naive_y),
                (pretrained_x, pretrained_y),
                arrowstyle="->",
                mutation_scale=25,
                linewidth=3,
                color="red",
                alpha=0.8,
                zorder=5,
            )
            ax.add_patch(arrow)

            ax.plot(
                naive_x,
                naive_y,
                "o",
                markersize=12,
                color=color_naive,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=4,
            )
            ax.plot(
                pretrained_x,
                pretrained_y,
                "s",
                markersize=12,
                color=color_pretrained,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=4,
            )

            shift_pc1 = pretrained_x - naive_x
            shift_pc3 = pretrained_y - naive_y

            if bivariate_results and genotype in bivariate_results:
                result = bivariate_results[genotype]
                p_val = result["p_value"]
                distance = result.get("observed_distance_residual", result.get("observed_distance", 0))

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                if p_val < 0.05:
                    box_color = "lightcoral"
                    text_color = "darkred"
                else:
                    box_color = "lightgray"
                    text_color = "gray"

                stats_text = f"D={distance:.2f}\np={p_val:.4f} {sig}\n"
                stats_text += f"ΔPC1={shift_pc1:+.2f}\nΔPC3={shift_pc3:+.2f}"

                ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
                    color=text_color,
                    fontweight="bold",
                )

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        title = f"{genotype}\n(n={len(naive_data)}, {len(pretrained_data)})"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--", zorder=0)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    for idx in range(n_genotypes, len(axes)):
        fig.delaxes(axes[idx])

    pc1_var = pcadf.attrs["explained_variance_ratio"][int(pc1[2]) - 1] * 100
    pc2_var = pcadf.attrs["explained_variance_ratio"][int(pc2[2]) - 1] * 100

    fig.text(0.5, 0.02, f"{pc1} ({pc1_var:.1f}% variance)", ha="center", fontsize=13, fontweight="bold")
    fig.text(
        0.02, 0.5, f"{pc2} ({pc2_var:.1f}% variance)", va="center", rotation="vertical", fontsize=13, fontweight="bold"
    )

    plt.suptitle(
        "Bivariate Analysis: PC1 vs PC3\nPretraining Effect per Genotype", fontsize=15, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    saved_files = save_figure_multiple_formats(fig, output_dir, f"bivariate_{pc1}_vs_{pc2}_faceted")
    print(f"  Saved {', '.join(saved_files)}")

    return fig


def plot_bivariate_vector_shift(pcadf, pc1, pc2, genotype_col, pretraining_col, output_dir, bivariate_results=None):
    """Vector-only plot showing naive→pretrained centroid shifts per genotype."""
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 10))

    genotypes = [g for g in SELECTED_GENOTYPES if g in pcadf[genotype_col].unique()]

    ax.arrow(
        0,
        0,
        2,
        -1,
        head_width=0.15,
        head_length=0.2,
        fc="lightgreen",
        ec="darkgreen",
        linewidth=3,
        alpha=0.3,
        label="Expected pretraining signature\n(+PC1, +PC3)",
        zorder=1,
    )

    for genotype in genotypes:
        df_geno = pcadf[pcadf[genotype_col] == genotype]

        naive = df_geno[df_geno[pretraining_col] == "n"]
        pretrained = df_geno[df_geno[pretraining_col] == "y"]

        if len(naive) < 2 or len(pretrained) < 2:
            continue

        naive_centroid = [naive[pc1].mean(), naive[pc2].mean()]
        pretrained_centroid = [pretrained[pc1].mean(), pretrained[pc2].mean()]

        color = get_secondary_color(genotype, ANALYSIS_MODE if ANALYSIS_MODE else "tnt_full")

        ax.plot(
            naive_centroid[0],
            naive_centroid[1],
            "o",
            markersize=14,
            color=color,
            markeredgecolor="black",
            markeredgewidth=2,
            alpha=0.7,
            zorder=3,
        )

        arrow = FancyArrowPatch(
            naive_centroid,
            pretrained_centroid,
            arrowstyle="->",
            mutation_scale=30,
            linewidth=3,
            color=color,
            alpha=0.8,
            zorder=4,
        )
        ax.add_patch(arrow)

        ax.plot(
            pretrained_centroid[0],
            pretrained_centroid[1],
            "s",
            markersize=14,
            color=color,
            markeredgecolor="black",
            markeredgewidth=2.5,
            alpha=0.9,
            zorder=5,
        )

        if bivariate_results and genotype in bivariate_results:
            result = bivariate_results[genotype]
            p_val = result["p_value"]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            label = f"{genotype} {sig}"
            text_color = "red" if p_val < 0.05 else "black"

            ax.annotate(
                label,
                xy=pretrained_centroid,
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold" if p_val < 0.05 else "normal",
                color=text_color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white" if p_val < 0.05 else "lightgray",
                    edgecolor=text_color if p_val < 0.05 else "gray",
                    linewidth=2 if p_val < 0.05 else 1,
                    alpha=0.9,
                ),
                arrowprops=dict(arrowstyle="-", color=text_color, linewidth=1.5, alpha=0.7),
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)

    pc1_var = pcadf.attrs["explained_variance_ratio"][int(pc1[2]) - 1] * 100
    pc2_var = pcadf.attrs["explained_variance_ratio"][int(pc2[2]) - 1] * 100

    ax.set_xlabel(f"{pc1} ({pc1_var:.1f}% variance)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{pc2} ({pc2_var:.1f}% variance)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Bivariate Pretraining Effect: PC1 vs PC3\nArrows show Naive→Pretrained centroid shift",
        fontsize=14,
        fontweight="bold",
    )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Naive centroid",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=2.5,
            label="Pretrained centroid",
        ),
        plt.Line2D(
            [0], [0], linestyle="-", color="gray", linewidth=3, marker=">", markersize=10, label="Pretraining shift"
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=11, framealpha=0.95)

    ax.grid(alpha=0.3, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, f"bivariate_{pc1}_vs_{pc2}_vectors")
    print(f"  Saved {', '.join(saved_files)}")

    return fig


def plot_3d_pca(pca_df, secondary_col, pretraining_col, output_dir, mode):
    """Create 3D scatter plot of first three PCs"""
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns or "PC3" not in pca_df.columns:
        print("  Warning: Not enough PCs for 3D plot")
        return None

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get unique values - preserve order from SELECTED_GENOTYPES/SELECTED_F1_CONDITIONS
    if mode == "tnt_full":
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
            style = PRETRAINING_STYLES.get(str(pretrain_val).lower(), PRETRAINING_STYLES["n"])

            label = f"{sec_val} - {style['label']}"

            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                subset["PC3"],
                color=color,
                alpha=style["alpha"],
                s=SCATTER_SIZE * 2,
                marker=style["marker"],
                edgecolors=style["edgecolor"],
                linewidths=style["linewidth"],
                label=label,
            )

    # Get explained variance
    var1 = pca_df.attrs["explained_variance_ratio"][0] * 100
    var2 = pca_df.attrs["explained_variance_ratio"][1] * 100
    var3 = pca_df.attrs["explained_variance_ratio"][2] * 100

    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_zlabel(f"PC3 ({var3:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_title("PCA: 3D View of First Three Components", fontsize=14, fontweight="bold", pad=20)

    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    saved_files = save_figure_multiple_formats(fig, output_dir, "pca_3d_scatter")
    print(f"  Saved: {', '.join(saved_files)}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PCA Analysis for F1 Behavioral Metrics")
    parser.add_argument(
        "--mode",
        type=str,
        default="tnt_full",
        choices=["control", "tnt_full"],
        help="Analysis mode: control (F1_condition) or tnt_full (genotypes)",
    )
    parser.add_argument("--n-components", type=int, default=3, help="Number of principal components")
    parser.add_argument("--show", action="store_true", help="Display plots instead of just saving")
    parser.add_argument("--metrics", type=str, help="Comma-separated list of metrics (default: predefined list)")
    parser.add_argument(
        "--detect-secondary", action="store_true", help="Print all secondary grouping values found in dataset and exit"
    )

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
        if args.mode == "tnt_full":
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
        metrics = [m.strip() for m in args.metrics.split(",")]
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

    # Perform multivariate statistical analysis
    stat_results = perform_multivariate_statistical_analysis(
        pca_df,
        secondary_col,
        pretraining_col,
        n_components=args.n_components,
        use_permanova=USE_PERMANOVA,
        use_pairwise=USE_PAIRWISE_PERMUTATION,
        use_lmm=USE_LMM_ON_PCS,
        use_mahalanobis=USE_MAHALANOBIS,
        use_mann_whitney=USE_MANN_WHITNEY_ON_PCS,
        use_permutation=USE_PERMUTATION_ON_PCS,
        use_residual_permutation=USE_RESIDUAL_PERMUTATION_ON_PCS,
        n_permutations=N_PERMUTATIONS,
        alpha=ALPHA,
    )

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
        pc_name = f"PC{i+1}"
        plot_pc_boxplot(
            pca_df, pc_name, secondary_col, pretraining_col, output_dir, args.mode, stat_results=stat_results
        )
        plt.close()

    # 4. 2D scatter plots
    print("\n4. Plotting 2D scatter plots...")
    plot_pca_scatter(pca_df, "PC1", "PC2", secondary_col, pretraining_col, output_dir, args.mode)
    plt.close()

    if args.n_components >= 3:
        plot_pca_scatter(pca_df, "PC1", "PC3", secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()
        plot_pca_scatter(pca_df, "PC2", "PC3", secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()

    # 5. 3D scatter plot
    if args.n_components >= 3:
        print("\n5. Plotting 3D scatter plot...")
        plot_3d_pca(pca_df, secondary_col, pretraining_col, output_dir, args.mode)
        plt.close()

    # 6. Bivariate analysis (PC1 + PC3 joint test)
    if args.n_components >= 3:
        print("\n6. Bivariate analysis: PC1 + PC3 joint test...")
        from pca_multivariate_stats import perform_bivariate_residual_permutation

        bivariate_results = perform_bivariate_residual_permutation(
            pca_df, ["PC1", "PC3"], secondary_col, pretraining_col, n_permutations=N_PERMUTATIONS
        )

        # Plot confidence ellipses with bivariate test results
        plot_bivariate_confidence_ellipses(
            pca_df,
            "PC1",
            "PC3",
            secondary_col,
            pretraining_col,
            output_dir,
            args.mode,
            bivariate_results=bivariate_results,
        )
        plt.close()

        # Faceted per-genotype plot
        plot_bivariate_faceted(
            pca_df,
            "PC1",
            "PC3",
            secondary_col,
            pretraining_col,
            output_dir,
            bivariate_results=bivariate_results,
        )
        plt.close()

        # Vector-only shift plot
        plot_bivariate_vector_shift(
            pca_df,
            "PC1",
            "PC3",
            secondary_col,
            pretraining_col,
            output_dir,
            bivariate_results=bivariate_results,
        )
        plt.close()

        # Save bivariate results
        if bivariate_results:
            bivariate_data = []
            for genotype, result in bivariate_results.items():
                bivariate_data.append(
                    {
                        "genotype": genotype,
                        "pcs_tested": ", ".join(result["pcs_tested"]),
                        "observed_distance_residual": result["observed_distance_residual"],
                        "observed_distance_original": result["observed_distance_original"],
                        "p_value_residual": result["p_value_residual"],
                        "p_value_original": result["p_value_original"],
                        "p_value": result["p_value"],
                        "effect_size": result["effect_size"],
                        "n_naive": result["n_naive"],
                        "n_pretrained": result["n_pretrained"],
                    }
                )
            bivariate_df = pd.DataFrame(bivariate_data)
            bivariate_csv = output_dir / "bivariate_PC1_PC3_results.csv"
            bivariate_df.to_csv(bivariate_csv, index=False)
            print(f"  Saved: {bivariate_csv.name}")

    # Save PCA results and loadings
    output_csv = output_dir / "pca_results.csv"
    pca_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}")

    # Save loadings to CSV
    loadings_df = pd.DataFrame(
        pca.components_.T, columns=[f"PC{i+1}" for i in range(args.n_components)], index=valid_metrics
    )
    loadings_csv = output_dir / "pca_loadings.csv"
    loadings_df.to_csv(loadings_csv)
    print(f"  Saved: {loadings_csv.name}")

    # Save statistical results
    print("\nSaving statistical results...")

    # Check if stat_results is valid
    if stat_results is None:
        stat_results = {}
        print("  Warning: No statistical results available")

    # Save PERMANOVA results
    if "permanova" in stat_results:
        permanova_df = pd.DataFrame([stat_results["permanova"]])
        permanova_csv = output_dir / "permanova_results.csv"
        permanova_df.to_csv(permanova_csv, index=False)
        print(f"  Saved: {permanova_csv.name}")

    # Save pairwise permutation results
    if "pairwise_permutation" in stat_results and stat_results["pairwise_permutation"]:
        pairwise_data = []
        for group, result in stat_results["pairwise_permutation"].items():
            pairwise_data.append(
                {
                    "group": group,
                    "mahalanobis_distance": result["statistic"],
                    "p_value": result["p_value"],
                    "cohens_d": result["cohens_d"],
                    "n_permutations": result["n_permutations"],
                }
            )
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_csv = output_dir / "pairwise_permutation_results.csv"
        pairwise_df.to_csv(pairwise_csv, index=False)
        print(f"  Saved: {pairwise_csv.name}")

    # Save LMM results on PCs
    if "lmm_on_pcs" in stat_results:
        lmm_data = []
        for pc_name, pc_result in stat_results["lmm_on_pcs"].items():
            if "error" in pc_result:
                continue

            for group, group_result in pc_result.get("group_effects", {}).items():
                lmm_data.append(
                    {
                        "PC": pc_name,
                        "group": group,
                        "effect_size": group_result["effect_size"],
                        "stderr": group_result["stderr"],
                        "pvalue": group_result["pvalue"],
                        "mean_naive": group_result["mean_naive"],
                        "mean_pretrained": group_result["mean_pretrained"],
                    }
                )

        if lmm_data:
            lmm_df = pd.DataFrame(lmm_data)
            lmm_csv = output_dir / "lmm_pc_results.csv"
            lmm_df.to_csv(lmm_csv, index=False)
            print(f"  Saved: {lmm_csv.name}")

    # Save centroid distances
    if "centroids" in stat_results and stat_results["centroids"]["distances"]:
        centroid_data = []
        for pair, dist_info in stat_results["centroids"]["distances"].items():
            centroid_data.append(
                {
                    "comparison": pair,
                    "mahalanobis_distance": dist_info["mahalanobis_distance"],
                    "euclidean_distance": dist_info["euclidean_distance"],
                }
            )
        centroid_df = pd.DataFrame(centroid_data)
        centroid_csv = output_dir / "centroid_distances.csv"
        centroid_df.to_csv(centroid_csv, index=False)
        print(f"  Saved: {centroid_csv.name}")

    # Save RESIDUAL PERMUTATION results on PCs (PRIMARY METHOD)
    if "residual_permutation_on_pcs" in stat_results and stat_results["residual_permutation_on_pcs"]:
        residual_perm_data = []
        for pc_name, test_results in stat_results["residual_permutation_on_pcs"].items():
            for result in test_results:
                residual_perm_data.append(
                    {
                        "PC": pc_name,
                        "group": result["group"],
                        "observed_diff_residual": result["observed_diff_residual"],
                        "observed_diff_original": result["observed_diff_original"],
                        "cohens_d_residual": result["cohens_d_residual"],
                        "cohens_d_original": result["cohens_d_original"],
                        "p_value_residual": result["p_value_residual"],
                        "p_value_original": result["p_value_original"],
                        "p_value": result["p_value"],
                        "p_value_corrected": result["p_value_corrected"],
                        "n_naive": result["n_naive"],
                        "n_pretrained": result["n_pretrained"],
                    }
                )
        if residual_perm_data:
            residual_perm_df = pd.DataFrame(residual_perm_data)
            residual_perm_csv = output_dir / "residual_permutation_pc_results.csv"
            residual_perm_df.to_csv(residual_perm_csv, index=False)
            print(f"  Saved: {residual_perm_csv.name}")

    # Save Mann-Whitney results on PCs (DEPRECATED)
    if "mann_whitney_on_pcs" in stat_results and stat_results["mann_whitney_on_pcs"]:
        mw_data = []
        for pc_name, test_results in stat_results["mann_whitney_on_pcs"].items():
            for result in test_results:
                mw_data.append(
                    {
                        "PC": pc_name,
                        "group": result["group"],
                        "U_statistic": result["statistic"],
                        "p_value": result["p_value"],
                        "p_value_corrected": result["p_value_corrected"],
                        "rank_biserial": result["rank_biserial"],
                        "median_naive": result["median_naive"],
                        "median_pretrained": result["median_pretrained"],
                        "n_naive": result["n_naive"],
                        "n_pretrained": result["n_pretrained"],
                    }
                )
        if mw_data:
            mw_df = pd.DataFrame(mw_data)
            mw_csv = output_dir / "mann_whitney_pc_results.csv"
            mw_df.to_csv(mw_csv, index=False)
            print(f"  Saved: {mw_csv.name}")

    # Save permutation test results on PCs
    if "permutation_on_pcs" in stat_results and stat_results["permutation_on_pcs"]:
        perm_data = []
        for pc_name, test_results in stat_results["permutation_on_pcs"].items():
            for result in test_results:
                perm_data.append(
                    {
                        "PC": pc_name,
                        "group": result["group"],
                        "observed_diff": result["observed_diff"],
                        "p_value": result["p_value"],
                        "p_value_corrected": result["p_value_corrected"],
                        "cohens_d": result["cohens_d"],
                        "mean_naive": result["mean_naive"],
                        "mean_pretrained": result["mean_pretrained"],
                        "n_naive": result["n_naive"],
                        "n_pretrained": result["n_pretrained"],
                    }
                )
        if perm_data:
            perm_df = pd.DataFrame(perm_data)
            perm_csv = output_dir / "permutation_pc_results.csv"
            perm_df.to_csv(perm_csv, index=False)
            print(f"  Saved: {perm_csv.name}")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(
        f"  - Total variance explained by {args.n_components} PCs: " f"{pca.explained_variance_ratio_.sum()*100:.1f}%"
    )
    print(f"  - Top contributors to PC1: " f"{loadings_df['PC1'].abs().nlargest(3).to_dict()}")
    if args.n_components >= 2:
        print(f"  - Top contributors to PC2: " f"{loadings_df['PC2'].abs().nlargest(3).to_dict()}")


if __name__ == "__main__":
    main()
