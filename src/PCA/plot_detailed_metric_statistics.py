#!/usr/bin/env python3
"""
Generate detailed metric plots using individual metrics instead of PCs.
Uses the same statistical approach as PC analysis but on raw metric values.
Only includes genotypes with high consistency scores (≥80% by default).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import sys
import json
import os
import argparse
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, set_link_color_palette
from pathlib import Path
from scipy.spatial.distance import pdist
import textwrap

warnings.filterwarnings("ignore")

# Try to import seaborn, but provide fallback if not available
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not available - using matplotlib only")


sys.path.append("/home/matthias/ballpushing_utils")
import Config

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use August dataset matching the reproduced original analysis
DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
# Use the exact reproduced analysis (August dataset, Triple test, 20 configs, edge cases disabled)
CONSISTENCY_DIR = "/home/matthias/ballpushing_utils/src/PCA/pca_analysis_results_tailored_20251219_153806/data_files"
# Default metrics path relative to this script for robustness
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics_lists", "final_metrics_for_pca_alt.txt")

# Consistency threshold for inclusion in detailed analysis (fraction 0–1 on Combined_Consistency)
MIN_COMBINED_CONSISTENCY = 0.80  # Only include hits with ≥80% combined consistency


def _parse_args():
    parser = argparse.ArgumentParser(description="Detailed metric statistics for high combined-consistency hits")
    parser.add_argument("--output-dir", default="best_metric_analysis", help="Directory for all outputs")
    parser.add_argument(
        "--consistency-dir", default=CONSISTENCY_DIR, help="Directory containing combined consistency CSVs"
    )
    parser.add_argument("--metrics-path", default=METRICS_PATH, help="Path to metrics list file")
    parser.add_argument("--data-path", default=DATA_PATH, help="Feather dataset with pooled summary data")
    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=MIN_COMBINED_CONSISTENCY,
        help="Minimum Combined_Consistency (0-1) required to include a genotype (default: %(default)s)",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=None,
        help="Optionally cap number of high-consistency genotypes (after filtering) to top-N",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["tailored", "emptysplit", "tnt_pr"],
        default="tailored",
        help="Control selection mode (default: tailored)",
    )
    parser.add_argument(
        "--discrete-effects",
        action="store_true",
        help="Use discrete effect size bins (negligible <0.2, small 0.2-0.5, medium 0.5-1.0, large >1.0) instead of continuous coloring",
    )
    parser.add_argument(
        "--clip-effects",
        type=float,
        default=1.5,
        help="Clip Cohen's d values to ±this threshold (default: 1.5) to prevent extreme values from dominating the color scale. Set to None to disable clipping.",
    )
    return parser.parse_args()


# These globals will be overridden by CLI if provided
OUTPUT_DIR = "best_metric_analysis"
CONTROL_MODE = "tailored"
DISCRETE_EFFECTS = False
CLIP_EFFECTS = 1.5


def _apply_args(args):
    global OUTPUT_DIR, CONSISTENCY_DIR, METRICS_PATH, DATA_PATH, MIN_COMBINED_CONSISTENCY, CONTROL_MODE, DISCRETE_EFFECTS, CLIP_EFFECTS
    OUTPUT_DIR = args.output_dir
    CONSISTENCY_DIR = args.consistency_dir
    METRICS_PATH = args.metrics_path
    DATA_PATH = args.data_path
    MIN_COMBINED_CONSISTENCY = args.combined_threshold
    CONTROL_MODE = args.control_mode
    DISCRETE_EFFECTS = args.discrete_effects
    CLIP_EFFECTS = args.clip_effects
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🎯 Output directory: {OUTPUT_DIR}")
    if DISCRETE_EFFECTS:
        print(f"🎨 Using discrete effect size bins (Cohen's d thresholds: 0.2/0.5/1.0)")
    if CLIP_EFFECTS is not None:
        print(f"✂️  Clipping Cohen's d values to ±{CLIP_EFFECTS} (prevents extreme values from dominating color scale)")

    # Check for consistency files in multiple locations (PCA_Static.py saves to data_files subdirectory)
    proposed_consistency_dir = args.consistency_dir

    # Check if consistency files exist in:
    # 1. Proposed consistency dir (e.g., old static consistency_analysis_with_edges)
    # 2. OUTPUT_DIR/data_files (where PCA_Static.py saves them)
    # 3. OUTPUT_DIR directly (fallback)
    enhanced_in_proposed = os.path.exists(os.path.join(proposed_consistency_dir, "enhanced_consistency_scores.csv"))
    enhanced_in_output_data = os.path.exists(os.path.join(OUTPUT_DIR, "data_files", "enhanced_consistency_scores.csv"))
    enhanced_in_output = os.path.exists(os.path.join(OUTPUT_DIR, "enhanced_consistency_scores.csv"))

    # Priority: OUTPUT_DIR/data_files > proposed_dir > OUTPUT_DIR
    if enhanced_in_output_data:
        CONSISTENCY_DIR = os.path.join(OUTPUT_DIR, "data_files")
        print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
    elif enhanced_in_proposed:
        CONSISTENCY_DIR = proposed_consistency_dir
        print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
    elif enhanced_in_output:
        CONSISTENCY_DIR = OUTPUT_DIR
        print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
    else:
        CONSISTENCY_DIR = proposed_consistency_dir
        print(f"⚠️  No consistency files found, will try: {CONSISTENCY_DIR}")


# ------------------------------------------------------------------
# Brain-region look-ups
# ------------------------------------------------------------------
try:
    nickname_to_brainregion = dict(zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"]))
    color_dict = Config.color_dict
except Exception as e:
    print(f"⚠️  Could not load region mapping from Config: {e}")
    nickname_to_brainregion = {}
    color_dict = {}

# ------------------------------------------------------------------
# Metric name mapping (internal -> informative display names)
# ------------------------------------------------------------------
METRIC_DISPLAY_NAMES = {
    "pulling_ratio": "Proportion pull vs push",
    "distance_ratio": "Dist. ball moved / corridor length",
    "distance_moved": "Dist. ball moved",
    "pulled": "Signif. (>0.3 mm) pulling events (#)",
    "max_event": "Event max. ball displ. (n)",
    "number_of_pauses": "Long pauses (>5s <5px) (#)",
    "first_major_event": "First major (>1.2mm) event(n)",
    "significant_ratio": "Fraction signif. (>0.3 mm) events",
    "max_distance": "Max ball displacement (mm)",
    "chamber_ratio": "Fraction time in chamber",
    "nb_events": "Events (< 2mm fly-ball dist.)(#)",
    "persistence_at_end": "Fraction time near end of corridor",
    "time_chamber_beginning": "Time in chamber first 25% exp. (s)",
    "normalized_velocity": "Normalized walking velocity",
    "first_major_event_time": "First major (>1.2mm) event time (s)",
    "max_event_time": "Max ball displ. time (s)",
    "nb_freeze": "short pauses (>2s <5px) (#)",
    "flailing": "Movement of front legs during contact",
    "velocity_during_interactions": "Fly speed during ball contact (mm/s)",
    "head_pushing_ratio": "Head pushing ratio",
    "fraction_not_facing_ball": "Fraction not facing (>30°) ball in corridor",
    "interaction_persistence": "Avg. duration ball interaction events (s)",
    "chamber_exit_time": "Time of first chamber exit (s)",
    "velocity_trend": "Slope linear fit to fly velocity over time",
}

BRAIN_REGION_DISPLAY_NAMES = {
    "CX": "Central complex neurons",
    "LH": "Lateral horn neurons",
    "Olfaction": "Olfactory sensory neurons & antennal lobe projection neurons",
    "MB": "Mushroom body and MB output neurons",
    "MB extrinsic neurons": "Mushroom body extrinsic neurons",
    "Neuropeptide": "Neuropeptide",
    "Vision": "Visual projection neurons",
}

# Custom ordering for brain regions in grouped heatmap
BRAIN_REGION_ORDER = [
    "Olfaction",
    "MB",
    "MB extrinsic neurons",
    "Neuropeptide",
    "CX",
    "LH",
    "Vision",
]

# Genotypes to display in bold font
BOLD_GENOTYPES = [
    "OR67d",
    "GR63a",
    "IR8a",
    "MB247",
    "LC10bc",
]


def get_display_name(metric_name):
    """Get informative display name for a metric, fallback to original name."""
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def get_brain_region_display_name(region_name):
    """Get informative display name for a brain region, fallback to original name."""
    return BRAIN_REGION_DISPLAY_NAMES.get(region_name, region_name)


# ------------------------------------------------------------------
# Helper functions (same as original)
# ------------------------------------------------------------------
def reorder_by_brain_region(df, nickname_to_region):
    """Group rows by brain region, then by nickname (index)."""
    tmp = df.copy()
    tmp["__region__"] = [nickname_to_region.get(idx, "Unknown") for idx in tmp.index]
    tmp_reset = tmp.reset_index()
    index_col = tmp_reset.columns[0]
    tmp_sorted = tmp_reset.sort_values(
        by=["__region__", index_col],
        ascending=[True, True],
    )
    tmp_sorted = tmp_sorted.set_index(index_col)
    return tmp_sorted.drop(columns="__region__")


def colour_y_ticklabels(ax, nickname_to_region, color_dict):
    """Paint y-tick labels according to the genotype's brain region."""
    for tick in ax.get_yticklabels():
        region = nickname_to_region.get(tick.get_text(), None)
        if region in color_dict:
            tick.set_color(color_dict[region])


def load_consistency_results(threshold: float = MIN_COMBINED_CONSISTENCY, max_hits=None):
    """Load consistency analysis results and return qualifying genotypes.

    Now uses statistical_criteria_comparison.csv with Triple test criterion to get
    the exact reproduced list of 30 hits from the original analysis.

    Parameters
    ----------
    threshold : float
        Minimum consistency (0–1) required to include a genotype (defaults to 0.80).
    max_hits : int | None
        If provided, cap the number of returned genotypes (after filtering) to top-N.
    """
    # NEW APPROACH: Use statistical_criteria_comparison.csv with Triple test
    # This gives us the EXACT 30 hits from the reproduced original analysis
    criteria_file = os.path.join(CONSISTENCY_DIR, "statistical_criteria_comparison.csv")

    if os.path.exists(criteria_file):
        df = pd.read_csv(criteria_file)
        source = "statistical_criteria_comparison.csv"

        # Use Triple test (MW + Perm + Maha all required) - the reproduced criterion
        if "Triple_Pass" in df.columns:
            filtered = df[df["Triple_Pass"] == True].copy()
            print("📊 LOADING EXACT REPRODUCED HITS (Triple Test):")
            print(f"   Source file: {source}")
            print(f"   Criterion: Triple test (MW + Perm + Maha, all required)")
            print(f"   Total genotypes evaluated: {len(df)}")
            print(f"   Hits passing Triple test: {len(filtered)}")

            if not filtered.empty:
                print("\n🏆 TRIPLE TEST HITS (all shown):")
                for _, row in filtered.iterrows():
                    consistency = row.get("Triple_Consistency_%", 0)
                    print(f"   {row['Genotype']:<30}  Consistency={consistency:.1f}%")

            return filtered["Genotype"].tolist()
        else:
            print(f"❌ statistical_criteria_comparison.csv found but missing Triple_Pass column")
            return []

    # FALLBACK: Try old approach with combined consistency files
    preferred_file = os.path.join(CONSISTENCY_DIR, "combined_consistency_ranking.csv")
    fallback_file = os.path.join(CONSISTENCY_DIR, "enhanced_consistency_scores.csv")

    if os.path.exists(preferred_file):
        df = pd.read_csv(preferred_file)
        source = os.path.basename(preferred_file)
    elif os.path.exists(fallback_file):
        df = pd.read_csv(fallback_file)
        source = os.path.basename(fallback_file)
    else:
        print(
            "❌ No consistency file found (expected statistical_criteria_comparison.csv, combined_consistency_ranking.csv, or enhanced_consistency_scores.csv)"
        )
        return []

    # Derive Combined_Consistency if missing and sufficient components exist
    if "Combined_Consistency" not in df.columns:
        # Try multiple naming variations for the optimized consistency column
        if "Overall_Consistency" in df.columns:
            # Check for various naming patterns
            if "Optimized_Only_Consistency" in df.columns:
                optimized_col = "Optimized_Only_Consistency"
            elif "Optimized_Consistency" in df.columns:
                optimized_col = "Optimized_Consistency"
            else:
                optimized_col = None

            if optimized_col:
                print(
                    f"⚠️  Combined_Consistency not present – deriving as mean of Overall_Consistency & {optimized_col}"
                )
                df["Combined_Consistency"] = 0.5 * df["Overall_Consistency"] + 0.5 * df[optimized_col]
            else:
                print(
                    "❌ Could not find Combined_Consistency nor components to derive it – aborting consistency filter"
                )
                return []
        else:
            print("❌ Could not find Combined_Consistency nor components to derive it – aborting consistency filter")
            return []

    # Standardize column names for downstream printing (support both Genotype / genotype variations)
    genotype_col = "Genotype" if "Genotype" in df.columns else ("genotype" if "genotype" in df.columns else None)
    if genotype_col is None:
        print("❌ No genotype column found in consistency file")
        return []

    # Filter
    filtered = df[df["Combined_Consistency"] >= threshold].copy()
    filtered = filtered.sort_values("Combined_Consistency", ascending=False)

    # Load statistical results from best configuration to apply additional filter
    stats_file = os.path.join(CONSISTENCY_DIR, "static_pca_stats_results_allmethods_tailoredctrls.csv")
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        # Apply Permutation + Mahalanobis criterion (matching the compare script)
        perm_maha_filter = stats_df["Permutation_FDR_significant"] & stats_df["Mahalanobis_FDR_significant"]
        perm_maha_genotypes = set(stats_df[perm_maha_filter]["Nickname"].unique())

        # Filter consistency results to only include statistically significant genotypes
        initial_count = len(filtered)
        filtered = filtered[filtered[genotype_col].isin(perm_maha_genotypes)]
        print(f"   📊 Statistical filter (Perm+Maha): {initial_count} → {len(filtered)} genotypes")
    else:
        print(f"   ⚠️  Statistical results file not found, skipping statistical filter")

    if max_hits is not None:
        filtered = filtered.head(max_hits)

    print("📊 CONSISTENCY FILTERING (Combined Consistency + Statistical Significance):")
    print(f"   Source file: {source}")
    print(f"   Total genotypes evaluated: {len(df)}")
    print(f"   Threshold: ≥{threshold:.0%} Combined_Consistency + Perm+Maha criterion")
    print(f"   High-consistency hits: {len(filtered)}")

    if not filtered.empty:
        print("\n🏆 HIGH-CONSISTENCY HITS (top 20 shown):")
        for _, row in filtered.head(20).iterrows():
            hit_count = row.get("Total_Hit_Count", row.get("Hit_Count", "?"))
            total_cfg = row.get("Total_Configs", row.get("Total_Config", "?"))
            print(
                f"   {row[genotype_col]:<30}  Combined={row['Combined_Consistency']:.1%}  (hits {hit_count}/{total_cfg})"
            )
        if len(filtered) > 20:
            print(f"   ... and {len(filtered) - 20} more")

    return filtered[genotype_col].tolist()


def load_metrics_list():
    """Load metrics list from file. If missing, fall back to best configuration metrics."""
    candidate_paths = []
    # 1) User-provided or default (could be absolute)
    candidate_paths.append(METRICS_PATH)
    # 2) If METRICS_PATH was given as a relative path, try relative to SCRIPT_DIR
    if not os.path.isabs(METRICS_PATH):
        candidate_paths.append(os.path.join(SCRIPT_DIR, METRICS_PATH))
    # 3) Standard script-local metric_lists folder
    candidate_paths.append(os.path.join(SCRIPT_DIR, "metric_lists", "final_metrics_for_pca.txt"))
    # 4) Workspace-level metric_lists (one directory up)
    candidate_paths.append(os.path.join(os.path.dirname(SCRIPT_DIR), "metric_lists", "final_metrics_for_pca.txt"))

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(metrics)} metrics from {path}")
            return metrics

    # Fallback: derive metrics from best configuration JSON if available
    configs_path = os.path.join(SCRIPT_DIR, "multi_condition_pca_optimization", "top_configurations.json")
    if os.path.exists(configs_path):
        try:
            with open(configs_path, "r") as f:
                all_configs = json.load(f)
            best_key, best_cfg, best_score = None, None, -1
            for k, cfg in all_configs.items():
                score = cfg.get("best_score", 0)
                if score > best_score:
                    best_key, best_cfg, best_score = k, cfg, score
            if best_cfg and isinstance(best_cfg.get("metrics", None), list):
                metrics = [m for m in best_cfg["metrics"] if isinstance(m, str) and m.strip()]
                print(
                    f"📋 Metrics file not found; using {len(metrics)} metrics from best configuration '{best_key}' (score={best_score:.3f})."
                )
                return metrics
        except Exception as e:
            print(f"⚠️  Could not load metrics from top_configurations.json: {e}")

    print(f"❌ Metrics file not found in expected locations and no configuration fallback available.")
    print("   Checked:")
    for path in candidate_paths:
        print(f"   • {path}")
    return []


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


def apply_simplified_nicknames(genotype_list, nickname_mapping):
    """Apply simplified nicknames to genotype list for visualization"""
    if not nickname_mapping:
        return genotype_list

    simplified = []
    for genotype in genotype_list:
        simplified_name = nickname_mapping.get(genotype, genotype)
        simplified.append(simplified_name)

    return simplified


def bin_cohens_d(cohens_d_value):
    """Bin Cohen's d into discrete categories based on interpretation guidelines.

    Standard interpretation (Cohen, 1988):
    - Negligible: |d| < 0.2
    - Small: 0.2 ≤ |d| < 0.5
    - Medium: 0.5 ≤ |d| < 1.0
    - Large: |d| ≥ 1.0

    Returns a binned value preserving sign for directionality:
    - 0: negligible effect (white)
    - ±0.35: small effect (faint color)
    - ±0.75: medium effect (medium color)
    - ±1.5: large effect (dark color)
    """
    abs_d = abs(cohens_d_value)
    sign = 1 if cohens_d_value >= 0 else -1

    if abs_d < 0.2:
        return 0  # Negligible
    elif abs_d < 0.5:
        return sign * 0.35  # Small effect - represented as 0.35
    elif abs_d < 1.0:
        return sign * 0.75  # Medium effect - represented as 0.75
    else:
        return sign * 1.5  # Large effect - represented as 1.5


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_standard_deviation

    This is a standardized measure of the difference between two groups,
    expressed in standard deviation units. It's scale-invariant and suitable
    for comparing effects across metrics with different units.

    Interpretation:
    - Small effect: |d| ~ 0.2
    - Medium effect: |d| ~ 0.5
    - Large effect: |d| ~ 0.8

    Parameters
    ----------
    group1 : array-like
        Treatment/experimental group values
    group2 : array-like
        Control group values

    Returns
    -------
    float
        Cohen's d effect size (positive = group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    mean_diff = np.mean(group1) - np.mean(group2)

    if pooled_std == 0:
        return 0.0

    return mean_diff / pooled_std


def prepare_data():
    """Load and preprocess data (same as consistency analysis)"""
    print("📊 Loading and preprocessing data...")

    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)

    # Exclude problematic nicknames
    exclude_nicknames = [
        "Ple-Gal4.F a.k.a TH-Gal4",
        "TNTxCS",
        "MB247-Gal4",
    ]  # "854 (OK107-Gal4)", "7362 (C739-Gal4)"]
    dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]

    # Rename columns
    dataset.rename(
        columns={
            "major_event": "first_major_event",
            "major_event_time": "first_major_event_time",
        },
        inplace=True,
    )

    # Convert boolean to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    return dataset


def run_metric_analysis(dataset, metrics_list, high_consistency_hits):
    """Run metric analysis focusing on high-consistency hits"""
    print(f"\n🔬 RUNNING METRIC ANALYSIS")
    print(f"   Metrics: {len(metrics_list)}")
    print(f"   Focus on {len(high_consistency_hits)} high-consistency genotypes")

    # Filter metrics that exist in dataset and handle missing values
    available_metrics = [m for m in metrics_list if m in dataset.columns]
    missing_metrics = len(metrics_list) - len(available_metrics)

    if missing_metrics > 0:
        print(f"   ⚠️  Missing metrics: {missing_metrics}")
        missing_list = [m for m in metrics_list if m not in dataset.columns]
        print(f"   Missing: {missing_list[:10]}{'...' if len(missing_list) > 10 else ''}")

    # Filter out binary metrics that aren't suitable for Mann-Whitney U tests
    binary_metrics = ["has_finished", "has_major", "has_significant"]
    binary_metrics_present = [m for m in available_metrics if m in binary_metrics]
    available_metrics = [m for m in available_metrics if m not in binary_metrics]

    if binary_metrics_present:
        print(f"   🚫 Excluded binary metrics (not suitable for Mann-Whitney U): {binary_metrics_present}")
        print(f"   📊 Remaining metrics for analysis: {len(available_metrics)}")

    na_counts = dataset[available_metrics].isna().sum()
    total_rows = len(dataset)
    missing_percentages = (na_counts / total_rows) * 100

    valid_metrics = [col for col in available_metrics if missing_percentages.loc[col] <= 5.0]
    valid_data = dataset[valid_metrics].copy()
    rows_with_missing = valid_data.isnull().any(axis=1)
    dataset_clean = dataset[~rows_with_missing].copy()

    print(f"   Final data: {dataset_clean.shape[0]} rows, {len(valid_metrics)} metrics")

    # Scale data for correlation analysis but keep raw for statistics
    metric_data = dataset_clean[valid_metrics].to_numpy()
    scaler = RobustScaler()
    metric_data_scaled = scaler.fit_transform(metric_data)

    # Create metric dataframe
    metric_df = pd.DataFrame(metric_data_scaled, columns=valid_metrics)

    # Combine with metadata
    metadata_cols = [col for col in dataset_clean.columns if col not in valid_metrics]
    metric_with_meta = pd.concat(
        [dataset_clean[metadata_cols].reset_index(drop=True), dataset_clean[valid_metrics].reset_index(drop=True)],
        axis=1,
    )

    # Save metric results
    scores_file = os.path.join(OUTPUT_DIR, f"metric_values.csv")
    dataset_clean[valid_metrics].to_csv(scores_file, index=False)

    # Calculate correlation matrix for clustering
    correlation_matrix = dataset_clean[valid_metrics].corr()
    corr_file = os.path.join(OUTPUT_DIR, f"metric_correlations.csv")
    correlation_matrix.to_csv(corr_file)

    with_meta_file = os.path.join(OUTPUT_DIR, f"metrics_with_metadata.feather")
    metric_with_meta.to_feather(with_meta_file)

    print(f"   💾 Saved metric results to {OUTPUT_DIR}")

    # Run statistical analysis focusing on high-consistency hits
    results = []

    # Only analyze the high-consistency genotypes
    analysis_genotypes = [g for g in high_consistency_hits if g in metric_with_meta["Nickname"].values]
    print(f"   🎯 Analyzing {len(analysis_genotypes)} high-consistency genotypes")

    for nickname in analysis_genotypes:
        # Determine which control to use based on CONTROL_MODE
        if CONTROL_MODE == "emptysplit":
            force_control = "Empty-Split"
        elif CONTROL_MODE == "tnt_pr":
            force_control = "TNTxPR"
        else:
            force_control = None  # Use tailored control from split registry

        subset = Config.get_subset_data(metric_with_meta, col="Nickname", value=nickname, force_control=force_control)
        if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
            continue

        control_names = subset["Nickname"].unique()
        control_names = [n for n in control_names if n != nickname]
        if not control_names:
            continue
        control_name = control_names[0]

        # Mann-Whitney U per metric + FDR correction
        # This is the main analysis - no need for multivariate tests since hits are pre-selected
        mannwhitney_pvals = []
        metrics_tested = []
        directions = {}
        effect_sizes = {}  # Store Cohen's d for each metric

        for metric in valid_metrics:
            group_values = subset[subset["Nickname"] == nickname][metric]
            control_values = subset[subset["Nickname"] == control_name][metric]
            if group_values.empty or control_values.empty:
                continue

            stat, pval = mannwhitneyu(group_values, control_values, alternative="two-sided")
            mannwhitney_pvals.append(pval)
            metrics_tested.append(metric)

            # Calculate Cohen's d effect size
            d = cohens_d(group_values, control_values)
            effect_sizes[metric] = d

            # Calculate direction (same as sign of Cohen's d)
            directions[metric] = 1 if d > 0 else -1

        if len(mannwhitney_pvals) > 0:
            rejected, pvals_corr, _, _ = multipletests(mannwhitney_pvals, alpha=0.05, method="fdr_bh")
            significant_metrics = [metrics_tested[i] for i, rej in enumerate(rejected) if rej]
            mannwhitney_any = any(rejected)
        else:
            mannwhitney_any = False
            significant_metrics = []
            pvals_corr = []

        # Build result with metric-specific information (no multivariate tests needed)
        result_dict = {
            "genotype": nickname,
            "control": control_name,
            "MannWhitney_any_metric_significant": mannwhitney_any,
            "MannWhitney_significant_metrics": significant_metrics,
            "num_significant_metrics": len(significant_metrics),
            "significant": mannwhitney_any,  # Use Mann-Whitney results as main criterion
        }

        # Add metric-specific results
        for i, metric in enumerate(metrics_tested):
            if i < len(mannwhitney_pvals) and i < len(pvals_corr):
                result_dict[f"{metric}_pval"] = mannwhitney_pvals[i]
                result_dict[f"{metric}_pval_corrected"] = pvals_corr[i]
                result_dict[f"{metric}_significant"] = metric in significant_metrics
                result_dict[f"{metric}_direction"] = directions.get(metric, 0)
                result_dict[f"{metric}_cohens_d"] = effect_sizes.get(metric, 0.0)

        results.append(result_dict)

    if not results:
        print("   ⚠️  No statistical results generated")
        return pd.DataFrame(), correlation_matrix

    results_df = pd.DataFrame(results)

    # (Removed) Multivariate test FDR correction block (Permutation/Mahalanobis) no longer applicable.

    # Save results
    results_file = os.path.join(OUTPUT_DIR, f"metric_stats_results.csv")
    results_df.to_csv(results_file, index=False)

    print(f"   📊 Found {len(results_df)} genotypes in analysis")
    print(f"   🎯 Significant hits: {len(results_df[results_df['significant']])}")
    print(f"   💾 Results saved to {results_file}")

    return results_df, correlation_matrix


def create_hits_heatmap(
    results_df, correlation_matrix, nickname_to_brainregion, color_dict, nickname_mapping=None, color_mode="effect_size"
):
    """Create detailed heatmap for high-consistency hits (all of them, not just statistically significant)

    Parameters
    ----------
    color_mode : str
        Coloring strategy for tiles:
        - "effect_size" (default): Color intensity based on Cohen's d magnitude (scale-invariant effect size)
        - "pvalue": Color intensity based on statistical significance (p-value thresholds)
    """

    if len(results_df) == 0:
        print("No hits to visualize")
        return None

    # Use ALL results (not just statistically significant ones) and sort by brain region
    all_df = results_df.copy()

    # Apply simplified nicknames for display
    if nickname_mapping:
        all_df["genotype"] = apply_simplified_nicknames(all_df["genotype"].tolist(), nickname_mapping)

    all_df = all_df.set_index("genotype")
    sorted_df = reorder_by_brain_region(all_df, nickname_to_brainregion).reset_index()

    # Get all metrics from correlation matrix
    all_metrics = list(correlation_matrix.columns)

    # Detect metric columns in results (those ending with _significant)
    metric_columns = [
        col
        for col in results_df.columns
        if col.endswith("_significant") and col.replace("_significant", "") in all_metrics
    ]
    metric_names = [col.replace("_significant", "") for col in metric_columns]

    # Create matrices for visualization
    significance_matrix = []
    annotation_matrix = []
    genotype_names = []

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        genotype_names.append(row["genotype"])

        sig_row = []
        annot_row = []

        for metric in metric_names:
            metric_sig_col = f"{metric}_significant"
            metric_pval_col = f"{metric}_pval_corrected"
            metric_dir_col = f"{metric}_direction"
            metric_cohens_col = f"{metric}_cohens_d"

            is_significant = row.get(metric_sig_col, False)
            pval_corrected = row.get(metric_pval_col, np.nan)
            direction = row.get(metric_dir_col, 0)
            cohens_d_value = row.get(metric_cohens_col, 0.0)

            if color_mode == "effect_size":
                # COLOR MODE: Effect Size (Cohen's d)
                # Intensity based on magnitude of effect, capped at |d| = 1.0 for color scale
                # Sign based on direction (positive = higher than control = red, negative = lower = blue)

                if DISCRETE_EFFECTS:
                    # DISCRETE BINNING: Use standard interpretation thresholds
                    # Negligible (<0.2) -> 0, Small (0.2-0.5) -> ±0.35, Medium (0.5-1.0) -> ±0.75, Large (≥1.0) -> ±1.5
                    binned_d = bin_cohens_d(cohens_d_value)
                    sig_row.append(binned_d)
                else:
                    # CONTINUOUS: Cap Cohen's d at ±1.0 for visualization (large effects)
                    # This makes small/medium/large effects visible on a -1 to +1 scale
                    capped_d = np.clip(cohens_d_value, -1.0, 1.0)
                    sig_row.append(capped_d)

                # Annotation: show significance stars only if statistically significant
                if is_significant and not pd.isna(pval_corrected):
                    if pval_corrected < 0.001:
                        annot_row.append("***")
                    elif pval_corrected < 0.01:
                        annot_row.append("**")
                    elif pval_corrected < 0.05:
                        annot_row.append("*")
                    else:
                        annot_row.append("")
                else:
                    annot_row.append("")

            else:  # color_mode == "pvalue"
                # COLOR MODE: P-value significance
                # Binary: significant vs not, with direction for sign
                if is_significant and not pd.isna(pval_corrected):
                    # Create significance code
                    if pval_corrected < 0.001:
                        sig_code = "***"
                    elif pval_corrected < 0.01:
                        sig_code = "**"
                    elif pval_corrected < 0.05:
                        sig_code = "*"
                    else:
                        sig_code = ""

                    sig_row.append(direction)  # 1 for higher than control (red), -1 for lower than control (blue)
                    annot_row.append(sig_code)
                else:
                    sig_row.append(0)  # No significance
                    annot_row.append("")

        significance_matrix.append(sig_row)
        annotation_matrix.append(annot_row)

    if len(significance_matrix) == 0:
        print("No hits matrix to create")
        return None

    # Create DataFrames - genotypes as rows, metrics as columns
    sig_df = pd.DataFrame(significance_matrix, index=sorted_df["genotype"], columns=metric_names)
    annot_df = pd.DataFrame(annotation_matrix, index=sorted_df["genotype"], columns=metric_names)

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(max(14, len(metric_names) * 0.4), max(8, len(genotype_names) * 0.4)))

    # Create custom colormap based on color_mode
    if color_mode == "effect_size":
        # Use continuous RdBu_r colormap for Cohen's d effect sizes
        # Blue (negative) = lower than control, Red (positive) = higher than control
        # Intensity represents magnitude of effect (0 to 1.0 standard deviations)
        import matplotlib.cm as cm

        cmap = cm.get_cmap("RdBu_r")

        # Set color scale limits based on discrete vs continuous mode
        if DISCRETE_EFFECTS:
            # Discrete: binned values are 0, ±0.35, ±0.75, ±1.5
            vmin_val, vmax_val = -1.5, 1.5
        elif CLIP_EFFECTS is not None:
            # Continuous with clipping: use the clip threshold
            vmin_val, vmax_val = -CLIP_EFFECTS, CLIP_EFFECTS
        else:
            # Continuous: default range ±1.0
            vmin_val, vmax_val = -1.0, 1.0
    else:
        # Binary colormap for p-value significance mode
        colors = ["lightblue", "white", "lightcoral"]  # blue for -1 (lower), white for 0, red for +1 (higher)
        cmap = ListedColormap(colors)
        vmin_val, vmax_val = -1, 1

    # Create the heatmap
    if HAS_SEABORN and "sns" in globals():
        if (DISCRETE_EFFECTS or CLIP_EFFECTS is not None) and color_mode == "effect_size":
            # In discrete or clipped mode, create heatmap without annotations first
            # so we can add colored annotations based on background
            sns.heatmap(
                sig_df,
                annot=False,  # Don't add annotations yet
                fmt="",
                cmap=cmap,
                center=0,
                vmin=vmin_val,
                vmax=vmax_val,
                cbar=False,
                ax=ax,
                linewidths=0.5,
                linecolor="gray",
            )

            # Add annotations manually with color based on background intensity
            # For discrete bins: ±0.35 (small), ±0.75 (medium), ±1.5 (large)
            for i, idx in enumerate(sig_df.index):
                for j, col in enumerate(sig_df.columns):
                    value = sig_df.iloc[i, j]  # Use iloc for guaranteed scalar access
                    annot_text = annot_df.iloc[i, j]

                    # Check if annotation exists (not NaN and not empty)
                    # Convert to string first to avoid Series ambiguity
                    annot_str = str(annot_text).strip()
                    if annot_str and annot_str != "nan":
                        # Determine text color based on background value
                        # Use white for medium and large effects (±0.75, ±1.5)
                        abs_value = abs(value)
                        if abs_value >= 0.5:  # Medium or large effect
                            text_color = "white"
                        else:  # Small or negligible effect
                            text_color = "black"

                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            annot_str,
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=text_color,
                        )
        else:
            # Normal mode: use seaborn's built-in annotations
            sns.heatmap(
                sig_df,
                annot=annot_df,
                fmt="",
                cmap=cmap,
                center=0,
                vmin=vmin_val,
                vmax=vmax_val,
                cbar=False,
                ax=ax,
                linewidths=0.5,
                linecolor="gray",
                annot_kws={"fontsize": 8, "fontweight": "bold"},
            )
    else:
        # Fallback using matplotlib imshow
        im = ax.imshow(sig_df.values, cmap=cmap, vmin=vmin_val, vmax=vmax_val, aspect="auto")
        ax.set_xticks(range(len(sig_df.columns)))
        ax.set_yticks(range(len(sig_df.index)))
        ax.set_xticklabels(sig_df.columns)
        ax.set_yticklabels(sig_df.index)

    # Apply informative metric names to x-axis
    display_metric_names = [get_display_name(m) for m in metric_names]
    ax.set_xticklabels(display_metric_names, rotation=45, ha="right")

    # Color y-tick labels by brain region
    colour_y_ticklabels(ax, nickname_to_brainregion, color_dict)

    # Customize the plot
    if color_mode == "effect_size":
        if DISCRETE_EFFECTS:
            title_suffix = "Cohen's d Effect Size (Discrete)"
            color_description = "Effect size bins: Negligible (<0.2), Small (0.2-0.5), Medium (0.5-1.0), Large (≥1.0)"
        else:
            title_suffix = "Cohen's d Effect Size"
            color_description = "Color intensity = Effect magnitude (|d| ≤ 1.0 SD)"
    else:
        title_suffix = "Statistical Significance"
        color_description = "Color = Significance at p < 0.05"

    ax.set_title(
        f"High-Consistency Hits - Detailed Metric Analysis ({title_suffix})\n"
        f"Individual Metrics (≥{MIN_COMBINED_CONSISTENCY:.0%} combined consistency)\n"
        f"{len(genotype_names)} genotypes across {len(metric_names)} metrics",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Genotypes (High-Consistency Hits)", fontsize=12, fontweight="bold")

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Add legend based on color_mode
    if color_mode == "effect_size":
        if DISCRETE_EFFECTS:
            legend_elements = [
                Patch(facecolor="darkred", label="Large Higher (d ≥ 1.0)"),
                Patch(facecolor="indianred", label="Medium Higher (0.5 ≤ d < 1.0)"),
                Patch(facecolor="lightcoral", label="Small Higher (0.2 ≤ d < 0.5)"),
                Patch(facecolor="white", edgecolor="gray", label="Negligible (|d| < 0.2)"),
                Patch(facecolor="lightblue", label="Small Lower (-0.5 < d ≤ -0.2)"),
                Patch(facecolor="steelblue", label="Medium Lower (-1.0 < d ≤ -0.5)"),
                Patch(facecolor="darkblue", label="Large Lower (d ≤ -1.0)"),
                Patch(facecolor="none", label=""),
                Patch(facecolor="none", label=color_description),
                Patch(facecolor="none", label="Stars = Statistical significance:"),
                Patch(facecolor="none", label="* p < 0.05"),
                Patch(facecolor="none", label="** p < 0.01"),
                Patch(facecolor="none", label="*** p < 0.001"),
                Patch(facecolor="none", label=""),
                Patch(facecolor="none", label="Brain Regions:"),
            ]
        else:
            legend_elements = [
                Patch(facecolor="darkred", label="Strong Higher (d > 0.8)"),
                Patch(facecolor="lightcoral", label="Moderate Higher (d ~ 0.5)"),
                Patch(facecolor="white", edgecolor="gray", label="Negligible (d ~ 0)"),
                Patch(facecolor="lightblue", label="Moderate Lower (d ~ -0.5)"),
                Patch(facecolor="darkblue", label="Strong Lower (d < -0.8)"),
                Patch(facecolor="none", label=""),
                Patch(facecolor="none", label=color_description),
                Patch(facecolor="none", label="Stars = Statistical significance:"),
                Patch(facecolor="none", label="* p < 0.05"),
                Patch(facecolor="none", label="** p < 0.01"),
                Patch(facecolor="none", label="*** p < 0.001"),
                Patch(facecolor="none", label=""),
                Patch(facecolor="none", label="Brain Regions:"),
            ]
    else:
        legend_elements = [
            Patch(facecolor="lightcoral", label="Higher than Control"),
            Patch(facecolor="lightblue", label="Lower than Control"),
            Patch(facecolor="white", edgecolor="gray", label="Not Significant"),
            Patch(facecolor="none", label=""),
            Patch(facecolor="none", label="P-value significance:"),
            Patch(facecolor="none", label="* p < 0.05"),
            Patch(facecolor="none", label="** p < 0.01"),
            Patch(facecolor="none", label="*** p < 0.001"),
            Patch(facecolor="none", label=""),
            Patch(facecolor="none", label="Brain Regions:"),
        ]

    # Add brain region colors
    for region, color in color_dict.items():
        legend_elements.append(Patch(facecolor=color, label=region))

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc="upper left")

    plt.tight_layout()

    # Save plots
    mode_suffix = "" if color_mode == "effect_size" else f"_{color_mode}"
    png_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}.png")
    pdf_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"   💾 Detailed heatmap ({color_mode} mode) saved: {png_file} and {pdf_file}")

    return fig


def _collect_metric_columns(results_df, all_metrics):
    """Return sorted metric names present in results_df."""
    metric_signif_cols = [
        c for c in results_df.columns if c.endswith("_significant") and c.replace("_significant", "") in all_metrics
    ]
    metric_names = [c.replace("_significant", "") for c in metric_signif_cols]

    # Sort alphabetically for consistency
    metric_names = sorted(set(metric_names))
    return metric_names


def _build_signed_weighted_matrix(
    results_df,
    all_metrics,
    only_significant_hits=False,  # Changed default to False
    alpha=0.05,
    weight_mode="linear_cap",  # "linear_cap", "neglog10", or "cohens_d"
):
    """
    Build matrix M (n_genotypes x n_metrics) with signed values.

    For weight_mode="cohens_d": Uses Cohen's d effect sizes directly (no normalization)
    For other modes: Sign = metric_direction (+1/-1), Magnitude = significance weight

    For p-value modes, includes gradient coloring for all p-values:
    - p < 0.05: Full intensity (100%) - clearly significant
    - 0.05 <= p < 0.1: Medium intensity (60%) - trending/marginal
    - 0.1 <= p < 0.2: Low intensity (30%) - weak evidence
    - p >= 0.2: Very faint (10%) - minimal evidence
    """
    if only_significant_hits:
        df = results_df[results_df["significant"]].copy()
    else:
        df = results_df.copy()  # Use all genotypes, not just statistically significant ones
    if df.empty:
        return pd.DataFrame(), []

    metric_names = _collect_metric_columns(results_df, all_metrics)
    if not metric_names:
        return pd.DataFrame(), []

    M = pd.DataFrame(0.0, index=df["genotype"].values, columns=metric_names)

    for _, row in df.iterrows():
        for metric in metric_names:
            sig_col = f"{metric}_significant"
            padj_col = f"{metric}_pval_corrected"
            dir_col = f"{metric}_direction"
            cohens_col = f"{metric}_cohens_d"

            if weight_mode == "cohens_d":
                # Use Cohen's d directly - optionally bin into discrete categories or clip
                cohens_d_value = row.get(cohens_col, 0.0)
                if pd.notna(cohens_d_value) and cohens_d_value != 0:
                    if DISCRETE_EFFECTS:
                        # Apply discrete binning based on effect size thresholds
                        M.loc[row["genotype"], metric] = float(bin_cohens_d(cohens_d_value))
                    else:
                        # Use continuous Cohen's d values, optionally clipped
                        if CLIP_EFFECTS is not None:
                            # Clip to ±CLIP_EFFECTS to prevent extreme values from dominating
                            cohens_d_value = np.clip(cohens_d_value, -CLIP_EFFECTS, CLIP_EFFECTS)
                        M.loc[row["genotype"], metric] = float(cohens_d_value)
            else:
                # P-value based weighting modes
                is_sig = bool(row.get(sig_col, False))
                p_adj = row.get(padj_col, np.nan)
                direction = row.get(dir_col, 0)

                # Only process if we have valid p-value and direction
                if pd.notna(p_adj) and direction != 0:
                    if weight_mode == "linear_cap":
                        # Base weight calculation
                        w = max(0.0, 1.0 - (p_adj / alpha))
                        w = min(1.0, w)

                        # Apply intensity scaling based on significance level
                        if is_sig:  # p < 0.05
                            intensity = 1.0  # Full intensity
                        elif p_adj < 0.1:  # 0.05 <= p < 0.1 (trending)
                            intensity = 0.6  # Medium intensity
                        elif p_adj < 0.2:  # 0.1 <= p < 0.2 (weak evidence)
                            intensity = 0.3  # Low intensity
                        else:  # p >= 0.2 (very weak)
                            intensity = 0.1  # Very faint

                        final_weight = w * intensity

                    elif weight_mode == "neglog10":
                        # Pure logarithmic scaling: -log10(p-value)
                        # This naturally makes small p-values dark and large p-values light
                        p_adj = max(float(p_adj), 1e-10)  # Avoid log(0)
                        log_p = -np.log10(p_adj)

                        # Normalize to reasonable range for visualization
                        # p=0.001 → log_p=3, p=0.01 → log_p=2, p=0.05 → log_p=1.3, p=0.1 → log_p=1
                        # Cap at reasonable maximum for very small p-values
                        final_weight = min(log_p / 3.0, 1.0)  # Normalize so p=0.001 gives weight=1.0
                    else:
                        raise ValueError("Unknown weight_mode")

                    M.loc[row["genotype"], metric] = np.sign(direction) * float(final_weight)
    return M, metric_names


def plot_two_way_dendrogram_metrics(
    results_df,
    correlation_matrix,
    OUTPUT_DIR,
    nickname_mapping=None,  # Add nickname mapping parameter
    simplified_to_region=None,  # Add simplified brain region mapping
    only_significant_hits=True,
    alpha=0.05,
    weight_mode="linear_cap",
    color_mode="effect_size",  # New parameter: "effect_size" or "pvalue"
    # clustering choices
    row_metric="euclidean",
    row_linkage="ward",
    col_metric="euclidean",
    col_linkage="ward",
    # dendrogram coloring
    color_threshold_rows="default",
    color_threshold_cols="default",
    row_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
    col_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
    above_threshold_color_rows="C0",
    above_threshold_color_cols="C0",
    # heatmap appearance
    cmap="RdBu_r",  # Use continuous colormap for gradient effect
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.4,
    linecolor="gray",
    cbar_label=None,  # Will be set based on color_mode if None
    # layout
    fig_size=(20, 16),  # Increased height from 12 to 16 for longer metric labels
    # label styling
    row_label_fontsize=9,
    col_label_fontsize=9,
    metric_label_rotation=45,
    truncate_row_labels=None,
    wrap_labels=True,
    annotate=False,
    despine=True,
):
    """
    Custom 2-way dendrogram where:
    - Rows (genotypes) are clustered using the signed p-weight matrix
    - Columns (metrics) are clustered by correlation distance
    """
    import numpy as np

    # Set labels and titles based on color_mode
    if cbar_label is None:
        if color_mode == "effect_size":
            cbar_label = "Cohen's d Effect Size (Blue: lower, Red: higher)"
        else:
            cbar_label = "Signed -log10(p-value)"

    if color_mode == "effect_size":
        clustering_description = "Genotypes by effect size, Metrics by correlation"
        # Use Cohen's d directly
        actual_weight_mode = "cohens_d"
    else:
        clustering_description = "Genotypes by p-values, Metrics by correlation"
        actual_weight_mode = weight_mode

    # 1) Build matrix M (genotypes x metrics)
    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df,
        all_metrics,
        only_significant_hits=only_significant_hits,
        alpha=alpha,
        weight_mode=actual_weight_mode,
    )
    if M.empty:
        print("No data available to build clustering matrix.")
        return None

    # Set vmin/vmax based on actual data range
    if color_mode == "effect_size":
        # Use the maximum absolute Cohen's d value to set symmetric color scale
        if DISCRETE_EFFECTS:
            # Discrete mode: use fixed bins
            vmin = -1.5  # Large negative effect
            vmax = 1.5  # Large positive effect
            print(f"   📊 Using discrete effect size bins (±1.5 scale)")
        elif CLIP_EFFECTS is not None:
            # Clipping mode: use the clip threshold
            vmin = -CLIP_EFFECTS
            vmax = CLIP_EFFECTS
            print(f"   ✂️  Cohen's d clipped to: [{vmin:.2f}, {vmax:.2f}]")
        else:
            # Continuous mode: use actual data range
            max_abs_d = M.abs().max().max()
            vmin = -max_abs_d
            vmax = max_abs_d
            print(f"   📊 Cohen's d range: [{vmin:.2f}, {vmax:.2f}]")

    # Apply simplified nicknames to matrix index for visualization
    # Keep mapping from simplified to original names for p-value lookup
    original_to_simplified = {}
    simplified_to_original = {}
    if nickname_mapping:
        original_names = M.index.tolist()
        simplified_genotypes = apply_simplified_nicknames(original_names, nickname_mapping)
        for orig, simp in zip(original_names, simplified_genotypes):
            original_to_simplified[orig] = simp
            simplified_to_original[simp] = orig
        M = M.set_axis(simplified_genotypes, axis=0)  # Use set_axis to rename index
    else:
        # No mapping, use identity
        for name in M.index:
            simplified_to_original[name] = name

    # 2) Linkages
    # Row linkage from the signed p-weight matrix
    row_Z = linkage(pdist(M.values, metric=row_metric), method=row_linkage) if M.shape[0] > 1 else None

    # Column linkage from correlation matrix (convert to distance)
    # Use correlation-based distance: distance = 1 - |correlation|
    corr_subset = correlation_matrix.loc[metric_names, metric_names]

    # Handle NaN values in correlation matrix
    corr_subset = corr_subset.fillna(0.0)  # Replace NaN with 0 correlation

    # Convert correlation to distance matrix
    distance_matrix = 1 - np.abs(corr_subset.values)

    # Ensure diagonal is 0 (distance from metric to itself)
    np.fill_diagonal(distance_matrix, 0.0)

    # Check for any remaining non-finite values
    if not np.all(np.isfinite(distance_matrix)):
        print("⚠️  Non-finite values in distance matrix, replacing with 1.0")
        distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1.0)

    # Extract upper triangle to get condensed distance matrix
    # squareform expects the upper triangle in condensed form
    n = distance_matrix.shape[0]
    col_distances = distance_matrix[np.triu_indices(n, k=1)]

    # Final check for finite values in condensed distance matrix
    if not np.all(np.isfinite(col_distances)):
        print("⚠️  Non-finite values in condensed distance matrix, replacing with 1.0")
        col_distances = np.where(np.isfinite(col_distances), col_distances, 1.0)

    col_Z = linkage(col_distances, method=col_linkage) if len(metric_names) > 1 else None

    # 3) BALANCED GridSpec layout - MORE space for metric columns
    plt.style.use("default")
    fig = plt.figure(figsize=fig_size)
    # Create a layout where metric dendrogram sits above the heatmap,
    # the heatmap is in the middle row, and metric labels are below the heatmap.
    # This keeps the dendrogram visually closer to the tiles and places labels like other plots.
    gs = gridspec.GridSpec(
        3,
        4,
        width_ratios=[1.4, 1.2, 8.0, 0.3],
        height_ratios=[0.8, 5.6, 0.6],  # reduce bottom padding so labels sit closer to heatmap
        wspace=0.04,
        hspace=0.06,  # slightly reduce vertical spacing to keep dendrogram close
    )

    # Create axes: top dendrogram, heatmap row (with left dendro and nicknames), and bottom metric labels
    ax_top_dendro = fig.add_subplot(gs[0, 2])
    ax_hm = fig.add_subplot(gs[1, 2])
    ax_left_dendro = fig.add_subplot(gs[1, 0])
    ax_nicknames = fig.add_subplot(gs[1, 1])
    ax_metric_labels = fig.add_subplot(gs[2, 2])
    ax_cbar = fig.add_subplot(gs[1, 3])

    # 4) Top dendrogram (horizontal) - metrics
    if col_Z is not None and M.shape[1] > 1:
        set_link_color_palette(list(col_palette))
        dg_col = dendrogram(
            col_Z,
            orientation="top",
            color_threshold=(None if color_threshold_cols == "default" else color_threshold_cols),
            above_threshold_color=above_threshold_color_cols,
            no_labels=True,
            ax=ax_top_dendro,
        )
        col_order_idx = dg_col["leaves"]
        col_labels_ordered = [metric_names[i] for i in col_order_idx]
    else:
        col_order_idx = list(range(M.shape[1]))
        col_labels_ordered = list(M.columns)
        dg_col = None
        ax_top_dendro.axis("off")

    # Save canonical metric order used by this PCA plotting routine so other scripts
    # can import a stable ordering. Save to src/PCA/metrics_lists/canonical_metrics_order.txt
    try:
        canonical_path = Path(__file__).parent / "metrics_lists" / "canonical_metrics_order.txt"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with canonical_path.open("w") as fh:
            for m in col_labels_ordered:
                fh.write(f"{m}\n")
        print(f"   💾 Saved canonical metric order to: {canonical_path}")
    except Exception as e:
        print(f"⚠️  Could not save canonical metric order: {e}")

    # 5) Metric labels: place below heatmap using tick positions derived from dendrogram
    metric_positions = []  # Initialize to ensure it's available later
    if dg_col is not None:
        # Position metric labels to align with dendrogram leaves
        leaf_positions = []
        for i in range(len(col_labels_ordered)):
            leaf_x = (i * 10) + 5
            leaf_positions.append(leaf_x)

        max_pos = max(leaf_positions) if leaf_positions else 1
        metric_positions = np.array([pos / max_pos for pos in leaf_positions]) * 0.975

        # We'll align ticks to the heatmap after the heatmap is drawn
    else:
        # No dendrogram: keep metric label axis but populate later if needed
        pass

    # 6) Left dendrogram (vertical) - genotypes
    if row_Z is not None and M.shape[0] > 1:
        set_link_color_palette(list(row_palette))
        dg_row = dendrogram(
            row_Z,
            orientation="left",
            color_threshold=(None if color_threshold_rows == "default" else color_threshold_rows),
            above_threshold_color=above_threshold_color_rows,
            no_labels=True,
            ax=ax_left_dendro,
        )
        row_order_idx = dg_row["leaves"]
        row_labels_ordered = [M.index[i] for i in row_order_idx]
    else:
        row_order_idx = list(range(M.shape[0]))
        row_labels_ordered = list(M.index)
        ax_left_dendro.axis("off")

    # 7) Process row labels with wrapping
    if wrap_labels:
        import textwrap

        max_width = 30
        row_labels_display = ["\n".join(textwrap.wrap(lbl, max_width)) for lbl in row_labels_ordered]
    else:

        def _truncate(s, n):
            if truncate_row_labels is None or n is None:
                return s
            return (s[: n - 1] + "…") if isinstance(s, str) and len(s) > n else s

        row_labels_display = [_truncate(lbl, truncate_row_labels) for lbl in row_labels_ordered]

    # POTENTIAL FIX: Reverse the row labels to match heatmap orientation
    # Seaborn heatmaps show first row at TOP, but dendrograms might expect first row at BOTTOM
    row_labels_display = row_labels_display[::-1]  # Reverse the list

    # 8) Create heatmap
    M_ord = M.iloc[row_order_idx, col_order_idx]

    if HAS_SEABORN and "sns" in globals():
        sns.heatmap(
            M_ord,
            ax=ax_hm,
            cmap=plt.get_cmap("RdBu_r"),
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=linewidths,
            linecolor=linecolor,
            square=False,
            xticklabels=False,
            yticklabels=False,
            annot=False if not annotate else None,
        )
    else:
        # Fallback using matplotlib imshow
        im = ax_hm.imshow(M_ord.values, cmap=plt.get_cmap("RdBu_r"), vmin=vmin, vmax=vmax, aspect="auto")
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])

    # Add significance stars to tiles
    # Use original genotype names for p-value lookup
    for i in range(M_ord.shape[0]):  # rows (genotypes)
        for j in range(M_ord.shape[1]):  # columns (metrics)
            genotype_display = M_ord.index[i]  # This is the simplified name
            metric = M_ord.columns[j]

            # Get original genotype name for lookup in results_df
            genotype_original = simplified_to_original.get(genotype_display, genotype_display)

            # Get p-value and check significance
            # Look up using original genotype name
            genotype_row = results_df[results_df["genotype"] == genotype_original]
            if genotype_row.empty:
                continue

            pval_col = f"{metric}_pval_corrected"
            pval = genotype_row.iloc[0].get(pval_col, np.nan)

            if pd.notna(pval):
                # Determine significance level
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    stars = None

                if stars:
                    # Determine text color based on background in discrete or clipped mode
                    if DISCRETE_EFFECTS or CLIP_EFFECTS is not None:
                        # Get the cell value to determine background darkness
                        cell_value = M_ord.iloc[i, j]
                        abs_value = abs(cell_value)
                        # White text for medium/large effects (≥0.5), black for small/negligible
                        text_color = "white" if abs_value >= 0.5 else "black"
                    else:
                        text_color = "black"

                    # Add text annotation
                    ax_hm.text(
                        j + 0.5,
                        i + 0.5,
                        stars,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

    # 9) Nicknames
    # Align bottom metric label axis to heatmap columns so text ends line up with column ticks
    try:
        num_cols = M_ord.shape[1]
        positions = np.arange(num_cols) + 0.5
        # Set heatmap ticks at column centers and show labels directly on heatmap
        try:
            ax_hm.set_xticks(positions)
            display_labels = [get_display_name(m) for m in col_labels_ordered]
            ax_hm.set_xticklabels(display_labels, fontsize=col_label_fontsize)
        except Exception:
            pass
        # Rotate and right-align the labels so their ends match the column ticks
        try:
            plt.setp(ax_metric_labels.get_xticklabels(), rotation=45, ha="right")
        except Exception:
            for lbl in ax_metric_labels.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")

        # Increase padding so rotated labels don't overlap heatmap
        ax_metric_labels.xaxis.set_tick_params(pad=8)

        # Rotate labels on heatmap and hide separate label axis
        try:
            plt.setp(ax_hm.get_xticklabels(), rotation=45, ha="right")
        except Exception:
            for lbl in ax_hm.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")

        # Hide the separate metric label axis (we place labels on heatmap)
        try:
            ax_metric_labels.set_visible(False)
        except Exception:
            pass
    except Exception:
        pass
    ax_nicknames.axis("off")
    heatmap_ylim = ax_hm.get_ylim()
    ax_nicknames.set_ylim(heatmap_ylim)
    y_positions = np.linspace(heatmap_ylim[0] - 0.5, heatmap_ylim[1] + 0.5, len(row_labels_display))

    # DEBUG: Check label/position alignment
    print(f"🔍 GENOTYPE LABEL ALIGNMENT DEBUG:")
    print(f"🔍   heatmap_ylim: {heatmap_ylim}")
    print(f"🔍   y_positions: {y_positions[:3]} ... {y_positions[-3:]}")
    print(f"🔍   row_labels_display[:3]: {row_labels_display[:3]}")
    print(f"🔍   M_ord.index[:3]: {list(M_ord.index[:3])}")

    for y_pos, nickname in zip(y_positions, row_labels_display):
        ax_nicknames.text(
            0.95,
            y_pos,
            nickname,
            ha="right",
            va="center",
            fontsize=row_label_fontsize,
            transform=ax_nicknames.transData,
        )

    # Add continuous guide lines from dendrogram leaves to heatmap columns
    if dg_col is not None and len(col_labels_ordered) > 0:
        # Draw continuous vertical guide lines using figure coordinates
        fig_width, fig_height = fig.get_size_inches()

        for i, metric_pos in enumerate(metric_positions):
            # Calculate x position in figure coordinates
            # Convert from dendrogram data coordinates to figure coordinates
            dendro_x_data = metric_pos * ax_top_dendro.get_xlim()[1]
            dendro_xlim = ax_top_dendro.get_xlim()
            dendro_pos = ax_top_dendro.get_position()

            # Normalize x position within dendrogram axis
            x_norm_dendro = dendro_x_data / (dendro_xlim[1] - dendro_xlim[0])

            # Convert to figure coordinates
            x_fig = dendro_pos.x0 + x_norm_dendro * dendro_pos.width

            # Get y coordinates for the line (from top of dendrogram to bottom of heatmap)
            y_top = dendro_pos.y1  # Top of dendrogram
            hm_pos = ax_hm.get_position()
            y_bottom = hm_pos.y1  # Top of heatmap (we'll draw from dendro bottom to hm top)

            # Draw the guide line in figure coordinates
            from matplotlib.lines import Line2D

            line = Line2D(
                [x_fig, x_fig], [y_bottom, y_top], color="lightgray", linestyle=":", alpha=0.5, linewidth=1.0, zorder=0
            )
            fig.add_artist(line)

    # 10) Colorbar with p-value interpretation
    from matplotlib.colors import Normalize
    import numpy as np

    norm = Normalize(vmin=vmin, vmax=vmax)
    pos = ax_cbar.get_position()
    new_height = pos.height * 0.35
    new_y = pos.y0 + (pos.height - new_height) / 2
    ax_cbar.set_position((pos.x0, new_y, pos.width, new_height))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)

    # Set label based on color_mode
    cbar.set_label(cbar_label, fontsize=8)

    # Create custom tick labels based on color_mode
    try:
        if DISCRETE_EFFECTS or CLIP_EFFECTS is not None:
            thresh = CLIP_EFFECTS if CLIP_EFFECTS is not None else 1.5
            lower = -thresh
            upper = thresh

            # Use rule-of-thumb ticks: ±1.0, ±0.5 and 0, plus clipped endpoints
            base_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
            ticks = [lower] + base_ticks + [upper]
            ticks = sorted(set(ticks))

            # Build tick labels, marking endpoints as clipped
            tick_labels = []
            for t in ticks:
                if np.isclose(t, lower):
                    tick_labels.append(f"≤ {lower:.2f}")
                elif np.isclose(t, upper):
                    tick_labels.append(f"≥ {upper:.2f}")
                else:
                    if color_mode == "effect_size":
                        tick_labels.append(f"{t:.1f}")
                    else:
                        # Convert to p-value equivalent for pvalue mode
                        if abs(t) < 1e-10:
                            tick_labels.append("~1.0")
                        else:
                            p_equiv = 10 ** (-abs(t) * 3)
                            if p_equiv >= 0.01:
                                tick_labels.append(f"{p_equiv:.2f}")
                            elif p_equiv >= 0.001:
                                tick_labels.append(f"{p_equiv:.3f}")
                            else:
                                tick_labels.append(f"{p_equiv:.1e}")

            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        else:
            tick_positions = np.linspace(vmin, vmax, 5)
            if color_mode == "effect_size":
                tick_labels = [f"{pos:.1f}" for pos in tick_positions]
            else:
                tick_labels = []
                for pos in tick_positions:
                    if abs(pos) < 1e-10:
                        tick_labels.append("~1.0")
                    else:
                        p_equiv = 10 ** (-abs(pos) * 3)
                        if p_equiv >= 0.01:
                            tick_labels.append(f"{p_equiv:.2f}")
                        elif p_equiv >= 0.001:
                            tick_labels.append(f"{p_equiv:.3f}")
                        else:
                            tick_labels.append(f"{p_equiv:.1e}")

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
    except Exception:
        # fallback to default linear ticks
        tick_positions = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(tick_positions)
        if color_mode == "effect_size":
            cbar.set_ticklabels([f"{pos:.1f}" for pos in tick_positions])
        cbar.ax.tick_params(labelsize=7)

    cbar.ax.tick_params(labelsize=7)

    # 11) Clean up dendrogram axes
    for ax in [ax_top_dendro, ax_left_dendro]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 12) Color nickname labels by brain region
    try:
        # Use the simplified brain region mapping if available
        brain_region_mapping = (
            simplified_to_region if simplified_to_region else globals().get("nickname_to_brainregion", {})
        )
        color_dict = globals().get("color_dict", {})

        print(f"🎨 Brain region coloring applied to {len(ax_nicknames.texts)} labels")

        if brain_region_mapping and color_dict:
            for i, text_obj in enumerate(ax_nicknames.texts):
                if i < len(row_labels_display):
                    # Get the simplified nickname from the display label (remove line wrapping)
                    display_label = row_labels_display[i]
                    simplified_nickname = display_label.replace("\n", " ")
                    region = brain_region_mapping.get(simplified_nickname, None)
                    if region in color_dict:
                        text_obj.set_color(color_dict[region])
    except Exception as e:
        print(f"🎨 Error in brain region coloring: {e}")
        pass

    # 13) Final styling
    if despine:
        if HAS_SEABORN:
            sns.despine(ax=ax_hm, top=True, right=True, left=False, bottom=False)
        else:
            # Manual despining
            ax_hm.spines["top"].set_visible(False)
            ax_hm.spines["right"].set_visible(False)

    fig.suptitle(
        f"Metric Analysis - Two-way Dendrogram\n({clustering_description})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 14) Save files
    png = os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.png")
    pdf = os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # 15) Save matrix and orders
    mat_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_matrix.csv")
    row_order_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_row_order.csv")
    col_order_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_col_order.csv")
    M.to_csv(mat_csv)
    pd.Series(row_labels_ordered, name="genotype").to_csv(row_order_csv, index=False)
    pd.Series(col_labels_ordered, name="metric").to_csv(col_order_csv, index=False)

    if row_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"metric_two_way_row_linkage.npy"), row_Z)
    if col_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"metric_two_way_col_linkage.npy"), col_Z)

    print(f"   💾 Saved two-way metric dendrogram: {png} / {pdf}")
    print(f"   💾 Matrix CSV: {mat_csv}")
    print(f"   💾 Row/Col orders: {row_order_csv} / {col_order_csv}")

    return {
        "matrix": M,
        "row_Z": row_Z,
        "col_Z": col_Z,
        "row_order": row_labels_ordered,
        "col_order": col_labels_ordered,
    }


def plot_simple_metric_heatmap(
    results_df,
    correlation_matrix,
    OUTPUT_DIR,
    nickname_mapping=None,
    simplified_to_region=None,
    only_significant_hits=False,
    alpha=0.05,
    weight_mode="neglog10",
    color_mode="effect_size",  # New parameter: "effect_size" or "pvalue"
    cbar_label=None,  # Will be set based on color_mode if None
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.4,
    linecolor="gray",
    fig_size=(16, 12),
    row_label_fontsize=9,
    col_label_fontsize=9,
    metric_label_rotation=45,
):
    """
    Create a simple heatmap (no dendrograms) with genotypes sorted by brain region
    and metrics sorted by correlation similarity.
    """
    import numpy as np

    # Determine actual weight_mode based on color_mode
    if color_mode == "effect_size":
        actual_weight_mode = "cohens_d"
    else:
        actual_weight_mode = weight_mode

    # 1) Build matrix M (genotypes x metrics)
    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df,
        all_metrics,
        only_significant_hits=only_significant_hits,
        alpha=alpha,
        weight_mode=actual_weight_mode,
    )
    if M.empty:
        print("   ⚠️  No data available to build heatmap matrix.")
        return None

    # Set colorbar label and vmin/vmax based on color_mode
    if cbar_label is None:
        if color_mode == "effect_size":
            cbar_label = "Cohen's d Effect Size"
        else:
            cbar_label = "Signed -log10(p-value)"

    if color_mode == "effect_size":
        # Use the maximum absolute Cohen's d value to set symmetric color scale
        if DISCRETE_EFFECTS:
            # Discrete mode: use fixed bins
            vmin = -1.5  # Large negative effect
            vmax = 1.5  # Large positive effect
            print(f"   📊 Using discrete effect size bins (±1.5 scale)")
        elif CLIP_EFFECTS is not None:
            # Clipping mode: use the clip threshold
            vmin = -CLIP_EFFECTS
            vmax = CLIP_EFFECTS
            print(f"   ✂️  Cohen's d clipped to: [{vmin:.2f}, {vmax:.2f}]")
        else:
            # Continuous mode: use actual data range
            max_abs_d = M.abs().max().max()
            vmin = -max_abs_d
            vmax = max_abs_d
            print(f"   📊 Cohen's d range: [{vmin:.2f}, {vmax:.2f}]")

    # Apply simplified nicknames to matrix index for visualization
    # Keep mapping from simplified to original names for p-value lookup
    original_to_simplified = {}
    simplified_to_original = {}
    if nickname_mapping:
        original_names = M.index.tolist()
        simplified_genotypes = apply_simplified_nicknames(original_names, nickname_mapping)
        for orig, simp in zip(original_names, simplified_genotypes):
            original_to_simplified[orig] = simp
            simplified_to_original[simp] = orig
        M = M.set_axis(simplified_genotypes, axis=0)
    else:
        # No mapping, use identity
        for name in M.index:
            simplified_to_original[name] = name

    # 2) Sort genotypes by brain region
    brain_region_mapping = simplified_to_region if simplified_to_region else {}

    # Create a DataFrame for sorting
    genotype_df = pd.DataFrame(
        {"genotype": M.index, "brain_region": [brain_region_mapping.get(g, "Unknown") for g in M.index]}
    )

    # Sort by brain region, then by genotype name
    genotype_df_sorted = genotype_df.sort_values(["brain_region", "genotype"])
    row_order = genotype_df_sorted["genotype"].tolist()

    # 3) Sort metrics by correlation similarity (hierarchical clustering)
    corr_subset = correlation_matrix.loc[metric_names, metric_names]
    corr_subset = corr_subset.fillna(0.0)

    # Convert correlation to distance and cluster
    distance_matrix = 1 - np.abs(corr_subset.values)
    np.fill_diagonal(distance_matrix, 0.0)

    if not np.all(np.isfinite(distance_matrix)):
        distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1.0)

    n = distance_matrix.shape[0]
    if n > 1:
        col_distances = distance_matrix[np.triu_indices(n, k=1)]
        if not np.all(np.isfinite(col_distances)):
            col_distances = np.where(np.isfinite(col_distances), col_distances, 1.0)

        col_Z = linkage(col_distances, method="ward")
        col_order_idx = leaves_list(col_Z)
        col_order = [metric_names[i] for i in col_order_idx]
    else:
        col_order = metric_names

    # 4) Reorder matrix
    M_ordered = M.loc[row_order, col_order]

    # 5) Create the plot
    plt.style.use("default")
    fig, (ax_main, ax_cbar) = plt.subplots(
        1, 2, figsize=fig_size, gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05}
    )

    # 6) Create heatmap
    if HAS_SEABORN and "sns" in globals():
        sns.heatmap(
            M_ordered,
            ax=ax_main,
            cmap=plt.get_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=linewidths,
            linecolor=linecolor,
            square=False,
            xticklabels=True,
            yticklabels=True,
            annot=False,
        )
    else:
        # Fallback using matplotlib imshow
        im = ax_main.imshow(M_ordered.values, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, aspect="auto")
        ax_main.set_xticks(range(len(M_ordered.columns)))
        ax_main.set_yticks(range(len(M_ordered.index)))
        ax_main.set_xticklabels(M_ordered.columns)
        ax_main.set_yticklabels(M_ordered.index)

    # Add significance stars to tiles
    # Use original genotype names for p-value lookup
    for i in range(M_ordered.shape[0]):  # rows (genotypes)
        for j in range(M_ordered.shape[1]):  # columns (metrics)
            genotype_display = M_ordered.index[i]  # This is the simplified name
            metric = M_ordered.columns[j]

            # Get original genotype name for lookup in results_df
            genotype_original = simplified_to_original.get(genotype_display, genotype_display)

            # Get p-value and check significance
            # Look up using original genotype name
            genotype_row = results_df[results_df["genotype"] == genotype_original]
            if genotype_row.empty:
                continue

            pval_col = f"{metric}_pval_corrected"
            pval = genotype_row.iloc[0].get(pval_col, np.nan)

            if pd.notna(pval):
                # Determine significance level
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    stars = None

                if stars:
                    # Determine text color based on background in discrete or clipped mode
                    if DISCRETE_EFFECTS or CLIP_EFFECTS is not None:
                        # Get the cell value to determine background darkness
                        cell_value = M_ordered.iloc[i, j]
                        abs_value = abs(cell_value)
                        # White text for medium/large effects (≥0.5), black for small/negligible
                        text_color = "white" if abs_value >= 0.5 else "black"
                    else:
                        text_color = "black"

                    # Add text annotation
                    ax_main.text(
                        j + 0.5,
                        i + 0.5,
                        stars,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                        fontweight="bold",
                    )

    # 7) Customize labels and appearance with informative metric names
    current_xlabels = [label.get_text() for label in ax_main.get_xticklabels()]
    display_xlabels = [get_display_name(m) for m in current_xlabels]
    ax_main.set_xticklabels(display_xlabels, rotation=metric_label_rotation, ha="right", fontsize=col_label_fontsize)
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=row_label_fontsize)

    # 8) Color genotype labels by brain region
    try:
        color_dict = globals().get("color_dict", {})
        if brain_region_mapping and color_dict:
            for tick in ax_main.get_yticklabels():
                genotype = tick.get_text()
                region = brain_region_mapping.get(genotype, None)
                if region in color_dict:
                    tick.set_color(color_dict[region])
    except Exception as e:
        print(f"🎨 Error in brain region coloring: {e}")
        pass

    # 9) Add colorbar
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)), cax=ax_cbar)

    # Set p-value oriented label and ticks
    cbar.set_label("p-value (Blue: lower, Red: higher)", fontsize=10)

    # Create custom tick labels showing p-value equivalents
    tick_positions = np.linspace(vmin, vmax, 5)
    tick_labels = []
    for pos in tick_positions:
        if abs(pos) < 1e-10:  # Very close to zero
            tick_labels.append("~1.0")
        else:
            # Convert back to p-value equivalent
            p_equiv = 10 ** (-abs(pos) * 3)
            if p_equiv >= 0.01:
                tick_labels.append(f"{p_equiv:.2f}")
            elif p_equiv >= 0.001:
                tick_labels.append(f"{p_equiv:.3f}")
            else:
                tick_labels.append(f"{p_equiv:.1e}")

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=9)

    # 10) Set title and labels
    ax_main.set_title(
        f"Metric Analysis - Simple Heatmap\n" f"(Genotypes sorted by brain region, Metrics by correlation)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_main.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Genotypes (sorted by brain region)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # 11) Save files
    png = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap.png")
    pdf = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # 12) Save matrix and orders
    simple_mat_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_matrix.csv")
    simple_row_order_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_row_order.csv")
    simple_col_order_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_col_order.csv")
    M_ordered.to_csv(simple_mat_csv)
    pd.Series(row_order, name="genotype").to_csv(simple_row_order_csv, index=False)
    pd.Series(col_order, name="metric").to_csv(simple_col_order_csv, index=False)

    print(f"   💾 Saved simple metric heatmap: {png} / {pdf}")
    print(f"   💾 Matrix CSV: {simple_mat_csv}")
    print(f"   � Row/Col orders: {simple_row_order_csv} / {simple_col_order_csv}")

    return {
        "matrix": M_ordered,
        "row_order": row_order,
        "col_order": col_order,
    }


def plot_brain_region_grouped_heatmap(
    results_df,
    correlation_matrix,
    OUTPUT_DIR,
    nickname_mapping=None,
    simplified_to_region=None,
    only_significant_hits=False,
    alpha=0.05,
    fig_size=(22, 16),
    row_label_fontsize=9,
    col_label_fontsize=9,
    metric_label_rotation=45,
):
    """
    Create a publication-style heatmap with:
    - Separate heatmap per brain region with spacing
    - Metrics clustered by correlation
    - Cohen's d coloring
    - Darkness-adapted significance asterisks
    - Full informative metric names
    - Single shared colorbar
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    import numpy as np
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # Build matrix using Cohen's d
    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df,
        all_metrics,
        only_significant_hits=only_significant_hits,
        alpha=alpha,
        weight_mode="cohens_d",
    )
    if M.empty:
        print("   ⚠️  No data available to build heatmap matrix.")
        return None

    # Set vmin/vmax based on clipping/discrete settings
    if DISCRETE_EFFECTS:
        vmin, vmax = -1.5, 1.5
        print(f"   📊 Using discrete effect size bins (±1.5 scale)")
    elif CLIP_EFFECTS is not None:
        vmin, vmax = -CLIP_EFFECTS, CLIP_EFFECTS
        print(f"   ✂️  Cohen's d clipped to: [{vmin:.2f}, {vmax:.2f}]")
    else:
        max_abs_d = M.abs().max().max()
        vmin, vmax = -max_abs_d, max_abs_d
        print(f"   📊 Cohen's d range: [{vmin:.2f}, {vmax:.2f}]")

    # Apply simplified nicknames
    original_to_simplified = {}
    if nickname_mapping:
        for orig_name in M.index:
            simplified = nickname_mapping.get(orig_name, orig_name)
            original_to_simplified[orig_name] = simplified
        M.index = [original_to_simplified[name] for name in M.index]

    simplified_to_original = {v: k for k, v in original_to_simplified.items()}

    # Group genotypes by brain region
    if simplified_to_region is None:
        simplified_to_region = {}

    genotypes_by_region = {}
    for genotype in M.index:
        region = simplified_to_region.get(genotype, "Unknown")
        if region not in genotypes_by_region:
            genotypes_by_region[region] = []
        genotypes_by_region[region].append(genotype)

    # Sort regions by custom order (not alphabetically)
    # Use BRAIN_REGION_ORDER, append any regions not in the list at the end
    sorted_regions = []
    for region in BRAIN_REGION_ORDER:
        if region in genotypes_by_region:
            sorted_regions.append(region)
    # Add any regions not in BRAIN_REGION_ORDER
    for region in genotypes_by_region.keys():
        if region not in sorted_regions:
            sorted_regions.append(region)

    # Cluster metrics by correlation (same order for all subplots)
    # Use same approach as dendrogram with NaN handling
    corr_subset = correlation_matrix.loc[metric_names, metric_names]

    # Handle NaN values in correlation matrix
    corr_subset = corr_subset.fillna(0.0)  # Replace NaN with 0 correlation

    # Convert correlation to distance matrix
    distance_matrix = 1 - np.abs(corr_subset.values)

    # Ensure diagonal is 0 (distance from metric to itself)
    np.fill_diagonal(distance_matrix, 0.0)

    # Check for any remaining non-finite values
    if not np.all(np.isfinite(distance_matrix)):
        print("⚠️  Non-finite values in distance matrix, replacing with 1.0")
        distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1.0)

    # Extract upper triangle to get condensed distance matrix
    n = distance_matrix.shape[0]
    col_distances = distance_matrix[np.triu_indices(n, k=1)]

    # Final check for finite values in condensed distance matrix
    if not np.all(np.isfinite(col_distances)):
        print("⚠️  Non-finite values in condensed distance matrix, replacing with 1.0")
        col_distances = np.where(np.isfinite(col_distances), col_distances, 1.0)

    col_Z = linkage(col_distances, method="ward") if len(metric_names) > 1 else None
    col_order_idx = leaves_list(col_Z) if col_Z is not None else list(range(len(metric_names)))
    col_order = [metric_names[i] for i in col_order_idx]

    # Get informative metric names for display
    display_metric_names = [get_display_name(m) for m in col_order]

    # Calculate height ratios for GridSpec (proportional to number of genotypes in each region)
    num_regions = len(sorted_regions)
    height_ratios = [len(genotypes_by_region[region]) for region in sorted_regions]

    # Create plot with GridSpec - one row per brain region + colorbar column
    plt.style.use("default")
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(
        num_regions,
        2,
        height_ratios=height_ratios,
        width_ratios=[20, 1],  # Main plot area and colorbar
        hspace=0.3,  # Spacing between brain regions
        wspace=0.05,
    )

    # Store axes for later use
    axes = []

    # Create one heatmap per brain region
    for region_idx, region in enumerate(sorted_regions):
        ax = fig.add_subplot(gs[region_idx, 0])
        axes.append(ax)

        # Get genotypes for this region
        region_genotypes = sorted(genotypes_by_region[region])
        M_region = M.loc[region_genotypes, col_order]

        # Create heatmap for this region
        if HAS_SEABORN and "sns" in globals():
            import seaborn as sns

            sns.heatmap(
                M_region,
                ax=ax,
                cmap=plt.get_cmap("RdBu_r"),
                vmin=vmin,
                vmax=vmax,
                cbar=False,  # We'll add a single colorbar later
                linewidths=0.5,
                linecolor="lightgray",
                square=False,
                xticklabels=False if region_idx < num_regions - 1 else display_metric_names,  # Only show on bottom
                yticklabels=True,
                annot=False,
            )
        else:
            im = ax.imshow(M_region.values, cmap=plt.get_cmap("RdBu_r"), vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(col_order)))
            ax.set_yticks(range(len(region_genotypes)))
            if region_idx == num_regions - 1:  # Only show on bottom
                ax.set_xticklabels(display_metric_names)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(region_genotypes)

        # Add significance stars with adaptive colors
        for i, genotype in enumerate(region_genotypes):
            for j, metric in enumerate(col_order):
                genotype_original = simplified_to_original.get(genotype, genotype)
                genotype_row = results_df[results_df["genotype"] == genotype_original]
                if genotype_row.empty:
                    continue

                pval_col = f"{metric}_pval_corrected"
                pval = genotype_row.iloc[0].get(pval_col, np.nan)

                if pd.notna(pval):
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                    else:
                        stars = None

                    if stars:
                        # Adaptive text color based on background
                        if DISCRETE_EFFECTS or CLIP_EFFECTS is not None:
                            cell_value = M_region.iloc[i, j]
                            abs_value = abs(cell_value)
                            text_color = "white" if abs_value >= 0.5 else "black"
                        else:
                            text_color = "black"

                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            stars,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=7,
                            fontweight="bold",
                        )

        # Customize y-axis labels with bold for specific genotypes
        ytick_labels = ax.get_yticklabels()
        for label in ytick_labels:
            label.set_rotation(0)
            label.set_fontsize(row_label_fontsize)
            # Check if this genotype should be bold
            genotype_name = label.get_text()
            if genotype_name in BOLD_GENOTYPES:
                label.set_fontweight("bold")

        # Color genotype labels by brain region
        try:
            colour_y_ticklabels(ax, simplified_to_region, color_dict)
        except Exception as e:
            print(f"   ⚠️  Could not color labels for {region}: {e}")

        # Add region title at top-left above heatmap
        region_color = color_dict.get(region, "black")
        region_display_name = get_brain_region_display_name(region)
        ax.text(
            0,  # Align with left edge of heatmap
            -0.5,
            region_display_name,
            fontsize=11,
            fontweight="bold",
            color=region_color,
            ha="left",
            va="top",
            transform=ax.transData,
        )

    # Customize x-axis labels on bottom subplot only
    if axes:
        axes[-1].set_xticklabels(
            axes[-1].get_xticklabels(), rotation=metric_label_rotation, ha="right", fontsize=col_label_fontsize
        )
        axes[-1].set_xlabel("Metrics", fontsize=12, fontweight="bold")

    # Add smaller colorbar on the right (centered, about 1/4 height)
    # Calculate middle rows to center the colorbar
    cbar_start = num_regions // 4
    cbar_end = cbar_start + max(1, num_regions // 4)
    ax_cbar = fig.add_subplot(gs[cbar_start:cbar_end, 1])  # Centered, smaller colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")),
        cax=ax_cbar,
        orientation="vertical",
    )
    cbar.set_label("Cohen's d Effect Size", fontsize=12, fontweight="bold")

    # If we are using clipping or discrete bins, use published rule-of-thumb ticks
    try:
        if DISCRETE_EFFECTS or CLIP_EFFECTS is not None:
            thresh = CLIP_EFFECTS if CLIP_EFFECTS is not None else 1.5
            lower = -thresh
            upper = thresh

            # Preferred tick positions (rule-of-thumb): 0.5, 1.0 (omit 0.2)
            base_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
            # Ensure endpoints (clipped) are included and unique
            ticks = [lower] + base_ticks + [upper]
            # Remove duplicates and sort
            ticks = sorted(set(ticks))

            # Apply ticks and custom labels where endpoints are marked as clipped
            cbar.set_ticks(ticks)
            ticklabels = []
            for t in ticks:
                if np.isclose(t, lower):
                    ticklabels.append(f"≤ {lower:.2f}")
                elif np.isclose(t, upper):
                    ticklabels.append(f"≥ {upper:.2f}")
                else:
                    ticklabels.append(f"{t:.2f}")
            cbar.set_ticklabels(ticklabels)
    except Exception:
        # If anything goes wrong, leave default ticks
        print("   ⚠️  Could not set clipped colorbar tick labels.")
        pass

    # Overall title
    fig.suptitle(
        "Metric Analysis by Brain Region\n(Separate panels per region, metrics clustered by correlation)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for title

    # Save
    png = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.png")
    pdf = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    print(f"   💾 Saved brain region grouped heatmap: {png} / {pdf}")

    return {
        "matrix": M,
        "col_order": col_order,
        "regions": sorted_regions,
        "genotypes_by_region": genotypes_by_region,
    }


def main():
    # Parse CLI arguments once here (avoid side-effects if imported)
    args = _parse_args()
    _apply_args(args)
    """Main analysis function"""
    print("DETAILED METRIC ANALYSIS")
    print("=" * 50)
    print(f"Minimum combined consistency threshold: {MIN_COMBINED_CONSISTENCY:.0%}")
    print("=" * 50)
    print(
        f"Control mode: {'Tailored controls (split-based)' if CONTROL_MODE == 'tailored' else 'Empty-Split (universal control)'}"
    )

    # Step 1: Load high-consistency hits
    high_consistency_hits = load_consistency_results(threshold=MIN_COMBINED_CONSISTENCY, max_hits=args.max_hits)
    if not high_consistency_hits:
        print("❌ No high-consistency hits found. Cannot proceed.")
        return

    # Step 2: Load metrics list
    metrics_list = load_metrics_list()
    if not metrics_list:
        print("❌ No metrics loaded. Cannot proceed.")
        return

    # Step 3: Prepare data
    dataset = prepare_data()

    # Step 4: Run metric analysis
    results_df, correlation_matrix = run_metric_analysis(dataset, metrics_list, high_consistency_hits)

    if results_df.empty:
        print("❌ No results generated from metric analysis.")
        return

    # Step 5: Load nickname mapping for visualization
    nickname_mapping, simplified_to_region = load_nickname_mapping()

    # Step 6: Create detailed visualizations
    print(f"\n🎨 CREATING DETAILED VISUALIZATIONS")
    print(f"🎨 Logarithmic p-value coloring:")
    print(f"   • Dark: p ≈ 0.001 (-log10 = 3.0)")
    print(f"   • Medium-dark: p ≈ 0.01 (-log10 = 2.0)")
    print(f"   • Medium: p ≈ 0.05 (-log10 = 1.3)")
    print(f"   • Light: p ≈ 0.1 (-log10 = 1.0)")
    print(f"   • Very light: p ≥ 0.2 (-log10 ≤ 0.7)")

    # Create hits heatmap
    create_hits_heatmap(results_df, correlation_matrix, simplified_to_region, color_dict, nickname_mapping)

    # Create simple heatmap (no dendrograms)
    print(f"\n🎨 Creating simple heatmap (no dendrograms)...")
    plot_simple_metric_heatmap(
        results_df,
        correlation_matrix,
        OUTPUT_DIR,
        nickname_mapping,
        simplified_to_region,
        only_significant_hits=False,
        alpha=0.05,
        color_mode="effect_size",  # Use effect size mode
        weight_mode="neglog10",
        fig_size=(18, 14),
    )

    # Create two-way dendrogram
    print(f"\n🎨 Creating two-way dendrogram...")
    plot_two_way_dendrogram_metrics(
        results_df,
        correlation_matrix,
        OUTPUT_DIR,
        nickname_mapping,  # Pass nickname mapping
        simplified_to_region,  # Pass simplified brain region mapping
        only_significant_hits=False,  # Show all high-consistency hits, not just statistically significant
        alpha=0.05,
        weight_mode="neglog10",  # Use logarithmic p-value scaling
        color_mode="effect_size",  # Use effect size mode
        row_metric="euclidean",
        row_linkage="ward",
        col_metric="euclidean",
        col_linkage="ward",
        fig_size=(24, 16),  # Much larger figure for better metric name spacing
        annotate=False,
    )

    # Create brain region grouped heatmap (publication style)
    print(f"\n🎨 Creating brain region grouped heatmap (publication style)...")
    plot_brain_region_grouped_heatmap(
        results_df,
        correlation_matrix,
        OUTPUT_DIR,
        nickname_mapping=nickname_mapping,
        simplified_to_region=simplified_to_region,
        only_significant_hits=False,
        alpha=0.05,
        fig_size=(20, 14),
    )

    # Summary statistics
    total_hits = len(results_df)  # All high-consistency genotypes
    total_statistically_significant = len(results_df[results_df["significant"]])
    total_metrics_significant = results_df["num_significant_metrics"].sum()

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   High-consistency genotypes analyzed: {len(high_consistency_hits)}")
    print(f"   All genotypes displayed: {total_hits}")
    print(f"   Statistically significant genotypes: {total_statistically_significant}")
    print(f"   Total significant metrics across all genotypes: {total_metrics_significant}")
    print(
        f"   Average significant metrics per genotype: {total_metrics_significant/total_hits:.1f}"
        if total_hits > 0
        else "   No genotypes to analyze"
    )

    # Create summary file
    summary_file = os.path.join(OUTPUT_DIR, "metric_analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("DETAILED METRIC ANALYSIS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Combined consistency threshold: ≥{MIN_COMBINED_CONSISTENCY:.0%}\n")
        f.write(f"High-consistency hits identified: {len(high_consistency_hits)}\n")
        f.write(f"Metrics analyzed: {len(metrics_list)}\n\n")
        f.write("RESULTS:\n")
        f.write(f"All genotypes displayed: {total_hits}\n")
        f.write(f"Statistically significant genotypes: {total_statistically_significant}\n")
        f.write(f"Total significant metrics: {total_metrics_significant}\n")
        f.write(
            f"Average metrics per genotype: {total_metrics_significant/total_hits:.1f}\n"
            if total_hits > 0
            else "No genotypes analyzed\n"
        )

        f.write("\nALL HIGH-CONSISTENCY GENOTYPES:\n")
        for _, row in results_df.iterrows():
            f.write(f"  {row['genotype']}: {row['num_significant_metrics']} significant metrics")
            if bool(row["significant"]):
                f.write(" (statistically significant)\n")
            else:
                f.write("\n")

    print(f"\n✅ DETAILED METRIC ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {OUTPUT_DIR}")
    print(f"📊 Generated plots:")
    print(f"   • Simple heatmap (brain region sorted): metric_simple_heatmap.png/pdf")
    print(f"   • Two-way dendrogram (clustered): metric_two_way_dendrogram.png/pdf")
    print(f"   • Brain region grouped heatmap (publication style): metric_brain_region_grouped_heatmap.png/pdf")


if __name__ == "__main__":
    main()
