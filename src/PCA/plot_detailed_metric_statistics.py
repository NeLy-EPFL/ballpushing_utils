#!/usr/bin/env python3
"""
Generate detailed metric plots using individual metrics instead of PCs.
Uses the same statistical approach as PC analysis but on raw metric values.
Only includes genotypes with high consistency scores (≥80% by default).
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

# Rebuild font cache if needed
fm._load_fontmanager(try_read_cache=False)


# Configure for Illustrator-editable text with Arial font
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts = editable text in Illustrator
plt.rcParams["ps.fonttype"] = 42  # Also for EPS files
plt.rcParams["svg.fonttype"] = "none"  # Embed fonts as text, not paths
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

import warnings
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# matplotlib.use("module://mplcairo.base")
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
CONSISTENCY_DIR = "/home/matthias/ballpushing_utils/src/PCA/pca_analysis_results_tailored_20251219_163028/data_files"
# Default metrics path relative to this script for robustness
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics_lists", "final_metrics_for_pca_alt.txt")

# Consistency threshold for inclusion in detailed analysis (fraction 0–1 on Combined_Consistency)
MIN_COMBINED_CONSISTENCY = 0.80  # Only include hits with ≥80% combined consistency


def permutation_test_1d(group1, group2, n_permutations=10000, random_state=None):
    """Univariate permutation test for a single dimension using mean difference"""
    rng = np.random.default_rng(random_state)
    observed = np.abs(group1.mean() - group2.mean())
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(combined))
        perm_combined = combined[perm_idx]
        perm_group1 = perm_combined[:n1]
        perm_group2 = perm_combined[n1:]
        stat = np.abs(perm_group1.mean() - perm_group2.mean())
        if stat >= observed:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return pval


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrap confidence interval for mean(group1) - mean(group2)."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_diffs[i] = np.mean(sample1) - np.mean(sample2)

    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)

    return float(lower), float(upper)


def _parse_args():
    parser = argparse.ArgumentParser(description="Detailed metric statistics for high combined-consistency hits")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for all outputs")
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
        "--use-mannwhitney-per-pc",
        dest="use_permutation_per_pc",
        action="store_false",
        default=True,
        help="Use Mann-Whitney U instead of permutation test for per-metric significance testing (default: permutation test)",
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
OUTPUT_DIR = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Detailed_metrics_statistics"
CONTROL_MODE = "tailored"
USE_PERMUTATION_PER_PC = True
DISCRETE_EFFECTS = False
CLIP_EFFECTS = 1.5


def _apply_args(args):
    global OUTPUT_DIR, CONSISTENCY_DIR, METRICS_PATH, DATA_PATH, MIN_COMBINED_CONSISTENCY, CONTROL_MODE, USE_PERMUTATION_PER_PC, DISCRETE_EFFECTS, CLIP_EFFECTS
    OUTPUT_DIR = args.output_dir
    CONSISTENCY_DIR = args.consistency_dir
    METRICS_PATH = args.metrics_path
    DATA_PATH = args.data_path
    MIN_COMBINED_CONSISTENCY = args.combined_threshold
    CONTROL_MODE = args.control_mode
    USE_PERMUTATION_PER_PC = args.use_permutation_per_pc
    DISCRETE_EFFECTS = args.discrete_effects
    CLIP_EFFECTS = args.clip_effects
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🎯 Output directory: {OUTPUT_DIR}")
    if DISCRETE_EFFECTS:
        print(f"🎨 Using discrete effect size bins (Cohen's d thresholds: 0.2/0.5/1.0)")
    if CLIP_EFFECTS is not None:
        print(f"✂️  Clipping Cohen's d values to ±{CLIP_EFFECTS} (prevents extreme values from dominating color scale)")
    if USE_PERMUTATION_PER_PC:
        print(f"🔀 Per-metric testing: Permutation test (10000 permutations + FDR, default)")
    else:
        print(f"📊 Per-metric testing: Mann-Whitney U (+ FDR, optional override)")

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

# Thematic metric groups for brain-region grouped heatmap (defines column ordering and bracket annotations)
# Each entry is (group_label, [metric_internal_names...])
METRIC_GROUPS = [
    (
        "Push efficiency",
        ["pulling_ratio", "distance_ratio", "distance_moved", "pulled"],
    ),
    (
        "Ball contact events",
        [
            "max_event",
            "nb_events",
            "first_major_event",
            "first_major_event_time",
            "max_event_time",
            "max_distance",
            "significant_ratio",
            "interaction_persistence",
        ],
    ),
    (
        "Kinematics",
        [
            "velocity_trend",
            "normalized_velocity",
            "fraction_not_facing_ball",
            "velocity_during_interactions",
            "flailing",
            "head_pushing_ratio",
        ],
    ),
    (
        "Pauses & durations",
        [
            "persistence_at_end",
            "time_chamber_beginning",
            "chamber_ratio",
            "chamber_exit_time",
            "number_of_pauses",
            "nb_freeze",
        ],
    ),
]

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
    if "Nickname" in dataset_clean.columns:
        plot_nicknames = set(high_consistency_hits)
        filtered_for_plot = dataset_clean[dataset_clean["Nickname"].isin(plot_nicknames)]
        nickname_counts = filtered_for_plot["Nickname"].value_counts()
        if not nickname_counts.empty:
            print("   Sample size per Nickname (min-max, plotted): " f"{nickname_counts.min()}-{nickname_counts.max()}")
            low_sample_nicknames = nickname_counts[nickname_counts < 12]
            if not low_sample_nicknames.empty:
                low_sample_list = ", ".join([f"{name} ({count})" for name, count in low_sample_nicknames.items()])
                print(f"   ⚠️  Nicknames with <12 samples: {low_sample_list}")

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
    print(f"   🔬 Computing statistical tests FRESH from raw data (not using pre-computed results)")
    if USE_PERMUTATION_PER_PC:
        print(f"   📊 Per-metric test: Permutation (10000 permutations + FDR correction, default)")
    else:
        print(f"   📊 Per-metric test: Mann-Whitney U (+ FDR correction, optional override)")
    print(f"   📐 Computing Cohen's d effect sizes for all {len(valid_metrics)} metrics")
    print(f"   🎯 Computing bootstrap CIs for mean differences and % changes (n=10000)")
    print()

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

        # Per-metric testing (Mann-Whitney U or Permutation test) + FDR correction
        # IMPORTANT: Tests are computed FRESH from raw data for each genotype-metric pair
        # This works regardless of which test was used in PCA_Static.py for hit selection
        ppc_pvals = []
        metrics_tested = []
        directions = {}
        effect_sizes = {}  # Store Cohen's d for each metric
        metric_stats = {}  # Store per-metric descriptive/CI stats

        for metric in valid_metrics:
            group_values = subset[subset["Nickname"] == nickname][metric]
            control_values = subset[subset["Nickname"] == control_name][metric]
            if group_values.empty or control_values.empty:
                continue

            group_arr = group_values.values.astype(float)
            control_arr = control_values.values.astype(float)

            genotype_n = len(group_arr)
            control_n = len(control_arr)
            genotype_mean = float(np.mean(group_arr))
            control_mean = float(np.mean(control_arr))
            genotype_median = float(np.median(group_arr))
            control_median = float(np.median(control_arr))

            mean_diff = genotype_mean - control_mean
            median_diff = genotype_median - control_median

            ci_lower, ci_upper = bootstrap_ci_difference(
                group_arr,
                control_arr,
                n_bootstrap=10000,
                ci=95,
                random_state=42,
            )

            if control_mean != 0:
                pct_change = (mean_diff / control_mean) * 100
                pct_ci_lower = (ci_lower / control_mean) * 100
                pct_ci_upper = (ci_upper / control_mean) * 100
            else:
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan

            # Compute test fresh from raw data (not pre-computed)
            if USE_PERMUTATION_PER_PC:
                pval = permutation_test_1d(group_arr, control_arr, random_state=42)
            else:
                stat, pval = mannwhitneyu(group_values, control_values, alternative="two-sided")

            ppc_pvals.append(pval)
            metrics_tested.append(metric)

            # Calculate Cohen's d effect size
            d = cohens_d(group_arr, control_arr)
            effect_sizes[metric] = d

            # Calculate direction (same as sign of Cohen's d)
            directions[metric] = 1 if d > 0 else -1

            metric_stats[metric] = {
                "genotype_n": genotype_n,
                "control_n": control_n,
                "genotype_mean": genotype_mean,
                "control_mean": control_mean,
                "mean_diff": mean_diff,
                "genotype_median": genotype_median,
                "control_median": control_median,
                "median_diff": median_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_change": pct_change,
                "pct_ci_lower": pct_ci_lower,
                "pct_ci_upper": pct_ci_upper,
                "n_bootstrap": 10000,
            }

        if len(ppc_pvals) > 0:
            rejected, pvals_corr, _, _ = multipletests(ppc_pvals, alpha=0.05, method="fdr_bh")
            significant_metrics = [metrics_tested[i] for i, rej in enumerate(rejected) if rej]
            ppc_any = any(rejected)
        else:
            ppc_any = False
            significant_metrics = []
            pvals_corr = []

        # Build result with metric-specific information (no multivariate tests needed)
        result_dict = {
            "genotype": nickname,
            "control": control_name,
            "MannWhitney_any_metric_significant": ppc_any,
            "MannWhitney_significant_metrics": significant_metrics,
            "num_significant_metrics": len(significant_metrics),
            "significant": ppc_any,  # Use per-metric test results as main criterion
        }

        # Add metric-specific results
        for i, metric in enumerate(metrics_tested):
            if i < len(ppc_pvals) and i < len(pvals_corr):
                result_dict[f"{metric}_pval"] = ppc_pvals[i]
                result_dict[f"{metric}_pval_corrected"] = pvals_corr[i]
                result_dict[f"{metric}_significant"] = metric in significant_metrics
                result_dict[f"{metric}_direction"] = directions.get(metric, 0)
                result_dict[f"{metric}_cohens_d"] = effect_sizes.get(metric, 0.0)
                stats = metric_stats.get(metric, {})
                result_dict[f"{metric}_genotype_n"] = stats.get("genotype_n", np.nan)
                result_dict[f"{metric}_control_n"] = stats.get("control_n", np.nan)
                result_dict[f"{metric}_genotype_mean"] = stats.get("genotype_mean", np.nan)
                result_dict[f"{metric}_control_mean"] = stats.get("control_mean", np.nan)
                result_dict[f"{metric}_mean_diff"] = stats.get("mean_diff", np.nan)
                result_dict[f"{metric}_genotype_median"] = stats.get("genotype_median", np.nan)
                result_dict[f"{metric}_control_median"] = stats.get("control_median", np.nan)
                result_dict[f"{metric}_median_diff"] = stats.get("median_diff", np.nan)
                result_dict[f"{metric}_ci_lower"] = stats.get("ci_lower", np.nan)
                result_dict[f"{metric}_ci_upper"] = stats.get("ci_upper", np.nan)
                result_dict[f"{metric}_pct_change"] = stats.get("pct_change", np.nan)
                result_dict[f"{metric}_pct_ci_lower"] = stats.get("pct_ci_lower", np.nan)
                result_dict[f"{metric}_pct_ci_upper"] = stats.get("pct_ci_upper", np.nan)
                result_dict[f"{metric}_n_bootstrap"] = stats.get("n_bootstrap", np.nan)

        results.append(result_dict)

    if not results:
        print("   ⚠️  No statistical results generated")
        return pd.DataFrame(), correlation_matrix

    results_df = pd.DataFrame(results)

    # (Removed) Multivariate test FDR correction block (Permutation/Mahalanobis) no longer applicable.

    # Save results
    test_method = "permutation" if USE_PERMUTATION_PER_PC else "mannwhitney"
    results_file = os.path.join(OUTPUT_DIR, f"metric_stats_results_{test_method}.csv")
    results_df.to_csv(results_file, index=False)

    print(f"\n   ✅ Statistical testing complete!")
    print(f"   📊 Genotypes analyzed: {len(results_df)}")
    print(f"   🎯 Genotypes with ≥1 significant metric: {len(results_df[results_df['significant']])}")
    print(f"   🧪 Test method: {'Permutation (10000 perms)' if USE_PERMUTATION_PER_PC else 'Mann-Whitney U'}")
    print(f"   💾 Results saved to {results_file}")

    return results_df, correlation_matrix


def format_p_value(p_value):
    """Format p-value for display with appropriate precision"""
    if p_value is None or np.isnan(p_value):
        return "N/A"
    # Use scientific notation for very small p-values
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    # For larger p-values, use fixed notation with 6 decimal places
    return f"{p_value:.6f}"


def significance_label(p_value):
    """Return significance stars or ns for a p-value."""
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def save_statistical_results_table(results_df, metrics_list, output_dir):
    """
    Generate detailed statistical results table for publication.

    Creates both CSV and Markdown tables with:
    - Genotype and control
    - Metric name
    - Sample sizes
    - Mean values (genotype vs control)
    - P-value (raw and FDR-corrected)
    - Cohen's d effect size
    - Significance label

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_metric_analysis with metric-level statistics
    metrics_list : list
        List of metrics analyzed
    output_dir : str
        Directory to save output files
    """
    print(f"\n📝 Generating statistical results table...")

    def _fmt_num(value, decimals=3):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}"

    def _fmt_pct(value, decimals=1):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:+.{decimals}f}%"

    # Prepare data for table
    table_rows = []

    for _, row in results_df.iterrows():
        genotype = row["genotype"]
        control = row["control"]

        # Iterate through all metrics
        for metric in metrics_list:
            pval_col = f"{metric}_pval"
            pval_corr_col = f"{metric}_pval_corrected"
            cohens_d_col = f"{metric}_cohens_d"
            genotype_n_col = f"{metric}_genotype_n"
            control_n_col = f"{metric}_control_n"
            genotype_mean_col = f"{metric}_genotype_mean"
            control_mean_col = f"{metric}_control_mean"
            mean_diff_col = f"{metric}_mean_diff"
            ci_lower_col = f"{metric}_ci_lower"
            ci_upper_col = f"{metric}_ci_upper"
            pct_change_col = f"{metric}_pct_change"
            pct_ci_lower_col = f"{metric}_pct_ci_lower"
            pct_ci_upper_col = f"{metric}_pct_ci_upper"
            n_bootstrap_col = f"{metric}_n_bootstrap"

            # Skip if metric not in results
            if pval_col not in row or pd.isna(row[pval_col]):
                continue

            pval = row[pval_col]
            pval_corrected = row[pval_corr_col] if pval_corr_col in row else pval
            cohens_d_value = row[cohens_d_col] if cohens_d_col in row else np.nan
            genotype_n = row[genotype_n_col] if genotype_n_col in row else np.nan
            control_n = row[control_n_col] if control_n_col in row else np.nan
            genotype_mean = row[genotype_mean_col] if genotype_mean_col in row else np.nan
            control_mean = row[control_mean_col] if control_mean_col in row else np.nan
            mean_diff = row[mean_diff_col] if mean_diff_col in row else np.nan
            ci_lower = row[ci_lower_col] if ci_lower_col in row else np.nan
            ci_upper = row[ci_upper_col] if ci_upper_col in row else np.nan
            pct_change = row[pct_change_col] if pct_change_col in row else np.nan
            pct_ci_lower = row[pct_ci_lower_col] if pct_ci_lower_col in row else np.nan
            pct_ci_upper = row[pct_ci_upper_col] if pct_ci_upper_col in row else np.nan
            n_bootstrap = row[n_bootstrap_col] if n_bootstrap_col in row else np.nan

            table_rows.append(
                {
                    "Genotype": genotype,
                    "Control": control,
                    "Metric": get_display_name(metric),
                    "Metric_ID": metric,
                    "Genotype_n": genotype_n,
                    "Control_n": control_n,
                    "Genotype_mean": genotype_mean,
                    "Control_mean": control_mean,
                    "Mean_diff": mean_diff,
                    "Bootstrap_CI_lower": ci_lower,
                    "Bootstrap_CI_upper": ci_upper,
                    "Percent_change": pct_change,
                    "Percent_CI_lower": pct_ci_lower,
                    "Percent_CI_upper": pct_ci_upper,
                    "N_bootstrap": n_bootstrap,
                    "P_value": pval,
                    "P_value_formatted": format_p_value(pval),
                    "P_value_corrected": pval_corrected,
                    "P_value_corrected_formatted": format_p_value(pval_corrected),
                    "Cohens_d": cohens_d_value,
                    "Significance": significance_label(pval_corrected),
                }
            )

    # Convert to DataFrame and save
    if table_rows:
        stats_table = pd.DataFrame(table_rows)

        # Sort by genotype, then by p-value (corrected)
        stats_table = stats_table.sort_values(["Genotype", "P_value_corrected"])

        # Save CSV version with all numeric precision
        csv_path = os.path.join(output_dir, "statistical_results_detailed.csv")
        stats_table.to_csv(csv_path, index=False)
        print(f"   💾 Saved CSV table: {csv_path}")

        # Create Markdown version for manuscript reporting
        md_path = os.path.join(output_dir, "statistical_results_detailed.md")

        with open(md_path, "w") as f:
            f.write("# Statistical Results: Detailed Metric Analysis\n\n")
            f.write(f"**Analysis Type:** {'Permutation test' if USE_PERMUTATION_PER_PC else 'Mann-Whitney U test'}\n")
            f.write(f"**FDR Correction:** Benjamini-Hochberg (α = 0.05)\n")
            f.write(f"**Total comparisons:** {len(table_rows)}\n\n")

            # Group by genotype for better readability
            for genotype in stats_table["Genotype"].unique():
                genotype_data = stats_table[stats_table["Genotype"] == genotype]
                control = genotype_data["Control"].iloc[0]

                f.write(f"\n## {genotype} vs {control}\n\n")

                # Count significant metrics
                n_significant = (genotype_data["Significance"] != "ns").sum()
                f.write(f"**Significant metrics:** {n_significant}/{len(genotype_data)}\n\n")

                # Create table
                f.write(
                    "| Metric | n (G/C) | Mean G | Mean C | ΔMean (G-C) | 95% BS CI ΔMean | %Δ vs C | 95% BS CI %Δ | P-value | P-value (FDR) | Cohen's d | Significance |\n"
                )
                f.write(
                    "|--------|---------|--------|--------|-------------|-----------------|---------|--------------|---------|---------------|-----------|--------------|\n"
                )

                for _, metric_row in genotype_data.iterrows():
                    metric_name = metric_row["Metric"]
                    genotype_n = f"{int(metric_row['Genotype_n'])}" if not pd.isna(metric_row["Genotype_n"]) else "N/A"
                    control_n = f"{int(metric_row['Control_n'])}" if not pd.isna(metric_row["Control_n"]) else "N/A"
                    n_text = f"{genotype_n}/{control_n}"
                    mean_g = _fmt_num(metric_row.get("Genotype_mean", np.nan), 3)
                    mean_c = _fmt_num(metric_row.get("Control_mean", np.nan), 3)
                    mean_diff = _fmt_num(metric_row.get("Mean_diff", np.nan), 3)
                    ci_low = _fmt_num(metric_row.get("Bootstrap_CI_lower", np.nan), 3)
                    ci_high = _fmt_num(metric_row.get("Bootstrap_CI_upper", np.nan), 3)
                    ci_text = f"[{ci_low}, {ci_high}]"
                    pct_change = _fmt_pct(metric_row.get("Percent_change", np.nan), 1)
                    pct_ci_low = _fmt_pct(metric_row.get("Percent_CI_lower", np.nan), 1)
                    pct_ci_high = _fmt_pct(metric_row.get("Percent_CI_upper", np.nan), 1)
                    pct_ci_text = f"[{pct_ci_low}, {pct_ci_high}]"
                    p_fmt = metric_row["P_value_formatted"]
                    p_corr_fmt = metric_row["P_value_corrected_formatted"]
                    cohens_d = f"{metric_row['Cohens_d']:.3f}" if not pd.isna(metric_row["Cohens_d"]) else "N/A"
                    sig = metric_row["Significance"]

                    f.write(
                        f"| {metric_name} | {n_text} | {mean_g} | {mean_c} | {mean_diff} | {ci_text} | {pct_change} | {pct_ci_text} | {p_fmt} | {p_corr_fmt} | {cohens_d} | {sig} |\n"
                    )

        print(f"   💾 Saved Markdown table: {md_path}")

        # Print summary statistics
        total_tests = len(stats_table)
        total_significant = (stats_table["Significance"] != "ns").sum()
        n_genotypes = stats_table["Genotype"].nunique()

        print(f"\n   📊 Statistical Summary:")
        print(f"      Genotypes analyzed: {n_genotypes}")
        print(f"      Total metric comparisons: {total_tests}")
        print(f"      Significant (FDR < 0.05): {total_significant} ({100*total_significant/total_tests:.1f}%)")

        # Show top 10 most significant results
        print(f"\n   🎯 Top 10 most significant results:")
        top_results = stats_table.nsmallest(10, "P_value_corrected")[
            ["Genotype", "Metric", "P_value_corrected_formatted", "Cohens_d", "Significance"]
        ]
        for idx, result in top_results.iterrows():
            print(
                f"      {result['Genotype']:15s} | {result['Metric']:40s} | p={result['P_value_corrected_formatted']:10s} | d={result['Cohens_d']:6.3f} | {result['Significance']}"
            )

    else:
        print("   ⚠️  No statistical results to tabulate")


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
                            fontsize=16,
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

    # # Save plots
    mode_suffix = "" if color_mode == "effect_size" else f"_{color_mode}"
    png_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}.png")
    pdf_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}.pdf")
    svg_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}.svg")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(svg_file, format="svg")
    print(f"   💾 Detailed heatmap ({color_mode} mode) saved: {png_file} and {pdf_file}")
    # base_path = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap{mode_suffix}")
    # save_figure_dual_versions(fig, base_path, axes_with_colored_labels=[ax])

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
    # row clustering and spacing
    row_cluster_level=None,  # Dendrogram level at which to cut for row clusters (None = no gaps)
    cluster_gap_size=2,  # Size of gap (in heatmap units) between clusters
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

    # 3) Compute row cluster assignments
    genotypes_by_cluster = {}
    cluster_order = []  # Track cluster IDs in dendrogram order

    if row_cluster_level is not None and row_Z is not None and M.shape[0] > 1:
        from scipy.cluster.hierarchy import fcluster

        row_cluster_assignments = fcluster(row_Z, row_cluster_level, criterion="maxclust")
        # Map original indices to cluster assignments
        cluster_map = {i: row_cluster_assignments[i] for i in range(len(row_cluster_assignments))}

        # Build row ordering from dendrogram
        set_link_color_palette(list(row_palette))
        dg_row = dendrogram(
            row_Z,
            orientation="left",
            color_threshold=(None if color_threshold_rows == "default" else color_threshold_rows),
            above_threshold_color=above_threshold_color_rows,
            no_labels=True,
            ax=plt.gca() if False else None,  # Don't draw yet, just get ordering
        )
        row_order_idx = dg_row["leaves"]

        # Group genotypes by cluster, maintaining dendrogram order
        ordered_clusters = [cluster_map[i] for i in row_order_idx]
        for i, genotype_idx in enumerate(row_order_idx):
            cluster_id = cluster_map[genotype_idx]
            if cluster_id not in genotypes_by_cluster:
                genotypes_by_cluster[cluster_id] = []
                cluster_order.append(cluster_id)
            genotypes_by_cluster[cluster_id].append(M.index[genotype_idx])

        print(f"   🔀 Row clustering: {row_cluster_level} clusters identified")
        print(f"      Cluster distribution: {[len(genotypes_by_cluster[c]) for c in cluster_order]}")
    else:
        # No clustering: treat all genotypes as a single cluster
        row_order_idx = list(range(M.shape[0]))
        genotypes_by_cluster[1] = [M.index[i] for i in row_order_idx]
        cluster_order = [1]
        row_cluster_level = None

    # 3b) GRIDSPEC layout - separate subplots for each cluster
    # Arrange clusters as: 3 at top, 1 in middle, 2 at bottom
    plt.style.use("default")
    fig = plt.figure(figsize=fig_size)
    num_clusters = len(cluster_order)

    # Reorder clusters: cluster 3, then cluster 1, then cluster 2
    # Assuming cluster_order contains [1, 2, 3] based on dendrogram
    if num_clusters == 3:
        # Custom display order: 3, 1, 2
        display_order = [cluster_order[2], cluster_order[0], cluster_order[1]]  # indices 2, 0, 1 = clusters 3, 1, 2
        print(f"   🎨 Arranging clusters in custom order: {display_order} (3 top, 1 middle, 2 bottom)")
    else:
        # For other numbers of clusters, use original order
        display_order = cluster_order
        print(f"   ℹ️  Using default cluster order: {display_order}")

    # Height ratios for each cluster (proportional to number of genotypes)
    cluster_heights = [len(genotypes_by_cluster[cid]) for cid in display_order]

    # GridSpec: rows are [top_dendro, cluster1, cluster2, cluster3]
    # columns are [left_dendro, labels, hm, cbar]
    total_grid_rows = 1 + num_clusters  # top dendro + clusters

    gs = gridspec.GridSpec(
        total_grid_rows,
        4,
        height_ratios=[2.5] + cluster_heights,  # top dendrogram + cluster rows
        width_ratios=[1.5, 1.2, 8.0, 0.3],  # left dendro, labels, heatmap, colorbar
        wspace=0.08,  # Moderate spacing to create visible gap between labels and heatmap
        hspace=0.15,  # Increased spacing between separate subplots
    )

    # Top dendrogram (spans heatmap column only)
    ax_top_dendro = fig.add_subplot(gs[0, 2])

    # DON'T force colors - allows individual line styling afterward
    set_link_color_palette(["#404040"])

    dg_col = dendrogram(
        col_Z,
        orientation="top",
        color_threshold=None,  # Don't force uniform color - allows individual line styling
        no_labels=True,
        ax=ax_top_dendro,
    )
    col_order_idx = dg_col["leaves"]
    col_labels_ordered = [metric_names[i] for i in col_order_idx]

    # Now manually recolor ALL dendrogram lines - override any default scipy colors
    # IMPORTANT: Only modify collections that belong to the dendrogram (in ax_top_dendro)
    if col_Z is not None and len(metric_names) > 1:
        max_dist_top = col_Z[:, 2].max()

        # Work with LineCollections (which is what dendrogram actually creates)
        # Store collections before modifying to ensure we only touch dendrogram collections
        from matplotlib.collections import LineCollection
        import numpy as np

        dendrogram_collections = list(ax_top_dendro.collections)

        for old_collection in dendrogram_collections:
            segments = old_collection.get_segments()

            # Separate segments into two groups based on their distance
            light_gray_segments = []
            dark_gray_segments = []

            for segment in segments:
                if len(segment) > 0:
                    # For TOP orientation, Y-values are distances
                    y_values = segment[:, 1]
                    max_y = max(y_values) if len(y_values) > 0 else 0

                    if max_y >= max_dist_top * 0.80:
                        # Root segments (highest distance) - light gray, below
                        light_gray_segments.append(segment)
                    else:
                        # Other segments - dark gray, above
                        dark_gray_segments.append(segment)
                else:
                    dark_gray_segments.append(segment)

            # Remove old collection
            old_collection.remove()

            # Create new collections with different z-orders
            # Light gray segments: z-order 8 (below dark gray)
            if light_gray_segments:
                lc_light = LineCollection(light_gray_segments, colors="lightgray", linewidths=2.0, zorder=8)
                ax_top_dendro.add_collection(lc_light)

            # Dark gray segments: z-order 10 (above light gray)
            if dark_gray_segments:
                lc_dark = LineCollection(dark_gray_segments, colors="#404040", linewidths=2.0, zorder=10)
                ax_top_dendro.add_collection(lc_dark)

        # Also handle any Line2D objects (just in case)
        for line in ax_top_dendro.get_lines():
            ydata = line.get_ydata()
            if len(ydata) > 0:
                max_y = max(ydata)
                if max_y >= max_dist_top * 0.80:
                    line.set_color("lightgray")
                    line.set_linewidth(2.0)
                    line.set_zorder(8)  # Light gray below
                else:
                    line.set_color("#404040")
                    line.set_linewidth(2.0)
                    line.set_zorder(10)  # Dark gray above
    else:
        # Just increase linewidth if no dendrogram
        for collection in ax_top_dendro.collections:
            collection.set_linewidth(2.0)
            collection.set_zorder(10)
        for line in ax_top_dendro.get_lines():
            line.set_linewidth(2.0)
            line.set_zorder(10)
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

    # 6) Create separate heatmap subplots for each cluster
    # Build ordered genotype list per cluster
    cluster_genotypes = {}
    for cluster_id in display_order:
        cluster_genotypes[cluster_id] = genotypes_by_cluster[cluster_id]

    # Create heatmap axes - one for each cluster
    ax_hm_list = []
    ax_left_dendro_list = []
    ax_label_list = []

    for i, cluster_id in enumerate(display_order):
        row_idx = 1 + i  # Skip top dendrogram row

        # Left dendrogram for this cluster
        ax_left = fig.add_subplot(gs[row_idx, 0])
        ax_left_dendro_list.append(ax_left)

        # Label column for this cluster
        ax_label = fig.add_subplot(gs[row_idx, 1])
        ax_label_list.append(ax_label)

        # Heatmap for this cluster
        ax_hm = fig.add_subplot(gs[row_idx, 2])
        ax_hm_list.append((cluster_id, ax_hm))

    # Colorbar axis - span all cluster rows
    first_cluster_row = 1
    last_cluster_row = num_clusters
    ax_cbar = fig.add_subplot(gs[first_cluster_row : last_cluster_row + 1, 3])

    # Prepare col order and cluster matrices
    cluster_matrices = {}
    for cluster_id in display_order:
        genotypes = cluster_genotypes[cluster_id]
        cluster_M = M.loc[genotypes, :]
        cluster_matrices[cluster_id] = cluster_M.iloc[:, col_order_idx] if len(col_order_idx) > 0 else cluster_M

    # Make NaN cells transparent in colormap
    cmap_obj = plt.get_cmap("RdBu_r").copy()
    try:
        cmap_obj.set_bad(alpha=0.0)
    except Exception:
        pass

    # Plot each cluster's heatmap
    for i, (cluster_id, ax_hm) in enumerate(ax_hm_list):
        display_matrix = cluster_matrices[cluster_id]
        ax_label = ax_label_list[i]

        if HAS_SEABORN and "sns" in globals():
            # Seaborn centers cells at integer coordinates, not +0.5
            # We need to manually adjust for proper alignment with labels
            sns.heatmap(
                display_matrix,
                ax=ax_hm,
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                linewidths=linewidths,
                linecolor=linecolor,
                square=False,
                xticklabels=False,
                yticklabels=False,  # Labels now in separate column
                annot=False,
            )
            # Seaborn heatmap uses cell centers at integer coordinates
            # But we need centers at +0.5 to match label positions
            # This is already handled by seaborn internally
        else:
            im = ax_hm.imshow(display_matrix.values, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
            ax_hm.set_xticks(range(display_matrix.shape[1]))
            ax_hm.set_xticklabels([])

        # Seaborn already adds gridlines, but for imshow we need to add them manually
        if not (HAS_SEABORN and "sns" in globals()):
            ax_hm.set_xticks(np.arange(display_matrix.shape[1] + 1) - 0.5, minor=True)
            ax_hm.set_yticks(np.arange(display_matrix.shape[0] + 1) - 0.5, minor=True)
            ax_hm.grid(which="minor", color=linecolor, linestyle="-", linewidth=linewidths)
            ax_hm.tick_params(which="minor", size=0)

        # Remove y-ticks from heatmap (but keep gridlines)
        ax_hm.set_yticks([])
        ax_hm.set_yticklabels([])

        # Set up label column - centered on row midpoints to match heatmap cells
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(-0.5, len(display_matrix.index) - 0.5)
        ax_label.invert_yaxis()
        ax_label.set_xticks([])

        # Move y-axis ticks to the RIGHT side of the label column (away from dendrogram)
        ax_label.yaxis.tick_right()
        ax_label.yaxis.set_label_position("right")
        # Position labels at integers to match seaborn heatmap cells (which center at 0, 1, 2...)
        ax_label.set_yticks(np.arange(len(display_matrix.index)))  # At 0, 1, 2... to match heatmap
        ax_label.set_yticklabels(display_matrix.index, fontsize=row_label_fontsize, ha="right")
        ax_label.tick_params(axis="y", which="both", length=0, pad=2)  # Remove tick marks, small padding
        for spine in ax_label.spines.values():
            spine.set_visible(False)
        # Only show x-axis labels on the bottom cluster
        if i == len(ax_hm_list) - 1:
            ax_hm.set_xticks(np.arange(len(col_labels_ordered)) + 0.5)
            ax_hm.set_xticklabels(
                [get_display_name(m) for m in col_labels_ordered],
                rotation=metric_label_rotation,
                ha="right",
                fontsize=col_label_fontsize,
            )
        else:
            ax_hm.set_xticks([])
            ax_hm.set_xticklabels([])

        # Color y-axis labels by brain region (in label column)
        try:
            region_mapping = simplified_to_region if simplified_to_region else nickname_to_brainregion
            colour_y_ticklabels(ax_label, region_mapping, color_dict)
        except Exception as e:
            print(f"   ⚠️  Could not color y-labels for cluster {cluster_id}: {e}")

        # Add significance stars
        for row_idx, genotype in enumerate(display_matrix.index):
            for j, metric in enumerate(col_labels_ordered):
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
                        cell_value = None
                        try:
                            cell_value = display_matrix.iloc[row_idx, j]
                        except Exception:
                            pass
                        text_color = "white" if (cell_value is not None and abs(cell_value) >= 0.5) else "black"
                        ax_hm.text(
                            j + 0.5,
                            row_idx + 0.5,
                            stars,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=16,
                            fontweight="bold",
                        )

    # Draw left dendrograms for each cluster
    if row_Z is not None and M.shape[0] > 1:
        # Get the full dendrogram structure
        # Set consistent dark gray color for all dendrogram lines
        set_link_color_palette(["#404040"])  # Dark gray
        dg_left = dendrogram(
            row_Z,
            orientation="left",
            color_threshold=0,  # Force all lines to use the same color
            above_threshold_color="#404040",  # Dark gray
            no_labels=True,
            no_plot=True,
        )

        # Map leaf indices to cluster IDs and positions
        leaf_to_cluster = {}
        leaf_to_position = {}  # Position within the ordered genotype list

        current_pos = 0
        for cluster_id in display_order:
            cluster_genotypes_list = cluster_genotypes[cluster_id]
            for genotype in cluster_genotypes_list:
                leaf_idx = list(M.index).index(genotype)
                leaf_to_cluster[leaf_idx] = cluster_id
                leaf_to_position[leaf_idx] = current_pos
                current_pos += 1

        # For each cluster, extract and plot relevant dendrogram segments
        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            cluster_genotypes_list = cluster_genotypes[cluster_id]
            cluster_size = len(cluster_genotypes_list)

            # Find leaf indices for this cluster
            cluster_leaf_indices = set([list(M.index).index(g) for g in cluster_genotypes_list])

            # Map global positions to local positions within this cluster
            global_to_local_y = {}
            for j, genotype in enumerate(cluster_genotypes_list):
                global_leaf_idx = list(M.index).index(genotype)
                # Map to integer positions (0, 1, 2...) to match seaborn heatmap and labels
                global_to_local_y[leaf_to_position[global_leaf_idx]] = j

            # Function to check if a dendrogram segment belongs to this cluster
            def get_segment_leaves(icoord_segment):
                """Extract leaf indices from a dendrogram segment's y-coordinates"""
                leaves = []
                for y_val in icoord_segment:
                    # Dendrogram y-coords are (leaf_idx * 10 + 5)
                    leaf_idx = int((y_val - 5) / 10)
                    if 0 <= leaf_idx < len(dg_left["leaves"]):
                        leaves.append(dg_left["leaves"][leaf_idx])
                return leaves

            # Plot dendrogram segments that belong to this cluster
            # First pass: find max distance for this cluster
            max_x = 0
            max_distance_in_cluster = 0
            for ic, dc, color in zip(dg_left["icoord"], dg_left["dcoord"], dg_left.get("color_list", [])):
                segment_leaves = get_segment_leaves(ic)
                if segment_leaves and all(leaf in cluster_leaf_indices for leaf in segment_leaves):
                    max_distance_in_cluster = max(max_distance_in_cluster, max(dc))

            # Second pass: plot segments with appropriate colors and z-orders
            for ic, dc, color in zip(dg_left["icoord"], dg_left["dcoord"], dg_left.get("color_list", [])):
                segment_leaves = get_segment_leaves(ic)

                # Check if all leaves in this segment belong to this cluster
                if segment_leaves and all(leaf in cluster_leaf_indices for leaf in segment_leaves):
                    # Remap y-coordinates to local cluster coordinates
                    new_ic = []
                    valid_segment = True
                    for y_val in ic:
                        leaf_idx = int((y_val - 5) / 10)
                        if 0 <= leaf_idx < len(dg_left["leaves"]):
                            actual_leaf = dg_left["leaves"][leaf_idx]
                            global_pos = leaf_to_position[actual_leaf]
                            if global_pos in global_to_local_y:
                                new_ic.append(global_to_local_y[global_pos])
                            else:
                                valid_segment = False
                                break
                        else:
                            # This is an internal node, estimate position
                            # Use average of the closest leaves
                            frac = (y_val - 5) / 10
                            if frac < 0:
                                frac = 0
                            elif frac >= len(dg_left["leaves"]) - 1:
                                frac = len(dg_left["leaves"]) - 1

                            low_idx = int(np.floor(frac))
                            high_idx = min(low_idx + 1, len(dg_left["leaves"]) - 1)
                            low_leaf = dg_left["leaves"][low_idx]
                            high_leaf = dg_left["leaves"][high_idx]

                            if low_leaf in cluster_leaf_indices and high_leaf in cluster_leaf_indices:
                                low_local = global_to_local_y.get(leaf_to_position[low_leaf], 0)
                                high_local = global_to_local_y.get(leaf_to_position[high_leaf], cluster_size - 1)
                                ratio = frac - low_idx
                                interpolated = low_local * (1 - ratio) + high_local * ratio
                                new_ic.append(interpolated)
                            else:
                                valid_segment = False
                                break

                    if valid_segment and len(new_ic) == 4:
                        # Determine if this is a root segment based on distance
                        max_distance_in_segment = max(dc)

                        # Root segments: light gray, z-order 8 (below)
                        # Other segments: dark gray, z-order 10 (above)
                        if max_distance_in_cluster > 0 and max_distance_in_segment >= max_distance_in_cluster * 0.90:
                            ax_left.plot(dc, new_ic, color="lightgray", linewidth=2.0, zorder=8)
                        else:
                            ax_left.plot(dc, new_ic, color="#404040", linewidth=2.0, zorder=10)
                        max_x = max(max_x, max(dc))

            ax_left.set_ylim(-0.5, cluster_size - 0.5)
            # Extend xlim to create space for connecting lines on the left
            # ax_left.set_xlim(-max_x * 0.15, max(max_x * 1.05, 1))  # Add 15% extra space on left
            ax_left.invert_xaxis()
            ax_left.invert_yaxis()
            ax_left.set_xticks([])
            ax_left.set_yticks([])
            for spine in ax_left.spines.values():
                spine.set_visible(False)
            # Ensure lines don't extend beyond axes bounds
            ax_left.set_clip_on(True)

        # Sub-dendrograms are drawn for visualization - they show cluster hierarchy independently

        # Now draw the overarching stem connecting all 3 sub-dendrograms
        from matplotlib.lines import Line2D

        # Collect the root information for each cluster
        cluster_roots = []  # List of (ax, y_position, max_distance) for each cluster

        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            cluster_genotypes_list = cluster_genotypes[cluster_id]
            cluster_size = len(cluster_genotypes_list)

            # Find the maximum distance (root) in this cluster's dendrogram
            cluster_leaf_indices = set([list(M.index).index(g) for g in cluster_genotypes_list])

            max_distance_cluster = 0
            for dc, ic in zip(dg_left["dcoord"], dg_left["icoord"]):
                segment_leaves = []
                for y_val in ic:
                    leaf_idx = int((y_val - 5) / 10)
                    if 0 <= leaf_idx < len(dg_left["leaves"]):
                        segment_leaves.append(dg_left["leaves"][leaf_idx])

                if segment_leaves and all(leaf in cluster_leaf_indices for leaf in segment_leaves):
                    max_distance_cluster = max(max_distance_cluster, max(dc))

            # Root y-position is at the center of the cluster
            root_y = (cluster_size - 1) / 2.0

            cluster_roots.append(
                {"ax": ax_left, "y_pos": root_y, "max_dist": max_distance_cluster, "cluster_size": cluster_size}
            )

        # Stem x-position: extend beyond the maximum dendrogram distance
        stem_x = max([cr["max_dist"] for cr in cluster_roots]) * 1.15

        # First, determine the target x-coordinate in figure space using the first axis
        ax_ref = cluster_roots[0]["ax"]
        y_ref = cluster_roots[0]["y_pos"]
        trans_ref = ax_ref.transData
        point_ref_data = trans_ref.transform([stem_x, y_ref])
        point_ref_fig = fig.transFigure.inverted().transform(point_ref_data)
        stem_x_fig = point_ref_fig[0]  # This is the consistent x-coordinate in figure space

        # Now calculate all endpoints
        stem_endpoints = []  # List of (x_fig, y_fig) for each cluster
        root_endpoints = []  # List of (x_fig, y_fig) for each dendrogram root

        for cr in cluster_roots:
            ax = cr["ax"]
            y_pos = cr["y_pos"]
            max_dist = cr["max_dist"]

            # Transform dendrogram root endpoint to figure coordinates
            trans = ax.transData
            point_root_data = trans.transform([max_dist, y_pos])
            point_root_fig = fig.transFigure.inverted().transform(point_root_data)
            root_endpoints.append(point_root_fig)

            # All stem endpoints have the same x (stem_x_fig) but different y
            # Transform the figure point back to y-position in this specific axis
            point_stem_fig_temp = [stem_x_fig, 0]  # Temp y-value
            point_stem_display = fig.transFigure.transform(point_stem_fig_temp)
            point_stem_data_temp = trans.inverted().transform(point_stem_display)
            # y-coordinate should be the cluster's y_pos
            stem_endpoints.append([stem_x_fig, point_root_fig[1]])  # Use root's y-position in fig coords

        # Draw horizontal and vertical stem lines entirely in figure coordinates
        for i in range(len(cluster_roots)):
            # Draw horizontal line from dendrogram root to stem
            # All stem_endpoints have the same x-coordinate (stem_x_fig)
            horizontal_line = Line2D(
                [root_endpoints[i][0], stem_x_fig],
                [root_endpoints[i][1], stem_endpoints[i][1]],
                color="lightgray",  # Light gray
                linestyle="-",
                linewidth=2.0,  # Increased linewidth
                transform=fig.transFigure,
                zorder=5,
            )
            fig.add_artist(horizontal_line)

        # Draw vertical stem line connecting all cluster stems
        if len(stem_endpoints) > 1:
            vertical_stem = Line2D(
                [stem_x_fig, stem_x_fig],
                [stem_endpoints[0][1], stem_endpoints[-1][1]],
                color="lightgray",  # Light gray
                linestyle="-",
                linewidth=2.0,  # Increased linewidth
                transform=fig.transFigure,
                zorder=5,
            )
            fig.add_artist(vertical_stem)
    else:
        # No dendrogram - just label clusters
        for i, cluster_id in enumerate(display_order):
            ax_left = ax_left_dendro_list[i]
            ax_left.text(
                0.5,
                0.5,
                f"C{cluster_id}",
                transform=ax_left.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            ax_left.axis("off")

    # Bottom metric labels axis (not needed since we show labels on bottom heatmap)
    # ax_metric_labels = fig.add_subplot(gs[-1, 1])
    # ax_metric_labels.axis("off")

    # 7) Add colorbar

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

    # 11) Clean up axes
    ax_top_dendro.set_xticks([])
    ax_top_dendro.set_yticks([])
    for spine in ax_top_dendro.spines.values():
        spine.set_visible(False)
    ax_top_dendro.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 12) Final styling on all heatmap axes
    if despine:
        for cluster_id, ax_hm in ax_hm_list:
            if HAS_SEABORN:
                sns.despine(ax=ax_hm, top=True, right=True, left=False, bottom=False)
            else:
                ax_hm.spines["top"].set_visible(False)
                ax_hm.spines["right"].set_visible(False)

    # 13) Save files
    png = os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.png")
    pdf = os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.pdf")
    svg = os.path.join(OUTPUT_DIR, f"metric_two_way_dendrogram.svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, format="svg")
    plt.close(fig)

    # 14) Save matrix and orders
    mat_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_matrix.csv")
    row_order_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_row_order.csv")
    col_order_csv = os.path.join(OUTPUT_DIR, f"metric_two_way_col_order.csv")
    M.to_csv(mat_csv)

    # Build ordered genotype list from display order
    ordered_genotypes = []
    for cluster_id in display_order:
        ordered_genotypes.extend(cluster_genotypes[cluster_id])

    pd.Series(ordered_genotypes, name="genotype").to_csv(row_order_csv, index=False)
    pd.Series(col_labels_ordered, name="metric").to_csv(col_order_csv, index=False)

    if row_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"metric_two_way_row_linkage.npy"), row_Z)
    if col_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"metric_two_way_col_linkage.npy"), col_Z)

    print(f"   💾 Matrix CSV: {mat_csv}")
    print(f"   💾 Row/Col orders: {row_order_csv} / {col_order_csv}")

    return {
        "matrix": M,
        "row_Z": row_Z,
        "col_Z": col_Z,
        "row_order": ordered_genotypes,
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
                        fontsize=16,
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

    # 10) Set labels
    ax_main.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Genotypes (sorted by brain region)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # 11) Save files
    png = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap.png")
    pdf = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap.pdf")
    svg = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap.svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, format="svg")
    plt.close(fig)

    # base_path = os.path.join(OUTPUT_DIR, "metric_simple_heatmap")
    # save_figure_dual_versions(fig, base_path, axes_with_colored_labels=[ax_main])
    plt.close(fig)

    # 12) Save matrix and orders
    simple_mat_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_matrix.csv")
    simple_row_order_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_row_order.csv")
    simple_col_order_csv = os.path.join(OUTPUT_DIR, f"metric_simple_heatmap_col_order.csv")
    M_ordered.to_csv(simple_mat_csv)
    pd.Series(row_order, name="genotype").to_csv(simple_row_order_csv, index=False)
    pd.Series(col_order, name="metric").to_csv(simple_col_order_csv, index=False)

    # print(f"   💾 Saved simple metric heatmap: {png} / {pdf}")
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

    # Order metrics by predefined thematic groups (METRIC_GROUPS) instead of correlation clustering
    metric_names_set = set(metric_names)
    col_order = []
    # valid_groups tracks (group_label, [metric_keys_in_order]) for annotation
    valid_groups = []
    for group_label, group_keys in METRIC_GROUPS:
        present = [k for k in group_keys if k in metric_names_set]
        if present:
            col_order.extend(present)
            valid_groups.append((group_label, present))
    # Append any metrics not covered by any group, preserving original order
    covered = set(col_order)
    extra = [m for m in metric_names if m not in covered]
    if extra:
        col_order.extend(extra)
        valid_groups.append(("Other", extra))

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
        width_ratios=[20, 0.4],  # Main plot area and colorbar
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
                            fontsize=11,
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
        # Color each metric label to match its thematic group (black / gray alternation)
        _group_colors_tick = ["black", "#888888"]
        _col_to_group_color = {}
        for _g_idx, (_g_label, _g_keys) in enumerate(valid_groups):
            _gc = _group_colors_tick[_g_idx % len(_group_colors_tick)]
            for _k in _g_keys:
                if _k in col_order:
                    _col_to_group_color[col_order.index(_k)] = _gc
        for _j, _lbl in enumerate(axes[-1].get_xticklabels()):
            _lbl.set_color(_col_to_group_color.get(_j, "black"))

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

    # Overall title removed

    # Reserve extra bottom margin for the group bracket annotations, then finalise layout
    plt.tight_layout(rect=[0, 0.12, 1, 0.99])

    # --- Thematic group bracket annotations below x-axis of bottom subplot ---
    # Must come AFTER tight_layout so that all axes positions are final and
    # tick-label bounding boxes reflect the actual rendered layout.
    # We work in figure-fraction coordinates (DPI-independent).
    if axes and valid_groups:
        from matplotlib.lines import Line2D

        ax_bottom = axes[-1]
        col_idx_map = {m: i for i, m in enumerate(col_order)}
        group_colors = ["black", "#888888"]  # alternating black / medium-gray
        line_lw = 2.0

        # Render into the Agg canvas so bounding boxes are populated
        fig.canvas.draw()
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            renderer = fig.canvas.renderer
        fig_inv = fig.transFigure.inverted()

        # Collect label bounding boxes in figure-fraction coords.
        # tick_labels[j] corresponds to col_order[j].
        tick_labels = ax_bottom.get_xticklabels()
        label_x0_fig = {}  # col_idx -> left edge of rotated label bbox (fig frac)
        y_min_fig = float("inf")  # lowest y of any label (fig frac)
        for j, lbl in enumerate(tick_labels):
            bb = lbl.get_window_extent(renderer=renderer)
            pts = fig_inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            label_x0_fig[j] = pts[0, 0]  # leftmost x = bottom-left of rotated label
            y_min_fig = min(y_min_fig, pts[0, 1])  # track the lowest y across all labels

        # Place bracket line just below the lowest label edge
        gap_frac = 0.005  # gap between lowest label bottom and bracket line (fig frac)
        label_gap = 0.012  # gap between bracket line and group text (fig frac)

        y_line_fig = y_min_fig - gap_frac
        y_label_fig = y_line_fig - label_gap

        for g_idx, (group_label, group_keys) in enumerate(valid_groups):
            present_in_col = [k for k in group_keys if k in col_idx_map]
            if not present_in_col:
                continue
            j_start_idx = col_idx_map[present_in_col[0]]
            j_end_idx = col_idx_map[present_in_col[-1]]
            color = group_colors[g_idx % len(group_colors)]

            # x_start: bottom-left (leftmost point) of the first label's rotated bbox
            x_start = label_x0_fig.get(j_start_idx, None)
            if x_start is None:
                continue

            # x_end: bottom-left (leftmost point) of the LAST label's rotated bbox —
            # so the bar spans from the visual tip of the first label to the visual
            # tip of the last label (not beyond it).
            x_end = label_x0_fig.get(j_end_idx, None)
            if x_end is None:
                continue

            # Horizontal bracket line (no end-cap ticks)
            fig.add_artist(
                Line2D(
                    [x_start, x_end],
                    [y_line_fig, y_line_fig],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=line_lw,
                    solid_capstyle="butt",
                    clip_on=False,
                )
            )
            # Group label — centred on the bracket bar
            fig.text(
                (x_start + x_end) / 2,
                y_label_fig,
                group_label,
                transform=fig.transFigure,
                color=color,
                ha="center",
                va="top",
                fontsize=col_label_fontsize,
                fontweight="bold",
                clip_on=False,
            )

    # Save
    png = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.png")
    pdf = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.pdf")
    svg = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.svg")
    eps = os.path.join(OUTPUT_DIR, "metric_brain_region_grouped_heatmap.eps")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, format="svg")
    plt.savefig(eps, format="eps")

    print(f"   💾 Saved brain region grouped heatmap: {png} / {pdf}")

    return {
        "matrix": M,
        "col_order": col_order,
        "regions": sorted_regions,
        "genotypes_by_region": genotypes_by_region,
    }


def main():

    # Check if Arial is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in available_fonts:
        print("✓ Arial is available")
    else:
        print("✗ Arial not found")
        print(
            "Available sans-serif fonts:",
            [f for f in available_fonts if "sans" in f.lower() or "helvetica" in f.lower()],
        )
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

    # Step 4.5: Generate detailed statistical results table
    save_statistical_results_table(results_df, metrics_list, OUTPUT_DIR)

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
        row_cluster_level=3,  # Split into 6 clusters (the second level of branching)
        cluster_gap_size=8,  # Size of gaps between clusters (increased for more spacing)
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
            f"zAverage metrics per genotype: {total_metrics_significant/total_hits:.1f}\n"
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
    print(f"📋 Statistical results tables:")
    print(f"   • CSV format: statistical_results_detailed.csv")
    print(f"   • Markdown format: statistical_results_detailed.md")


if __name__ == "__main__":
    main()
