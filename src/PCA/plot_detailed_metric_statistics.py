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
DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250924_14_summary_TNT_screen_Data/summary/pooled_summary.feather"
CONSISTENCY_DIR = "/home/durrieu/ballpushing_utils/src/PCA/consistency_analysis_with_edges"
METRICS_PATH = "metric_lists/final_metrics_for_pca.txt"

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
    return parser.parse_args()


# These globals will be overridden by CLI if provided
OUTPUT_DIR = "best_metric_analysis"


def _apply_args(args):
    global OUTPUT_DIR, CONSISTENCY_DIR, METRICS_PATH, DATA_PATH, MIN_COMBINED_CONSISTENCY
    OUTPUT_DIR = args.output_dir
    CONSISTENCY_DIR = args.consistency_dir
    METRICS_PATH = args.metrics_path
    DATA_PATH = args.data_path
    MIN_COMBINED_CONSISTENCY = args.combined_threshold
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🎯 Output directory: {OUTPUT_DIR}")


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
    """Load combined consistency analysis results and return qualifying genotypes.

    Priority order of sources:
      1. combined_consistency_ranking.csv (preferred; must contain Combined_Consistency column)
      2. enhanced_consistency_scores.csv (fallback; will attempt to derive Combined_Consistency if absent)

    Parameters
    ----------
    threshold : float
        Minimum Combined_Consistency (0–1) required to include a genotype.
    max_hits : int | None
        If provided, cap the number of returned genotypes (after filtering) to top-N by Combined_Consistency.
    """
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
            "❌ No consistency ranking file found (expected combined_consistency_ranking.csv or enhanced_consistency_scores.csv)"
        )
        return []

    # Derive Combined_Consistency if missing and sufficient components exist
    if "Combined_Consistency" not in df.columns:
        if {"Overall_Consistency", "Optimized_Only_Consistency"}.issubset(df.columns):
            print("⚠️  Combined_Consistency not present – deriving as mean of Overall_ & Optimized_Only_Consistencies")
            df["Combined_Consistency"] = 0.5 * df["Overall_Consistency"] + 0.5 * df["Optimized_Only_Consistency"]
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
    if max_hits is not None:
        filtered = filtered.head(max_hits)

    print("📊 CONSISTENCY FILTERING (Combined Consistency):")
    print(f"   Source file: {source}")
    print(f"   Total genotypes evaluated: {len(df)}")
    print(f"   Threshold: ≥{threshold:.0%} Combined_Consistency")
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
    """Load metrics list from file"""
    if not os.path.exists(METRICS_PATH):
        print(f"❌ Metrics file not found: {METRICS_PATH}")
        return []

    with open(METRICS_PATH, "r") as f:
        metrics = [line.strip() for line in f if line.strip()]

    print(f"📋 Loaded {len(metrics)} metrics from {METRICS_PATH}")
    return metrics


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


def prepare_data():
    """Load and preprocess data (same as consistency analysis)"""
    print("📊 Loading and preprocessing data...")

    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)

    # Exclude problematic nicknames
    exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "MB247-Gal4"]
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
        subset = Config.get_subset_data(metric_with_meta, col="Nickname", value=nickname, force_control=None)
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

        for metric in valid_metrics:
            group_values = subset[subset["Nickname"] == nickname][metric]
            control_values = subset[subset["Nickname"] == control_name][metric]
            if group_values.empty or control_values.empty:
                continue

            stat, pval = mannwhitneyu(group_values, control_values, alternative="two-sided")
            mannwhitney_pvals.append(pval)
            metrics_tested.append(metric)

            # Calculate direction
            directions[metric] = 1 if group_values.mean() > control_values.mean() else -1

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


def create_hits_heatmap(results_df, correlation_matrix, nickname_to_brainregion, color_dict, nickname_mapping=None):
    """Create detailed heatmap for high-consistency hits (all of them, not just statistically significant)"""

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

            is_significant = row.get(metric_sig_col, False)
            pval_corrected = row.get(metric_pval_col, np.nan)
            direction = row.get(metric_dir_col, 0)

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

                sig_row.append(direction)  # 1 for up (red), -1 for down (blue)
                annot_row.append(sig_code)

                # DEBUG: Check our test cases
                if row["genotype"] in ["41731 (IR8a-GAL4)", "LC10-2"] and metric == "pulling_ratio":
                    print(f"🔍 SIGNIFICANCE MATRIX: {row['genotype']} - {metric}")
                    print(f"🔍   is_significant: {is_significant}")
                    print(f"🔍   direction: {direction}")
                    print(f"🔍   appending to sig_row: {direction}")
            else:
                sig_row.append(0)  # No significance
                annot_row.append("")

                # DEBUG: Check our test cases when not significant
                if row["genotype"] in ["41731 (IR8a-GAL4)", "LC10-2"] and metric == "pulling_ratio":
                    print(f"🔍 SIGNIFICANCE MATRIX (NOT SIG): {row['genotype']} - {metric}")
                    print(f"🔍   is_significant: {is_significant}")
                    print(f"🔍   direction: {direction}")
                    print(f"🔍   appending to sig_row: 0 (not significant)")

                # DEBUG: Check first few rows for pulling_ratio
                if i < 6 and metric == "pulling_ratio":
                    print(f"🔍 FIRST ROWS [{i}] {row['genotype']} - pulling_ratio:")
                    print(f"🔍   significant: {is_significant}, direction: {direction}, appending: 0")

        significance_matrix.append(sig_row)
        annotation_matrix.append(annot_row)

    if len(significance_matrix) == 0:
        print("No hits matrix to create")
        return None

    # Create DataFrames - genotypes as rows, metrics as columns
    sig_df = pd.DataFrame(significance_matrix, index=sorted_df["genotype"], columns=metric_names)
    annot_df = pd.DataFrame(annotation_matrix, index=sorted_df["genotype"], columns=metric_names)

    # DEBUG: Show the exact mapping for the first 6 rows and pulling_ratio column
    print(f"🔍 FINAL sig_df DEBUG:")
    print(f"🔍   sig_df shape: {sig_df.shape}")
    print(f"🔍   First 6 genotypes (rows):")
    for i in range(min(6, len(sig_df))):
        genotype = sig_df.index[i]
        pulling_value = sig_df.loc[genotype, "pulling_ratio"] if "pulling_ratio" in sig_df.columns else "N/A"
        print(f"🔍     Row {i}: {genotype} -> pulling_ratio = {pulling_value}")
    print(f"🔍   Metrics (columns): {list(sig_df.columns)}")
    if "pulling_ratio" in sig_df.columns:
        pulling_col_idx = list(sig_df.columns).index("pulling_ratio")
        print(f"🔍   pulling_ratio is column index {pulling_col_idx}")

    # Also check if LC10-2 and IR8a are in the DataFrame at all
    if "LC10-2" in sig_df.index:
        lc10_idx = sig_df.index.get_loc("LC10-2")
        lc10_pulling = sig_df.loc["LC10-2", "pulling_ratio"] if "pulling_ratio" in sig_df.columns else "N/A"
        print(f"🔍   LC10-2 is at row index {lc10_idx}, pulling_ratio = {lc10_pulling}")
    if "41731 (IR8a-GAL4)" in sig_df.index:
        ir8a_idx = sig_df.index.get_loc("41731 (IR8a-GAL4)")
        ir8a_pulling = sig_df.loc["41731 (IR8a-GAL4)", "pulling_ratio"] if "pulling_ratio" in sig_df.columns else "N/A"
        print(f"🔍   41731 (IR8a-GAL4) is at row index {ir8a_idx}, pulling_ratio = {ir8a_pulling}")

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(max(14, len(metric_names) * 0.4), max(8, len(genotype_names) * 0.4)))

    # Create custom colormap - FINAL FIX: invert colors to match the actual data
    # Based on debug: -1 values show red but should show blue, so colormap is inverted
    colors = ["lightcoral", "white", "lightblue"]  # red for -1, white for 0, blue for +1
    cmap = ListedColormap(colors)

    # Create the heatmap
    if HAS_SEABORN and "sns" in globals():
        sns.heatmap(
            sig_df,
            annot=annot_df,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"fontsize": 8, "fontweight": "bold"},
        )
    else:
        # Fallback using matplotlib imshow
        im = ax.imshow(sig_df.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sig_df.columns)))
        ax.set_yticks(range(len(sig_df.index)))
        ax.set_xticklabels(sig_df.columns)
        ax.set_yticklabels(sig_df.index)

    # Color y-tick labels by brain region
    colour_y_ticklabels(ax, nickname_to_brainregion, color_dict)

    # Customize the plot
    ax.set_title(
        f"High-Consistency Hits - Detailed Metric Analysis\n"
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

    # Add legend
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
    png_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap.png")
    pdf_file = os.path.join(OUTPUT_DIR, f"metric_high_consistency_heatmap.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"   💾 Detailed heatmap saved: {png_file} and {pdf_file}")

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
    weight_mode="linear_cap",  # "linear_cap" or "neglog10"
):
    """
    Build matrix M (n_genotypes x n_metrics) with entries in [-1,1]:
    Sign = metric_direction (+1/-1). Magnitude = significance weight with intensity scaling.

    Now includes gradient coloring for all p-values:
    - p < 0.05: Full intensity (100%) - clearly significant
    - 0.05 <= p < 0.1: Medium intensity (60%) - trending/marginal
    - 0.1 <= p < 0.2: Low intensity (30%) - weak evidence
    - p >= 0.2: Very faint (10%) - minimal evidence

    This preserves information about near-significant effects while maintaining
    clear distinction between significant and non-significant results.
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
    cbar_label="Signed -log10(p-value)",
    # layout
    fig_size=(20, 12),
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

    # 1) Build matrix M (genotypes x metrics, values in [-1, 1])
    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df, all_metrics, only_significant_hits=only_significant_hits, alpha=alpha, weight_mode=weight_mode
    )
    if M.empty:
        print("No data available to build clustering matrix.")
        return None

    # Apply simplified nicknames to matrix index for visualization
    if nickname_mapping:
        simplified_genotypes = apply_simplified_nicknames(M.index.tolist(), nickname_mapping)
        M = M.set_axis(simplified_genotypes, axis=0)  # Use set_axis to rename index

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
    gs = gridspec.GridSpec(
        3,
        4,
        width_ratios=[1.4, 1.2, 8.0, 0.3],  # More space for heatmap: 6.0->8.0, less for nicknames
        height_ratios=[1.2, 0.2, 6.0],  # More space for metric labels: 0.8->1.2, 0.075->0.2
        wspace=0.04,  # Tighter horizontal spacing
        hspace=0.15,  # Slightly more vertical spacing
    )

    # Create axes
    ax_top_dendro = fig.add_subplot(gs[0, 2])
    ax_metric_labels = fig.add_subplot(gs[1, 2])
    ax_left_dendro = fig.add_subplot(gs[2, 0])
    ax_nicknames = fig.add_subplot(gs[2, 1])
    ax_hm = fig.add_subplot(gs[2, 2])
    ax_cbar = fig.add_subplot(gs[2, 3])

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

    # 5) Metric labels with alternating up/down positioning
    ax_metric_labels.axis("off")
    metric_positions = []  # Initialize to ensure it's available later
    if dg_col is not None:
        # Position metric labels to align with dendrogram leaves
        leaf_positions = []
        for i in range(len(col_labels_ordered)):
            leaf_x = (i * 10) + 5
            leaf_positions.append(leaf_x)

        max_pos = max(leaf_positions) if leaf_positions else 1
        metric_positions = [pos / max_pos for pos in leaf_positions]
        metric_positions = np.array(metric_positions) * 0.975

        # Place labels diagonally but with better spacing for readability
        for i, (metric_pos, metric_label) in enumerate(zip(metric_positions, col_labels_ordered)):
            ax_metric_labels.text(
                metric_pos,
                0.3,  # Fixed position for all labels
                metric_label,
                ha="center",  # Right align for diagonal text
                va="center",
                fontsize=col_label_fontsize,
                rotation=45,  # Diagonal rotation
                transform=ax_metric_labels.transAxes,
                zorder=10,  # Put text above guide lines
            )

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

    # 9) Nicknames
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

    # Set p-value oriented label and ticks
    cbar.set_label("p-value (Blue: lower, Red: higher)", fontsize=8)

    # Create custom tick labels showing p-value equivalents
    # For our log scale: matrix_value = -log10(p_value) / 3, so p_value = 10^(-matrix_value * 3)
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
        f"Metric Analysis - Two-way Dendrogram\n(Genotypes by p-values, Metrics by correlation)",
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

    # 1) Build matrix M (genotypes x metrics, values in [-1, 1])
    all_metrics = list(correlation_matrix.columns)
    M, metric_names = _build_signed_weighted_matrix(
        results_df, all_metrics, only_significant_hits=only_significant_hits, alpha=alpha, weight_mode=weight_mode
    )
    if M.empty:
        print("No data available to build heatmap matrix.")
        return None

    # Apply simplified nicknames to matrix index for visualization
    if nickname_mapping:
        simplified_genotypes = apply_simplified_nicknames(M.index.tolist(), nickname_mapping)
        M = M.set_axis(simplified_genotypes, axis=0)

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

    # 7) Customize labels and appearance
    ax_main.set_xticklabels(
        ax_main.get_xticklabels(), rotation=metric_label_rotation, ha="right", fontsize=col_label_fontsize
    )
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


def main():
    # Parse CLI arguments once here (avoid side-effects if imported)
    args = _parse_args()
    _apply_args(args)
    """Main analysis function"""
    print("�🔬 DETAILED METRIC ANALYSIS")
    print("=" * 50)
    print(f"Minimum combined consistency threshold: {MIN_COMBINED_CONSISTENCY:.0%}")
    print("=" * 50)

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
        row_metric="euclidean",
        row_linkage="ward",
        col_metric="euclidean",
        col_linkage="ward",
        fig_size=(24, 16),  # Much larger figure for better metric name spacing
        annotate=False,
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


if __name__ == "__main__":
    main()
