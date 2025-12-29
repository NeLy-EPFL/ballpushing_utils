#!/usr/bin/env python3
"""
Generate detailed PCA plots using the best configuration from consistency analysis.
Only includes genotypes with high consistency scores (≥80% by default).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import sys
import json
import os
import argparse
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, set_link_color_palette
from scipy.spatial.distance import pdist


sys.path.append("/home/matthias/ballpushing_utils")
import Config

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
CONSISTENCY_DIR = "consistency_analysis"
CONFIGS_PATH = os.path.join(SCRIPT_DIR, "multi_condition_pca_optimization/top_configurations.json")

# Consistency threshold for inclusion in detailed analysis
MIN_CONSISTENCY_PERCENT = 80.0  # Only include hits with ≥80% consistency

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate detailed PCA plots with control mode selection")
parser.add_argument("output_dir", nargs="?", default="best_pca_analysis", help="Output directory for plots")
parser.add_argument(
    "--control-mode",
    type=str,
    choices=["tailored", "emptysplit"],
    default="tailored",
    help="Control selection mode: 'tailored' for split-based controls or 'emptysplit' for universal Empty-Split control (default: tailored)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Path to pooled_summary.feather dataset (if not provided, uses hardcoded DATA_PATH)",
)
args = parser.parse_args()

# Output directory
OUTPUT_DIR = args.output_dir
CONTROL_MODE = args.control_mode

# Use dataset from command line if provided, otherwise use hardcoded path
if args.dataset:
    DATA_PATH = args.dataset
    print(f"📊 Using dataset from command line: {DATA_PATH}")
else:
    print(f"📊 Using default dataset: {DATA_PATH}")

DATA_FILES_DIR = os.path.join(OUTPUT_DIR, "data_files")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_FILES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"🎯 Output directory: {OUTPUT_DIR}")
print(f"📁 Data files directory: {DATA_FILES_DIR}")
print(f"🎨 Plots directory: {PLOTS_DIR}")
print(
    f"🎛️  Control mode: {'Tailored controls (split-based)' if CONTROL_MODE == 'tailored' else 'Empty-Split (universal control)'}"
)

# Update CONSISTENCY_DIR to look in OUTPUT_DIR/data_files first (where PCA_Static.py saves files)
# Priority: OUTPUT_DIR/data_files > default CONSISTENCY_DIR > OUTPUT_DIR
consistency_in_output_data = os.path.exists(os.path.join(OUTPUT_DIR, "data_files", "genotype_consistency_scores.csv"))
consistency_in_default = os.path.exists(os.path.join(CONSISTENCY_DIR, "genotype_consistency_scores.csv"))
consistency_in_output = os.path.exists(os.path.join(OUTPUT_DIR, "genotype_consistency_scores.csv"))

if consistency_in_output_data:
    CONSISTENCY_DIR = os.path.join(OUTPUT_DIR, "data_files")
    print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
elif consistency_in_default:
    print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
elif consistency_in_output:
    CONSISTENCY_DIR = OUTPUT_DIR
    print(f"📁 Using consistency files from: {CONSISTENCY_DIR}")
else:
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


def permutation_test(group1, group2, n_permutations=1000, random_state=None):
    """Permutation test for multivariate difference"""
    rng = np.random.default_rng(random_state)
    observed = np.linalg.norm(group1.mean(axis=0) - group2.mean(axis=0))
    combined = np.vstack([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        stat = np.linalg.norm(perm_group1.mean(axis=0) - perm_group2.mean(axis=0))
        if stat >= observed:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return observed, pval


def mahalanobis_distance(group1, group2):
    """Calculate Mahalanobis distance between two groups"""
    mean1 = group1.mean(axis=0)
    mean2 = group2.mean(axis=0)
    pooled = np.vstack([group1, group2])
    cov = np.cov(pooled, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    dist = distance.mahalanobis(mean1, mean2, inv_cov)
    return dist


def mahalanobis_permutation_test(group1, group2, n_permutations=1000, random_state=None):
    """Permutation test using Mahalanobis distance"""
    rng = np.random.default_rng(random_state)
    observed = mahalanobis_distance(group1, group2)
    combined = np.vstack([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        stat = mahalanobis_distance(perm_group1, perm_group2)
        if stat >= observed:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return observed, pval


def load_consistency_results():
    """Load consistency analysis results and filter high-consistency hits"""
    # Try new format first (enhanced_consistency_scores.csv)
    new_format_file = os.path.join(CONSISTENCY_DIR, "enhanced_consistency_scores.csv")
    old_format_file = os.path.join(CONSISTENCY_DIR, "genotype_consistency_scores.csv")

    consistency_file = None
    if os.path.exists(new_format_file):
        consistency_file = new_format_file
        use_new_format = True
        print(f"📊 Using new format consistency file: {new_format_file}")
    elif os.path.exists(old_format_file):
        consistency_file = old_format_file
        use_new_format = False
        print(f"📊 Using old format consistency file: {old_format_file}")
    else:
        print(f"❌ Consistency file not found. Tried:")
        print(f"   - {new_format_file}")
        print(f"   - {old_format_file}")
        return []

    consistency_df = pd.read_csv(consistency_file)

    # Handle different column names for different file formats
    if use_new_format:
        # New format uses Combined_Consistency (0.0-1.0 scale)
        consistency_column = "Combined_Consistency"
        threshold_value = MIN_CONSISTENCY_PERCENT / 100.0

        # Also support Optimized_Only_Consistency if Combined isn't available
        if consistency_column not in consistency_df.columns and "Optimized_Only_Consistency" in consistency_df.columns:
            consistency_column = "Optimized_Only_Consistency"
            print(f"⚠️  Using Optimized_Only_Consistency instead of Combined_Consistency")

        high_consistency = consistency_df[consistency_df[consistency_column] >= threshold_value]

        print(f"📊 CONSISTENCY FILTERING:")
        print(f"   Total genotypes in consistency analysis: {len(consistency_df)}")
        print(f"   High-consistency hits (≥{MIN_CONSISTENCY_PERCENT}%): {len(high_consistency)}")
        print(f"   Using column: {consistency_column}")

        if len(high_consistency) > 0:
            print(f"\n🏆 HIGH-CONSISTENCY HITS:")
            for _, row in high_consistency.head(20).iterrows():  # Show top 20
                cons_pct = row[consistency_column] * 100
                hit_count = row.get("Total_Hit_Count", row.get("Hit_Count", "?"))
                total_configs = row.get("Total_Configs", "?")
                print(f"   {row['Genotype']:<25}: {cons_pct:5.1f}% ({hit_count}/{total_configs})")
    else:
        # Old format uses Consistency_Percent (0-100 scale)
        high_consistency = consistency_df[consistency_df["Consistency_Percent"] >= MIN_CONSISTENCY_PERCENT]

        print(f"📊 CONSISTENCY FILTERING:")
        print(f"   Total genotypes in consistency analysis: {len(consistency_df)}")
        print(f"   High-consistency hits (≥{MIN_CONSISTENCY_PERCENT}%): {len(high_consistency)}")

        if len(high_consistency) > 0:
            print(f"\n🏆 HIGH-CONSISTENCY HITS:")
            for _, row in high_consistency.head(20).iterrows():  # Show top 20
                print(
                    f"   {row['Genotype']:<25}: {row['Consistency_Percent']:5.1f}% ({row['Hit_Count']}/{row['Total_Configs']})"
                )

    if len(high_consistency) > 20:
        print(f"   ... and {len(high_consistency) - 20} more")

    return high_consistency["Genotype"].tolist()


def select_best_pca_configuration():
    """Select the best performing PCA configuration from optimization results"""
    if not os.path.exists(CONFIGS_PATH):
        print(f"❌ Configuration file not found: {CONFIGS_PATH}")
        return None, None, None, None

    with open(CONFIGS_PATH, "r") as f:
        all_configs = json.load(f)

    # Find the configuration with the highest best_score across all conditions
    best_config = None
    best_score = -1
    best_key = None

    for config_key, config_data in all_configs.items():
        score = config_data.get("best_score", 0)
        if score > best_score:
            best_score = score
            best_config = config_data
            best_key = config_key

    if best_config is None:
        print("❌ No valid configuration found")
        return None, None, None, None

    print(f"\n🏆 BEST PCA CONFIGURATION:")
    print(f"   Configuration: {best_key}")
    print(f"   Score: {best_score:.4f}")
    print(f"   Condition: {best_config['condition']}")
    print(f"   Method: {best_config['method']}")
    print(f"   Metrics: {len(best_config['metrics'])}")

    # Get the best parameter set (rank 1)
    best_params = best_config["top_params"][0]["params"]
    print(f"   Best parameters: {best_params}")

    return (best_config["condition"], best_config["method"], best_config["metrics"], best_params)


def prepare_data():
    """Load and preprocess data (same as consistency analysis)"""
    print("📊 Loading and preprocessing data...")

    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)

    # Exclude problematic nicknames
    exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "MB247-Gal4", "854 (OK107-Gal4)", "7362 (C739-Gal4)"]
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


def run_best_pca_analysis(dataset, metrics_list, method_type, params, high_consistency_hits):
    """Run PCA analysis with the best configuration, focusing on high-consistency hits"""
    print(f"\n🔬 RUNNING BEST PCA ANALYSIS")
    print(f"   Method: {method_type}")
    print(f"   Metrics: {len(metrics_list)}")
    print(f"   Focus on {len(high_consistency_hits)} high-consistency genotypes")

    # Filter metrics and handle missing values
    available_metrics = [m for m in metrics_list if m in dataset.columns]
    na_counts = dataset[available_metrics].isna().sum()
    total_rows = len(dataset)
    missing_percentages = (na_counts / total_rows) * 100

    valid_metrics = [col for col in available_metrics if missing_percentages.loc[col] <= 5.0]
    valid_data = dataset[valid_metrics].copy()
    rows_with_missing = valid_data.isnull().any(axis=1)
    dataset_clean = dataset[~rows_with_missing].copy()

    # Prepare static metrics
    temporal_metrics = [
        col for col in valid_metrics if col.startswith(("binned_slope_", "interaction_rate_bin_", "binned_auc_"))
    ]
    static_metrics = [col for col in valid_metrics if col not in temporal_metrics]

    print(f"   Final data: {dataset_clean.shape[0]} rows, {len(static_metrics)} static metrics")

    # Scale data
    static_data = dataset_clean[static_metrics].to_numpy()
    scaler = RobustScaler()
    static_data_scaled = scaler.fit_transform(static_data)

    # Run PCA with best parameters
    if method_type == "PCA":
        pca = PCA(random_state=42, **params)
        method_name = "PCA"
    else:
        pca = SparsePCA(random_state=42, **params)
        method_name = "SparsePCA"

    pca_scores = pca.fit_transform(static_data_scaled)

    # Calculate explained variance
    if method_type == "SparsePCA":
        total_variance = np.var(static_data_scaled, axis=0).sum()
        explained_variance = []
        for component in pca.components_:
            proj_data = static_data_scaled @ component.reshape(-1, 1)
            explained_variance.append(np.var(proj_data))
        explained_variance_ratio = np.array(explained_variance) / total_variance
    else:
        explained_variance_ratio = pca.explained_variance_ratio_

    print(f"   PCA completed: {pca_scores.shape[1]} components")
    print(f"   Total explained variance: {explained_variance_ratio.sum():.3f}")

    # Create PCA scores dataframe
    n_components = pca_scores.shape[1]
    pca_scores_df = pd.DataFrame(pca_scores, columns=[f"{method_name}{i+1}" for i in range(n_components)])

    # Combine with metadata
    metadata_cols = [col for col in dataset_clean.columns if col not in valid_metrics]
    pca_with_meta = pd.concat([dataset_clean[metadata_cols].reset_index(drop=True), pca_scores_df], axis=1)

    # Save PCA results
    scores_file = os.path.join(DATA_FILES_DIR, f"best_{method_name.lower()}_scores.csv")
    pca_scores_df.to_csv(scores_file, index=False)

    loadings_file = os.path.join(DATA_FILES_DIR, f"best_{method_name.lower()}_loadings.csv")
    pca_loadings_df = pd.DataFrame(
        pca.components_, columns=static_metrics, index=[f"{method_name}{i+1}" for i in range(pca.components_.shape[0])]
    )
    pca_loadings_df.to_csv(loadings_file)

    with_meta_file = os.path.join(DATA_FILES_DIR, f"best_{method_name.lower()}_with_metadata.feather")
    pca_with_meta.to_feather(with_meta_file)

    print(f"   💾 Saved PCA results to {DATA_FILES_DIR}")

    # Run statistical analysis focusing on high-consistency hits
    selected_dims = pca_scores_df.columns
    results = []

    # Only analyze the high-consistency genotypes
    analysis_genotypes = [g for g in high_consistency_hits if g in pca_with_meta["Nickname"].values]
    print(f"   🎯 Analyzing {len(analysis_genotypes)} high-consistency genotypes")

    for nickname in analysis_genotypes:
        # Determine which control to use based on CONTROL_MODE
        if CONTROL_MODE == "emptysplit":
            force_control = "Empty-Split"
        else:
            force_control = None  # Use tailored control from split registry

        subset = Config.get_subset_data(pca_with_meta, col="Nickname", value=nickname, force_control=force_control)
        if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
            continue

        control_names = subset["Nickname"].unique()
        control_names = [n for n in control_names if n != nickname]
        if not control_names:
            continue
        control_name = control_names[0]

        # Mann-Whitney U per dimension + FDR
        mannwhitney_pvals = []
        dims_tested = []
        directions = {}

        for dim in selected_dims:
            group_scores = subset[subset["Nickname"] == nickname][dim]
            control_scores = subset[subset["Nickname"] == control_name][dim]
            if group_scores.empty or control_scores.empty:
                continue

            stat, pval = mannwhitneyu(group_scores, control_scores, alternative="two-sided")
            mannwhitney_pvals.append(pval)
            dims_tested.append(dim)

            # Calculate direction
            directions[dim] = 1 if group_scores.mean() > control_scores.mean() else -1

        if len(mannwhitney_pvals) > 0:
            rejected, pvals_corr, _, _ = multipletests(mannwhitney_pvals, alpha=0.05, method="fdr_bh")
            significant_dims = [dims_tested[i] for i, rej in enumerate(rejected) if rej]
            mannwhitney_any = any(rejected)
        else:
            mannwhitney_any = False
            significant_dims = []
            pvals_corr = []

        # Multivariate tests
        group_matrix = subset[subset["Nickname"] == nickname][selected_dims].values
        control_matrix = subset[subset["Nickname"] == control_name][selected_dims].values

        if group_matrix.size == 0 or control_matrix.size == 0:
            continue

        perm_stat, perm_pval = permutation_test(group_matrix, control_matrix, random_state=42)
        maha_stat, maha_pval = mahalanobis_permutation_test(group_matrix, control_matrix, random_state=42)

        # Build result with PC-specific information
        result_dict = {
            "genotype": nickname,
            "control": control_name,
            "MannWhitney_any_dim_significant": mannwhitney_any,
            "MannWhitney_significant_dims": significant_dims,
            "num_significant_PCs": len(significant_dims),
            "Permutation_pval": perm_pval,
            "Mahalanobis_pval": maha_pval,
            "significant": perm_pval < 0.05,  # Use permutation test as main criterion
        }

        # Add PC-specific results
        for i, dim in enumerate(dims_tested):
            pc_name = dim.replace(method_name, "PC")  # Standardize to PC1, PC2, etc.
            if i < len(mannwhitney_pvals) and i < len(pvals_corr):
                result_dict[f"{pc_name}_pval"] = mannwhitney_pvals[i]
                result_dict[f"{pc_name}_pval_corrected"] = pvals_corr[i]
                result_dict[f"{pc_name}_significant"] = dim in significant_dims
                result_dict[f"{pc_name}_direction"] = directions.get(dim, 0)

        results.append(result_dict)

    if not results:
        print("   ⚠️  No statistical results generated")
        return pd.DataFrame(), method_name.lower()

    results_df = pd.DataFrame(results)

    # Apply FDR correction to multivariate tests
    for col in ["Permutation_pval", "Mahalanobis_pval"]:
        rejected, pvals_corrected, _, _ = multipletests(results_df[col], alpha=0.05, method="fdr_bh")
        results_df[col.replace("_pval", "_FDR_significant")] = rejected

    # Save results
    results_file = os.path.join(DATA_FILES_DIR, f"best_{method_name.lower()}_stats_results.csv")
    results_df.to_csv(results_file, index=False)

    print(f"   📊 Found {len(results_df)} genotypes in analysis")
    print(f"   🎯 Significant hits: {len(results_df[results_df['significant']])}")
    print(f"   💾 Results saved to {results_file}")

    return results_df, method_name.lower()


def create_hits_heatmap(results_df, method_name, nickname_to_brainregion, color_dict):
    """Create detailed heatmap for high-consistency hits only"""

    if len(results_df[results_df["significant"]]) == 0:
        print("No significant hits to visualize")
        return None

    # Filter to significant results and sort by brain region
    sig_df = results_df[results_df["significant"]].copy()
    sig_df = sig_df.set_index("genotype")
    sorted_df = reorder_by_brain_region(sig_df, nickname_to_brainregion).reset_index()

    # Detect PC columns
    pc_columns = [col for col in results_df.columns if col.endswith("_significant") and col.startswith("PC")]
    pc_names = [col.replace("_significant", "") for col in pc_columns]

    # Sort PC names numerically
    def pc_sort_key(pc):
        import re

        m = re.match(r"PC(\d+)", pc)
        return int(m.group(1)) if m else 999

    pc_names_sorted = sorted(pc_names, key=pc_sort_key)

    # Create matrices for visualization
    significance_matrix = []
    annotation_matrix = []
    genotype_names = []

    for _, row in sorted_df.iterrows():
        genotype_names.append(row["genotype"])

        sig_row = []
        annot_row = []

        for pc in pc_names_sorted:
            pc_sig_col = f"{pc}_significant"
            pc_pval_col = f"{pc}_pval_corrected"
            pc_dir_col = f"{pc}_direction"

            is_significant = row.get(pc_sig_col, False)
            pval_corrected = row.get(pc_pval_col, np.nan)
            direction = row.get(pc_dir_col, 0)

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
            else:
                sig_row.append(0)  # No significance
                annot_row.append("")

        significance_matrix.append(sig_row)
        annotation_matrix.append(annot_row)

    if len(significance_matrix) == 0:
        print("No hits matrix to create")
        return None

    # Create DataFrames
    sig_df = pd.DataFrame(significance_matrix, index=sorted_df["genotype"], columns=pc_names_sorted)
    annot_df = pd.DataFrame(annotation_matrix, index=sorted_df["genotype"], columns=pc_names_sorted)

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(max(14, len(pc_names_sorted) * 0.8), max(8, len(genotype_names) * 0.4)))

    # Create custom colormap
    colors = ["lightblue", "white", "lightcoral"]
    cmap = ListedColormap(colors)

    # Create the heatmap
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
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    # Color y-tick labels by brain region
    colour_y_ticklabels(ax, nickname_to_brainregion, color_dict)

    # Customize the plot
    method_display = "Sparse PCA" if method_name == "sparsepca" else "Regular PCA"
    ax.set_title(
        f"High-Consistency Hits - Detailed PC Analysis\n"
        f"Best {method_display} Configuration (≥{MIN_CONSISTENCY_PERCENT}% consistency)\n"
        f"{len(genotype_names)} genotypes across {len(pc_names_sorted)} PCs",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Principal Components", fontsize=12, fontweight="bold")
    ax.set_ylabel("Genotypes (High-Consistency Hits)", fontsize=12, fontweight="bold")

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Add legend
    legend_elements = [
        Patch(facecolor="lightcoral", label="Upregulated vs Control"),
        Patch(facecolor="lightblue", label="Downregulated vs Control"),
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
    png_file = os.path.join(PLOTS_DIR, f"best_{method_name}_high_consistency_heatmap.png")
    pdf_file = os.path.join(PLOTS_DIR, f"best_{method_name}_high_consistency_heatmap.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"   💾 Detailed heatmap saved: {png_file} and {pdf_file}")

    return fig


def create_explained_variance_plot(explained_variance_ratio, method_name):
    """Create explained variance plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(explained_variance_ratio * 100), "o-", linewidth=2, markersize=6)
    plt.xlabel(f"Principal Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title(f"Best Configuration - Cumulative Explained Variance\n{method_name.upper()}")
    plt.grid(True, alpha=0.3)

    # Add text annotations for key components
    cum_var = np.cumsum(explained_variance_ratio * 100)
    for i in range(min(5, len(cum_var))):
        plt.annotate(f"{cum_var[i]:.1f}%", (i, cum_var[i]), textcoords="offset points", xytext=(0, 10), ha="center")

    variance_file = os.path.join(PLOTS_DIR, f"best_{method_name}_explained_variance.png")
    plt.savefig(variance_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"   📈 Explained variance plot saved: {variance_file}")


def _collect_pc_columns(results_df):
    """Return sorted PC names ['PC1','PC2',...] present in results_df."""
    pc_signif_cols = [c for c in results_df.columns if c.startswith("PC") and c.endswith("_significant")]
    pc_names = [c.replace("_significant", "") for c in pc_signif_cols]

    def pc_sort_key(pc):
        import re

        m = re.match(r"PC(\d+)$", pc)
        return int(m.group(1)) if m else 99999

    pc_names = sorted(set(pc_names), key=pc_sort_key)
    return pc_names


def _build_signed_weighted_matrix(
    results_df,
    only_significant_hits=True,
    alpha=0.05,
    weight_mode="linear_cap",  # "linear_cap" or "neglog10"
):
    """
    Build matrix M (n_genotypes x n_PCs) with entries in [-1,1]:
    Sign = PCk_direction (+1/-1). Magnitude = significance weight.
    Only PCs that pass per-PC FDR in results_df get nonzero weight.
    """
    if only_significant_hits:
        df = results_df[results_df["significant"]].copy()
    else:
        df = results_df.copy()
    if df.empty:
        return pd.DataFrame(), []

    pc_names = _collect_pc_columns(results_df)
    if not pc_names:
        return pd.DataFrame(), []

    M = pd.DataFrame(0.0, index=df["genotype"].values, columns=pc_names)

    for _, row in df.iterrows():
        for pc in pc_names:
            sig_col = f"{pc}_significant"
            padj_col = f"{pc}_pval_corrected"
            dir_col = f"{pc}_direction"
            is_sig = bool(row.get(sig_col, False))
            p_adj = row.get(padj_col, np.nan)
            direction = row.get(dir_col, 0)
            if is_sig and pd.notna(p_adj) and direction != 0:
                if weight_mode == "linear_cap":
                    w = max(0.0, 1.0 - (p_adj / alpha))
                    w = min(1.0, w)
                elif weight_mode == "neglog10":
                    p_adj = max(float(p_adj), 1e-300)
                    denom = -np.log10(alpha)
                    denom = denom if denom > 0 else 1.0
                    w = min(1.0, (-np.log10(p_adj)) / denom)
                else:
                    raise ValueError("Unknown weight_mode")
                M.loc[row["genotype"], pc] = np.sign(direction) * float(w)
    return M, pc_names


def create_hits_dendrograms_no_heatmap(
    results_df,
    method_name,
    OUTPUT_DIR,
    only_significant_hits=True,
    alpha=0.05,
    weight_mode="linear_cap",
    row_distance_metric="euclidean",
    row_linkage_method="ward",
    col_distance_metric="euclidean",
    col_linkage_method="ward",
    figsize_rows=(6.0, 10.0),  # (width, height) for row dendrogram
    figsize_cols=(10.0, 4.0),  # (width, height) for column dendrogram
):
    """
    Create and save two dendrograms:
      - Row dendrogram (genotypes), orientation='left'
      - Column dendrogram (PCs), orientation='top'
    Also saves:
      - matrix CSV (input to clustering)
      - row and column leaf order CSVs
      - scipy linkage arrays (as .npy)
    """
    # 1) Build signed, p-weighted matrix
    M, pc_names = _build_signed_weighted_matrix(
        results_df, only_significant_hits=only_significant_hits, alpha=alpha, weight_mode=weight_mode
    )
    if M.empty:
        print("No data available to build clustering matrix.")
        return None, None

    genotypes = M.index.tolist()

    # 2) Row clustering (genotypes)
    if len(genotypes) >= 2:
        D_rows = pdist(M.values, metric=row_distance_metric)
        Z_rows = linkage(D_rows, method=row_linkage_method)
        row_order = leaves_list(Z_rows)
        ordered_genotypes = [genotypes[i] for i in row_order]
    else:
        Z_rows = None
        ordered_genotypes = genotypes

    # 3) Column clustering (PCs)
    if len(pc_names) >= 2:
        D_cols = pdist(M.values.T, metric=col_distance_metric)
        Z_cols = linkage(D_cols, method=col_linkage_method)
        col_order = leaves_list(Z_cols)
        ordered_pcs = [pc_names[i] for i in col_order]
    else:
        Z_cols = None
        ordered_pcs = pc_names

    # 4) Plot and save ROW dendrogram
    plt.style.use("default")
    fig_rows = plt.figure(figsize=figsize_rows)
    ax_r = fig_rows.add_subplot(111)
    if Z_rows is not None:
        dendrogram(Z_rows, orientation="left", labels=[genotypes[i] for i in row_order], leaf_font_size=8, ax=ax_r)
        ax_r.set_title("Genotype dendrogram (row clustering)", fontsize=12, fontweight="bold")
        ax_r.set_xlabel("Distance")
        ax_r.set_ylabel("Genotypes")
    else:
        ax_r.text(0.5, 0.5, "Not enough genotypes to cluster", ha="center", va="center", transform=ax_r.transAxes)
        ax_r.axis("off")
    fig_rows.tight_layout()
    rows_png = os.path.join(OUTPUT_DIR, f"best_{method_name}_row_dendrogram.png")
    rows_pdf = os.path.join(OUTPUT_DIR, f"best_{method_name}_row_dendrogram.pdf")
    fig_rows.savefig(rows_png, dpi=300, bbox_inches="tight")
    fig_rows.savefig(rows_pdf, bbox_inches="tight")
    plt.close(fig_rows)

    # 5) Plot and save COLUMN dendrogram
    fig_cols = plt.figure(figsize=figsize_cols)
    ax_c = fig_cols.add_subplot(111)
    if Z_cols is not None:
        dendrogram(
            Z_cols,
            orientation="top",
            labels=[pc_names[i] for i in col_order],
            leaf_rotation=0,
            leaf_font_size=9,
            ax=ax_c,
        )
        ax_c.set_title("PC dendrogram (column clustering)", fontsize=12, fontweight="bold")
        ax_c.set_ylabel("Distance")
        ax_c.set_xlabel("PCs")
    else:
        ax_c.text(0.5, 0.5, "Not enough PCs to cluster", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.axis("off")
    fig_cols.tight_layout()
    cols_png = os.path.join(OUTPUT_DIR, f"best_{method_name}_column_dendrogram.png")
    cols_pdf = os.path.join(OUTPUT_DIR, f"best_{method_name}_column_dendrogram.pdf")
    fig_cols.savefig(cols_png, dpi=300, bbox_inches="tight")
    fig_cols.savefig(cols_pdf, bbox_inches="tight")
    plt.close(fig_cols)

    # 6) Save matrix and orders to CSV; linkages to .npy (optional)
    mat_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_hits_cluster_matrix.csv")
    row_order_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_row_order.csv")
    col_order_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_col_order.csv")
    M.to_csv(mat_csv)
    pd.Series(ordered_genotypes, name="genotype").to_csv(row_order_csv, index=False)
    pd.Series(ordered_pcs, name="PC").to_csv(col_order_csv, index=False)

    if "np" in globals():
        if Z_rows is not None:
            np.save(os.path.join(OUTPUT_DIR, f"best_{method_name}_row_linkage.npy"), Z_rows)
        if Z_cols is not None:
            np.save(os.path.join(OUTPUT_DIR, f"best_{method_name}_col_linkage.npy"), Z_cols)

    print(f"   💾 Row dendrogram saved: {rows_png} and {rows_pdf}")
    print(f"   💾 Column dendrogram saved: {cols_png} and {cols_pdf}")
    print(f"   💾 Clustering matrix saved: {mat_csv}")
    print(f"   💾 Row order CSV: {row_order_csv}")
    print(f"   💾 Column order CSV: {col_order_csv}")

    return {
        "matrix": M,
        "row_order": ordered_genotypes,
        "col_order": ordered_pcs,
        "row_linkage": Z_rows,
        "col_linkage": Z_cols,
    }


def plot_two_way_dendrogram(
    M,  # DataFrame: genotypes x PCs, values in [-1,1]
    OUTPUT_DIR,
    method_name,
    row_metric="euclidean",
    row_linkage="ward",
    col_metric="euclidean",
    col_linkage="ward",
    cmap=plt.get_cmap("RdBu_r"),
    vmin=-1,
    vmax=1,
):
    # Precompute linkages to ensure Ward+Euclidean correctness
    row_Z = linkage(pdist(M.values, metric=row_metric), method=row_linkage) if M.shape[0] > 1 else None
    col_Z = linkage(pdist(M.values.T, metric=col_metric), method=col_linkage) if M.shape[1] > 1 else None

    # Note: clustermap does not support branch coloring like scipy dendrogram, but we can color y-tick labels by brain region
    g = sns.clustermap(
        M,
        row_linkage=row_Z,
        col_linkage=col_Z,
        method=row_linkage,  # kept for metadata; actual linkages provided
        metric=row_metric,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "Signed p-weight (graded by p-value)"},
        figsize=(max(8, 0.35 * M.shape[1] + 3), max(8, 0.25 * M.shape[0] + 3)),
        yticklabels=True,  # ensure row labels are shown on the left
    )
    # Color y-tick labels by brain region if available
    try:
        from matplotlib import pyplot as plt

        # Try to get nickname_to_brainregion and color_dict from globals
        nickname_to_brainregion = globals().get("nickname_to_brainregion", {})
        color_dict = globals().get("color_dict", {})
        if nickname_to_brainregion and color_dict:
            colour_y_ticklabels(g.ax_heatmap, nickname_to_brainregion, color_dict)
    except Exception as e:
        print(f"[WARN] Could not color y-tick labels in two-way dendrogram: {e}")

    g.fig.suptitle("Two-way dendrogram (genotypes × PCs)", y=1.02, fontsize=14, fontweight="bold")

    png = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_dendrogram.png")
    pdf = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_dendrogram.pdf")
    g.savefig(png, dpi=300, bbox_inches="tight")
    g.savefig(pdf, bbox_inches="tight")
    print(f"   💾 Two-way dendrogram saved: {png} and {pdf}")

    # Get orders
    row_order = [M.index[i] for i in g.dendrogram_row.reordered_ind] if M.shape[0] > 1 else list(M.index)
    col_order = [M.columns[i] for i in g.dendrogram_col.reordered_ind] if M.shape[1] > 1 else list(M.columns)
    return {"row_order": row_order, "col_order": col_order}


def plot_two_way_dendrogram_custom(
    results_df,
    method_name,
    OUTPUT_DIR,
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
    cmap=ListedColormap(["#6baed6", "#f7fbff", "#fcae91"]),
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.4,
    linecolor="gray",
    cbar_label="Signed p-weight",
    # layout
    fig_size=(20, 12),
    # label styling
    row_label_fontsize=9,
    col_label_fontsize=9,
    pc_label_rotation=45,
    truncate_row_labels=None,
    wrap_labels=True,
    annotate=False,
    despine=True,
):
    """
    Custom 2-way dendrogram with perfect alignment and optimal spacing.
    """
    # 1) Build matrix M (genotypes x PCs, values in [-1, 1])
    M, pc_names = _build_signed_weighted_matrix(
        results_df, only_significant_hits=only_significant_hits, alpha=alpha, weight_mode=weight_mode
    )
    if M.empty:
        print("No data available to build clustering matrix.")
        return None

    # 2) Linkages (allow callers to override precomputed linkages via kwargs row_Z / col_Z)
    # If the caller supplied row_Z/col_Z (positional kwargs), use them; otherwise compute from M
    # Note: function signature accepts optional row_Z and col_Z at the end of kwargs
    row_Z = globals().get("_plot_custom_row_Z", None)
    col_Z = globals().get("_plot_custom_col_Z", None)
    # If caller did not set the globals, compute from the matrix
    if row_Z is None:
        row_Z = linkage(pdist(M.values, metric=row_metric), method=row_linkage) if M.shape[0] > 1 else None
    if col_Z is None:
        col_Z = linkage(pdist(M.values.T, metric=col_metric), method=col_linkage) if M.shape[1] > 1 else None

    # 3) BALANCED GridSpec layout: more space for nicknames, proper spacing
    plt.style.use("default")
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(
        3,
        4,
        width_ratios=[1.6, 1.4, 6.0, 0.3],  # INCREASED nickname space: 1.0→1.4, left dendro: 1.2→1.7
        height_ratios=[0.8, 0.075, 6.0],  # balanced heights
        wspace=0.06,
        hspace=0.08,  # slightly more spacing
    )

    # Create axes
    ax_top_dendro = fig.add_subplot(gs[0, 2])
    ax_pc_labels = fig.add_subplot(gs[1, 2])
    ax_left_dendro = fig.add_subplot(gs[2, 0])
    ax_nicknames = fig.add_subplot(gs[2, 1])
    ax_hm = fig.add_subplot(gs[2, 2])
    ax_cbar = fig.add_subplot(gs[2, 3])

    # 4) Top dendrogram (horizontal)
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
        col_labels_ordered = [M.columns[i] for i in col_order_idx]
    else:
        col_order_idx = list(range(M.shape[1]))
        col_labels_ordered = list(M.columns)
        dg_col = None
        ax_top_dendro.axis("off")

    # 5) PC labels with CORRECTED leaf alignment
    ax_pc_labels.axis("off")
    if dg_col is not None:
        # FIXED: Extract exact leaf positions from dendrogram
        # Each leaf corresponds to the midpoint of its terminal branch
        leaf_positions = []
        for i in range(len(col_labels_ordered)):
            # Use the fact that leaves are positioned at i*10 + 5 in scipy's coordinate system
            leaf_x = (i * 10) + 5
            leaf_positions.append(leaf_x)

        # Normalize to [0, 1] range and adjust to match axis coordinates
        max_pos = max(leaf_positions)
        pc_positions = [pos / max_pos for pos in leaf_positions]

        # Fine-tune positioning to align perfectly with dendrogram leaves
        # Adjust the range to match the dendrogram's actual span
        pc_positions = np.array(pc_positions)
        pc_positions = 0 + (pc_positions * 0.975)  # map to [0.05, 0.95] range

        # Place PC labels at exact dendrogram leaf positions
        for i, (pc_pos, pc_label) in enumerate(zip(pc_positions, col_labels_ordered)):
            ax_pc_labels.text(
                pc_pos,
                0.5,
                pc_label,
                ha="center",
                va="center",
                fontsize=col_label_fontsize,
                rotation=pc_label_rotation,
                transform=ax_pc_labels.transAxes,
            )

    # 6) Left dendrogram (vertical)
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

        max_width = 20  # good balance for readability
        row_labels_display = ["\n".join(textwrap.wrap(lbl, max_width)) for lbl in row_labels_ordered]
    else:

        def _truncate(s, n):
            if truncate_row_labels is None or n is None:
                return s
            return (s[: n - 1] + "…") if isinstance(s, str) and len(s) > n else s

        row_labels_display = [_truncate(lbl, truncate_row_labels) for lbl in row_labels_ordered]

    # 8) Create heatmap
    M_ord = M.iloc[row_order_idx, col_order_idx]

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

    # 9) Nicknames with PROPER spacing - no overlap with dendrogram
    ax_nicknames.axis("off")

    # Get heatmap y-axis limits for perfect alignment
    heatmap_ylim = ax_hm.get_ylim()
    ax_nicknames.set_ylim(heatmap_ylim)

    # Position nicknames to align with heatmap rows
    y_positions = np.linspace(heatmap_ylim[0] - 0.5, heatmap_ylim[1] + 0.5, len(row_labels_display))

    for y_pos, nickname in zip(y_positions, row_labels_display):
        ax_nicknames.text(
            0.95,
            y_pos,
            nickname,  # RESTORED to 0.95 for proper spacing from dendrogram
            ha="right",
            va="center",
            fontsize=row_label_fontsize,
            transform=ax_nicknames.transData,
        )

    # 10) Ultra-compact colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    pos = ax_cbar.get_position()
    new_height = pos.height * 0.35
    new_y = pos.y0 + (pos.height - new_height) / 2
    ax_cbar.set_position([pos.x0, new_y, pos.width, new_height])

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=8)

    # 11) Clean up dendrogram axes
    for ax in [ax_top_dendro, ax_left_dendro]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 12) Color nickname labels by brain region if available
    try:
        nickname_to_brainregion = globals().get("nickname_to_brainregion", {})
        color_dict = globals().get("color_dict", {})
        if nickname_to_brainregion and color_dict:
            for i, (nickname, text_obj) in enumerate(zip(row_labels_ordered, ax_nicknames.texts)):
                region = nickname_to_brainregion.get(nickname, None)
                if region in color_dict:
                    text_obj.set_color(color_dict[region])
    except Exception:
        pass

    # 13) Final styling
    if despine:
        sns.despine(ax=ax_hm, top=True, right=True, left=False, bottom=False)

    method_display = "Sparse PCA" if method_name == "sparsepca" else "Regular PCA"
    fig.suptitle(f"", fontsize=14, fontweight="bold", y=0.98)

    # 14) Save files
    png = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_custom.png")
    pdf = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_custom.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # 15) Save matrix and orders
    mat_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_matrix.csv")
    row_order_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_row_order.csv")
    col_order_csv = os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_col_order.csv")
    M.to_csv(mat_csv)
    pd.Series(row_labels_ordered, name="genotype").to_csv(row_order_csv, index=False)
    pd.Series(col_labels_ordered, name="PC").to_csv(col_order_csv, index=False)

    if row_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_row_linkage.npy"), row_Z)
    if col_Z is not None:
        np.save(os.path.join(OUTPUT_DIR, f"best_{method_name}_two_way_col_linkage.npy"), col_Z)

    print(f"   💾 Saved two-way custom dendrogram: {png} / {pdf}")
    print(f"   💾 Matrix CSV: {mat_csv}")
    print(f"   💾 Row/Col orders: {row_order_csv} / {col_order_csv}")

    return {
        "matrix": M,
        "row_Z": row_Z,
        "col_Z": col_Z,
        "row_order": row_labels_ordered,
        "col_order": col_labels_ordered,
    }


def plot_two_way_dendrogram_mixed(
    results_df,
    method_name,
    OUTPUT_DIR,
    loadings_csv_path,
    only_significant_hits=True,
    alpha=0.05,
    weight_mode="linear_cap",
    # clustering choices
    row_metric="euclidean",
    row_linkage="ward",
    col_metric="euclidean",
    col_linkage="ward",
    # plotting passthroughs
    **plot_kwargs,
):
    """
    Two-way dendrogram where rows (genotypes) are clustered using the signed p-weight matrix
    (same as the other functions), but columns (PCs) are clustered by similarity of their
    loadings (read from `loadings_csv_path`). The loadings CSV should have PC rows and feature
    columns (this matches the saved output from the PCA step).

    Returns the same dict as `plot_two_way_dendrogram_custom`.
    """
    # Build M and pc_names
    M, pc_names = _build_signed_weighted_matrix(
        results_df, only_significant_hits=only_significant_hits, alpha=alpha, weight_mode=weight_mode
    )
    if M.empty:
        print("No data available to build clustering matrix (mixed).")
        return None

    # Row linkage from the signed p-weight matrix
    row_Z = linkage(pdist(M.values, metric=row_metric), method=row_linkage) if M.shape[0] > 1 else None

    # Load loadings CSV and construct PC x features matrix aligned to pc_names
    if not os.path.exists(loadings_csv_path):
        print(f"Loadings file not found: {loadings_csv_path}")
        return None

    loadings_df = pd.read_csv(loadings_csv_path, index_col=0)

    # Attempt to align loadings rows to pc_names by numeric suffix (e.g. PCA1 / PC1)
    def find_row_for_pc(pc_label, loadings_index):
        import re

        m = re.match(r"PC(\d+)$", pc_label)
        if not m:
            return None
        idx_num = m.group(1)
        # Try exact match first
        if pc_label in loadings_index:
            return pc_label
        # Try any index that ends with the same number
        for r in loadings_index:
            m2 = re.match(r".*?(\d+)$", str(r))
            if m2 and m2.group(1) == idx_num:
                return r
        return None

    loadings_index = list(loadings_df.index.astype(str))
    pc_row_names = []
    missing = []
    for pc in pc_names:
        r = find_row_for_pc(pc, loadings_index)
        if r is None:
            missing.append(pc)
            pc_row_names.append(None)
        else:
            pc_row_names.append(r)

    if any(x is None for x in pc_row_names):
        print(f"Warning: could not find loadings rows for PCs: {missing}. Falling back to clustering PCs by M.T")
        col_Z = linkage(pdist(M.values.T, metric=col_metric), method=col_linkage) if M.shape[1] > 1 else None
    else:
        # Build matrix where rows are in pc_names order.
        # Use absolute loadings to break orthogonality and normalize rows so clustering focuses
        # on profile shape rather than raw scale.
        col_matrix = loadings_df.loc[pc_row_names].values.astype(float)
        # take absolute values
        col_matrix = np.abs(col_matrix)
        # Replace NaNs with zero
        if np.isnan(col_matrix).any():
            col_matrix = np.nan_to_num(col_matrix)
        # Normalize rows to unit L2 norm to focus on relative feature contributions
        row_norms = np.linalg.norm(col_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        col_matrix = col_matrix / row_norms
        col_Z = linkage(pdist(col_matrix, metric=col_metric), method=col_linkage) if col_matrix.shape[0] > 1 else None

    # Use the existing custom plot by temporarily passing precomputed linkages via globals
    # (less invasive than refactoring the function signature in many call sites)
    prev_row_Z = globals().get("_plot_custom_row_Z", None)
    prev_col_Z = globals().get("_plot_custom_col_Z", None)
    try:
        globals()["_plot_custom_row_Z"] = row_Z
        globals()["_plot_custom_col_Z"] = col_Z
        out = plot_two_way_dendrogram_custom(
            results_df,
            method_name,
            OUTPUT_DIR,
            only_significant_hits=only_significant_hits,
            alpha=alpha,
            weight_mode=weight_mode,
            row_metric=row_metric,
            row_linkage=row_linkage,
            col_metric=col_metric,
            col_linkage=col_linkage,
            **plot_kwargs,
        )
    finally:
        # restore globals
        if prev_row_Z is None:
            globals().pop("_plot_custom_row_Z", None)
        else:
            globals()["_plot_custom_row_Z"] = prev_row_Z
        if prev_col_Z is None:
            globals().pop("_plot_custom_col_Z", None)
        else:
            globals()["_plot_custom_col_Z"] = prev_col_Z

    return out


def main():
    """Main analysis function"""
    print("🏆 BEST PCA CONFIGURATION DETAILED ANALYSIS")
    print("=" * 80)
    print(f"Minimum consistency threshold: {MIN_CONSISTENCY_PERCENT}%")
    print("=" * 80)

    # Step 1: Load high-consistency hits
    high_consistency_hits = load_consistency_results()
    if not high_consistency_hits:
        print("❌ No high-consistency hits found. Cannot proceed.")
        return

    # Step 2: Select best PCA configuration
    condition, method_type, metrics_list, best_params = select_best_pca_configuration()
    if condition is None:
        print("❌ Could not select best configuration. Cannot proceed.")
        return

    # Step 3: Prepare data
    dataset = prepare_data()

    # Step 4: Run PCA analysis with best configuration
    results_df, method_name = run_best_pca_analysis(
        dataset, metrics_list, method_type, best_params, high_consistency_hits
    )

    if results_df.empty:
        print("❌ No results generated from PCA analysis.")
        return

    # Step 5: Create detailed visualizations
    print(f"\n🎨 CREATING DETAILED VISUALIZATIONS")

    # Explained variance plot (if we have the data)
    # Note: We'd need to modify run_best_pca_analysis to return this

    # Create hits heatmap
    create_hits_heatmap(results_df, method_name, nickname_to_brainregion, color_dict)

    info = create_hits_dendrograms_no_heatmap(
        results_df,
        method_name,
        OUTPUT_DIR,
        only_significant_hits=True,  # keep parity with your heatmap subset
        alpha=0.05,
        weight_mode="linear_cap",  # or "neglog10"
        row_distance_metric="euclidean",
        row_linkage_method="ward",
        col_distance_metric="euclidean",
        col_linkage_method="ward",
    )

    # Also create two-way dendrogram
    if info and "matrix" in info:
        plot_two_way_dendrogram(
            info["matrix"],
            OUTPUT_DIR,
            method_name,
            row_metric="euclidean",
            row_linkage="ward",
            col_metric="euclidean",
            col_linkage="ward",
            cmap=plt.get_cmap("RdBu_r"),
            vmin=-1,
            vmax=1,
        )

        # Make the custom one

        plot_two_way_dendrogram_custom(
            results_df,
            method_name,
            OUTPUT_DIR,
            only_significant_hits=True,
            alpha=0.05,
            weight_mode="linear_cap",
            row_metric="euclidean",
            row_linkage="ward",
            col_metric="euclidean",
            col_linkage="ward",
            color_threshold_rows="default",
            color_threshold_cols="default",
            row_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
            col_palette=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
            above_threshold_color_rows="C0",
            above_threshold_color_cols="C0",
            cmap=ListedColormap(["#6baed6", "#f7fbff", "#fcae91"]),
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.4,
            linecolor="gray",
            cbar_label="Signed p-weight",
            fig_size=(12, 12),
            annotate=False,
        )

        # If explicit loadings CSV is available, create a mixed dendrogram:
        loadings_csv = "/home/matthias/ballpushing_utils/src/PCA/best_pca_analysis/best_pca_loadings.csv"
        if os.path.exists(loadings_csv):
            print(f"\n🎯 Creating mixed dendrogram using PC loadings from: {loadings_csv}")
            plot_two_way_dendrogram_mixed(
                results_df,
                method_name,
                OUTPUT_DIR,
                loadings_csv_path=loadings_csv,
                only_significant_hits=True,
                alpha=0.05,
                weight_mode="linear_cap",
                row_metric="euclidean",
                row_linkage="ward",
                col_metric="euclidean",
                col_linkage="ward",
                fig_size=(12, 12),
            )

    # Summary statistics
    total_hits = len(results_df[results_df["significant"]])
    total_pcs_significant = results_df[results_df["significant"]]["num_significant_PCs"].sum()

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Best configuration: {condition}_{method_type}")
    print(f"   High-consistency genotypes analyzed: {len(high_consistency_hits)}")
    print(f"   Significant genotypes in best analysis: {total_hits}")
    print(f"   Total significant PCs across all hits: {total_pcs_significant}")
    print(
        f"   Average significant PCs per hit: {total_pcs_significant/total_hits:.1f}"
        if total_hits > 0
        else "   No hits to analyze"
    )

    # Create summary file
    summary_file = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("BEST PCA CONFIGURATION - DETAILED ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Consistency threshold: ≥{MIN_CONSISTENCY_PERCENT}%\n")
        f.write(f"High-consistency hits identified: {len(high_consistency_hits)}\n")
        f.write(f"Best configuration: {condition}_{method_type}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Metrics used: {len(metrics_list)}\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"Significant genotypes: {total_hits}\n")
        f.write(f"Total significant PCs: {total_pcs_significant}\n")
        f.write(
            f"Average PCs per hit: {total_pcs_significant/total_hits:.1f}\n"
            if total_hits > 0
            else "No significant hits\n"
        )

        if total_hits > 0:
            f.write(f"\nSIGNIFICANT GENOTYPES:\n")
            for _, row in results_df[results_df["significant"]].iterrows():
                f.write(f"  {row['genotype']}: {row['num_significant_PCs']} significant PCs\n")

    print(f"\n✅ DETAILED ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
