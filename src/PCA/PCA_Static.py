#!/usr/bin/env python3
"""
Enhanced Consistency Analysis with Edge Cases
Runs PCA analysis with optimized configurations PLUS edge case scenarios:
- Fixed components (10, 15) with optimized metric lists
- Fixed components (10, 15) with full unfiltered metrics
- Both PCA and SparsePCA for all scenarios
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance
import sys
import os
import matplotlib.pyplot as plt
import Config
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
CONFIGS_PATH = "multi_condition_pca_optimization/top_configurations.json"
FULL_METRICS_PATH = "/home/matthias/ballpushing_utils/src/PCA/full_metrics_pca.txt"

# Edge case testing parameters
EDGE_CASE_COMPONENTS = [10, 15]  # Fixed component counts to test
INCLUDE_EDGE_CASES = False  # Enable/disable edge case testing

# Output directory
if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
else:
    OUTPUT_DIR = "consistency_analysis_with_edges"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🎯 Output directory: {OUTPUT_DIR}")


def load_metrics_list(path):
    """Load metrics list from file"""
    if not os.path.exists(path):
        print(f"⚠️ Metrics file not found: {path}")
        return None
    with open(path, "r") as f:
        metrics = [line.strip() for line in f if line.strip()]
    return metrics


def generate_edge_case_configs(optimized_configs, full_metrics_list):
    """
    Generate edge case configurations for supplementary analysis

    Edge cases include:
    1. Fixed components (10, 15) with optimized metric lists
    2. Fixed components (10, 15) with full unfiltered metrics
    3. Both scenarios for PCA and SparsePCA

    Returns: Dictionary of edge case configurations
    """
    print("\n🔧 Generating edge case configurations...")

    edge_configs = {}

    # Get unique conditions and methods from optimized configs
    unique_conditions = set()
    unique_methods = set()

    for config_key, config_data in optimized_configs.items():
        unique_conditions.add(config_data["condition"])
        unique_methods.add(config_data["method"])

    print(f"   Found conditions: {list(unique_conditions)}")
    print(f"   Found methods: {list(unique_methods)}")

    edge_case_counter = 1

    for condition in unique_conditions:
        # Get the original metrics for this condition
        original_metrics = None
        for config_data in optimized_configs.values():
            if config_data["condition"] == condition:
                original_metrics = config_data["metrics"]
                break

        if not original_metrics:
            continue

        for method in unique_methods:
            for n_comp in EDGE_CASE_COMPONENTS:
                # Edge case 1: Fixed components with optimized metrics
                edge_key = f"EdgeCase_{edge_case_counter:02d}_{condition}_{method}_fixed{n_comp}comp"

                # Basic parameters for the method
                if method == "PCA":
                    base_params = {"n_components": n_comp}
                else:  # SparsePCA
                    # Use reasonable default SparsePCA parameters
                    base_params = {
                        "n_components": n_comp,
                        "alpha": 1.0,  # Moderate sparsity
                        "ridge_alpha": 0.1,
                        "method": "lars",
                        "max_iter": 2000,
                        "tol": 1e-4,
                    }

                edge_configs[edge_key] = {
                    "condition": f"{condition}_fixed{n_comp}",
                    "method": method,
                    "metrics": original_metrics,  # Use optimized metrics
                    "top_params": [{"params": base_params}],
                    "edge_case_type": "fixed_components_optimized_metrics",
                    "description": f"Fixed {n_comp} components with {condition} optimized metrics",
                }
                edge_case_counter += 1

                # Edge case 2: Fixed components with full metrics (no filtering)
                edge_key = f"EdgeCase_{edge_case_counter:02d}_FullMetrics_{method}_fixed{n_comp}comp"

                edge_configs[edge_key] = {
                    "condition": f"FullMetrics_fixed{n_comp}",
                    "method": method,
                    "metrics": full_metrics_list,  # Use ALL metrics
                    "top_params": [{"params": base_params}],
                    "edge_case_type": "fixed_components_full_metrics",
                    "description": f"Fixed {n_comp} components with ALL metrics ({len(full_metrics_list)} total)",
                }
                edge_case_counter += 1

    print(f"   ✅ Generated {len(edge_configs)} edge case configurations")

    return edge_configs


def run_single_pca_analysis(dataset, config_id, condition_name, method_type, metrics_list, params, is_edge_case=False):
    """
    Run PCA analysis for a single parameter configuration
    Enhanced to handle edge cases with different filtering approaches
    """
    print(f"  🔬 Running {config_id}: {condition_name}_{method_type}")

    try:
        # Filter metrics that exist in dataset
        available_metrics = [m for m in metrics_list if m in dataset.columns]
        missing_metrics = len(metrics_list) - len(available_metrics)

        if missing_metrics > 0:
            print(f"    ⚠️ {missing_metrics} metrics not found in dataset")

        if len(available_metrics) < 3:
            print(f"    ❌ Insufficient available metrics: {len(available_metrics)}")
            return []

        # Handle missing values differently for edge cases vs optimized
        if is_edge_case and "full_metrics" in config_id.lower():
            # For full metrics edge cases: more lenient missing value handling
            na_counts = dataset[available_metrics].isna().sum()
            total_rows = len(dataset)
            missing_percentages = (na_counts / total_rows) * 100

            # Keep metrics with ≤10% missing for full metrics (more lenient)
            valid_metrics = [col for col in available_metrics if missing_percentages.loc[col] <= 10.0]
            print(f"    📊 Edge case (full metrics): Using {len(valid_metrics)} metrics (≤10% missing)")

        else:
            # Standard filtering: ≤5% missing
            na_counts = dataset[available_metrics].isna().sum()
            total_rows = len(dataset)
            missing_percentages = (na_counts / total_rows) * 100

            valid_metrics = [col for col in available_metrics if missing_percentages.loc[col] <= 5.0]
            print(f"    📊 Standard filtering: Using {len(valid_metrics)} metrics (≤5% missing)")

        # Remove rows with any missing values
        valid_data = dataset[valid_metrics].copy()
        rows_with_missing = valid_data.isnull().any(axis=1)
        dataset_clean = dataset[~rows_with_missing].copy()

        if len(dataset_clean) < 10:
            print(f"    ❌ Insufficient data: {len(dataset_clean)} rows")
            return []

        # Prepare static metrics (exclude temporal ones)
        temporal_metrics = [
            col for col in valid_metrics if col.startswith(("binned_slope_", "interaction_rate_bin_", "binned_auc_"))
        ]
        static_metrics = [col for col in valid_metrics if col not in temporal_metrics]

        if len(static_metrics) < 3:
            print(f"    ❌ Insufficient static metrics: {len(static_metrics)}")
            return []

        # Scale data
        static_data = dataset_clean[static_metrics].to_numpy()
        scaler = RobustScaler()
        static_data_scaled = scaler.fit_transform(static_data)

        # Validate component count vs available metrics
        n_components = params["n_components"]
        max_components = min(static_data_scaled.shape[0] - 1, len(static_metrics))

        if n_components > max_components:
            print(f"    ⚠️ Reducing components from {n_components} to {max_components} (data limitation)")
            params = params.copy()
            params["n_components"] = max_components
            n_components = max_components

        # Run PCA with specific parameters
        if method_type == "PCA":
            pca = PCA(random_state=42, **params)
        else:  # SparsePCA
            pca = SparsePCA(random_state=42, **params)

        pca_scores = pca.fit_transform(static_data_scaled)

        # Calculate explained variance
        if method_type == "SparsePCA":
            # Manual calculation for SparsePCA
            total_variance = np.var(static_data_scaled, axis=0).sum()
            explained_variance = []
            for component in pca.components_:
                proj_data = static_data_scaled @ component.reshape(-1, 1)
                explained_variance.append(np.var(proj_data))
            explained_variance_ratio = np.array(explained_variance) / total_variance
        else:
            explained_variance_ratio = pca.explained_variance_ratio_

        # Create PCA scores dataframe
        actual_components = pca_scores.shape[1]
        pca_scores_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(actual_components)])

        # Combine with metadata
        metadata_cols = [col for col in dataset_clean.columns if col not in valid_metrics]
        pca_with_meta = pd.concat([dataset_clean[metadata_cols].reset_index(drop=True), pca_scores_df], axis=1)

        # Statistical testing (same as original)
        selected_dims = pca_scores_df.columns
        results = []
        all_nicknames = pca_with_meta["Nickname"].unique()

        for nickname in all_nicknames:
            subset = Config.get_subset_data(pca_with_meta, col="Nickname", value=nickname, force_control=None)
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
            for dim in selected_dims:
                group_scores = subset[subset["Nickname"] == nickname][dim]
                control_scores = subset[subset["Nickname"] == control_name][dim]
                if group_scores.empty or control_scores.empty:
                    continue
                stat, pval = stats.mannwhitneyu(group_scores, control_scores, alternative="two-sided")
                mannwhitney_pvals.append(pval)
                dims_tested.append(dim)

            if len(mannwhitney_pvals) > 0:
                rejected, pvals_corr, _, _ = multipletests(mannwhitney_pvals, alpha=0.05, method="fdr_bh")
                mannwhitney_any = any(rejected)
                significant_dims = [dims_tested[i] for i, rej in enumerate(rejected) if rej]
            else:
                mannwhitney_any = False
                significant_dims = []

            # Multivariate tests
            group_matrix = subset[subset["Nickname"] == nickname][selected_dims].values
            control_matrix = subset[subset["Nickname"] == control_name][selected_dims].values

            if group_matrix.size == 0 or control_matrix.size == 0:
                continue

            perm_stat, perm_pval = permutation_test(group_matrix, control_matrix, random_state=42)
            maha_stat, maha_pval = mahalanobis_permutation_test(group_matrix, control_matrix, random_state=42)

            results.append(
                {
                    "Nickname": nickname,
                    "Control": control_name,
                    "MannWhitney_any_dim_significant": mannwhitney_any,
                    "MannWhitney_significant_dims": significant_dims,
                    "Permutation_pval": perm_pval,
                    "Mahalanobis_pval": maha_pval,
                }
            )

        if not results:
            print(f"    ⚠️ No statistical results generated")
            return []

        results_df = pd.DataFrame(results)

        # Apply FDR correction to multivariate tests
        for col in ["Permutation_pval", "Mahalanobis_pval"]:
            rejected, pvals_corrected, _, _ = multipletests(results_df[col], alpha=0.05, method="fdr_bh")
            results_df[col.replace("_pval", "_FDR_significant")] = rejected

        # Extract significant hits
        all_methods_significant = results_df[
            (results_df["MannWhitney_any_dim_significant"])
            & (results_df["Permutation_FDR_significant"])
            & (results_df["Mahalanobis_FDR_significant"])
        ]

        significant_genotypes = all_methods_significant["Nickname"].tolist()

        print(f"    ✅ Found {len(significant_genotypes)} significant hits")

        # Store detailed results
        config_results = {
            "config_id": config_id,
            "condition": condition_name,
            "method": method_type,
            "params": params,
            "n_metrics": len(static_metrics),
            "n_components": actual_components,
            "explained_variance_total": float(explained_variance_ratio.sum()),
            "n_genotypes_tested": len(results_df),
            "significant_hits": significant_genotypes,
            "all_tested_genotypes": all_nicknames.tolist(),  # Include ALL tested genotypes
            "is_edge_case": is_edge_case,
            "edge_case_type": getattr(params, "edge_case_type", None) if is_edge_case else None,
            "all_results": results_df.to_dict("records"),
        }

        return significant_genotypes, config_results

    except Exception as e:
        print(f"    ❌ Error in {config_id}: {str(e)}")
        return []


# [Keep all the original helper functions: permutation_test, mahalanobis_distance, etc.]
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


def prepare_data():
    """Load and preprocess data (same as original script)"""
    print("📊 Loading and preprocessing data...")

    dataset = pd.read_feather(DATA_PATH)
    dataset = Config.cleanup_data(dataset)

    # Exclude problematic nicknames
    exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS"]
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


def main():
    global INCLUDE_EDGE_CASES
    print("🚀 ENHANCED CONSISTENCY ANALYSIS WITH EDGE CASES")
    print("=" * 65)

    # Load optimized configurations
    if not os.path.exists(CONFIGS_PATH):
        print(f"❌ Configuration file not found: {CONFIGS_PATH}")
        return

    with open(CONFIGS_PATH, "r") as f:
        optimized_configs = json.load(f)

    print(f"✅ Loaded {len(optimized_configs)} optimized configurations")

    # Load full metrics list for edge cases
    full_metrics = None
    if INCLUDE_EDGE_CASES:
        full_metrics = load_metrics_list(FULL_METRICS_PATH)
        if full_metrics:
            print(f"✅ Loaded {len(full_metrics)} metrics for edge case testing")
        else:
            print("⚠️ Could not load full metrics - edge cases will be skipped")
            INCLUDE_EDGE_CASES = False

    # Generate edge case configurations
    edge_configs = {}
    if INCLUDE_EDGE_CASES and full_metrics:
        edge_configs = generate_edge_case_configs(optimized_configs, full_metrics)

    # Combine all configurations
    all_configs = {}
    all_configs.update(optimized_configs)
    all_configs.update(edge_configs)

    print(f"\n📋 TOTAL ANALYSIS SCOPE:")
    print(f"   Optimized configurations: {len(optimized_configs)}")
    print(f"   Edge case configurations: {len(edge_configs)}")
    print(f"   Total configurations: {len(all_configs)}")

    # Prepare data once
    dataset = prepare_data()
    print(f"📊 Dataset prepared: {dataset.shape}")

    # Track results separately for optimized vs edge cases
    optimized_hits = []
    edge_case_hits = []
    all_hits = []  # Combined

    hit_counts = {}
    optimized_hit_counts = {}
    edge_case_hit_counts = {}

    # Track ALL genotypes tested (not just significant ones)
    all_tested_genotypes = set()
    genotype_tested_counts = {}  # How many configs each genotype was tested in

    config_details = []

    total_configs = 0
    successful_configs = 0
    successful_optimized = 0
    successful_edge_cases = 0

    # Process all configurations
    for config_key, config_data in all_configs.items():
        condition_name = config_data["condition"]
        method_type = config_data["method"]
        metrics_list = config_data["metrics"]
        top_params_list = config_data["top_params"]
        is_edge_case = config_key.startswith("EdgeCase_")

        config_type = "Edge Case" if is_edge_case else "Optimized"
        print(f"\n📋 Processing {config_type} {config_key}: {len(top_params_list)} parameter sets")

        for i, param_data in enumerate(top_params_list):
            config_id = f"{config_key}_rank{i+1}"
            params = param_data["params"]
            total_configs += 1

            # Run analysis
            result = run_single_pca_analysis(
                dataset, config_id, condition_name, method_type, metrics_list, params, is_edge_case=is_edge_case
            )

            if isinstance(result, tuple):
                significant_genotypes, detailed_results = result
                successful_configs += 1

                if is_edge_case:
                    successful_edge_cases += 1
                else:
                    successful_optimized += 1

                # Track ALL tested genotypes (not just significant ones)
                all_tested_in_config = detailed_results.get("all_tested_genotypes", [])
                for genotype in all_tested_in_config:
                    all_tested_genotypes.add(genotype)
                    genotype_tested_counts[genotype] = genotype_tested_counts.get(genotype, 0) + 1

                # Track hits by category
                for genotype in significant_genotypes:
                    all_hits.append((config_id, genotype))
                    hit_counts[genotype] = hit_counts.get(genotype, 0) + 1

                    if is_edge_case:
                        edge_case_hits.append((config_id, genotype))
                        edge_case_hit_counts[genotype] = edge_case_hit_counts.get(genotype, 0) + 1
                    else:
                        optimized_hits.append((config_id, genotype))
                        optimized_hit_counts[genotype] = optimized_hit_counts.get(genotype, 0) + 1

                # Store detailed results with edge case info
                detailed_results["is_edge_case"] = is_edge_case
                if is_edge_case:
                    detailed_results["edge_case_type"] = config_data.get("edge_case_type", "unknown")
                config_details.append(detailed_results)

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Total configurations attempted: {total_configs}")
    print(f"   Successful configurations: {successful_configs}")
    print(f"     - Optimized successful: {successful_optimized}")
    print(f"     - Edge case successful: {successful_edge_cases}")
    print(f"   Overall success rate: {successful_configs/total_configs*100:.1f}%")

    if successful_configs == 0:
        print("❌ No successful configurations - cannot generate analysis")
        return

    # === ENHANCED CONSISTENCY ANALYSIS ===
    print(f"\n🎯 ENHANCED CONSISTENCY ANALYSIS")
    print("=" * 45)

    # Overall statistics
    total_unique_hits = len(hit_counts)
    optimized_unique_hits = len(optimized_hit_counts)
    edge_case_unique_hits = len(edge_case_hit_counts)

    print(f"OVERALL RESULTS:")
    print(f"   Total unique hits: {total_unique_hits}")
    print(f"   Optimized unique hits: {optimized_unique_hits}")
    print(f"   Edge case unique hits: {edge_case_unique_hits}")
    print(f"   Total genotypes tested: {len(all_tested_genotypes)}")

    # Create comprehensive consistency dataframes for ALL tested genotypes
    all_consistency_data = []
    for genotype in all_tested_genotypes:
        count = hit_counts.get(genotype, 0)  # 0 if never significant
        consistency_score = count / successful_configs
        optimized_count = optimized_hit_counts.get(genotype, 0)
        edge_case_count = edge_case_hit_counts.get(genotype, 0)

        optimized_consistency = optimized_count / successful_optimized if successful_optimized > 0 else 0
        edge_case_consistency = edge_case_count / successful_edge_cases if successful_edge_cases > 0 else 0

        all_consistency_data.append(
            {
                "Genotype": genotype,
                "Total_Hit_Count": count,
                "Total_Configs": successful_configs,
                "Overall_Consistency": consistency_score,
                "Optimized_Hit_Count": optimized_count,
                "Optimized_Configs": successful_optimized,
                "Optimized_Consistency": optimized_consistency,
                "Edge_Case_Hit_Count": edge_case_count,
                "Edge_Case_Configs": successful_edge_cases,
                "Edge_Case_Consistency": edge_case_consistency,
                "Found_In_Both": (optimized_count > 0) and (edge_case_count > 0),
                "Optimized_Only": (optimized_count > 0) and (edge_case_count == 0),
                "Edge_Case_Only": (optimized_count == 0) and (edge_case_count > 0),
            }
        )

    consistency_df = pd.DataFrame(all_consistency_data)
    consistency_df = consistency_df.sort_values("Overall_Consistency", ascending=False)

    # === COMPARATIVE ANALYSIS ===
    print(f"\n📈 COMPARATIVE ANALYSIS:")
    print("-" * 40)

    found_in_both = consistency_df[consistency_df["Found_In_Both"]]
    optimized_only = consistency_df[consistency_df["Optimized_Only"]]
    edge_case_only = consistency_df[consistency_df["Edge_Case_Only"]]

    print(f"   Hits found in BOTH optimized AND edge cases: {len(found_in_both):3d}")
    print(f"   Hits found ONLY in optimized configs:      {len(optimized_only):3d}")
    print(f"   Hits found ONLY in edge cases:             {len(edge_case_only):3d}")

    # Top consistent hits
    print(f"\n🏆 TOP 10 OVERALL CONSISTENT HITS:")
    print("-" * 50)
    top_10 = consistency_df.head(10)
    for _, row in top_10.iterrows():
        opt_pct = row["Optimized_Consistency"] * 100
        edge_pct = row["Edge_Case_Consistency"] * 100
        both_marker = "✓" if row["Found_In_Both"] else " "
        print(f"{row['Genotype']:<30}: Opt={opt_pct:5.1f}% | Edge={edge_pct:5.1f}% | Both={both_marker}")

    # === SAVE ENHANCED RESULTS ===

    # 1. Comprehensive consistency results
    consistency_file = os.path.join(OUTPUT_DIR, "enhanced_consistency_scores.csv")
    consistency_df.to_csv(consistency_file, index=False)

    # 2. Configuration summary with edge case info
    config_summary = []
    for details in config_details:
        config_summary.append(
            {
                "Config_ID": details["config_id"],
                "Condition": details["condition"],
                "Method": details["method"],
                "Is_Edge_Case": details["is_edge_case"],
                "Edge_Case_Type": details.get("edge_case_type", "N/A"),
                "N_Metrics": details["n_metrics"],
                "N_Components": details["n_components"],
                "Explained_Variance": details["explained_variance_total"],
                "N_Significant_Hits": len(details["significant_hits"]),
            }
        )

    config_summary_df = pd.DataFrame(config_summary)
    config_file = os.path.join(OUTPUT_DIR, "enhanced_configuration_summary.csv")
    config_summary_df.to_csv(config_file, index=False)

    # 3. Edge case specific analysis
    edge_case_summary = config_summary_df[config_summary_df["Is_Edge_Case"]]
    if len(edge_case_summary) > 0:
        edge_case_file = os.path.join(OUTPUT_DIR, "edge_case_analysis.csv")
        edge_case_summary.to_csv(edge_case_file, index=False)

        print(f"\n🔧 EDGE CASE ANALYSIS:")
        print(
            f"   Fixed 10 components average hits: {edge_case_summary[edge_case_summary['N_Components']==10]['N_Significant_Hits'].mean():.1f}"
        )
        print(
            f"   Fixed 15 components average hits: {edge_case_summary[edge_case_summary['N_Components']==15]['N_Significant_Hits'].mean():.1f}"
        )

    # 4. Enhanced visualization
    plt.figure(figsize=(20, 15))

    # Plot 1: Overall vs Optimized vs Edge Case consistency
    plt.subplot(3, 3, 1)
    plt.scatter(consistency_df["Optimized_Consistency"], consistency_df["Edge_Case_Consistency"], alpha=0.6)
    plt.xlabel("Optimized Consistency")
    plt.ylabel("Edge Case Consistency")
    plt.title("Optimized vs Edge Case Consistency")
    plt.plot([0, 1], [0, 1], "r--", alpha=0.5)

    # Plot 2: Hit distribution by category
    plt.subplot(3, 3, 2)
    categories = ["Both", "Optimized Only", "Edge Case Only"]
    counts = [len(found_in_both), len(optimized_only), len(edge_case_only)]
    colors = ["green", "blue", "orange"]
    plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.ylabel("Number of Genotypes")
    plt.title("Hit Distribution by Category")
    plt.xticks(rotation=45)

    # Plot 3: Components vs hits for edge cases
    plt.subplot(3, 3, 3)
    edge_10_comp = edge_case_summary[edge_case_summary["N_Components"] == 10]
    edge_15_comp = edge_case_summary[edge_case_summary["N_Components"] == 15]

    plt.boxplot(
        [edge_10_comp["N_Significant_Hits"], edge_15_comp["N_Significant_Hits"]],
        labels=["10 Components", "15 Components"],
    )
    plt.ylabel("Number of Significant Hits")
    plt.title("Edge Case: Components vs Hits")

    # Plot 4-6: Method comparisons for edge cases
    plt.subplot(3, 3, 4)
    pca_edge = edge_case_summary[edge_case_summary["Method"] == "PCA"]
    sparse_edge = edge_case_summary[edge_case_summary["Method"] == "SparsePCA"]

    if len(pca_edge) > 0 and len(sparse_edge) > 0:
        plt.boxplot([pca_edge["N_Significant_Hits"], sparse_edge["N_Significant_Hits"]], labels=["PCA", "SparsePCA"])
        plt.ylabel("Number of Significant Hits")
        plt.title("Edge Case: Method Comparison")

    # Plot 7: Top 20 most consistent overall
    plt.subplot(3, 3, 7)
    top_20 = consistency_df.head(20)
    y_pos = range(len(top_20))
    plt.barh(y_pos, top_20["Overall_Consistency"])
    plt.yticks(y_pos, top_20["Genotype"], fontsize=6)
    plt.xlabel("Overall Consistency Score")
    plt.title("Top 20 Most Consistent Hits")
    plt.gca().invert_yaxis()

    # Plot 8: Edge case type comparison
    plt.subplot(3, 3, 8)
    if len(edge_case_summary) > 0:
        edge_type_hits = edge_case_summary.groupby("Edge_Case_Type")["N_Significant_Hits"].mean()
        plt.bar(range(len(edge_type_hits)), edge_type_hits.values)
        plt.xticks(range(len(edge_type_hits)), edge_type_hits.index, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Average Hits")
        plt.title("Edge Case Type Performance")

    # Plot 9: Consistency distribution comparison
    plt.subplot(3, 3, 9)
    plt.hist(consistency_df["Optimized_Consistency"], alpha=0.5, label="Optimized", bins=20)
    plt.hist(consistency_df["Edge_Case_Consistency"], alpha=0.5, label="Edge Cases", bins=20)
    plt.xlabel("Consistency Score")
    plt.ylabel("Number of Genotypes")
    plt.title("Consistency Distribution Comparison")
    plt.legend()

    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "enhanced_consistency_plots.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n💾 ENHANCED FILES SAVED:")
    print(f"   📊 {consistency_file}")
    print(f"   ⚙️  {config_file}")
    if len(edge_case_summary) > 0:
        print(f"   🔧 {edge_case_file}")
    print(f"   📈 {plot_file}")

    print(f"\n✅ ENHANCED CONSISTENCY ANALYSIS COMPLETE!")
    print(f"🎯 Found {len(found_in_both)} genotypes consistent across BOTH optimized and edge cases")
    print(f"🔧 Edge cases tested {len(edge_configs)} additional configurations")


if __name__ == "__main__":
    main()
