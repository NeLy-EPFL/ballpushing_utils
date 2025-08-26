#!/usr/bin/env python3
"""
Multi-Condition PCA Parameter Optimization Script with BIC Penalty
Tests both PCA and SparsePCA on two different metric lists,
keeping top 5 parameter sets for each condition for consistency analysis.
Includes BIC penalty to prevent overfitting with too many components.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ParameterGrid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import your Config module
import Config

# === CONFIGURATION ===
DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"

# Two metric lists to compare
METRICS_LIST_1_PATH = "final_metrics_for_pca.txt"  # Your correlation-based selection
METRICS_LIST_2_PATH = "final_metrics_for_pca_alt.txt"  # Your family-based selection

# BIC penalty weight (0.0 = no penalty, higher values = stronger penalty for complexity)
BIC_PENALTY_WEIGHT = 0.15  # Adjust this value to control complexity penalty

# Output directory
if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
else:
    OUTPUT_DIR = "multi_condition_pca_optimization"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of top parameter sets to keep per condition
TOP_N_PARAMS = 5


def load_metrics_list(path):
    """Load metrics list from file"""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        cols = [ln.strip() for ln in f if ln.strip()]
    return cols


def calculate_bic(reconstruction_error, n_components, n_samples):
    """
    Calculate BIC (Bayesian Information Criterion) for model selection
    BIC = n * log(RSS/n) + k * log(n)

    Parameters:
    - reconstruction_error: Mean squared error from PCA reconstruction
    - n_components: Number of PCA components (model complexity)
    - n_samples: Number of data samples

    Returns:
    - BIC score (lower is better)
    """
    if reconstruction_error <= 0 or n_samples <= 0:
        return np.inf

    # Convert MSE to RSS (Residual Sum of Squares)
    rss = reconstruction_error * n_samples

    # Calculate BIC: n*log(RSS/n) + k*log(n)
    bic = n_samples * np.log(rss / n_samples) + n_components * np.log(n_samples)

    return bic


def evaluate_pca_params(X, params):
    """Evaluate regular PCA parameters"""
    try:
        pca = PCA(random_state=42, **params)
        X_transformed = pca.fit_transform(X)

        explained_variance_ratio = sum(pca.explained_variance_ratio_)
        overall_sparsity = 0.0
        mean_sparsity_per_component = 0.0
        interpretable_components = params["n_components"]

        # Reconstruction error
        X_reconstructed = X_transformed @ pca.components_
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        return {
            "method_type": "PCA",
            "params": params,
            "explained_variance_ratio": explained_variance_ratio,
            "overall_sparsity": overall_sparsity,
            "mean_sparsity_per_component": mean_sparsity_per_component,
            "interpretable_components": interpretable_components,
            "reconstruction_error": reconstruction_error,
            "converged": True,
            "n_iter": None,
            "success": True,
        }

    except Exception as e:
        return {
            "method_type": "PCA",
            "params": params,
            "explained_variance_ratio": 0,
            "overall_sparsity": 0,
            "mean_sparsity_per_component": 0,
            "interpretable_components": 0,
            "reconstruction_error": np.inf,
            "converged": False,
            "n_iter": None,
            "success": False,
            "error": str(e),
        }


def evaluate_sparsepca_params(X, params):
    """Evaluate SparsePCA parameters"""
    try:
        spca = SparsePCA(random_state=42, **params)
        X_transformed = spca.fit_transform(X)

        # Compute explained variance manually for SparsePCA
        total_variance = np.var(X, axis=0).sum()
        explained_variances = []

        for k in range(spca.components_.shape[0]):
            component = spca.components_[k]
            projected_data = X @ component.reshape(-1, 1)
            explained_variances.append(np.var(projected_data))

        explained_variance_ratio = sum(explained_variances) / total_variance

        # Compute sparsity metrics
        loadings = spca.components_
        overall_sparsity = np.mean(loadings == 0)
        sparsity_per_component = np.mean(loadings == 0, axis=1)

        # Convergence check
        converged = hasattr(spca, "n_iter_") and spca.n_iter_ < params.get("max_iter", 1000)
        n_iter = getattr(spca, "n_iter_", None)

        # Component interpretability
        interpretable_components = np.sum((sparsity_per_component > 0.1) & (sparsity_per_component < 0.9))

        # Reconstruction error
        X_reconstructed = X_transformed @ spca.components_
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        return {
            "method_type": "SparsePCA",
            "params": params,
            "explained_variance_ratio": explained_variance_ratio,
            "overall_sparsity": overall_sparsity,
            "mean_sparsity_per_component": np.mean(sparsity_per_component),
            "interpretable_components": interpretable_components,
            "reconstruction_error": reconstruction_error,
            "converged": converged,
            "n_iter": n_iter,
            "success": True,
        }

    except Exception as e:
        return {
            "method_type": "SparsePCA",
            "params": params,
            "explained_variance_ratio": 0,
            "overall_sparsity": 0,
            "mean_sparsity_per_component": 0,
            "interpretable_components": 0,
            "reconstruction_error": np.inf,
            "converged": False,
            "n_iter": None,
            "success": False,
            "error": str(e),
        }


def optimize_single_condition(X, condition_name, method_type):
    """Optimize parameters for a single condition (metric_list + method combination)"""
    print(f"\n🔍 Optimizing {condition_name} - {method_type}...")

    if method_type == "PCA":
        param_grid = {"n_components": [7, 10, 12, 15, 20]}
        evaluate_func = evaluate_pca_params
    else:  # SparsePCA
        param_grid = {
            "n_components": [7, 10, 12, 15],
            "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "ridge_alpha": [0.01, 0.1, 1.0],
            "method": ["lars", "cd"],
            "max_iter": [2000, 3000],
            "tol": [1e-4, 1e-5],
        }
        evaluate_func = evaluate_sparsepca_params

    combinations = list(ParameterGrid(param_grid))
    print(f"   Testing {len(combinations)} parameter combinations...")

    results = []
    for i, params in enumerate(combinations):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(combinations)}")

        result = evaluate_func(X, params)
        result["condition"] = condition_name
        result["combination_id"] = i
        results.append(result)

    results_df = pd.DataFrame(results)
    successful_df = results_df[results_df["success"] == True].copy()

    if len(successful_df) == 0:
        print(f"   ❌ No successful combinations for {condition_name} - {method_type}")
        return pd.DataFrame()

    print(f"   ✅ {len(successful_df)}/{len(results_df)} combinations succeeded")
    return successful_df


def score_parameter_combinations_with_bic(results_df, n_samples, bic_weight=0.15):
    """
    Score parameter combinations using multiple criteria including BIC penalty

    Parameters:
    - results_df: DataFrame with optimization results
    - n_samples: Number of samples in the dataset
    - bic_weight: Weight for BIC penalty (higher = stronger penalty for complexity)
    """
    if len(results_df) == 0:
        return results_df

    results_df = results_df.copy()

    # Calculate BIC scores
    bic_scores = []
    for _, row in results_df.iterrows():
        n_components = row["params"]["n_components"]
        reconstruction_error = row["reconstruction_error"]
        bic = calculate_bic(reconstruction_error, n_components, n_samples)
        bic_scores.append(bic)

    results_df["bic_score"] = bic_scores

    # Normalize BIC scores (lower BIC is better, so we invert for penalty)
    bic_min = results_df["bic_score"].min()
    bic_max = results_df["bic_score"].max()
    if bic_max > bic_min:
        results_df["bic_penalty"] = (results_df["bic_score"] - bic_min) / (bic_max - bic_min)
    else:
        results_df["bic_penalty"] = 0.0

    # Normalize other metrics for scoring (0-1 scale)
    results_df["explained_var_norm"] = (
        results_df["explained_variance_ratio"] - results_df["explained_variance_ratio"].min()
    ) / (results_df["explained_variance_ratio"].max() - results_df["explained_variance_ratio"].min() + 1e-8)

    results_df["interpretable_comp_norm"] = results_df["interpretable_components"] / (
        results_df["interpretable_components"].max() + 1e-8
    )

    # Method-aware sparsity scoring
    sparsity_scores = []
    for _, row in results_df.iterrows():
        if row["method_type"] == "PCA":
            sparsity_scores.append(1.0)
        else:
            sparsity = row["overall_sparsity"]
            optimal_sparsity_range = (0.3, 0.7)
            low, high = optimal_sparsity_range
            center = (low + high) / 2
            if low <= sparsity <= high:
                score = 1 - abs(sparsity - center) / (high - center)
            elif sparsity < low:
                score = (sparsity / low) * 0.5
            else:  # sparsity > high
                score = ((1 - sparsity) / (1 - high)) * 0.5
            sparsity_scores.append(score)

    results_df["sparsity_score"] = sparsity_scores

    # Reconstruction score
    if results_df["reconstruction_error"].max() > results_df["reconstruction_error"].min():
        results_df["reconstruction_score"] = 1 - (
            results_df["reconstruction_error"] - results_df["reconstruction_error"].min()
        ) / (results_df["reconstruction_error"].max() - results_df["reconstruction_error"].min())
    else:
        results_df["reconstruction_score"] = 1.0

    # Convergence score
    results_df["convergence_score"] = results_df["converged"].astype(float)

    # Combined score WITH BIC penalty
    combined_scores = []
    for _, row in results_df.iterrows():
        if row["method_type"] == "PCA":
            weights = {
                "explained_var_norm": 0.4,
                "sparsity_score": 0.1,
                "interpretable_comp_norm": 0.2,
                "reconstruction_score": 0.2,
                "convergence_score": 0.1,
            }
            method_bic_weight = bic_weight
        else:
            weights = {
                "explained_var_norm": 0.3,
                "sparsity_score": 0.25,
                "interpretable_comp_norm": 0.2,
                "reconstruction_score": 0.15,
                "convergence_score": 0.1,
            }
            method_bic_weight = bic_weight * 1.2  # Slightly stronger penalty for SparsePCA

        # Calculate base score
        base_score = sum(row[metric] * weight for metric, weight in weights.items())

        # Apply BIC penalty (subtract penalty, weighted by method_bic_weight)
        final_score = base_score - (method_bic_weight * row["bic_penalty"])

        combined_scores.append(final_score)

    results_df["combined_score"] = combined_scores
    return results_df


def create_condition_summary_plot(all_results, output_dir):
    """Create summary plots across all conditions including BIC analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Group by condition and method
    conditions = all_results["condition"].unique()
    methods = all_results["method_type"].unique()

    # Plot 1: Combined scores by condition
    ax1 = axes[0, 0]
    box_data = []
    labels = []
    for condition in conditions:
        for method in methods:
            subset = all_results[(all_results["condition"] == condition) & (all_results["method_type"] == method)]
            if len(subset) > 0:
                box_data.append(subset["combined_score"])
                labels.append(f"{condition}\n{method}")

    if box_data:
        ax1.boxplot(box_data, labels=labels)
        ax1.set_ylabel("Combined Score (with BIC penalty)")
        ax1.set_title("Performance by Condition & Method")
        ax1.tick_params(axis="x", rotation=45)

    # Plot 2: BIC vs Number of Components
    ax2 = axes[0, 1]
    for condition in conditions:
        subset = all_results[all_results["condition"] == condition]
        n_components = [row["n_components"] for row in subset["params"]]
        ax2.scatter(n_components, subset["bic_score"], alpha=0.6, label=condition)
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("BIC Score")
    ax2.set_title("BIC vs Model Complexity")
    ax2.legend()

    # Plot 3: Explained Variance vs BIC Penalty
    ax3 = axes[0, 2]
    ax3.scatter(
        all_results["explained_variance_ratio"],
        all_results["bic_penalty"],
        c=[conditions.tolist().index(c) for c in all_results["condition"]],
        alpha=0.6,
    )
    ax3.set_xlabel("Explained Variance Ratio")
    ax3.set_ylabel("BIC Penalty (normalized)")
    ax3.set_title("Variance-Complexity Tradeoff")

    # Plot 4: Top performers with BIC info
    ax4 = axes[1, 0]
    top_per_condition = []
    for condition in conditions:
        for method in methods:
            subset = all_results[(all_results["condition"] == condition) & (all_results["method_type"] == method)]
            if len(subset) > 0:
                best = subset.nlargest(1, "combined_score").iloc[0]
                top_per_condition.append(
                    {
                        "condition": condition,
                        "method": method,
                        "score": best["combined_score"],
                        "n_components": best["params"]["n_components"],
                        "bic_penalty": best["bic_penalty"],
                        "label": f"{condition}-{method}",
                    }
                )

    if top_per_condition:
        top_df = pd.DataFrame(top_per_condition)
        bars = ax4.bar(range(len(top_df)), top_df["score"])
        ax4.set_xticks(range(len(top_df)))
        ax4.set_xticklabels(top_df["label"], rotation=45)
        ax4.set_ylabel("Best Combined Score")
        ax4.set_title("Best Performance per Condition-Method")

        # Add component count as text on bars
        for i, (bar, n_comp) in enumerate(zip(bars, top_df["n_components"])):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{n_comp}c",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 5: BIC penalty effect
    ax5 = axes[1, 1]
    # Compare scores with and without BIC penalty
    n_components = [row["n_components"] for row in all_results["params"]]
    ax5.scatter(n_components, all_results["combined_score"], alpha=0.6, label="With BIC penalty")
    # Calculate scores without BIC penalty for comparison
    scores_without_bic = all_results["combined_score"] + BIC_PENALTY_WEIGHT * all_results["bic_penalty"]
    ax5.scatter(n_components, scores_without_bic, alpha=0.4, label="Without BIC penalty")
    ax5.set_xlabel("Number of Components")
    ax5.set_ylabel("Combined Score")
    ax5.set_title(f"BIC Penalty Effect (weight={BIC_PENALTY_WEIGHT})")
    ax5.legend()

    # Plot 6: Method comparison with complexity
    ax6 = axes[1, 2]
    for method in methods:
        method_data = all_results[all_results["method_type"] == method]
        n_components = [row["n_components"] for row in method_data["params"]]
        ax6.scatter(n_components, method_data["combined_score"], alpha=0.6, label=method)
    ax6.set_xlabel("Number of Components")
    ax6.set_ylabel("Combined Score")
    ax6.set_title("Score vs Complexity by Method")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_condition_summary_with_bic.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main():
    print("🚀 Multi-Condition PCA Parameter Optimization with BIC Penalty")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🔢 BIC penalty weight: {BIC_PENALTY_WEIGHT}")

    # === LOAD METRIC LISTS ===
    print("\n📋 Loading metric lists...")

    metrics_list_1 = load_metrics_list(METRICS_LIST_1_PATH)
    metrics_list_2 = load_metrics_list(METRICS_LIST_2_PATH)

    if metrics_list_1 is None:
        print(f"❌ Could not load {METRICS_LIST_1_PATH}")
        return
    if metrics_list_2 is None:
        print(f"❌ Could not load {METRICS_LIST_2_PATH}")
        return

    print(f"✅ List 1 ({METRICS_LIST_1_PATH}): {len(metrics_list_1)} metrics")
    print(f"✅ List 2 ({METRICS_LIST_2_PATH}): {len(metrics_list_2)} metrics")

    # === LOAD AND PREPROCESS DATA ===
    print("\n📊 Loading and preprocessing data...")

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

    print(f"📈 Dataset shape: {dataset.shape}")

    # === PREPARE CONDITIONS ===
    conditions = [
        {"name": "List1", "metrics": metrics_list_1},
        {"name": "List2", "metrics": metrics_list_2},
    ]

    methods = ["PCA", "SparsePCA"]

    # Storage for all results
    all_results = []
    top_configs = {}  # Will store top 5 configs per condition

    # === OPTIMIZE EACH CONDITION ===
    print(f"\n🎯 Optimizing {len(conditions)} × {len(methods)} = {len(conditions) * len(methods)} conditions...")

    for condition in conditions:
        condition_name = condition["name"]
        metric_names = condition["metrics"]

        # Filter metrics that exist in dataset
        available_metrics = [m for m in metric_names if m in dataset.columns]
        if len(available_metrics) < len(metric_names):
            missing = set(metric_names) - set(available_metrics)
            print(f"⚠️  {condition_name}: {len(missing)} metrics not found in dataset")

        print(f"\n📊 {condition_name}: Using {len(available_metrics)} metrics")

        # Filter missing values
        valid_data = dataset[available_metrics].copy()
        na_counts = valid_data.isna().sum()
        total_rows = len(valid_data)
        missing_percentages = (na_counts / total_rows) * 100

        # Keep metrics with ≤5% missing
        valid_metrics = [col for col in available_metrics if missing_percentages.loc[col] <= 5.0]

        if len(valid_metrics) < len(available_metrics):
            excluded = len(available_metrics) - len(valid_metrics)
            print(f"   ⚠️  Excluded {excluded} metrics with >5% missing values")

        # Remove rows with any missing values in valid metrics
        clean_data = dataset[valid_metrics].copy()
        rows_with_missing = clean_data.isnull().any(axis=1)
        final_data = clean_data[~rows_with_missing]

        print(f"   📉 Final: {final_data.shape[0]} rows, {len(valid_metrics)} metrics")

        if len(valid_metrics) < 3:
            print(f"   ❌ Too few metrics for {condition_name}, skipping")
            continue

        # Scale data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(final_data.values)
        n_samples = X_scaled.shape[0]

        # Test both methods for this condition
        for method in methods:
            condition_key = f"{condition_name}_{method}"

            # Optimize this specific condition
            results_df = optimize_single_condition(X_scaled, condition_name, method)

            if len(results_df) == 0:
                continue

            # Score the results WITH BIC penalty
            scored_results = score_parameter_combinations_with_bic(results_df, n_samples, BIC_PENALTY_WEIGHT)

            # Get top N configurations
            top_n = scored_results.nlargest(TOP_N_PARAMS, "combined_score")

            # Store for later use
            top_configs[condition_key] = {
                "condition": condition_name,
                "method": method,
                "metrics": valid_metrics,
                "top_params": top_n.to_dict("records"),
                "best_score": top_n.iloc[0]["combined_score"] if len(top_n) > 0 else 0,
                "best_bic": top_n.iloc[0]["bic_score"] if len(top_n) > 0 else np.inf,
                "best_n_components": top_n.iloc[0]["params"]["n_components"] if len(top_n) > 0 else 0,
            }

            # Add to overall results
            all_results.append(scored_results)

            best_n_comp = top_n.iloc[0]["params"]["n_components"] if len(top_n) > 0 else 0
            print(
                f"   ✅ {condition_key}: Top score = {top_n.iloc[0]['combined_score']:.4f} ({best_n_comp} components)"
            )

    # === COMBINE AND ANALYZE RESULTS ===
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Create summary plots with BIC analysis
        create_condition_summary_plot(combined_results, OUTPUT_DIR)

        # Save detailed results
        results_file = os.path.join(OUTPUT_DIR, "all_conditions_results_with_bic.csv")
        combined_results.to_csv(results_file, index=False)

        # === SAVE TOP CONFIGURATIONS ===
        configs_file = os.path.join(OUTPUT_DIR, "top_configurations.json")
        with open(configs_file, "w") as f:
            json.dump(top_configs, f, indent=2)

        # Create summary of all top configs
        summary_file = os.path.join(OUTPUT_DIR, "optimization_summary_with_bic.txt")
        with open(summary_file, "w") as f:
            f.write("🏆 MULTI-CONDITION PCA OPTIMIZATION SUMMARY (with BIC penalty)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"BIC penalty weight: {BIC_PENALTY_WEIGHT}\n")

            total_configs = sum(len(config["top_params"]) for config in top_configs.values())
            f.write(f"Total configurations for consistency analysis: {total_configs}\n")
            f.write(f"Conditions tested: {len(top_configs)}\n")
            f.write(f"Top parameters per condition: {TOP_N_PARAMS}\n\n")

            # Analyze complexity distribution
            all_n_components = []
            for config in top_configs.values():
                all_n_components.extend([p["params"]["n_components"] for p in config["top_params"]])

            if all_n_components:
                f.write(f"COMPLEXITY ANALYSIS:\n")
                f.write(f"   Average components selected: {np.mean(all_n_components):.1f}\n")
                f.write(f"   Component range: {min(all_n_components)} - {max(all_n_components)}\n")
                f.write(f"   Most common component count: {max(set(all_n_components), key=all_n_components.count)}\n\n")

            for condition_key, config in top_configs.items():
                f.write(f"\n📊 {condition_key}:\n")
                f.write(f"   Metrics used: {len(config['metrics'])}\n")
                f.write(f"   Best score: {config['best_score']:.4f}\n")
                f.write(f"   Best BIC: {config['best_bic']:.2f}\n")
                f.write(f"   Best components: {config['best_n_components']}\n")

                for i, params in enumerate(config["top_params"], 1):
                    n_comp = params["params"]["n_components"]
                    bic = params.get("bic_score", "N/A")
                    f.write(f"   #{i}: Score={params['combined_score']:.4f}, Components={n_comp}, BIC={bic:.2f}")
                    if params["method_type"] == "SparsePCA":
                        f.write(f", alpha={params['params']['alpha']}")
                    f.write("\n")

        print(f"\n🏆 OPTIMIZATION SUMMARY (with BIC penalty):")
        print(f"   BIC penalty weight: {BIC_PENALTY_WEIGHT}")
        print(f"   Total conditions: {len(top_configs)}")
        print(f"   Total configurations: {total_configs}")
        print(f"   Average per condition: {total_configs/len(top_configs):.1f}")

        if all_n_components:
            print(f"   Average components selected: {np.mean(all_n_components):.1f}")
            print(f"   Component range: {min(all_n_components)} - {max(all_n_components)}")

        # Show best per condition
        print(f"\n🥇 BEST CONFIGURATION PER CONDITION:")
        for condition_key, config in sorted(top_configs.items(), key=lambda x: x[1]["best_score"], reverse=True):
            print(f"   {condition_key}: {config['best_score']:.4f} ({config['best_n_components']} components)")

        print(f"\n💾 FILES SAVED:")
        print(f"   📄 {configs_file}")
        print(f"   📊 {results_file}")
        print(f"   📋 {summary_file}")
        print(f"   📈 multi_condition_summary_with_bic.png")

        print(f"\n✅ Ready for consistency analysis with {total_configs} parameter combinations!")
        print(f"🔢 BIC penalty helped balance model complexity vs performance")

    else:
        print("\n❌ No successful optimizations completed!")


if __name__ == "__main__":
    main()
