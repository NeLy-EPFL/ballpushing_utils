#!/usr/bin/env python3
"""
Correlation analysis script for ballpushing metrics.
This script analyzes the correlation between different metrics to identify
redundant features before adding them to the PCA pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_metrics_data(file_path):
    """Load the metrics dataset."""
    try:
        dataset = pd.read_feather(file_path)
        print(f"✓ Successfully loaded dataset with shape: {dataset.shape}")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def identify_metric_columns(dataset):
    """Identify which columns are metrics vs metadata."""
    # Based on your PCA_Static.py, identify metric patterns
    potential_metrics = []

    # Known metric patterns from your code
    base_metrics = [
        "nb_events",
        "max_event",
        "max_event_time",
        "max_distance",
        "final_event",
        "final_event_time",
        "nb_significant_events",
        "significant_ratio",
        "first_major_event",
        "first_major_event_time",
        "major_event",
        "major_event_time",  # Alternative names
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_velocity",
        "auc",
        # "overall_slope",  # Excluded as requested
        "overall_interaction_rate",
    ]

    # Pattern-based metrics (excluding binned metrics as requested)
    # binned_patterns = ["binned_slope_", "interaction_rate_bin_", "binned_auc_"]

    # Check which base metrics exist
    for metric in base_metrics:
        if metric in dataset.columns:
            potential_metrics.append(metric)

    # Skip pattern-based metrics (binned metrics) as requested
    # for pattern in binned_patterns:
    #     pattern_cols = [col for col in dataset.columns if col.startswith(pattern)]
    #     potential_metrics.extend(pattern_cols)

    # Additional metrics that might exist (from ballpushing_metrics.py methods)
    # Excluding slope, r2, and binned metrics as requested
    additional_patterns = [
        "velocity",
        "speed",
        "pause",
        "freeze",
        "persistence",
        "learning",
        "logistic",
        "influence",
        "trend",
        # "slope",  # Excluded as requested
        "interaction_rate",
        "finished",
        "chamber",
        "freeze",
        "facing",
        "flailing",
        "head",
        # "leg_visibility",
        "median_",
        "mean_",
    ]

    for pattern in additional_patterns:
        pattern_cols = [col for col in dataset.columns if pattern in col.lower()]
        potential_metrics.extend(pattern_cols)

    # Remove duplicates and sort
    potential_metrics = sorted(list(set(potential_metrics)))

    # Filter out obvious metadata columns and excluded metrics
    metadata_keywords = [
        "nickname",
        "genotype",
        "condition",
        "experiment",
        "date",
        "path",
        "file",
        "id",
        "index",
        "fly_idx",
        "ball_idx",
    ]

    # Additional exclusions as requested
    excluded_patterns = [
        "binned_",
        "r2",
        "slope",
        "_bin_",
        "logistic_" "persistence_at_end",
        "number_of_pauses",
        "overall_interaction_rate",
        "max_distance",
        "nb_significant_events",
        "final_event_time",
    ]

    actual_metrics = []
    for col in potential_metrics:
        # Skip metadata columns
        if any(keyword in col.lower() for keyword in metadata_keywords):
            continue
        # Skip excluded patterns
        if any(pattern in col.lower() for pattern in excluded_patterns):
            continue
        actual_metrics.append(col)

    return actual_metrics


def analyze_correlations(dataset, metric_columns, correlation_threshold=0.8, nan_threshold=0.1):
    """Analyze correlations between metrics."""
    print(f"\nAnalyzing correlations for {len(metric_columns)} metrics...")

    # Extract metrics data
    metrics_data = dataset[metric_columns].copy()

    # Convert boolean columns to numeric
    for col in metrics_data.columns:
        if metrics_data[col].dtype == "bool":
            metrics_data[col] = metrics_data[col].astype(int)

    # Analyze missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)

    total_rows = len(metrics_data)
    missing_counts = metrics_data.isnull().sum()
    missing_percentages = (missing_counts / total_rows) * 100

    print(f"Total number of rows: {total_rows}")
    print(f"NaN threshold for exclusion: {nan_threshold*100}% (allowing up to 5% missing values)")
    print("\nMissing values per metric (sorted by % missing):")
    print("-" * 60)

    # Create a summary DataFrame
    missing_summary = pd.DataFrame(
        {
            "metric": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_percentage": missing_percentages.values,
        }
    ).sort_values("missing_percentage", ascending=False)

    # Display all metrics with their missing percentages
    for _, row in missing_summary.iterrows():
        metric = row["metric"]
        count = row["missing_count"]
        percentage = row["missing_percentage"]
        status = "❌ EXCLUDE" if percentage > (nan_threshold * 100) else "✓ KEEP"
        print(f"{metric:<35} | {count:5.0f} missing ({percentage:5.1f}%) | {status}")

    # Filter out metrics with too many missing values
    valid_metrics = missing_summary[missing_summary["missing_percentage"] <= (nan_threshold * 100)]["metric"].tolist()
    excluded_metrics = missing_summary[missing_summary["missing_percentage"] > (nan_threshold * 100)]["metric"].tolist()

    print(f"\n📊 SUMMARY:")
    print(f"   • Metrics to keep: {len(valid_metrics)}")
    print(f"   • Metrics to exclude: {len(excluded_metrics)}")

    if excluded_metrics:
        print(f"\n❌ EXCLUDED METRICS (>{nan_threshold*100}% missing):")
        for metric in excluded_metrics:
            percentage = missing_percentages[metric]
            print(f"   • {metric} ({percentage:.1f}% missing)")

    # Filter the metrics data to only include valid metrics
    metrics_data = metrics_data[valid_metrics]
    print(f"\n✓ Proceeding with {len(valid_metrics)} metrics for correlation analysis...")

    # Calculate correlation matrix
    correlation_matrix = metrics_data.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    valid_metric_names = list(metrics_data.columns)  # Use filtered metric names
    n_metrics = len(valid_metric_names)

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= correlation_threshold and not pd.isna(corr_value):
                high_corr_pairs.append(
                    {"metric1": valid_metric_names[i], "metric2": valid_metric_names[j], "correlation": corr_value}
                )

    # Sort by absolute correlation value
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return correlation_matrix, high_corr_pairs


def create_correlation_heatmap(correlation_matrix, output_path="correlation_heatmap.png"):
    """Create and save a correlation heatmap."""
    plt.figure(figsize=(20, 16))

    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=False,  # Don't annotate due to size
        cmap="RdBu_r",
        center=0,
        square=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Metrics Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Correlation heatmap saved to: {output_path}")


def suggest_metrics_to_remove(high_corr_pairs, correlation_matrix):
    """Suggest which metrics to remove based on correlation analysis."""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR REDUNDANT METRICS")
    print("=" * 50)

    if not high_corr_pairs:
        print("✓ No highly correlated metric pairs found!")
        return []

    # Count how many high correlations each metric has
    metric_corr_counts = {}
    for pair in high_corr_pairs:
        metric1, metric2 = pair["metric1"], pair["metric2"]
        metric_corr_counts[metric1] = metric_corr_counts.get(metric1, 0) + 1
        metric_corr_counts[metric2] = metric_corr_counts.get(metric2, 0) + 1

    # Suggest removal candidates (metrics with many high correlations)
    removal_candidates = []
    processed_pairs = set()

    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs:")
    print("-" * 50)

    for pair in high_corr_pairs:
        metric1, metric2, corr = pair["metric1"], pair["metric2"], pair["correlation"]
        pair_key = tuple(sorted([metric1, metric2]))

        if pair_key not in processed_pairs:
            print(f"{metric1:<30} ↔ {metric2:<30} | r = {corr:6.3f}")

            # Suggest which one to remove based on various criteria
            if metric_corr_counts[metric1] > metric_corr_counts[metric2]:
                suggestion = metric1
            elif metric_corr_counts[metric2] > metric_corr_counts[metric1]:
                suggestion = metric2
            else:
                # If equal, prefer removing more complex/derived metrics
                if any(pattern in metric1 for pattern in ["binned_", "rate_", "ratio"]):
                    suggestion = metric1
                elif any(pattern in metric2 for pattern in ["binned_", "rate_", "ratio"]):
                    suggestion = metric2
                else:
                    suggestion = metric2  # Default to second one

            removal_candidates.append(suggestion)
            processed_pairs.add(pair_key)

    # Get unique removal candidates
    unique_candidates = list(set(removal_candidates))

    print(f"\n📋 SUGGESTED METRICS TO CONSIDER REMOVING:")
    print("-" * 50)
    for candidate in sorted(unique_candidates):
        corr_count = metric_corr_counts.get(candidate, 0)
        print(f"• {candidate:<30} (involved in {corr_count} high correlations)")

    return unique_candidates


def create_final_metrics_list(correlation_matrix, removal_candidates, keep_for_biological_significance=None):
    """Create the final list of metrics for PCA after removing highly correlated ones."""
    if keep_for_biological_significance is None:
        keep_for_biological_significance = ["pulled"]

    # Start with all metrics that passed the NaN filtering
    final_metrics = list(correlation_matrix.columns)

    # Remove the suggested candidates, except those we want to keep for biological significance
    metrics_to_remove = [metric for metric in removal_candidates if metric not in keep_for_biological_significance]

    final_metrics_for_pca = [metric for metric in final_metrics if metric not in metrics_to_remove]

    print(f"\n🎯 FINAL METRICS LIST FOR PCA")
    print("=" * 60)
    print(f"   • Started with: {len(correlation_matrix.columns)} metrics (after NaN filtering)")
    print(f"   • Removal candidates: {len(removal_candidates)}")
    print(f"   • Kept for biological significance: {keep_for_biological_significance}")
    print(f"   • Actually removed: {len(metrics_to_remove)}")
    print(f"   • Final metrics for PCA: {len(final_metrics_for_pca)}")

    if metrics_to_remove:
        print(f"\n❌ REMOVED METRICS:")
        for metric in sorted(metrics_to_remove):
            print(f"   • {metric}")

    if keep_for_biological_significance:
        kept_candidates = [m for m in removal_candidates if m in keep_for_biological_significance]
        if kept_candidates:
            print(f"\n🧬 KEPT FOR BIOLOGICAL SIGNIFICANCE:")
            for metric in sorted(kept_candidates):
                print(f"   • {metric}")

    print(f"\n✅ FINAL METRICS FOR PCA ({len(final_metrics_for_pca)} metrics):")
    print("-" * 60)
    for i, metric in enumerate(sorted(final_metrics_for_pca), 1):
        print(f"{i:3d}. {metric}")

    return final_metrics_for_pca


def save_correlation_analysis(correlation_matrix, high_corr_pairs, metric_columns, output_dir="."):
    """Save correlation analysis results."""
    output_dir = Path(output_dir)

    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / "metrics_correlation_matrix.csv")

    # Save high correlation pairs
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    # Save metrics list
    metrics_df = pd.DataFrame({"metric": metric_columns})
    metrics_df.to_csv(output_dir / "metrics_list.csv", index=False)

    print(f"\n✓ Analysis results saved to {output_dir}/")


def main():
    """Main analysis function."""
    # Configuration
    DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250804_17_summary_TNT_screen_Data/summary/pooled_summary.feather"
    CORRELATION_THRESHOLD = 0.8  # Adjust this threshold as needed
    NAN_THRESHOLD = 0.05  # Exclude metrics with more than 5% missing values

    print("🔍 BALLPUSHING METRICS CORRELATION ANALYSIS")
    print("=" * 50)

    # Load data
    print(f"📂 Loading data from: {DATA_PATH}")
    dataset = load_metrics_data(DATA_PATH)
    if dataset is None:
        return

    # Identify metrics
    print(f"\n🎯 Identifying metric columns...")
    metric_columns = identify_metric_columns(dataset)

    print(f"\n📊 Found {len(metric_columns)} potential metrics:")
    for i, metric in enumerate(metric_columns, 1):
        print(f"{i:3d}. {metric}")

    # Analyze correlations (now includes NaN analysis and filtering)
    print(f"\n🔗 Analyzing correlations (threshold: {CORRELATION_THRESHOLD})...")
    print(f"⚠️  Using 5% NaN filtering: excluding metrics with >5% missing values")
    correlation_matrix, high_corr_pairs = analyze_correlations(
        dataset, metric_columns, CORRELATION_THRESHOLD, NAN_THRESHOLD
    )

    # Get the final list of metrics used (after NaN filtering)
    final_metrics = list(correlation_matrix.columns)

    # Create visualizations
    print(f"\n📈 Creating correlation heatmap...")
    create_correlation_heatmap(correlation_matrix)

    # Suggest metrics to remove
    removal_candidates = suggest_metrics_to_remove(high_corr_pairs, correlation_matrix)

    # Create final metrics list for PCA
    final_metrics_for_pca = create_final_metrics_list(
        correlation_matrix, removal_candidates, keep_for_biological_significance=["pulled"]
    )

    # Save results
    save_correlation_analysis(correlation_matrix, high_corr_pairs, final_metrics)

    # Save the final metrics list for PCA
    final_metrics_df = pd.DataFrame({"metric": final_metrics_for_pca})
    final_metrics_df.to_csv("final_metrics_for_pca.csv", index=False)
    print(f"\n💾 Final PCA metrics list saved to: final_metrics_for_pca.csv")

    print(f"\n✅ Analysis complete!")
    print(f"   • Initial metrics found: {len(metric_columns)}")
    print(f"   • Final metrics analyzed: {len(final_metrics)}")
    print(f"   • High correlation pairs found: {len(high_corr_pairs)}")
    print(f"   • Metrics suggested for removal: {len(removal_candidates)}")
    print(f"   • Final metrics for PCA: {len(final_metrics_for_pca)}")

    return final_metrics_for_pca


if __name__ == "__main__":
    main()
