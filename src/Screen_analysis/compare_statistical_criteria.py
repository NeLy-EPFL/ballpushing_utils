#!/usr/bin/env python3
"""
Comprehensive Statistical Criteria Comparison
Tests all possible combinations of statistical tests to identify hits.
Compares results across different criteria to find which matches the old analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from pathlib import Path
import json

# Add path for Config
sys.path.append("/home/matthias/ballpushing_utils")
import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Compare statistical criteria for hit identification")
    parser.add_argument("results_dir", help="Directory containing PCA analysis results")
    parser.add_argument("--output-dir", default="statistical_criteria_comparison", help="Output directory")
    parser.add_argument("--consistency-threshold", type=float, default=0.80, help="Minimum consistency threshold")
    return parser.parse_args()


def load_all_configuration_results(results_dir):
    """
    Load detailed statistical results.
    We need the BEST configuration's detailed results to apply different criteria.
    The consistency calculation will be based on how criteria affect the 28-config consensus.
    """
    data_files_dir = os.path.join(results_dir, "data_files")

    if not os.path.exists(data_files_dir):
        print(f"❌ Data files directory not found: {data_files_dir}")
        return None, None, None

    # Load the combined statistical results (best configuration)
    stats_file = os.path.join(data_files_dir, "static_pca_stats_results_allmethods_tailoredctrls.csv")

    if not os.path.exists(stats_file):
        print(f"❌ Statistical results file not found: {stats_file}")
        return None, None, None

    stats_df = pd.read_csv(stats_file)
    print(f"📊 Loaded statistical results from best configuration")
    print(f"   Genotypes tested: {len(stats_df)}")

    # Load enhanced consistency scores (tracks hits across all 28 configurations)
    consistency_file = os.path.join(data_files_dir, "enhanced_consistency_scores.csv")

    if not os.path.exists(consistency_file):
        print(f"❌ Enhanced consistency file not found: {consistency_file}")
        return None, None, None

    consistency_df = pd.read_csv(consistency_file)
    print(f"   Genotypes in consistency analysis: {len(consistency_df)}")
    print(f"   Total configurations: {consistency_df['Total_Configs'].iloc[0]}")

    total_configs = int(consistency_df["Total_Configs"].iloc[0])

    return stats_df, consistency_df, total_configs


def define_statistical_criteria():
    """Define all statistical criteria to test"""
    criteria = {
        # Single test criteria
        "MW_only": {
            "name": "Mann-Whitney Only",
            "description": "Requires ≥1 significant PC (Mann-Whitney FDR < 0.05)",
            "filter": lambda df: df["MannWhitney_any_dim_significant"] == True,
        },
        "Perm_only": {
            "name": "Permutation Only",
            "description": "Requires Permutation test FDR < 0.05",
            "filter": lambda df: df["Permutation_FDR_significant"] == True,
        },
        "Maha_only": {
            "name": "Mahalanobis Only",
            "description": "Requires Mahalanobis test FDR < 0.05",
            "filter": lambda df: df["Mahalanobis_FDR_significant"] == True,
        },
        # Two-test criteria (AND logic)
        "Perm_MW": {
            "name": "Permutation + Mann-Whitney",
            "description": "Requires Permutation FDR < 0.05 AND ≥1 significant PC",
            "filter": lambda df: (df["Permutation_FDR_significant"] == True)
            & (df["MannWhitney_any_dim_significant"] == True),
        },
        "Perm_Maha": {
            "name": "Permutation + Mahalanobis",
            "description": "Requires Permutation FDR < 0.05 AND Mahalanobis FDR < 0.05",
            "filter": lambda df: (df["Permutation_FDR_significant"] == True)
            & (df["Mahalanobis_FDR_significant"] == True),
        },
        "MW_Maha": {
            "name": "Mann-Whitney + Mahalanobis",
            "description": "Requires ≥1 significant PC AND Mahalanobis FDR < 0.05",
            "filter": lambda df: (df["MannWhitney_any_dim_significant"] == True)
            & (df["Mahalanobis_FDR_significant"] == True),
        },
        # Triple test criterion
        "Triple": {
            "name": "Triple Test (All Three)",
            "description": "Requires Mann-Whitney AND Permutation AND Mahalanobis (all FDR < 0.05)",
            "filter": lambda df: (df["MannWhitney_any_dim_significant"] == True)
            & (df["Permutation_FDR_significant"] == True)
            & (df["Mahalanobis_FDR_significant"] == True),
        },
        # Two-test criteria (OR logic) - for completeness
        "Perm_OR_MW": {
            "name": "Permutation OR Mann-Whitney",
            "description": "Requires Permutation FDR < 0.05 OR ≥1 significant PC",
            "filter": lambda df: (df["Permutation_FDR_significant"] == True)
            | (df["MannWhitney_any_dim_significant"] == True),
        },
        "Perm_OR_Maha": {
            "name": "Permutation OR Mahalanobis",
            "description": "Requires Permutation FDR < 0.05 OR Mahalanobis FDR < 0.05",
            "filter": lambda df: (df["Permutation_FDR_significant"] == True)
            | (df["Mahalanobis_FDR_significant"] == True),
        },
    }

    return criteria


def apply_criteria_and_calculate_consistency(
    stats_df, consistency_df, total_configs, criteria, consistency_threshold=0.80
):
    """
    Apply each statistical criterion to the best configuration results,
    then filter by overall consistency across all 28 configurations.

    Logic:
    1. Apply statistical criterion to best configuration (determines IF genotype is significant)
    2. For significant genotypes, use their actual consistency from all 28 configs
    3. Filter by consistency threshold
    """

    results_by_criterion = {}

    print(f"\n🔬 Applying {len(criteria)} statistical criteria...")
    print(f"   Total configurations (for consistency): {total_configs}")
    print(f"   Consistency threshold: {consistency_threshold:.0%}")
    print(f"\n   Strategy: Apply criteria to best config, then filter by 28-config consistency")

    for criterion_id, criterion_info in criteria.items():
        criterion_name = criterion_info["name"]
        filter_func = criterion_info["filter"]

        # Apply filter to best configuration stats
        significant_in_best = stats_df[filter_func(stats_df)]["Nickname"].tolist()

        # Get consistency scores for these genotypes from all 28 configs
        genotype_consistency = consistency_df[consistency_df["Genotype"].isin(significant_in_best)].copy()
        genotype_consistency = genotype_consistency.sort_values("Combined_Consistency", ascending=False)

        # Filter by consistency threshold
        high_consistency = genotype_consistency[
            genotype_consistency["Combined_Consistency"] >= consistency_threshold
        ].copy()

        results_by_criterion[criterion_id] = {
            "name": criterion_name,
            "description": criterion_info["description"],
            "total_hits_in_best": len(significant_in_best),
            "high_consistency_hits": len(high_consistency),
            "genotypes": high_consistency["Genotype"].tolist(),
            "consistency_df": genotype_consistency,
        }

        print(
            f"   ✓ {criterion_name:35s}: {len(significant_in_best):3d} in best config → {len(high_consistency):3d} high-consistency (≥{consistency_threshold:.0%})"
        )

    return results_by_criterion


def create_comparison_matrix(results_by_criterion):
    """Create comparison matrix showing overlap between criteria"""

    criteria_ids = list(results_by_criterion.keys())
    n_criteria = len(criteria_ids)

    # Create overlap matrix
    overlap_matrix = pd.DataFrame(0, index=criteria_ids, columns=criteria_ids)

    for i, crit1 in enumerate(criteria_ids):
        genotypes1 = set(results_by_criterion[crit1]["genotypes"])
        for j, crit2 in enumerate(criteria_ids):
            genotypes2 = set(results_by_criterion[crit2]["genotypes"])
            overlap = len(genotypes1 & genotypes2)
            overlap_matrix.loc[crit1, crit2] = overlap

    return overlap_matrix


def plot_comparison_heatmap(overlap_matrix, results_by_criterion, output_dir):
    """Create heatmap showing overlap between criteria"""

    # Create labels with hit counts
    labels = [
        f"{results_by_criterion[crit]['name']}\n(n={results_by_criterion[crit]['high_consistency_hits']})"
        for crit in overlap_matrix.index
    ]

    plt.figure(figsize=(14, 12))

    # Plot heatmap
    ax = sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Number of Overlapping Genotypes"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(
        "Overlap Between Statistical Criteria\n(High-Consistency Hits ≥80%)", fontsize=14, fontweight="bold", pad=20
    )
    plt.xlabel("Statistical Criterion", fontsize=12, fontweight="bold")
    plt.ylabel("Statistical Criterion", fontsize=12, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    png_file = os.path.join(output_dir, "criteria_overlap_heatmap.png")
    pdf_file = os.path.join(output_dir, "criteria_overlap_heatmap.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"   💾 Overlap heatmap saved: {png_file}")

    plt.close()


def plot_hit_counts_comparison(results_by_criterion, output_dir):
    """Bar plot comparing hit counts across criteria"""

    # Prepare data
    criteria_names = [results_by_criterion[crit]["name"] for crit in results_by_criterion.keys()]
    hit_counts = [results_by_criterion[crit]["high_consistency_hits"] for crit in results_by_criterion.keys()]

    # Create color based on hit count (highlight the one close to 29)
    colors = ["red" if abs(count - 29) <= 1 else "steelblue" for count in hit_counts]

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(criteria_names, hit_counts, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, hit_counts)):
        ax.text(count + 0.5, bar.get_y() + bar.get_height() / 2, f"{count}", va="center", fontweight="bold")

    # Add reference line at 29 (old analysis target)
    ax.axvline(x=29, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Target: 29 hits (old analysis)")

    ax.set_xlabel("Number of High-Consistency Hits (≥80%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Statistical Criterion", fontsize=12, fontweight="bold")
    ax.set_title("Comparison of Hit Counts Across Statistical Criteria", fontsize=14, fontweight="bold", pad=20)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    png_file = os.path.join(output_dir, "hit_counts_comparison.png")
    pdf_file = os.path.join(output_dir, "hit_counts_comparison.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"   💾 Hit counts comparison saved: {png_file}")

    plt.close()


def create_detailed_comparison_table(results_by_criterion, output_dir):
    """Create detailed CSV showing which genotypes appear in which criteria"""

    # Get all unique genotypes
    all_genotypes = set()
    for result in results_by_criterion.values():
        all_genotypes.update(result["genotypes"])

    all_genotypes = sorted(all_genotypes)

    # Create DataFrame
    comparison_data = []

    for genotype in all_genotypes:
        row = {"Genotype": genotype}

        # Check presence in each criterion
        for crit_id, result in results_by_criterion.items():
            row[result["name"]] = genotype in result["genotypes"]

        # Count how many criteria include this genotype
        row["Total_Criteria"] = sum(genotype in result["genotypes"] for result in results_by_criterion.values())

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Total_Criteria", ascending=False)

    # Save to CSV
    csv_file = os.path.join(output_dir, "genotype_by_criteria_comparison.csv")
    comparison_df.to_csv(csv_file, index=False)
    print(f"   💾 Detailed comparison table saved: {csv_file}")

    return comparison_df


def find_best_matching_criterion(results_by_criterion, target_count=29):
    """Find which criterion(a) give closest to target hit count"""

    print(f"\n🎯 FINDING BEST MATCH FOR {target_count} HITS:")
    print("=" * 70)

    matches = []
    for crit_id, result in results_by_criterion.items():
        count = result["high_consistency_hits"]
        diff = abs(count - target_count)
        matches.append(
            {"criterion": result["name"], "count": count, "difference": diff, "genotypes": result["genotypes"]}
        )

    # Sort by difference from target
    matches_sorted = sorted(matches, key=lambda x: x["difference"])

    print(f"\nCriteria sorted by closeness to {target_count} hits:\n")
    for i, match in enumerate(matches_sorted[:5], 1):
        print(f"{i}. {match['criterion']:35s}: {match['count']:3d} hits (diff: {match['difference']:+3d})")

    # Return best match(es)
    best_matches = [m for m in matches_sorted if m["difference"] == matches_sorted[0]["difference"]]

    if best_matches[0]["difference"] == 0:
        print(f"\n✅ EXACT MATCH FOUND!")
    else:
        print(f"\n⚠️  No exact match. Closest is {best_matches[0]['difference']} away.")

    return best_matches


def create_summary_report(results_by_criterion, overlap_matrix, output_dir, target_count=29):
    """Create comprehensive summary report"""

    summary_file = os.path.join(output_dir, "statistical_criteria_comparison_summary.txt")

    with open(summary_file, "w") as f:
        f.write("STATISTICAL CRITERIA COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("OBJECTIVE:\n")
        f.write(f"Compare different statistical criteria to identify which produces\n")
        f.write(f"{target_count} high-consistency hits (matching old analysis).\n\n")

        f.write("CRITERIA TESTED:\n")
        f.write("-" * 80 + "\n")
        for crit_id, result in results_by_criterion.items():
            f.write(f"\n{result['name']}:\n")
            f.write(f"  Description: {result['description']}\n")
            f.write(f"  High-consistency hits (≥80%): {result['high_consistency_hits']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 80 + "\n\n")

        # Find best matches
        best_matches = find_best_matching_criterion(results_by_criterion, target_count)

        for match in best_matches:
            f.write(f"\nBEST MATCH: {match['criterion']}\n")
            f.write(f"  Hit count: {match['count']}\n")
            f.write(f"  Difference from target ({target_count}): {match['difference']}\n")
            f.write(f"\n  Genotypes:\n")
            for i, genotype in enumerate(sorted(match["genotypes"]), 1):
                f.write(f"    {i:2d}. {genotype}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERLAP ANALYSIS:\n")
        f.write("-" * 80 + "\n\n")

        # Find most conservative (smallest overlap with all)
        criteria_list = list(results_by_criterion.keys())
        for crit_id in criteria_list:
            genotypes = set(results_by_criterion[crit_id]["genotypes"])
            f.write(f"\n{results_by_criterion[crit_id]['name']} ({len(genotypes)} hits):\n")

            # Show overlap with other criteria
            for other_crit_id in criteria_list:
                if other_crit_id != crit_id:
                    other_genotypes = set(results_by_criterion[other_crit_id]["genotypes"])
                    overlap = len(genotypes & other_genotypes)
                    unique = len(genotypes - other_genotypes)
                    f.write(
                        f"  vs {results_by_criterion[other_crit_id]['name']:35s}: {overlap:3d} common, {unique:3d} unique\n"
                    )

    print(f"   💾 Summary report saved: {summary_file}")


def main():
    args = parse_args()

    print("🔬 COMPREHENSIVE STATISTICAL CRITERIA COMPARISON")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Consistency threshold: {args.consistency_threshold:.0%}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print("\n📊 Loading configuration results...")
    stats_df, consistency_df, total_configs = load_all_configuration_results(args.results_dir)

    if stats_df is None or consistency_df is None:
        return

    # Define criteria
    criteria = define_statistical_criteria()

    # Apply criteria
    results_by_criterion = apply_criteria_and_calculate_consistency(
        stats_df, consistency_df, total_configs, criteria, args.consistency_threshold
    )

    # Create comparison matrix
    print("\n📊 Creating comparison visualizations...")
    overlap_matrix = create_comparison_matrix(results_by_criterion)

    # Generate plots
    plot_comparison_heatmap(overlap_matrix, results_by_criterion, args.output_dir)
    plot_hit_counts_comparison(results_by_criterion, args.output_dir)

    # Create detailed table
    comparison_df = create_detailed_comparison_table(results_by_criterion, args.output_dir)

    # Find best match
    best_matches = find_best_matching_criterion(results_by_criterion, target_count=29)

    # Create summary report
    create_summary_report(results_by_criterion, overlap_matrix, args.output_dir, target_count=29)

    print("\n✅ ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {args.output_dir}")
    print(f"\n📊 Key files:")
    print(f"   • criteria_overlap_heatmap.png/pdf")
    print(f"   • hit_counts_comparison.png/pdf")
    print(f"   • genotype_by_criteria_comparison.csv")
    print(f"   • statistical_criteria_comparison_summary.txt")


if __name__ == "__main__":
    main()
