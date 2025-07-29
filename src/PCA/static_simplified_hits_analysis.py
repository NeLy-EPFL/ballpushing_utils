#!/usr/bin/env python3
"""
Create simplified statistical analysis and hits heatmap for static PCA (NaN metrics removed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys
import glob
import os

sys.path.append("/home/matthias/ballpushing_utils")
import Config


def detect_most_recent_pca_method():
    """Detect the most recent PCA method used based on file timestamps"""
    pca_files = glob.glob("static_pca_stats_results_allmethods*tailored*.csv")
    sparsepca_files = glob.glob("static_sparsepca_stats_results_allmethods*tailored*.csv")

    all_files = [(f, "pca") for f in pca_files] + [(f, "sparsepca") for f in sparsepca_files]

    if not all_files:
        print("No PCA result files found")
        return "pca", "static_pca_stats_results_allmethods_tailoredctrls.csv"

    # Sort by modification time (most recent first)
    all_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    most_recent_file, method = all_files[0]

    print(f"Most recent PCA method detected: {method.upper()}")
    print(f"Using file: {most_recent_file}")

    return method, most_recent_file


def perform_simplified_static_analysis():
    """Apply simplified Mann-Whitney analysis to significant hits from permutation test"""

    # Detect the most recent PCA method and get the appropriate file
    pca_method, comprehensive_file = detect_most_recent_pca_method()
    method_prefix = "SparsePCA" if pca_method == "sparsepca" else "PCA"

    # Load the existing comprehensive results to get permutation-significant hits
    try:
        comprehensive_results = pd.read_csv(comprehensive_file)
        print(f"Loaded comprehensive results: {len(comprehensive_results)} genotypes")
    except FileNotFoundError:
        print(f"Error: Could not find {comprehensive_file}")
        print("Please run the main PCA analysis first.")
        return pd.DataFrame()

    # Filter to permutation-significant hits (overall significant distance from control)
    # Use same significance criteria as Summary_Heatmap_Simplified.py
    permutation_significant = comprehensive_results[comprehensive_results["Permutation_pval"] < 0.05].copy()

    print(f"Found {len(permutation_significant)} genotypes with significant permutation p-values (<0.05)")
    if len(permutation_significant) > 0:
        print("Permutation-significant genotypes:")
        for _, row in permutation_significant.iterrows():
            print(f"  {row['Nickname']}: p = {row['Permutation_pval']:.3f}")

    # For these hits, examine which specific PCs are significant
    # Extract Mann-Whitney results that are already computed
    results = []

    for _, row in permutation_significant.iterrows():
        nickname = row["Nickname"]
        control_name = row["Control"]

        # Parse Mann-Whitney results from the comprehensive analysis
        # Use try-except for each field to handle any pandas/parsing issues
        try:
            significant_dims = eval(str(row["MannWhitney_significant_dims"]))
        except:
            significant_dims = []

        try:
            mannwhitney_pvals = eval(str(row["MannWhitney_raw_pvals"]))
        except:
            mannwhitney_pvals = []

        try:
            mannwhitney_corrected = eval(str(row["MannWhitney_corrected_pvals"]))
        except:
            mannwhitney_corrected = []

        try:
            dims_tested = eval(str(row["MannWhitney_dims_tested"]))
        except:
            dims_tested = []

        # Build PC-specific results
        pc_results = {}
        # Only use the PCs that were actually tested (from dims_tested)
        available_components = dims_tested if dims_tested else []

        for i, pc in enumerate(available_components):
            if i < len(mannwhitney_pvals) and i < len(mannwhitney_corrected):
                pc_results[f"{pc}_pval"] = mannwhitney_pvals[i]
                pc_results[f"{pc}_pval_corrected"] = mannwhitney_corrected[i]
                pc_results[f"{pc}_significant"] = pc in significant_dims
            else:
                pc_results[f"{pc}_pval"] = np.nan
                pc_results[f"{pc}_pval_corrected"] = np.nan
                pc_results[f"{pc}_significant"] = False

        # Build result dictionary
        result_dict = {
            "genotype": nickname,
            "control": control_name,
            "significant": True,  # All these are permutation-significant
            "significant_PCs": significant_dims,
            "num_significant_PCs": len(significant_dims),
            "permutation_pval": row["Permutation_pval"],
        }
        result_dict.update(pc_results)
        results.append(result_dict)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No permutation-significant hits found")
        return pd.DataFrame()

    # Save results with method-specific naming
    output_file = f"static_{pca_method}_stats_simplified_tailoredctrls.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")

    # Determine the number of PCs that were tested (dynamically from the data)
    max_pcs_tested = 0
    if len(results_df) > 0:
        # Get the maximum number of tested PCs across all genotypes
        for _, row in results_df.iterrows():
            pc_cols = [col for col in results_df.columns if col.startswith(method_prefix) and col.endswith("_pval")]
            max_pcs_tested = max(max_pcs_tested, len(pc_cols))

    # Print summary
    print(f"\nSimplified hits analysis results (from permutation-significant genotypes):")
    print(f"Found {len(results_df)} genotypes with significant overall distance from control")

    if len(results_df) > 0:
        print("Genotypes and their significant PCs:")
        for _, row in results_df.iterrows():
            print(
                f"  {row['genotype']}: {row['num_significant_PCs']}/{max_pcs_tested} PCs significant ({row['significant_PCs']})"
            )

        # Show distribution of significant PCs
        pc_distribution = results_df["num_significant_PCs"].value_counts().sort_index()
        print(f"\nDistribution of significant PCs per genotype: {dict(pc_distribution)}")
        print(f"Note: Analysis used {max_pcs_tested} PCs (up to 95% cumulative variance)")

    return results_df


def create_hits_heatmap(results_df):
    """Create a heatmap showing which genotypes have significant hits in which PCs"""

    if len(results_df[results_df["significant"]]) == 0:
        print("No significant hits to visualize")
        return None

    # Detect the most recent PCA method for proper component naming
    pca_method, _ = detect_most_recent_pca_method()
    method_prefix = "SparsePCA" if pca_method == "sparsepca" else "PCA"

    # Get significant results
    significant_results = results_df[results_df["significant"]].copy()

    # Get all PC components that were tested (dynamic method detection)
    pc_columns = [col for col in results_df.columns if col.endswith("_significant") and col.startswith(method_prefix)]
    pc_names = [col.replace("_significant", "") for col in pc_columns]

    # Create binary matrix for heatmap
    hit_matrix = []
    genotype_names = []

    for _, row in significant_results.iterrows():
        genotype_names.append(row["genotype"])
        hit_row = []
        for pc_col in pc_columns:
            hit_row.append(1 if row.get(pc_col, False) else 0)
        hit_matrix.append(hit_row)

    if len(hit_matrix) == 0:
        print("No hits matrix to create")
        return None

    hit_df = pd.DataFrame(hit_matrix, index=genotype_names, columns=pc_names)

    # Set up the plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, max(8, len(genotype_names) * 0.4)))

    # Create the heatmap
    sns.heatmap(
        hit_df,
        annot=True,
        fmt="d",
        cmap=["white", "red"],
        cbar_kws={"label": "Significant (1) vs Not Significant (0)"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    # Customize the plot
    method_display = "Sparse PCA" if pca_method == "sparsepca" else "Regular PCA"
    ax.set_title(
        f"Significant Hits Heatmap - Simplified Static {method_display}\n"
        f"Mann-Whitney Test with FDR Correction (α=0.05)\n"
        f"{len(genotype_names)} significant genotypes across {len(pc_names)} PCs",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Principal Components", fontsize=12, fontweight="bold")
    ax.set_ylabel("Genotypes", fontsize=12, fontweight="bold")

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    # Save with method-specific naming
    png_file = f"static_{pca_method}_hits_simplified_heatmap.png"
    pdf_file = f"static_{pca_method}_hits_simplified_heatmap.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"Hits heatmap saved: {png_file} and {pdf_file}")

    return fig


def create_comparison_summary():
    """Create a summary comparing permutation significance vs PC-level significance"""

    # Detect the most recent PCA method
    pca_method, comprehensive_file = detect_most_recent_pca_method()
    method_display = "Sparse PCA" if pca_method == "sparsepca" else "Regular PCA"

    # Load comprehensive results
    try:
        comprehensive_df = pd.read_csv(comprehensive_file)
        permutation_hits = len(comprehensive_df[comprehensive_df["Permutation_pval"] < 0.05])
        complex_hits = len(
            comprehensive_df[
                (comprehensive_df["MannWhitney_any_dim_significant"])
                & (comprehensive_df["Permutation_FDR_significant"])
                & (comprehensive_df["Mahalanobis_FDR_significant"])
            ]
        )
    except FileNotFoundError:
        permutation_hits = "Not available"
        complex_hits = "Not available"

    # Load simplified results (PC-level analysis of permutation hits)
    simplified_file = f"static_{pca_method}_stats_simplified_tailoredctrls.csv"
    try:
        simplified_df = pd.read_csv(simplified_file)
        pc_hits = len(simplified_df)  # All entries are permutation-significant
        total_significant_pcs = simplified_df["num_significant_PCs"].sum()
    except FileNotFoundError:
        pc_hits = "Not available"
        total_significant_pcs = "Not available"

    print(f"\n{'='*60}")
    print(f"STATIC {method_display.upper()} ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Data source: {comprehensive_file}")
    print(f"Permutation-significant hits (p < 0.05): {permutation_hits}")
    print(f"Complex approach (all 3 tests significant): {complex_hits}")
    print(f"PC-level analysis of permutation hits: {pc_hits}")
    if isinstance(total_significant_pcs, int):
        print(f"Total significant PCs across all hits: {total_significant_pcs}")

    print(f"\nApproach:")
    print(f"1. Use permutation test for overall significance (ANOVA-style)")
    print(f"2. For permutation-significant hits, examine which PCs differ via Mann-Whitney")


def main():
    """Main analysis function"""
    print("=" * 80)
    print("PC-LEVEL ANALYSIS OF PERMUTATION-SIGNIFICANT HITS")
    print("=" * 80)
    print("Step 1: Find hits using permutation test (overall significance)")
    print("Step 2: For each hit, examine which PCs are significantly different")
    print("=" * 80)

    # Perform PC-level analysis of permutation-significant hits
    results_df = perform_simplified_static_analysis()

    # Create hits heatmap
    if not results_df.empty:
        create_hits_heatmap(results_df)

    # Create comparison summary
    create_comparison_summary()

    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()
