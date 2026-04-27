#!/usr/bin/env python3
"""
Loadings heatmap for best PCA configuration from consistency analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import json

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONSISTENCY_DIR = "consistency_analysis"
CONFIGS_PATH = os.path.join(SCRIPT_DIR, "multi_condition_pca_optimization/top_configurations.json")
BEST_PCA_DIR = "best_pca_analysis"  # Directory where best PCA results are stored

# Get output directory from command line argument
if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
    DATA_FILES_DIR = os.path.join(OUTPUT_DIR, "data_files")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    print(f"🎯 Using output directory: {OUTPUT_DIR}")
    print(f"📁 Data files directory: {DATA_FILES_DIR}")
    print(f"🎨 Plots directory: {PLOTS_DIR}")
else:
    OUTPUT_DIR = BEST_PCA_DIR
    DATA_FILES_DIR = OUTPUT_DIR
    PLOTS_DIR = OUTPUT_DIR
    print(f"⚠️  No output directory specified, using best PCA directory: {OUTPUT_DIR}")


def load_best_configuration():
    """Load the best PCA configuration details"""
    if not os.path.exists(CONFIGS_PATH):
        print(f"❌ Configuration file not found: {CONFIGS_PATH}")
        return None, None, None, None

    with open(CONFIGS_PATH, "r") as f:
        all_configs = json.load(f)

    # Find the configuration with the highest best_score
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
        return None, None, None, None

    print(f"🏆 BEST PCA CONFIGURATION:")
    print(f"   Configuration: {best_key}")
    print(f"   Score: {best_score:.4f}")
    print(f"   Condition: {best_config['condition']}")
    print(f"   Method: {best_config['method']}")
    print(f"   Metrics: {len(best_config['metrics'])}")

    return (best_config["condition"], best_config["method"], best_config["metrics"], best_config.get("best_score", 0))


def get_consistency_stats():
    """Get consistency statistics for additional context"""
    consistency_file = os.path.join(CONSISTENCY_DIR, "genotype_consistency_scores.csv")

    if not os.path.exists(consistency_file):
        return None

    consistency_df = pd.read_csv(consistency_file)

    # Calculate tier counts
    tier_100 = len(consistency_df[consistency_df["Consistency_Percent"] == 100])
    tier_80_99 = len(
        consistency_df[(consistency_df["Consistency_Percent"] >= 80) & (consistency_df["Consistency_Percent"] < 100)]
    )
    tier_50_79 = len(
        consistency_df[(consistency_df["Consistency_Percent"] >= 50) & (consistency_df["Consistency_Percent"] < 80)]
    )

    return {
        "total_genotypes": len(consistency_df),
        "perfect_consistency": tier_100,
        "high_consistency": tier_80_99,
        "moderate_consistency": tier_50_79,
        "max_consistency": consistency_df["Consistency_Percent"].max(),
        "mean_consistency": consistency_df["Consistency_Percent"].mean(),
    }


def create_best_loadings_heatmap():
    """Create a heatmap of PCA loadings for the best configuration"""

    # Load best configuration details
    condition, method_type, metrics_list, best_score = load_best_configuration()
    if condition is None:
        print("❌ Could not load best configuration. Falling back to auto-detection.")
        return create_fallback_heatmap()

    # Get consistency statistics
    consistency_stats = get_consistency_stats()

    # Determine method name and loadings file
    method_name = "Sparse PCA" if method_type == "SparsePCA" else "Regular PCA"
    method_suffix = method_type.lower()

    loadings_file = os.path.join(DATA_FILES_DIR, f"best_{method_suffix}_loadings.csv")

    # Fallbacks: try OUTPUT_DIR root, then best_pca_analysis directory
    if not os.path.exists(loadings_file):
        alt1 = os.path.join(OUTPUT_DIR, f"best_{method_suffix}_loadings.csv")
        alt2 = os.path.join(SCRIPT_DIR, BEST_PCA_DIR, f"best_{method_suffix}_loadings.csv")
        if os.path.exists(alt1):
            loadings_file = alt1
        elif os.path.exists(alt2):
            loadings_file = alt2
        else:
            print(f"❌ Best loadings file not found in:")
            print(f"   {DATA_FILES_DIR}")
            print(f"   {OUTPUT_DIR}")
            print(f"   {alt2}")
            print("Please run the best PCA analysis first.")
            return None

    print(f"✅ Using {method_name} loadings from: {loadings_file}")

    # Load the loadings data
    loadings_df = pd.read_csv(loadings_file, index_col=0)

    print(f"📊 Loadings shape: {loadings_df.shape}")
    print(f"📋 PCA components: {list(loadings_df.index)}")
    print(f"📈 Metrics included: {len(loadings_df.columns)}")

    # Set up the plot with extra height for text
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(max(14, len(loadings_df.columns) * 0.6), max(10, len(loadings_df.index) * 0.8)))

    # Create the heatmap
    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",  # Diverging colormap
        center=0,
        cbar_kws={"label": "Loading Value"},
        ax=ax,
        annot_kws={"size": 9},
        linewidths=0.1,
        linecolor="gray",
    )

    # Customize the plot title
    title_lines = [f"{method_name} Loadings - Best Configuration", f"Condition: {condition} | Score: {best_score:.3f}"]

    if consistency_stats:
        title_lines.append(f"Based on {consistency_stats['total_genotypes']} genotypes consistency analysis")

    ax.set_title(
        "\n".join(title_lines),
        fontsize=16,
        fontweight="bold",
        pad=25,
    )
    ax.set_xlabel("Behavioral Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Principal Components", fontsize=12, fontweight="bold")

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Create detailed explanation text
    metrics_count = len(loadings_df.columns)
    components_count = len(loadings_df.index)

    explanation_lines = [
        f"📊 Analysis: {metrics_count} metrics across {components_count} components",
        f"🏆 Best configuration: {condition}_{method_type}",
        f"📈 Optimization score: {best_score:.3f}",
    ]

    if consistency_stats:
        explanation_lines.extend(
            [
                f"🎯 Consistency analysis: {consistency_stats['total_genotypes']} genotypes",
                f"🥇 Perfect consistency: {consistency_stats['perfect_consistency']} genotypes",
                f"🥈 High consistency (≥80%): {consistency_stats['high_consistency']} genotypes",
            ]
        )

    explanation_lines.append("📋 Values represent contribution of each metric to each PC")

    explanation_text = "\n".join(explanation_lines)

    # Add text box with explanation
    ax.text(
        0,
        -0.3,  # Position below the plot
        explanation_text,
        transform=ax.transAxes,
        fontsize=10,
        style="italic",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Save with descriptive filenames
    png_path = os.path.join(PLOTS_DIR, f"best_{method_suffix}_loadings_heatmap.png")
    pdf_path = os.path.join(PLOTS_DIR, f"best_{method_suffix}_loadings_heatmap.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"💾 Loadings heatmap saved: {png_path} and {pdf_path}")

    # Print summary of highest loadings by component
    print(f"\n🔍 HIGHEST ABSOLUTE LOADINGS BY COMPONENT:")
    print("-" * 60)
    for pc in loadings_df.index:
        pc_loadings = loadings_df.loc[pc].abs().sort_values(ascending=False)
        top_5 = pc_loadings.head(5)  # Show top 5 instead of 3
        loadings_str = ", ".join([f"{metric}({val:.2f})" for metric, val in top_5.items()])
        print(f"{pc:>12}: {loadings_str}")

    # Additional analysis: Find metrics with highest overall contribution
    print(f"\n📈 METRICS WITH HIGHEST OVERALL CONTRIBUTION:")
    print("-" * 60)
    overall_contribution = loadings_df.abs().mean(axis=0).sort_values(ascending=False)
    top_10_metrics = overall_contribution.head(10)
    for metric, contrib in top_10_metrics.items():
        print(f"{metric:>30}: {contrib:.3f}")

    # Sparsity analysis for Sparse PCA
    if method_type == "SparsePCA":
        total_loadings = loadings_df.size
        zero_loadings = (loadings_df == 0).sum().sum()
        sparsity = zero_loadings / total_loadings
        print(f"\n🕸️  SPARSITY ANALYSIS:")
        print(f"   Total loadings: {total_loadings}")
        print(f"   Zero loadings: {zero_loadings}")
        print(f"   Sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)")

    # Save detailed loadings summary
    summary_file = os.path.join(DATA_FILES_DIR, f"best_{method_suffix}_loadings_summary.txt")
    with open(summary_file, "w") as f:
        f.write("BEST PCA CONFIGURATION - LOADINGS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration: {condition}_{method_type}\n")
        f.write(f"Optimization score: {best_score:.4f}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Components: {components_count}\n")
        f.write(f"Metrics: {metrics_count}\n")

        if consistency_stats:
            f.write(f"\nConsistency Analysis:\n")
            f.write(f"  Total genotypes: {consistency_stats['total_genotypes']}\n")
            f.write(f"  Perfect consistency: {consistency_stats['perfect_consistency']}\n")
            f.write(f"  High consistency: {consistency_stats['high_consistency']}\n")
            f.write(f"  Mean consistency: {consistency_stats['mean_consistency']:.1f}%\n")

        f.write(f"\nTop Contributing Metrics (by average absolute loading):\n")
        for i, (metric, contrib) in enumerate(top_10_metrics.items(), 1):
            f.write(f"  {i:2d}. {metric}: {contrib:.3f}\n")

        f.write(f"\nHighest Loadings by Component:\n")
        for pc in loadings_df.index:
            pc_loadings = loadings_df.loc[pc].abs().sort_values(ascending=False)
            top_3 = pc_loadings.head(3)
            f.write(f"  {pc}: {', '.join([f'{metric}({val:.3f})' for metric, val in top_3.items()])}\n")

        if method_type == "SparsePCA":
            f.write(f"\nSparsity: {sparsity:.3f} ({sparsity*100:.1f}% zero loadings)\n")

    print(f"📄 Detailed summary saved: {summary_file}")

    return fig


def create_fallback_heatmap():
    """Fallback to original auto-detection method if best config not available"""
    print("⚠️  Falling back to auto-detection method...")

    # Auto-detect which method was used by checking available files
    # Try data_files subdirectory first, then OUTPUT_DIR root
    pca_file = os.path.join(DATA_FILES_DIR, "static_pca_loadings.csv")
    sparsepca_file = os.path.join(DATA_FILES_DIR, "static_sparsepca_loadings.csv")

    # Fallback to OUTPUT_DIR root if not in data_files
    if not os.path.exists(pca_file):
        pca_file = os.path.join(OUTPUT_DIR, "static_pca_loadings.csv")
    if not os.path.exists(sparsepca_file):
        sparsepca_file = os.path.join(OUTPUT_DIR, "static_sparsepca_loadings.csv")

    if Path(sparsepca_file).exists() and Path(pca_file).exists():
        # Both files exist, check which is more recent
        pca_time = Path(pca_file).stat().st_mtime
        sparsepca_time = Path(sparsepca_file).stat().st_mtime
        if sparsepca_time > pca_time:
            loadings_file = sparsepca_file
            method_name = "Sparse PCA"
        else:
            loadings_file = pca_file
            method_name = "Regular PCA"
    elif Path(sparsepca_file).exists():
        loadings_file = sparsepca_file
        method_name = "Sparse PCA"
    elif Path(pca_file).exists():
        loadings_file = pca_file
        method_name = "Regular PCA"
    else:
        raise FileNotFoundError(f"Neither {pca_file} nor {sparsepca_file} found!")

    print(f"Using {method_name} loadings from: {loadings_file}")

    # Load and create heatmap (original logic)
    loadings_df = pd.read_csv(loadings_file, index_col=0)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Loading Value"},
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title(
        f"{method_name} Loadings Heatmap - Fallback Analysis\n(Metrics with NaNs Removed)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Behavioral Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Principal Components", fontsize=12, fontweight="bold")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save with fallback naming
    method_suffix = "sparsepca" if "sparse" in method_name.lower() else "pca"
    plt.tight_layout()
    png_path = os.path.join(PLOTS_DIR, f"fallback_{method_suffix}_loadings_heatmap.png")
    pdf_path = os.path.join(PLOTS_DIR, f"fallback_{method_suffix}_loadings_heatmap.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Fallback loadings heatmap saved: {png_path} and {pdf_path}")

    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("CREATING LOADINGS HEATMAP - BEST PCA CONFIGURATION")
    print("=" * 80)

    try:
        fig = create_best_loadings_heatmap()
        if fig is not None:
            print("\n✅ Best configuration loadings heatmap completed!")
        else:
            print("\n❌ Could not create loadings heatmap")
    except Exception as e:
        print(f"\n❌ Error creating loadings heatmap: {e}")
        print("Please ensure the best PCA analysis has been run first.")

    print("=" * 80)
