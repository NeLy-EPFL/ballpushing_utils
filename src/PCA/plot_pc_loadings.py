#!/usr/bin/env python3
"""
Loadings heatmap for simplified PCA Static analysis (NaN metrics removed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_loadings_heatmap():
    """Create a heatmap of PCA loadings for the simplified analysis"""

    # Auto-detect which method was used by checking available files
    pca_file = "static_pca_loadings.csv"
    sparsepca_file = "static_sparsepca_loadings.csv"

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
        raise FileNotFoundError("Neither static_pca_loadings.csv nor static_sparsepca_loadings.csv found!")

    print(f"Using {method_name} loadings from: {loadings_file}")

    # Load the loadings data
    loadings_df = pd.read_csv(loadings_file, index_col=0)

    print(f"Loadings shape: {loadings_df.shape}")
    print(f"PCA components: {list(loadings_df.index)}")
    print(f"Metrics included: {list(loadings_df.columns)}")

    # Set up the plot with extra height for the text
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased height from 8 to 10

    # Create the heatmap
    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",  # Diverging colormap
        center=0,
        cbar_kws={"label": "Loading Value"},
        ax=ax,
        annot_kws={"size": 9},
    )

    # Customize the plot
    ax.set_title(
        f"{method_name} Loadings Heatmap - Simplified Static Analysis\n(Metrics with NaNs Removed)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Behavioral Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Principal Components", fontsize=12, fontweight="bold")

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add explanation text using a text box positioned within the figure
    metrics_count = len(loadings_df.columns)
    components_count = len(loadings_df.index)

    explanation_text = (
        f"Analysis includes {metrics_count} metrics (no missing values)\n"
        f"{components_count} principal components shown\n"
        f"Method: {method_name}\n"
        f"Values represent contribution of each metric to each PC"
    )

    # Add text box at the bottom left of the plot area, with more spacing
    ax.text(
        0,
        -0.25,  # Moved further down from -0.15 to -0.25
        explanation_text,
        transform=ax.transAxes,
        fontsize=10,
        style="italic",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Generate method-specific output filenames
    method_suffix = "sparsepca" if "sparse" in method_name.lower() else "pca"

    plt.tight_layout()
    plt.savefig(f"static_{method_suffix}_loadings_simplified_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"static_{method_suffix}_loadings_simplified_heatmap.pdf", bbox_inches="tight")
    print(f"Loadings heatmap saved: static_{method_suffix}_loadings_simplified_heatmap.png/pdf")

    # Print summary of highest loadings
    print(f"\nHighest absolute loadings by component:")
    for pc in loadings_df.index:
        pc_loadings = loadings_df.loc[pc].abs().sort_values(ascending=False)
        top_3 = pc_loadings.head(3)
        print(f"{pc}: {', '.join([f'{metric}({val:.3f})' for metric, val in top_3.items()])}")

    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("CREATING LOADINGS HEATMAP - SIMPLIFIED STATIC PCA")
    print("=" * 80)
    create_loadings_heatmap()
    print("Loadings heatmap completed!")
