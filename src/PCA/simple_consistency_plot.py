#!/usr/bin/env python3
"""
Simple Consistency Score Visualization
Creates a clean plot showing genotype consistency scores with 80% threshold line.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import argparse
import os
import sys

# Add path for Config module
sys.path.append("/home/matthias/ballpushing_utils")
try:
    from src.Config import Config

    HAS_CONFIG = True
except ImportError:
    print("⚠️  Could not import Config module - brain region coloring will be limited")
    HAS_CONFIG = False

# Set plotting style
plt.style.use("default")
sns.set_palette("viridis")


def load_nickname_mapping():
    """Load the simplified nickname mapping for visualization"""
    region_map_path = "/mnt/upramdya_data/MD/Region_map_250908.csv"
    print(f"📋 Loading nickname mapping from {region_map_path}")

    try:
        region_map = pd.read_csv(region_map_path)

        # Create mapping from original Nickname to Simplified Nickname
        nickname_mapping = dict(zip(region_map["Nickname"], region_map["Simplified Nickname"]))
        print(f"📋 Loaded {len(nickname_mapping)} nickname mappings")

        # Create brain region mapping for simplified nicknames
        simplified_to_region = dict(zip(region_map["Simplified Nickname"], region_map["Simplified region"]))
        print(f"📋 Loaded {len(simplified_to_region)} simplified nickname → brain region mappings")

        return nickname_mapping, simplified_to_region
    except Exception as e:
        print(f"⚠️  Could not load region mapping: {e}")
        return {}, {}


def get_brain_region_colors():
    """Get brain region color mapping"""
    if HAS_CONFIG:
        try:
            if hasattr(Config, "color_dict"):
                return Config.color_dict
        except Exception as e:
            print(f"⚠️  Could not load color dictionary from Config: {e}")

    # Use the exact color scheme from Config.py
    return {
        "MB": "#1f77b4",  # Blue
        "Vision": "#ff7f0e",  # Orange
        "LH": "#2ca02c",  # Green
        "Neuropeptide": "#d62728",  # Red
        "Olfaction": "#9467bd",  # Purple
        "MB extrinsic neurons": "#8c564b",  # Brown
        "CX": "#e377c2",  # Pink
        "Control": "#7f7f7f",  # Gray
        "None": "#bcbd22",  # Yellow-green
        "fchON": "#17becf",  # Cyan
        "JON": "#ffbb78",  # Light orange
        "DN": "#c5b0d5",  # Light purple
        "Unknown": "#D3D3D3",  # Light gray for unknown
    }


def colour_y_ticklabels(ax, nickname_to_region, color_dict):
    """Paint y-tick labels according to the genotype's brain region."""
    for tick in ax.get_yticklabels():
        region = nickname_to_region.get(tick.get_text(), None)
        if region in color_dict:
            tick.set_color(color_dict[region])


def apply_simplified_nicknames(genotype_list, nickname_mapping):
    """Apply simplified nicknames to genotype list for visualization"""
    if not nickname_mapping:
        return genotype_list

    simplified = []
    for genotype in genotype_list:
        simplified_name = nickname_mapping.get(genotype, genotype)
        simplified.append(simplified_name)

    return simplified


def create_simple_consistency_plot(
    consistency_df,
    output_dir=".",
    threshold=0.8,
    suffix="",
    nickname_mapping=None,
    simplified_to_region=None,
    color_dict=None,
):
    """
    Create a simple plot showing genotype consistency scores

    Args:
        consistency_df: DataFrame with genotype consistency data
        output_dir: Output directory
        threshold: Consistency threshold to highlight (default: 0.8 = 80%)
        suffix: Suffix for output filename
        nickname_mapping: Dictionary mapping nicknames to simplified nicknames
        simplified_to_region: Dictionary mapping simplified nicknames to brain regions
        color_dict: Dictionary mapping brain regions to colors
    """
    print(f"🎨 Creating simple consistency plot with {threshold*100}% threshold{suffix}...")

    # Sort by consistency score (descending)
    sorted_df = consistency_df.sort_values("Overall_Consistency", ascending=True).copy()

    # Apply simplified nicknames if available
    if nickname_mapping:
        sorted_df["Display_Name"] = sorted_df["Genotype"].map(nickname_mapping).fillna(sorted_df["Genotype"])
    else:
        sorted_df["Display_Name"] = sorted_df["Genotype"]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_df) * 0.3)))

    # Create color map based on consistency scores
    consistency_scores = sorted_df["Overall_Consistency"].values
    cmap = plt.colormaps.get_cmap("RdYlBu_r")  # Red-Yellow-Blue reversed (red=high, blue=low)
    colors = cmap(consistency_scores)

    # Create horizontal bar plot
    y_positions = np.arange(len(sorted_df))
    bars = ax.barh(y_positions, consistency_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Set genotype labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_df["Display_Name"], fontsize=10)

    # Color the y-tick labels by brain region if mappings are available
    if simplified_to_region and color_dict:
        colour_y_ticklabels(ax, simplified_to_region, color_dict)

    # Add threshold line
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"{threshold*100}% Threshold", zorder=10)

    # Highlight genotypes above threshold
    above_threshold = consistency_scores >= threshold
    n_above = above_threshold.sum()

    if n_above > 0:
        # Add background highlighting for above-threshold genotypes
        threshold_positions = y_positions[above_threshold]
        for pos in threshold_positions:
            ax.axhspan(pos - 0.4, pos + 0.4, alpha=0.2, color="green", zorder=1)

    # Formatting
    ax.set_xlabel("Overall Consistency Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Genotype", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Genotype Consistency Scores\n"
        f"{len(sorted_df)} genotypes tested | "
        f"{n_above} above {threshold*100}% threshold",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis limits and ticks
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{x:.0%}" for x in np.arange(0, 1.1, 0.1)])

    # Add grid
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)

    # Create legend elements
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="red", linestyle="--", label=f"{threshold*100}% Threshold")]

    # Add brain region colors to legend if available
    if simplified_to_region and color_dict:
        # Find unique brain regions in the data
        unique_regions = set()
        for genotype in sorted_df["Display_Name"]:
            region = simplified_to_region.get(genotype, "Unknown")
            unique_regions.add(region)

        # Add separator
        legend_elements.append(Patch(facecolor="none", label=""))
        legend_elements.append(Patch(facecolor="none", label="Brain Regions:"))

        # Add brain region colors
        for region in sorted(unique_regions):
            if region in color_dict:
                legend_elements.append(Patch(facecolor=color_dict[region], label=region))

    # Add legend
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add colorbar with more spacing
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.08)  # Increased pad from 0.02 to 0.08
    cbar.set_label("Consistency Score", rotation=270, labelpad=15, fontsize=10)

    # Add text annotations for values
    for i, (bar, score) in enumerate(zip(bars, consistency_scores)):
        # Only annotate if there's space and score is significant
        if score >= 0.05:  # Only annotate scores >= 5%
            ax.text(
                score + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1%}",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save plot
    filename = f"simple_consistency_scores{suffix}.png"
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # Save a pdf version as well
    pdf_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"💾 Saved plot as: {output_file} and {pdf_file}")
    plt.show()

    # Print summary statistics
    print(f"\n📊 CONSISTENCY SUMMARY:")
    print(f"   Total genotypes tested: {len(sorted_df)}")
    print(f"   Above {threshold*100}% threshold: {n_above} ({n_above/len(sorted_df)*100:.1f}%)")
    print(f"   Mean consistency: {consistency_scores.mean():.1%}")
    print(f"   Median consistency: {np.median(consistency_scores):.1%}")
    print(f"   Max consistency: {consistency_scores.max():.1%}")

    return output_file


def create_threshold_comparison_plot(consistency_df, output_dir=".", thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Create a plot showing how many genotypes pass different threshold levels
    """
    print(f"📈 Creating threshold comparison plot...")

    # Calculate counts for each threshold
    threshold_data = []
    for threshold in thresholds:
        count = (consistency_df["Overall_Consistency"] >= threshold).sum()
        percentage = count / len(consistency_df) * 100
        threshold_data.append(
            {"Threshold": f"{threshold:.0%}", "Count": count, "Percentage": percentage, "Threshold_Value": threshold}
        )

    threshold_df = pd.DataFrame(threshold_data)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Bar chart of counts
    bars1 = ax1.bar(threshold_df["Threshold"], threshold_df["Count"], color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Consistency Threshold", fontsize=12)
    ax1.set_ylabel("Number of Genotypes", fontsize=12)
    ax1.set_title("Genotypes Above Different Consistency Thresholds", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars1, threshold_df["Count"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Percentage chart
    bars2 = ax2.bar(threshold_df["Threshold"], threshold_df["Percentage"], color="coral", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Consistency Threshold", fontsize=12)
    ax2.set_ylabel("Percentage of Genotypes (%)", fontsize=12)
    ax2.set_title("Percentage of Genotypes Above Thresholds", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, pct in zip(bars2, threshold_df["Percentage"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Highlight 80% threshold
    for ax, bars in [(ax1, bars1), (ax2, bars2)]:
        # Find 80% threshold bar
        threshold_80_idx = threshold_df[threshold_df["Threshold_Value"] == 0.8].index
        if len(threshold_80_idx) > 0:
            idx = threshold_80_idx[0]
            bars[idx].set_color("red")
            bars[idx].set_alpha(0.8)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "threshold_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Create simple consistency score visualization")
    parser.add_argument("analysis_dir", help="Directory containing consistency analysis results")
    parser.add_argument("-o", "--output", default=".", help="Output directory for plots")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Consistency threshold to highlight (default: 0.8)"
    )
    parser.add_argument("--no-comparison", action="store_true", help="Skip threshold comparison plot")
    parser.add_argument(
        "--with-hits-only", action="store_true", help="Also create version with only genotypes that have >0 consistency"
    )

    args = parser.parse_args()

    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        return 1

    # Load consistency data
    consistency_file = os.path.join(args.analysis_dir, "enhanced_consistency_scores.csv")
    if not os.path.exists(consistency_file):
        print(f"❌ Consistency scores file not found: {consistency_file}")
        return 1

    consistency_df = pd.read_csv(consistency_file)
    print(f"📊 Loaded consistency data: {len(consistency_df)} genotypes")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"🎯 Output directory: {args.output}")

    # Load nickname mapping and brain region colors
    print(f"\n📋 Loading nickname mapping and brain region colors...")
    nickname_mapping, simplified_to_region = load_nickname_mapping()
    color_dict = get_brain_region_colors()

    try:
        created_plots = []

        # Create main consistency plot (all genotypes)
        print(f"\n🎨 Creating plot with ALL genotypes...")
        main_plot = create_simple_consistency_plot(
            consistency_df,
            output_dir=args.output,
            threshold=args.threshold,
            suffix="_all_genotypes",
            nickname_mapping=nickname_mapping,
            simplified_to_region=simplified_to_region,
            color_dict=color_dict,
        )
        created_plots.append(("All genotypes", main_plot))

        # Create with-hits-only plot if requested or if there are many genotypes
        with_hits_df = consistency_df[consistency_df["Overall_Consistency"] > 0]

        if args.with_hits_only or len(consistency_df) > 50:  # Auto-create if too many genotypes
            print(f"\n🎨 Creating plot with genotypes that have HITS (>0% consistency)...")
            print(f"   Filtering {len(consistency_df)} → {len(with_hits_df)} genotypes")

            if len(with_hits_df) > 0:
                hits_plot = create_simple_consistency_plot(
                    with_hits_df,
                    output_dir=args.output,
                    threshold=args.threshold,
                    suffix="_with_hits_only",
                    nickname_mapping=nickname_mapping,
                    simplified_to_region=simplified_to_region,
                    color_dict=color_dict,
                )
                created_plots.append(("With hits only", hits_plot))
            else:
                print(f"   ⚠️  No genotypes with >0% consistency - skipping with-hits-only plot")

        # Create threshold comparison plot
        comparison_plot = None
        if not args.no_comparison:
            comparison_plot = create_threshold_comparison_plot(consistency_df, output_dir=args.output)
            if comparison_plot:
                created_plots.append(("Threshold comparison", comparison_plot))

        # Report results
        print(f"\n💾 PLOTS CREATED:")
        for plot_name, plot_file in created_plots:
            print(f"   🎨 {plot_name}: {plot_file}")

        print(f"\n✅ SIMPLE CONSISTENCY VISUALIZATION COMPLETE!")
        return 0

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
