#!/usr/bin/env python3
"""
LAL-2 vs Empty-Split Long-Lasting Pause Analysis

This script focuses on comparing LAL-2 (experimental) vs Empty-Split (control)
to capture the phenotype of more/longer lasting pauses in LAL-2 flies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
from pathlib import Path


def parse_raw_pauses(raw_pauses_str):
    """Parse the string representation of raw pauses back into a list of tuples."""
    if pd.isna(raw_pauses_str) or raw_pauses_str == "[]":
        return []

    try:
        # Clean up the string - remove numpy float64 wrappers
        cleaned = re.sub(r"np\.float64\((.*?)\)", r"\1", str(raw_pauses_str))
        # Parse as literal
        pauses = ast.literal_eval(cleaned)
        return pauses
    except Exception as e:
        print(f"Error parsing pauses: {e}")
        return []


def load_and_filter_data():
    """Load dataset and extract LAL-2 and Empty-Split groups"""
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250918_16_summary_TNT_screen_Data/summary/pooled_summary.feather"
    df = pd.read_feather(dataset_path)

    # Filter for our two genotypes of interest
    lal2_data = df[df["Nickname"].str.contains("LAL-2", na=False)].copy()
    empty_split_data = df[df["Nickname"] == "Empty-Split"].copy()

    print(f"LAL-2 experiments: {len(lal2_data)}")
    print(f"Empty-Split experiments: {len(empty_split_data)}")

    # Parse raw_pauses for both groups
    lal2_data["parsed_pauses"] = lal2_data["raw_pauses"].apply(parse_raw_pauses)
    empty_split_data["parsed_pauses"] = empty_split_data["raw_pauses"].apply(parse_raw_pauses)

    return lal2_data, empty_split_data


def analyze_long_pause_phenotype(lal2_data, empty_split_data, thresholds=[1, 2, 3, 5, 10]):
    """
    Analyze long-lasting pause phenotype across different duration thresholds
    """
    results = []

    for threshold in thresholds:
        print(f"\nAnalyzing pauses >= {threshold}s...")

        # Analyze LAL-2
        lal2_metrics = []
        for _, row in lal2_data.iterrows():
            pauses = row["parsed_pauses"]
            if pauses:
                # Filter for long pauses
                long_pauses = [p for p in pauses if p[2] >= threshold]  # p[2] is duration

                count_long = len(long_pauses)
                total_duration_long = sum(p[2] for p in long_pauses)
                mean_duration_long = np.mean([p[2] for p in long_pauses]) if long_pauses else 0
                max_duration_long = max([p[2] for p in long_pauses]) if long_pauses else 0

                # Proportion of time in long pauses vs all pauses
                total_all_pauses = sum(p[2] for p in pauses)
                prop_time_long = total_duration_long / total_all_pauses if total_all_pauses > 0 else 0

                lal2_metrics.append(
                    {
                        "count_long": count_long,
                        "total_duration_long": total_duration_long,
                        "mean_duration_long": mean_duration_long,
                        "max_duration_long": max_duration_long,
                        "prop_time_long": prop_time_long,
                    }
                )

        # Analyze Empty-Split
        empty_metrics = []
        for _, row in empty_split_data.iterrows():
            pauses = row["parsed_pauses"]
            if pauses:
                # Filter for long pauses
                long_pauses = [p for p in pauses if p[2] >= threshold]

                count_long = len(long_pauses)
                total_duration_long = sum(p[2] for p in long_pauses)
                mean_duration_long = np.mean([p[2] for p in long_pauses]) if long_pauses else 0
                max_duration_long = max([p[2] for p in long_pauses]) if long_pauses else 0

                # Proportion of time in long pauses vs all pauses
                total_all_pauses = sum(p[2] for p in pauses)
                prop_time_long = total_duration_long / total_all_pauses if total_all_pauses > 0 else 0

                empty_metrics.append(
                    {
                        "count_long": count_long,
                        "total_duration_long": total_duration_long,
                        "mean_duration_long": mean_duration_long,
                        "max_duration_long": max_duration_long,
                        "prop_time_long": prop_time_long,
                    }
                )

        # Convert to DataFrames
        lal2_df = pd.DataFrame(lal2_metrics)
        empty_df = pd.DataFrame(empty_metrics)

        if len(lal2_df) > 0 and len(empty_df) > 0:
            # Calculate statistics
            lal2_stats = {
                "genotype": "LAL-2",
                "threshold": threshold,
                "n_experiments": len(lal2_df),
                "count_long_mean": lal2_df["count_long"].mean(),
                "count_long_std": lal2_df["count_long"].std(),
                "total_duration_long_mean": lal2_df["total_duration_long"].mean(),
                "total_duration_long_std": lal2_df["total_duration_long"].std(),
                "mean_duration_long_mean": lal2_df["mean_duration_long"].mean(),
                "mean_duration_long_std": lal2_df["mean_duration_long"].std(),
                "max_duration_long_mean": lal2_df["max_duration_long"].mean(),
                "max_duration_long_std": lal2_df["max_duration_long"].std(),
                "prop_time_long_mean": lal2_df["prop_time_long"].mean(),
                "prop_time_long_std": lal2_df["prop_time_long"].std(),
            }

            empty_stats = {
                "genotype": "Empty-Split",
                "threshold": threshold,
                "n_experiments": len(empty_df),
                "count_long_mean": empty_df["count_long"].mean(),
                "count_long_std": empty_df["count_long"].std(),
                "total_duration_long_mean": empty_df["total_duration_long"].mean(),
                "total_duration_long_std": empty_df["total_duration_long"].std(),
                "mean_duration_long_mean": empty_df["mean_duration_long"].mean(),
                "mean_duration_long_std": empty_df["mean_duration_long"].std(),
                "max_duration_long_mean": empty_df["max_duration_long"].mean(),
                "max_duration_long_std": empty_df["max_duration_long"].std(),
                "prop_time_long_mean": empty_df["prop_time_long"].mean(),
                "prop_time_long_std": empty_df["prop_time_long"].std(),
            }

            results.extend([lal2_stats, empty_stats])

            # Print comparison
            ratio_count = (
                lal2_stats["count_long_mean"] / empty_stats["count_long_mean"]
                if empty_stats["count_long_mean"] > 0
                else float("inf")
            )
            ratio_total = (
                lal2_stats["total_duration_long_mean"] / empty_stats["total_duration_long_mean"]
                if empty_stats["total_duration_long_mean"] > 0
                else float("inf")
            )
            ratio_prop = (
                lal2_stats["prop_time_long_mean"] / empty_stats["prop_time_long_mean"]
                if empty_stats["prop_time_long_mean"] > 0
                else float("inf")
            )

            print(f"  LAL-2 vs Empty-Split ratios:")
            print(f"    Count of long pauses: {ratio_count:.2f}x")
            print(f"    Total time in long pauses: {ratio_total:.2f}x")
            print(f"    Proportion of time in long pauses: {ratio_prop:.2f}x")

    return pd.DataFrame(results)


def create_comparison_plots(results_df):
    """Create comprehensive comparison plots"""

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("LAL-2 vs Empty-Split: Long-Lasting Pause Phenotype Analysis", fontsize=16, fontweight="bold")

    thresholds = results_df["threshold"].unique()
    lal2_results = results_df[results_df["genotype"] == "LAL-2"]
    empty_results = results_df[results_df["genotype"] == "Empty-Split"]

    # 1. Count of long pauses
    ax = axes[0, 0]
    ax.errorbar(
        lal2_results["threshold"],
        lal2_results["count_long_mean"],
        yerr=lal2_results["count_long_std"],
        label="LAL-2",
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.errorbar(
        empty_results["threshold"],
        empty_results["count_long_mean"],
        yerr=empty_results["count_long_std"],
        label="Empty-Split",
        marker="s",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("Count of Long Pauses")
    ax.set_title("A. Number of Long Pauses per Experiment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Total duration in long pauses
    ax = axes[0, 1]
    ax.errorbar(
        lal2_results["threshold"],
        lal2_results["total_duration_long_mean"],
        yerr=lal2_results["total_duration_long_std"],
        label="LAL-2",
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.errorbar(
        empty_results["threshold"],
        empty_results["total_duration_long_mean"],
        yerr=empty_results["total_duration_long_std"],
        label="Empty-Split",
        marker="s",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("Total Duration in Long Pauses (s)")
    ax.set_title("B. Total Time Spent in Long Pauses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Proportion of time in long pauses
    ax = axes[0, 2]
    ax.errorbar(
        lal2_results["threshold"],
        lal2_results["prop_time_long_mean"],
        yerr=lal2_results["prop_time_long_std"],
        label="LAL-2",
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.errorbar(
        empty_results["threshold"],
        empty_results["prop_time_long_mean"],
        yerr=empty_results["prop_time_long_std"],
        label="Empty-Split",
        marker="s",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("Proportion of Time in Long Pauses")
    ax.set_title("C. Fraction of Pause Time in Long Pauses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Ratios - Count
    ax = axes[1, 0]
    ratios_count = []
    for threshold in thresholds:
        lal2_val = lal2_results[lal2_results["threshold"] == threshold]["count_long_mean"].iloc[0]
        empty_val = empty_results[empty_results["threshold"] == threshold]["count_long_mean"].iloc[0]
        ratio = lal2_val / empty_val if empty_val > 0 else float("inf")
        ratios_count.append(ratio)

    ax.plot(thresholds, ratios_count, "ro-", linewidth=2, markersize=6)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("LAL-2 / Empty-Split Ratio")
    ax.set_title("D. Ratio: Count of Long Pauses")
    ax.grid(True, alpha=0.3)

    # 5. Ratios - Total Duration
    ax = axes[1, 1]
    ratios_duration = []
    for threshold in thresholds:
        lal2_val = lal2_results[lal2_results["threshold"] == threshold]["total_duration_long_mean"].iloc[0]
        empty_val = empty_results[empty_results["threshold"] == threshold]["total_duration_long_mean"].iloc[0]
        ratio = lal2_val / empty_val if empty_val > 0 else float("inf")
        ratios_duration.append(ratio)

    ax.plot(thresholds, ratios_duration, "go-", linewidth=2, markersize=6)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("LAL-2 / Empty-Split Ratio")
    ax.set_title("E. Ratio: Total Duration in Long Pauses")
    ax.grid(True, alpha=0.3)

    # 6. Ratios - Proportion
    ax = axes[1, 2]
    ratios_prop = []
    for threshold in thresholds:
        lal2_val = lal2_results[lal2_results["threshold"] == threshold]["prop_time_long_mean"].iloc[0]
        empty_val = empty_results[empty_results["threshold"] == threshold]["prop_time_long_mean"].iloc[0]
        ratio = lal2_val / empty_val if empty_val > 0 else float("inf")
        ratios_prop.append(ratio)

    ax.plot(thresholds, ratios_prop, "bo-", linewidth=2, markersize=6)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Pause Duration Threshold (s)")
    ax.set_ylabel("LAL-2 / Empty-Split Ratio")
    ax.set_title("F. Ratio: Proportion of Time in Long Pauses")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = "/home/matthias/ballpushing_utils/lal2_vs_emptysplit_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")

    return fig


def create_summary_table(results_df):
    """Create a summary table of key metrics"""

    print("\n" + "=" * 80)
    print("LAL-2 vs Empty-Split: Long-Lasting Pause Phenotype Summary")
    print("=" * 80)

    thresholds = sorted(results_df["threshold"].unique())

    for threshold in thresholds:
        lal2_row = results_df[(results_df["genotype"] == "LAL-2") & (results_df["threshold"] == threshold)].iloc[0]
        empty_row = results_df[(results_df["genotype"] == "Empty-Split") & (results_df["threshold"] == threshold)].iloc[
            0
        ]

        print(f"\nPause Duration Threshold: ≥ {threshold}s")
        print("-" * 40)

        # Count of long pauses
        ratio_count = (
            lal2_row["count_long_mean"] / empty_row["count_long_mean"]
            if empty_row["count_long_mean"] > 0
            else float("inf")
        )
        print(f"Count of Long Pauses:")
        print(f"  LAL-2: {lal2_row['count_long_mean']:.1f} ± {lal2_row['count_long_std']:.1f}")
        print(f"  Empty-Split: {empty_row['count_long_mean']:.1f} ± {empty_row['count_long_std']:.1f}")
        print(f"  Ratio (LAL-2/Empty-Split): {ratio_count:.2f}x")

        # Total duration
        ratio_duration = (
            lal2_row["total_duration_long_mean"] / empty_row["total_duration_long_mean"]
            if empty_row["total_duration_long_mean"] > 0
            else float("inf")
        )
        print(f"Total Time in Long Pauses (s):")
        print(f"  LAL-2: {lal2_row['total_duration_long_mean']:.1f} ± {lal2_row['total_duration_long_std']:.1f}")
        print(
            f"  Empty-Split: {empty_row['total_duration_long_mean']:.1f} ± {empty_row['total_duration_long_std']:.1f}"
        )
        print(f"  Ratio (LAL-2/Empty-Split): {ratio_duration:.2f}x")

        # Proportion
        ratio_prop = (
            lal2_row["prop_time_long_mean"] / empty_row["prop_time_long_mean"]
            if empty_row["prop_time_long_mean"] > 0
            else float("inf")
        )
        print(f"Proportion of Time in Long Pauses:")
        print(f"  LAL-2: {lal2_row['prop_time_long_mean']:.3f} ± {lal2_row['prop_time_long_std']:.3f}")
        print(f"  Empty-Split: {empty_row['prop_time_long_mean']:.3f} ± {empty_row['prop_time_long_std']:.3f}")
        print(f"  Ratio (LAL-2/Empty-Split): {ratio_prop:.2f}x")


def main():
    print("Loading data and filtering for LAL-2 and Empty-Split...")
    lal2_data, empty_split_data = load_and_filter_data()

    print("\nAnalyzing long-lasting pause phenotype...")
    results_df = analyze_long_pause_phenotype(lal2_data, empty_split_data)

    print("\nCreating comparison plots...")
    fig = create_comparison_plots(results_df)

    print("\nGenerating summary table...")
    create_summary_table(results_df)

    print("\nAnalysis complete!")
    print(f"Results saved to: /home/matthias/ballpushing_utils/lal2_vs_emptysplit_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
