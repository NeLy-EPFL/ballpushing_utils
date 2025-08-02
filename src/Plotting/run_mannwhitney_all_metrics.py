#!/usr/bin/env python3
"""
Script to generate Mann-Whitney U test jitterboxplots for all metrics.

This script generates comprehensive Mann-Whitney U test visualizations with:
- Significance-based sorting (significantly increased/neutral/significantly decreased)
- Secondary sorting by median within each significance group
- Color-coded backgrounds for easy interpretation
- Statistical significance annotations (*, **, ***)
- External legend for improved readability
- Proper spacing between boxplots

Usage:
    python run_mannwhitney_all_metrics.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from Plotting.All_metrics import generate_jitterboxplots_with_mannwhitney
from Plotting.Config import color_dict
import pandas as pd


def load_and_clean_dataset():
    """Load and clean the dataset following the same process as All_metrics.py"""
    # Load the dataset
    dataset = pd.read_feather(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
    )

    # Convert boolean columns to int
    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    # Clean NA values following the same logic as All_metrics.py
    dataset["max_event"].fillna(-1, inplace=True)
    dataset["first_significant_event"].fillna(-1, inplace=True)
    dataset["major_event"].fillna(-1, inplace=True)
    dataset["final_event"].fillna(-1, inplace=True)

    # Fill time columns with 3600
    dataset["max_event_time"].fillna(3600, inplace=True)
    dataset["first_significant_event_time"].fillna(3600, inplace=True)
    dataset["major_event_time"].fillna(3600, inplace=True)
    dataset["final_event_time"].fillna(3600, inplace=True)

    # Drop problematic columns
    columns_to_drop = ["insight_effect", "insight_effect_log", "exit_time"]
    columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    if columns_to_drop:
        dataset.drop(columns=columns_to_drop, inplace=True)

    # Fill remaining NA values with 0
    dataset["pulling_ratio"].fillna(0, inplace=True)
    dataset["avg_displacement_after_success"].fillna(0, inplace=True)
    dataset["avg_displacement_after_failure"].fillna(0, inplace=True)
    dataset["influence_ratio"].fillna(0, inplace=True)

    return dataset


def main():
    """Main function to run Mann-Whitney plots for all metrics."""
    print("Starting Mann-Whitney U test analysis for all metrics...")

    # Define output directories
    base_output_dir = Path(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Plots/Summary_metrics/All_metrics_bysplit_BS_CI_99"
    )

    # Load the dataset
    print("Loading dataset...")
    dataset = load_and_clean_dataset()

    # Get all metric columns (excluding metadata)
    metadata_cols = [
        "index",
        "fly",
        "flypath",
        "experiment",
        "Nickname",
        "Brain region",
        "Simplified Nickname",
        "Split",
        "Date",
        "Genotype",
        "Period",
        "FeedingState",
        "Orientation",
        "Light",
        "Crossing",
    ]

    all_cols = dataset.columns.tolist()
    metric_cols = [col for col in all_cols if col not in metadata_cols]

    print(f"Found {len(metric_cols)} metrics to analyze:")
    for i, metric in enumerate(metric_cols, 1):
        print(f"  {i:2d}. {metric}")

    # Clean the dataset (already done in load_and_clean_dataset)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Unique genotypes: {len(dataset['Nickname'].unique())}")

    # Generate plots for each split condition
    conditions = [
        ("Split lines", lambda df: df[df["Split"] == "Split"], "mannwhitney_split"),
        ("No Split lines", lambda df: df[df["Split"] == "No Split"], "mannwhitney_nosplit"),
        ("Mutants", lambda df: df[~df["Split"].isin(["Split", "No Split"])], "mannwhitney_mutant"),
    ]

    total_stats = []

    for condition_name, filter_func, output_subdir in conditions:
        print(f"\n{'='*60}")
        print(f"Processing {condition_name}...")
        print("=" * 60)

        # Filter dataset for this condition
        filtered_data = filter_func(dataset)
        print(f"Filtered dataset shape: {filtered_data.shape}")
        print(f"Unique genotypes in {condition_name}: {len(filtered_data['Nickname'].unique())}")

        if len(filtered_data) == 0:
            print(f"No data found for {condition_name}, skipping...")
            continue

        # Generate plots
        try:
            stats_df = generate_jitterboxplots_with_mannwhitney(
                data=filtered_data,
                metrics=metric_cols,
                y="Nickname",
                hue=None,
                palette=color_dict,
                output_dir=str(base_output_dir / output_subdir),
            )

            if stats_df is not None:
                stats_df["condition"] = condition_name
                total_stats.append(stats_df)

                # Print summary for this condition
                sig_count = len(stats_df[stats_df["significant"] == True])
                total_count = len(stats_df)
                print(f"\n{condition_name} Results:")
                print(f"  Total comparisons: {total_count}")
                print(f"  Significant results: {sig_count}")
                print(f"  Significance rate: {sig_count/total_count*100:.1f}%")

                # Breakdown by significance level
                for level in ["***", "**", "*"]:
                    count = len(stats_df[stats_df["sig_level"] == level])
                    if count > 0:
                        print(f"  {level}: {count} results")

        except Exception as e:
            print(f"Error processing {condition_name}: {str(e)}")
            continue

    # Save combined statistics
    if total_stats:
        print(f"\n{'='*60}")
        print("Saving combined statistics...")
        print("=" * 60)

        combined_stats = pd.concat(total_stats, ignore_index=True)
        output_file = base_output_dir / "all_mannwhitney_statistics.csv"
        combined_stats.to_csv(output_file, index=False)

        print(f"\nAll Mann-Whitney statistics saved to: {output_file}")

        # Overall summary
        total_significant = len(combined_stats[combined_stats["significant"] == True])
        total_comparisons = len(combined_stats)

        print(f"\nOVERALL SUMMARY:")
        print(f"Total comparisons: {total_comparisons}")
        print(f"Total significant results: {total_significant}")
        print(f"Overall significance rate: {total_significant/total_comparisons*100:.1f}%")

        print(f"\nSignificance breakdown:")
        for level in ["***", "**", "*", "ns"]:
            count = len(combined_stats[combined_stats["sig_level"] == level])
            print(f"  {level}: {count} results")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
