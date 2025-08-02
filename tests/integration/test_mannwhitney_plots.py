#!/usr/bin/env python3
"""
Test script for the new Mann-Whitney U test jitterboxplot function.
This script tests the function on a small subset of metrics to verify it works correctly.
"""

import sys

sys.path.append("/home/matthias/ballpushing_utils/src/Plotting")
sys.path.append("/home/matthias/ballpushing_utils/src")

from All_metrics import generate_jitterboxplots_with_mannwhitney
import pandas as pd
import Config
from pathlib import Path

# Load the dataset
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# Clean up the dataset
dataset = Config.cleanup_data(dataset)

# Convert boolean metrics to numeric
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)

# Handle missing values as in the original script
dataset["max_event"].fillna(-1, inplace=True)
dataset["first_significant_event"].fillna(-1, inplace=True)
dataset["major_event"].fillna(-1, inplace=True)
dataset["final_event"].fillna(-1, inplace=True)

dataset["max_event_time"].fillna(3600, inplace=True)
dataset["first_significant_event_time"].fillna(3600, inplace=True)
dataset["major_event_time"].fillna(3600, inplace=True)
dataset["final_event_time"].fillna(3600, inplace=True)

dataset.drop(columns=["insight_effect", "insight_effect_log", "exit_time"], inplace=True, errors="ignore")

dataset["pulling_ratio"].fillna(0, inplace=True)
dataset["avg_displacement_after_success"].fillna(0, inplace=True)
dataset["avg_displacement_after_failure"].fillna(0, inplace=True)
dataset["influence_ratio"].fillna(0, inplace=True)

# Filter datasets
SplitLines = dataset[dataset["Split"] == "y"]

# Test metrics (just a few to start)
test_metrics = ["nb_events", "max_event", "interaction_proportion", "chamber_ratio", "pulling_ratio"]

# Output directory for test
test_output_dir = Path("/home/matthias/ballpushing_utils/test_mannwhitney_output")

print("Testing Mann-Whitney U jitterboxplot function...")
print(f"Dataset shape: {SplitLines.shape}")
print(f"Test metrics: {test_metrics}")
print(f"Output directory: {test_output_dir}")

try:
    # Run the function
    stats_df = generate_jitterboxplots_with_mannwhitney(
        data=SplitLines,
        metrics=test_metrics,
        y="Nickname",
        split_col="Split",
        hue="Brain region",
        palette=Config.color_dict,
        figsize=(12, 15),
        output_dir=test_output_dir,
    )

    print("Success! Function completed without errors.")
    print(f"Statistics shape: {stats_df.shape}")
    print(f"Significant results: {stats_df['significant'].sum()}")
    print("\nFirst few results:")
    print(stats_df.head(10))

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
