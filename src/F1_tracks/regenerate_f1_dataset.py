#!/usr/bin/env python3
"""
Script to regenerate F1_coordinates dataset with normalized percentage-based adjusted time.
"""

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from Ballpushing_utils import Config, Dataset
from pathlib import Path
import pandas as pd


def main():
    """Regenerate F1_coordinates dataset with normalized time."""

    # Load experiment configuration
    config = Config()
    config.experiment_type = "F1"

    # Path to the F1 experiment YAML
    yaml_path = "/mnt/upramdya_data/MD/Experimental_data/experiments_yaml/F1_New.yaml"

    if not Path(yaml_path).exists():
        print(f"Error: YAML file not found at {yaml_path}")
        return

    print("Loading F1 experiments from YAML...")

    try:
        # Create dataset with F1_coordinates metrics
        print("Generating F1_coordinates dataset with normalized time...")
        dataset = Dataset(yaml_path, dataset_type="F1_coordinates")

        if dataset.data is None or dataset.data.empty:
            print("Error: Generated dataset is empty")
            return

        print(f"Dataset generated successfully!")
        print(f"Shape: {dataset.data.shape}")
        print(
            f"Adjusted time range: {dataset.data['adjusted_time'].min():.2f} to {dataset.data['adjusted_time'].max():.2f}"
        )

        # Check a few sample rows
        print("\nSample data:")
        print(dataset.data[["adjusted_time", "test_ball", "fly"]].head(10))

        # Save to a new location for testing
        output_path = Path("/home/matthias/ballpushing_utils/src/F1_tracks/test_normalized_f1_coordinates.feather")
        dataset.data.to_feather(output_path)
        print(f"\nDataset saved to: {output_path}")

        # Quick validation
        print(f"\nValidation:")
        print(f"Unique flies: {dataset.data['fly'].nunique()}")
        print(f"Adjusted time stats:")
        print(dataset.data["adjusted_time"].describe())

    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
