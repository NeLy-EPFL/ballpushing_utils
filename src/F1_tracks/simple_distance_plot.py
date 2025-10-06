#!/usr/bin/env python3
"""
Simple script to plot distance_moved by Pretraining and ball_condition for F1 dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def main():
    """Main function to create the distance moved plot."""

    # Dataset path
    dataset_path = (
        "/Volumes/upramdya/data/MD/F1_Tracks/Datasets/251001_12_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Load dataset
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Try to identify the correct column names
    distance_col = None
    pretraining_col = None
    ball_condition_col = None

    # Look for distance column
    for col in df.columns:
        if "distance" in col.lower() and ("move" in col.lower() or "total" in col.lower()):
            distance_col = col
            break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower() or "training" in col.lower():
            pretraining_col = col
            break

    # Look for ball condition column
    for col in df.columns:
        if "ball" in col.lower() and "condition" in col.lower():
            ball_condition_col = col
            break

    # If exact matches not found, use manual mapping based on your dataset structure
    if distance_col is None:
        # Look for any distance column
        distance_cols = [col for col in df.columns if "distance" in col.lower()]
        if distance_cols:
            distance_col = distance_cols[0]
            print(f"Using '{distance_col}' as distance column")

    if pretraining_col is None:
        # Look for any training-related column
        train_cols = [col for col in df.columns if "train" in col.lower()]
        if train_cols:
            pretraining_col = train_cols[0]
            print(f"Using '{pretraining_col}' as pretraining column")

    if ball_condition_col is None:
        # Check if there's a direct 'ball_condition' column
        if "ball_condition" in df.columns:
            ball_condition_col = "ball_condition"
        else:
            # Look for any ball-related column
            ball_cols = [col for col in df.columns if "ball" in col.lower()]
            if ball_cols:
                ball_condition_col = ball_cols[0]
                print(f"Using '{ball_condition_col}' as ball condition column")

    # Check if we found all required columns
    required_columns = {"distance": distance_col, "pretraining": pretraining_col, "ball_condition": ball_condition_col}

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  Distance: {distance_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Ball condition: {ball_condition_col}")

    # Clean the data
    df_clean = df[[distance_col, pretraining_col, ball_condition_col]].dropna()
    print(f"Data shape after removing NaNs: {df_clean.shape}")

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_clean[pretraining_col].unique()}")
    print(f"Unique {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use seaborn for a nice boxplot
    sns.set_style("whitegrid")
    sns.boxplot(data=df_clean, x=pretraining_col, y=distance_col, hue=ball_condition_col)

    plt.title("Distance Moved by Pretraining and Ball Condition", fontsize=14, fontweight="bold")
    plt.ylabel("Distance Moved", fontsize=12)
    plt.xlabel("Pretraining", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Ball Condition")
    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "distance_moved_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    summary = (
        df_clean.groupby([pretraining_col, ball_condition_col])[distance_col]
        .agg(["count", "mean", "std", "median"])
        .round(3)
    )
    print(summary)

    plt.show()


if __name__ == "__main__":
    main()
