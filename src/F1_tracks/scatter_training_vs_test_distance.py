#!/usr/bin/env python3
"""
Script to create a scatter plot of training vs test ball distance moved for Pretraining == "y" flies.
X-axis: distance_moved for ball_condition "training"
Y-axis: distance_moved for ball_condition "test"
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    """Main function to create the training vs test scatter plot."""

    # Dataset path
    dataset_path = (
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251001_13_summary_F1_New_Data/summary/pooled_summary.feather"
    )

    # Load dataset
    try:
        df = pd.read_feather(dataset_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Remove the data for Date 250904

    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")
    else:
        print("Column 'Date' not found in dataset. Skipping date-based filtering.")

    # Print available columns to help identify the correct ones
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Try to identify the correct column names
    distance_col = None
    pretraining_col = None
    ball_condition_col = None

    # Look for distance column - prioritize ball-specific distance over fly distance
    for col in df.columns:
        if col.lower() == "distance_moved":  # Exact match for ball-specific distance
            distance_col = col
            break
        elif "distance" in col.lower() and "move" in col.lower() and "fly" not in col.lower():
            distance_col = col
            break

    # If still no distance column found, try any distance column as fallback
    if distance_col is None:
        for col in df.columns:
            if "distance" in col.lower() and ("move" in col.lower() or "total" in col.lower()):
                distance_col = col
                break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower() or "training" in col.lower():
            pretraining_col = col
            break

    # Look for ball condition column - prioritize ball_condition over ball_identity
    if "ball_condition" in df.columns:
        ball_condition_col = "ball_condition"
    elif "ball_identity" in df.columns:
        ball_condition_col = "ball_identity"
    else:
        for col in df.columns:
            if "ball" in col.lower() and ("condition" in col.lower() or "identity" in col.lower()):
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

    # Add fly identifier column if it doesn't exist
    fly_id_col = "fly"
    if fly_id_col not in df.columns:
        # Look for fly identifier column
        fly_cols = [col for col in df.columns if "fly" in col.lower()]
        if fly_cols:
            fly_id_col = fly_cols[0]
            print(f"Using '{fly_id_col}' as fly identifier column")
        else:
            print("Warning: No fly identifier column found. Using row index.")
            df["fly"] = df.index
            fly_id_col = "fly"

    # Clean the data
    df_clean = df[[distance_col, pretraining_col, ball_condition_col, fly_id_col]].dropna()
    print(f"Data shape after removing NaNs: {df_clean.shape}")

    # Debug: Show some sample data to verify the distance values are different
    sample_data = df_clean[[distance_col, ball_condition_col, fly_id_col]].head(10)
    print(f"\nSample data to verify distance values:")
    print(sample_data)

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_clean[pretraining_col].unique()}")
    print(f"Unique {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")

    # Filter for Pretraining == "y" flies
    pretrained_flies = df_clean[df_clean[pretraining_col] == "y"]
    print(f"Number of pretrained flies: {len(pretrained_flies[fly_id_col].unique())}")

    if pretrained_flies.empty:
        print("No flies with Pretraining == 'y' found. Available pretraining values:")
        print(df_clean[pretraining_col].value_counts())
        return

    # Pivot the data to get training and test distances for each fly
    pivot_data = pretrained_flies.pivot_table(
        index=fly_id_col,
        columns=ball_condition_col,
        values=distance_col,
        aggfunc="first",  # In case there are duplicates, take the first value
    )

    print(f"Pivot table shape: {pivot_data.shape}")
    print(f"Pivot table columns: {pivot_data.columns.tolist()}")

    # Check if we have both training and test conditions
    if "training" not in pivot_data.columns or "test" not in pivot_data.columns:
        print("Missing 'training' or 'test' ball conditions in the data.")
        print(f"Available ball conditions: {pivot_data.columns.tolist()}")
        return

    # Remove flies that don't have both training and test data
    complete_data = pivot_data.dropna(subset=["training", "test"])
    print(f"Number of flies with both training and test data: {len(complete_data)}")

    if complete_data.empty:
        print("No flies have both training and test data.")
        return

    # Create the scatter plot
    plt.figure(figsize=(10, 8))

    # Set style
    sns.set_style("whitegrid")

    # Create scatter plot
    x = complete_data["training"]
    y = complete_data["test"]

    plt.scatter(x, y, alpha=0.7, s=60, edgecolors="black", linewidth=0.5)

    # Calculate correlation and trend line
    correlation_coef, p_value = stats.pearsonr(x, y)

    # Add trend line (best fit)
    z = np.polyfit(x, y, 1)
    fit_line = np.poly1d(z)
    plt.plot(
        x,
        fit_line(x),
        "b-",
        alpha=0.8,
        linewidth=2,
        label=f"Linear fit (r = {correlation_coef:.3f}, p = {p_value:.3f})",
    )

    # Add reference line showing expected relationship if test ball has systematic offset
    # Calculate the ratio of means to show expected scaling
    mean_ratio = y.mean() / x.mean()
    x_range = np.linspace(x.min(), x.max(), 100)
    expected_y = x_range * mean_ratio
    plt.plot(
        x_range, expected_y, "r--", alpha=0.6, linewidth=2, label=f"Expected scaling (test/training = {mean_ratio:.2f})"
    )

    # Optionally add y=x line for absolute reference (but label it appropriately)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "gray",
        alpha=0.4,
        linewidth=1,
        linestyle=":",
        label="Equal performance (y = x)",
    )

    # Formatting
    plt.xlabel(f"Distance Moved - Training Ball", fontsize=12)
    plt.ylabel(f"Distance Moved - Test Ball", fontsize=12)
    plt.title("Training vs Test Ball Performance\n(Pretrained Flies Only)", fontsize=14, fontweight="bold")

    # Add correlation text box with better interpretation
    slope, intercept = z
    # Note: correlation_coef should be a number at runtime despite type checker warnings
    r_squared = correlation_coef * correlation_coef  # type: ignore
    textstr = (
        f"Pearson r = {correlation_coef:.3f}\n"
        f"R² = {r_squared:.3f}\n"
        f"p-value = {p_value:.3f}\n"
        f"Slope = {slope:.3f}\n"
        f"n = {len(complete_data)} flies"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set axis limits to match the actual data range with some padding
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_padding = x_range * 0.05  # 5% padding
    y_padding = y_range * 0.05  # 5% padding

    plt.xlim(x.min() - x_padding, x.max() + x_padding)
    plt.ylim(y.min() - y_padding, y.max() + y_padding)

    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "training_vs_test_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Training ball - Mean: {x.mean():.3f}, Std: {x.std():.3f}")
    print(f"Test ball - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    print(f"Correlation: {correlation_coef:.3f} (p = {p_value:.3f})")
    print(f"Mean ratio (test/training): {mean_ratio:.3f}")

    # Additional statistics
    print(f"\nPerformance comparison:")
    print(f"Flies performing better on test than training: {sum(y > x)} ({100*sum(y > x)/len(complete_data):.1f}%)")
    print(f"Flies performing better on training than test: {sum(x > y)} ({100*sum(x > y)/len(complete_data):.1f}%)")
    print(
        f"Flies with similar performance (±5% difference): {sum(abs(y - x) <= 0.05 * x)} ({100*sum(abs(y - x) <= 0.05 * x)/len(complete_data):.1f}%)"
    )

    return complete_data


if __name__ == "__main__":
    data = main()
