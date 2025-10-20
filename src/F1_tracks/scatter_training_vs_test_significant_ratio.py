#!/usr/bin/env python3
"""
Script to create a scatter plot of training vs test ball significant_ratio for Pretraining == "y" flies.
X-axis: significant_ratio for ball_condition "training"
Y-axis: significant_ratio for ball_condition "test"
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
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather"
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
    significant_ratio_col = None
    pretraining_col = None
    ball_condition_col = None
    f1_condition_col = None

    # Look for significant_ratio column
    if "significant_ratio" in df.columns:
        significant_ratio_col = "significant_ratio"
    else:
        for col in df.columns:
            if "significant" in col.lower() and "ratio" in col.lower():
                significant_ratio_col = col
                break

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower():
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

    # Look for F1_condition column
    if "F1_condition" in df.columns:
        f1_condition_col = "F1_condition"
    else:
        for col in df.columns:
            if "f1" in col.lower() and "condition" in col.lower():
                f1_condition_col = col
                break

    # If exact matches not found, use manual mapping based on your dataset structure
    if significant_ratio_col is None:
        ratio_cols = [col for col in df.columns if "ratio" in col.lower()]
        if ratio_cols:
            significant_ratio_col = ratio_cols[0]
            print(f"Using '{significant_ratio_col}' as significant ratio column")

    if pretraining_col is None:
        train_cols = [col for col in df.columns if "train" in col.lower()]
        if train_cols:
            pretraining_col = train_cols[0]
            print(f"Using '{pretraining_col}' as pretraining column")

    # Check if we found all required columns
    required_columns = {
        "significant_ratio": significant_ratio_col,
        "pretraining": pretraining_col,
        "ball_condition": ball_condition_col,
        "f1_condition": f1_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col is None]
    if missing:
        print(f"Could not find columns for: {missing}")
        print("Please check the column names in your dataset.")
        return

    print(f"Using columns:")
    print(f"  Significant ratio: {significant_ratio_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  Ball condition: {ball_condition_col}")
    print(f"  F1 condition: {f1_condition_col}")

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
    df_clean = df[[significant_ratio_col, pretraining_col, ball_condition_col, f1_condition_col, fly_id_col]].dropna()
    print(f"Data shape after removing NaNs: {df_clean.shape}")

    # Debug: Show some sample data to verify the ratio values are different
    sample_data = df_clean[[significant_ratio_col, ball_condition_col, f1_condition_col, fly_id_col]].head(10)
    print(f"\nSample data to verify significant ratio values:")
    print(sample_data)

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_clean[pretraining_col].unique()}")
    print(f"Unique {ball_condition_col} values: {df_clean[ball_condition_col].unique()}")
    print(f"Unique {f1_condition_col} values: {df_clean[f1_condition_col].unique()}")

    # Filter for Pretraining == "y" flies
    pretrained_flies = df_clean[df_clean[pretraining_col] == "y"]
    print(f"Number of pretrained flies: {len(pretrained_flies[fly_id_col].unique())}")

    if pretrained_flies.empty:
        print("No flies with Pretraining == 'y' found. Available pretraining values:")
        print(df_clean[pretraining_col].value_counts())
        return

    # Get unique F1_condition values for pretrained flies
    f1_conditions = sorted(pretrained_flies[f1_condition_col].unique())
    print(f"F1_condition values for pretrained flies: {f1_conditions}")

    # Filter to only include 'pretrained' and 'pretrained_unlocked' conditions
    conditions_to_include = ["pretrained", "pretrained_unlocked"]
    available_conditions = [cond for cond in conditions_to_include if cond in f1_conditions]

    if not available_conditions:
        print(f"No data found for conditions: {conditions_to_include}")
        print(f"Available F1_condition values: {f1_conditions}")
        return

    print(f"Analyzing F1_conditions: {available_conditions}")

    # Create subplots
    fig, axes = plt.subplots(1, len(available_conditions), figsize=(6 * len(available_conditions), 6))
    if len(available_conditions) == 1:
        axes = [axes]  # Make it a list for consistent indexing

    # Set style
    sns.set_style("whitegrid")

    all_results = {}

    for idx, condition in enumerate(available_conditions):
        ax = axes[idx]

        # Filter data for this F1_condition
        condition_data = pretrained_flies[pretrained_flies[f1_condition_col] == condition]
        print(f"\n{condition} - Number of flies: {len(condition_data[fly_id_col].unique())}")

        # Pivot the data to get training and test ratios for each fly
        pivot_data = condition_data.pivot_table(
            index=fly_id_col,
            columns=ball_condition_col,
            values=significant_ratio_col,
            aggfunc="first",  # In case there are duplicates, take the first value
        )

        print(f"{condition} - Pivot table shape: {pivot_data.shape}")
        print(f"{condition} - Pivot table columns: {pivot_data.columns.tolist()}")

        # Check if we have both training and test conditions
        if "training" not in pivot_data.columns or "test" not in pivot_data.columns:
            print(f"{condition} - Missing 'training' or 'test' ball conditions in the data.")
            print(f"{condition} - Available ball conditions: {pivot_data.columns.tolist()}")
            continue

        # Remove flies that don't have both training and test data
        complete_data = pivot_data.dropna(subset=["training", "test"])
        print(f"{condition} - Number of flies with both training and test data: {len(complete_data)}")

        if complete_data.empty:
            print(f"{condition} - No flies have both training and test data.")
            continue

        # Store results
        all_results[condition] = complete_data

        # Create scatter plot
        x = complete_data["training"]
        y = complete_data["test"]

        ax.scatter(x, y, alpha=0.7, s=60, edgecolors="black", linewidth=0.5)

        # Calculate correlation and trend line
        correlation_coef = float("nan")
        p_value = float("nan")
        normalized_correlation_coef = float("nan")
        normalized_p_value = float("nan")

        if len(x) > 1 and x.std() > 0 and y.std() > 0:
            correlation_coef, p_value = stats.pearsonr(x, y)

            # Normalized correlation: scale both training and test to [0,1] based on their ranges
            x_normalized = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
            y_normalized = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0

            # Calculate correlation on normalized data
            if x_normalized.std() > 0 and y_normalized.std() > 0:
                normalized_correlation_coef, normalized_p_value = stats.pearsonr(x_normalized, y_normalized)

            # Add trend line (best fit)
            z = np.polyfit(x, y, 1)
            fit_line = np.poly1d(z)
            ax.plot(
                x,
                fit_line(x),
                "b-",
                alpha=0.8,
                linewidth=2,
                label=f"Linear fit (r = {correlation_coef:.3f}, p = {p_value:.3f})",
            )

            # Add reference line showing expected relationship if test ball has systematic offset
            mean_ratio = y.mean() / x.mean() if x.mean() != 0 else 1
            x_range = np.linspace(x.min(), x.max(), 100)
            expected_y = x_range * mean_ratio
            ax.plot(
                x_range,
                expected_y,
                "r--",
                alpha=0.6,
                linewidth=2,
                label=f"Expected scaling (test/training = {mean_ratio:.2f})",
            )

            # Add y=x line for reference (since both are ratios, this makes sense)
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "gray",
                alpha=0.4,
                linewidth=1,
                linestyle=":",
                label="Equal performance (y = x)",
            )

            # Add correlation text box
            slope, intercept = z
            r_squared = correlation_coef * correlation_coef  # type: ignore
            textstr = (
                f"Raw: r = {correlation_coef:.3f} (p = {p_value:.3f})\n"
                f"Norm: r = {normalized_correlation_coef:.3f} (p = {normalized_p_value:.3f})\n"
                f"R² = {r_squared:.3f}\n"
                f"Slope = {slope:.3f}\n"
                f"n = {len(complete_data)} flies"
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)
        else:
            ax.text(
                0.05,
                0.95,
                f"n = {len(complete_data)} flies\n(insufficient variation for correlation)",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Formatting
        ax.set_xlabel(f"Significant Ratio - Training Ball", fontsize=11)
        ax.set_ylabel(f"Significant Ratio - Test Ball", fontsize=11)
        ax.set_title(f"{condition.replace('_', ' ').title()}", fontsize=12, fontweight="bold")

        # Set axis limits with padding - special handling for ratio data (0-1 range)
        if len(x) > 0:
            # For ratios, we want to show the full meaningful range
            x_min, x_max = max(0, x.min() - 0.05), min(1, x.max() + 0.05)
            y_min, y_max = max(0, y.min() - 0.05), min(1, y.max() + 0.05)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print summary statistics for this condition
        print(f"\n{condition} Summary Statistics:")
        print(f"Training ball - Mean: {x.mean():.3f}, Std: {x.std():.3f}")
        print(f"Test ball - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
        if not np.isnan(correlation_coef):
            print(f"Raw correlation: {correlation_coef:.3f} (p = {p_value:.3f})")
        if not np.isnan(normalized_correlation_coef):
            print(f"Normalized correlation: {normalized_correlation_coef:.3f} (p = {normalized_p_value:.3f})")
            print(f"Mean ratio (test/training): {mean_ratio:.3f}")

    # Main title
    fig.suptitle(
        "Training vs Test Ball Significant Ratio by F1_condition\n(Pretrained Flies Only)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save the plot
    output_path = Path(__file__).parent / "training_vs_test_significant_ratio_by_f1condition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Compare conditions if we have data for both
    if len(all_results) >= 2:
        conditions = list(all_results.keys())
        print(f"\n=== Comparison between {conditions[0]} and {conditions[1]} ===")

        for i, (cond1, cond2) in enumerate([(conditions[0], conditions[1])]):
            data1 = all_results[cond1]
            data2 = all_results[cond2]

            x1, y1 = data1["training"], data1["test"]
            x2, y2 = data2["training"], data2["test"]

            print(f"\nTraining performance:")
            print(f"  {cond1}: {x1.mean():.3f} ± {x1.std():.3f} (n={len(x1)})")
            print(f"  {cond2}: {x2.mean():.3f} ± {x2.std():.3f} (n={len(x2)})")

            # Statistical test for training performance
            if len(x1) > 1 and len(x2) > 1:
                try:
                    stat_mw, p_mw = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                    stat_t, p_t = stats.ttest_ind(x1, x2)
                    print(f"  Training difference: Mann-Whitney U p = {p_mw:.3f}, t-test p = {p_t:.3f}")
                except:
                    print(f"  Could not perform statistical tests for training data")

            print(f"\nTest performance:")
            print(f"  {cond1}: {y1.mean():.3f} ± {y1.std():.3f} (n={len(y1)})")
            print(f"  {cond2}: {y2.mean():.3f} ± {y2.std():.3f} (n={len(y2)})")

            # Statistical test for test performance
            if len(y1) > 1 and len(y2) > 1:
                try:
                    stat_mw, p_mw = stats.mannwhitneyu(y1, y2, alternative="two-sided")
                    stat_t, p_t = stats.ttest_ind(y1, y2)
                    print(f"  Test difference: Mann-Whitney U p = {p_mw:.3f}, t-test p = {p_t:.3f}")
                except:
                    print(f"  Could not perform statistical tests for test data")

    return all_results


if __name__ == "__main__":
    data = main()
