#!/usr/bin/env python3
"""
Script to plot F1 coordinates over adjusted time with two approaches:
1. Percentage-based normalization (0-100%) - normalizes each fly's time individually
2. Trimmed time approach - uses actual seconds but only up to max time all flies share

The script can optionally filter the F1 coordinates dataset to only include flies
that are present in the corresponding summary dataset for consistency.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import argparse


def filter_data_by_time_window(data, time_col, start_time=-10, end_time=30):
    """Filter data to a specific time window."""
    return data[(data[time_col] >= start_time) & (data[time_col] <= end_time)]


def create_line_plot(data, x_col, y_cols, hue_col, title, ax, color_mapping=None):
    """Create a line plot with binned data and confidence intervals."""

    # Use all the data - no time window filtering
    filtered_data = data

    if filtered_data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Create color mapping if not provided
    if color_mapping is None:
        unique_hues = filtered_data[hue_col].unique()
        colors = sns.color_palette("Set1", n_colors=len(unique_hues))
        color_mapping = dict(zip(unique_hues, colors))

    # Plot each y column
    for i, y_col in enumerate(y_cols):
        if y_col not in filtered_data.columns:
            print(f"Warning: Column '{y_col}' not found in data")
            continue

        # Plot for each hue value
        for hue_val in filtered_data[hue_col].unique():
            subset = filtered_data[filtered_data[hue_col] == hue_val]
            if subset.empty:
                continue

            # Skip training_ball for control condition (will be all NaN)
            if y_col == "training_ball" and subset[y_col].isnull().all():
                continue

            # Remove NaN values for this y_col
            subset_clean = subset.dropna(subset=[y_col])
            if subset_clean.empty:
                continue

            # Bin the data into 1-second intervals and get median per fly per second
            fly_col = "fly" if "fly" in subset_clean.columns else None
            if not fly_col:
                print("Warning: No 'fly' column found, cannot bin by individual flies")
                continue

            # Create time bins (0.1% intervals for percentage-based adjusted time)
            subset_clean = subset_clean.copy()
            subset_clean["time_bin"] = np.floor(subset_clean[x_col] / 0.1) * 0.1  # 0.1% bins

            # Get median value per fly per time bin
            binned_data = subset_clean.groupby([fly_col, "time_bin"])[y_col].median().reset_index()
            binned_data = binned_data.rename(columns={y_col: f"{y_col}_median"})

            # Now calculate mean and confidence intervals across flies for each time bin
            time_stats = (
                binned_data.groupby("time_bin")[f"{y_col}_median"].agg(["mean", "std", "count", "sem"]).reset_index()
            )

            # Calculate 95% confidence intervals
            from scipy import stats as scipy_stats

            confidence_level = 0.95
            alpha = 1 - confidence_level
            time_stats["ci"] = time_stats.apply(
                lambda row: scipy_stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0,
                axis=1,
            )

            # Plot settings
            linestyle = "-" if i == 0 else "--"
            label = f"{y_col.replace('_', ' ').title()} ({hue_val})"
            color = color_mapping.get(hue_val, "black")

            # Plot mean trajectory
            ax.plot(
                time_stats["time_bin"],
                time_stats["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=0.8,
                label=label,
            )

            # Add confidence intervals
            ax.fill_between(
                time_stats["time_bin"],
                time_stats["mean"] - time_stats["ci"],
                time_stats["mean"] + time_stats["ci"],
                color=color,
                alpha=0.2,
            )

    # Add vertical line at time 0
    ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time (0%)")

    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Adjusted Time (%)", fontsize=12)
    ax.set_ylabel("Distance (pixels)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add sample size annotation
    n_flies = len(filtered_data["fly"].unique()) if "fly" in filtered_data.columns else len(filtered_data)
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    return ax


def normalize_adjusted_time_to_percentage(df):
    """
    Normalize adjusted_time to 0-100% for each fly and resample to standard 0.1% intervals.

    Args:
        df: DataFrame with 'adjusted_time' and 'fly' columns

    Returns:
        DataFrame with normalized adjusted_time and standardized intervals
    """
    if "adjusted_time" not in df.columns or "fly" not in df.columns:
        print("Warning: Cannot normalize - missing 'adjusted_time' or 'fly' columns")
        return df

    print(f"Original adjusted_time range: {df['adjusted_time'].min():.2f} to {df['adjusted_time'].max():.2f} seconds")

    # Create standard time grid (0-100% in 0.1% steps)
    standard_time_grid = np.arange(0, 100.1, 0.1)

    normalized_data = []

    # Group by fly and normalize each fly's data separately
    for fly_id in df["fly"].unique():
        fly_data = df[df["fly"] == fly_id].copy()

        # Skip flies with no valid adjusted_time data
        if fly_data["adjusted_time"].isna().all():
            continue

        # Get valid adjusted time values for this fly
        valid_mask = ~fly_data["adjusted_time"].isna()
        if not valid_mask.any():
            continue

        valid_times = fly_data.loc[valid_mask, "adjusted_time"]
        min_time = valid_times.min()
        max_time = valid_times.max()

        # Skip if all times are the same or we have less than 2 points
        if max_time <= min_time or valid_mask.sum() < 2:
            continue

        # Normalize to 0-100% for this fly
        normalized_time = ((fly_data["adjusted_time"] - min_time) / (max_time - min_time)) * 100

        # Create DataFrame for this fly with standard time grid
        fly_normalized = pd.DataFrame({"adjusted_time": standard_time_grid})

        # Add fly identifier and other metadata columns
        for col in ["fly", "Pretraining", "F1_condition"]:
            if col in fly_data.columns:
                fly_normalized[col] = fly_data[col].iloc[0]  # Use first value for categorical data

        # Interpolate ball distances to standard time grid
        valid_indices = valid_mask
        for ball_col in ["training_ball", "test_ball"]:
            if ball_col in fly_data.columns:
                if valid_indices.sum() > 1:
                    # Interpolate to standard grid
                    fly_normalized[ball_col] = np.interp(
                        standard_time_grid, normalized_time[valid_indices], fly_data.loc[valid_indices, ball_col]
                    )
                else:
                    # Not enough points for interpolation
                    fly_normalized[ball_col] = np.nan

        # Add other metadata columns without interpolation
        for col in df.columns:
            if col not in fly_normalized.columns and col not in [
                "adjusted_time",
                "training_ball",
                "test_ball",
                "time",
                "frame",
            ]:
                fly_normalized[col] = fly_data[col].iloc[0]

        normalized_data.append(fly_normalized)

    if not normalized_data:
        print("Warning: No flies could be normalized")
        return df

    # Combine all normalized fly data
    result_df = pd.concat(normalized_data, ignore_index=True)

    print(f"Normalized {len(df['fly'].unique())} flies to standard 0-100% time grid")
    print(f"Resulting dataset shape: {result_df.shape}")

    return result_df


def find_maximum_shared_time(df, fly_col="fly", time_col="adjusted_time"):
    """
    Find the maximum adjusted time that ALL flies have data for.

    Args:
        df: DataFrame with fly tracking data
        fly_col: Column name for fly identifier
        time_col: Column name for adjusted time

    Returns:
        Maximum time that all flies share
    """
    print(f"\nFinding maximum shared time across all flies...")

    # Get the maximum time for each fly
    max_times_per_fly = df.groupby(fly_col)[time_col].max()

    print(f"Number of flies: {len(max_times_per_fly)}")
    print(f"Max time per fly range: {max_times_per_fly.min():.2f} to {max_times_per_fly.max():.2f} seconds")

    # The maximum shared time is the minimum of all maximum times
    max_shared_time = max_times_per_fly.min()

    print(f"Maximum shared time across all flies: {max_shared_time:.2f} seconds")

    # Print some statistics
    flies_with_longer_data = (max_times_per_fly > max_shared_time).sum()
    print(f"Number of flies with data beyond {max_shared_time:.2f}s: {flies_with_longer_data}")

    return max_shared_time


def trim_data_to_shared_time(df, max_shared_time, time_col="adjusted_time"):
    """
    Trim dataset to only include data up to the maximum shared time.

    Args:
        df: DataFrame with fly tracking data
        max_shared_time: Maximum time to include
        time_col: Column name for adjusted time

    Returns:
        Trimmed DataFrame
    """
    print(f"\nTrimming data to shared time limit: {max_shared_time:.2f}s")

    original_shape = df.shape
    df_trimmed = df[df[time_col] <= max_shared_time].copy()

    print(f"Original shape: {original_shape}")
    print(f"Trimmed shape: {df_trimmed.shape}")
    print(f"Rows removed: {original_shape[0] - df_trimmed.shape[0]}")

    return df_trimmed


def create_line_plot_trimmed_time(
    data,
    x_col,
    y_cols,
    hue_col,
    title,
    ax,
    color_mapping=None,
    x_label="Adjusted Time (seconds)",
    y_label="Distance to Ball (pixels)",
    time_bins=0.1,
):
    """
    Create a line plot with trimmed time data, time binning, and confidence intervals.

    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis (time)
        y_cols: List of column names for y-axis (ball distances)
        hue_col: Column name for color grouping
        title: Plot title
        ax: Matplotlib axes object
        color_mapping: Dictionary mapping hue values to colors
        x_label: Label for x-axis
        y_label: Label for y-axis
        time_bins: Bin size in seconds for time averaging
    """
    from scipy import stats as scipy_stats

    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Create time bins for averaging
    data = data.copy()
    min_time = data[x_col].min()
    max_time = data[x_col].max()
    time_bin_edges = np.arange(min_time, max_time + time_bins, time_bins)
    data["time_bin"] = pd.cut(data[x_col], bins=time_bin_edges, labels=False, include_lowest=True)
    data["time_bin_center"] = data["time_bin"] * time_bins + time_bins / 2 + min_time

    # Set colors
    if color_mapping is not None:
        color_dict = color_mapping
    else:
        unique_hues = data[hue_col].unique()
        colors = sns.color_palette("Set1", n_colors=len(unique_hues))
        color_dict = dict(zip(unique_hues, colors))

    # Plot each y-column
    for i, y_col in enumerate(y_cols):
        if y_col not in data.columns:
            print(f"Warning: Column '{y_col}' not found in data")
            continue

        # Skip training_ball for control condition (will be all NaN)
        if y_col == "training_ball" and data[y_col].isnull().all():
            continue

        # Group by fly, hue, and time bin to get median per fly per time bin first
        fly_col = "fly" if "fly" in data.columns else None
        if not fly_col:
            print("Warning: No 'fly' column found, cannot calculate confidence intervals properly")
            continue

        # Remove NaN values for this y_col
        data_clean = data.dropna(subset=[y_col])
        if data_clean.empty:
            continue

        # Get median value per fly per time bin for each hue value
        binned_data = data_clean.groupby([fly_col, hue_col, "time_bin_center"])[y_col].median().reset_index()
        binned_data = binned_data.rename(columns={y_col: f"{y_col}_median"})

        # Now calculate statistics across flies for each time bin and hue value
        for hue_val in data[hue_col].unique():
            hue_data = binned_data[binned_data[hue_col] == hue_val]

            if hue_data.empty:
                continue

            # Skip training_ball for control condition
            if y_col == "training_ball" and hue_data[f"{y_col}_median"].isnull().all():
                continue

            # Calculate statistics for each time bin
            time_stats = (
                hue_data.groupby("time_bin_center")[f"{y_col}_median"]
                .agg(["mean", "std", "count", "sem"])
                .reset_index()
            )

            # Calculate 95% confidence intervals
            confidence_level = 0.95
            alpha = 1 - confidence_level
            time_stats["ci"] = time_stats.apply(
                lambda row: scipy_stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0,
                axis=1,
            )

            # Plot settings
            color = color_dict[hue_val] if color_dict else None
            linestyle = "-" if i == 0 else "--"  # Different linestyles for different y_cols
            label = f"{hue_val} - {y_col}" if len(y_cols) > 1 else str(hue_val)

            # Plot mean trajectory
            ax.plot(
                time_stats["time_bin_center"],
                time_stats["mean"],
                color=color,
                linestyle=linestyle,
                label=label,
                linewidth=2,
                alpha=0.8,
            )

            # Add confidence intervals
            ax.fill_between(
                time_stats["time_bin_center"],
                time_stats["mean"] - time_stats["ci"],
                time_stats["mean"] + time_stats["ci"],
                color=color,
                alpha=0.2,
            )

    # Add vertical line at time 0
    ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time (0s)")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add sample size annotation
    n_flies = len(data["fly"].unique()) if "fly" in data.columns else len(data)
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    return ax


def main():
    """Main function to create the F1 coordinates plots."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot F1 coordinates over adjusted time")
    parser.add_argument(
        "--filter-summary",
        action="store_true",
        default=True,
        help="Filter F1 coordinates to only include flies present in summary dataset (default: True)",
    )
    parser.add_argument(
        "--no-filter-summary",
        action="store_false",
        dest="filter_summary",
        help="Do not filter F1 coordinates based on summary dataset",
    )
    parser.add_argument(
        "--coordinates-path",
        type=str,
        default="/mnt/upramdya_data/MD/F1_Tracks/Datasets/251014_10_F1_coordinates_F1_New_Data/F1_coordinates/pooled_F1_coordinates.feather",
        help="Path to F1 coordinates dataset",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_17_summary_F1_New_Data/summary/pooled_summary.feather",
        help="Path to summary dataset",
    )

    args = parser.parse_args()

    # Dataset paths
    coordinates_dataset_path = args.coordinates_path
    summary_dataset_path = args.summary_path

    # Alternative paths to try if the main ones don't exist
    alternative_coordinates_paths = [
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251014_F1_coordinates_F1_New_Data/F1_coordinates/pooled_F1_coordinates.feather",
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/latest_F1_coordinates/pooled_F1_coordinates.feather",
        # Add more potential paths as needed
    ]

    alternative_summary_paths = [
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251013_summary_F1_New_Data/summary/pooled_summary.feather",
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/latest_summary/pooled_summary.feather",
        # Add more potential paths as needed
    ]

    # Try to load coordinates dataset from multiple possible paths
    df = None
    used_coordinates_path = None

    for path in [coordinates_dataset_path] + alternative_coordinates_paths:
        try:
            if Path(path).exists():
                df = pd.read_feather(path)
                used_coordinates_path = path
                print(f"F1 coordinates dataset loaded successfully from: {path}")
                print(f"F1 coordinates dataset shape: {df.shape}")
                break
        except Exception as e:
            print(f"Could not load F1 coordinates dataset from {path}: {e}")
            continue

    if df is None:
        print("Could not load F1 coordinates dataset from any of the specified paths.")
        print("Please check the dataset path and ensure the F1_coordinates dataset exists.")
        print("You may need to generate it first using the Dataset class with metrics='F1_coordinates'")
        return

    # Try to load summary dataset for filtering
    summary_df = None
    used_summary_path = None

    if args.filter_summary:
        for path in [summary_dataset_path] + alternative_summary_paths:
            try:
                if Path(path).exists():
                    summary_df = pd.read_feather(path)
                    used_summary_path = path
                    print(f"Summary dataset loaded successfully from: {path}")
                    print(f"Summary dataset shape: {summary_df.shape}")
                    break
            except Exception as e:
                print(f"Could not load summary dataset from {path}: {e}")
                continue

        if summary_df is None:
            print("Warning: Could not load summary dataset from any of the specified paths.")
            print("Proceeding without filtering F1 coordinates dataset.")
            print("This may result in inconsistencies between coordinate and summary data.")
        else:
            # Filter F1 coordinates dataset to only include flies present in summary dataset
            print(f"\nFiltering F1 coordinates dataset based on summary dataset...")

            # Get unique flies from summary dataset
            if "fly" in summary_df.columns:
                summary_flies = set(summary_df["fly"].unique())
                print(f"Summary dataset contains {len(summary_flies)} unique flies")
            else:
                print("Warning: 'fly' column not found in summary dataset")
                print("Available columns in summary dataset:", list(summary_df.columns))
                summary_flies = set()

            # Get unique flies from coordinates dataset
            if "fly" in df.columns:
                coordinates_flies = set(df["fly"].unique())
                print(f"F1 coordinates dataset contains {len(coordinates_flies)} unique flies")

                # Find flies present in both datasets
                common_flies = summary_flies.intersection(coordinates_flies)
                flies_only_in_coordinates = coordinates_flies - summary_flies
                flies_only_in_summary = summary_flies - coordinates_flies

                print(f"Flies present in both datasets: {len(common_flies)}")
                print(f"Flies only in coordinates dataset: {len(flies_only_in_coordinates)}")
                print(f"Flies only in summary dataset: {len(flies_only_in_summary)}")

                if flies_only_in_coordinates:
                    print(f"Examples of flies only in coordinates: {list(flies_only_in_coordinates)[:5]}")
                if flies_only_in_summary:
                    print(f"Examples of flies only in summary: {list(flies_only_in_summary)[:5]}")

                # Filter coordinates dataset to only include flies present in summary
                if common_flies:
                    original_shape = df.shape
                    df = df[df["fly"].isin(common_flies)].copy()
                    print(f"Filtered F1 coordinates dataset: {original_shape} -> {df.shape}")
                    print(f"Removed {original_shape[0] - df.shape[0]} rows for flies not in summary dataset")
                else:
                    print("Warning: No common flies found between datasets!")
                    print("This suggests a serious mismatch between the datasets.")
            else:
                print("Warning: 'fly' column not found in F1 coordinates dataset")
                print("Available columns in F1 coordinates dataset:", list(df.columns))
    else:
        print("Skipping summary dataset filtering (--no-filter-summary specified)")
        summary_df = None

    # Filter out specific dates if needed (based on existing scripts)
    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")

    # Store original data before any transformations for the trimmed time approach
    original_df = df.copy()

    # Transform adjusted_time to normalized percentage (0-100%) with standardized intervals for first approach
    print("\\nTransforming adjusted_time to normalized percentages...")
    df_percentage = normalize_adjusted_time_to_percentage(df)
    print(
        f"After normalization: adjusted_time range {df_percentage['adjusted_time'].min():.1f}% to {df_percentage['adjusted_time'].max():.1f}%"
    )

    # Print available columns to help identify the correct ones
    print("\\nAvailable columns:")
    for i, col in enumerate(df_percentage.columns):
        print(f"  {i+1:2d}. {col}")

    # Identify the required columns
    time_col = "adjusted_time"
    training_ball_col = "training_ball"
    test_ball_col = "test_ball"
    pretraining_col = None
    f1_condition_col = None

    # Look for pretraining column
    for col in df_percentage.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    # Look for F1_condition column
    for col in df_percentage.columns:
        if "f1" in col.lower() and "condition" in col.lower():
            f1_condition_col = col
            break

    # Check if we have the required columns
    required_columns = {
        "time": time_col,
        "training_ball": training_ball_col,
        "test_ball": test_ball_col,
        "pretraining": pretraining_col,
        "f1_condition": f1_condition_col,
    }

    missing = [name for name, col in required_columns.items() if col not in df_percentage.columns or col is None]
    if missing:
        print(f"\\nMissing required columns: {missing}")
        print("Available columns that might be relevant:")
        relevant_cols = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["time", "ball", "train", "f1", "condition"])
        ]
        for col in relevant_cols:
            print(f"  - {col}")
        return

    print(f"\\nUsing columns:")
    print(f"  Time: {time_col}")
    print(f"  Training ball: {training_ball_col}")
    print(f"  Test ball: {test_ball_col}")
    print(f"  Pretraining: {pretraining_col}")
    print(f"  F1 condition: {f1_condition_col}")

    # Clean the data - be more careful with NaN handling
    # Essential columns that must not be NaN
    essential_cols = [time_col, test_ball_col, pretraining_col, f1_condition_col]

    # First, let's see what we have before cleaning (percentage dataset)
    print(f"\nPercentage dataset shape before cleaning: {df_percentage.shape}")
    print(f"NaN counts in key columns:")
    for col in essential_cols + [training_ball_col]:
        if col in df_percentage.columns:
            nan_count = df_percentage[col].isnull().sum()
            total_count = len(df_percentage)
            print(f"  {col}: {nan_count}/{total_count} ({100*nan_count/total_count:.1f}%)")

    # Clean data for percentage approach: only drop rows where essential columns are NaN
    # training_ball can be NaN for control flies, so we'll handle it separately
    df_percentage_clean = (
        df_percentage[essential_cols + [training_ball_col, "fly"]].copy()
        if "fly" in df_percentage.columns
        else df_percentage[essential_cols + [training_ball_col]].copy()
    )

    # Drop rows where essential columns are NaN
    df_percentage_clean = df_percentage_clean.dropna(subset=essential_cols)

    print(f"\nPercentage dataset shape after removing NaNs in essential columns: {df_percentage_clean.shape}")

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_percentage_clean[pretraining_col].unique()}")
    print(f"Unique {f1_condition_col} values: {df_percentage_clean[f1_condition_col].unique()}")

    # Print time range for percentage data
    print(
        f"Percentage time range: {df_percentage_clean[time_col].min():.2f}% to {df_percentage_clean[time_col].max():.2f}%"
    )

    # =============================================================================
    # IMPLEMENTATION 1: PERCENTAGE-BASED NORMALIZATION (0-100%)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 1: PERCENTAGE-BASED NORMALIZATION")
    print(f"{'='*60}")

    # Create figure for percentage-based plots
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Define color mappings for consistent styling
    pretraining_colors = {"n": "steelblue", "y": "orange"}
    f1_condition_colors = {"control": "steelblue", "pretrained": "orange", "pretrained_unlocked": "lightgreen"}

    # Plot 1: Test ball only by Pretraining condition (percentage-based)
    create_line_plot(
        data=df_percentage_clean,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col=pretraining_col,
        title="Test Ball Distance Over Normalized Time (%) by Pretraining Condition",
        ax=ax1,
        color_mapping=pretraining_colors,
    )

    # Format and save first percentage-based plot
    plt.tight_layout()
    output_path_percentage_pretraining = Path(__file__).parent / "f1_coordinates_percentage_time_pretraining.png"
    plt.savefig(output_path_percentage_pretraining, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots (Pretraining) saved to: {output_path_percentage_pretraining}")

    # Create second figure for percentage-based plots
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot 2: Test ball only by F1_condition (percentage-based)
    create_line_plot(
        data=df_percentage_clean,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col=f1_condition_col,
        title="Test Ball Distance Over Normalized Time (%) by F1 Condition",
        ax=ax2,
        color_mapping=f1_condition_colors,
    )

    # Format and save second percentage-based plot
    plt.tight_layout()
    output_path_percentage_f1 = Path(__file__).parent / "f1_coordinates_percentage_time_f1condition.png"
    plt.savefig(output_path_percentage_f1, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots (F1 Condition) saved to: {output_path_percentage_f1}")

    # =============================================================================
    # IMPLEMENTATION 2: TRIMMED TIME APPROACH (ACTUAL SECONDS)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 2: TRIMMED TIME APPROACH")
    print(f"{'='*60}")

    # Start with original data (before percentage normalization)
    # We need to reload or use the original df before normalization
    original_df = df.copy()  # Use original data before percentage normalization

    # Clean the original data the same way
    original_df_clean = original_df[essential_cols + [training_ball_col, "fly"]].copy()
    original_df_clean = original_df_clean.dropna(subset=essential_cols)

    print(f"Original data shape for trimmed approach: {original_df_clean.shape}")
    print(
        f"Original time range: {original_df_clean[time_col].min():.2f} to {original_df_clean[time_col].max():.2f} seconds"
    )

    # Find maximum shared time across all flies
    max_shared_time = find_maximum_shared_time(original_df_clean, "fly", time_col)

    # Trim data to shared time
    df_trimmed = trim_data_to_shared_time(original_df_clean, max_shared_time, time_col)

    # Create first figure for trimmed time plots
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot 3: Test ball only by Pretraining condition (trimmed time)
    create_line_plot_trimmed_time(
        data=df_trimmed,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col=pretraining_col,
        title="Test Ball Distance Over Trimmed Adjusted Time by Pretraining Condition",
        ax=ax3,
        color_mapping=pretraining_colors,
        x_label="Adjusted Time (seconds)",
        time_bins=0.1,
    )

    # Format and save first trimmed time plot
    plt.tight_layout()
    output_path_trimmed_pretraining = Path(__file__).parent / "f1_coordinates_trimmed_time_pretraining.png"
    plt.savefig(output_path_trimmed_pretraining, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (Pretraining) saved to: {output_path_trimmed_pretraining}")

    # Create second figure for trimmed time plots
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot 4: Test ball only by F1_condition (trimmed time)
    create_line_plot_trimmed_time(
        data=df_trimmed,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col=f1_condition_col,
        title="Test Ball Distance Over Trimmed Adjusted Time by F1 Condition",
        ax=ax4,
        color_mapping=f1_condition_colors,
        x_label="Adjusted Time (seconds)",
        time_bins=0.1,
    )

    # Format and save second trimmed time plot
    plt.tight_layout()
    output_path_trimmed_f1 = Path(__file__).parent / "f1_coordinates_trimmed_time_f1condition.png"
    plt.savefig(output_path_trimmed_f1, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (F1 Condition) saved to: {output_path_trimmed_f1}")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    print("\\n" + "=" * 80)
    print("SUMMARY STATISTICS FOR BOTH APPROACHES")
    print("=" * 80)

    print(f"\\nPercentage-based approach:")
    print(f"  - Data normalized to 0-100% per fly")
    print(
        f"  - Final dataset: {df_percentage_clean.shape[0]} data points from {df_percentage_clean['fly'].nunique()} flies"
    )
    print(f"  - Time range: {df_percentage_clean[time_col].min():.1f}% to {df_percentage_clean[time_col].max():.1f}%")

    print(f"\\nTrimmed time approach:")
    print(f"  - Maximum shared time: {max_shared_time:.2f} seconds")
    print(f"  - Final dataset: {df_trimmed.shape[0]} data points from {df_trimmed['fly'].nunique()} flies")
    print(f"  - Time range: {df_trimmed[time_col].min():.2f}s to {df_trimmed[time_col].max():.2f}s")

    # Statistics for different time windows (using trimmed data for actual time windows)
    time_windows = [(-5, 0), (0, 5), (5, 15), (15, max_shared_time)]
    window_names = [
        "Pre-exit (-5 to 0s)",
        "Early post-exit (0 to 5s)",
        "Mid post-exit (5 to 15s)",
        f"Late post-exit (15 to {max_shared_time:.1f}s)",
    ]

    print(f"\\nTime window analysis (using trimmed time data):")
    for (start, end), window_name in zip(time_windows, window_names):
        print(f"\\n{window_name}:")
        window_data = filter_data_by_time_window(df_trimmed, time_col, start, end)

        if not window_data.empty:
            # Test ball statistics by pretraining
            test_stats = (
                window_data.groupby(pretraining_col)[test_ball_col].agg(["count", "mean", "std", "median"]).round(3)
            )
            print(f"Test ball by {pretraining_col}:")
            print(test_stats)

            # Test ball statistics by F1 condition
            f1_stats = (
                window_data.groupby(f1_condition_col)[test_ball_col].agg(["count", "mean", "std", "median"]).round(3)
            )
            print(f"Test ball by {f1_condition_col}:")
            print(f1_stats)
        else:
            print("No data in this time window")

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("Two visualization approaches created:")
    print("1. Percentage-based: Normalizes each fly's time to 0-100%, allows comparison of relative progression")
    print("2. Trimmed time: Uses actual seconds but only up to maximum shared time, preserves absolute timing")

    # Dataset filtering summary
    if args.filter_summary:
        if summary_df is not None:
            print(f"\nDataset filtering summary:")
            print(f"  - F1 coordinates dataset path: {used_coordinates_path}")
            print(f"  - Summary dataset path: {used_summary_path}")
            if "fly" in df.columns and "fly" in summary_df.columns:
                coordinates_flies = set(df["fly"].unique())
                summary_flies = set(summary_df["fly"].unique())
                common_flies = coordinates_flies.intersection(summary_flies)
                print(f"  - Flies in filtered coordinates dataset: {len(coordinates_flies)}")
                print(f"  - Flies in summary dataset: {len(summary_flies)}")
                print(f"  - Common flies used for analysis: {len(common_flies)}")
            print(f"  - Final coordinates dataset shape: {df.shape}")
        else:
            print(f"\nDataset filtering attempted but summary dataset could not be loaded")
            print(f"  - F1 coordinates dataset path: {used_coordinates_path}")
            print(f"  - Final coordinates dataset shape: {df.shape}")
    else:
        print(f"\nNo dataset filtering applied (--no-filter-summary specified)")
        print(f"  - F1 coordinates dataset path: {used_coordinates_path}")
        print(f"  - Final coordinates dataset shape: {df.shape}")

    print(f"\nFiles saved:")
    print(f"  - {output_path_percentage_pretraining}")
    print(f"  - {output_path_percentage_f1}")
    print(f"  - {output_path_trimmed_pretraining}")
    print(f"  - {output_path_trimmed_f1}")

    plt.show()

    return df_percentage_clean, df_trimmed


if __name__ == "__main__":
    data = main()
