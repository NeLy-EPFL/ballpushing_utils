#!/usr/bin/env python3
"""
Script to plot dark olfaction coordinates over adjusted time with two approaches:
1. Percentage-based normalization (0-100%) - normalizes each fly's time individually
2. Trimmed time approach - uses actual seconds but only up to max time all flies share

This script is adapted for dark olfaction experiments with grouping by:
- Light (on/off)
- Genotype (TNTxEmptyGal4, TNTxIR8a)
- BallType (ctrl, sand)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats


def filter_data_by_time_window(data, time_col, start_time=-10, end_time=30):
    """Filter data to a specific time window."""
    return data[(data[time_col] >= start_time) & (data[time_col] <= end_time)]


def create_line_plot(data, x_col, y_col, hue_col, title, ax, color_mapping=None, linestyle_mapping=None):
    """Create a line plot with binned data and confidence intervals."""

    filtered_data = data.copy()

    if filtered_data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Create color mapping if not provided
    if color_mapping is None:
        unique_hues = filtered_data[hue_col].unique()
        colors = sns.color_palette("Set1", n_colors=len(unique_hues))
        color_mapping = dict(zip(unique_hues, colors))

    # Skip if y_col contains all NaN values
    if y_col not in filtered_data.columns:
        print(f"Warning: Column '{y_col}' not found in data")
        return ax

    # Remove NaN values for this y_col
    subset_clean = filtered_data.dropna(subset=[y_col])
    if subset_clean.empty:
        print(f"Warning: No valid data for '{y_col}'")
        return ax

    # Check for fly column
    fly_col = "fly" if "fly" in subset_clean.columns else None
    if not fly_col:
        print("Warning: No 'fly' column found")
        return ax

    # Create time bins (0.1% intervals for percentage-based adjusted time)
    subset_clean = subset_clean.copy()
    subset_clean["time_bin"] = np.floor(subset_clean[x_col] / 0.1) * 0.1  # 0.1% bins

    # Get median value per fly per time bin
    binned_data = subset_clean.groupby([fly_col, "time_bin", hue_col])[y_col].median().reset_index()
    binned_data = binned_data.rename(columns={y_col: f"{y_col}_median"})

    # Now calculate mean and confidence intervals across flies for each time bin and hue
    from scipy import stats as scipy_stats

    for hue_val in subset_clean[hue_col].unique():
        hue_data = binned_data[binned_data[hue_col] == hue_val]

        if hue_data.empty:
            continue

        # Calculate statistics for each time bin
        time_stats = hue_data.groupby("time_bin")[f"{y_col}_median"].agg(["mean", "std", "count", "sem"]).reset_index()

        # Calculate 95% confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        time_stats["ci"] = time_stats.apply(
            lambda row: scipy_stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0,
            axis=1,
        )

        # Plot settings
        color = color_mapping.get(hue_val, "black")
        linestyle = linestyle_mapping.get(hue_val, "-") if linestyle_mapping else "-"
        label = f"{hue_val}"

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
    ax.set_ylabel("Distance to Ball (pixels)", fontsize=12)
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


def normalize_adjusted_time_to_percentage(df, time_col="adjusted_time"):
    """
    Normalize time to 0-100% for each fly and resample to standard 0.1% intervals.

    Args:
        df: DataFrame with time and 'fly' columns
        time_col: Name of the time column to normalize (default: 'adjusted_time')

    Returns:
        DataFrame with normalized time and standardized intervals
    """
    if time_col not in df.columns or "fly" not in df.columns:
        print(f"Warning: Cannot normalize - missing '{time_col}' or 'fly' columns")
        return df

    print(f"Original {time_col} range: {df[time_col].min():.2f} to {df[time_col].max():.2f} seconds")

    # Create standard time grid (0-100% in 0.1% steps)
    standard_time_grid = np.arange(0, 100.1, 0.1)

    normalized_data = []

    # Group by fly and normalize each fly's data separately
    for fly_id in df["fly"].unique():
        fly_data = df[df["fly"] == fly_id].copy()

        # Skip flies with no valid time data
        if fly_data[time_col].isna().all():
            continue

        # Get valid time values for this fly
        valid_mask = ~fly_data[time_col].isna()
        if not valid_mask.any():
            continue

        valid_times = fly_data.loc[valid_mask, time_col]
        min_time = valid_times.min()
        max_time = valid_times.max()

        # Skip if all times are the same or we have less than 2 points
        if max_time <= min_time or valid_mask.sum() < 2:
            continue

        # Normalize to 0-100% for this fly
        normalized_time = ((fly_data[time_col] - min_time) / (max_time - min_time)) * 100

        # Create DataFrame for this fly with standard time grid
        # Use the original time_col name to maintain consistency
        fly_normalized = pd.DataFrame({time_col: standard_time_grid})

        # Add fly identifier and other metadata columns
        for col in ["fly", "Light", "Genotype", "BallType"]:
            if col in fly_data.columns:
                fly_normalized[col] = fly_data[col].iloc[0]

        # Interpolate ball distance to standard time grid
        ball_col = "distance_ball_0"
        if ball_col in fly_data.columns:
            valid_indices = valid_mask
            interp_values = np.interp(
                standard_time_grid,
                normalized_time[valid_indices],
                fly_data.loc[valid_indices, ball_col],
                left=np.nan,
                right=np.nan,
            )
            fly_normalized[ball_col] = interp_values

        # Add other metadata columns without interpolation
        for col in df.columns:
            if col not in fly_normalized.columns and col not in [
                "adjusted_time",
                "time",
                "distance_ball_0",
                "frame",
                "x_fly_0",
                "y_fly_0",
                "x_ball_0",
                "y_ball_0",
                "distance_fly_0",
                "interaction_event",
                "interaction_event_onset",
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
    y_col,
    hue_col,
    title,
    ax,
    color_mapping=None,
    linestyle_mapping=None,
    x_label="Adjusted Time (seconds)",
    y_label="Distance to Ball (pixels)",
    time_bins=0.1,
):
    """
    Create a line plot with trimmed time data, time binning, and confidence intervals.

    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis (time)
        y_col: Column name for y-axis (ball distance)
        hue_col: Column name for color grouping
        title: Plot title
        ax: Matplotlib axes object
        color_mapping: Dictionary mapping hue values to colors
        linestyle_mapping: Dictionary mapping hue values to linestyles
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

    if y_col not in data.columns:
        print(f"Warning: Column '{y_col}' not found in data")
        return ax

    # Check for fly column
    fly_col = "fly" if "fly" in data.columns else None
    if not fly_col:
        print("Warning: No 'fly' column found, cannot calculate confidence intervals properly")
        return ax

    # Remove NaN values for this y_col
    data_clean = data.dropna(subset=[y_col])
    if data_clean.empty:
        return ax

    # Get median value per fly per time bin for each hue value
    binned_data = data_clean.groupby([fly_col, hue_col, "time_bin_center"])[y_col].median().reset_index()
    binned_data = binned_data.rename(columns={y_col: f"{y_col}_median"})

    # Now calculate statistics across flies for each time bin and hue value
    for hue_val in data[hue_col].unique():
        hue_data = binned_data[binned_data[hue_col] == hue_val]

        if hue_data.empty:
            continue

        # Calculate statistics for each time bin
        time_stats = (
            hue_data.groupby("time_bin_center")[f"{y_col}_median"].agg(["mean", "std", "count", "sem"]).reset_index()
        )

        # Calculate 95% confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        time_stats["ci"] = time_stats.apply(
            lambda row: scipy_stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0,
            axis=1,
        )

        # Plot settings
        color = color_dict.get(hue_val, "black")
        linestyle = linestyle_mapping.get(hue_val, "-") if linestyle_mapping else "-"
        label = str(hue_val)

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
    """Main function to create the dark olfaction coordinates plots."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot dark olfaction coordinates over adjusted time")
    parser.add_argument(
        "--coordinates-path",
        type=str,
        default="/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Datasets/251106_08_summary_TNT_Olfaction_Dark_Data/coordinates/pooled_coordinates.feather",
        help="Path to dark olfaction coordinates dataset",
    )

    args = parser.parse_args()

    # Dataset path
    coordinates_dataset_path = args.coordinates_path

    # Load coordinates dataset
    print(f"Loading dark olfaction coordinates dataset from: {coordinates_dataset_path}")
    try:
        df = pd.read_feather(coordinates_dataset_path)
        print(f"✅ Dark olfaction coordinates dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Could not load dark olfaction coordinates dataset: {e}")
        return

    # Store original data before any transformations
    original_df = df.copy()

    # Print available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Identify required columns
    # Check if adjusted_time is available, otherwise use regular time
    if "adjusted_time" in df.columns and not df["adjusted_time"].isna().all():
        time_col = "adjusted_time"
        print("Using adjusted_time (flies exited corridor)")
    else:
        time_col = "time"
        print("Using regular time (no corridor exit data)")

    ball_distance_col = "distance_ball_0"

    # Grouping columns
    light_col = "Light"
    genotype_col = "Genotype"
    balltype_col = "BallType"

    # Check required columns
    required_columns = {
        "time": time_col,
        "ball_distance": ball_distance_col,
        "light": light_col,
        "genotype": genotype_col,
        "balltype": balltype_col,
        "fly": "fly",
    }

    missing = [name for name, col in required_columns.items() if col not in df.columns]
    if missing:
        print(f"\nMissing required columns: {missing}")
        return

    print(f"\nUsing columns:")
    print(f"  Time: {time_col}")
    print(f"  Ball distance: {ball_distance_col}")
    print(f"  Light: {light_col}")
    print(f"  Genotype: {genotype_col}")
    print(f"  BallType: {balltype_col}")

    # Clean the data
    essential_cols = [time_col, ball_distance_col, light_col, genotype_col, balltype_col, "fly"]

    print(f"\nDataset shape before cleaning: {df.shape}")
    print(f"NaN counts in key columns:")
    for col in essential_cols:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            total_count = len(df)
            print(f"  {col}: {nan_count}/{total_count} ({100*nan_count/total_count:.1f}%)")

    # Clean data: only drop rows where essential columns are NaN
    df_clean = df[essential_cols].copy()
    df_clean = df_clean.dropna(subset=essential_cols)

    print(f"\nDataset shape after removing NaNs in essential columns: {df_clean.shape}")

    # Print unique values
    print(f"Unique {light_col} values: {df_clean[light_col].unique()}")
    print(f"Unique {genotype_col} values: {df_clean[genotype_col].unique()}")
    print(f"Unique {balltype_col} values: {df_clean[balltype_col].unique()}")

    # Print time range
    print(f"Time range: {df_clean[time_col].min():.2f}s to {df_clean[time_col].max():.2f}s")

    # =============================================================================
    # IMPLEMENTATION 1: PERCENTAGE-BASED NORMALIZATION (0-100%)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 1: PERCENTAGE-BASED NORMALIZATION")
    print(f"{'='*60}")

    # Transform time to normalized percentage
    df_percentage = normalize_adjusted_time_to_percentage(df_clean, time_col=time_col)
    print(
        f"After normalization: {time_col} range {df_percentage[time_col].min():.1f}% to {df_percentage[time_col].max():.1f}%"
    )

    # Define color mappings for consistent styling
    genotype_colors = {"TNTxEmptyGal4": "blue", "TNTxIR8a": "orange"}
    balltype_colors = {"ctrl": "green", "sand": "brown"}

    # Combined grouping colors (Light_Genotype_BallType)
    # Create combined group column - convert categorical columns to strings first
    df_percentage["Combined_Group"] = (
        df_percentage[light_col].astype(str)
        + "_"
        + df_percentage[genotype_col].astype(str)
        + "_"
        + df_percentage[balltype_col].astype(str)
    )

    # Plot 1: Ball distance by Genotype (percentage-based) - SIDE-BY-SIDE BY LIGHT
    # Get unique light conditions
    light_conditions = sorted(df_percentage[light_col].unique())

    # Create side-by-side subplots for light conditions
    fig1, axes1 = plt.subplots(1, len(light_conditions), figsize=(12 * len(light_conditions) / 2, 6))
    if len(light_conditions) == 1:
        axes1 = [axes1]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions):
        # Filter data for this light condition
        df_light = df_percentage[df_percentage[light_col] == light_condition].copy()

        create_line_plot(
            data=df_light,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col=genotype_col,
            title=f"Light: {light_condition}",
            ax=axes1[idx],
            color_mapping=genotype_colors,
        )

    # Add overall title
    fig1.suptitle("Ball Distance Over Normalized Time (%) by Genotype", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path_percentage_genotype = Path(__file__).parent / "dark_olfaction_coordinates_percentage_time_genotype.png"
    plt.savefig(output_path_percentage_genotype, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots (Genotype, side-by-side) saved to: {output_path_percentage_genotype}")

    # Plot 2: Ball distance by BallType (percentage-based) - SIDE-BY-SIDE BY LIGHT
    # Create side-by-side subplots for light conditions
    fig2, axes2 = plt.subplots(1, len(light_conditions), figsize=(12 * len(light_conditions) / 2, 6))
    if len(light_conditions) == 1:
        axes2 = [axes2]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions):
        # Filter data for this light condition
        df_light = df_percentage[df_percentage[light_col] == light_condition].copy()

        create_line_plot(
            data=df_light,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col=balltype_col,
            title=f"Light: {light_condition}",
            ax=axes2[idx],
            color_mapping=balltype_colors,
        )

    # Add overall title
    fig2.suptitle("Ball Distance Over Normalized Time (%) by BallType", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path_percentage_balltype = Path(__file__).parent / "dark_olfaction_coordinates_percentage_time_balltype.png"
    plt.savefig(output_path_percentage_balltype, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots (BallType, side-by-side) saved to: {output_path_percentage_balltype}")

    # Plot 3: Ball distance by combined grouping (percentage-based) - SIDE-BY-SIDE BY LIGHT
    # Create color mapping for combined groups
    combined_colors = {}
    for group in df_percentage["Combined_Group"].unique():
        light, genotype, balltype = group.split("_", 2)
        # Use genotype color as base
        combined_colors[group] = genotype_colors.get(genotype, "gray")

    # Create linestyle mapping based on balltype
    combined_linestyles = {}
    for group in df_percentage["Combined_Group"].unique():
        light, genotype, balltype = group.split("_", 2)
        combined_linestyles[group] = ":" if balltype == "sand" else "-"

    # Get unique light conditions
    light_conditions = sorted(df_percentage[light_col].unique())

    # Create side-by-side subplots for light conditions
    fig3, axes3 = plt.subplots(1, len(light_conditions), figsize=(14 * len(light_conditions) / 2, 6))
    if len(light_conditions) == 1:
        axes3 = [axes3]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions):
        # Filter data for this light condition
        df_light = df_percentage[df_percentage[light_col] == light_condition].copy()

        create_line_plot(
            data=df_light,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col="Combined_Group",
            title=f"Light: {light_condition}",
            ax=axes3[idx],
            color_mapping=combined_colors,
            linestyle_mapping=combined_linestyles,
        )

    # Add overall title
    fig3.suptitle(
        "Ball Distance Over Normalized Time (%) by Genotype × BallType", fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    output_path_percentage_combined = Path(__file__).parent / "dark_olfaction_coordinates_percentage_time_combined.png"
    plt.savefig(output_path_percentage_combined, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots (Combined, side-by-side) saved to: {output_path_percentage_combined}")

    # =============================================================================
    # IMPLEMENTATION 2: TRIMMED TIME APPROACH (ACTUAL SECONDS)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 2: TRIMMED TIME APPROACH")
    print(f"{'='*60}")

    # Use original cleaned data
    original_df_clean = df_clean.copy()

    print(f"Original data shape for trimmed approach: {original_df_clean.shape}")
    print(
        f"Original time range: {original_df_clean[time_col].min():.2f} to {original_df_clean[time_col].max():.2f} seconds"
    )

    # Find maximum shared time across all flies
    max_shared_time = find_maximum_shared_time(original_df_clean, "fly", time_col)

    # Trim data to shared time
    df_trimmed = trim_data_to_shared_time(original_df_clean, max_shared_time, time_col)

    # Add combined group column - convert categorical columns to strings first
    df_trimmed["Combined_Group"] = (
        df_trimmed[light_col].astype(str)
        + "_"
        + df_trimmed[genotype_col].astype(str)
        + "_"
        + df_trimmed[balltype_col].astype(str)
    )

    # Plot 4: Ball distance by Genotype (trimmed time) - SIDE-BY-SIDE BY LIGHT
    # Get unique light conditions
    light_conditions_trimmed = sorted(df_trimmed[light_col].unique())

    # Create side-by-side subplots for light conditions
    fig4, axes4 = plt.subplots(1, len(light_conditions_trimmed), figsize=(12 * len(light_conditions_trimmed) / 2, 6))
    if len(light_conditions_trimmed) == 1:
        axes4 = [axes4]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions_trimmed):
        # Filter data for this light condition
        df_light = df_trimmed[df_trimmed[light_col] == light_condition].copy()

        create_line_plot_trimmed_time(
            data=df_light,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col=genotype_col,
            title=f"Light: {light_condition}",
            ax=axes4[idx],
            color_mapping=genotype_colors,
            x_label="Adjusted Time (seconds)",
            time_bins=0.1,
        )

    # Add overall title
    fig4.suptitle("Ball Distance Over Trimmed Adjusted Time by Genotype", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path_trimmed_genotype = Path(__file__).parent / "dark_olfaction_coordinates_trimmed_time_genotype.png"
    plt.savefig(output_path_trimmed_genotype, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (Genotype, side-by-side) saved to: {output_path_trimmed_genotype}")

    # Plot 5: Ball distance by BallType (trimmed time) - SIDE-BY-SIDE BY LIGHT
    # Create side-by-side subplots for light conditions
    fig5, axes5 = plt.subplots(1, len(light_conditions_trimmed), figsize=(12 * len(light_conditions_trimmed) / 2, 6))
    if len(light_conditions_trimmed) == 1:
        axes5 = [axes5]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions_trimmed):
        # Filter data for this light condition
        df_light = df_trimmed[df_trimmed[light_col] == light_condition].copy()

        create_line_plot_trimmed_time(
            data=df_light,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col=balltype_col,
            title=f"Light: {light_condition}",
            ax=axes5[idx],
            color_mapping=balltype_colors,
            x_label="Adjusted Time (seconds)",
            time_bins=0.1,
        )

    # Add overall title
    fig5.suptitle("Ball Distance Over Trimmed Adjusted Time by BallType", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path_trimmed_balltype = Path(__file__).parent / "dark_olfaction_coordinates_trimmed_time_balltype.png"
    plt.savefig(output_path_trimmed_balltype, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (BallType, side-by-side) saved to: {output_path_trimmed_balltype}")
    plt.savefig(output_path_trimmed_balltype, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (BallType) saved to: {output_path_trimmed_balltype}")

    # Plot 6: Ball distance by combined grouping (trimmed time) - SIDE-BY-SIDE BY LIGHT
    # Get unique light conditions
    light_conditions_trimmed = sorted(df_trimmed[light_col].unique())

    # Create side-by-side subplots for light conditions
    fig6, axes6 = plt.subplots(1, len(light_conditions_trimmed), figsize=(14 * len(light_conditions_trimmed) / 2, 6))
    if len(light_conditions_trimmed) == 1:
        axes6 = [axes6]  # Make it iterable if only one subplot

    for idx, light_condition in enumerate(light_conditions_trimmed):
        # Filter data for this light condition
        df_light_trimmed = df_trimmed[df_trimmed[light_col] == light_condition].copy()

        create_line_plot_trimmed_time(
            data=df_light_trimmed,
            x_col=time_col,
            y_col=ball_distance_col,
            hue_col="Combined_Group",
            title=f"Light: {light_condition}",
            ax=axes6[idx],
            color_mapping=combined_colors,
            linestyle_mapping=combined_linestyles,
            x_label="Adjusted Time (seconds)",
            time_bins=0.1,
        )

    # Add overall title
    fig6.suptitle(
        "Ball Distance Over Trimmed Adjusted Time by Genotype × BallType", fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    output_path_trimmed_combined = Path(__file__).parent / "dark_olfaction_coordinates_trimmed_time_combined.png"
    plt.savefig(output_path_trimmed_combined, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (Combined, side-by-side) saved to: {output_path_trimmed_combined}")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS FOR BOTH APPROACHES")
    print("=" * 80)

    print(f"\nPercentage-based approach:")
    print(f"  - Data normalized to 0-100% per fly")
    print(f"  - Final dataset: {df_percentage.shape[0]} data points from {df_percentage['fly'].nunique()} flies")
    print(f"  - Time range: {df_percentage[time_col].min():.1f}% to {df_percentage[time_col].max():.1f}%")

    print(f"\nTrimmed time approach:")
    print(f"  - Maximum shared time: {max_shared_time:.2f} seconds")
    print(f"  - Final dataset: {df_trimmed.shape[0]} data points from {df_trimmed['fly'].nunique()} flies")
    print(f"  - Time range: {df_trimmed[time_col].min():.2f}s to {df_trimmed[time_col].max():.2f}s")

    # Statistics for different time windows (using trimmed data)
    time_windows = [(-5, 0), (0, 5), (5, 15), (15, max_shared_time)]
    window_names = [
        "Pre-exit (-5 to 0s)",
        "Early post-exit (0 to 5s)",
        "Mid post-exit (5 to 15s)",
        f"Late post-exit (15 to {max_shared_time:.1f}s)",
    ]

    print(f"\nTime window analysis (using trimmed time data):")
    for (start, end), window_name in zip(time_windows, window_names):
        print(f"\n{window_name}:")
        window_data = filter_data_by_time_window(df_trimmed, time_col, start, end)

        if not window_data.empty:
            # Ball distance statistics by genotype
            genotype_stats = (
                window_data.groupby(genotype_col)[ball_distance_col].agg(["count", "mean", "std", "median"]).round(3)
            )
            print(f"Ball distance by {genotype_col}:")
            print(genotype_stats)

            # Ball distance statistics by balltype
            balltype_stats = (
                window_data.groupby(balltype_col)[ball_distance_col].agg(["count", "mean", "std", "median"]).round(3)
            )
            print(f"\nBall distance by {balltype_col}:")
            print(balltype_stats)
        else:
            print(f"  No data in this window")

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("Two visualization approaches created:")
    print("1. Percentage-based: Normalizes each fly's time to 0-100%, allows comparison of relative progression")
    print("2. Trimmed time: Uses actual seconds but only up to maximum shared time, preserves absolute timing")

    print(f"\nGrouping factors:")
    print(f"  - Light: {sorted(df_clean[light_col].unique())}")
    print(f"  - Genotype: {sorted(df_clean[genotype_col].unique())}")
    print(f"  - BallType: {sorted(df_clean[balltype_col].unique())}")

    print(f"\nFiles saved:")
    print(f"  Percentage-based:")
    print(f"    - {output_path_percentage_genotype}")
    print(f"    - {output_path_percentage_balltype}")
    print(f"    - {output_path_percentage_combined} (side-by-side comparison)")
    print(f"  Trimmed time:")
    print(f"    - {output_path_trimmed_genotype}")
    print(f"    - {output_path_trimmed_balltype}")
    print(f"    - {output_path_trimmed_combined} (side-by-side comparison)")

    print(f"\n✅ All plots generated successfully!")

    return df_percentage, df_trimmed


if __name__ == "__main__":
    data = main()
