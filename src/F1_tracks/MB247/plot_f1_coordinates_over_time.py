#!/usr/bin/env python3
"""
Script to plot F1 coordinates over adjusted time with two approaches:
1. Percentage-based normalization (0-100%) - normalizes each fly's time individually
2. Trimmed time approach - uses actual seconds but only up to max time all flies share
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
from scipy import stats

# Add path to import Config for brain region mappings
sys.path.append("/home/matthias/ballpushing_utils/src")
try:
    from PCA import Config

    # Load brain region mappings
    nickname_to_brainregion = dict(zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"]))
    brain_region_color_dict = Config.color_dict
    HAS_BRAIN_REGIONS = True
    print(f"✅ Loaded brain region mappings: {len(nickname_to_brainregion)} genotypes")
except Exception as e:
    print(f"⚠️  Could not load brain region mappings from Config: {e}")
    nickname_to_brainregion = {}
    brain_region_color_dict = {}
    HAS_BRAIN_REGIONS = False


def get_brain_region_for_genotype(genotype):
    """Get brain region for a genotype, with manual mapping for MB247 dataset."""

    # Manual mapping for MB247 dataset genotypes
    manual_mapping = {
        "TNTxMB247": "MB",
        "TNTxEmptyGal4": "Control",
    }

    if genotype in manual_mapping:
        return manual_mapping[genotype]

    if not HAS_BRAIN_REGIONS:
        return "Unknown"

    # Direct lookup from Config
    if genotype in nickname_to_brainregion:
        return nickname_to_brainregion[genotype]

    # Try common variations
    variations = [
        genotype.replace("TNTx", ""),  # Remove TNTx prefix
        genotype.replace("x", ""),  # Remove x's
        genotype.replace("-", ""),  # Remove hyphens
        genotype.split("x")[-1] if "x" in genotype else genotype,  # Take last part after x
    ]

    for variation in variations:
        if variation in nickname_to_brainregion:
            return nickname_to_brainregion[variation]

    return "Unknown"


def get_color_for_brain_region(brain_region):
    """Get color for a brain region."""
    if not HAS_BRAIN_REGIONS or brain_region not in brain_region_color_dict:
        # Fallback colors if brain region mapping not available
        fallback_colors = {
            "Unknown": "#808080",  # Gray
            "MB": "#1f77b4",  # Blue
            "EmptyGal4": "#ff7f0e",  # Orange
        }
        return fallback_colors.get(brain_region, "#808080")

    return brain_region_color_dict[brain_region]


def create_genotype_color_linestyle_mapping(genotypes, pretraining_vals):
    """
    Create mapping of (genotype, pretraining) -> (color, linestyle).
    Same brain region gets same color, pretraining determines linestyle.
    """
    # Map each genotype to its brain region and color
    genotype_to_region = {}
    genotype_to_color = {}

    for genotype in genotypes:
        brain_region = get_brain_region_for_genotype(genotype)
        color = get_color_for_brain_region(brain_region)
        genotype_to_region[genotype] = brain_region
        genotype_to_color[genotype] = color

    # Create combined mapping
    style_mapping = {}
    linestyle_map = {"n": ":", "y": "-"}  # dotted for 'n', solid for 'y'

    for genotype in genotypes:
        for pretraining in pretraining_vals:
            key = f"{pretraining} + {genotype}"
            style_mapping[key] = {
                "color": genotype_to_color[genotype],
                "linestyle": linestyle_map.get(pretraining, "-"),
                "brain_region": genotype_to_region[genotype],
            }

    return style_mapping


def filter_data_by_time_window(data, time_col, start_time=-10, end_time=30):
    """Filter data to a specific time window."""
    return data[(data[time_col] >= start_time) & (data[time_col] <= end_time)]


def create_line_plot(data, x_col, y_cols, hue_col, title, ax, color_palette=None, style_mapping=None):
    """Create a line plot with binned data and confidence intervals."""

    # Use all the data - no time window filtering
    filtered_data = data

    if filtered_data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Create color palette if not provided and no style mapping
    if color_palette is None and style_mapping is None:
        unique_hues = filtered_data[hue_col].unique()
        color_palette = sns.color_palette("Set1", n_colors=len(unique_hues))
        color_dict = dict(zip(unique_hues, color_palette))
    elif color_palette is not None and style_mapping is None:
        unique_hues = filtered_data[hue_col].unique()
        color_dict = dict(zip(unique_hues, color_palette))
    else:
        # Use style mapping for colors and linestyles
        color_dict = {}
        unique_hues = filtered_data[hue_col].unique()

    # Plot each y column
    for i, y_col in enumerate(y_cols):
        if y_col not in filtered_data.columns:
            print(f"Warning: Column '{y_col}' not found in data")
            continue

        # Plot for each hue value
        for hue_val in unique_hues:
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

            # Plot settings - use style mapping if available
            if style_mapping and hue_val in style_mapping:
                style_info = style_mapping[hue_val]
                color = style_info["color"]
                linestyle = style_info["linestyle"]
                brain_region = style_info.get("brain_region", "Unknown")
                label = f"{y_col.replace('_', ' ').title()} ({hue_val}) [{brain_region}]"
            else:
                # Fallback to original style
                color = color_dict.get(hue_val, "#808080")
                linestyle = "-" if i == 0 else "--"
                label = f"{y_col.replace('_', ' ').title()} ({hue_val})"

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
        for col in ["fly", "Pretraining", "Genotype"]:
            if col in fly_data.columns:
                fly_normalized[col] = fly_data[col].iloc[0]  # Use first value for categorical data

        # Interpolate ball distances to standard time grid
        valid_indices = valid_mask
        for ball_col in ["training_ball_euclidean_distance", "test_ball_euclidean_distance"]:
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
                "training_ball_euclidean_distance",
                "test_ball_euclidean_distance",
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
    color_palette=None,
    style_mapping=None,
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
        color_palette: Color palette for different hue values
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
    if color_palette is not None:
        color_dict = dict(zip(data[hue_col].unique(), color_palette))
    else:
        color_dict = None

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

            # Plot settings - use style mapping if available
            if style_mapping and hue_val in style_mapping:
                style_info = style_mapping[hue_val]
                color = style_info["color"]
                linestyle = style_info["linestyle"]
                brain_region = style_info.get("brain_region", "Unknown")
                label = f"{hue_val} - {y_col} [{brain_region}]" if len(y_cols) > 1 else f"{hue_val} [{brain_region}]"
            else:
                # Fallback to original style
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

    # Dataset path - MB247 F1 coordinates dataset
    dataset_path = "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251015_17_F1_coordinates_F1_TNT_MB247_Data/F1_coordinates/pooled_F1_coordinates.feather"

    # Alternative paths to try if the main one doesn't exist
    alternative_paths = [
        # Add more potential paths as needed
    ]

    # Try to load dataset from multiple possible paths
    df = None
    used_path = None

    for path in [dataset_path] + alternative_paths:
        try:
            if Path(path).exists():
                df = pd.read_feather(path)
                used_path = path
                print(f"Dataset loaded successfully from: {path}")
                print(f"Dataset shape: {df.shape}")
                break
        except Exception as e:
            print(f"Could not load dataset from {path}: {e}")
            continue

    if df is None:
        print("Could not load dataset from any of the specified paths.")
        print("Please check the dataset path and ensure the F1_coordinates dataset exists.")
        print("You may need to generate it first using the Dataset class with metrics='F1_coordinates'")
        return

    # Filter out specific dates if needed (based on existing scripts)
    if "Date" in df.columns:
        initial_shape = df.shape
        df = df[df["Date"] != "250904"]
        print(f"Removed data for Date 250904. Shape changed from {initial_shape} to {df.shape}.")

    # Store original data before any transformations
    original_df = df.copy()

    # =============================================================================
    # OUTLIER DETECTION AND EXCLUSION (BEFORE NORMALIZATION)
    # =============================================================================
    print(f"\n{'='*60}")
    print("OUTLIER DETECTION AND EXCLUSION (BEFORE NORMALIZATION)")
    print(f"{'='*60}")

    # Identify required columns for outlier detection
    time_col = "adjusted_time"
    training_ball_col = "training_ball_euclidean_distance"
    test_ball_col = "test_ball_euclidean_distance"
    pretraining_col = None
    genotype_col = None

    # Look for pretraining column
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    # Look for genotype column
    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
            break

    # Essential columns for outlier detection
    essential_cols = [time_col, test_ball_col, pretraining_col, genotype_col]

    # Clean data for outlier detection
    df_for_outlier_detection = df[essential_cols + ["fly"]].copy()
    df_for_outlier_detection = df_for_outlier_detection.dropna(subset=essential_cols)
    df_for_outlier_detection["combined_group"] = (
        df_for_outlier_detection[pretraining_col].astype(str)
        + " + "
        + df_for_outlier_detection[genotype_col].astype(str)
    )

    # Calculate outlier threshold
    overall_stats = df_for_outlier_detection[test_ball_col].describe()
    Q1 = overall_stats["25%"]
    Q3 = overall_stats["75%"]
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    print(f"Overall test ball distance statistics:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Outlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.2f}")

    # Find outlier flies
    outlier_flies_to_exclude = set()
    for group in sorted(df_for_outlier_detection["combined_group"].unique()):
        group_data = df_for_outlier_detection[df_for_outlier_detection["combined_group"] == group]
        outliers = group_data[group_data[test_ball_col] > outlier_threshold]

        if not outliers.empty:
            outlier_flies = outliers["fly"].unique() if "fly" in outliers.columns else []
            print(f"\n{group}: {len(outliers)} outlier data points (>{outlier_threshold:.0f})")
            print(f"  Outlier flies: {list(outlier_flies)}")
            outlier_flies_to_exclude.update(outlier_flies)
        else:
            print(f"\n{group}: No outliers detected")

    # Exclude outlier flies from original dataset BEFORE normalization
    if outlier_flies_to_exclude:
        print(f"\n{'='*40}")
        print(f"EXCLUDING {len(outlier_flies_to_exclude)} OUTLIER FLIES BEFORE NORMALIZATION")
        print(f"{'='*40}")
        print(f"Outlier flies: {list(outlier_flies_to_exclude)}")

        original_shape = df.shape
        df = df[~df["fly"].isin(outlier_flies_to_exclude)]
        print(f"Dataset after outlier removal: {original_shape} -> {df.shape}")

        # Also filter original_df for later use
        original_df = df.copy()
    else:
        print(f"\nNo outlier flies to exclude.")

    # Transform adjusted_time to normalized percentage (0-100%) with standardized intervals AFTER outlier removal
    print("\\nTransforming adjusted_time to normalized percentages (after outlier removal)...")

    # DEBUG: Check data before normalization
    print(f"DEBUG - Before normalization:")
    print(f"  Shape: {df.shape}")
    print(
        f"  Distance range: {df['test_ball_euclidean_distance'].min():.2f} to {df['test_ball_euclidean_distance'].max():.2f}"
    )
    print(f"  Distance std: {df['test_ball_euclidean_distance'].std():.2f}")

    df_percentage = normalize_adjusted_time_to_percentage(df)

    # DEBUG: Check data after normalization
    print(f"DEBUG - After normalization:")
    print(f"  Shape: {df_percentage.shape}")
    if "test_ball_euclidean_distance" in df_percentage.columns:
        print(
            f"  Distance range: {df_percentage['test_ball_euclidean_distance'].min():.2f} to {df_percentage['test_ball_euclidean_distance'].max():.2f}"
        )
        print(f"  Distance std: {df_percentage['test_ball_euclidean_distance'].std():.2f}")

    print(
        f"After normalization: adjusted_time range {df_percentage['adjusted_time'].min():.1f}% to {df_percentage['adjusted_time'].max():.1f}%"
    )

    # Print available columns to help identify the correct ones
    print("\\nAvailable columns:")
    for i, col in enumerate(df_percentage.columns):
        print(f"  {i+1:2d}. {col}")

    # Clean the percentage data
    df_percentage_clean = (
        df_percentage[essential_cols + [training_ball_col, "fly"]].copy()
        if "fly" in df_percentage.columns
        else df_percentage[essential_cols + [training_ball_col]].copy()
    )

    # Only drop rows where essential columns are NaN
    df_percentage_clean = df_percentage_clean.dropna(subset=essential_cols)

    print(f"\nPercentage dataset shape after cleaning: {df_percentage_clean.shape}")

    # Create combined group variable
    df_percentage_clean["combined_group"] = (
        df_percentage_clean[pretraining_col].astype(str) + " + " + df_percentage_clean[genotype_col].astype(str)
    )

    # Print unique values
    print(f"Unique {pretraining_col} values: {df_percentage_clean[pretraining_col].unique()}")
    print(f"Unique {genotype_col} values: {df_percentage_clean[genotype_col].unique()}")

    # Print time range for percentage data
    print(
        f"Percentage time range: {df_percentage_clean[time_col].min():.2f}% to {df_percentage_clean[time_col].max():.2f}%"
    )

    # =============================================================================
    # OUTLIER DETECTION AND EXCLUSION (BEFORE PLOTTING)
    # =============================================================================
    print(f"\n{'='*60}")
    print("OUTLIER DETECTION AND EXCLUSION")
    print(f"{'='*60}")

    # Use original data for outlier detection
    original_df_clean = original_df[essential_cols + [training_ball_col, "fly"]].copy()
    original_df_clean = original_df_clean.dropna(subset=essential_cols)
    original_df_clean["combined_group"] = (
        original_df_clean[pretraining_col].astype(str) + " + " + original_df_clean[genotype_col].astype(str)
    )

    # Calculate outlier threshold
    overall_stats = original_df_clean[test_ball_col].describe()
    Q1 = overall_stats["25%"]
    Q3 = overall_stats["75%"]
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    print(f"Overall test ball distance statistics:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Outlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.2f}")

    # Find outlier flies
    outlier_flies_to_exclude = set()
    for group in sorted(original_df_clean["combined_group"].unique()):
        group_data = original_df_clean[original_df_clean["combined_group"] == group]
        outliers = group_data[group_data[test_ball_col] > outlier_threshold]

        if not outliers.empty:
            outlier_flies = outliers["fly"].unique() if "fly" in outliers.columns else []
            print(f"\n{group}: {len(outliers)} outlier data points (>{outlier_threshold:.0f})")
            print(f"  Outlier flies: {list(outlier_flies)}")
            outlier_flies_to_exclude.update(outlier_flies)
        else:
            print(f"\n{group}: No outliers detected")

    # Exclude outlier flies from datasets BEFORE plotting
    if outlier_flies_to_exclude:
        print(f"\n{'='*40}")
        print(f"EXCLUDING {len(outlier_flies_to_exclude)} OUTLIER FLIES FROM ALL ANALYSES")
        print(f"{'='*40}")
        print(f"Outlier flies: {list(outlier_flies_to_exclude)}")

        # Filter percentage data
        original_percentage_shape = df_percentage_clean.shape
        df_percentage_clean = df_percentage_clean[~df_percentage_clean["fly"].isin(outlier_flies_to_exclude)]
        print(f"Percentage data: {original_percentage_shape} -> {df_percentage_clean.shape}")

        # Filter original data
        original_original_shape = original_df_clean.shape
        original_df_clean = original_df_clean[~original_df_clean["fly"].isin(outlier_flies_to_exclude)]
        print(f"Original data: {original_original_shape} -> {original_df_clean.shape}")

        # Recalculate combined groups after filtering
        df_percentage_clean["combined_group"] = (
            df_percentage_clean[pretraining_col].astype(str) + " + " + df_percentage_clean[genotype_col].astype(str)
        )
        original_df_clean["combined_group"] = (
            original_df_clean[pretraining_col].astype(str) + " + " + original_df_clean[genotype_col].astype(str)
        )

        print(f"\nUpdated group statistics after outlier removal:")
        for group in sorted(df_percentage_clean["combined_group"].unique()):
            group_data = df_percentage_clean[df_percentage_clean["combined_group"] == group]
            n_flies = group_data["fly"].nunique() if "fly" in group_data.columns else len(group_data)
            print(f"  {group}: {n_flies} flies")
    else:
        print(f"\nNo outlier flies to exclude.")

    # =============================================================================
    # DIAGNOSTIC: DISTANCES AFTER OUTLIER REMOVAL
    # =============================================================================
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: DISTANCES AFTER OUTLIER REMOVAL")
    print(f"{'='*60}")

    # Check initial distances after filtering
    initial_time_data = df_percentage_clean[df_percentage_clean[time_col] <= 10.0]  # First 10% of time
    print(f"\nInitial distances (0-10% time) by combined group after filtering:")

    for group in sorted(df_percentage_clean["combined_group"].unique()):
        group_data = initial_time_data[initial_time_data["combined_group"] == group]

        if not group_data.empty:
            test_ball_stats = group_data[test_ball_col].describe()
            n_flies = group_data["fly"].nunique() if "fly" in group_data.columns else len(group_data)

            print(f"\n{group} (n_flies={n_flies}, n_points={len(group_data)}):")
            print(f"  Test ball distance - Mean: {test_ball_stats['mean']:.2f}, Std: {test_ball_stats['std']:.2f}")
            print(f"  Test ball distance - Median: {test_ball_stats['50%']:.2f}")
        else:
            print(f"\n{group}: No data in initial time period")

    # =============================================================================
    # DIAGNOSTIC: CHECK INITIAL DISTANCES FOR EACH GROUP
    # =============================================================================
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: INITIAL EUCLIDEAN DISTANCES BY GROUP")
    print(f"{'='*60}")

    # Create combined group variable for diagnostics
    df_percentage_clean["combined_group"] = (
        df_percentage_clean[pretraining_col].astype(str) + " + " + df_percentage_clean[genotype_col].astype(str)
    )

    # Check initial distances (first 10% of time) for each group
    initial_time_data = df_percentage_clean[df_percentage_clean[time_col] <= 10.0]  # First 10% of time

    print(f"\nInitial distances (0-10% time) by combined group:")
    print(f"Data points in initial period: {len(initial_time_data)}")

    for group in sorted(df_percentage_clean["combined_group"].unique()):
        group_data = initial_time_data[initial_time_data["combined_group"] == group]

        if not group_data.empty:
            test_ball_stats = group_data[test_ball_col].describe()
            n_flies = group_data["fly"].nunique() if "fly" in group_data.columns else len(group_data)

            print(f"\n{group} (n_flies={n_flies}, n_points={len(group_data)}):")
            print(f"  Test ball distance - Mean: {test_ball_stats['mean']:.2f}, Std: {test_ball_stats['std']:.2f}")
            print(f"  Test ball distance - Min: {test_ball_stats['min']:.2f}, Max: {test_ball_stats['max']:.2f}")
            print(f"  Test ball distance - Median: {test_ball_stats['50%']:.2f}")
        else:
            print(f"\n{group}: No data in initial time period")

    # Also check for the original (non-percentage) data at time 0 area
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: DISTANCES AROUND TIME=0 (ORIGINAL TIME SCALE)")
    print(f"{'='*60}")

    # Use original data around time 0
    original_df_clean = original_df[essential_cols + [training_ball_col, "fly"]].copy()
    original_df_clean = original_df_clean.dropna(subset=essential_cols)
    original_df_clean["combined_group"] = (
        original_df_clean[pretraining_col].astype(str) + " + " + original_df_clean[genotype_col].astype(str)
    )

    # Check distances around time 0 (±2 seconds)
    time_zero_data = original_df_clean[(original_df_clean[time_col] >= -2.0) & (original_df_clean[time_col] <= 2.0)]

    print(f"\nDistances around time=0 (±2 seconds) by combined group:")
    print(f"Data points around time=0: {len(time_zero_data)}")

    for group in sorted(original_df_clean["combined_group"].unique()):
        group_data = time_zero_data[time_zero_data["combined_group"] == group]

        if not group_data.empty:
            test_ball_stats = group_data[test_ball_col].describe()
            n_flies = group_data["fly"].nunique() if "fly" in group_data.columns else len(group_data)

            print(f"\n{group} (n_flies={n_flies}, n_points={len(group_data)}):")
            print(f"  Test ball distance - Mean: {test_ball_stats['mean']:.2f}, Std: {test_ball_stats['std']:.2f}")
            print(f"  Test ball distance - Min: {test_ball_stats['min']:.2f}, Max: {test_ball_stats['max']:.2f}")
            print(f"  Test ball distance - Median: {test_ball_stats['50%']:.2f}")

            # Show time distribution for this group
            time_stats = group_data[time_col].describe()
            print(f"  Time range: {time_stats['min']:.2f} to {time_stats['max']:.2f} seconds")
        else:
            print(f"\n{group}: No data around time=0")

    # Check for any extreme outliers
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: EXTREME VALUES CHECK")
    print(f"{'='*60}")

    overall_stats = original_df_clean[test_ball_col].describe()
    print(f"\nOverall test ball distance statistics:")
    print(f"  Mean: {overall_stats['mean']:.2f}, Std: {overall_stats['std']:.2f}")
    print(f"  Min: {overall_stats['min']:.2f}, Max: {overall_stats['max']:.2f}")
    print(f"  Q1: {overall_stats['25%']:.2f}, Median: {overall_stats['50%']:.2f}, Q3: {overall_stats['75%']:.2f}")

    # Find potential outliers (values > Q3 + 1.5*IQR)
    Q1 = overall_stats["25%"]
    Q3 = overall_stats["75%"]
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    print(f"\nOutlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.2f}")

    outlier_flies_to_exclude = set()

    for group in sorted(original_df_clean["combined_group"].unique()):
        group_data = original_df_clean[original_df_clean["combined_group"] == group]
        outliers = group_data[group_data[test_ball_col] > outlier_threshold]

        if not outliers.empty:
            print(f"\n{group}: {len(outliers)} outlier data points (>{outlier_threshold:.0f})")
            print(f"  Outlier values range: {outliers[test_ball_col].min():.2f} to {outliers[test_ball_col].max():.2f}")
            outlier_flies = outliers["fly"].unique() if "fly" in outliers.columns else []
            print(f"  Outlier flies: {len(outlier_flies)} flies")
            print(f"  Fly names: {list(outlier_flies)}")

            # Add outlier flies to exclusion set
            outlier_flies_to_exclude.update(outlier_flies)
        else:
            print(f"\n{group}: No outliers detected")

    # Exclude outlier flies from all datasets
    if outlier_flies_to_exclude:
        print(f"\n{'='*60}")
        print(f"EXCLUDING {len(outlier_flies_to_exclude)} OUTLIER FLIES")
        print(f"{'='*60}")
        print(f"Outlier flies to exclude: {list(outlier_flies_to_exclude)}")

        # Filter percentage data
        original_percentage_shape = df_percentage_clean.shape
        df_percentage_clean = df_percentage_clean[~df_percentage_clean["fly"].isin(outlier_flies_to_exclude)]
        print(f"Percentage data: {original_percentage_shape} -> {df_percentage_clean.shape}")

        # Filter original data
        original_original_shape = original_df_clean.shape
        original_df_clean = original_df_clean[~original_df_clean["fly"].isin(outlier_flies_to_exclude)]
        print(f"Original data: {original_original_shape} -> {original_df_clean.shape}")

        # Recalculate combined groups after filtering
        df_percentage_clean["combined_group"] = (
            df_percentage_clean[pretraining_col].astype(str) + " + " + df_percentage_clean[genotype_col].astype(str)
        )
        original_df_clean["combined_group"] = (
            original_df_clean[pretraining_col].astype(str) + " + " + original_df_clean[genotype_col].astype(str)
        )

        # Print updated group statistics
        print(f"\nUpdated group statistics after outlier removal:")
        for group in sorted(df_percentage_clean["combined_group"].unique()):
            group_data = df_percentage_clean[df_percentage_clean["combined_group"] == group]
            n_flies = group_data["fly"].nunique() if "fly" in group_data.columns else len(group_data)
            print(f"  {group}: {n_flies} flies, {len(group_data)} data points")
    else:
        print(f"\nNo outlier flies to exclude.")

    # =============================================================================
    # IMPLEMENTATION 1: PERCENTAGE-BASED NORMALIZATION (0-100%) - FILTERED DATA
    # =============================================================================
    print(f"\n{'='*60}")
    print("COMBINED PRETRAINING + GENOTYPE ANALYSIS - FILTERED DATA")
    print(f"{'='*60}")

    # Create figure for combined plots
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))

    # Create brain region-based style mapping for combined groups
    unique_genotypes = df_percentage_clean[genotype_col].unique()
    unique_pretraining = df_percentage_clean[pretraining_col].unique()
    style_mapping = create_genotype_color_linestyle_mapping(unique_genotypes, unique_pretraining)

    # Debug: Print style mapping
    if HAS_BRAIN_REGIONS:
        print(f"\n🎨 Brain region-based styling:")
        for combined_group, style_info in style_mapping.items():
            print(
                f"  {combined_group}: {style_info['brain_region']} (color: {style_info['color']}, line: {style_info['linestyle']})"
            )
    else:
        print(f"\n⚠️  Using fallback colors (brain region mapping not available)")

    # Set up color palette for combined groups (fallback if style mapping fails)
    n_combined_groups = len(df_percentage_clean["combined_group"].unique())
    combined_palette = sns.color_palette("Set1", n_colors=n_combined_groups)

    # DEBUG: Check data before plotting
    print(f"\nDEBUG - Data passed to percentage plotting:")
    print(f"  Shape: {df_percentage_clean.shape}")
    print(f"  Time range: {df_percentage_clean[time_col].min():.1f}% to {df_percentage_clean[time_col].max():.1f}%")
    print(f"  Unique time values: {len(df_percentage_clean[time_col].unique())}")
    print(
        f"  Distance range: {df_percentage_clean[test_ball_col].min():.2f} to {df_percentage_clean[test_ball_col].max():.2f}"
    )
    print(f"  Distance std: {df_percentage_clean[test_ball_col].std():.2f}")

    # Check variation by group
    for group in sorted(df_percentage_clean["combined_group"].unique()):
        group_data = df_percentage_clean[df_percentage_clean["combined_group"] == group]
        print(f"  {group}: {len(group_data)} points, distance std: {group_data[test_ball_col].std():.2f}")

    # Plot: Test ball only by Combined Pretraining + Genotype (percentage-based)
    create_line_plot(
        data=df_percentage_clean,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col="combined_group",
        title="Test Ball Distance Over Normalized Time (%) by Pretraining + Genotype",
        ax=ax1,
        color_palette=combined_palette,
        style_mapping=style_mapping,
    )

    # Format and save percentage-based plots
    plt.tight_layout()
    output_path_percentage = Path(__file__).parent / "f1_coordinates_percentage_time.png"
    plt.savefig(output_path_percentage, dpi=300, bbox_inches="tight")
    print(f"Percentage-based plots saved to: {output_path_percentage}")

    # =============================================================================
    # IMPLEMENTATION 2: TRIMMED TIME APPROACH (ACTUAL SECONDS) - FILTERED DATA
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 2: TRIMMED TIME APPROACH - FILTERED DATA")
    print(f"{'='*60}")

    # Use the already filtered original_df_clean from outlier detection
    print(f"Filtered data shape for trimmed approach: {original_df_clean.shape}")
    print(f"Time range: {original_df_clean[time_col].min():.2f} to {original_df_clean[time_col].max():.2f} seconds")

    # Find maximum shared time across all flies (using filtered data)
    max_shared_time = find_maximum_shared_time(original_df_clean, "fly", time_col)

    # Trim data to shared time
    df_trimmed = trim_data_to_shared_time(original_df_clean, max_shared_time, time_col)

    # Create figure for trimmed time plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))

    # Plot: Test ball only by Combined Pretraining + Genotype (trimmed time)
    create_line_plot_trimmed_time(
        data=df_trimmed,
        x_col=time_col,
        y_cols=[test_ball_col],
        hue_col="combined_group",
        title="Test Ball Distance Over Trimmed Adjusted Time by Pretraining + Genotype",
        ax=ax2,
        color_palette=combined_palette,
        style_mapping=style_mapping,
        x_label="Adjusted Time (seconds)",
        time_bins=0.1,
    )

    # Format and save trimmed time plots
    plt.tight_layout()
    output_path_trimmed = Path(__file__).parent / "f1_coordinates_trimmed_time.png"
    plt.savefig(output_path_trimmed, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots saved to: {output_path_trimmed}")

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

        # Ensure combined_group is available in window_data
        if "combined_group" not in window_data.columns:
            window_data["combined_group"] = (
                window_data[pretraining_col].astype(str) + " + " + window_data[genotype_col].astype(str)
            )

        if not window_data.empty:
            # Test ball statistics by pretraining
            test_stats = (
                window_data.groupby("combined_group")[test_ball_col].agg(["count", "mean", "std", "median"]).round(3)
            )
            print(f"Test ball by Combined Pretraining + Genotype:")
            print(test_stats)
        else:
            print("No data in this time window")

    print(f"\\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("Two visualization approaches created:")
    print("1. Percentage-based: Normalizes each fly's time to 0-100%, allows comparison of relative progression")
    print("2. Trimmed time: Uses actual seconds but only up to maximum shared time, preserves absolute timing")
    print(f"\\nFiles saved:")
    print(f"  - {output_path_percentage}")
    print(f"  - {output_path_trimmed}")

    plt.show()

    return df_percentage_clean, df_trimmed


if __name__ == "__main__":
    data = main()
