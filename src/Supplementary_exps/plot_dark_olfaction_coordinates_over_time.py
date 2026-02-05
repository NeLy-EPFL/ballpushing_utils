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

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def filter_data_by_time_window(data, time_col, start_time=-10, end_time=30):
    """Filter data to a specific time window."""
    return data[(data[time_col] >= start_time) & (data[time_col] <= end_time)]


def downsample_to_seconds(data, time_col, value_col, group_cols, subject_col="fly"):
    """
    Downsample data to 1 datapoint per second per fly by rounding time to nearest second.

    This mirrors the alignment approach used in light trajectory plots.
    """
    if data.empty:
        return data

    data = data.sort_values([subject_col, time_col]).copy()
    data["time_rounded"] = data[time_col].round(0).astype(int)

    agg_map = {value_col: "mean"}
    for col in group_cols:
        if col in data.columns:
            agg_map[col] = "first"

    downsampled = data.groupby([subject_col, "time_rounded"], as_index=False).agg(agg_map)
    downsampled = downsampled.rename(columns={"time_rounded": time_col})

    return downsampled


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
    # ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time (0%)")

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
    # ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time (0s)")

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


def preprocess_data_for_permutation(
    data, time_col="adjusted_time", value_col="distance_ball_0", group_col="Genotype", subject_col="fly", n_bins=12
):
    """
    Preprocess data by binning time and computing statistics per bin.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw trajectory data
    time_col : str
        Column name for time
    value_col : str
        Column name for distance/position values
    group_col : str
        Column name for grouping (Genotype)
    subject_col : str
        Column name for individual subjects (flies)
    n_bins : int
        Number of time bins

    Returns:
    --------
    pd.DataFrame
        Processed data with time bins and statistics
    """
    # Create time bins
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    # Assign bins
    data = data.copy()
    data["time_bin"] = pd.cut(data[time_col], bins=bin_edges, labels=range(n_bins), include_lowest=True)
    data["time_bin"] = data["time_bin"].astype(int)

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute statistics per bin, group, and subject
    grouped = data.groupby([group_col, subject_col, "time_bin"])[value_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{value_col}"]

    # Add bin centers and edges
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))

    return grouped, bin_edges


def compute_permutation_test_genotype(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="Genotype",
    control_group="TNTxEmptyGal4",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Compute permutation tests for each time bin comparing two genotypes.

    Parameters:
    -----------
    processed_data : pd.DataFrame
        Preprocessed data with time bins
    metric : str
        Column name for the metric to test
    group_col : str
        Column name for grouping
    control_group : str
        Name of the control group
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level
    progress : bool
        Show progress bar

    Returns:
    --------
    dict
        Dictionary with test results
    """
    # Get all groups
    groups = processed_data[group_col].unique()

    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups for genotype analysis, found {len(groups)}: {groups}")

    if control_group not in groups:
        raise ValueError(f"Control group '{control_group}' not found in data")

    test_group = [g for g in groups if g != control_group][0]

    print(f"\n  Testing {test_group} vs {control_group}...")

    # Get all time bins
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

    # Store results for each bin
    observed_diffs = []
    p_values = []

    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    for time_bin in iterator:
        bin_data = processed_data[processed_data["time_bin"] == time_bin]

        # Get control and test values
        control_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        if len(control_vals) == 0 or len(test_vals) == 0:
            observed_diffs.append(np.nan)
            p_values.append(1.0)
            continue

        # Observed difference
        obs_diff = np.mean(test_vals) - np.mean(control_vals)
        observed_diffs.append(obs_diff)

        # Permutation test
        combined = np.concatenate([control_vals, test_vals])
        n_control = len(control_vals)
        n_test = len(test_vals)

        perm_diffs = []
        for _ in range(n_permutations):
            # Shuffle and split
            np.random.shuffle(combined)
            perm_control = combined[:n_control]
            perm_test = combined[n_control:]
            perm_diff = np.mean(perm_test) - np.mean(perm_control)
            perm_diffs.append(perm_diff)

        perm_diffs = np.array(perm_diffs)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
        p_values.append(p_value)

    # Apply FDR correction across all time bins
    p_values_raw = np.array(p_values)
    rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

    results = {
        "test_group": test_group,
        "control_group": control_group,
        "time_bins": time_bins,
        "observed_diffs": observed_diffs,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_values_corrected,
        "p_values": p_values_corrected,  # For backward compatibility with plotting code
        "significant_timepoints": np.where(rejected)[0],
        "n_significant": np.sum(rejected),
        "n_significant_raw": np.sum(p_values_raw < alpha),
    }

    print(f"    Raw significant bins: {results['n_significant_raw']}/{n_bins}")
    print(f"    FDR significant bins: {results['n_significant']}/{n_bins} (α={alpha})")

    return results


def create_trajectory_plot_with_permutation(
    data,
    time_col,
    value_col,
    group_col,
    subject_col,
    light_val,
    n_bins=12,
    permutation_results=None,
    color_mapping=None,
    ax=None,
    bin_edges=None,
):
    """
    Create a trajectory plot with permutation test significance annotations.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data for a specific light condition
    time_col : str
        Column name for time
    value_col : str
        Column name for distance/position
    group_col : str
        Column name for grouping (Genotype)
    subject_col : str
        Column name for subjects (flies)
    light_val : str
        Light condition value
    n_bins : int
        Number of time bins
    permutation_results : dict
        Results from permutation test
    color_mapping : dict
        Color mapping for groups
    ax : matplotlib axis
        Axis to plot on
    bin_edges : array
        Bin edges for time bins
    """
    # Pixel to mm conversion factor (500 pixels = 30 mm)
    PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Light: {light_val}", fontweight="bold", fontsize=14)
        return

    # Get groups
    groups = sorted(data[group_col].unique())

    if len(groups) != 2:
        ax.text(0.5, 0.5, f"Expected 2 groups, found {len(groups)}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Light: {light_val}", fontweight="bold", fontsize=14)
        return

    # Determine control and test groups
    control_group = "TNTxEmptyGal4"
    test_group = [g for g in groups if g != control_group][0]

    # Get sample sizes for labels
    n_control = data[data[group_col] == control_group][subject_col].nunique() if control_group in groups else 0
    n_test = data[data[group_col] == test_group][subject_col].nunique() if test_group in groups else 0

    # Convert to mm (don't shift individual fly data)
    data_copy = data.copy()
    data_copy[f"{value_col}_mm"] = data_copy[value_col] / PIXELS_PER_MM
    value_col_plot = f"{value_col}_mm"

    # Create time bins for averaging (finer bins for smoothness)
    time_bin_size = 0.1
    min_time = data_copy[time_col].min()
    max_time = data_copy[time_col].max()
    fine_bin_edges = np.arange(min_time, max_time + time_bin_size, time_bin_size)
    data_copy["time_bin_fine"] = pd.cut(data_copy[time_col], bins=fine_bin_edges, labels=False, include_lowest=True)
    data_copy["time_bin_center"] = data_copy["time_bin_fine"] * time_bin_size + time_bin_size / 2 + min_time

    # Compute mean trajectories to find baseline for y-axis shift
    # Same approach as light trajectories script - group by time directly
    all_means = []
    for group in groups:
        group_data = data_copy[data_copy[group_col] == group]
        # Group by time_bin_center to get mean and SEM
        time_grouped = group_data.groupby("time_bin_center")[value_col_plot].agg(["mean", "sem"]).reset_index()
        all_means.extend(time_grouped["mean"].values)

    # Find the minimum mean value to use as y-axis baseline
    y_baseline = min(all_means) if all_means else 0

    # Plot mean trajectories with error bands (no individual flies)
    for group in groups:
        group_data = data_copy[data_copy[group_col] == group]

        # Group by time bin centers to compute mean and SEM
        # Same approach as light trajectories script
        time_grouped = group_data.groupby("time_bin_center")[value_col_plot].agg(["mean", "sem"]).reset_index()

        # Determine line style and color (IR8a purple, Empty grey)
        linestyle = "dashed" if group == test_group else "solid"
        if color_mapping is not None:
            color = color_mapping.get(group, "gray")
        else:
            color = "#9467bd" if group == test_group else "#7f7f7f"
        alpha_band = 0.15 if group == test_group else 0.25

        # Plot mean trajectory - shift by baseline
        ax.plot(
            time_grouped["time_bin_center"],
            time_grouped["mean"] - y_baseline,
            linestyle=linestyle,
            color=color,
            linewidth=2.5,
            label=f"{group} (n={n_control if group == control_group else n_test})",
            zorder=10,
        )

        # Error band (SEM)
        ax.fill_between(
            time_grouped["time_bin_center"],
            time_grouped["mean"] - y_baseline - time_grouped["sem"],
            time_grouped["mean"] - y_baseline + time_grouped["sem"],
            color=color,
            alpha=alpha_band,
            zorder=5,
        )

    # Draw vertical dotted lines for permutation test bins
    if bin_edges is not None:
        for edge in bin_edges:
            ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, zorder=1)

    # Add vertical line at time 0
    # ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, linewidth=2, label="Exit time (0)", zorder=8)

    # Get current y-axis limits and extend them to make room for significance annotations
    # Compute range from shifted data
    y_max_shifted = max(all_means) - y_baseline if all_means else 0
    y_range = y_max_shifted

    # Set y-axis limits - start at 0, add modest space at top for annotations
    ax.set_ylim(0, y_max_shifted + 0.08 * y_range)

    # Annotate significance levels from permutation test
    if permutation_results is not None and bin_edges is not None:
        # Position for annotations (slightly above the top of data)
        y_annotation_stars = y_max_shifted + 0.02 * y_range
        y_annotation_pval = y_max_shifted + 0.05 * y_range

        for idx in permutation_results["significant_timepoints"]:
            if idx >= len(bin_edges) - 1:
                continue

            bin_start = bin_edges[idx]
            bin_end = bin_edges[idx + 1]
            x_pos = (bin_start + bin_end) / 2

            p_value = permutation_results["p_values"][idx]

            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = ""

            if significance:
                # Add significance stars
                ax.text(
                    x_pos,
                    y_annotation_stars,
                    significance,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="red",
                    fontweight="bold",
                    zorder=15,
                )

                # Add p-value below the stars
                ax.text(
                    x_pos,
                    y_annotation_pval,
                    f"p={p_value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="darkgray",
                    zorder=15,
                )

    # Formatting
    ax.set_xlabel(
        "Adjusted Time (%)" if "%" in str(data[time_col].max()) or data[time_col].max() > 100 else "Adjusted Time (s)",
        fontsize=12,
    )
    ax.set_ylabel("Relative ball distance (mm)", fontsize=12)
    ax.set_title(f"Light: {light_val}", fontweight="bold", fontsize=14, pad=10)
    # Legend outside plot window, top right
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    # Remove background grid - only keep the time bin vertical lines
    ax.grid(False)


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

    # Downsample to 1 datapoint per second per fly (align to integer seconds)
    group_cols = [light_col, genotype_col, balltype_col]
    df_aligned = downsample_to_seconds(
        df_clean,
        time_col=time_col,
        value_col=ball_distance_col,
        group_cols=group_cols,
        subject_col="fly",
    )

    print(f"\nDataset shape after downsampling: {df_aligned.shape}")
    print(f"Time range after downsampling: {df_aligned[time_col].min():.2f}s to {df_aligned[time_col].max():.2f}s")

    # Define color mappings for consistent styling
    genotype_colors = {"TNTxEmptyGal4": "#7f7f7f", "TNTxIR8a": "#9467bd"}
    balltype_colors = {"ctrl": "green", "sand": "brown"}
    # =============================================================================
    # IMPLEMENTATION 1: TRIMMED TIME APPROACH (ACTUAL SECONDS)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 1: TRIMMED TIME APPROACH")
    print(f"{'='*60}")

    # Use downsampled data
    original_df_clean = df_aligned.copy()

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

    # Create color mapping for combined groups
    combined_colors = {}
    for group in df_trimmed["Combined_Group"].unique():
        _, genotype, _ = group.split("_", 2)
        combined_colors[group] = genotype_colors.get(genotype, "gray")

    # Create linestyle mapping based on balltype
    combined_linestyles = {}
    for group in df_trimmed["Combined_Group"].unique():
        _, _, balltype = group.split("_", 2)
        combined_linestyles[group] = ":" if balltype == "sand" else "-"

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
    fig4.suptitle("", fontsize=16, fontweight="bold", y=1.02)

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
    fig5.suptitle("", fontsize=16, fontweight="bold", y=1.02)

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
    fig6.suptitle("", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path_trimmed_combined = Path(__file__).parent / "dark_olfaction_coordinates_trimmed_time_combined.png"
    plt.savefig(output_path_trimmed_combined, dpi=300, bbox_inches="tight")
    print(f"Trimmed time plots (Combined, side-by-side) saved to: {output_path_trimmed_combined}")

    # =============================================================================
    # IMPLEMENTATION 2: POOLED BALLTYPES WITH PERMUTATION TESTS (GENOTYPE EFFECT)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 2: POOLED BALLTYPES (Genotype Effect with Permutation Tests)")
    print(f"{'='*60}")

    # Pool across BallTypes
    pooled_data_trimmed = df_trimmed[[time_col, ball_distance_col, light_col, genotype_col, "fly"]].copy()

    print(f"Pooled BallTypes dataset shape: {pooled_data_trimmed.shape}")
    print(f"Unique genotypes: {sorted(pooled_data_trimmed[genotype_col].unique())}")

    # Run permutation tests for each Light condition
    n_permutation_bins = 12
    n_permutations = 10000

    light_conditions_pooled = sorted(pooled_data_trimmed[light_col].unique())

    # Create side-by-side subplots
    fig7, axes7 = plt.subplots(1, len(light_conditions_pooled), figsize=(10 * len(light_conditions_pooled), 6))
    if len(light_conditions_pooled) == 1:
        axes7 = [axes7]

    for idx, light_condition in enumerate(light_conditions_pooled):
        print(f"\nProcessing Light condition: {light_condition}")

        # Filter data for this light condition
        light_data = pooled_data_trimmed[pooled_data_trimmed[light_col] == light_condition].copy()

        print(f"  Data shape: {light_data.shape}")
        print(f"  Genotypes: {sorted(light_data[genotype_col].unique())}")

        # Preprocess for permutation test
        print(f"  Preprocessing data into {n_permutation_bins} time bins...")
        processed, bin_edges = preprocess_data_for_permutation(
            light_data,
            time_col=time_col,
            value_col=ball_distance_col,
            group_col=genotype_col,
            subject_col="fly",
            n_bins=n_permutation_bins,
        )

        # Compute permutation test
        print(f"  Computing permutation test ({n_permutations} permutations)...")
        perm_results = compute_permutation_test_genotype(
            processed,
            metric=f"avg_{ball_distance_col}",
            group_col=genotype_col,
            control_group="TNTxEmptyGal4",
            n_permutations=n_permutations,
            alpha=0.05,
            progress=True,
        )

        # Create trajectory plot with permutation annotations
        create_trajectory_plot_with_permutation(
            light_data,
            time_col=time_col,
            value_col=ball_distance_col,
            group_col=genotype_col,
            subject_col="fly",
            light_val=light_condition,
            n_bins=n_permutation_bins,
            permutation_results=perm_results,
            color_mapping=genotype_colors,
            ax=axes7[idx],
            bin_edges=bin_edges,
        )

    fig7.suptitle(
        "",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_dir = Path("/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Plots/Trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_pooled_genotype = output_dir / "dark_olfaction_coordinates_pooled_genotype_permutation.png"
    plt.savefig(output_path_pooled_genotype, dpi=300, bbox_inches="tight")
    # Also save as PDF and SVG
    pdf_path = output_path_pooled_genotype.with_suffix(".pdf")
    svg_path = output_path_pooled_genotype.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    print(f"\nPooled BallTypes plots (Genotype with permutation tests) saved to: {output_path_pooled_genotype}")

    # =============================================================================
    # IMPLEMENTATION 3: CTRL BALLTYPE ONLY WITH PERMUTATION TESTS
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 3: CTRL BALLTYPE ONLY (with Permutation Tests)")
    print(f"{'='*60}")

    # Filter to only ctrl BallType
    ctrl_data_trimmed = df_trimmed[df_trimmed[balltype_col] == "ctrl"][
        [time_col, ball_distance_col, light_col, genotype_col, "fly"]
    ].copy()

    print(f"Ctrl BallType only dataset shape: {ctrl_data_trimmed.shape}")
    print(f"Unique genotypes: {sorted(ctrl_data_trimmed[genotype_col].unique())}")

    light_conditions_ctrl = sorted(ctrl_data_trimmed[light_col].unique())

    # Create side-by-side subplots
    fig8, axes8 = plt.subplots(1, len(light_conditions_ctrl), figsize=(10 * len(light_conditions_ctrl), 6))
    if len(light_conditions_ctrl) == 1:
        axes8 = [axes8]

    for idx, light_condition in enumerate(light_conditions_ctrl):
        print(f"\nProcessing Light condition: {light_condition}")

        # Filter data for this light condition
        light_data = ctrl_data_trimmed[ctrl_data_trimmed[light_col] == light_condition].copy()

        print(f"  Data shape: {light_data.shape}")
        print(f"  Genotypes: {sorted(light_data[genotype_col].unique())}")

        # Preprocess for permutation test
        print(f"  Preprocessing data into {n_permutation_bins} time bins...")
        processed, bin_edges = preprocess_data_for_permutation(
            light_data,
            time_col=time_col,
            value_col=ball_distance_col,
            group_col=genotype_col,
            subject_col="fly",
            n_bins=n_permutation_bins,
        )

        # Compute permutation test
        print(f"  Computing permutation test ({n_permutations} permutations)...")
        perm_results = compute_permutation_test_genotype(
            processed,
            metric=f"avg_{ball_distance_col}",
            group_col=genotype_col,
            control_group="TNTxEmptyGal4",
            n_permutations=n_permutations,
            alpha=0.05,
            progress=True,
        )

        # Create trajectory plot with permutation annotations
        create_trajectory_plot_with_permutation(
            light_data,
            time_col=time_col,
            value_col=ball_distance_col,
            group_col=genotype_col,
            subject_col="fly",
            light_val=light_condition,
            n_bins=n_permutation_bins,
            permutation_results=perm_results,
            color_mapping=genotype_colors,
            ax=axes8[idx],
            bin_edges=bin_edges,
        )

    fig8.suptitle(
        "",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_dir = Path("/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Plots/Trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_ctrl_genotype = output_dir / "dark_olfaction_coordinates_ctrl_genotype_permutation.png"
    plt.savefig(output_path_ctrl_genotype, dpi=300, bbox_inches="tight")
    # Also save as PDF and SVG
    pdf_path = output_path_ctrl_genotype.with_suffix(".pdf")
    svg_path = output_path_ctrl_genotype.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    print(f"\nCtrl BallType only plots (Genotype with permutation tests) saved to: {output_path_ctrl_genotype}")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

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
    print("SUMMARY")
    print(f"{'='*80}")
    print("Three visualization approaches created:")
    print("1. Trimmed time: Uses actual seconds but only up to maximum shared time, preserves absolute timing")
    print("2. Pooled BallTypes: Focus on genotype effect across all ball types with permutation tests")
    print("3. Ctrl BallType only: Focus on standard ball condition with permutation tests")

    print(f"\nGrouping factors:")
    print(f"  - Light: {sorted(df_clean[light_col].unique())}")
    print(f"  - Genotype: {sorted(df_clean[genotype_col].unique())}")
    print(f"  - BallType: {sorted(df_clean[balltype_col].unique())}")

    print(f"\nFiles saved:")
    print(f"  Trimmed time:")
    print(f"    - {output_path_trimmed_genotype}")
    print(f"    - {output_path_trimmed_balltype}")
    print(f"    - {output_path_trimmed_combined} (side-by-side comparison)")
    print(f"  Pooled BallTypes with permutation tests:")
    print(f"    - {output_path_pooled_genotype}")
    print(f"  Ctrl BallType only with permutation tests:")
    print(f"    - {output_path_ctrl_genotype}")

    print(f"\n✅ All plots generated successfully!")

    return df_trimmed


if __name__ == "__main__":
    data = main()
