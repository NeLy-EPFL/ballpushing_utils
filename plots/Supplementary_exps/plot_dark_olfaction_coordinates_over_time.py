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


# Global conversion constant used by plotting and CSV export helpers
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


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
    ax.set_xlabel("Adjusted Time (%)", fontsize=14)
    ax.set_ylabel("Distance to Ball (pixels)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
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

        # Calculate statistics for each time bin using bootstrap for 95% CI
        time_bin_centers = sorted(hue_data["time_bin_center"].unique())

        means = []
        ci_lower = []
        ci_upper = []

        for time_bin in time_bin_centers:
            bin_values = hue_data[hue_data["time_bin_center"] == time_bin][f"{y_col}_median"].values

            if len(bin_values) > 0:
                # Calculate mean
                mean_val = np.mean(bin_values)
                means.append(mean_val)

                # Bootstrap confidence intervals (1000 iterations)
                if len(bin_values) > 1:
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(bin_values, size=len(bin_values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))

                    # 95% CI: 2.5th and 97.5th percentiles
                    ci_low = np.percentile(bootstrap_means, 2.5)
                    ci_high = np.percentile(bootstrap_means, 97.5)
                else:
                    # Single data point: no CI
                    ci_low = mean_val
                    ci_high = mean_val

                ci_lower.append(ci_low)
                ci_upper.append(ci_high)

        time_stats = pd.DataFrame(
            {"time_bin_center": time_bin_centers, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper}
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

        # Add bootstrapped 95% confidence intervals
        ax.fill_between(
            time_stats["time_bin_center"],
            time_stats["ci_lower"],
            time_stats["ci_upper"],
            color=color,
            alpha=0.2,
        )

    # Add vertical line at time 0
    # ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time (0s)")

    ax.set_xlabel("Time (min)", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

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


def bootstrap_ci_difference(group1_data, group2_data, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrap confidence interval for mean(group2) - mean(group1)."""
    if len(group1_data) == 0 or len(group2_data) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1_data, size=len(group1_data), replace=True)
        sample2 = rng.choice(group2_data, size=len(group2_data), replace=True)
        bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

    alpha = 100 - ci
    ci_lower = float(np.percentile(bootstrap_diffs, alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 - alpha / 2))
    return ci_lower, ci_upper


def compute_permutation_test_genotype(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="Genotype",
    control_group="TNTxEmptyGal4",
    n_permutations=10000,
    alpha=0.05,
    n_bootstrap=10000,
    collect_detailed_stats=False,
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
    detailed_rows = []

    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    for time_bin in iterator:
        bin_data = processed_data[processed_data["time_bin"] == time_bin]

        bin_center = float(bin_data["bin_center"].iloc[0]) if len(bin_data) > 0 and "bin_center" in bin_data else np.nan
        bin_start = float(bin_data["bin_start"].iloc[0]) if len(bin_data) > 0 and "bin_start" in bin_data else np.nan
        bin_end = float(bin_data["bin_end"].iloc[0]) if len(bin_data) > 0 and "bin_end" in bin_data else np.nan

        # Get control and test values
        control_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        if len(control_vals) == 0 or len(test_vals) == 0:
            observed_diffs.append(np.nan)
            p_values.append(1.0)
            if collect_detailed_stats:
                detailed_rows.append(
                    {
                        "time_bin": int(time_bin),
                        "bin_center": bin_center,
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                        "n_control": int(len(control_vals)),
                        "n_test": int(len(test_vals)),
                        "control_mean": np.nan,
                        "test_mean": np.nan,
                        "control_median": np.nan,
                        "test_median": np.nan,
                        "mean_diff_test_minus_control": np.nan,
                        "median_diff_test_minus_control": np.nan,
                        "effect_size_raw": np.nan,
                        "pct_change": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "pct_ci_lower": np.nan,
                        "pct_ci_upper": np.nan,
                        "cohens_d": np.nan,
                        "p_value_raw": 1.0,
                    }
                )
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

        # Two-tailed p-value with permutation-resolution floor.
        # This avoids exact 0.0 values and keeps p-values reportable in CSV outputs.
        extreme_count = int(np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)))
        p_value = (extreme_count + 1) / (n_permutations + 1)
        p_values.append(p_value)

        if collect_detailed_stats:
            control_mean = float(np.mean(control_vals))
            test_mean = float(np.mean(test_vals))
            control_median = float(np.median(control_vals))
            test_median = float(np.median(test_vals))
            mean_diff = test_mean - control_mean
            median_diff = test_median - control_median

            pooled_std = np.sqrt((np.var(control_vals, ddof=1) + np.var(test_vals, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if np.isfinite(pooled_std) and pooled_std > 0 else np.nan

            ci_lower, ci_upper = bootstrap_ci_difference(
                control_vals,
                test_vals,
                n_bootstrap=n_bootstrap,
                ci=95,
                random_state=42 + int(time_bin),
            )

            if np.isfinite(control_mean) and control_mean != 0:
                pct_change = (mean_diff / control_mean) * 100.0
                pct_ci_lower = (ci_lower / control_mean) * 100.0 if np.isfinite(ci_lower) else np.nan
                pct_ci_upper = (ci_upper / control_mean) * 100.0 if np.isfinite(ci_upper) else np.nan
            else:
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan

            detailed_rows.append(
                {
                    "time_bin": int(time_bin),
                    "bin_center": bin_center,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "n_control": int(len(control_vals)),
                    "n_test": int(len(test_vals)),
                    "control_mean": control_mean,
                    "test_mean": test_mean,
                    "control_median": control_median,
                    "test_median": test_median,
                    "mean_diff_test_minus_control": mean_diff,
                    "median_diff_test_minus_control": median_diff,
                    "effect_size_raw": mean_diff,
                    "pct_change": pct_change,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "pct_ci_lower": pct_ci_lower,
                    "pct_ci_upper": pct_ci_upper,
                    "cohens_d": cohens_d,
                    "p_value_raw": float(p_value),
                }
            )

    # Apply FDR correction across all time bins
    p_values_raw = np.array(p_values)
    rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

    if collect_detailed_stats:
        for i, row in enumerate(detailed_rows):
            row["p_value_fdr"] = float(p_values_corrected[i])
            row["significant_fdr"] = bool(rejected[i])

            p_fdr = float(p_values_corrected[i])
            if p_fdr < 0.001:
                row["significance_label"] = "***"
            elif p_fdr < 0.01:
                row["significance_label"] = "**"
            elif p_fdr < 0.05:
                row["significance_label"] = "*"
            else:
                row["significance_label"] = ""

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

    if collect_detailed_stats:
        results["bin_stats"] = detailed_rows

    print(f"    Raw significant bins: {results['n_significant_raw']}/{n_bins}")
    print(f"    FDR significant bins: {results['n_significant']}/{n_bins} (α={alpha})")

    return results


def format_bin_stats_for_publication(
    bin_stats,
    comparison,
    condition_a,
    condition_b,
    pixels_per_mm=PIXELS_PER_MM,
):
    """Convert internal bin stats to standardized, mm-based publication CSV schema."""
    if not bin_stats:
        return pd.DataFrame()

    df = pd.DataFrame(bin_stats).copy()

    if "time_bin" in df.columns:
        df = df.rename(columns={"time_bin": "Time_Bin"})
    if "bin_start" in df.columns:
        df = df.rename(columns={"bin_start": "Bin_Start_min"})
    if "bin_end" in df.columns:
        df = df.rename(columns={"bin_end": "Bin_End_min"})
    if "bin_center" in df.columns:
        df = df.rename(columns={"bin_center": "Bin_Center_min"})

    df["Comparison"] = comparison
    df["ConditionA"] = condition_a
    df["ConditionB"] = condition_b

    df["N_Control"] = df.get("n_control", np.nan)
    df["N_Test"] = df.get("n_test", np.nan)

    df["Mean_Control_mm"] = df.get("control_mean", np.nan) / pixels_per_mm
    df["Mean_Test_mm"] = df.get("test_mean", np.nan) / pixels_per_mm
    df["Difference_mm"] = df.get("mean_diff_test_minus_control", np.nan) / pixels_per_mm
    df["CI_Lower_mm"] = df.get("ci_lower", np.nan) / pixels_per_mm
    df["CI_Upper_mm"] = df.get("ci_upper", np.nan) / pixels_per_mm

    df["Pct_Change"] = df.get("pct_change", np.nan)
    df["Pct_CI_Lower"] = df.get("pct_ci_lower", np.nan)
    df["Pct_CI_Upper"] = df.get("pct_ci_upper", np.nan)
    df["Cohens_D"] = df.get("cohens_d", np.nan)
    df["P_Value_Raw"] = df.get("p_value_raw", np.nan)
    df["P_Value_FDR"] = df.get("p_value_fdr", np.nan)

    if "significance_label" in df.columns:
        df["Significant_FDR"] = df["significance_label"].replace({"": "ns"})
    elif "significant_fdr" in df.columns:
        df["Significant_FDR"] = np.where(df["significant_fdr"], "*", "ns")
    else:
        df["Significant_FDR"] = "ns"

    keep_cols = [
        "Comparison",
        "ConditionA",
        "ConditionB",
        "Time_Bin",
        "Bin_Start_min",
        "Bin_End_min",
        "Bin_Center_min",
        "N_Control",
        "N_Test",
        "Mean_Control_mm",
        "Mean_Test_mm",
        "Difference_mm",
        "CI_Lower_mm",
        "CI_Upper_mm",
        "Pct_Change",
        "Pct_CI_Lower",
        "Pct_CI_Upper",
        "Cohens_D",
        "P_Value_Raw",
        "P_Value_FDR",
        "Significant_FDR",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


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
        return

    # Get groups
    groups = sorted(data[group_col].unique())

    if len(groups) != 2:
        ax.text(0.5, 0.5, f"Expected 2 groups, found {len(groups)}", ha="center", va="center", transform=ax.transAxes)
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

        # First, aggregate per fly within each time bin to get one value per fly per bin
        fly_binned = group_data.groupby([subject_col, "time_bin_center"])[value_col_plot].mean().reset_index()

        # Bootstrap approach: resample flies with replacement to compute confidence intervals
        time_bin_centers = sorted(fly_binned["time_bin_center"].unique())

        means = []
        ci_lower = []
        ci_upper = []

        for time_bin in time_bin_centers:
            # Get one value per fly for this time bin
            fly_values = fly_binned[fly_binned["time_bin_center"] == time_bin][value_col_plot].values

            if len(fly_values) > 0:
                # Calculate mean across flies
                mean_val = np.mean(fly_values)
                means.append(mean_val)

                # Bootstrap confidence intervals by resampling flies (1000 iterations)
                if len(fly_values) > 1:
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        # Resample flies with replacement
                        bootstrap_flies = np.random.choice(fly_values, size=len(fly_values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_flies))

                    # 95% CI: 2.5th and 97.5th percentiles
                    ci_low = np.percentile(bootstrap_means, 2.5)
                    ci_high = np.percentile(bootstrap_means, 97.5)
                else:
                    # Single fly: no CI
                    ci_low = mean_val
                    ci_high = mean_val

                ci_lower.append(ci_low)
                ci_upper.append(ci_high)

        time_grouped = pd.DataFrame(
            {"time_bin_center": time_bin_centers, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper}
        )

        # Determine line style and color (IR8a purple, Empty grey)
        # Use solid lines for both genotypes (color distinguishes them)
        if color_mapping is not None:
            color = color_mapping.get(group, "gray")
        else:
            color = "#9467bd" if group == test_group else "#7f7f7f"

        # Plot mean trajectory - shift by baseline
        ax.plot(
            time_grouped["time_bin_center"],
            time_grouped["mean"] - y_baseline,
            linestyle="solid",
            color=color,
            linewidth=2.5,
            label=group,
            zorder=10,
        )

        # Error band (bootstrapped 95% CI)
        ax.fill_between(
            time_grouped["time_bin_center"],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=color,
            alpha=0.2,
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
                # Add significance stars only (remove p-value annotation)
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

    # Formatting
    ax.set_xlabel("Time (min)", fontsize=14)
    ax.set_ylabel("Relative ball distance (mm)", fontsize=14)
    # No legend - will be added separately only to the first subplot
    ax.tick_params(axis="both", labelsize=10)
    # Remove background grid - only keep the time bin vertical lines
    ax.grid(False)


def create_ir8a_light_trajectory_plot_with_permutation(
    data,
    time_col,
    value_col,
    light_col,
    subject_col,
    permutation_results=None,
    color_mapping=None,
    alpha_mapping=None,
    linestyle_mapping=None,
    ax=None,
    bin_edges=None,
    show_legend=True,
    show_ylabel=True,
):
    """
    Create a trajectory plot comparing Light conditions within IR8a only.
    """
    PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    groups = sorted(data[light_col].unique(), key=lambda x: (x != "off", x))
    if len(groups) != 2:
        ax.text(
            0.5,
            0.5,
            f"Expected 2 light conditions, found {len(groups)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    data_copy = data.copy()
    data_copy[f"{value_col}_mm"] = data_copy[value_col] / PIXELS_PER_MM
    value_col_plot = f"{value_col}_mm"

    time_bin_size = 0.1
    min_time = data_copy[time_col].min()
    max_time = data_copy[time_col].max()
    fine_bin_edges = np.arange(min_time, max_time + time_bin_size, time_bin_size)
    data_copy["time_bin_fine"] = pd.cut(data_copy[time_col], bins=fine_bin_edges, labels=False, include_lowest=True)
    data_copy["time_bin_center"] = data_copy["time_bin_fine"] * time_bin_size + time_bin_size / 2 + min_time

    all_means = []
    for group in groups:
        group_data = data_copy[data_copy[light_col] == group]
        time_grouped = group_data.groupby("time_bin_center")[value_col_plot].agg(["mean", "sem"]).reset_index()
        all_means.extend(time_grouped["mean"].values)

    y_baseline = min(all_means) if all_means else 0

    for group in groups:
        group_data = data_copy[data_copy[light_col] == group]
        fly_binned = group_data.groupby([subject_col, "time_bin_center"])[value_col_plot].mean().reset_index()
        time_bin_centers = sorted(fly_binned["time_bin_center"].unique())

        means = []
        ci_lower = []
        ci_upper = []

        for time_bin in time_bin_centers:
            fly_values = fly_binned[fly_binned["time_bin_center"] == time_bin][value_col_plot].values

            if len(fly_values) > 0:
                mean_val = np.mean(fly_values)
                means.append(mean_val)

                if len(fly_values) > 1:
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        bootstrap_flies = np.random.choice(fly_values, size=len(fly_values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_flies))

                    ci_low = np.percentile(bootstrap_means, 2.5)
                    ci_high = np.percentile(bootstrap_means, 97.5)
                else:
                    ci_low = mean_val
                    ci_high = mean_val

                ci_lower.append(ci_low)
                ci_upper.append(ci_high)

        time_grouped = pd.DataFrame(
            {"time_bin_center": time_bin_centers, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper}
        )

        if color_mapping is not None:
            color = color_mapping.get(group, "#9467bd")
        else:
            color = "#9467bd"

        if alpha_mapping is not None:
            line_alpha = alpha_mapping.get(group, 0.9)
        else:
            line_alpha = 0.95 if group == "on" else 0.55

        if linestyle_mapping is not None:
            linestyle = linestyle_mapping.get(group, "solid")
        else:
            linestyle = "solid" if group == "on" else "dashed"

        fill_alpha = max(0.08, min(0.3, line_alpha * 0.25))

        ax.plot(
            time_grouped["time_bin_center"],
            time_grouped["mean"] - y_baseline,
            linestyle=linestyle,
            color=color,
            linewidth=2.5,
            label=f"Light {group}",
            alpha=line_alpha,
            zorder=10,
        )

        ax.fill_between(
            time_grouped["time_bin_center"],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=color,
            alpha=fill_alpha,
            zorder=5,
        )

    if bin_edges is not None:
        for edge in bin_edges:
            ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, zorder=1)

    y_max_shifted = max(all_means) - y_baseline if all_means else 0
    y_range = y_max_shifted
    ax.set_ylim(0, y_max_shifted + 0.08 * y_range)

    if permutation_results is not None and bin_edges is not None:
        y_annotation_stars = y_max_shifted + 0.02 * y_range

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

    ax.set_xlabel("Time (min)", fontsize=14)
    if show_ylabel:
        ax.set_ylabel("Relative ball distance (mm)", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    if show_legend:
        ax.legend(loc="upper left", fontsize=12)


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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations for confidence intervals in detailed stats",
    )

    args = parser.parse_args()

    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be a positive integer")

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

    # Convert time from seconds to minutes
    df_aligned[time_col] = df_aligned[time_col] / 60.0
    print(
        f"Time range after conversion to minutes: {df_aligned[time_col].min():.2f}min to {df_aligned[time_col].max():.2f}min"
    )

    # Define color mappings for consistent styling
    genotype_colors = {"TNTxEmptyGal4": "#7f7f7f", "TNTxIR8a": "#9467bd"}
    balltype_colors = {"ctrl": "green", "sand": "brown"}

    # Use downsampled data for trimmed time approach
    original_df_clean = df_aligned.copy()

    print(f"Original data shape for trimmed approach: {original_df_clean.shape}")
    print(
        f"Original time range: {original_df_clean[time_col].min():.2f} to {original_df_clean[time_col].max():.2f} seconds"
    )

    # Find maximum shared time across all flies
    max_shared_time = find_maximum_shared_time(original_df_clean, "fly", time_col)

    # Trim data to shared time
    df_trimmed = trim_data_to_shared_time(original_df_clean, max_shared_time, time_col)

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

    # Sort light conditions with 'on' first, then 'off'
    all_light_conditions = sorted(pooled_data_trimmed[light_col].unique())
    light_conditions_pooled = sorted(all_light_conditions, key=lambda x: (x != "on", x))

    output_dir = Path("/mnt/upramdya_data/MD/TNT_Olfaction_Dark/Plots/Trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_stats_tables = []

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

        # Print sample sizes for each genotype
        for genotype in sorted(light_data[genotype_col].unique()):
            n_flies = light_data[light_data[genotype_col] == genotype]["fly"].nunique()
            print(f"    {genotype}: n={n_flies} flies")

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
            n_bootstrap=args.n_bootstrap,
            collect_detailed_stats=True,
            progress=True,
        )

        # Save standardized mm-based binwise stats for this light condition
        pooled_bin_stats = perm_results.get("bin_stats", [])
        pooled_stats_df = format_bin_stats_for_publication(
            pooled_bin_stats,
            comparison=f"pooled_genotype_{light_condition}",
            condition_a="TNTxEmptyGal4",
            condition_b=perm_results.get("test_group", "TNTxIR8a"),
            pixels_per_mm=PIXELS_PER_MM,
        )
        if not pooled_stats_df.empty:
            pooled_stats_df["Light_Condition"] = light_condition
            pooled_stats_df["Metric"] = f"avg_{ball_distance_col}"
            pooled_stats_df["N_Permutations"] = n_permutations
            pooled_stats_df["N_Bootstrap"] = args.n_bootstrap
            pooled_stats_df["Value_Unit"] = "mm"

            per_light_csv = (
                output_dir / f"dark_olfaction_pooled_genotype_binwise_statistics_light_{light_condition}.csv"
            )
            pooled_stats_df.to_csv(per_light_csv, index=False, float_format="%.6f")
            print(f"  Saved pooled-genotype stats CSV: {per_light_csv}")
            pooled_stats_tables.append(pooled_stats_df)

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

        # Add light condition text box to bottom right of subplot
        ax = axes7[idx]
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Position text box in bottom right corner
        x_pos = xlim[1] - 0.15 * (xlim[1] - xlim[0])
        y_pos = ylim[0] + 0.08 * (ylim[1] - ylim[0])
        # Add text with black box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=1.5)
        ax.text(
            x_pos,
            y_pos,
            f"Light: {light_condition}",
            fontsize=14,
            ha="center",
            va="bottom",
            bbox=bbox_props,
            zorder=20,
        )

    # Add legend only on the far right (after all subplots)
    if len(light_conditions_pooled) > 0:
        ax_last = axes7[-1]
        # Get current handles and labels
        handles, labels = ax_last.get_legend_handles_labels()
        if handles:
            ax_last.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=12,
            )

    fig7.suptitle(
        "",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path_pooled_genotype = output_dir / "dark_olfaction_coordinates_pooled_genotype_permutation.png"
    plt.savefig(output_path_pooled_genotype, dpi=300, bbox_inches="tight")
    # Also save as PDF and SVG
    pdf_path = output_path_pooled_genotype.with_suffix(".pdf")
    svg_path = output_path_pooled_genotype.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    print(f"\nPooled BallTypes plots (Genotype with permutation tests) saved to: {output_path_pooled_genotype}")

    if pooled_stats_tables:
        pooled_combined_df = pd.concat(pooled_stats_tables, ignore_index=True)
        pooled_combined_csv = output_dir / "dark_olfaction_pooled_genotype_binwise_statistics_all_lights.csv"
        pooled_combined_df.to_csv(pooled_combined_csv, index=False, float_format="%.6f")
        print(f"Combined pooled-genotype stats CSV saved to: {pooled_combined_csv}")

    # =============================================================================
    # IMPLEMENTATION 3: GENOTYPE-WISE LIGHT EFFECT (ON VS OFF)
    # =============================================================================
    print(f"\n{'='*60}")
    print("IMPLEMENTATION 3: GENOTYPE-WISE LIGHT EFFECT (Light on vs off)")
    print(f"{'='*60}")

    preferred_genotype_order = ["TNTxEmptyGal4", "TNTxIR8a"]
    available_genotypes = pooled_data_trimmed[genotype_col].dropna().unique().tolist()
    genotype_order = [g for g in preferred_genotype_order if g in available_genotypes]

    if len(genotype_order) == 0:
        print("⚠️ No genotype data available; skipping genotype-wise light comparison trajectory plot.")
    else:
        print(f"Genotypes selected for light comparison: {genotype_order}")
        genotype_light_output_dir = output_dir / "genotype_light_comparison"
        genotype_light_output_dir.mkdir(parents=True, exist_ok=True)

        fig_light, axes_light = plt.subplots(1, len(genotype_order), figsize=(10 * len(genotype_order), 6), sharey=True)
        if len(genotype_order) == 1:
            axes_light = [axes_light]

        genotype_light_stats_tables = []

        for idx, genotype in enumerate(genotype_order):
            genotype_data = pooled_data_trimmed[pooled_data_trimmed[genotype_col] == genotype].copy()
            print(f"\nProcessing genotype for light comparison: {genotype} (shape={genotype_data.shape})")

            genotype_lights = sorted(genotype_data[light_col].dropna().unique(), key=lambda x: (x != "off", x))
            print(f"  Light conditions: {genotype_lights}")

            if len(genotype_lights) != 2 or "off" not in genotype_lights or "on" not in genotype_lights:
                print(f"  ⚠️ Skipping {genotype}: expected both 'off' and 'on' light conditions.")
                axes_light[idx].text(
                    0.5,
                    0.5,
                    f"Insufficient data for {genotype}",
                    ha="center",
                    va="center",
                    transform=axes_light[idx].transAxes,
                )
                axes_light[idx].set_title(f"{genotype}: Light effect", fontsize=14)
                continue

            print(f"  Preprocessing {genotype} data into {n_permutation_bins} time bins...")
            genotype_processed, genotype_bin_edges = preprocess_data_for_permutation(
                genotype_data,
                time_col=time_col,
                value_col=ball_distance_col,
                group_col=light_col,
                subject_col="fly",
                n_bins=n_permutation_bins,
            )

            print(f"  Computing {genotype} Light on vs off permutation test ({n_permutations} permutations)...")
            genotype_perm_results = compute_permutation_test_genotype(
                genotype_processed,
                metric=f"avg_{ball_distance_col}",
                group_col=light_col,
                control_group="off",
                n_permutations=n_permutations,
                alpha=0.05,
                n_bootstrap=args.n_bootstrap,
                collect_detailed_stats=True,
                progress=True,
            )

            genotype_color = genotype_colors.get(genotype, "#7f7f7f")
            light_colors = {"off": genotype_color, "on": genotype_color}
            light_alphas = {"off": 0.5, "on": 0.95}
            light_linestyles = {"off": "dashed", "on": "solid"}

            create_ir8a_light_trajectory_plot_with_permutation(
                genotype_data,
                time_col=time_col,
                value_col=ball_distance_col,
                light_col=light_col,
                subject_col="fly",
                permutation_results=genotype_perm_results,
                color_mapping=light_colors,
                alpha_mapping=light_alphas,
                linestyle_mapping=light_linestyles,
                ax=axes_light[idx],
                bin_edges=genotype_bin_edges,
                show_legend=True,
                show_ylabel=(idx == 0),
            )

            axes_light[idx].set_title(f"{genotype}: Light effect", fontsize=14)

            genotype_bin_stats = genotype_perm_results.get("bin_stats", [])
            if genotype_bin_stats:
                genotype_stats_df = format_bin_stats_for_publication(
                    genotype_bin_stats,
                    comparison=f"{genotype}_light_on_vs_off",
                    condition_a="off",
                    condition_b=genotype_perm_results.get("test_group", "on"),
                    pixels_per_mm=PIXELS_PER_MM,
                )
                genotype_stats_df["Genotype"] = genotype
                genotype_stats_df["Metric"] = f"avg_{ball_distance_col}"
                genotype_stats_df["N_Permutations"] = n_permutations
                genotype_stats_df["N_Bootstrap"] = args.n_bootstrap
                genotype_stats_df["Value_Unit"] = "mm"
                genotype_light_stats_tables.append(genotype_stats_df)

                per_genotype_stats_path = (
                    genotype_light_output_dir / f"dark_olfaction_{genotype}_light_binwise_statistics_mm.csv"
                )
                genotype_stats_df.to_csv(per_genotype_stats_path, index=False, float_format="%.6f")
                print(f"  {genotype} light-comparison detailed stats saved to: {per_genotype_stats_path}")

        plt.tight_layout()
        output_path_genotype_light = (
            genotype_light_output_dir / "dark_olfaction_coordinates_genotype_light_permutation.png"
        )
        plt.savefig(output_path_genotype_light, dpi=300, bbox_inches="tight")
        plt.savefig(output_path_genotype_light.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.savefig(output_path_genotype_light.with_suffix(".svg"), dpi=300, bbox_inches="tight")
        plt.close(fig_light)

        if genotype_light_stats_tables:
            combined_genotype_light_stats = pd.concat(genotype_light_stats_tables, ignore_index=True)
            combined_genotype_light_stats_path = (
                genotype_light_output_dir / "dark_olfaction_genotype_light_binwise_statistics_mm.csv"
            )
            combined_genotype_light_stats.to_csv(combined_genotype_light_stats_path, index=False, float_format="%.6f")
            print(f"Combined genotype light-comparison detailed stats saved to: {combined_genotype_light_stats_path}")

        print(f"\nGenotype-wise light-comparison trajectory plot saved to: {output_path_genotype_light}")

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nDataset summary:")
    print(f"  - Maximum shared time: {max_shared_time:.2f} seconds ({max_shared_time/60.0:.2f} minutes)")
    print(f"  - Total data points: {df_trimmed.shape[0]}")
    print(f"  - Total flies: {df_trimmed['fly'].nunique()}")
    print(f"  - Time range: {df_trimmed[time_col].min():.2f} to {df_trimmed[time_col].max():.2f} minutes")

    print(f"\nGrouping factors:")
    print(f"  - Light conditions: {sorted(df_clean[light_col].unique())}")
    print(f"  - Genotypes: {sorted(df_clean[genotype_col].unique())}")
    print(f"  - BallTypes (pooled in main plot): {sorted(df_clean[balltype_col].unique())}")

    print(f"\nGenerated plots:")
    print(f"  - Pooled BallTypes with permutation tests (side-by-side by Light condition)")
    print(f"    Saved as PNG, PDF, and SVG in: {output_dir}")
    print(f"  - Pooled BallTypes detailed binwise statistics (CSV, mm, standardized schema)")
    print(f"    Saved in: {output_dir / 'dark_olfaction_pooled_genotype_binwise_statistics_all_lights.csv'}")
    print(f"  - Genotype-wise Light comparison trajectory plot (EmptyGal4 and IR8a side by side; each: on vs off)")
    print(f"    Saved as PNG, PDF, and SVG in: {output_dir / 'genotype_light_comparison'}")
    print(f"  - Genotype-wise Light comparison detailed binwise statistics (CSV, mm, standardized schema)")
    print(
        f"    Saved in: {output_dir / 'genotype_light_comparison' / 'dark_olfaction_genotype_light_binwise_statistics_mm.csv'}"
    )

    print(f"\n✅ All plots generated successfully!")

    return df_trimmed


if __name__ == "__main__":
    data = main()
