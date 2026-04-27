#!/usr/bin/env python3
"""
Script to generate ball speed plots over time for MagnetBlock experiments.

This script generates speed visualizations with:
- Speed calculation from distance data
- Rolling window smoothing
- Time-binned analysis of speed
- Permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between Magnet and non-Magnet conditions

Usage:
    python plot_magnetblock_speeds.py [--n-bins N] [--n-permutations N] [--output-dir PATH]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/speeds)
    --rolling-window: Size of rolling window for smoothing (default: 150)
    --no-stats: Skip permutation testing and statistics visualization
    --no-smoothing: Skip rolling window smoothing of speed data
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.family"] = "Arial"

# matplotlib.rcParams["path.simplify"] = True
# matplotlib.rcParams["path.simplify_threshold"] = 1.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Conversion from pixels/frame to mm/s (assuming 30 fps)
PIXELS_PER_FRAME_TO_MM_PER_S = 0.06


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def bootstrap_ci_difference(group1, group2, n_bootstrap=10000, ci=95, random_state=42):
    """Bootstrap CI for mean(group2) - mean(group1)."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs[i] = np.mean(s2) - np.mean(s1)

    alpha = 100 - ci
    lower = np.percentile(diffs, alpha / 2)
    upper = np.percentile(diffs, 100 - alpha / 2)
    return lower, upper


def load_coordinates_dataset():
    """Load the MagnetBlock experiments coordinates dataset and subset to summary flies"""
    # Load full coordinates dataset (2 hours)
    coordinates_path = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/260127_15_coordinates_magnet_block_folders_Data/coordinates/pooled_coordinates.feather"

    print(f"Loading coordinates dataset from: {coordinates_path}")
    try:
        dataset = pd.read_feather(coordinates_path)
        print(f"✅ MagnetBlock coordinates dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {coordinates_path}")
        raise FileNotFoundError(f"Coordinates dataset not found at {coordinates_path}")

    # Load summary dataset to get list of valid flies
    summary_path = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather"

    print(f"Loading summary dataset from: {summary_path}")
    try:
        summary = pd.read_feather(summary_path)
        print(f"✅ Summary dataset loaded successfully! Shape: {summary.shape}")
    except FileNotFoundError:
        print(f"❌ Summary dataset not found at {summary_path}")
        raise FileNotFoundError(f"Summary dataset not found at {summary_path}")

    # Get list of valid flies from summary
    if "fly" in summary.columns:
        valid_flies = summary["fly"].unique()
        print(f"Valid flies from summary: {len(valid_flies)}")

        # Subset coordinates to only include valid flies
        n_before = len(dataset)
        dataset = dataset[dataset["fly"].isin(valid_flies)]
        n_after = len(dataset)
        print(f"Filtered coordinates: {n_before} -> {n_after} rows ({n_before - n_after} removed)")
    else:
        print("Warning: 'fly' column not found in summary dataset, using all flies")

    # Check for Magnet column
    if "Magnet" not in dataset.columns:
        raise ValueError(f"'Magnet' column not found in dataset. Available columns: {list(dataset.columns)}")

    print(f"Magnet groups: {sorted(dataset['Magnet'].unique())}")

    # Print sample counts
    if "fly" in dataset.columns:
        print(f"\nSample counts by Magnet status:")
        for status in sorted(dataset["Magnet"].unique()):
            n_flies = dataset[dataset["Magnet"] == status]["fly"].nunique()
            n_points = len(dataset[dataset["Magnet"] == status])
            print(f"  {status}: {n_flies} flies, {n_points} data points")

    return dataset


def calculate_speeds(data, position_col="y_fly_0", time_col="time", rolling_window=150, apply_smoothing=True):
    """Calculate speeds per fly and smooth within each fly's trajectory"""
    data = data.copy()

    # Sort by fly and time for proper per-fly diff
    data = data.sort_values(by=["fly", time_col]).reset_index(drop=True)

    # Calculate speed PER FLY
    def calc_per_fly_speed(group):
        group = group.sort_values(time_col).reset_index(drop=True)
        group["speed_raw"] = group[position_col].diff() / group[time_col].diff()
        return group

    data = data.groupby("fly", group_keys=False).apply(calc_per_fly_speed)

    # Calculate speed and convert to mm/s
    data["speed_raw"] = data["speed_raw"].abs()
    data["speed_mm_s"] = data["speed_raw"] * 0.06

    if apply_smoothing:
        # CRITICAL FIX: Apply rolling window PER FLY (not across flies!)
        def smooth_per_fly(group):
            group = group.sort_values(time_col).reset_index(drop=True)
            group["speed_mm_s_smooth"] = group["speed_mm_s"].rolling(window=rolling_window, min_periods=1).mean()
            return group

        data = data.groupby("fly", group_keys=False).apply(smooth_per_fly)
    else:
        data["speed_mm_s_smooth"] = data["speed_mm_s"]

    return data


def preprocess_speed_data(
    data,
    time_col="time",
    speed_col="speed_mm_s_smooth",
    group_col="Magnet",
    subject_col="fly",
    n_bins=12,
):
    """
    Preprocess speed data by binning time and computing statistics per bin.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw speed data
    time_col : str
        Column name for time
    speed_col : str
        Column name for speed values
    group_col : str
        Column name for grouping (Magnet)
    subject_col : str
        Column name for individual subjects (flies)
    n_bins : int
        Number of time bins

    Returns:
    --------
    pd.DataFrame
        Processed data with time bins and statistics
    """
    # Remove NaN speeds
    data = data.dropna(subset=[speed_col])

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
    grouped = data.groupby([group_col, subject_col, "time_bin"])[speed_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{speed_col}"]

    # Add bin centers and edges
    grouped["bin_center"] = grouped["time_bin"].map(dict(enumerate(bin_centers)))
    grouped["bin_start"] = grouped["time_bin"].map(dict(enumerate(bin_edges[:-1])))
    grouped["bin_end"] = grouped["time_bin"].map(dict(enumerate(bin_edges[1:])))

    return grouped


def compute_permutation_test(
    processed_data,
    metric="avg_speed_mm_s_smooth",
    group_col="Magnet",
    control_group="non-Magnet",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Compute permutation tests for each time bin.
    No FDR correction needed for 2-group comparison.

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
        raise ValueError(f"Expected 2 groups for MagnetBlock analysis, found {len(groups)}: {groups}")

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
    n_control_values = []
    n_test_values = []
    mean_control_values = []
    mean_test_values = []
    ci_lower_values = []
    ci_upper_values = []
    pct_change_values = []
    pct_ci_lower_values = []
    pct_ci_upper_values = []
    cohens_d_values = []

    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    for time_bin in iterator:
        bin_data = processed_data[processed_data["time_bin"] == time_bin]

        # Get control and test values
        control_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        if len(control_vals) == 0 or len(test_vals) == 0:
            observed_diffs.append(np.nan)
            p_values.append(1.0)
            n_control_values.append(0)
            n_test_values.append(0)
            mean_control_values.append(np.nan)
            mean_test_values.append(np.nan)
            ci_lower_values.append(np.nan)
            ci_upper_values.append(np.nan)
            pct_change_values.append(np.nan)
            pct_ci_lower_values.append(np.nan)
            pct_ci_upper_values.append(np.nan)
            cohens_d_values.append(np.nan)
            continue

        # Observed difference
        mean_control = np.mean(control_vals)
        mean_test = np.mean(test_vals)
        obs_diff = mean_test - mean_control
        observed_diffs.append(obs_diff)
        n_control_values.append(len(control_vals))
        n_test_values.append(len(test_vals))
        mean_control_values.append(mean_control)
        mean_test_values.append(mean_test)

        ci_lower, ci_upper = bootstrap_ci_difference(control_vals, test_vals, n_bootstrap=10000, ci=95, random_state=42)
        ci_lower_values.append(ci_lower)
        ci_upper_values.append(ci_upper)

        if mean_control != 0:
            pct_change = (obs_diff / mean_control) * 100
            pct_ci_lower = (ci_lower / mean_control) * 100
            pct_ci_upper = (ci_upper / mean_control) * 100
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan
        pct_change_values.append(pct_change)
        pct_ci_lower_values.append(pct_ci_lower)
        pct_ci_upper_values.append(pct_ci_upper)

        cohens_d_values.append(cohens_d(control_vals, test_vals))

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

    # Prepare p-values array
    p_values_raw = np.array(p_values)

    # Benjamini-Hochberg FDR correction (control across the n_bins tests)
    def benjamini_hochberg(pvals, alpha):
        m = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_p = pvals[sorted_idx]
        thresholds = (np.arange(1, m + 1) / m) * alpha
        below = sorted_p <= thresholds
        if not np.any(below):
            reject = np.zeros(m, dtype=bool)
        else:
            max_i = np.max(np.where(below)[0])
            reject = np.zeros(m, dtype=bool)
            reject[: max_i + 1] = True
        # map back to original ordering
        reject_orig = np.zeros(m, dtype=bool)
        reject_orig[sorted_idx] = reject

        # Adjusted p-values (BH step-up)
        # p_adj_sorted[i] = min_{j>=i} (m/j * p_sorted[j])
        p_adj_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * sorted_p[::-1])[::-1]
        p_adj = np.empty_like(p_adj_sorted)
        p_adj[sorted_idx] = p_adj_sorted

        return reject_orig, p_adj

    # Handle NaNs in p-values (keep NaNs in place)
    valid_mask = ~np.isnan(p_values_raw)
    p_values_adj = np.full_like(p_values_raw, np.nan)
    significant_mask = np.zeros_like(valid_mask, dtype=bool)

    if valid_mask.any():
        rej, p_adj_compact = benjamini_hochberg(p_values_raw[valid_mask], alpha)
        p_values_adj[valid_mask] = p_adj_compact
        significant_mask[valid_mask] = rej

    significant_timepoints = np.where(significant_mask)[0]
    n_significant = int(np.sum(significant_mask))

    results = {
        "test_group": test_group,
        "control_group": control_group,
        "time_bins": time_bins,
        "observed_diffs": observed_diffs,
        "n_control": n_control_values,
        "n_test": n_test_values,
        "mean_control": mean_control_values,
        "mean_test": mean_test_values,
        "ci_lower": ci_lower_values,
        "ci_upper": ci_upper_values,
        "pct_change": pct_change_values,
        "pct_ci_lower": pct_ci_lower_values,
        "pct_ci_upper": pct_ci_upper_values,
        "cohens_d": cohens_d_values,
        "p_values_raw": p_values_raw,
        "p_values": p_values_adj,
        "significant_timepoints": significant_timepoints,
        "n_significant": n_significant,
    }

    print(f"    Significant bins after FDR: {n_significant}/{n_bins} (α={alpha}, FDR)")

    return results


def create_speed_plot(
    data,
    time_col="time",
    speed_col="speed_mm_s_smooth",
    group_col="Magnet",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
    time_window=None,
    ax=None,
    add_title=True,
    figsize_mm=(245, 120),
    font_size_labels=14,
    font_size_title=18,
    font_size_pval=11,
    font_size_legend=12,
    show_pvalues=False,
    asterisk_scale=1.0,
    bootstrap_n=1000,
    ci_level=95,
):
    """
    Create a speed plot comparing Magnet to non-Magnet conditions.
    Aggregates data by grouping time and Magnet, calculates mean and SEM.
    Filters to time window around 60 minutes (±10 minutes) and adds vertical line at start point.

    Parameters:
    -----------
    data : pd.DataFrame
        Full speed data
    time_col : str
        Column name for time
    speed_col : str
        Column name for speed/speed
    group_col : str
        Column name for grouping
    subject_col : str
        Column name for subjects
    n_bins : int
        Number of time bins
    permutation_results : dict
        Results from permutation test
    output_path : Path or str
        Where to save the plot
    figsize_mm : tuple
        Figure size in millimeters (width, height)
    font_size_labels : float
        Font size for axis labels in points
    font_size_title : float
        Font size for title in points
    font_size_pval : float
        Font size for p-values and asterisks in points
    font_size_legend : float
        Font size for legend in points
    show_pvalues : bool
        Whether to show p-values in annotations (default: False)
    asterisk_scale : float
        Scaling factor for asterisk font size relative to font_size_pval (default: 1.0)
    bootstrap_n : int
        Number of bootstrap samples for confidence intervals (default: 1000)
    ci_level : float
        Confidence interval level (default: 95)
    """
    # Filter out NaN speeds
    data = data.dropna(subset=[speed_col])

    if time_window is not None:
        tmin, tmax = time_window
        print(f"  Filtering data to time window: {tmin}s - {tmax}s ({tmin/60:.1f} - {tmax/60:.1f} min)")
        data = data[(data[time_col] >= tmin) & (data[time_col] <= tmax)]
        if len(data) == 0:
            print(f"Warning: No data in time window")
            return

    # Get groups
    groups = sorted(data[group_col].unique())

    if len(groups) != 2:
        print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
        return

    # Determine control and test groups
    if "n" in groups and "y" in groups:
        control_group = "n"
        test_group = "y"
    else:
        control_group = groups[0]
        test_group = groups[1]

    # Get sample sizes (within window)
    n_control = data[data[group_col] == control_group][subject_col].nunique()
    n_test = data[data[group_col] == test_group][subject_col].nunique()

    # Create figure or use provided axis
    if ax is None:
        # Convert figsize from mm to inches (1 inch = 25.4 mm)
        figsize_inches = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize_inches)

    # Colors for each group (matching trajectory plot)
    colors = {control_group: "#faa41a", test_group: "#3953A4"}
    labels = {
        control_group: f"No access to ball (n = {n_control})",
        test_group: f"Access to immobile ball (n = {n_test})",
    }

    def bootstrap_ci(values, n_samples, ci):
        values = np.asarray(values)
        if values.size == 0:
            return np.nan, np.nan
        rng = np.random.default_rng()
        boot_means = rng.choice(values, size=(n_samples, values.size), replace=True).mean(axis=1)
        alpha = (100 - ci) / 2
        lower = np.percentile(boot_means, alpha)
        upper = np.percentile(boot_means, 100 - alpha)
        return lower, upper

    # Process each group
    for group in groups:
        group_data = data[data[group_col] == group]

        # Group by time and calculate mean and bootstrap CI
        grouped = group_data.groupby(time_col)[speed_col].apply(list).reset_index(name="values")
        grouped["mean"] = grouped["values"].apply(np.mean)
        grouped[["ci_lower", "ci_upper"]] = grouped["values"].apply(
            lambda vals: pd.Series(bootstrap_ci(vals, bootstrap_n, ci_level))
        )

        time = grouped[time_col].values / 60.0
        mean_speed = grouped["mean"].values
        lower_bound = grouped["ci_lower"].values
        upper_bound = grouped["ci_upper"].values

        # Plot the mean line
        ax.plot(time, mean_speed, color=colors[group], linewidth=1, label=labels[group])

        # Plot the confidence interval as a shaded area (rasterized to avoid Illustrator issues)
        ax.fill_between(time, lower_bound, upper_bound, color=colors[group], alpha=0.3, rasterized=True, zorder=0)

    # Add vertical line at start point (60 minutes) if in range
    start = 60 * 60
    start_min = start / 60.0
    if (time_window is None) or (start >= data[time_col].min() and start <= data[time_col].max()):
        ax.axvline(start_min, color="red", linestyle="dashed", linewidth=3, label="Start Point", zorder=5)

    # Set x-axis limits
    if time_window is not None:
        time_window_min = (time_window[0] / 60.0, time_window[1] / 60.0)
        ax.set_xlim(*time_window_min)

    # Get current y-axis limits and extend them to make room for significance annotations
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Extend y-axis by 20% to make room for significance stars and p-values
    ax.set_ylim(y_min, y_max + 0 * y_range)

    # Calculate time bin edges
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_width = (time_max - time_min) / n_bins

    # Draw vertical dotted lines for time bins and annotate significance
    if permutation_results is not None:
        significant_bins = permutation_results["significant_timepoints"]

        for time_bin in range(n_bins):
            bin_start = time_min + time_bin * bin_width
            bin_end = bin_start + bin_width
            bin_start_min = bin_start / 60.0
            bin_end_min = bin_end / 60.0

            # Draw faint dotted lines for bins
            ax.axvline(bin_start_min, color="gray", linestyle="dotted", alpha=0.5)
            ax.axvline(bin_end_min, color="gray", linestyle="dotted", alpha=0.5)

            # Annotate significance levels and p-values for all bins
            p_value = permutation_results["p_values"][time_bin]

            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = ""

            # Add significance stars if significant
            if significance and time_bin in significant_bins:
                # Position stars at top of plot
                y_star = y_max + 0.05 * y_range
                ax.text(
                    (bin_start_min + bin_end_min) / 2,
                    y_star,
                    significance,
                    ha="center",
                    va="bottom",
                    fontsize=font_size_pval * asterisk_scale,
                    color="red",
                    fontweight="bold",
                )

            # Add p-value for all bins (only if show_pvalues is True)
            if show_pvalues:
                y_pval = y_max + 0.11 * y_range
                p_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"
                p_color = "red" if time_bin in significant_bins else "gray"
                ax.text(
                    (bin_start_min + bin_end_min) / 2,
                    y_pval,
                    p_text,
                    ha="center",
                    va="bottom",
                    fontsize=font_size_pval,
                    color=p_color,
                )

    # Formatting
    # Custom formatter to display time as MM:SS instead of decimal minutes
    def format_time_mmss(x, pos):
        """Convert decimal minutes to MM:SS format"""
        minutes = int(x)
        seconds = int((x - minutes) * 60)
        return f"{minutes}:{seconds:02d}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_time_mmss))
    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Average Speed (mm/s)", fontsize=font_size_labels)
    if add_title:
        ax.set_title("Average Speed Across Flies Grouped by Magnet Positions", fontsize=font_size_title)

    # Remove gridlines for cleaner look
    ax.grid(False)

    # Set legend with configured font size (only if showing labels)
    if time_window is None:
        # Full plot - show legend
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=font_size_legend)
    # Window plot - no legend since already in full plot

    if output_path is not None and ax is None:
        plt.tight_layout()
        # Save as PDF and PNG
        pdf_path = output_path.with_suffix(".pdf")
        png_path = output_path.with_suffix(".png")
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"  Saved PDF: {pdf_path}")
        print(f"  Saved PNG: {png_path}")
        plt.close()


def generate_speed_plot(
    data,
    n_bins=12,
    n_permutations=10000,
    rolling_window=150,
    output_dir=None,
    show_progress=True,
    compute_stats=True,
    apply_smoothing=True,
):
    """
    Generate speed plot for Magnet vs non-Magnet experiments.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    n_bins : int
        Number of time bins
    n_permutations : int
        Number of permutations for testing
    rolling_window : int
        Window size for rolling average smoothing
    output_dir : Path or str
        Output directory
    show_progress : bool
        Show progress bars
    compute_stats : bool
        Whether to compute permutation statistics (default: True)
    apply_smoothing : bool
        Whether to apply rolling window smoothing (default: True)
    """
    if output_dir is None:
        output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/speeds")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING MAGNETBLOCK SPEED PLOT")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
    print(f"Rolling window: {rolling_window}")
    print(f"Permutations: {n_permutations}")

    # Check required columns
    required_cols = ["time", "distance_ball_0", "Magnet", "fly"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get Magnet groups
    groups = sorted(data["Magnet"].unique())

    if len(groups) != 2:
        raise ValueError(f"Expected 2 Magnet groups, found {len(groups)}: {groups}")

    # Determine control and test groups
    if "n" in groups and "y" in groups:
        control_group = "n"
        test_group = "y"
    else:
        control_group = groups[0]
        test_group = groups[1]

    print(f"\nMagnet groups in dataset: {groups}")
    print(f"Control group: {control_group}")
    print(f"Test group: {test_group}")

    # Calculate speeds
    smoothing_msg = "with" if apply_smoothing else "without"
    print(f"\nCalculating speeds {smoothing_msg} rolling window smoothing...")
    data_with_speeds = calculate_speeds(
        data, position_col="y_fly_0", rolling_window=rolling_window, apply_smoothing=apply_smoothing
    )
    print(f"Speed data shape: {data_with_speeds.shape}")

    def compute_pre_post_median(
        df,
        time_col="time",
        speed_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        pre_window=(3400, 3600),
        post_window=(3800, 4000),
    ):
        """Return DataFrame with per-fly median speed in pre and post windows.

        Only flies that have values in both windows are returned (so lines can be drawn).
        """
        pre_min, pre_max = pre_window
        post_min, post_max = post_window

        pre = (
            df[(df[time_col] >= pre_min) & (df[time_col] <= pre_max)]
            .groupby([group_col, subject_col])[speed_col]
            .median()
            .reset_index()
            .rename(columns={speed_col: "pre_median"})
        )

        post = (
            df[(df[time_col] >= post_min) & (df[time_col] <= post_max)]
            .groupby([group_col, subject_col])[speed_col]
            .median()
            .reset_index()
            .rename(columns={speed_col: "post_median"})
        )

        merged = pd.merge(pre, post, on=[group_col, subject_col], how="inner")
        return merged

    def plot_pre_post_by_group(
        merged_df,
        group_col="Magnet",
        subject_col="fly",
        output_path=None,
        change_results=None,
        pre_label=None,
        post_label=None,
        pre_window=(3400, 3600),
        post_window=(3800, 4000),
        figsize_mm=(245, 120),
        font_size_labels=14,
        font_size_title=18,
        font_size_pval=11,
        font_size_legend=12,
    ):
        """Create two-subplot figure (one per group) showing pre/post medians per fly.

        Each fly is a line connecting its pre and post median.
        """
        if pre_label is None:
            # Convert seconds to MM:SS format
            pre_start_min = int(pre_window[0] // 60)
            pre_start_sec = int(pre_window[0] % 60)
            pre_end_min = int(pre_window[1] // 60)
            pre_end_sec = int(pre_window[1] % 60)
            pre_label = f"{pre_start_min}:{pre_start_sec:02d}-{pre_end_min}:{pre_end_sec:02d} min"
        if post_label is None:
            # Convert seconds to MM:SS format
            post_start_min = int(post_window[0] // 60)
            post_start_sec = int(post_window[0] % 60)
            post_end_min = int(post_window[1] // 60)
            post_end_sec = int(post_window[1] % 60)
            post_label = f"{post_start_min}:{post_start_sec:02d}-{post_end_min}:{post_end_sec:02d} min"

        groups = sorted(merged_df[group_col].unique())
        if len(groups) == 0:
            print("No flies found for pre/post comparison")
            return

        # Convert figsize from mm to inches (1 inch = 25.4 mm)
        # For multiple subplots, the width applies to each subplot, so divide by number of groups
        figsize_inches = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, axes = plt.subplots(1, len(groups), figsize=figsize_inches, sharey=True)
        if len(groups) == 1:
            axes = [axes]

        colors = {groups[0]: "#faa41a"} if len(groups) == 1 else {g: c for g, c in zip(groups, ["#faa41a", "#3953A4"])}

        for ax, grp in zip(axes, groups):
            dfg = merged_df[merged_df[group_col] == grp]
            if dfg.empty:
                ax.set_title(f"{grp} (no data)")
                continue

            x = [0, 1]
            for _, row in dfg.iterrows():
                ax.plot(
                    x, [row["pre_median"], row["post_median"]], marker="o", color=colors.get(grp, "gray"), alpha=0.7
                )

            # Plot per-group median lines for pre and post
            med_pre = dfg["pre_median"].median()
            med_post = dfg["post_median"].median()
            ax.plot(x, [med_pre, med_post], marker=None, color="k", linewidth=2)

            ax.set_xticks(x)
            ax.set_xticklabels([pre_label, post_label], fontsize=font_size_labels)
            if grp == "n":
                title_label = "No access to ball"
            elif grp == "y":
                title_label = "Access to immobile ball"
            else:
                title_label = str(grp)
            ax.set_title(f"{title_label} (n={dfg.shape[0]})", fontsize=font_size_title)
            ax.set_ylabel("Median Average Speed (mm/s)", fontsize=font_size_labels)
            ax.tick_params(axis="y", labelsize=font_size_labels)
            ax.grid(True, alpha=0.3)
            # (group-level annotation moved to figure-level to avoid duplicate labels)

        fig.tight_layout()
        # Add a single figure-level annotation for the between-group permutation test (absolute change)
        if change_results is not None and "absolute_change" in change_results:
            p_val_adj = change_results["absolute_change"].get(
                "p_value_adj", change_results["absolute_change"].get("p_value")
            )
            if p_val_adj is None or np.isnan(p_val_adj):
                sig = "ns"
            elif p_val_adj < 0.001:
                sig = "***"
            elif p_val_adj < 0.01:
                sig = "**"
            elif p_val_adj < 0.05:
                sig = "*"
            else:
                sig = "ns"

            fig.suptitle(
                f"Between-group absolute-change: {sig} (p_adj={p_val_adj:.3f})",
                y=1.02,
                fontsize=font_size_pval,
                color="red",
            )

        if output_path is not None:
            pdfp = Path(output_path).with_suffix(".pdf")
            pngp = Path(output_path).with_suffix(".png")
            svgp = Path(output_path).with_suffix(".svg")
            fig.savefig(pdfp, format="pdf", dpi=300, bbox_inches="tight")
            fig.savefig(pngp, format="png", dpi=300, bbox_inches="tight")
            fig.savefig(svgp, format="svg", dpi=300, bbox_inches="tight")
            print(f"  Saved pre/post comparison PDF: {pdfp}")
            print(f"  Saved pre/post comparison PNG: {pngp}")
            print(f"  Saved pre/post comparison SVG: {svgp}")
            plt.close(fig)

    # (Pre/post plotting will be done after change comparison so we can annotate significance)

    def compare_speed_changes(
        data,
        time_col="time",
        speed_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        pre_window=(3400, 3600),
        post_window=(3800, 4000),
        n_permutations=10000,
    ):
        """
        Compare speed changes between groups using permutation test.

        Returns both absolute change and percent change comparisons.
        """
        pre_min, pre_max = pre_window
        post_min, post_max = post_window

        # Calculate pre and post medians per fly
        pre = (
            data[(data[time_col] >= pre_min) & (data[time_col] <= pre_max)]
            .groupby([group_col, subject_col])[speed_col]
            .median()
            .reset_index()
            .rename(columns={speed_col: "pre_median"})
        )

        post = (
            data[(data[time_col] >= post_min) & (data[time_col] <= post_max)]
            .groupby([group_col, subject_col])[speed_col]
            .median()
            .reset_index()
            .rename(columns={speed_col: "post_median"})
        )

        merged = pd.merge(pre, post, on=[group_col, subject_col], how="inner")

        # Calculate change scores
        merged["absolute_change"] = merged["post_median"] - merged["pre_median"]
        merged["percent_change"] = ((merged["post_median"] - merged["pre_median"]) / merged["pre_median"]) * 100

        # Get groups
        groups = sorted(merged[group_col].unique())
        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(groups)}")

        control_group = groups[0]
        test_group = groups[1]

        results = {}

        # Test absolute change
        for metric in ["absolute_change", "percent_change"]:
            control_vals = merged[merged[group_col] == control_group][metric].values
            test_vals = merged[merged[group_col] == test_group][metric].values

            # Observed difference in mean change
            obs_diff = np.mean(test_vals) - np.mean(control_vals)

            # Permutation test
            combined = np.concatenate([control_vals, test_vals])
            n_control = len(control_vals)

            perm_diffs = []
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_control = combined[:n_control]
                perm_test = combined[n_control:]
                perm_diff = np.mean(perm_test) - np.mean(perm_control)
                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

            results[metric] = {
                "control_mean": np.mean(control_vals),
                "control_sem": stats.sem(control_vals),
                "test_mean": np.mean(test_vals),
                "test_sem": stats.sem(test_vals),
                "observed_diff": obs_diff,
                "p_value": p_value,
            }

        # Apply Benjamini-Hochberg FDR correction across the two metrics
        def benjamini_hochberg_array(pvals, alpha=0.05):
            pvals = np.array(pvals)
            m = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_p = pvals[sorted_idx]
            thresholds = (np.arange(1, m + 1) / m) * alpha
            below = sorted_p <= thresholds
            reject = np.zeros(m, dtype=bool)
            if np.any(below):
                max_i = np.max(np.where(below)[0])
                reject[: max_i + 1] = True
            # adjusted p-values
            p_adj_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * sorted_p[::-1])[::-1]
            p_adj = np.empty_like(p_adj_sorted)
            p_adj[sorted_idx] = p_adj_sorted
            return reject[sorted_idx.argsort()], p_adj

        pvals = [results["absolute_change"]["p_value"], results["percent_change"]["p_value"]]
        rej, p_adj = benjamini_hochberg_array(pvals, alpha=0.05)
        # assign adjusted p-values and significance
        metrics = ["absolute_change", "percent_change"]
        for mname, p_raw, p_a, r in zip(metrics, pvals, p_adj, rej):
            results[mname]["p_value_adj"] = float(p_a)
            results[mname]["significant_adj"] = bool(r)

        # Print results
        print(f"\n{'='*60}")
        print("SPEED CHANGE COMPARISON")
        print(f"{'='*60}")
        print(f"Control group: {control_group} (n={len(control_vals)})")
        print(f"Test group: {test_group} (n={len(test_vals)})")
        print(f"Pre window: {pre_window[0]/60:.1f}-{pre_window[1]/60:.1f} min")
        print(f"Post window: {post_window[0]/60:.1f}-{post_window[1]/60:.1f} min")

        print(f"\nABSOLUTE CHANGE (mm/s):")
        print(
            f"  {control_group}: {results['absolute_change']['control_mean']:.3f} ± "
            f"{results['absolute_change']['control_sem']:.3f}"
        )
        print(
            f"  {test_group}: {results['absolute_change']['test_mean']:.3f} ± "
            f"{results['absolute_change']['test_sem']:.3f}"
        )
        print(f"  Difference: {results['absolute_change']['observed_diff']:.3f}")
        print(f"  P-value: {results['absolute_change']['p_value']:.6f}")

        print(f"\nPERCENT CHANGE (%):")
        print(
            f"  {control_group}: {results['percent_change']['control_mean']:.1f}% ± "
            f"{results['percent_change']['control_sem']:.1f}%"
        )
        print(
            f"  {test_group}: {results['percent_change']['test_mean']:.1f}% ± "
            f"{results['percent_change']['test_sem']:.1f}%"
        )
        print(f"  Difference: {results['percent_change']['observed_diff']:.1f}%")
        print(f"  P-value: {results['percent_change']['p_value']:.6f}")

        return merged, results

    def save_permutation_results_csv(results, output_csv_path):
        """Save per-bin permutation stats to CSV."""
        df = pd.DataFrame(
            {
                "Time_Bin": results["time_bins"],
                "N_Control": results.get("n_control", [np.nan] * len(results["time_bins"])),
                "N_Test": results.get("n_test", [np.nan] * len(results["time_bins"])),
                "Mean_Control_mm_s": results.get("mean_control", [np.nan] * len(results["time_bins"])),
                "Mean_Test_mm_s": results.get("mean_test", [np.nan] * len(results["time_bins"])),
                "Difference_mm_s": results["observed_diffs"],
                "CI_Lower_mm_s": results.get("ci_lower", [np.nan] * len(results["time_bins"])),
                "CI_Upper_mm_s": results.get("ci_upper", [np.nan] * len(results["time_bins"])),
                "Pct_Change": results.get("pct_change", [np.nan] * len(results["time_bins"])),
                "Pct_CI_Lower": results.get("pct_ci_lower", [np.nan] * len(results["time_bins"])),
                "Pct_CI_Upper": results.get("pct_ci_upper", [np.nan] * len(results["time_bins"])),
                "Cohens_d": results.get("cohens_d", [np.nan] * len(results["time_bins"])),
                "P_Value_Raw": results["p_values_raw"],
                "P_Value_FDR": results["p_values"],
            }
        )
        df["Significant_FDR"] = np.where(df["P_Value_FDR"] < 0.05, "*", "ns")
        df.to_csv(output_csv_path, index=False, float_format="%.6f")
        print(f"✅ Statistical CSV saved: {output_csv_path}")

    def save_change_results_csv(results, output_csv_path):
        """Save pre/post change comparison stats to CSV."""
        rows = []
        for metric_name, metric_results in results.items():
            rows.append(
                {
                    "Comparison": metric_name,
                    "Control_Mean": metric_results.get("control_mean", np.nan),
                    "Control_SEM": metric_results.get("control_sem", np.nan),
                    "Test_Mean": metric_results.get("test_mean", np.nan),
                    "Test_SEM": metric_results.get("test_sem", np.nan),
                    "Observed_Difference": metric_results.get("observed_diff", np.nan),
                    "P_Value_Raw": metric_results.get("p_value", np.nan),
                    "P_Value_FDR": metric_results.get("p_value_adj", np.nan),
                    "Significant_FDR": bool(metric_results.get("significant_adj", False)),
                }
            )

        pd.DataFrame(rows).to_csv(output_csv_path, index=False, float_format="%.6f")
        print(f"✅ Change-comparison CSV saved: {output_csv_path}")

    # Initialize permutation results
    permutation_results = None

    if compute_stats:
        # Preprocess data
        print(f"\nPreprocessing speed data into {n_bins} time bins...")
        processed = preprocess_speed_data(
            data_with_speeds,
            time_col="time",
            speed_col="speed_mm_s_smooth",
            group_col="Magnet",
            subject_col="fly",
            n_bins=n_bins,
        )
        print(f"Processed data shape: {processed.shape}")

        # Compute permutation test
        print(f"\nComputing permutation test ({n_permutations} permutations)...")

        permutation_results = compute_permutation_test(
            processed,
            metric="avg_speed_mm_s_smooth",
            group_col="Magnet",
            control_group=control_group,
            n_permutations=n_permutations,
            alpha=0.05,
            progress=show_progress,
        )
    else:
        print(f"\nSkipping statistical analysis (--no-stats flag enabled)")

    # Compare speed changes
    merged_changes, change_results = compare_speed_changes(
        data_with_speeds,
        pre_window=(3400, 3600),
        post_window=(3800, 4000),
        n_permutations=n_permutations,
    )

    # Save pre/post change-comparison CSV
    change_csv = output_dir / f"speed_change_comparison_{test_group}_vs_{control_group}.csv"
    save_change_results_csv(change_results, change_csv)

    # Generate and save full range plot
    print(f"\nGenerating full range speed plot...")
    fig_full, ax_full = plt.subplots(figsize=((245 / 25.4), (120 / 25.4)))
    create_speed_plot(
        data_with_speeds,
        time_col="time",
        speed_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,  # show trajectory permutation stats on full plot
        output_path=None,
        time_window=None,
        ax=ax_full,
        add_title=False,
        figsize_mm=(245, 120),
        font_size_labels=14,
        font_size_title=18,
        font_size_pval=11,
        font_size_legend=12,
        show_pvalues=False,
        asterisk_scale=1.5,
    )
    pdf_path_full = output_dir / f"speed_{test_group}_vs_{control_group}_full.pdf"
    png_path_full = output_dir / f"speed_{test_group}_vs_{control_group}_full.png"
    svg_path_full = output_dir / f"speed_{test_group}_vs_{control_group}_full.svg"
    fig_full.tight_layout()
    fig_full.savefig(pdf_path_full, format="pdf", dpi=300, bbox_inches="tight")
    fig_full.savefig(png_path_full, format="png", dpi=300, bbox_inches="tight")
    fig_full.savefig(svg_path_full, format="svg", dpi=300, bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path_full}")
    print(f"  Saved PNG: {png_path_full}")
    print(f"  Saved SVG: {svg_path_full}")
    plt.close(fig_full)

    # Generate and save windowed plot
    print(f"\nGenerating windowed speed plot...")
    start = 60 * 60
    window = 10 * 60
    time_window = (start - window, start + window)

    # For the windowed plot, compute separate permutation statistics on the windowed data
    permutation_results_window = None
    if compute_stats:
        print(f"  Computing permutation test for windowed data ({time_window[0]}s - {time_window[1]}s)...")
        # Subset data to time window
        data_windowed = data_with_speeds[
            (data_with_speeds["time"] >= time_window[0]) & (data_with_speeds["time"] <= time_window[1])
        ].copy()

        # Preprocess windowed data
        processed_window = preprocess_speed_data(
            data_windowed,
            time_col="time",
            speed_col="speed_mm_s_smooth",
            group_col="Magnet",
            subject_col="fly",
            n_bins=n_bins,
        )

        # Compute permutation test on windowed data
        permutation_results_window = compute_permutation_test(
            processed_window,
            metric="avg_speed_mm_s_smooth",
            group_col="Magnet",
            control_group=control_group,
            n_permutations=n_permutations,
            alpha=0.05,
            progress=show_progress,
        )

        # Save windowed permutation statistics CSV
        window_csv = output_dir / f"speed_permutation_statistics_window_{test_group}_vs_{control_group}.csv"
        save_permutation_results_csv(permutation_results_window, window_csv)

    fig_window, ax_window = plt.subplots(figsize=((163.33 / 25.4), (80 / 25.4)))
    create_speed_plot(
        data_with_speeds,
        time_col="time",
        speed_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results_window,  # Use windowed-specific statistics
        output_path=None,
        time_window=time_window,
        ax=ax_window,
        add_title=False,
        figsize_mm=(163.33, 80),
        font_size_labels=9,
        font_size_title=12,
        font_size_pval=7,
        font_size_legend=8,
        show_pvalues=False,
        asterisk_scale=1.5,
    )
    pdf_path_window = output_dir / f"speed_{test_group}_vs_{control_group}_window.pdf"
    png_path_window = output_dir / f"speed_{test_group}_vs_{control_group}_window.png"
    svg_path_window = output_dir / f"speed_{test_group}_vs_{control_group}_window.svg"
    fig_window.tight_layout()
    fig_window.savefig(pdf_path_window, format="pdf", dpi=300, bbox_inches="tight")
    fig_window.savefig(png_path_window, format="png", dpi=300, bbox_inches="tight")
    fig_window.savefig(svg_path_window, format="svg", dpi=300, bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path_window}")
    print(f"  Saved PNG: {png_path_window}")
    print(f"  Saved SVG: {svg_path_window}")
    plt.close(fig_window)

    # Now plot pre/post per-fly comparisons and annotate with permutation test results
    try:
        pre_window = (3400, 3600)
        post_window = (3800, 4000)
        merged_pre_post = compute_pre_post_median(data_with_speeds, pre_window=pre_window, post_window=post_window)
        prepost_out = output_dir / f"speed_pre_post_by_fly_{test_group}_vs_{control_group}"
        plot_pre_post_by_group(
            merged_pre_post,
            change_results=change_results,
            output_path=prepost_out,
            pre_window=pre_window,
            post_window=post_window,
            figsize_mm=(180, 95),
            font_size_labels=14,
            font_size_title=18,
            font_size_pval=11,
            font_size_legend=12,
        )
    except Exception as e:
        print(f"Warning: failed to compute/plot pre/post medians after change comparison: {e}")

    # Save statistical results if statistics were computed
    if compute_stats and permutation_results is not None:
        # Save full-range permutation statistics CSV
        full_csv = output_dir / f"speed_permutation_statistics_full_{test_group}_vs_{control_group}.csv"
        save_permutation_results_csv(permutation_results, full_csv)

        stats_file = output_dir / "speed_permutation_statistics.txt"
        with open(stats_file, "w") as f:
            f.write("MagnetBlock Speed Permutation Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Control group: {permutation_results['control_group']}\n")
            f.write(f"Test group: {permutation_results['test_group']}\n")
            f.write(f"Number of permutations: {n_permutations}\n")
            f.write(f"Number of time bins: {n_bins}\n")
            f.write(f"Rolling window size: {rolling_window}\n")
            f.write(f"Significance level: α = 0.05\n\n")

            f.write(f"Significant bins: {permutation_results['n_significant']}/{n_bins}\n")
            f.write(f"Significant time bins: {list(permutation_results['significant_timepoints'])}\n\n")

            f.write("Bin | Obs. Diff | P-value | Significant\n")
            f.write("-" * 60 + "\n")
            for i, time_bin in enumerate(permutation_results["time_bins"]):
                obs_diff = permutation_results["observed_diffs"][i]
                p_val = permutation_results["p_values"][i]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                f.write(f"{time_bin:3d} | {obs_diff:9.3f} | {p_val:7.6f} | {sig}\n")

        print(f"\n✅ Statistical results saved to: {stats_file}")

    print(f"\n{'='*60}")
    print("✅ Speed plot generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball speed plot for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_magnetblock_speeds.py
  python plot_magnetblock_speeds.py --n-bins 15 --rolling-window 200
  python plot_magnetblock_speeds.py --output-dir /path/to/output
        """,
    )

    parser.add_argument("--n-bins", type=int, default=12, help="Number of time bins for analysis (default: 12)")

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical testing (default: 10000)",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=150,
        help="Size of rolling window for smoothing (default: 150)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/upramdya_data/MD/MagnetBlock/Plots/speeds",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/speeds)",
    )

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument("--no-stats", action="store_true", help="Skip permutation testing and statistics visualization")

    parser.add_argument("--no-smoothing", action="store_true", help="Skip rolling window smoothing of speed data")

    args = parser.parse_args()

    # Load data
    print("Loading MagnetBlock coordinates data...")
    data = load_coordinates_dataset()

    # Generate plot
    generate_speed_plot(
        data,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        rolling_window=args.rolling_window,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        compute_stats=not args.no_stats,
        apply_smoothing=not args.no_smoothing,
    )


if __name__ == "__main__":
    main()
