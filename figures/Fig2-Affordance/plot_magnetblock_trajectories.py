#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for MagnetBlock experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- Permutation tests for each time bin (no FDR correction needed for 2-group comparison)
- Significance annotations (*, **, ***)
- Comparison between Magnet and non-Magnet conditions
- Individual fly trajectories with mean overlay

Usage:
    python plot_magnetblock_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/trajectories)
"""

import argparse
import os
from pathlib import Path

# Keep sibling scripts importable (legacy; harmless).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from ballpushing_utils import dataset as dataset_path_for  # noqa: F401 (used below)
from ballpushing_utils import figure_output_dir
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def format_p_value(p_value, n_permutations):
    """Format p-value for display, handling very small values

    Parameters:
    -----------
    p_value : float
        P-value to format
    n_permutations : int
        Number of permutations used (determines precision)

    Returns:
    --------
    str : Formatted p-value string
    """
    # Permutation tests have finite precision: minimum p = 1/n_permutations
    min_p = 1.0 / n_permutations

    if p_value <= min_p:
        # Format as "p ≤ 1e-4" etc.
        return f"p ≤ {min_p:.0e}"
    elif p_value < 0.001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.4f}"


def calculate_cohens_d(group1_data, group2_data):
    """Calculate Cohen's d effect size between two groups

    Parameters:
    -----------
    group1_data : array-like
        Data from first group
    group2_data : array-like
        Data from second group

    Returns:
    --------
    float : Cohen's d effect size
    """
    n1 = len(group1_data)
    n2 = len(group2_data)

    mean1 = np.mean(group1_data)
    mean2 = np.mean(group2_data)

    var1 = np.var(group1_data, ddof=1)
    var2 = np.var(group2_data, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean2 - mean1) / pooled_std


def bootstrap_ci_difference(group1_data, group2_data, n_bootstrap=10000, ci=95):
    """Calculate bootstrapped confidence interval for difference between groups

    Parameters:
    -----------
    group1_data : array-like
        Data from first group
    group2_data : array-like
        Data from second group
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (e.g., 95 for 95% CI)

    Returns:
    --------
    tuple : (lower_bound, upper_bound) of the CI
    """
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(group1_data, size=len(group1_data), replace=True)
        sample2 = np.random.choice(group2_data, size=len(group2_data), replace=True)

        # Calculate difference in means
        diff = np.mean(sample2) - np.mean(sample1)
        bootstrap_diffs.append(diff)

    # Calculate CI
    alpha = 100 - ci
    lower = np.percentile(bootstrap_diffs, alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 - alpha / 2)

    return lower, upper


def load_coordinates_dataset():
    """Load the MagnetBlock experiments coordinates dataset"""
    dataset_path = dataset_path_for(
        "MagnetBlock/Datasets/251126_10_coordinates_magnet_block_folders_Data/coordinates/pooled_coordinates.feather"
    )

    print(f"Loading coordinates dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ MagnetBlock coordinates dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Coordinates dataset not found at {dataset_path}")

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


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="Magnet", subject_col="fly", n_bins=12
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

    return grouped


def compute_permutation_test(
    processed_data,
    metric="avg_distance_ball_0",
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
    cohens_d_values = []
    ci_lower_values = []
    ci_upper_values = []
    pct_change_values = []
    pct_ci_lower_values = []
    pct_ci_upper_values = []

    iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

    for time_bin in iterator:
        bin_data = processed_data[processed_data["time_bin"] == time_bin]

        # Get control and test values
        control_vals = bin_data[bin_data[group_col] == control_group][metric].values
        test_vals = bin_data[bin_data[group_col] == test_group][metric].values

        if len(control_vals) == 0 or len(test_vals) == 0:
            observed_diffs.append(np.nan)
            p_values.append(1.0)
            cohens_d_values.append(np.nan)
            ci_lower_values.append(np.nan)
            ci_upper_values.append(np.nan)
            pct_change_values.append(np.nan)
            pct_ci_lower_values.append(np.nan)
            pct_ci_upper_values.append(np.nan)
            continue

        # Observed difference
        control_mean = np.mean(control_vals)
        test_mean = np.mean(test_vals)
        obs_diff = test_mean - control_mean
        observed_diffs.append(obs_diff)

        # Calculate Cohen's d
        cohens_d = calculate_cohens_d(control_vals, test_vals)
        cohens_d_values.append(cohens_d)

        # Calculate bootstrapped confidence interval
        ci_lower, ci_upper = bootstrap_ci_difference(control_vals, test_vals, n_bootstrap=10000, ci=95)
        ci_lower_values.append(ci_lower)
        ci_upper_values.append(ci_upper)

        # Calculate percentage change relative to control
        if control_mean != 0:
            pct_change = (obs_diff / control_mean) * 100
            pct_ci_lower = (ci_lower / control_mean) * 100
            pct_ci_upper = (ci_upper / control_mean) * 100
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan

        pct_change_values.append(pct_change)
        pct_ci_lower_values.append(pct_ci_lower)
        pct_ci_upper_values.append(pct_ci_upper)

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

    # No FDR correction needed for single comparison
    p_values = np.array(p_values)
    significant_timepoints = np.where(p_values < alpha)[0]
    n_significant = np.sum(p_values < alpha)

    results = {
        "test_group": test_group,
        "control_group": control_group,
        "time_bins": time_bins,
        "observed_diffs": observed_diffs,
        "p_values": p_values,
        "cohens_d": cohens_d_values,
        "ci_lower": ci_lower_values,
        "ci_upper": ci_upper_values,
        "pct_change": pct_change_values,
        "pct_ci_lower": pct_ci_lower_values,
        "pct_ci_upper": pct_ci_upper_values,
        "significant_timepoints": significant_timepoints,
        "n_significant": n_significant,
    }

    print(f"    Significant bins: {n_significant}/{n_bins} (α={alpha})")

    return results


def create_trajectory_plot(
    data,
    time_col="time",
    value_col="distance_ball_0",
    group_col="Magnet",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
):
    """
    Create a trajectory plot comparing Magnet to non-Magnet conditions.
    Uses seaborn lineplot for mean ± CI visualization.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    time_col : str
        Column name for time
    value_col : str
        Column name for distance/position
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
    """
    if len(data) == 0:
        print(f"Warning: No data for trajectory plot")
        return

    # Get groups
    groups = sorted(data[group_col].unique())

    if len(groups) != 2:
        print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
        return

    # Determine control and test groups
    # Map to labels: "n" -> "Control", "y" -> "Magnet block"
    if "n" in groups and "y" in groups:
        control_group = "n"
        test_group = "y"
        data = data.copy()
        # Convert distance to mm
        data[value_col] = data[value_col] / PIXELS_PER_MM

        # Get sample sizes
        n_control = data[data[group_col] == control_group][subject_col].nunique()
        n_test = data[data[group_col] == test_group][subject_col].nunique()

        data["label"] = data[group_col].apply(
            lambda x: f"Magnet block (n={n_test})" if x == "y" else f"Control (n={n_control})"
        )
        group_col_plot = "label"
    else:
        control_group = groups[0]
        test_group = groups[1]
        data = data.copy()
        # Convert distance to mm
        data[value_col] = data[value_col] / PIXELS_PER_MM

        # Get sample sizes
        n_control = data[data[group_col] == control_group][subject_col].nunique()
        n_test = data[data[group_col] == test_group][subject_col].nunique()

        data["label"] = data[group_col].apply(
            lambda x: f"{test_group} (n={n_test})" if x == test_group else f"{control_group} (n={n_control})"
        )
        group_col_plot = "label"

    # Downsample data for plotting (every 290 frames as in notebook)
    print(f"  Downsampling data for plotting...")
    data_ds = data.groupby(subject_col, group_keys=False).apply(lambda df: df.iloc[::290, :]).reset_index(drop=True)
    print(f"  Downsampled from {len(data)} to {len(data_ds)} points")

    # Convert time to minutes for plotting
    time_col_min = "time_min"
    data_ds[time_col_min] = data_ds[time_col] / 60.0

    # Create figure — start with a placeholder size; will be resized to exact axes dimensions
    fig, ax = plt.subplots(figsize=(6, 3))

    # Color palette matching other Fig2 scripts
    COLOR_TEST = "#3953A4"  # blue  — Access to immobile ball
    COLOR_CTRL = "#FAA41A"  # orange — No initial access to ball (trajectory line)
    LABEL_COLOR_CTRL = "#cb7c29"  # darker orange — for text labels only
    label_colors = {f"Magnet block (n={n_test})": COLOR_TEST, f"Control (n={n_control})": COLOR_CTRL}

    # Calculate time bin edges in minutes
    time_min_sec = data[time_col].min()
    time_max_sec = data[time_col].max()
    bin_width_min = (time_max_sec - time_min_sec) / n_bins / 60.0
    time_min_min = time_min_sec / 60.0

    # Gray shading for first two time bins with a white gap between shades
    GAP_MIN = 0.2
    ax.axvspan(
        time_min_min, time_min_min + bin_width_min - GAP_MIN / 2, color="lightgray", alpha=0.5, zorder=0, linewidth=0
    )
    ax.axvspan(
        time_min_min + bin_width_min + GAP_MIN / 2,
        time_min_min + 2 * bin_width_min,
        color="lightgray",
        alpha=0.5,
        zorder=0,
        linewidth=0,
    )

    # Compute bootstrapped 95% CI per timepoint across flies, then plot manually.
    # 1. Get the sorted unique timepoints (in minutes) present after downsampling.
    # 2. For each timepoint, collect per-fly values and bootstrap the mean CI.
    N_BOOT = 2000
    rng_boot = np.random.default_rng(42)

    def _boot_ci(values, n_boot=N_BOOT, ci=95):
        if len(values) < 2:
            m = float(np.mean(values))
            return m, m, m
        boot_means = np.sort([np.mean(rng_boot.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
        lo = np.percentile(boot_means, (100 - ci) / 2)
        hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
        return float(np.mean(values)), lo, hi

    groups_order = [f"Magnet block (n={n_test})", f"Control (n={n_control})"]
    group_colors = {groups_order[0]: COLOR_TEST, groups_order[1]: COLOR_CTRL}

    for grp_lbl in groups_order:
        grp_data = data_ds[data_ds[group_col_plot] == grp_lbl]
        timepoints = np.sort(grp_data[time_col_min].unique())

        means, ci_lo, ci_hi = [], [], []
        for t in timepoints:
            # Per-fly mean at this timepoint to keep the unit of observation = fly
            fly_vals = grp_data[grp_data[time_col_min] == t].groupby(subject_col)[value_col].mean().values
            m, lo, hi = _boot_ci(fly_vals)
            means.append(m)
            ci_lo.append(lo)
            ci_hi.append(hi)

        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)
        color = group_colors[grp_lbl]

        ax.fill_between(timepoints, ci_lo, ci_hi, color=color, alpha=0.3, linewidth=0, zorder=1)
        ax.plot(timepoints, means, color=color, linewidth=1.0, zorder=2)

    # Axis limits and ticks
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["0", "1", "2", "3", "4"])

    xticks = list(range(60, 121, 10))
    xticklabels = [str(x) if x in (60, 120) else "" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Black significance asterisks — inside the gray shadings, at the top of the shaded area
    if permutation_results is not None:
        significant_bins = permutation_results["significant_timepoints"]
        y_star = 4.3  # near top of ylim (0–4.5), inside the gray bands

        for time_bin in range(n_bins):
            # Only annotate the first two bins (shaded region)
            if time_bin >= 2:
                break
            bin_center_min = time_min_min + time_bin * bin_width_min + bin_width_min / 2
            p_value = permutation_results["p_values"][time_bin]

            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = ""

            if significance and time_bin in significant_bins:
                ax.text(
                    bin_center_min,
                    y_star,
                    significance,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                    fontname="Arial",
                )

    # Axis labels — tight padding
    ax.set_xlabel("Time (min)", fontsize=7, fontname="Arial", labelpad=2)
    ax.set_ylabel("Distance ball\npushed (mm)", fontsize=7, fontname="Arial", labelpad=2)

    # Tick appearance — reduce pad to minimise whitespace
    ax.tick_params(axis="both", direction="out", length=2, width=0.6, labelsize=6, pad=1)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Arial")

    # Condition labels in the bottom right (manual legend lines)
    ax.text(
        0.99,
        0.14,
        "No initial access to ball",
        ha="right",
        va="bottom",
        fontsize=6,
        color=LABEL_COLOR_CTRL,
        fontname="Arial",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.05,
        "Access to immobile ball",
        ha="right",
        va="bottom",
        fontsize=6,
        color=COLOR_TEST,
        fontname="Arial",
        transform=ax.transAxes,
    )

    # n= annotations: bootstrapped 95% CI at the right edge (last 5 min), matching the plotted band.
    right_data = data_ds[data_ds[time_col_min] >= (data_ds[time_col_min].max() - 5)]

    blue_fly_vals = (
        right_data[right_data[group_col_plot] == groups_order[0]].groupby(subject_col)[value_col].mean().values
    )
    ctrl_fly_vals = (
        right_data[right_data[group_col_plot] == groups_order[1]].groupby(subject_col)[value_col].mean().values
    )

    blue_mean, blue_ci_lo, blue_ci_hi = _boot_ci(blue_fly_vals)
    ctrl_mean, ctrl_ci_lo, ctrl_ci_hi = _boot_ci(ctrl_fly_vals)

    # Blue (Access to immobile ball): above the CI band top
    ax.text(
        119.5,
        blue_ci_hi + 0.3,
        f"n={n_test} flies",
        ha="right",
        va="bottom",
        fontsize=5,
        color=COLOR_TEST,
        fontname="Arial",
    )
    # Orange (No initial access to ball): below the CI band bottom
    ax.text(
        119.5,
        ctrl_ci_lo - 0.3,
        f"n={n_control} flies",
        ha="right",
        va="top",
        fontsize=5,
        color=LABEL_COLOR_CTRL,
        fontname="Arial",
    )

    # Spine and background styling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # Set exact axes dimensions: 4.56 cm wide × 2.28 cm tall.
    # tight_layout arranges elements; we then scale the figure so the axes area
    # is exactly the desired physical size (fractions stay fixed after tight_layout).
    AX_W_CM = 4.56
    AX_H_CM = 2.28
    plt.tight_layout(pad=0.2)
    ax_pos = ax.get_position()
    new_fig_w_in = (AX_W_CM / 2.54) / ax_pos.width
    new_fig_h_in = (AX_H_CM / 2.54) / ax_pos.height
    fig.set_size_inches(new_fig_w_in, new_fig_h_in)

    # Move x-label closer to the axis: set_label_coords bypasses labelpad and
    # positions the label in axes fraction coordinates.  The value is set after
    # the figure is resized so the axes-fraction ↔ physical-size mapping is final.
    # -0.10 puts the label just below the tick labels for the small font / tight layout.
    ax.xaxis.set_label_coords(0.5, -0.10)

    # Save with bbox_inches="tight" so nothing outside the axes area is clipped.
    if output_path:
        pdf_path = output_path.with_suffix(".pdf")
        png_path = output_path.with_suffix(".png")

        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

        print(f"  Saved PDF: {pdf_path}")
        print(f"  Saved PNG: {png_path}")

    plt.close()


def generate_trajectory_plot(data, n_bins=12, n_permutations=10000, output_dir=None, show_progress=True):
    """
    Generate trajectory plot for Magnet vs non-Magnet experiments.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    n_bins : int
        Number of time bins
    n_permutations : int
        Number of permutations for testing
    output_dir : Path or str
        Output directory
    show_progress : bool
        Show progress bars
    """
    if output_dir is None:
        output_dir = figure_output_dir("Figure2", __file__)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING MAGNETBLOCK TRAJECTORY PLOT")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
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

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="Magnet", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation test
    print(f"\nComputing permutation test ({n_permutations} permutations)...")

    permutation_results = compute_permutation_test(
        processed,
        metric="avg_distance_ball_0",
        group_col="Magnet",
        control_group=control_group,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    # Generate plot
    print(f"\nGenerating trajectory plot...")
    output_path = output_dir / f"trajectory_{test_group}_vs_{control_group}.pdf"

    create_trajectory_plot(
        data,
        time_col="time",
        value_col="distance_ball_0",
        group_col="Magnet",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,
        output_path=output_path,
    )

    # Save statistical results to markdown
    stats_file_md = output_dir / "trajectory_permutation_statistics.md"
    with open(stats_file_md, "w") as f:
        f.write("# MagnetBlock Trajectory Permutation Test Results\n\n")
        f.write(f"**Control group:** {permutation_results['control_group']}\n\n")
        f.write(f"**Test group:** {permutation_results['test_group']}\n\n")
        f.write(f"**Number of permutations:** {n_permutations}\n\n")
        f.write(f"**Number of time bins:** {n_bins}\n\n")
        f.write(f"**Significance level:** α = 0.05\n\n")
        f.write(f"**Significant bins:** {permutation_results['n_significant']}/{n_bins}\n\n")
        f.write(f"**Significant time bins:** {list(permutation_results['significant_timepoints'])}\n\n")

        # Write markdown table
        f.write("## Statistical Results by Time Bin\n\n")
        f.write(
            "| Time Bin | Difference (mm) | % Change | 95% CI Lower (mm) | 95% CI Upper (mm) | 95% CI % Lower | 95% CI % Upper | Cohen's D | P-value | Sig |\n"
        )
        f.write(
            "|----------|----------------|----------|-------------------|-------------------|----------------|----------------|-----------|---------|-----|\n"
        )

        for i, time_bin in enumerate(permutation_results["time_bins"]):
            obs_diff = permutation_results["observed_diffs"][i]
            pct_change = permutation_results["pct_change"][i]
            ci_lower = permutation_results["ci_lower"][i]
            ci_upper = permutation_results["ci_upper"][i]
            pct_ci_lower = permutation_results["pct_ci_lower"][i]
            pct_ci_upper = permutation_results["pct_ci_upper"][i]
            cohens_d = permutation_results["cohens_d"][i]
            p_val = permutation_results["p_values"][i]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            # Format p-value
            p_val_str = format_p_value(p_val, n_permutations)

            f.write(
                f"| {time_bin} | {obs_diff:.3f} | {pct_change:.2f}% | {ci_lower:.3f} | {ci_upper:.3f} | {pct_ci_lower:.2f}% | {pct_ci_upper:.2f}% | {cohens_d:.3f} | {p_val_str} | {sig} |\n"
            )

    print(f"\n✅ Statistical results saved to: {stats_file_md}")

    # Also save as CSV for easier data analysis
    stats_file_csv = output_dir / "trajectory_permutation_statistics.csv"
    import pandas as pd

    stats_df = pd.DataFrame(
        {
            "Time_Bin": permutation_results["time_bins"],
            "Difference_mm": permutation_results["observed_diffs"],
            "Pct_Change": permutation_results["pct_change"],
            "CI_Lower_mm": permutation_results["ci_lower"],
            "CI_Upper_mm": permutation_results["ci_upper"],
            "Pct_CI_Lower": permutation_results["pct_ci_lower"],
            "Pct_CI_Upper": permutation_results["pct_ci_upper"],
            "Cohens_D": permutation_results["cohens_d"],
            "P_value": permutation_results["p_values"],
            "Significant": [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                for p in permutation_results["p_values"]
            ],
        }
    )
    stats_df.to_csv(stats_file_csv, index=False)
    print(f"✅ Statistical results saved to: {stats_file_csv}")

    print(f"\n{'='*60}")
    print("✅ Trajectory plot generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball trajectory plot for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_magnetblock_trajectories.py
  python plot_magnetblock_trajectories.py --n-bins 15 --n-permutations 5000
  python plot_magnetblock_trajectories.py --output-dir /path/to/output
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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save plots. Defaults to "
            "$BALLPUSHING_FIGURES_ROOT/Figure2/plot_magnetblock_trajectories/."
        ),
    )

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    args = parser.parse_args()

    # Load data
    print("Loading MagnetBlock coordinates data...")
    data = load_coordinates_dataset()

    # Generate plot
    generate_trajectory_plot(
        data,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
