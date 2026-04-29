#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for ball types experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- FDR-corrected permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between ctrl (control) and test ball types
- Individual fly trajectories with mean overlay
- Incremental dataset loading to avoid RAM overload
- Downsampling to 1 datapoint per second (aligned to integer seconds)

Usage:
    python plot_balltypes_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH] [--test]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/Ballpushing_Balltypes/Plots/Trajectories)
    --test: Test mode - only load first 2 datasets and limit to 10 flies per condition for quick verification
"""

import argparse
import sys
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # To avoid type 3 fonts in PDFs
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from ballpushing_utils import read_feather


# Pixel to mm conversion factor (500 px = 30 mm)
PIXELS_PER_MM = 500 / 30


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


def load_dark_olfaction_data(coordinates_path, test_mode=False):
    """
    Load dark olfaction coordinates dataset.

    Parameters:
    -----------
    coordinates_path : str or Path
        Path to the coordinates feather file
    test_mode : bool
        If True, only load first subset of data

    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed coordinates dataset
    """
    coordinates_path = Path(coordinates_path)

    if not coordinates_path.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coordinates_path}")

    print(f"\n{'='*60}")
    print(f"LOADING COORDINATES DATASET")
    print(f"{'='*60}")
    print(f"Loading from: {coordinates_path.name}")

    try:
        df = read_feather(coordinates_path)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

    # Check for required columns
    required_cols = ["time", "distance_ball_0", "fly", "BallType"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing columns: {missing}")

    print(f"✅ All required columns present")
    print(f"  Unique ball types: {sorted(df['BallType'].unique())}")
    print(f"  Total flies: {df['fly'].nunique()}")
    print(f"  Time range: {df['time'].min():.2f}s to {df['time'].max():.2f}s")

    # Rename ball types for clarity
    print(f"\nRenaming ball types...")
    balltype_mapping = {"ctrl": "control", "sand": "sandpaper"}
    df["BallType"] = df["BallType"].map(balltype_mapping).fillna(df["BallType"])

    # Downsample to 1 datapoint per second per fly (align to integer seconds)
    print(f"Downsampling data to 1 datapoint per second per fly...")
    df = df.sort_values(["fly", "time"]).copy()
    df["time_rounded"] = df["time"].round(0).astype(int)

    # Aggregate by fly and rounded time
    downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg(
        {
            "distance_ball_0": "mean",
            "BallType": "first",
        }
    )

    # Rename time_rounded back to time for consistency
    downsampled = downsampled.rename(columns={"time_rounded": "time"})

    # Convert time to minutes
    print(f"Converting time to minutes...")
    downsampled["time"] = downsampled["time"] / 60.0

    print(f"✅ Downsampled shape: {downsampled.shape}")
    for balltype in sorted(downsampled["BallType"].unique()):
        n_flies = downsampled[downsampled["BallType"] == balltype]["fly"].nunique()
        n_points = len(downsampled[downsampled["BallType"] == balltype])
        print(f"  {balltype}: {n_flies} flies, {n_points} datapoints")

    if test_mode:
        limited_flies = []
        for balltype in downsampled["BallType"].unique():
            balltype_flies = downsampled[downsampled["BallType"] == balltype]["fly"].unique()[:10]
            limited_flies.extend(balltype_flies)

        downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
        print(f"⚠️  TEST MODE: Limited to {downsampled['fly'].nunique()} flies total")

    return downsampled


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="BallType", subject_col="fly", n_bins=12
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
        Column name for grouping (BallType)
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


def compute_permutation_test_with_fdr(
    processed_data,
    metric="avg_distance_ball_0",
    group_col="BallType",
    control_group="ctrl",
    n_permutations=10000,
    alpha=0.05,
    progress=True,
):
    """
    Compute permutation tests for each time bin with FDR correction.

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
        Significance level for FDR correction
    progress : bool
        Show progress bar

    Returns:
    --------
    dict
        Dictionary with test results for each comparison
    """
    # Get all groups
    groups = processed_data[group_col].unique()
    test_groups = [g for g in groups if g != control_group]

    if control_group not in groups:
        raise ValueError(f"Control group '{control_group}' not found in data")

    # Get all time bins
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

    results = {}

    # For each test group, compare to control across all time bins
    for test_group in test_groups:
        print(f"\n  Testing {test_group} vs {control_group}...")

        # Filter data for this comparison
        comparison_data = processed_data[processed_data[group_col].isin([control_group, test_group])]

        # Store results for each bin
        observed_diffs = []
        p_values_raw = []
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
        bin_center_values = []
        bin_start_values = []
        bin_end_values = []

        iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

        for time_bin in iterator:
            # Get data for this time bin
            bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

            # Get values for each group
            control_vals = bin_data[bin_data[group_col] == control_group][metric].values
            test_vals = bin_data[bin_data[group_col] == test_group][metric].values

            # Skip if either group is empty
            if len(control_vals) == 0 or len(test_vals) == 0:
                observed_diffs.append(np.nan)
                p_values_raw.append(1.0)
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
                if len(bin_data) > 0:
                    bin_center_values.append(float(bin_data["bin_center"].iloc[0]))
                    bin_start_values.append(float(bin_data["bin_start"].iloc[0]))
                    bin_end_values.append(float(bin_data["bin_end"].iloc[0]))
                else:
                    bin_center_values.append(np.nan)
                    bin_start_values.append(np.nan)
                    bin_end_values.append(np.nan)
                continue

            # Observed difference (test - control)
            mean_control = np.mean(control_vals)
            mean_test = np.mean(test_vals)
            obs_diff = mean_test - mean_control
            observed_diffs.append(obs_diff)

            n_control_values.append(len(control_vals))
            n_test_values.append(len(test_vals))
            mean_control_values.append(mean_control)
            mean_test_values.append(mean_test)

            ci_low, ci_high = bootstrap_ci_difference(
                control_vals, test_vals, n_bootstrap=10000, ci=95, random_state=42
            )
            ci_lower_values.append(ci_low)
            ci_upper_values.append(ci_high)

            if mean_control != 0:
                pct_change = (obs_diff / mean_control) * 100
                pct_ci_lower = (ci_low / mean_control) * 100
                pct_ci_upper = (ci_high / mean_control) * 100
            else:
                pct_change = np.nan
                pct_ci_lower = np.nan
                pct_ci_upper = np.nan

            pct_change_values.append(pct_change)
            pct_ci_lower_values.append(pct_ci_lower)
            pct_ci_upper_values.append(pct_ci_upper)
            cohens_d_values.append(cohens_d(control_vals, test_vals))

            bin_center_values.append(float(bin_data["bin_center"].iloc[0]))
            bin_start_values.append(float(bin_data["bin_start"].iloc[0]))
            bin_end_values.append(float(bin_data["bin_end"].iloc[0]))

            # Permutation test
            combined = np.concatenate([control_vals, test_vals])
            n_control = len(control_vals)
            n_test = len(test_vals)

            perm_diffs = []
            for _ in range(n_permutations):
                # Shuffle and split
                shuffled = np.random.permutation(combined)
                perm_control = shuffled[:n_control]
                perm_test = shuffled[n_control:]

                # Compute difference
                perm_diff = np.mean(perm_test) - np.mean(perm_control)
                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)

            # Two-tailed p-value
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
            p_values_raw.append(p_value)

        # Apply FDR correction across all time bins for this comparison
        p_values_raw = np.array(p_values_raw)
        rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

        # Store results
        results[test_group] = {
            "time_bins": time_bins,
            "bin_center": bin_center_values,
            "bin_start": bin_start_values,
            "bin_end": bin_end_values,
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
            "p_values_corrected": p_values_corrected,
            "significant_timepoints": np.where(rejected)[0],
            "n_significant": np.sum(rejected),
            "n_significant_raw": np.sum(p_values_raw < alpha),
        }

        print(f"    Raw significant bins: {results[test_group]['n_significant_raw']}/{n_bins}")
        print(f"    FDR significant bins: {results[test_group]['n_significant']}/{n_bins} (α={alpha})")

    return results


def create_trajectory_plot(
    data,
    balltype_condition,
    control_condition="ctrl",
    time_col="time",
    value_col="distance_ball_0",
    group_col="BallType",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    color_mapping=None,
    ax=None,
    global_ylim=None,
):
    """
    Create a trajectory plot comparing a ball type to control.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    balltype_condition : str
        Name of the ball type to compare
    control_condition : str
        Name of the control ball type
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
    color_mapping : dict
        Color mapping for ball types
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure
    global_ylim : tuple, optional
        Global (min, max) for y-axis to ensure consistency across subplots

    Returns:
    --------
    matplotlib.axes.Axes
        The matplotlib axis with the plot
    """
    # Pixel to mm conversion factor (500 pixels = 30 mm)
    PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

    # Filter data for this comparison
    subset_data = data[data[group_col].isin([control_condition, balltype_condition])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {balltype_condition} vs {control_condition}")
        return None

    # Convert to mm (don't shift individual fly data)
    subset_data[f"{value_col}_mm"] = subset_data[value_col] / PIXELS_PER_MM
    value_col_plot = f"{value_col}_mm"

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    # Color palette
    if color_mapping is None:
        color_mapping = {control_condition: "#7f7f7f", balltype_condition: "#1f77b4"}

    colors = {
        control_condition: color_mapping.get(control_condition, "#7f7f7f"),
        balltype_condition: color_mapping.get(balltype_condition, "#1f77b4"),
    }

    # Get sample sizes for labels
    n_control = subset_data[subset_data[group_col] == control_condition][subject_col].nunique()
    n_test = subset_data[subset_data[group_col] == balltype_condition][subject_col].nunique()

    labels = {
        control_condition: control_condition,
        balltype_condition: balltype_condition,
    }

    # Compute mean trajectories to find baseline for y-axis shift
    all_means = []
    for balltype in [control_condition, balltype_condition]:
        balltype_data = subset_data[subset_data[group_col] == balltype]
        time_grouped = balltype_data.groupby(time_col)[value_col_plot].agg(["mean"]).reset_index()
        all_means.extend(time_grouped["mean"].values)

    # Find the minimum mean value to use as y-axis baseline
    y_baseline = min(all_means)

    # Plot mean trajectories with bootstrapped confidence intervals
    for balltype in [control_condition, balltype_condition]:
        balltype_data = subset_data[subset_data[group_col] == balltype]

        # First, aggregate per fly within each time bin to get one value per fly per time
        fly_aggregated = balltype_data.groupby([subject_col, time_col])[value_col_plot].mean().reset_index()

        # Get unique time points
        time_points = sorted(fly_aggregated[time_col].unique())

        means = []
        ci_lower = []
        ci_upper = []

        for time_point in time_points:
            # Get fly-level values for this time point
            fly_values = fly_aggregated[fly_aggregated[time_col] == time_point][value_col_plot].values

            if len(fly_values) > 0:
                # Calculate mean across flies
                mean_val = np.mean(fly_values)
                means.append(mean_val)

                # Bootstrap confidence intervals by resampling flies
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

        time_grouped = pd.DataFrame({time_col: time_points, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper})

        # Plot mean line with solid line for both
        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"] - y_baseline,
            color=colors[balltype],
            linestyle="solid",
            linewidth=2.5,
            label=labels[balltype],
            zorder=10,
        )

        # Plot bootstrapped 95% CI band
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=colors[balltype],
            alpha=0.2,
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = subset_data[time_col].min()
    time_max = subset_data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Get current y-axis limits and extend for significance annotations
    y_max_shifted = max(all_means) - y_baseline
    y_range = y_max_shifted

    # Set y-axis limits - start at 0, add modest space at top for annotations
    if global_ylim is not None:
        ax.set_ylim(global_ylim)
        # Use global range for annotation positioning
        y_max_shifted = global_ylim[1] - global_ylim[1] * 0.08
        y_range = global_ylim[1]
    else:
        ax.set_ylim(0, y_max_shifted + 0.08 * y_range)

    # Annotate significance levels (only red asterisks, no p-values)
    if permutation_results is not None and balltype_condition in permutation_results:
        perm_result = permutation_results[balltype_condition]

        # Position for annotations (slightly above the top of data)
        y_annotation_stars = y_max_shifted + 0.02 * y_range

        for idx in perm_result["significant_timepoints"]:
            bin_center = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            p_val = perm_result["p_values_corrected"][idx]

            # Determine significance level and marker
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            else:
                continue

            # Add significance stars only
            ax.text(
                bin_center,
                y_annotation_stars,
                marker,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="red",
            )

    # Formatting
    ax.set_xlabel("Time (min)", fontsize=14)
    ax.set_ylabel("Relative ball distance (mm)", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(False)

    return fig, ax


def generate_all_trajectory_plots(
    data, control_condition="ctrl", n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate trajectory plots for all ball types vs control.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    control_condition : str
        Name of control ball type
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
        output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Balltypes/Plots/Trajectories")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING BALL TYPE TRAJECTORY PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
    print(f"Permutations: {n_permutations}")
    print(f"Control: {control_condition}")

    # Check required columns
    required_cols = ["time", "distance_ball_0", "BallType", "fly"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get ball types
    ball_types = data["BallType"].unique()
    test_conditions = [c for c in ball_types if c != control_condition]

    print(f"\nBall types in dataset: {sorted(ball_types)}")
    print(f"Test conditions: {test_conditions}")
    print(f"Control condition: {control_condition}")

    if control_condition not in ball_types:
        raise ValueError(f"Control condition '{control_condition}' not found in data")

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="BallType", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation tests with FDR correction
    print(f"\nComputing permutation tests ({n_permutations} permutations)...")
    print(f"Using FDR correction across {n_bins} time bins for each comparison...")

    permutation_results = compute_permutation_test_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="BallType",
        control_group=control_condition,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    # Create color mapping for test ball types
    test_colors = {}
    test_colors[control_condition] = "#7f7f7f"  # Grey for control

    # Use a color palette for test ball types
    import matplotlib.cm as cm

    if len(test_conditions) > 0:
        colors = cm.tab10(np.linspace(0, 1, len(test_conditions)))
        for i, test_cond in enumerate(sorted(test_conditions)):
            test_colors[test_cond] = colors[i]

    # Generate plots for each test condition
    print(f"\nGenerating trajectory plots...")
    n_test = len(test_conditions)

    # First pass: compute global y-axis range across all comparisons
    print(f"\nComputing global y-axis range...")
    global_y_max = 0
    PIXELS_PER_MM = 500 / 30

    for test_condition in sorted(test_conditions):
        subset_data = data[data["BallType"].isin([control_condition, test_condition])].copy()
        if len(subset_data) == 0:
            continue
        subset_data["distance_ball_0_mm"] = subset_data["distance_ball_0"] / PIXELS_PER_MM

        # Compute baseline and max for this comparison
        all_means = []
        for balltype in [control_condition, test_condition]:
            balltype_data = subset_data[subset_data["BallType"] == balltype]
            if len(balltype_data) > 0:
                time_grouped = balltype_data.groupby("time")["distance_ball_0_mm"].agg(["mean"]).reset_index()
                all_means.extend(time_grouped["mean"].values)

        if all_means:
            y_baseline = min(all_means)
            y_max_shifted = max(all_means) - y_baseline
            global_y_max = max(global_y_max, y_max_shifted)

    # Add space for annotations
    global_ylim = (0, global_y_max + 0.08 * global_y_max)
    print(f"  Global y-axis range: {global_ylim}")

    for idx, test_condition in enumerate(sorted(test_conditions)):
        print(f"\n  Creating plot for {test_condition} vs {control_condition}...")

        fig_single, ax_single = create_trajectory_plot(
            data,
            balltype_condition=test_condition,
            control_condition=control_condition,
            time_col="time",
            value_col="distance_ball_0",
            group_col="BallType",
            subject_col="fly",
            n_bins=n_bins,
            permutation_results=permutation_results,
            color_mapping=test_colors,
            global_ylim=global_ylim,
        )

        if fig_single is None:
            continue

        # Save individual plots
        output_path = output_dir / f"trajectory_balltype_{test_condition}_vs_{control_condition}.pdf"
        fig_single.savefig(output_path, dpi=300, bbox_inches="tight")

        # Also save as PNG and SVG
        png_path = output_path.with_suffix(".png")
        svg_path = output_path.with_suffix(".svg")
        fig_single.savefig(png_path, dpi=300, bbox_inches="tight")
        fig_single.savefig(svg_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")

        plt.close(fig_single)

    # Create combined figure with all subplots
    print(f"\nCreating combined figure with all subplots...")
    fig_combined, axes_combined = plt.subplots(1, n_test, figsize=(12 * n_test, 7))
    if n_test == 1:
        axes_combined = [axes_combined]

    for idx, test_condition in enumerate(sorted(test_conditions)):
        print(f"  Adding subplot for {test_condition} vs {control_condition}...")

        create_trajectory_plot(
            data,
            balltype_condition=test_condition,
            control_condition=control_condition,
            time_col="time",
            value_col="distance_ball_0",
            group_col="BallType",
            subject_col="fly",
            n_bins=n_bins,
            permutation_results=permutation_results,
            color_mapping=test_colors,
            ax=axes_combined[idx],
            global_ylim=global_ylim,
        )

    # Adjust layout and save combined figure
    plt.tight_layout()

    combined_output_path = output_dir / "trajectory_balltype_all_comparisons_combined.pdf"
    fig_combined.savefig(combined_output_path, dpi=300, bbox_inches="tight")

    # Also save as PNG and SVG
    png_combined_path = combined_output_path.with_suffix(".png")
    svg_combined_path = combined_output_path.with_suffix(".svg")
    fig_combined.savefig(png_combined_path, dpi=300, bbox_inches="tight")
    fig_combined.savefig(svg_combined_path, dpi=300, bbox_inches="tight")

    print(f"\n✅ Combined figure saved to: {combined_output_path}")
    plt.close(fig_combined)

    # Save statistical results
    stats_file = output_dir / "trajectory_balltype_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("BALL TYPE TRAJECTORY PERMUTATION TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control: {control_condition}\n")
        f.write(f"Time bins: {n_bins}\n")
        f.write(f"Permutations: {n_permutations}\n")
        f.write(f"FDR method: fdr_bh\n")
        f.write(f"Alpha: 0.05\n\n")

        for test_condition in sorted(test_conditions):
            if test_condition not in permutation_results:
                continue

            result = permutation_results[test_condition]
            f.write(f"\n{test_condition} vs {control_condition}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Raw significant bins: {result['n_significant_raw']}/{n_bins}\n")
            f.write(f"FDR significant bins: {result['n_significant']}/{n_bins}\n")
            f.write(f"Significant time bins: {list(result['significant_timepoints'])}\n\n")

            f.write("Per-bin results:\n")
            for i, time_bin in enumerate(result["time_bins"]):
                obs_diff = result["observed_diffs"][i]
                p_raw = result["p_values_raw"][i]
                p_corr = result["p_values_corrected"][i]
                sig = "*" if i in result["significant_timepoints"] else ""

                f.write(f"  Bin {time_bin}: diff={obs_diff:.2f}, p_raw={p_raw:.4f}, p_FDR={p_corr:.4f} {sig}\n")

    print(f"\n✅ Statistical results saved to: {stats_file}")

    # Save concatenated CSV across all balltype-vs-control comparisons
    csv_rows = []
    for test_condition in sorted(test_conditions):
        if test_condition not in permutation_results:
            continue

        result = permutation_results[test_condition]
        df_comp = pd.DataFrame(
            {
                "Comparison": f"{test_condition} vs {control_condition}",
                "ConditionA": control_condition,
                "ConditionB": test_condition,
                "Time_Bin": result["time_bins"],
                "Bin_Start_min": result.get("bin_start", [np.nan] * len(result["time_bins"])),
                "Bin_End_min": result.get("bin_end", [np.nan] * len(result["time_bins"])),
                "Bin_Center_min": result.get("bin_center", [np.nan] * len(result["time_bins"])),
                "N_Control": result.get("n_control", [np.nan] * len(result["time_bins"])),
                "N_Test": result.get("n_test", [np.nan] * len(result["time_bins"])),
                "Mean_Control_mm": np.array(result.get("mean_control", [np.nan] * len(result["time_bins"])))
                / PIXELS_PER_MM,
                "Mean_Test_mm": np.array(result.get("mean_test", [np.nan] * len(result["time_bins"]))) / PIXELS_PER_MM,
                "Difference_mm": np.array(result["observed_diffs"]) / PIXELS_PER_MM,
                "CI_Lower_mm": np.array(result.get("ci_lower", [np.nan] * len(result["time_bins"]))) / PIXELS_PER_MM,
                "CI_Upper_mm": np.array(result.get("ci_upper", [np.nan] * len(result["time_bins"]))) / PIXELS_PER_MM,
                "Pct_Change": result.get("pct_change", [np.nan] * len(result["time_bins"])),
                "Pct_CI_Lower": result.get("pct_ci_lower", [np.nan] * len(result["time_bins"])),
                "Pct_CI_Upper": result.get("pct_ci_upper", [np.nan] * len(result["time_bins"])),
                "Cohens_D": result.get("cohens_d", [np.nan] * len(result["time_bins"])),
                "P_Value_Raw": result["p_values_raw"],
                "P_Value_FDR": result["p_values_corrected"],
            }
        )
        df_comp["Significant_FDR"] = np.where(df_comp["P_Value_FDR"] < 0.05, "*", "ns")
        csv_rows.append(df_comp)

    if csv_rows:
        stats_csv = output_dir / "trajectory_balltype_permutation_statistics.csv"
        pd.concat(csv_rows, ignore_index=True).to_csv(stats_csv, index=False, float_format="%.6f")
        print(f"✅ Concatenated CSV saved to: {stats_csv}")

    print(f"\n{'='*60}")
    print("✅ All trajectory plots generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball trajectory plots for ball types experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_balltypes_trajectories.py
  python plot_balltypes_trajectories.py --n-bins 15 --n-permutations 5000
  python plot_balltypes_trajectories.py --output-dir /path/to/output
  python plot_balltypes_trajectories.py --test  # Quick test mode
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
        default="/mnt/upramdya_data/MD/Ballpushing_Balltypes/Plots/Trajectories",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/Ballpushing_Balltypes/Plots/Trajectories)",
    )

    parser.add_argument(
        "--coordinates-path",
        type=str,
        default="/mnt/upramdya_data/MD/Ballpushing_Balltypes/Datasets/250815_17_summary_ball_types_Data/coordinates/pooled_coordinates.feather",
        help="Path to dark olfaction coordinates feather file",
    )

    parser.add_argument("--control", type=str, default="control", help="Control ball type name (default: control)")

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument(
        "--test", action="store_true", help="Test mode: only load subset of data and limit to 10 flies per condition"
    )

    args = parser.parse_args()

    # Load data
    print("Loading ball types coordinates data...")
    start_time = time.time()
    data = load_dark_olfaction_data(args.coordinates_path, test_mode=args.test)
    elapsed = time.time() - start_time
    print(f"\n✅ Data loading completed in {elapsed:.1f} seconds")

    # Generate plots
    generate_all_trajectory_plots(
        data,
        control_condition=args.control,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
