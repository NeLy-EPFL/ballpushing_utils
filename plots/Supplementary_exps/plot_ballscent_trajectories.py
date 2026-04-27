#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for ball scent experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- FDR-corrected permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between control and other ball scents
- Mean trajectories with error bands
- Downsampling to 1 datapoint per second (aligned to integer seconds)
- Combined figure with all comparisons in a subplot layout

Usage:
    python plot_ballscent_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH] [--test]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/Ball_scents/Plots/trajectories)
    --test: Test mode - limit to 10 flies per condition for quick verification
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

matplotlib.rcParams["pdf.fonttype"] = 42  # To avoid type 3 fonts in PDFs
matplotlib.rcParams["font.family"] = "Arial"

# Fixed color mapping for ball scents - matches colors from run_permutation_ballscents.py
# Ensures consistent colors across all plots
BALLSCENT_COLORS = {
    "New": "#7f7f7f",  # Grey — control (new clean ball)
    "Pre-exposed": "#1f77b4",  # Blue  (CtrlScent: pre-exposed to fly odors)
    "Washed": "#2ca02c",  # Green
    "Washed + Pre-exposed": "#ff7f0e",  # Orange
}

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


def get_ballscent_color(ballscent):
    """Get the fixed color for a ball scent condition.

    Parameters:
    -----------
    ballscent : str
        Name of the ball scent condition

    Returns:
    --------
    str
        Hex color code for the ball scent condition
    """
    return BALLSCENT_COLORS.get(ballscent, "#7f7f7f")


def normalize_ball_scent_labels(df, group_col="BallScent"):
    """Normalize/alias BallScent values to canonical factorial labels.

    Maps existing values to one of: Ctrl, CtrlScent, Washed, Scented, New, NewScent
    using substring and fuzzy matching when needed.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe")

    design_keys = ["Ctrl", "CtrlScent", "Washed", "Scented", "New", "NewScent"]
    available = pd.Series(df[group_col].dropna().unique()).astype(str).tolist()
    from difflib import get_close_matches

    mapping = {}
    for val in available:
        # Try exact match first
        if val in design_keys:
            mapping[val] = val
            continue
        # Try fuzzy matching
        matches = get_close_matches(val, design_keys, n=1, cutoff=0.6)
        if matches:
            mapping[val] = matches[0]
        else:
            # Keep original if no match
            mapping[val] = val

    # For any entries that were not mapped, try simple substring heuristics
    for val in list(mapping.keys()):
        if mapping[val] == val and val not in design_keys:
            # Try substring matching
            val_lower = val.lower()
            if "ctrl" in val_lower and "scent" in val_lower:
                mapping[val] = "CtrlScent"
            elif "ctrl" in val_lower:
                mapping[val] = "Ctrl"
            elif "scent" in val_lower and "new" not in val_lower and "wash" not in val_lower:
                mapping[val] = "Scented"
            elif "new" in val_lower and "scent" in val_lower:
                mapping[val] = "NewScent"
            elif "new" in val_lower:
                mapping[val] = "New"
            elif "wash" in val_lower and "scent" in val_lower:
                mapping[val] = "Scented"
            elif "wash" in val_lower:
                mapping[val] = "Washed"

    remapped = {k: v for k, v in mapping.items() if k != v}
    print(f"Normalizing {group_col} labels (total variants: {len(mapping)}):")
    for k, v in mapping.items():
        if k != v:
            print(f"  {k} -> {v}")

    df = df.copy()
    df[group_col] = df[group_col].map(lambda x: mapping.get(str(x), x))
    # Preserve canonical mapping in a separate column for downstream matching
    canonical_col = f"{group_col}_canonical"
    df[group_col + "_canonical"] = df[group_col]
    # Map canonical keys to descriptive factorial labels for plots/reports
    display_map = {
        "Ctrl": "Ctrl",
        "CtrlScent": "Pre-exposed",
        "Washed": "Washed",
        "Scented": "Washed + Pre-exposed",
        "New": "New",
        "NewScent": "New + Pre-exposed",
    }

    df[group_col] = df[group_col].map(lambda x: display_map.get(x, x))
    return df


def load_coordinates_dataset(test_mode=False):
    """
    Load the ball scent coordinates dataset.

    Parameters:
    -----------
    test_mode : bool
        If True, only load first subset of data for testing

    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed coordinates dataset
    """
    # Pooled ball scents dataset
    dataset_path = "/mnt/upramdya_data/MD/Ball_scents/Datasets/251103_10_summary_ballscents_Data/coordinates/pooled_coordinates.feather"

    print(f"\n{'='*60}")
    print(f"LOADING COORDINATES DATASET")
    print(f"{'='*60}")

    print(f"Loading from: {Path(dataset_path).name}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Ball scents coordinates dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Coordinates dataset not found at {dataset_path}")

    # Check for required columns
    required_cols = ["time", "distance_ball_0", "fly", "BallScent"]
    missing = [col for col in required_cols if col not in dataset.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"Available columns: {list(dataset.columns)}")
        raise ValueError(f"Missing columns: {missing}")

    print(f"✅ All required columns present")
    print(f"  Unique ball scents: {sorted(dataset['BallScent'].unique())}")
    print(f"  Total flies: {dataset['fly'].nunique()}")
    print(f"  Time range: {dataset['time'].min():.2f}s to {dataset['time'].max():.2f}s")

    # Downsample to 1 datapoint per second per fly (align to integer seconds)
    print(f"\nDownsampling data to 1 datapoint per second per fly...")
    dataset = dataset.sort_values(["fly", "time"]).copy()
    dataset["time_rounded"] = dataset["time"].round(0).astype(int)

    # Aggregate by fly and rounded time
    downsampled = dataset.groupby(["fly", "time_rounded"], as_index=False).agg(
        {
            "distance_ball_0": "mean",
            "BallScent": "first",
        }
    )

    # Rename time_rounded back to time for consistency
    downsampled = downsampled.rename(columns={"time_rounded": "time"})

    # Convert time to minutes
    print(f"Converting time to minutes...")
    downsampled["time"] = downsampled["time"] / 60.0

    print(f"✅ Downsampled shape: {downsampled.shape}")

    # Normalize BallScent labels to canonical factorial names
    print(f"\nNormalizing BallScent labels...")
    downsampled = normalize_ball_scent_labels(downsampled, group_col="BallScent")

    for scent in sorted(downsampled["BallScent"].unique()):
        n_flies = downsampled[downsampled["BallScent"] == scent]["fly"].nunique()
        n_points = len(downsampled[downsampled["BallScent"] == scent])
        print(f"  {scent}: {n_flies} flies, {n_points} datapoints")

    # Filter to only include specific ball scent conditions
    # Pre-exposed = CtrlScent (ball pre-exposed to fly odors); excludes NewScent
    allowed_scents = ["New", "Pre-exposed", "Washed", "Washed + Pre-exposed"]
    initial_shape = downsampled.shape
    downsampled = downsampled[downsampled["BallScent"].isin(allowed_scents)].copy()
    print(f"\n📊 Filtered to allowed ball scents: {allowed_scents}")
    print(f"   Shape: {initial_shape} -> {downsampled.shape}")
    print(f"   Remaining conditions: {sorted(downsampled['BallScent'].unique())}")

    if test_mode:
        limited_flies = []
        for scent in downsampled["BallScent"].unique():
            scent_flies = downsampled[downsampled["BallScent"] == scent]["fly"].unique()[:10]
            limited_flies.extend(scent_flies)

        downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
        print(f"⚠️  TEST MODE: Limited to {downsampled['fly'].nunique()} flies total")

    return downsampled


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="BallScent", subject_col="fly", n_bins=12
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
        Column name for grouping (BallScent)
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
    group_col="BallScent",
    control_group="New",
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
            bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

            # Get control and test values
            control_vals = bin_data[bin_data[group_col] == control_group][metric].values
            test_vals = bin_data[bin_data[group_col] == test_group][metric].values

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

            # Observed difference
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
                np.random.shuffle(combined)
                perm_control = combined[:n_control]
                perm_test = combined[n_control:]
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
    ball_scent,
    control_scent="New",
    time_col="time",
    value_col="distance_ball_0",
    group_col="BallScent",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    color_mapping=None,
    ax=None,
    global_ylim=None,
):
    """
    Create a trajectory plot comparing a ball scent to control.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    ball_scent : str
        Name of the ball scent to compare
    control_scent : str
        Name of the control ball scent
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
        Color mapping for ball scents
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
    subset_data = data[data[group_col].isin([control_scent, ball_scent])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {ball_scent} vs {control_scent}")
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
        color_mapping = {}
        for scent in [control_scent, ball_scent]:
            color_mapping[scent] = get_ballscent_color(scent)

    colors = {
        control_scent: color_mapping.get(control_scent, get_ballscent_color(control_scent)),
        ball_scent: color_mapping.get(ball_scent, get_ballscent_color(ball_scent)),
    }
    labels = {
        control_scent: control_scent,
        ball_scent: ball_scent,
    }

    # Compute mean trajectories to find baseline for y-axis shift
    all_means = []
    for scent in [control_scent, ball_scent]:
        scent_data = subset_data[subset_data[group_col] == scent]
        time_grouped = scent_data.groupby(time_col)[value_col_plot].agg(["mean", "sem"]).reset_index()
        all_means.extend(time_grouped["mean"].values)

    # Find the minimum mean value to use as y-axis baseline
    y_baseline = min(all_means)

    # Plot mean trajectories with bootstrapped confidence intervals
    for scent in [control_scent, ball_scent]:
        scent_data = subset_data[subset_data[group_col] == scent]

        # Aggregate per fly within each time point
        fly_aggregated = scent_data.groupby([subject_col, time_col])[value_col_plot].mean().reset_index()

        time_points = sorted(fly_aggregated[time_col].unique())
        means = []
        ci_lower = []
        ci_upper = []

        for time_point in time_points:
            fly_values = fly_aggregated[fly_aggregated[time_col] == time_point][value_col_plot].values

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

        time_grouped = pd.DataFrame({time_col: time_points, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper})

        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"] - y_baseline,
            color=colors[scent],
            linestyle="solid",
            linewidth=2.5,
            label=labels[scent],
            zorder=10,
        )

        ax.fill_between(
            time_grouped[time_col],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=colors[scent],
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
    if permutation_results is not None and ball_scent in permutation_results:
        perm_result = permutation_results[ball_scent]

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

    return ax


def create_combined_trajectory_plot(
    data,
    control_scent="New",
    time_col="time",
    value_col="distance_ball_0",
    group_col="BallScent",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
    show_individual_flies=True,
):
    """
    Create a combined trajectory plot showing all ball scent conditions together.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    control_scent : str
        Name of the control ball scent
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
    show_individual_flies : bool
        Whether to show individual fly trajectories
    """
    # Get all ball scents
    ball_scents = sorted(data[group_col].unique())

    if len(data) == 0:
        print(f"Warning: No data for combined plot")
        return

    # Pixel to mm conversion factor (500 pixels = 30 mm)
    PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm
    data = data.copy()
    data[f"{value_col}_mm"] = data[value_col] / PIXELS_PER_MM
    value_col_plot = f"{value_col}_mm"

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette - use fixed colors matching permutation tests
    color_map = {}
    for scent in ball_scents:
        color_map[scent] = get_ballscent_color(scent)

    # Plot individual fly trajectories if requested
    if show_individual_flies:
        for scent in ball_scents:
            scent_data = data[data[group_col] == scent]
            for fly in scent_data[subject_col].unique():
                fly_data = scent_data[scent_data[subject_col] == fly]
                ax.plot(
                    fly_data[time_col],
                    fly_data[value_col_plot],
                    color=color_map[scent],
                    alpha=0.15,
                    linewidth=0.8,
                )

    # Compute baseline for relative distance
    all_means = []
    for scent in ball_scents:
        scent_data = data[data[group_col] == scent]
        time_grouped = scent_data.groupby(time_col)[value_col_plot].agg(["mean"]).reset_index()
        all_means.extend(time_grouped["mean"].values)

    y_baseline = min(all_means) if all_means else 0.0

    # Plot mean trajectories with bootstrapped confidence intervals
    for scent in ball_scents:
        scent_data = data[data[group_col] == scent]

        fly_aggregated = scent_data.groupby([subject_col, time_col])[value_col_plot].mean().reset_index()
        time_points = sorted(fly_aggregated[time_col].unique())
        means = []
        ci_lower = []
        ci_upper = []

        for time_point in time_points:
            fly_values = fly_aggregated[fly_aggregated[time_col] == time_point][value_col_plot].values

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

        time_grouped = pd.DataFrame({time_col: time_points, "mean": means, "ci_lower": ci_lower, "ci_upper": ci_upper})

        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"] - y_baseline,
            color=color_map[scent],
            linewidth=3,
            label=scent,
            zorder=10,
        )

        ax.fill_between(
            time_grouped[time_col],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=color_map[scent],
            alpha=0.2,
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, zorder=1)

    # Annotate significance levels for each test condition vs control
    if permutation_results is not None:
        # Get test scents (non-control)
        test_scents = [s for s in ball_scents if s != control_scent]

        y_max_shifted = max(all_means) - y_baseline if all_means else 0.0
        y_range = y_max_shifted if y_max_shifted > 0 else 1.0

        # Offset annotations vertically for different comparisons
        for test_idx, test_scent in enumerate(test_scents):
            if test_scent not in permutation_results:
                continue

            perm_result = permutation_results[test_scent]

            # Vertical offset for this comparison (to avoid overlap)
            y_offset = 0.06 + (test_idx * 0.06)
            y_annotation = y_max_shifted + y_offset * y_range

            for idx in perm_result["significant_timepoints"]:
                bin_start = bin_edges[idx]
                bin_end = bin_edges[idx + 1]
                x_pos = (bin_start + bin_end) / 2

                p_value = perm_result["p_values_corrected"][idx]

                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = ""

                if significance:
                    # Use the test condition's color for the annotation
                    ax.annotate(
                        significance,
                        xy=(x_pos, y_annotation),
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        color="red",
                        fontweight="bold",
                        zorder=15,
                    )

    # Formatting
    ax.set_xlabel("Time (min)", fontsize=14)
    ax.set_ylabel("Relative ball distance (mm)", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(False)

    plt.tight_layout()

    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # Also save as PNG
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def generate_all_trajectory_plots(
    data, control_scent="New", n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate combined trajectory plot with all ball scents.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    control_scent : str
        Name of control ball scent
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
        output_dir = Path("/mnt/upramdya_data/MD/Ball_scents/Plots/trajectories")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING BALL SCENT TRAJECTORY PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
    print(f"Permutations: {n_permutations}")
    print(f"Control: {control_scent}")

    # Check required columns
    required_cols = ["time", "distance_ball_0", "BallScent", "fly"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get ball scents
    ball_scents = sorted(data["BallScent"].unique())
    test_scents = [s for s in ball_scents if s != control_scent]

    print(f"\nBall scents in dataset: {ball_scents}")
    print(f"Test scents: {test_scents}")
    print(f"Control scent: {control_scent}")

    if control_scent not in ball_scents:
        raise ValueError(f"Control scent '{control_scent}' not found in data")

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="BallScent", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation tests with FDR correction
    print(f"\nComputing permutation tests ({n_permutations} permutations)...")
    print(f"Using FDR correction across {n_bins} time bins for each comparison...")

    permutation_results = compute_permutation_test_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="BallScent",
        control_group=control_scent,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    # Create color mapping for all ball scents using fixed colors
    color_mapping = {}
    for scent in [control_scent] + test_scents:
        color_mapping[scent] = get_ballscent_color(scent)

    # Compute global y-axis range across all comparisons
    print(f"\nComputing global y-axis range...")
    global_y_max = 0
    PIXELS_PER_MM = 500 / 30

    for test_scent in sorted(test_scents):
        subset = data[data["BallScent"].isin([control_scent, test_scent])].copy()
        subset["distance_mm"] = subset["distance_ball_0"] / PIXELS_PER_MM

        all_means = []
        for scent in [control_scent, test_scent]:
            scent_data = subset[subset["BallScent"] == scent]
            if len(scent_data) > 0:
                time_grouped = scent_data.groupby("time")["distance_mm"].agg(["mean"]).reset_index()
                all_means.extend(time_grouped["mean"].values)

        if all_means:
            y_baseline = min(all_means)
            y_max_shifted = max(all_means) - y_baseline
            global_y_max = max(global_y_max, y_max_shifted)

    # Add space for annotations
    global_ylim = (0, global_y_max + 0.08 * global_y_max)
    print(f"  Global y-axis range: {global_ylim}")

    # Create combined figure with all subplots
    print(f"\nCreating combined figure with all comparisons...")
    n_test = len(test_scents)

    # Determine layout (prefer wide layout)
    if n_test == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 7))
        axes = [axes]
    elif n_test == 2:
        fig, axes = plt.subplots(1, 2, figsize=(24, 7))
    elif n_test == 3:
        fig, axes = plt.subplots(1, 3, figsize=(36, 7))
    else:
        # For more than 3, use 2 rows
        n_cols = (n_test + 1) // 2
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 7 * n_rows))
        axes = axes.flatten()

    # Create plot for each test scent
    for idx, test_scent in enumerate(sorted(test_scents)):
        print(f"  [{idx+1}/{n_test}] Creating plot for {test_scent} vs {control_scent}...")

        create_trajectory_plot(
            data,
            ball_scent=test_scent,
            control_scent=control_scent,
            time_col="time",
            value_col="distance_ball_0",
            group_col="BallScent",
            subject_col="fly",
            n_bins=n_bins,
            permutation_results=permutation_results,
            color_mapping=color_mapping,
            ax=axes[idx],
            global_ylim=global_ylim,
        )

    # Hide any unused subplots
    for idx in range(n_test, len(axes)):
        axes[idx].set_visible(False)

    # Adjust layout and save combined figure
    plt.tight_layout()

    combined_output_path = output_dir / "trajectory_ballscent_combined.pdf"
    fig.savefig(combined_output_path, dpi=300, bbox_inches="tight")

    # Also save as PNG and SVG
    png_combined_path = combined_output_path.with_suffix(".png")
    svg_combined_path = combined_output_path.with_suffix(".svg")
    fig.savefig(png_combined_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_combined_path, dpi=300, bbox_inches="tight")

    print(f"\n✅ Combined figure saved to:")
    print(f"   PDF: {combined_output_path}")
    print(f"   PNG: {png_combined_path}")
    print(f"   SVG: {svg_combined_path}")
    plt.close(fig)

    # Save statistical results
    stats_file = output_dir / "trajectory_ballscent_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("Ball Scent Trajectory Permutation Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control group: {control_scent}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Number of time bins: {n_bins}\n")
        f.write(f"FDR correction method: Benjamini-Hochberg\n")
        f.write(f"Significance level: α = 0.05\n\n")

        for test_scent in sorted(test_scents):
            if test_scent not in permutation_results:
                continue

            result = permutation_results[test_scent]
            f.write(f"\n{test_scent} vs {control_scent}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Raw significant bins: {result['n_significant_raw']}/{n_bins}\n")
            f.write(f"FDR significant bins: {result['n_significant']}/{n_bins}\n")
            f.write(f"Significant time bins: {list(result['significant_timepoints'])}\n\n")

            f.write("Bin | Obs. Diff | P-value (raw) | P-value (FDR) | Significant\n")
            f.write("-" * 60 + "\n")
            for i, time_bin in enumerate(result["time_bins"]):
                obs_diff = result["observed_diffs"][i]
                p_raw = result["p_values_raw"][i]
                p_fdr = result["p_values_corrected"][i]
                sig = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else "ns"
                f.write(f"{time_bin:3d} | {obs_diff:9.3f} | {p_raw:13.6f} | {p_fdr:13.6f} | {sig}\n")

    print(f"\n✅ Statistical results saved to: {stats_file}")

    # Save concatenated CSV across all ballscent-vs-control comparisons
    csv_rows = []
    for test_scent in sorted(test_scents):
        if test_scent not in permutation_results:
            continue

        result = permutation_results[test_scent]
        df_comp = pd.DataFrame(
            {
                "Comparison": f"{test_scent} vs {control_scent}",
                "ConditionA": control_scent,
                "ConditionB": test_scent,
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
        stats_csv = output_dir / "trajectory_ballscent_permutation_statistics.csv"
        pd.concat(csv_rows, ignore_index=True).to_csv(stats_csv, index=False, float_format="%.6f")
        print(f"✅ Concatenated CSV saved to: {stats_csv}")

    print(f"\n{'='*60}")
    print("✅ All trajectory plots generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball trajectory plots for ball scent experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_ballscent_trajectories.py
  python plot_ballscent_trajectories.py --n-bins 15 --n-permutations 5000
  python plot_ballscent_trajectories.py --output-dir /path/to/output
  python plot_ballscent_trajectories.py --test
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
        default="/mnt/upramdya_data/MD/Ball_scents/Plots/trajectories",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/Ball_scents/Plots/trajectories)",
    )

    parser.add_argument("--control", type=str, default="New", help="Control ball scent name (default: New)")

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - limit to 10 flies per condition for quick verification",
    )

    args = parser.parse_args()

    # Load data
    print("Loading ball scent coordinates data...")
    data = load_coordinates_dataset(test_mode=args.test)

    # Generate plots
    generate_all_trajectory_plots(
        data,
        control_scent=args.control,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
