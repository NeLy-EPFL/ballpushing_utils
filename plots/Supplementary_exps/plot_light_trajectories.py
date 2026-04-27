#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for light condition experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- FDR-corrected permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between Light ON (control) and Light OFF
- Individual fly trajectories with mean overlay
- Incremental dataset loading to avoid RAM overload
- Downsampling to 1 datapoint per second (29 frames)

Usage:
    python plot_light_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH] [--test]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_light)
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


# Pixel to mm conversion factor (500 pixels = 30 mm)
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


def load_coordinates_incrementally(coordinates_dir, test_mode=False):
    """
    Load coordinate datasets one by one, downsample, and combine.

    Parameters:
    -----------
    coordinates_dir : str or Path
        Directory containing coordinate feather files
    test_mode : bool
        If True, only load first 2 datasets and limit flies

    Returns:
    --------
    pd.DataFrame
        Combined and downsampled coordinates dataset
    """
    coordinates_dir = Path(coordinates_dir)

    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    # Find all feather files
    feather_files = sorted(list(coordinates_dir.glob("*_coordinates.feather")))

    if len(feather_files) == 0:
        raise FileNotFoundError(f"No coordinate feather files found in {coordinates_dir}")

    print(f"\n{'='*60}")
    print(f"LOADING COORDINATES DATASETS")
    print(f"{'='*60}")
    print(f"Found {len(feather_files)} coordinate files in {coordinates_dir}")

    if test_mode:
        feather_files = feather_files[:2]
        print(f"⚠️  TEST MODE: Only loading first {len(feather_files)} files")

    all_downsampled = []

    for i, file_path in enumerate(feather_files, 1):
        print(f"\n[{i}/{len(feather_files)}] Loading: {file_path.name}")

        try:
            # Load dataset
            df = pd.read_feather(file_path)
            print(f"  Original shape: {df.shape}")

            # Check for required columns
            required_cols = ["time", "distance_ball_0", "fly"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"  ⚠️  Skipping - missing columns: {missing}")
                continue

            # Check for Light column
            if "Light" not in df.columns:
                print(f"  ⚠️  Skipping - no 'Light' column")
                continue

            # Filter for FeedingState if available
            if "FeedingState" in df.columns:
                original_len = len(df)
                df = df[df["FeedingState"] == "starved_noWater"].copy()
                print(f"  Filtered for FeedingState='starved_noWater': {len(df)} rows (was {original_len})")

            # Remove empty Light values
            original_len = len(df)
            df = df[df["Light"] != ""].copy()
            if len(df) < original_len:
                print(f"  Removed {original_len - len(df)} rows with empty Light values")

            if len(df) == 0:
                print(f"  ⚠️  Skipping - no data after filtering")
                continue

            print(f"  Light conditions: {sorted(df['Light'].unique())}")
            print(f"  Unique flies: {df['fly'].nunique()}")

            # Downsample to 1 datapoint per second by rounding time to nearest integer
            # This ensures all flies are aligned to the same time points (1s, 2s, 3s, etc.)
            print(f"  Downsampling from {len(df)} to ~{len(df)//29} datapoints...")

            df = df.sort_values(["fly", "time"]).copy()

            # Round time to nearest integer second to align all flies
            df["time_rounded"] = df["time"].round(0).astype(int)

            # Aggregate by fly and rounded time
            downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg(
                {
                    "distance_ball_0": "mean",
                    "Light": "first",  # Should be same within group
                    "FeedingState": "first" if "FeedingState" in df.columns else lambda x: None,
                }
            )

            # Rename time_rounded back to time for consistency
            downsampled = downsampled.rename(columns={"time_rounded": "time"})

            print(f"  Downsampled shape: {downsampled.shape}")
            print(f"  Flies per Light condition:")
            for light in sorted(downsampled["Light"].unique()):
                n_flies = downsampled[downsampled["Light"] == light]["fly"].nunique()
                print(f"    {light}: {n_flies} flies")

            # In test mode, limit to 10 flies per condition
            if test_mode:
                limited_flies = []
                for light in downsampled["Light"].unique():
                    light_flies = downsampled[downsampled["Light"] == light]["fly"].unique()[:10]
                    limited_flies.extend(light_flies)

                downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
                print(f"  ⚠️  TEST MODE: Limited to {downsampled['fly'].nunique()} flies total")

            all_downsampled.append(downsampled)

        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")
            continue

    if len(all_downsampled) == 0:
        raise ValueError("No datasets could be loaded successfully")

    # Combine all datasets
    print(f"\n{'='*60}")
    print(f"COMBINING DATASETS")
    print(f"{'='*60}")
    print(f"Combining {len(all_downsampled)} datasets...")

    combined = pd.concat(all_downsampled, ignore_index=True)

    print(f"✅ Combined dataset shape: {combined.shape}")
    print(f"   Light conditions: {sorted(combined['Light'].unique())}")
    print(f"   Total flies: {combined['fly'].nunique()}")

    # Print sample counts per condition
    for light in sorted(combined["Light"].unique()):
        light_data = combined[combined["Light"] == light]
        n_flies = light_data["fly"].nunique()
        n_points = len(light_data)
        print(f"   {light}: {n_flies} flies, {n_points} datapoints")

    return combined


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="Light", subject_col="fly", n_bins=12
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
        Column name for grouping (Light)
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
    group_col="Light",
    control_group="on",
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

            ci_lower, ci_upper = bootstrap_ci_difference(
                control_vals, test_vals, n_bootstrap=10000, ci=95, random_state=42
            )
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
    light_condition,
    control_condition="on",
    time_col="time",
    value_col="distance_ball_0",
    group_col="Light",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
):
    """
    Create a trajectory plot comparing a light condition to control.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    light_condition : str
        Name of the light condition to compare
    control_condition : str
        Name of the control light condition
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
    # Pixel to mm conversion factor (500 pixels = 30 mm)
    PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

    # Filter data for this comparison
    subset_data = data[data[group_col].isin([control_condition, light_condition])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {light_condition} vs {control_condition}")
        return

    # Convert to mm (don't shift individual fly data)
    subset_data[f"{value_col}_mm"] = subset_data[value_col] / PIXELS_PER_MM
    value_col_plot = f"{value_col}_mm"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define font sizes for consistency
    font_size_ticks = 10
    font_size_labels = 14
    font_size_legend = 12
    font_size_annotations = 11

    # Color palette and line styles - matching permutation test plots
    colors = {control_condition: "black", light_condition: "black"}
    line_styles = {control_condition: "solid", light_condition: "dashed"}  # solid for ON, dashed for OFF

    # Different shades of gray for error bands
    error_colors = {
        control_condition: "black",  # Darker gray for ON
        light_condition: "dimgray",  # Lighter gray for OFF
    }
    error_alphas = {control_condition: 0.25, light_condition: 0.15}  # More opaque for ON  # More transparent for OFF

    # Get sample sizes for labels
    n_control = subset_data[subset_data[group_col] == control_condition][subject_col].nunique()
    n_test = subset_data[subset_data[group_col] == light_condition][subject_col].nunique()

    labels = {
        control_condition: f"Light {control_condition.upper()} (n={n_control})",
        light_condition: f"Light {light_condition.upper()} (n={n_test})",
    }

    # Compute mean trajectories to find baseline for y-axis shift
    all_means = []
    for light in [control_condition, light_condition]:
        light_data = subset_data[subset_data[group_col] == light]
        time_grouped = light_data.groupby(time_col)[value_col_plot].mean().reset_index()
        all_means.extend(time_grouped[value_col_plot].values)

    # Find the minimum mean value to use as y-axis baseline
    y_baseline = min(all_means)

    # Plot mean trajectories with bootstrapped 95% confidence intervals
    for light in [control_condition, light_condition]:
        light_data = subset_data[subset_data[group_col] == light]

        # First aggregate per fly at each time point (should already be 1 per fly, but make it explicit)
        fly_aggregated = light_data.groupby([subject_col, time_col])[value_col_plot].mean().reset_index()

        # Compute mean and bootstrapped 95% CI for each time point
        time_stats = []
        for t in sorted(fly_aggregated[time_col].unique()):
            # Get one value per fly at this time point
            fly_values = fly_aggregated[fly_aggregated[time_col] == t][value_col_plot].values
            if len(fly_values) == 0:
                continue

            # Compute mean across flies
            mean_val = np.mean(fly_values)

            # Bootstrap 95% CI by resampling flies
            if len(fly_values) > 1:
                n_bootstrap = 1000
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    # Resample flies with replacement
                    bootstrap_flies = np.random.choice(fly_values, size=len(fly_values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_flies))

                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
            else:
                # Single fly: no CI
                ci_lower = mean_val
                ci_upper = mean_val

            time_stats.append({time_col: t, "mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper})

        time_grouped = pd.DataFrame(time_stats)

        # Plot mean line with appropriate line style
        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"] - y_baseline,  # Shift by baseline
            color=colors[light],
            linestyle=line_styles[light],
            linewidth=2.5,
            label=labels[light],
            zorder=10,
        )

        # Plot bootstrapped 95% confidence interval
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["ci_lower"] - y_baseline,
            time_grouped["ci_upper"] - y_baseline,
            color=error_colors[light],
            alpha=error_alphas[light],
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = subset_data[time_col].min()
    time_max = subset_data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Get current y-axis limits and extend for significance annotations
    # Compute range from shifted data
    y_max_shifted = max(all_means) - y_baseline
    y_range = y_max_shifted

    # Set y-axis limits - start at 0, add modest space at top for annotations
    ax.set_ylim(0, y_max_shifted + 0.08 * y_range)

    # Annotate significance levels - matching magnetblock speed plot style
    if permutation_results is not None and light_condition in permutation_results:
        perm_result = permutation_results[light_condition]

        # Position for annotations (slightly above the top of data)
        y_annotation_stars = y_max_shifted + 0.02 * y_range
        y_annotation_pval = y_max_shifted + 0.05 * y_range

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

            # Add significance stars
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
    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=font_size_labels)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.legend(fontsize=font_size_legend, loc="upper left")
    # Remove background grid - only keep the time bin vertical lines
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
    data, control_condition="on", n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate trajectory plots for all light conditions vs control.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    control_condition : str
        Name of control light condition
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
        output_dir = Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_light")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING LIGHT CONDITION TRAJECTORY PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
    print(f"Permutations: {n_permutations}")
    print(f"Control: Light {control_condition.upper()}")

    # Check required columns
    required_cols = ["time", "distance_ball_0", "Light", "fly"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get light conditions
    light_conditions = data["Light"].unique()
    test_conditions = [c for c in light_conditions if c != control_condition]

    print(f"\nLight conditions in dataset: {sorted(light_conditions)}")
    print(f"Test conditions: {test_conditions}")
    print(f"Control condition: {control_condition}")

    if control_condition not in light_conditions:
        raise ValueError(f"Control condition '{control_condition}' not found in data")

    # Convert time from seconds to minutes before preprocessing
    print(f"\nConverting time from seconds to minutes...")
    data = data.copy()
    data["time"] = data["time"] / 60.0
    print(f"Time range: {data['time'].min():.2f} to {data['time'].max():.2f} minutes")

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="Light", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation tests with FDR correction
    print(f"\nComputing permutation tests ({n_permutations} permutations)...")
    print(f"Using FDR correction across {n_bins} time bins for each comparison...")

    permutation_results = compute_permutation_test_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="Light",
        control_group=control_condition,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    # Generate plots for each test condition
    print(f"\nGenerating trajectory plots...")
    for test_condition in test_conditions:
        print(f"\n  Creating plot for Light {test_condition.upper()} vs Light {control_condition.upper()}...")

        output_path = output_dir / f"trajectory_light_{test_condition}_vs_{control_condition}.pdf"

        create_trajectory_plot(
            data,
            light_condition=test_condition,
            control_condition=control_condition,
            time_col="time",
            value_col="distance_ball_0",
            group_col="Light",
            subject_col="fly",
            n_bins=n_bins,
            permutation_results=permutation_results,
            output_path=output_path,
        )

    # Save statistical results
    stats_file = output_dir / "trajectory_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("LIGHT CONDITION TRAJECTORY PERMUTATION TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control: Light {control_condition.upper()}\n")
        f.write(f"Time bins: {n_bins}\n")
        f.write(f"Permutations: {n_permutations}\n")
        f.write(f"FDR method: fdr_bh\n")
        f.write(f"Alpha: 0.05\n\n")

        for test_condition in test_conditions:
            if test_condition not in permutation_results:
                continue

            result = permutation_results[test_condition]
            f.write(f"\nLight {test_condition.upper()} vs Light {control_condition.upper()}\n")
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

    # Save publication-style CSV statistics (per comparison + combined)
    combined_rows = []
    for test_condition in test_conditions:
        if test_condition not in permutation_results:
            continue

        result = permutation_results[test_condition]
        df_comp = pd.DataFrame(
            {
                "Comparison": f"{test_condition} vs {control_condition}",
                "ConditionA": control_condition,
                "ConditionB": test_condition,
                "Time_Bin": result["time_bins"],
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

        comp_csv = output_dir / f"trajectory_light_stats_{test_condition}_vs_{control_condition}.csv"
        df_comp.to_csv(comp_csv, index=False, float_format="%.6f")
        print(f"✅ Comparison CSV saved: {comp_csv}")

        combined_rows.append(df_comp)

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_csv = output_dir / "trajectory_light_stats_all_comparisons.csv"
        combined_df.to_csv(combined_csv, index=False, float_format="%.6f")
        print(f"✅ Combined CSV saved: {combined_csv}")

    print(f"\n{'='*60}")
    print("✅ All trajectory plots generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball trajectory plots for light condition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_light_trajectories.py
  python plot_light_trajectories.py --n-bins 15 --n-permutations 5000
  python plot_light_trajectories.py --output-dir /path/to/output
  python plot_light_trajectories.py --test  # Quick test mode
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
        default="/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_light",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/trajectories_light)",
    )

    parser.add_argument(
        "--coordinates-dir",
        type=str,
        default="/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/Other/coordinates",
        help="Directory containing coordinate feather files",
    )

    parser.add_argument("--control", type=str, default="on", help="Control light condition name (default: on)")

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument(
        "--test", action="store_true", help="Test mode: only load 2 datasets and 10 flies per condition"
    )

    args = parser.parse_args()

    # Load data incrementally
    print("Loading light condition coordinates data incrementally...")
    start_time = time.time()
    data = load_coordinates_incrementally(args.coordinates_dir, test_mode=args.test)
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
