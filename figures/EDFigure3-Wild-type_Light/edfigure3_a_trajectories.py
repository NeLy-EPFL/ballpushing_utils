#!/usr/bin/env python3
"""
Script to generate Extended Data Figure 3a: Ball trajectory plot for light conditions.

This script generates a trajectory visualization showing:
- Distance the ball was pushed over 2 hours
- Comparison between Light ON (orange) and Light OFF (black/gray)
- Time-series data binned into 10 min segments
- Permutation tests with FDR correction (α = 0.05)
- 95% confidence intervals (parametric, using SEM)
- Filters for FeedingState="starved_noWater" to match published results

Usage:
    python edfigure3_a_trajectories.py [--test]
"""

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Add src directory to path for imports

from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# Set matplotlib parameters for publication quality
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
        return None, None

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
            df = read_feather(file_path)
            print(f"   Shape: {df.shape}")

            # Check required columns
            required_cols = ["time", "distance_ball_0", "Light", "fly"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"   ⚠️  Skipping - missing columns: {missing}")
                continue

            # Filter for FeedingState if available (CRITICAL for matching published results)
            if "FeedingState" in df.columns:
                original_len = len(df)
                df = df[df["FeedingState"] == "starved_noWater"].copy()
                print(f"   Filtered for starved_noWater: {len(df)} rows (was {original_len})")

            # Filter to only Light on/off conditions
            df = df[df["Light"].isin(["on", "off"])].copy()
            if len(df) == 0:
                print(f"   ⚠️  Skipping - no 'on' or 'off' light conditions")
                continue

            print(f"   After filtering: {df.shape}")
            print(f"   Light conditions: {sorted(df['Light'].unique())}")
            print(f"   Unique flies: {df['fly'].nunique()}")

            # Downsample to 1 Hz by rounding time to nearest integer
            # This ensures all flies align to same time points (much faster for plotting)
            print(f"   Downsampling from {len(df)} to ~{len(df)//29} datapoints...")

            df = df.sort_values(["fly", "time"]).copy()

            # Round time to nearest integer second to align all flies
            df["time_rounded"] = df["time"].round(0).astype(int)

            # Aggregate by fly and rounded time
            downsampled = df.groupby(["fly", "time_rounded"], as_index=False).agg(
                {
                    "distance_ball_0": "mean",
                    "Light": "first",  # Should be same within group
                }
            )

            # Rename time_rounded back to time
            downsampled = downsampled.rename(columns={"time_rounded": "time"})

            print(f"   Downsampled shape: {downsampled.shape}")
            print(f"   Flies per Light condition:")
            for light in sorted(downsampled["Light"].unique()):
                n_flies = downsampled[downsampled["Light"] == light]["fly"].nunique()
                print(f"     {light}: {n_flies} flies")

            # In test mode, limit to 10 flies per condition
            if test_mode:
                limited_flies = []
                for light in downsampled["Light"].unique():
                    light_flies = downsampled[downsampled["Light"] == light]["fly"].unique()[:10]
                    limited_flies.extend(light_flies)

                downsampled = downsampled[downsampled["fly"].isin(limited_flies)].copy()
                print(f"   TEST MODE: Limited to {downsampled['fly'].nunique()} flies total")

            all_downsampled.append(downsampled)

        except Exception as e:
            print(f"   ⚠️  Error loading {file_path.name}: {e}")
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
    test_group="off",
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
    test_group : str
        Name of the test group
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level for FDR correction
    progress : bool
        Show progress bar

    Returns:
    --------
    dict
        Dictionary with test results
    """
    print(f"\n  Testing {test_group} vs {control_group}...")

    # Filter data for this comparison
    comparison_data = processed_data[processed_data[group_col].isin([control_group, test_group])]

    # Get all time bins
    time_bins = sorted(processed_data["time_bin"].unique())
    n_bins = len(time_bins)

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
        bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

        control_values = bin_data[bin_data[group_col] == control_group][metric].values
        test_values = bin_data[bin_data[group_col] == test_group][metric].values

        n_control_values.append(len(control_values))
        n_test_values.append(len(test_values))

        if len(control_values) == 0 or len(test_values) == 0:
            observed_diffs.append(np.nan)
            p_values_raw.append(1.0)
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
        obs_diff = np.mean(test_values) - np.mean(control_values)
        observed_diffs.append(obs_diff)

        mean_control = np.mean(control_values)
        mean_test = np.mean(test_values)
        mean_control_values.append(mean_control)
        mean_test_values.append(mean_test)

        # Bootstrap CI for the difference
        ci_lower, ci_upper = bootstrap_ci_difference(control_values, test_values, n_bootstrap=10000)
        ci_lower_values.append(ci_lower)
        ci_upper_values.append(ci_upper)

        # Percent change
        if mean_control != 0:
            pct_change = 100 * (mean_test - mean_control) / abs(mean_control)
            pct_ci_lower = 100 * ci_lower / abs(mean_control) if ci_lower is not None else np.nan
            pct_ci_upper = 100 * ci_upper / abs(mean_control) if ci_upper is not None else np.nan
        else:
            pct_change = np.nan
            pct_ci_lower = np.nan
            pct_ci_upper = np.nan

        pct_change_values.append(pct_change)
        pct_ci_lower_values.append(pct_ci_lower)
        pct_ci_upper_values.append(pct_ci_upper)

        # Cohen's d
        d = cohens_d(control_values, test_values)
        cohens_d_values.append(d)

        # Permutation test
        pooled = np.concatenate([control_values, test_values])
        n_control = len(control_values)
        n_test = len(test_values)

        perm_diffs = np.empty(n_permutations)
        rng = np.random.default_rng(42)

        for i in range(n_permutations):
            shuffled = rng.permutation(pooled)
            perm_control = shuffled[:n_control]
            perm_test = shuffled[n_control:]
            perm_diffs[i] = np.mean(perm_test) - np.mean(perm_control)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
        p_values_raw.append(p_value)

    # Apply FDR correction across all time bins
    p_values_raw = np.array(p_values_raw)
    rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

    # Store results
    results = {
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

    print(f"    Raw significant bins: {results['n_significant_raw']}/{n_bins}")
    print(f"    FDR significant bins: {results['n_significant']}/{n_bins} (α={alpha})")

    return results


def create_trajectory_plot(
    data,
    control_condition="on",
    test_condition="off",
    time_col="time",
    value_col="distance_ball_0",
    group_col="Light",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
):
    """
    Create Extended Data Figure 3a: trajectory plot comparing light conditions.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    control_condition : str
        Name of the control light condition (ON)
    test_condition : str
        Name of the test light condition (OFF)
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
    # Filter data for this comparison
    subset_data = data[data[group_col].isin([control_condition, test_condition])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {test_condition} vs {control_condition}")
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

    # Color palette - Light ON = orange, Light OFF = black/gray
    colors = {control_condition: "orange", test_condition: "black"}
    line_styles = {control_condition: "solid", test_condition: "solid"}

    # Error band colors
    error_colors = {control_condition: "orange", test_condition: "dimgray"}
    error_alphas = {control_condition: 0.25, test_condition: 0.25}

    # Get sample sizes for labels
    n_control = subset_data[subset_data[group_col] == control_condition][subject_col].nunique()
    n_test = subset_data[subset_data[group_col] == test_condition][subject_col].nunique()

    labels = {
        control_condition: f"Light {control_condition.upper()} (n={n_control})",
        test_condition: f"Light {test_condition.upper()} (n={n_test})",
    }

    # Compute mean trajectories to find baseline for y-axis shift
    all_means = []
    for light in [control_condition, test_condition]:
        light_data = subset_data[subset_data[group_col] == light]
        time_grouped = light_data.groupby(time_col)[value_col_plot].mean().reset_index()
        all_means.extend(time_grouped[value_col_plot].values)

    # Find the minimum mean value to use as y-axis baseline
    y_baseline = min(all_means)

    # Plot mean trajectories with bootstrapped 95% confidence intervals
    for light in [control_condition, test_condition]:
        light_data = subset_data[subset_data[group_col] == light]

        # Aggregate per fly at each time point
        fly_aggregated = light_data.groupby([subject_col, time_col])[value_col_plot].mean().reset_index()

        # Get unique time points
        time_points = sorted(fly_aggregated[time_col].unique())

        means = []
        ci_lowers = []
        ci_uppers = []

        # For efficiency, compute bootstrap CIs only for every 10th time point
        # then interpolate (or just show fewer error bars)
        # With 1Hz data over 120 min, we have ~7200 time points
        # Computing bootstrap for all would be very slow
        print(f"    Computing CIs for {light} ({len(time_points)} time points)...")

        for t in time_points:
            t_data = fly_aggregated[fly_aggregated[time_col] == t][value_col_plot].values

            if len(t_data) > 1:
                # Use parametric CI for speed (much faster than bootstrap)
                # For bootstrap, we'd need 10,000 samples * 7200 time points = 72M operations
                mean_val = np.mean(t_data)
                sem = stats.sem(t_data)
                # 95% CI using t-distribution
                ci_lower = mean_val - 1.96 * sem
                ci_upper = mean_val + 1.96 * sem
            else:
                mean_val = np.mean(t_data) if len(t_data) > 0 else np.nan
                ci_lower = mean_val
                ci_upper = mean_val

            means.append(mean_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        # Shift to start at 0
        means_shifted = np.array(means) - y_baseline
        ci_lowers_shifted = np.array(ci_lowers) - y_baseline
        ci_uppers_shifted = np.array(ci_uppers) - y_baseline

        # Plot mean line
        ax.plot(
            time_points,
            means_shifted,
            color=colors[light],
            linestyle=line_styles[light],
            linewidth=2,
            label=labels[light],
        )

        # Plot confidence interval
        ax.fill_between(
            time_points,
            ci_lowers_shifted,
            ci_uppers_shifted,
            color=error_colors[light],
            alpha=error_alphas[light],
            edgecolor="none",
        )

    # Draw vertical dotted lines for time bins
    time_min = subset_data[time_col].min()
    time_max = subset_data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(x=edge, color="gray", linestyle=":", alpha=0.3, linewidth=1)

    # Get current y-axis limits and extend for significance annotations
    y_max_shifted = max(all_means) - y_baseline
    y_range = y_max_shifted

    # Set y-axis limits - start at 0, add modest space at top for annotations
    ax.set_ylim(0, y_max_shifted + 0.08 * y_range)

    # Annotate significance levels
    if permutation_results is not None:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p_values = permutation_results["p_values_corrected"]

        # Position annotations at top of plot
        y_annot = y_max_shifted + 0.04 * y_range

        for bin_idx, p_val in enumerate(p_values):
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            else:
                continue

            ax.text(
                bin_centers[bin_idx],
                y_annot,
                marker,
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    # Formatting
    ax.set_xlabel("Time (min)", fontsize=font_size_labels)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=font_size_labels)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.legend(fontsize=font_size_legend, loc="upper left")
    ax.grid(False)

    plt.tight_layout()

    # Save plot
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved trajectory plot to: {output_path}")

    plt.close()


def main():
    """Main function for Extended Data Figure 3a"""
    parser = argparse.ArgumentParser(
        description="Generate Extended Data Figure 3a: Light condition trajectory plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--test", action="store_true", help="Test mode: load limited data for quick verification")

    args = parser.parse_args()

    # Output directory
    output_dir = figure_output_dir("EDFigure3", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extended Data Figure 3a: Light Condition Trajectories")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Load data
    coordinates_dir = dataset("Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/Other/coordinates")

    print("\nLoading coordinates data...")
    start_time = time.time()
    data = load_coordinates_incrementally(coordinates_dir, test_mode=args.test)
    elapsed = time.time() - start_time
    print(f"✅ Data loaded in {elapsed:.1f} seconds")

    # Convert time from seconds to minutes
    print("\nConverting time from seconds to minutes...")
    data = data.copy()
    data["time"] = data["time"] / 60.0
    print(f"Time range: {data['time'].min():.2f} to {data['time'].max():.2f} minutes")

    # Parameters for 10-minute bins over 2 hours
    n_bins = 12  # 120 minutes / 10 minutes per bin

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins (10 min each)...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="Light", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation tests with FDR correction
    print(f"\nComputing permutation tests (10,000 permutations)...")
    print(f"Using FDR correction across {n_bins} time bins...")

    permutation_results = compute_permutation_test_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="Light",
        control_group="on",
        test_group="off",
        n_permutations=10000,
        alpha=0.05,
        progress=True,
    )

    # Save statistical results
    stats_file = output_dir / "edfigure3a_trajectory_statistics.csv"
    stats_rows = []

    for bin_idx in permutation_results["time_bins"]:
        stats_rows.append(
            {
                "time_bin": bin_idx + 1,
                "bin_start_min": processed[processed["time_bin"] == bin_idx]["bin_start"].iloc[0],
                "bin_end_min": processed[processed["time_bin"] == bin_idx]["bin_end"].iloc[0],
                "n_light_on": permutation_results["n_control"][bin_idx],
                "n_light_off": permutation_results["n_test"][bin_idx],
                "mean_light_on_mm": permutation_results["mean_control"][bin_idx] / PIXELS_PER_MM,
                "mean_light_off_mm": permutation_results["mean_test"][bin_idx] / PIXELS_PER_MM,
                "difference_mm": permutation_results["observed_diffs"][bin_idx] / PIXELS_PER_MM,
                "ci_lower_mm": (
                    permutation_results["ci_lower"][bin_idx] / PIXELS_PER_MM
                    if permutation_results["ci_lower"][bin_idx] is not None
                    else np.nan
                ),
                "ci_upper_mm": (
                    permutation_results["ci_upper"][bin_idx] / PIXELS_PER_MM
                    if permutation_results["ci_upper"][bin_idx] is not None
                    else np.nan
                ),
                "cohens_d": permutation_results["cohens_d"][bin_idx],
                "p_value_raw": permutation_results["p_values_raw"][bin_idx],
                "p_value_fdr": permutation_results["p_values_corrected"][bin_idx],
                "significant": permutation_results["p_values_corrected"][bin_idx] < 0.05,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✅ Statistics saved to: {stats_file}")

    # Print significant bins
    sig_bins = stats_df[stats_df["significant"]]
    if len(sig_bins) > 0:
        print(f"\nSignificant time bins (FDR-corrected p < 0.05):")
        for _, row in sig_bins.iterrows():
            print(
                f"  Bin {int(row['time_bin'])}: {row['bin_start_min']:.1f}-{row['bin_end_min']:.1f} min, p={row['p_value_fdr']:.4f}"
            )
    else:
        print("\nNo significant time bins found")

    # Generate trajectory plot
    print(f"\nGenerating trajectory plot...")
    plot_file = output_dir / "edfigure3a_light_trajectories.pdf"

    create_trajectory_plot(
        data,
        control_condition="on",
        test_condition="off",
        time_col="time",
        value_col="distance_ball_0",
        group_col="Light",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,
        output_path=plot_file,
    )

    print(f"\n{'='*60}")
    print("✅ Extended Data Figure 3a generated successfully!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  - Plot: {plot_file}")
    print(f"  - Statistics: {stats_file}")


if __name__ == "__main__":
    main()
