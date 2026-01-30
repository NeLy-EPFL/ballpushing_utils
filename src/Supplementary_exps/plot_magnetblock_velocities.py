#!/usr/bin/env python3
"""
Script to generate ball velocity plots over time for MagnetBlock experiments.

This script generates velocity visualizations with:
- Velocity calculation from distance data
- Rolling window smoothing
- Time-binned analysis of velocity
- Permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between Magnet and non-Magnet conditions

Usage:
    python plot_magnetblock_velocities.py [--n-bins N] [--n-permutations N] [--output-dir PATH]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/velocities)
    --rolling-window: Size of rolling window for smoothing (default: 150)
    --no-stats: Skip permutation testing and statistics visualization
    --no-smoothing: Skip rolling window smoothing of velocity data
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Conversion from pixels/frame to mm/s (assuming 30 fps)
PIXELS_PER_FRAME_TO_MM_PER_S = 0.06


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


def calculate_velocities(data, position_col="y_fly_0", time_col="time", rolling_window=150, apply_smoothing=True):
    """Calculate velocities per fly and smooth within each fly's trajectory"""
    data = data.copy()

    # Sort by fly and time for proper per-fly diff
    data = data.sort_values(by=["fly", time_col]).reset_index(drop=True)

    # Calculate velocity PER FLY
    def calc_per_fly_velocity(group):
        group = group.sort_values(time_col).reset_index(drop=True)
        group["velocity_raw"] = group[position_col].diff() / group[time_col].diff()
        return group

    data = data.groupby("fly", group_keys=False).apply(calc_per_fly_velocity)

    # Calculate speed and convert to mm/s
    data["speed_raw"] = data["velocity_raw"].abs()
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


def preprocess_velocity_data(
    data,
    time_col="time",
    velocity_col="speed_mm_s_smooth",
    group_col="Magnet",
    subject_col="fly",
    n_bins=12,
):
    """
    Preprocess velocity data by binning time and computing statistics per bin.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw velocity data
    time_col : str
        Column name for time
    velocity_col : str
        Column name for velocity values
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
    # Remove NaN velocities
    data = data.dropna(subset=[velocity_col])

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
    grouped = data.groupby([group_col, subject_col, "time_bin"])[velocity_col].mean().reset_index()
    grouped.columns = [group_col, subject_col, "time_bin", f"avg_{velocity_col}"]

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
        "significant_timepoints": significant_timepoints,
        "n_significant": n_significant,
    }

    print(f"    Significant bins: {n_significant}/{n_bins} (α={alpha})")

    return results


def create_velocity_plot(
    data,
    time_col="time",
    velocity_col="speed_mm_s_smooth",
    group_col="Magnet",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
    time_window=None,
    ax=None,
    add_title=True,
):
    """
    Create a velocity plot comparing Magnet to non-Magnet conditions.
    Aggregates data by grouping time and Magnet, calculates mean and SEM.
    Filters to time window around 60 minutes (±10 minutes) and adds vertical line at start point.

    Parameters:
    -----------
    data : pd.DataFrame
        Full velocity data
    time_col : str
        Column name for time
    velocity_col : str
        Column name for velocity/speed
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
    # Filter out NaN velocities
    data = data.dropna(subset=[velocity_col])

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
        fig, ax = plt.subplots(figsize=(12, 6))

    # Colors for each group (matching trajectory plot)
    colors = {control_group: "orange", test_group: "blue"}
    labels = {control_group: f"Control (n = {n_control})", test_group: f"Magnet block (n = {n_test})"}

    # Process each group
    for group in groups:
        group_data = data[data[group_col] == group]

        # Group by time and calculate mean and SEM
        grouped = group_data.groupby(time_col)[velocity_col].agg(["mean", "sem"]).reset_index()

        time = grouped[time_col].values
        mean_speed = grouped["mean"].values
        sem_speed = grouped["sem"].values

        # Calculate confidence bounds
        lower_bound = mean_speed - sem_speed
        upper_bound = mean_speed + sem_speed

        # Plot the mean line
        ax.plot(time, mean_speed, color=colors[group], linewidth=1, label=labels[group])

        # Plot the confidence interval as a shaded area
        ax.fill_between(time, lower_bound, upper_bound, color=colors[group], alpha=0.3)

    # Add vertical line at start point (60 minutes) if in range
    start = 60 * 60
    if (time_window is None) or (start >= data[time_col].min() and start <= data[time_col].max()):
        ax.axvline(start, color="red", linestyle="dashed", linewidth=3, label="Start Point", zorder=5)

    # Set x-axis limits
    if time_window is not None:
        ax.set_xlim(*time_window)

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

            # Draw faint dotted lines for bins
            ax.axvline(bin_start, color="gray", linestyle="dotted", alpha=0.5)
            ax.axvline(bin_end, color="gray", linestyle="dotted", alpha=0.5)

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
                    bin_start + bin_width / 2,
                    y_star,
                    significance,
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    color="red",
                    fontweight="bold",
                )

            # Add p-value for all bins (significant and non-significant)
            y_pval = y_max + 0.11 * y_range
            p_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"
            p_color = "red" if time_bin in significant_bins else "gray"
            ax.text(
                bin_start + bin_width / 2,
                y_pval,
                p_text,
                ha="center",
                va="bottom",
                fontsize=8,
                color=p_color,
            )

    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Average Speed (mm/s)", fontsize=12)
    if add_title:
        ax.set_title("Average Speed Across Flies Grouped by Magnet Positions", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set legend position to right
    # ax.legend(loc="right", fontsize=12)

    # Set legend outside the plot top right
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=12)

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


def generate_velocity_plot(
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
    Generate velocity plot for Magnet vs non-Magnet experiments.

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
        output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/velocities")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING MAGNETBLOCK VELOCITY PLOT")
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

    # Calculate velocities
    smoothing_msg = "with" if apply_smoothing else "without"
    print(f"\nCalculating velocities {smoothing_msg} rolling window smoothing...")
    data_with_velocities = calculate_velocities(
        data, position_col="y_fly_0", rolling_window=rolling_window, apply_smoothing=apply_smoothing
    )
    print(f"Velocity data shape: {data_with_velocities.shape}")

    # Initialize permutation results
    permutation_results = None

    if compute_stats:
        # Preprocess data
        print(f"\nPreprocessing velocity data into {n_bins} time bins...")
        processed = preprocess_velocity_data(
            data_with_velocities,
            time_col="time",
            velocity_col="speed_mm_s_smooth",
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

    # Generate and save full range plot
    print(f"\nGenerating full range velocity plot...")
    fig_full, ax_full = plt.subplots(figsize=(12, 6))
    create_velocity_plot(
        data_with_velocities,
        time_col="time",
        velocity_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=None,  # No stats for full plot
        output_path=None,
        time_window=None,
        ax=ax_full,
        add_title=True,
    )
    pdf_path_full = output_dir / f"velocity_{test_group}_vs_{control_group}_full.pdf"
    png_path_full = output_dir / f"velocity_{test_group}_vs_{control_group}_full.png"
    fig_full.tight_layout()
    fig_full.savefig(pdf_path_full, format="pdf", dpi=300, bbox_inches="tight")
    fig_full.savefig(png_path_full, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path_full}")
    print(f"  Saved PNG: {png_path_full}")
    plt.close(fig_full)

    # Generate and save windowed plot
    print(f"\nGenerating windowed velocity plot...")
    start = 60 * 60
    window = 10 * 60
    fig_window, ax_window = plt.subplots(figsize=(12, 6))
    create_velocity_plot(
        data_with_velocities,
        time_col="time",
        velocity_col="speed_mm_s_smooth",
        group_col="Magnet",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,
        output_path=None,
        time_window=(start - window, start + window),
        ax=ax_window,
        add_title=True,
    )
    pdf_path_window = output_dir / f"velocity_{test_group}_vs_{control_group}_window.pdf"
    png_path_window = output_dir / f"velocity_{test_group}_vs_{control_group}_window.png"
    fig_window.tight_layout()
    fig_window.savefig(pdf_path_window, format="pdf", dpi=300, bbox_inches="tight")
    fig_window.savefig(png_path_window, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path_window}")
    print(f"  Saved PNG: {png_path_window}")
    plt.close(fig_window)

    # Save statistical results if statistics were computed
    if compute_stats and permutation_results is not None:
        stats_file = output_dir / "velocity_permutation_statistics.txt"
        with open(stats_file, "w") as f:
            f.write("MagnetBlock Velocity Permutation Test Results\n")
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
    print("✅ Velocity plot generated successfully!")
    print(f"{'='*60}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate ball velocity plot for MagnetBlock experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_magnetblock_velocities.py
  python plot_magnetblock_velocities.py --n-bins 15 --rolling-window 200
  python plot_magnetblock_velocities.py --output-dir /path/to/output
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
        default="/mnt/upramdya_data/MD/MagnetBlock/Plots/velocities",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/velocities)",
    )

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument("--no-stats", action="store_true", help="Skip permutation testing and statistics visualization")

    parser.add_argument("--no-smoothing", action="store_true", help="Skip rolling window smoothing of velocity data")

    args = parser.parse_args()

    # Load data
    print("Loading MagnetBlock coordinates data...")
    data = load_coordinates_dataset()

    # Generate plot
    generate_velocity_plot(
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
