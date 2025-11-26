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


def load_coordinates_dataset():
    """Load the MagnetBlock experiments coordinates dataset"""
    dataset_path = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/251125_14_summary_magnet_block_folders_Data/coordinates/pooled_coordinates.feather"

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
    Shows only mean trajectories with 95% confidence intervals.

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
    # Assuming alphabetical order: "Magnet" comes before "non-Magnet"
    if "Magnet" in groups and "non-Magnet" in groups:
        control_group = "non-Magnet"
        test_group = "Magnet"
    else:
        control_group = groups[0]
        test_group = groups[1]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette - red for control, blue for test
    colors = {control_group: "#e74c3c", test_group: "#3498db"}

    # Find earliest interaction event onset after 3600 seconds
    earliest_onset_time = None
    if "interaction_event_onset" in data.columns:
        # Get all rows with interaction event onsets (non-NaN) and time >= 3600
        onset_data = data[(data["interaction_event_onset"].notna()) & (data[time_col] >= 3600)]
        if len(onset_data) > 0:
            earliest_onset_time = onset_data[time_col].min()
            print(f"  Earliest interaction event onset after 3600s: {earliest_onset_time:.1f}s")
        else:
            print("  Warning: No interaction event onsets found after 3600s")
    else:
        print("  Warning: 'interaction_event_onset' column not found in data")

    # Plot mean trajectories with 95% confidence intervals
    for group in groups:
        group_data = data[data[group_col] == group]

        # Group by time to compute mean and 95% CI
        grouped = group_data.groupby(time_col)[value_col]

        # Calculate statistics
        means = grouped.mean()
        sems = grouped.sem()
        counts = grouped.count()

        # Calculate 95% CI using t-distribution
        ci_values = sems * counts.apply(lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else 0)

        # Create DataFrame
        time_grouped = pd.DataFrame({"time": means.index, "mean": means.values, "ci": ci_values.values})

        # Plot mean line
        ax.plot(time_grouped["time"], time_grouped["mean"], color=colors[group], linewidth=3, label=group, zorder=10)

        # Plot 95% confidence interval
        ax.fill_between(
            time_grouped["time"],
            time_grouped["mean"] - time_grouped["ci"],
            time_grouped["mean"] + time_grouped["ci"],
            color=colors[group],
            alpha=0.3,
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Annotate significance levels
    if permutation_results is not None:
        y_max = data[value_col].max()
        y_annotation = y_max * 1.05

        for idx in permutation_results["significant_timepoints"]:
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
                ax.annotate(
                    significance,
                    xy=(x_pos, y_annotation),
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    color="red",
                    fontweight="bold",
                    zorder=15,
                )

    # Set x-axis limits based on earliest interaction onset
    if earliest_onset_time is not None:
        # Start from earliest onset, end at data maximum
        x_max = data[time_col].max()
        ax.set_xlim(earliest_onset_time, x_max)
        print(f"  X-axis range: {earliest_onset_time:.1f}s to {x_max:.1f}s")

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Ball distance from start (px)", fontsize=14)
    ax.set_title(f"Ball Trajectory: {test_group} vs {control_group}\n(Mean ± 95% CI, Permutation test)", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # Add sample sizes to legend
    n_control = data[data[group_col] == control_group][subject_col].nunique()
    n_test = data[data[group_col] == test_group][subject_col].nunique()
    ax.text(
        0.02,
        0.98,
        f"n({control_group})={n_control}, n({test_group})={n_test}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # Also save as PNG
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"  Saved PDF: {output_path}")
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
        output_dir = Path("/mnt/upramdya_data/MD/MagnetBlock/Plots/trajectories")
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
    if "Magnet" in groups and "non-Magnet" in groups:
        control_group = "non-Magnet"
        test_group = "Magnet"
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

    # Save statistical results
    stats_file = output_dir / "trajectory_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("MagnetBlock Trajectory Permutation Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control group: {permutation_results['control_group']}\n")
        f.write(f"Test group: {permutation_results['test_group']}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Number of time bins: {n_bins}\n")
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
        default="/mnt/upramdya_data/MD/MagnetBlock/Plots/trajectories",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/MagnetBlock/Plots/trajectories)",
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
