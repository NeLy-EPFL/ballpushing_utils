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

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def load_coordinates_dataset():
    """Load the MagnetBlock experiments coordinates dataset"""
    dataset_path = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/251126_10_coordinates_magnet_block_folders_Data/coordinates/pooled_coordinates.feather"

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

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette - blue for Magnet block, orange for Control (matching notebook)
    # Note: label_colors keys must match the actual labels with sample sizes
    label_colors = {f"Magnet block (n={n_test})": "blue", f"Control (n={n_control})": "orange"}

    # Plot using seaborn lineplot (gives mean ± CI automatically)
    sns.lineplot(data=data_ds, x=time_col, y=value_col, hue=group_col_plot, palette=label_colors, ax=ax)

    # Set x-axis limits first (3600 to 7200 seconds = 60 to 120 minutes)
    ax.set_xlim(3600, 7200)

    # Get current y-axis limits and extend them to make room for significance annotations
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Extend y-axis by 20% to make room for significance stars and p-values
    ax.set_ylim(y_min, y_max + 0.20 * y_range)

    # Update y_max after extension
    y_max_extended = y_max + 0.20 * y_range

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

    # Set x-axis ticks: every 10 min from 60 min to 120 min
    xticks = list(range(3600, 7201, 600))
    xticklabels = [f"{int(x//60)} min" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Ball distance from start (mm)", fontsize=12)

    # Set legend position to lower right
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()

    # Save plot
    if output_path:
        # Save as PDF and PNG
        pdf_path = output_path.with_suffix(".pdf")
        png_path = output_path.with_suffix(".png")

        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

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
