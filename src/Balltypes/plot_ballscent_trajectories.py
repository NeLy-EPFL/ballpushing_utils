#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for ball scent experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- FDR-corrected permutation tests for each time bin
- Significance annotations (*, **, ***)
- Comparison between Scented control and other ball types (Washed, New)
- Individual fly trajectories with mean overlay

Usage:
    python plot_ballscent_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/Ball_scents/Plots/trajectories)
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
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def load_coordinates_dataset():
    """Load the ball scent coordinates dataset and additional control datasets"""
    # Main ball scents dataset
    dataset_path = "/mnt/upramdya_data/MD/Ball_scents/Datasets/251103_10_summary_ballscents_Data/coordinates/pooled_coordinates.feather"

    print(f"Loading coordinates dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Ball scents coordinates dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Coordinates dataset not found at {dataset_path}")

    # Additional control datasets
    ctrl_paths = [
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/Other/coordinates/230704_FeedingState_1_AM_Videos_Tracked_coordinates.feather",
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/Other/coordinates/230705_FeedingState_2_AM_Videos_Tracked_coordinates.feather",
    ]

    ctrl_datasets = []
    for ctrl_path in ctrl_paths:
        print(f"\nLoading additional control dataset from: {ctrl_path}")
        try:
            ctrl_data = pd.read_feather(ctrl_path)
            print(f"  Loaded shape: {ctrl_data.shape}")

            # Check if FeedingState column exists
            if "FeedingState" not in ctrl_data.columns:
                print(f"  ⚠️  Warning: FeedingState column not found, skipping this dataset")
                print(f"  Available columns: {list(ctrl_data.columns)}")
                continue

            # Filter for starved_noWater flies only
            initial_shape = ctrl_data.shape
            ctrl_data = ctrl_data[ctrl_data["FeedingState"] == "starved_noWater"].copy()
            print(f"  Filtered to starved_noWater: {initial_shape} -> {ctrl_data.shape}")

            if len(ctrl_data) == 0:
                print(f"  ⚠️  Warning: No starved_noWater flies found in this dataset")
                continue

            # Add BallScent column as "Ctrl"
            ctrl_data["BallScent"] = "Ctrl"

            # Check for unique flies
            if "fly" in ctrl_data.columns:
                n_flies = ctrl_data["fly"].nunique()
                print(f"  Number of flies: {n_flies}")

            ctrl_datasets.append(ctrl_data)

        except FileNotFoundError:
            print(f"  ⚠️  Warning: Control dataset not found at {ctrl_path}")
            continue
        except Exception as e:
            print(f"  ⚠️  Error loading control dataset: {e}")
            continue

    # Combine all datasets
    if ctrl_datasets:
        print(f"\n📊 Combining {len(ctrl_datasets)} control datasets with main dataset...")
        all_datasets = [dataset] + ctrl_datasets
        combined_dataset = pd.concat(all_datasets, ignore_index=True)
        print(f"✅ Combined dataset shape: {combined_dataset.shape}")
        print(f"   BallScent conditions: {sorted(combined_dataset['BallScent'].unique())}")

        # Print sample counts
        if "fly" in combined_dataset.columns:
            print(f"\n   Sample counts by BallScent:")
            for scent in sorted(combined_dataset["BallScent"].unique()):
                n_flies = combined_dataset[combined_dataset["BallScent"] == scent]["fly"].nunique()
                n_points = len(combined_dataset[combined_dataset["BallScent"] == scent])
                print(f"     {scent}: {n_flies} flies, {n_points} data points")

        return combined_dataset
    else:
        print(f"\n⚠️  No control datasets could be loaded, using only main dataset")
        return dataset


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
    control_group="Scented",
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

        iterator = tqdm(time_bins, desc=f"  {test_group} vs {control_group}") if progress else time_bins

        for time_bin in iterator:
            bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

            # Get control and test values
            control_vals = bin_data[bin_data[group_col] == control_group][metric].values
            test_vals = bin_data[bin_data[group_col] == test_group][metric].values

            if len(control_vals) == 0 or len(test_vals) == 0:
                observed_diffs.append(np.nan)
                p_values_raw.append(1.0)
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
            p_values_raw.append(p_value)

        # Apply FDR correction across all time bins for this comparison
        p_values_raw = np.array(p_values_raw)
        rejected, p_values_corrected, _, _ = multipletests(p_values_raw, alpha=alpha, method="fdr_bh")

        # Store results
        results[test_group] = {
            "time_bins": time_bins,
            "observed_diffs": observed_diffs,
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
    control_scent="Scented",
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
    output_path : Path or str
        Where to save the plot
    show_individual_flies : bool
        Whether to show individual fly trajectories
    """
    # Filter data for this comparison
    subset_data = data[data[group_col].isin([control_scent, ball_scent])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {ball_scent} vs {control_scent}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    colors = {control_scent: "red", ball_scent: "blue"}

    # Plot individual fly trajectories if requested
    if show_individual_flies:
        for scent in [control_scent, ball_scent]:
            scent_data = subset_data[subset_data[group_col] == scent]
            for fly in scent_data[subject_col].unique():
                fly_data = scent_data[scent_data[subject_col] == fly]
                ax.plot(fly_data[time_col], fly_data[value_col], color=colors[scent], alpha=0.2, linewidth=0.8)

    # Plot mean trajectories with error bands
    for scent in [control_scent, ball_scent]:
        scent_data = subset_data[subset_data[group_col] == scent]

        # Group by time to compute mean and SEM
        time_grouped = scent_data.groupby(time_col)[value_col].agg(["mean", "sem"]).reset_index()

        # Plot mean line
        ax.plot(time_grouped[time_col], time_grouped["mean"], color=colors[scent], linewidth=3, label=scent, zorder=10)

        # Plot error band (SEM)
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["mean"] - time_grouped["sem"],
            time_grouped["mean"] + time_grouped["sem"],
            color=colors[scent],
            alpha=0.3,
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = subset_data[time_col].min()
    time_max = subset_data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5, zorder=1)

    # Annotate significance levels
    if permutation_results is not None and ball_scent in permutation_results:
        perm_result = permutation_results[ball_scent]

        y_max = subset_data[value_col].max()
        y_annotation = y_max * 1.05

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

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Ball distance from start (px)", fontsize=14)
    ax.set_title(f"Ball Trajectory: {ball_scent} vs {control_scent}\n(FDR-corrected permutation test)", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # Add sample sizes to legend
    n_control = subset_data[subset_data[group_col] == control_scent][subject_col].nunique()
    n_test = subset_data[subset_data[group_col] == ball_scent][subject_col].nunique()
    ax.text(
        0.02,
        0.98,
        f"n({control_scent})={n_control}, n({ball_scent})={n_test}",
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
        print(f"  Saved: {output_path}")

    plt.close()


def create_combined_trajectory_plot(
    data,
    control_scent="Scented",
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

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette - distinct colors for each condition
    color_map = {
        "Scented": "#e74c3c",  # Red
        "CtrlScent": "#c0392b",  # Dark Red
        "Washed": "#3498db",  # Blue
        "New": "#2ecc71",  # Green
        "NewScent": "#27ae60",  # Dark Green
        "Ctrl": "#9b59b6",  # Purple
    }

    # Fallback colors if conditions don't match expected names
    default_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]
    for i, scent in enumerate(ball_scents):
        if scent not in color_map:
            color_map[scent] = default_colors[i % len(default_colors)]

    # Plot individual fly trajectories if requested
    if show_individual_flies:
        for scent in ball_scents:
            scent_data = data[data[group_col] == scent]
            for fly in scent_data[subject_col].unique():
                fly_data = scent_data[scent_data[subject_col] == fly]
                ax.plot(
                    fly_data[time_col],
                    fly_data[value_col],
                    color=color_map[scent],
                    alpha=0.15,
                    linewidth=0.8,
                )

    # Plot mean trajectories with error bands
    for scent in ball_scents:
        scent_data = data[data[group_col] == scent]

        # Group by time to compute mean and SEM
        time_grouped = scent_data.groupby(time_col)[value_col].agg(["mean", "sem"]).reset_index()

        # Plot mean line
        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"],
            color=color_map[scent],
            linewidth=3,
            label=scent,
            zorder=10,
        )

        # Plot error band (SEM)
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["mean"] - time_grouped["sem"],
            time_grouped["mean"] + time_grouped["sem"],
            color=color_map[scent],
            alpha=0.25,
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

        y_max = data[value_col].max()

        # Offset annotations vertically for different comparisons
        for test_idx, test_scent in enumerate(test_scents):
            if test_scent not in permutation_results:
                continue

            perm_result = permutation_results[test_scent]

            # Vertical offset for this comparison (to avoid overlap)
            y_offset = 0.05 + (test_idx * 0.06)
            y_annotation = y_max * (1.0 + y_offset)

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
                        color=color_map[test_scent],
                        fontweight="bold",
                        zorder=15,
                    )

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Ball distance from start (px)", fontsize=14)
    ax.set_title(
        f"Ball Trajectory: All Conditions Combined\n(FDR-corrected permutation test vs {control_scent})", fontsize=16
    )
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add sample sizes
    sample_sizes = []
    for scent in ball_scents:
        n = data[data[group_col] == scent][subject_col].nunique()
        sample_sizes.append(f"n({scent})={n}")

    sample_text = ", ".join(sample_sizes)
    ax.text(
        0.02,
        0.98,
        sample_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    # Add legend for significance annotations if there are any
    if permutation_results:
        test_scents = [s for s in ball_scents if s != control_scent and s in permutation_results]
        if test_scents:
            sig_text = "Significance markers:\n"
            for test_scent in test_scents:
                sig_text += f"  {test_scent} vs {control_scent} (color coded)\n"

            ax.text(
                0.98,
                0.98,
                sig_text.strip(),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
            )

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
    data, control_scent="Scented", n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate trajectory plots for all ball scents vs control.

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
    ball_scents = data["BallScent"].unique()
    test_scents = [s for s in ball_scents if s != control_scent]

    print(f"\nBall scents in dataset: {sorted(ball_scents)}")
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

    # Generate plots for each test scent
    print(f"\nGenerating trajectory plots...")
    for test_scent in test_scents:
        print(f"\n  Creating plot for {test_scent} vs {control_scent}...")

        output_path = output_dir / f"trajectory_{test_scent}_vs_{control_scent}.pdf"

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
            output_path=output_path,
            show_individual_flies=True,
        )

    # Generate combined plot with all conditions
    print(f"\n  Creating combined plot with all conditions...")
    output_path_combined = output_dir / f"trajectory_all_conditions_combined.pdf"
    create_combined_trajectory_plot(
        data,
        control_scent=control_scent,
        time_col="time",
        value_col="distance_ball_0",
        group_col="BallScent",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,
        output_path=output_path_combined,
        show_individual_flies=True,
    )

    # Save statistical results
    stats_file = output_dir / "trajectory_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("Ball Scent Trajectory Permutation Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control group: {control_scent}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Number of time bins: {n_bins}\n")
        f.write(f"FDR correction method: Benjamini-Hochberg\n")
        f.write(f"Significance level: α = 0.05\n\n")

        for test_scent in test_scents:
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

    parser.add_argument("--control", type=str, default="Scented", help="Control ball scent name (default: Scented)")

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    args = parser.parse_args()

    # Load data
    print("Loading ball scent coordinates data...")
    data = load_coordinates_dataset()

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
