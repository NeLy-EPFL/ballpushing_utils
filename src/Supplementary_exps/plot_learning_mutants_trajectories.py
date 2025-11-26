#!/usr/bin/env python3
"""
Script to generate ball trajectory plots over time for learning mutants experiments.

This script generates trajectory visualizations with:
- Time-binned analysis of ball positions
- FDR-corrected permutation tests for each time bin (multiple genotype comparisons)
- Significance annotations (*, **, ***)
- Comparison between control genotype and test genotypes
- Individual fly trajectories with mean overlay

Usage:
    python plot_learning_mutants_trajectories.py [--n-bins N] [--n-permutations N] [--output-dir PATH]

Arguments:
    --n-bins: Number of time bins for analysis (default: 12)
    --n-permutations: Number of permutations for statistical testing (default: 10000)
    --output-dir: Directory to save plots (default: /mnt/upramdya_data/MD/Learning_mutants/Plots/trajectories)
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
    """Load the learning mutants coordinates dataset"""
    dataset_path = "/mnt/upramdya_data/MD/Learning_mutants/Datasets/251124_13_summary_learning_mutants_Data/coordinates/pooled_coordinates.feather"

    print(f"Loading coordinates dataset from: {dataset_path}")
    try:
        dataset = pd.read_feather(dataset_path)
        print(f"✅ Learning mutants coordinates dataset loaded successfully! Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Coordinates dataset not found at {dataset_path}")

    # Check for Genotype column
    if "Genotype" not in dataset.columns:
        raise ValueError(f"'Genotype' column not found in dataset. Available columns: {list(dataset.columns)}")

    print(f"Genotypes: {sorted(dataset['Genotype'].unique())}")

    # Print sample counts
    if "fly" in dataset.columns:
        print(f"\nSample counts by Genotype:")
        for genotype in sorted(dataset["Genotype"].unique()):
            n_flies = dataset[dataset["Genotype"] == genotype]["fly"].nunique()
            n_points = len(dataset[dataset["Genotype"] == genotype])
            print(f"  {genotype}: {n_flies} flies, {n_points} data points")

    return dataset


def preprocess_data(
    data, time_col="time", value_col="distance_ball_0", group_col="Genotype", subject_col="fly", n_bins=12
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
        Column name for grouping (Genotype)
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
    group_col="Genotype",
    control_group=None,
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
        Name of the control group. If None, uses alphabetically first group.
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
    groups = sorted(processed_data[group_col].unique())

    # Determine control group
    if control_group is None:
        control_group = groups[0]
        print(f"  Using {control_group} as control genotype")

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
    genotype,
    control_genotype=None,
    time_col="time",
    value_col="distance_ball_0",
    group_col="Genotype",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
    show_individual_flies=True,
):
    """
    Create a trajectory plot comparing a genotype to control.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    genotype : str
        Name of the genotype to compare
    control_genotype : str
        Name of the control genotype
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
    # Determine control genotype
    if control_genotype is None:
        genotypes = sorted(data[group_col].unique())
        control_genotype = genotypes[0]

    # Filter data for this comparison
    subset_data = data[data[group_col].isin([control_genotype, genotype])].copy()

    if len(subset_data) == 0:
        print(f"Warning: No data for {genotype} vs {control_genotype}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    colors = {control_genotype: "red", genotype: "blue"}

    # Plot individual fly trajectories if requested
    if show_individual_flies:
        for gen in [control_genotype, genotype]:
            gen_data = subset_data[subset_data[group_col] == gen]
            for fly in gen_data[subject_col].unique():
                fly_data = gen_data[gen_data[subject_col] == fly]
                ax.plot(fly_data[time_col], fly_data[value_col], color=colors[gen], alpha=0.2, linewidth=0.8)

    # Plot mean trajectories with error bands
    for gen in [control_genotype, genotype]:
        gen_data = subset_data[subset_data[group_col] == gen]

        # Group by time to compute mean and SEM
        time_grouped = gen_data.groupby(time_col)[value_col].agg(["mean", "sem"]).reset_index()

        # Plot mean line
        ax.plot(time_grouped[time_col], time_grouped["mean"], color=colors[gen], linewidth=3, label=gen, zorder=10)

        # Plot error band (SEM)
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["mean"] - time_grouped["sem"],
            time_grouped["mean"] + time_grouped["sem"],
            color=colors[gen],
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
    if permutation_results is not None and genotype in permutation_results:
        perm_result = permutation_results[genotype]

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
    ax.set_title(f"Ball Trajectory: {genotype} vs {control_genotype}\n(FDR-corrected permutation test)", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # Add sample sizes to legend
    n_control = subset_data[subset_data[group_col] == control_genotype][subject_col].nunique()
    n_test = subset_data[subset_data[group_col] == genotype][subject_col].nunique()
    ax.text(
        0.02,
        0.98,
        f"n({control_genotype})={n_control}, n({genotype})={n_test}",
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


def create_combined_trajectory_plot(
    data,
    control_genotype=None,
    time_col="time",
    value_col="distance_ball_0",
    group_col="Genotype",
    subject_col="fly",
    n_bins=12,
    permutation_results=None,
    output_path=None,
    show_individual_flies=True,
):
    """
    Create a combined trajectory plot showing all genotypes together.

    Parameters:
    -----------
    data : pd.DataFrame
        Full trajectory data
    control_genotype : str
        Name of the control genotype
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
    # Get all genotypes
    genotypes = sorted(data[group_col].unique())

    # Determine control genotype
    if control_genotype is None:
        control_genotype = genotypes[0]

    if len(data) == 0:
        print(f"Warning: No data for combined plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette - distinct colors for each genotype
    n_genotypes = len(genotypes)
    colors_palette = sns.color_palette("husl", n_genotypes)
    color_map = {}

    # Assign colors - control gets red, others get palette colors
    for i, genotype in enumerate(genotypes):
        if genotype == control_genotype:
            color_map[genotype] = "#e74c3c"  # Red for control
        else:
            color_map[genotype] = colors_palette[i]

    # Plot individual fly trajectories if requested
    if show_individual_flies:
        for genotype in genotypes:
            genotype_data = data[data[group_col] == genotype]
            for fly in genotype_data[subject_col].unique():
                fly_data = genotype_data[genotype_data[subject_col] == fly]
                ax.plot(
                    fly_data[time_col],
                    fly_data[value_col],
                    color=color_map[genotype],
                    alpha=0.15,
                    linewidth=0.8,
                )

    # Plot mean trajectories with error bands
    for genotype in genotypes:
        genotype_data = data[data[group_col] == genotype]

        # Group by time to compute mean and SEM
        time_grouped = genotype_data.groupby(time_col)[value_col].agg(["mean", "sem"]).reset_index()

        # Plot mean line
        ax.plot(
            time_grouped[time_col],
            time_grouped["mean"],
            color=color_map[genotype],
            linewidth=3,
            label=genotype,
            zorder=10,
        )

        # Plot error band (SEM)
        ax.fill_between(
            time_grouped[time_col],
            time_grouped["mean"] - time_grouped["sem"],
            time_grouped["mean"] + time_grouped["sem"],
            color=color_map[genotype],
            alpha=0.25,
            zorder=5,
        )

    # Draw vertical dotted lines for time bins
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)

    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.4, zorder=1)

    # Annotate significance levels for each test genotype vs control
    if permutation_results is not None:
        # Get test genotypes (non-control)
        test_genotypes = [g for g in genotypes if g != control_genotype]

        y_max = data[value_col].max()

        # Offset annotations vertically for different comparisons
        for test_idx, test_genotype in enumerate(test_genotypes):
            if test_genotype not in permutation_results:
                continue

            perm_result = permutation_results[test_genotype]

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
                    # Use the test genotype's color for the annotation
                    ax.annotate(
                        significance,
                        xy=(x_pos, y_annotation),
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        color=color_map[test_genotype],
                        fontweight="bold",
                        zorder=15,
                    )

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Ball distance from start (px)", fontsize=14)
    ax.set_title(
        f"Ball Trajectory: All Genotypes Combined\n(FDR-corrected permutation test vs {control_genotype})", fontsize=16
    )
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add sample sizes
    sample_sizes = []
    for genotype in genotypes:
        n = data[data[group_col] == genotype][subject_col].nunique()
        sample_sizes.append(f"n({genotype})={n}")

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
        test_genotypes = [g for g in genotypes if g != control_genotype and g in permutation_results]
        if test_genotypes:
            sig_text = "Significance markers:\n"
            for test_genotype in test_genotypes:
                sig_text += f"  {test_genotype} vs {control_genotype} (color coded)\n"

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
        print(f"  Saved PDF: {output_path}")
        print(f"  Saved PNG: {png_path}")

    plt.close()


def generate_all_trajectory_plots(
    data, control_genotype=None, n_bins=12, n_permutations=10000, output_dir=None, show_progress=True
):
    """
    Generate trajectory plots for all genotypes vs control.

    Parameters:
    -----------
    data : pd.DataFrame
        Coordinates data
    control_genotype : str
        Name of control genotype. If None, uses alphabetically first genotype.
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
        output_dir = Path("/mnt/upramdya_data/MD/Learning_mutants/Plots/trajectories")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING LEARNING MUTANTS TRAJECTORY PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Time bins: {n_bins}")
    print(f"Permutations: {n_permutations}")

    # Check required columns
    required_cols = ["time", "distance_ball_0", "Genotype", "fly"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get genotypes
    genotypes = sorted(data["Genotype"].unique())

    # Determine control genotype
    if control_genotype is None:
        control_genotype = genotypes[0]

    test_genotypes = [g for g in genotypes if g != control_genotype]

    print(f"\nGenotypes in dataset: {genotypes}")
    print(f"Test genotypes: {test_genotypes}")
    print(f"Control genotype: {control_genotype}")

    if control_genotype not in genotypes:
        raise ValueError(f"Control genotype '{control_genotype}' not found in data")

    # Preprocess data
    print(f"\nPreprocessing data into {n_bins} time bins...")
    processed = preprocess_data(
        data, time_col="time", value_col="distance_ball_0", group_col="Genotype", subject_col="fly", n_bins=n_bins
    )
    print(f"Processed data shape: {processed.shape}")

    # Compute permutation tests with FDR correction
    print(f"\nComputing permutation tests ({n_permutations} permutations)...")
    print(f"Using FDR correction across {n_bins} time bins for each comparison...")

    permutation_results = compute_permutation_test_with_fdr(
        processed,
        metric="avg_distance_ball_0",
        group_col="Genotype",
        control_group=control_genotype,
        n_permutations=n_permutations,
        alpha=0.05,
        progress=show_progress,
    )

    # Generate plots for each test genotype
    print(f"\nGenerating trajectory plots...")
    for test_genotype in test_genotypes:
        print(f"\n  Creating plot for {test_genotype} vs {control_genotype}...")

        output_path = output_dir / f"trajectory_{test_genotype}_vs_{control_genotype}.pdf"

        create_trajectory_plot(
            data,
            genotype=test_genotype,
            control_genotype=control_genotype,
            time_col="time",
            value_col="distance_ball_0",
            group_col="Genotype",
            subject_col="fly",
            n_bins=n_bins,
            permutation_results=permutation_results,
            output_path=output_path,
            show_individual_flies=True,
        )

    # Generate combined plot with all genotypes
    print(f"\n  Creating combined plot with all genotypes...")
    output_path_combined = output_dir / f"trajectory_all_genotypes_combined.pdf"
    create_combined_trajectory_plot(
        data,
        control_genotype=control_genotype,
        time_col="time",
        value_col="distance_ball_0",
        group_col="Genotype",
        subject_col="fly",
        n_bins=n_bins,
        permutation_results=permutation_results,
        output_path=output_path_combined,
        show_individual_flies=True,
    )

    # Save statistical results
    stats_file = output_dir / "trajectory_permutation_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("Learning Mutants Trajectory Permutation Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control genotype: {control_genotype}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Number of time bins: {n_bins}\n")
        f.write(f"FDR correction method: Benjamini-Hochberg\n")
        f.write(f"Significance level: α = 0.05\n\n")

        for test_genotype in test_genotypes:
            if test_genotype not in permutation_results:
                continue

            result = permutation_results[test_genotype]
            f.write(f"\n{test_genotype} vs {control_genotype}\n")
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
        description="Generate ball trajectory plots for learning mutants experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_learning_mutants_trajectories.py
  python plot_learning_mutants_trajectories.py --n-bins 15 --n-permutations 5000
  python plot_learning_mutants_trajectories.py --output-dir /path/to/output
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
        default="/mnt/upramdya_data/MD/Learning_mutants/Plots/trajectories",
        help="Directory to save plots (default: /mnt/upramdya_data/MD/Learning_mutants/Plots/trajectories)",
    )

    parser.add_argument(
        "--control", type=str, default=None, help="Control genotype name (default: alphabetically first genotype)"
    )

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    args = parser.parse_args()

    # Load data
    print("Loading learning mutants coordinates data...")
    data = load_coordinates_dataset()

    # Generate plots
    generate_all_trajectory_plots(
        data,
        control_genotype=args.control,
        n_bins=args.n_bins,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
