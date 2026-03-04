#!/usr/bin/env python3
"""
Compute distance moved at specific time bins for F1 TNT genotypes.

For each time bin (10, 20, 30, 40, 50, 60 minutes), compute the cumulative
distance moved up to that time point and create boxplots grouped by pretraining.

Usage:
    python compute_distance_at_timepoints.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

COORDINATES_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
)
SUMMARY_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather"
)

OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_At_Timepoints")

SELECTED_GENOTYPES = [
    "TNTxEmptySplit",
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

# Time bins in seconds (10, 20, 30, 40, 50, 60 minutes)
TIME_BINS = [600, 1200, 1800, 2400, 3000, 3600]
TIME_BIN_LABELS = ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"]

# Color scheme matching f1_tnt_genotype_comparison.py
GENOTYPE_COLORS = {
    "TNTxEmptySplit": "#7f7f7f",
    "TNTxDDC": "#8B4513",
    "TNTxTH": "#8B4513",
    "TNTxTRH": "#8B4513",
    "TNTxMB247": "#1f77b4",
    "TNTxLC10-2": "#ff7f0e",
    "TNTxLC16-1": "#ff7f0e",
}

# Pretraining styles matching f1_tnt_genotype_comparison.py
PRETRAINING_STYLES = {
    "Naive": {"alpha": 0.6, "edgecolor": "black", "linewidth": 1.5},
    "Pretrained": {"alpha": 1.0, "edgecolor": "black", "linewidth": 2.5},
}

# Pixel to mm conversion
PIXELS_PER_MM = 500 / 30

# Plot styling
JITTER_AMOUNT = 0.15
SCATTER_SIZE = 40
SCATTER_ALPHA = 0.5

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================


def load_datasets():
    """Load datasets."""
    print("Loading datasets...")
    df_coords = pd.read_feather(COORDINATES_PATH)
    df_summary = pd.read_feather(SUMMARY_PATH)
    print(f"  F1 coordinates: {df_coords.shape}")
    print(f"  Summary: {df_summary.shape}")
    return df_coords, df_summary


def compute_distance_at_timepoint(df_fly, time_cutoff, ball_col="test_ball_euclidean_distance"):
    """
    Compute NET distance moved (using euclidean distance displacement) up to a specific time point.

    For each interaction event that occurs before time_cutoff, compute the net displacement
    as: euclidean_distance[end_of_event] - euclidean_distance[start_of_event]
    Sum these displacements to get total NET distance moved.

    Parameters
    ----------
    df_fly : DataFrame
        Data for a single fly
    time_cutoff : float
        Time cutoff in seconds from adjusted_time
    ball_col : str
        Which ball column to use ('test_ball_euclidean_distance' or 'training_ball_euclidean_distance')

    Returns
    -------
    float
        Total NET distance moved up to time_cutoff
    """
    # Filter to frames up to time cutoff
    df_time = df_fly[df_fly["adjusted_time"] <= time_cutoff].copy()

    if df_time.empty:
        return 0.0

    # Check if ball column exists
    if ball_col not in df_time.columns:
        return 0.0

    # Get interaction events and euclidean distances
    euclidean_dist = df_time[ball_col].values
    events = df_time["interaction_event"].values
    frames = df_time["frame"].values

    # Find unique events and compute NET displacement for each
    total_distance = 0.0
    unique_events = np.unique(events[~np.isnan(events)])

    for event_num in unique_events:
        event_num = int(event_num)
        # Find indices where this event occurs
        event_mask = events == event_num
        event_idx = np.where(event_mask)[0]

        if len(event_idx) > 0:
            start_idx = event_idx[0]
            end_idx = event_idx[-1]

            # Get euclidean distances at start and end
            start_dist = euclidean_dist[start_idx]
            end_dist = euclidean_dist[end_idx]

            # NET displacement for this event
            if not (np.isnan(start_dist) or np.isnan(end_dist)):
                distance = abs(end_dist - start_dist)
                total_distance += distance

    return float(total_distance)


def process_all_flies_at_timepoints(df_coords, genotypes):
    """
    Process all flies and compute distance at each time point.

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns: fly, genotype, pretraining, timepoint, distance
    """
    results = []

    for genotype in genotypes:
        print(f"\n  Processing {genotype}...")
        df_gen = df_coords[df_coords["Genotype"] == genotype].copy()

        if df_gen.empty:
            continue

        # Check for Pretraining column
        if "Pretraining" not in df_gen.columns:
            print(f"    Warning: No Pretraining column for {genotype}")
            continue

        # Process each fly
        for fly_name in df_gen["fly"].unique():
            df_fly = df_gen[df_gen["fly"] == fly_name].copy()

            # Get pretraining status
            pretraining_val = df_fly["Pretraining"].iloc[0]
            value_str = str(pretraining_val).lower()
            pretraining_label = "Pretrained" if value_str in ["y", "yes", "true", "1"] else "Naive"

            # Compute distance at each time point
            for time_bin, time_label in zip(TIME_BINS, TIME_BIN_LABELS):
                distance = compute_distance_at_timepoint(df_fly, time_bin)

                results.append(
                    {
                        "fly": fly_name,
                        "genotype": genotype,
                        "pretraining": pretraining_label,
                        "timepoint": time_label,
                        "timepoint_sec": time_bin,
                        "distance_pixels": distance,
                        "distance_mm": distance / PIXELS_PER_MM,
                    }
                )

        # Count flies processed
        n_flies = len(df_gen["fly"].unique())
        print(f"    Processed {n_flies} flies")

    return pd.DataFrame(results)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_distance_by_timepoint(df_results, output_dir):
    """
    Create boxplots with jittered scatter for each timepoint, all genotypes combined.
    Returns paths to saved plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}
    n_genotypes = df_results["genotype"].nunique()
    jitter = JITTER_AMOUNT * (3 / max(n_genotypes, 3))  # Adjust for number of genotypes

    for timepoint, time_label in zip(TIME_BINS, TIME_BIN_LABELS):
        df_time = df_results[df_results["timepoint"] == time_label].copy()

        if df_time.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        genotypes = sorted(df_time["genotype"].unique())
        pretraining_vals = sorted(df_time["pretraining"].unique())

        # Create positions for grouped boxplots
        n_groups = len(genotypes)
        n_conditions = len(pretraining_vals)
        group_width = 0.8
        box_width = group_width / n_conditions

        x_positions = np.arange(n_groups)

        # Plot for each pretraining condition
        for i, pretrain in enumerate(pretraining_vals):
            df_pretrain = df_time[df_time["pretraining"] == pretrain]

            box_positions = x_positions + (i - n_conditions / 2 + 0.5) * box_width

            # Prepare data for boxplot
            data_by_genotype = [df_pretrain[df_pretrain["genotype"] == g]["distance_mm"].values for g in genotypes]

            # Get color (use first genotype's color for now, will customize per box)
            pretrain_style = PRETRAINING_STYLES.get(pretrain, {})

            # Create boxplots
            bp = ax.boxplot(
                data_by_genotype,
                positions=box_positions,
                widths=box_width * 0.6,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(alpha=pretrain_style.get("alpha", 0.8)),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Color boxes by genotype
            for patch, genotype in zip(bp["boxes"], genotypes):
                color = GENOTYPE_COLORS.get(genotype, "#808080")
                patch.set_facecolor(color)
                patch.set_edgecolor(pretrain_style.get("edgecolor", "black"))
                patch.set_linewidth(pretrain_style.get("linewidth", 1.5))

            # Add jittered scatter points
            for j, genotype in enumerate(genotypes):
                data = df_pretrain[df_pretrain["genotype"] == genotype]["distance_mm"].values

                if len(data) == 0:
                    continue

                # Add jitter
                x_jitter = np.random.normal(box_positions[j], jitter, len(data))

                color = GENOTYPE_COLORS.get(genotype, "#808080")
                ax.scatter(
                    x_jitter,
                    data,
                    s=SCATTER_SIZE,
                    alpha=SCATTER_ALPHA,
                    color=color,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )

        # Customize plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(genotypes, rotation=45, ha="right")
        ax.set_ylabel("Distance Moved (mm)", fontsize=13, fontweight="bold")
        ax.set_title(f"Distance Moved at {time_label} ({timepoint} seconds)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Create legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gray",
                edgecolor=PRETRAINING_STYLES[p].get("edgecolor", "black"),
                linewidth=PRETRAINING_STYLES[p].get("linewidth", 1.5),
                alpha=PRETRAINING_STYLES[p].get("alpha", 0.8),
                label=p,
            )
            for p in pretraining_vals
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

        plt.tight_layout()

        # Save
        output_path = output_dir / f"distance_at_{timepoint}sec.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plot_paths[time_label] = output_path
        print(f"  Saved: {output_path.name}")
        plt.close()

    return plot_paths


def plot_distance_by_genotype(df_results, output_dir):
    """
    Create individual plots for each genotype showing all timepoints.
    Returns paths to saved plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    for genotype in sorted(df_results["genotype"].unique()):
        df_gen = df_results[df_results["genotype"] == genotype].copy()

        if df_gen.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        timepoints = sorted(df_gen["timepoint_sec"].unique())
        pretraining_vals = sorted(df_gen["pretraining"].unique())

        # Create positions
        n_timepoints = len(timepoints)
        n_conditions = len(pretraining_vals)
        group_width = 0.8
        box_width = group_width / n_conditions

        x_positions = np.arange(n_timepoints)

        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")

        # Plot for each pretraining condition
        for i, pretrain in enumerate(pretraining_vals):
            df_pretrain = df_gen[df_gen["pretraining"] == pretrain]

            box_positions = x_positions + (i - n_conditions / 2 + 0.5) * box_width

            # Prepare data
            data_by_time = [df_pretrain[df_pretrain["timepoint_sec"] == t]["distance_mm"].values for t in timepoints]

            pretrain_style = PRETRAINING_STYLES.get(pretrain, {})

            # Create boxplots
            bp = ax.boxplot(
                data_by_time,
                positions=box_positions,
                widths=box_width * 0.6,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=genotype_color,
                    alpha=pretrain_style.get("alpha", 0.8),
                    edgecolor=pretrain_style.get("edgecolor", "black"),
                    linewidth=pretrain_style.get("linewidth", 1.5),
                ),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Add jittered scatter
            for j, timepoint_sec in enumerate(timepoints):
                data = df_pretrain[df_pretrain["timepoint_sec"] == timepoint_sec]["distance_mm"].values

                if len(data) == 0:
                    continue

                x_jitter = np.random.normal(box_positions[j], JITTER_AMOUNT * 0.5, len(data))

                ax.scatter(
                    x_jitter,
                    data,
                    s=SCATTER_SIZE,
                    alpha=SCATTER_ALPHA,
                    color=genotype_color,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )

        # Customize
        ax.set_xticks(x_positions)
        ax.set_xticklabels(TIME_BIN_LABELS, rotation=0)
        ax.set_xlabel("Time Point", fontsize=13, fontweight="bold")
        ax.set_ylabel("Distance Moved (mm)", fontsize=13, fontweight="bold")
        ax.set_title(f"{genotype} - Distance Moved Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=genotype_color,
                edgecolor=PRETRAINING_STYLES[p].get("edgecolor", "black"),
                linewidth=PRETRAINING_STYLES[p].get("linewidth", 1.5),
                alpha=PRETRAINING_STYLES[p].get("alpha", 0.8),
                label=p,
            )
            for p in pretraining_vals
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

        plt.tight_layout()

        # Save
        output_path = output_dir / f"distance_timepoints_{genotype}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plot_paths[genotype] = output_path
        print(f"  Saved: {output_path.name}")
        plt.close()

    return plot_paths


def create_distance_timepoint_grid(timepoint_paths, output_dir, time_bins_labels):
    """
    Create a grid layout showing all timepoint plots.
    """
    if len(timepoint_paths) == 0:
        return

    n_cols = 3
    n_rows = (len(timepoint_paths) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 5 * n_rows))

    for idx, (time_label, path) in enumerate(timepoint_paths.items(), 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(time_label, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = output_dir / "distance_timepoint_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid layout: {grid_path.name}")


def create_distance_genotype_grid(genotype_paths, output_dir, genotypes):
    """
    Create a grid layout showing all genotype plots.
    """
    genotypes_with_plots = [g for g in genotypes if g in genotype_paths]

    if len(genotypes_with_plots) == 0:
        return

    n_cols = 3
    n_rows = (len(genotypes_with_plots) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 5 * n_rows))

    for idx, genotype in enumerate(genotypes_with_plots, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        img = plt.imread(genotype_paths[genotype])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(genotype, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = output_dir / "distance_genotype_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid layout: {grid_path.name}")


def validate_60min_values(df_results, df_summary):
    """
    Compare 60-minute values with summary dataset to validate computation.
    """
    print("\n" + "=" * 70)
    print("VALIDATION: Comparing 60-min values with summary dataset")
    print("=" * 70)

    df_60min = df_results[df_results["timepoint_sec"] == 3600].copy()

    if df_60min.empty:
        print("  Warning: No 60-minute data found")
        return

    # Check if summary has distance_moved column
    if "distance_moved" not in df_summary.columns:
        print("  Warning: No distance_moved column in summary dataset")
        return

    # Match flies
    comparison = []
    for _, row in df_60min.iterrows():
        fly_name = row["fly"]
        computed_dist = row["distance_pixels"]

        # Find in summary
        summary_row = df_summary[df_summary["fly"] == fly_name]

        if not summary_row.empty:
            summary_dist = summary_row["distance_moved"].iloc[0]

            if pd.notna(summary_dist) and summary_dist > 0:
                pct_diff = 100 * (computed_dist - summary_dist) / summary_dist

                comparison.append(
                    {
                        "fly": fly_name,
                        "computed": computed_dist,
                        "summary": summary_dist,
                        "pct_diff": pct_diff,
                    }
                )

    if comparison:
        df_comp = pd.DataFrame(comparison)
        print(f"\n  Compared {len(df_comp)} flies")
        print(f"  Mean % difference: {df_comp['pct_diff'].mean():.2f}%")
        print(f"  Median % difference: {df_comp['pct_diff'].median():.2f}%")
        print(f"  Std % difference: {df_comp['pct_diff'].std():.2f}%")

        # Show some examples
        print("\n  Sample comparisons:")
        print(df_comp.head(10).to_string(index=False))
    else:
        print("  No matching flies found for comparison")


def generate_summary_statistics(df_results, output_dir):
    """Generate summary CSV with statistics per genotype, pretraining, and timepoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = (
        df_results.groupby(["genotype", "pretraining", "timepoint"])
        .agg({"distance_mm": ["count", "mean", "std", "median", "min", "max"]})
        .reset_index()
    )

    summary_stats.columns = ["genotype", "pretraining", "timepoint", "n_flies", "mean", "std", "median", "min", "max"]

    output_path = output_dir / "distance_timepoints_summary.csv"
    summary_stats.to_csv(output_path, index=False)
    print(f"\n  Saved summary statistics: {output_path.name}")

    # Also save full results
    full_output = output_dir / "distance_timepoints_full.csv"
    df_results.to_csv(full_output, index=False)
    print(f"  Saved full results: {full_output.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("DISTANCE MOVED AT TIME POINTS ANALYSIS")
    print("=" * 70 + "\n")

    # Load datasets
    df_coords, df_summary = load_datasets()

    # Check required columns
    required_cols = ["adjusted_time", "test_ball_euclidean_distance", "Genotype", "Pretraining"]
    missing = [col for col in required_cols if col not in df_coords.columns]
    if missing:
        print(f"\nError: Missing columns: {missing}")
        return False

    # Process all flies at all timepoints
    print("\nProcessing flies at all timepoints...")
    df_results = process_all_flies_at_timepoints(df_coords, SELECTED_GENOTYPES)

    if df_results.empty:
        print("\nError: No results generated!")
        return False

    print(f"\nProcessed {df_results['fly'].nunique()} flies total")

    # Validate 60-minute values
    validate_60min_values(df_results, df_summary)

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    timepoint_paths = plot_distance_by_timepoint(df_results, OUTPUT_DIR)
    genotype_paths = plot_distance_by_genotype(df_results, OUTPUT_DIR)

    # Create grid layouts
    create_distance_timepoint_grid(timepoint_paths, OUTPUT_DIR, TIME_BIN_LABELS)
    create_distance_genotype_grid(genotype_paths, OUTPUT_DIR, SELECTED_GENOTYPES)

    # Generate summary statistics
    print("\n" + "=" * 70)
    print("Generating summary statistics...")
    print("=" * 70)

    generate_summary_statistics(df_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
