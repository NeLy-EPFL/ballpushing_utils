#!/usr/bin/env python3
"""
Compute distance moved over time for F1 TNT genotypes.

This script processes the F1_coordinates dataset to compute cumulative distance moved
by the ball during interaction events over time, grouped by TNT genotype.

The script:
1. Loads the F1_coordinates dataset with event annotations
2. For each TNT genotype and fly, computes cumulative distance moved over events
3. Validates the final computed values against the summary dataset to ensure coherence

Usage:
    python compute_distance_moved_over_time.py
    python compute_distance_moved_over_time.py --genotypes TNTxMB247,TNTxLC10-2
    python compute_distance_moved_over_time.py --output /custom/output/path
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ballpushing_utils import read_feather

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_PATHS = {
    "coordinates": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather",
    "summary": "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather",
}

OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_Over_Time")

# TNT Groups from f1_tnt_genotype_comparison.py
SELECTED_GENOTYPES = [
    # === CONTROLS ===
    "TNTxEmptySplit",
    # === EXPERIMENTAL GENOTYPES ===
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm

# Ball euclidean distance columns (for different ball identities)
BALL_DISTANCE_COLUMNS = {
    "training": "training_ball_euclidean_distance",
    "test": "test_ball_euclidean_distance",
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_datasets():
    """Load F1 coordinates and summary datasets."""
    print("Loading datasets...")
    df_coords = read_feather(DATASET_PATHS["coordinates"])
    df_summary = read_feather(DATASET_PATHS["summary"])

    print(f"  F1 coordinates: {df_coords.shape}")
    print(f"  Summary: {df_summary.shape}")

    return df_coords, df_summary


def detect_genotype_column(df):
    """Detect the genotype column name."""
    for col in df.columns:
        if "genotype" in col.lower():
            return col
    raise ValueError("Could not find genotype column in dataset")


def detect_ball_identity_column(df):
    """Detect the ball identity column name."""
    for col in df.columns:
        if "ball" in col.lower() and "identity" in col.lower():
            return col
    # Fallback: check for "ball_condition"
    if "ball_condition" in df.columns:
        return "ball_condition"
    return None


def get_available_ball_names(df, ball_col):
    """Get available ball identity values in dataset."""
    if ball_col is None:
        return []
    return df[ball_col].dropna().unique().tolist()


def compute_event_distance(euclidean_dist, events, frames):
    """
    Compute distance moved per event using NET displacement.

    For each interaction event, computes the difference between euclidean distance
    at the end and start of the event. This gives the net displacement during that event.

    Parameters
    ----------
    euclidean_dist : array-like
        Euclidean distances from start position at each frame
    events : array-like
        Interaction event numbers (NaN when not interacting)
    frames : array-like
        Frame numbers

    Returns
    -------
    event_distances : dict
        Dictionary mapping event number to distance moved
    event_indices : dict
        Dictionary mapping event number to (start_idx, end_idx)
    """
    event_distances = {}
    event_indices = {}

    # Find unique events (excluding NaN)
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

            # Net displacement for this event
            if not (np.isnan(start_dist) or np.isnan(end_dist)):
                distance = abs(end_dist - start_dist)
                event_distances[event_num] = distance
                event_indices[event_num] = (start_idx, end_idx)

    return event_distances, event_indices


def process_fly_ball_combination(df_subset, fly_name, ball_distance_col):
    """
    Process a single fly's data to extract distance moved over time using event annotations.

    Parameters
    ----------
    df_subset : DataFrame
        Data for a single fly
    fly_name : str
        Fly identifier
    ball_distance_col : str
        Column name for ball euclidean distance

    Returns
    -------
    dict
        Distance information for the fly, organized by events
    """
    if ball_distance_col not in df_subset.columns:
        return None

    # Get interaction event column
    if "interaction_event" not in df_subset.columns:
        return None

    # Get euclidean distance column
    euclidean_dist = df_subset[ball_distance_col].values
    events = df_subset["interaction_event"].values
    frames = df_subset["frame"].values
    times = df_subset["time"].values

    # Compute distance per event using NET displacement
    event_distances_dict, event_indices_dict = compute_event_distance(euclidean_dist, events, frames)

    if not event_distances_dict:
        return None

    # Build ordered lists
    event_numbers = sorted(event_distances_dict.keys())
    event_distances = [event_distances_dict[e] for e in event_numbers]
    event_times = [times[event_indices_dict[e][1]] for e in event_numbers]  # End time of each event
    event_frames = [frames[event_indices_dict[e][1]] for e in event_numbers]  # End frame of each event

    # Compute cumulative distance over events
    cumulative_distances = np.cumsum(event_distances)

    # Total distance is sum of all event distances
    total_distance = sum(event_distances)

    return {
        "fly_name": fly_name,
        "total_distance": total_distance,
        "event_numbers": event_numbers,
        "event_distances": event_distances,
        "event_times": event_times,
        "event_frames": event_frames,
        "cumulative_distances": cumulative_distances,
    }


def detect_pretraining_column(df):
    """Detect the pretraining column name."""
    for col in df.columns:
        if "pretraining" in col.lower():
            return col
    return None


def process_genotype_group(df_coords, genotype_col, genotype, selected_genotypes=None):
    """
    Process all flies in a genotype group, organized by Pretraining status.

    Parameters
    ----------
    df_coords : DataFrame
        F1 coordinates dataset
    genotype_col : str
        Name of genotype column
    genotype : str
        Genotype to process
    selected_genotypes : list, optional
        List of genotypes to process

    Returns
    -------
    dict
        Distance metrics for all flies in the genotype, organized by pretraining
    """
    # Filter for genotype
    df_genotype = df_coords[df_coords[genotype_col] == genotype].copy()

    if df_genotype.empty:
        print(f"  ⚠️  No data for {genotype}")
        return {}

    # Detect pretraining column
    pretraining_col = detect_pretraining_column(df_coords)

    # Get unique flies and pretraining combinations
    if pretraining_col and pretraining_col in df_genotype.columns:
        unique_combos = df_genotype.groupby("fly")[pretraining_col].unique()
        total_flies = len(df_genotype["fly"].unique())
    else:
        unique_combos = {fly: [None] for fly in df_genotype["fly"].unique()}
        total_flies = len(df_genotype["fly"].unique())

    print(f"  Processing {genotype}: {total_flies} unique flies")

    results = {
        "genotype": genotype,
        "pretraining_data": {},
    }

    # Detect ball identity column
    ball_col = detect_ball_identity_column(df_coords)
    ball_names = get_available_ball_names(df_genotype, ball_col) if ball_col else ["unknown"]

    # Process each pretraining condition
    pretraining_values = df_genotype[pretraining_col].unique() if pretraining_col else [None]

    for pretraining in pretraining_values:
        # Filter by pretraining if available
        if pretraining_col and pretraining is not None:
            df_pretrain = df_genotype[df_genotype[pretraining_col] == pretraining].copy()
            pretrain_label = "Pretrained" if str(pretraining).lower() in ["y", "yes", "true", "1"] else "Naive"
        else:
            df_pretrain = df_genotype.copy()
            pretrain_label = "All"

        if df_pretrain.empty:
            continue

        pretrain_results = {
            "pretraining": pretrain_label,
            "flies": {},
        }

        # Process each fly in this pretraining condition
        for fly_name in df_pretrain["fly"].unique():
            df_fly = df_pretrain[df_pretrain["fly"] == fly_name]
            fly_results = {}

            # Process training and test balls
            for ball_identity, ball_distance_col in BALL_DISTANCE_COLUMNS.items():
                # Check if this ball column exists in the data
                if ball_distance_col not in df_fly.columns:
                    continue

                # Check if there's any non-NaN data for this ball
                if df_fly[ball_distance_col].isna().all():
                    continue

                result = process_fly_ball_combination(df_fly, fly_name, ball_distance_col)
                if result:
                    result["ball_identity"] = ball_identity
                    fly_results[ball_identity] = result

            if fly_results:
                pretrain_results["flies"][fly_name] = fly_results

        if pretrain_results["flies"]:
            results["pretraining_data"][pretrain_label] = pretrain_results

    return results


def compare_with_summary(results, df_summary, genotype_col):
    """
    Compare computed distance_moved values with summary dataset.

    Parameters
    ----------
    results : dict
        Computed distance results
    df_summary : DataFrame
        Summary dataset with reference distance_moved values
    genotype_col : str
        Name of genotype column

    Returns
    -------
    DataFrame
        Comparison results
    """
    comparison_data = []

    print(f"  Summary dataset columns: {df_summary.columns.tolist()}")
    print(f"  Distance_moved in summary: {'distance_moved' in df_summary.columns}")
    print(f"  Unique flies in summary: {df_summary['fly'].nunique()}")
    print(f"  Sample fly names from summary: {df_summary['fly'].unique()[:5].tolist()}")

    pretraining_col = detect_pretraining_column(df_summary)

    for genotype, genotype_results in results.items():
        if not isinstance(genotype_results, dict) or "pretraining_data" not in genotype_results:
            continue

        for pretrain_label, pretrain_results in genotype_results["pretraining_data"].items():
            for fly_name, fly_results in pretrain_results["flies"].items():
                for ball_id, ball_results in fly_results.items():
                    # Try multiple match strategies
                    summary_match = None

                    # Strategy 1: Exact fly name match
                    summary_match = df_summary[df_summary["fly"] == fly_name]

                    # Strategy 2: Partial match if exact didn't work
                    if summary_match.empty and "_" in fly_name:
                        fly_parts = fly_name.split("_")
                        # Try matching by last part (usually the fly number)
                        for part in reversed(fly_parts):
                            if not part.isdigit():
                                continue
                            partial_matches = df_summary[
                                df_summary["fly"].astype(str).str.contains(part, na=False, case=False, regex=False)
                            ]
                            if not partial_matches.empty:
                                summary_match = partial_matches.iloc[[0]]
                                break

                    computed_dist = ball_results["total_distance"]
                    summary_dist = np.nan
                    pretraining_match = None

                    if not summary_match.empty and "distance_moved" in df_summary.columns:
                        summary_dist = summary_match["distance_moved"].values[0]
                        if pretraining_col and pretraining_col in summary_match.columns:
                            pretraining_match = summary_match[pretraining_col].values[0]

                    difference = computed_dist - summary_dist if not np.isnan(summary_dist) else np.nan
                    pct_diff = (
                        (difference / summary_dist * 100)
                        if (not np.isnan(summary_dist) and summary_dist != 0)
                        else np.nan
                    )

                    comparison_data.append(
                        {
                            "fly_name": fly_name,
                            "genotype": genotype,
                            "pretraining": pretrain_label,
                            "ball_identity": ball_id,
                            "computed_distance": computed_dist,
                            "summary_distance": summary_dist,
                            "difference": difference,
                            "pct_difference": pct_diff,
                        }
                    )

    if comparison_data:
        return pd.DataFrame(comparison_data)
    else:
        # Return empty dataframe with correct columns if no data
        return pd.DataFrame(
            {
                "fly_name": [],
                "genotype": [],
                "pretraining": [],
                "ball_identity": [],
                "computed_distance": [],
                "summary_distance": [],
                "difference": [],
                "pct_difference": [],
            }
        )


def plot_distance_over_time(df_coords, selected_genotypes, output_dir):
    """
    Create plots of cumulative distance over TIME for each genotype, grouped by pretraining.
    Averages across all flies in each genotype-pretraining combination.
    Uses shared y-axis range across all genotypes for easier comparison.

    Parameters
    ----------
    df_coords : DataFrame
        Full F1 coordinates dataset
    selected_genotypes : list
        List of genotypes to plot
    output_dir : Path
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define colors for pretraining conditions
    pretrain_colors = {"Naive": "#1f77b4", "Pretrained": "#ff7f0e"}
    pretrain_linestyles = {"Naive": "-", "Pretrained": "--"}

    plot_data = {}  # Store plot info for grid layout

    # ===== FIRST PASS: Collect all data and compute global y-range =====
    all_genotype_data = {}
    global_min = float("inf")
    global_max = float("-inf")

    for genotype in selected_genotypes:
        # Filter data for this genotype
        df_geno = df_coords[df_coords["Genotype"] == genotype].copy()
        if df_geno.empty:
            continue

        # Get pretraining column
        pretraining_col = None
        for col in df_geno.columns:
            if "pretraining" in col.lower():
                pretraining_col = col
                break

        # Process each pretraining condition
        if pretraining_col:
            pretraining_values = df_geno[pretraining_col].unique()
        else:
            pretraining_values = [None]

        genotype_data = {}

        for pretraining_val in pretraining_values:
            # Filter by pretraining
            if pretraining_col and pretraining_val is not None:
                df_pretrain = df_geno[df_geno[pretraining_col] == pretraining_val].copy()
                pretrain_label = "Pretrained" if str(pretraining_val).lower() in ["y", "yes", "true", "1"] else "Naive"
            else:
                df_pretrain = df_geno.copy()
                pretrain_label = "All"

            if df_pretrain.empty:
                continue

            # Get list of flies
            flies = df_pretrain["fly"].unique()

            # For each fly, compute cumulative distance over time
            time_distance_data = []

            for fly in flies:
                df_fly = df_pretrain[df_pretrain["fly"] == fly].copy()

                # Try both ball types
                for ball_col in ["training_ball_euclidean_distance", "test_ball_euclidean_distance"]:
                    if ball_col not in df_fly.columns:
                        continue

                    # Get euclidean distances and times
                    euclidean_dist = df_fly[ball_col].values
                    events = df_fly["interaction_event"].values
                    times = df_fly["adjusted_time"].values

                    # Compute distance per event
                    event_distances_dict, event_indices_dict = compute_event_distance(
                        euclidean_dist, events, df_fly["frame"].values
                    )

                    if not event_distances_dict:
                        continue

                    # Build cumulative distance over time
                    for event_num in sorted(event_distances_dict.keys()):
                        event_dist = event_distances_dict[event_num]
                        event_time = times[event_indices_dict[event_num][1]]  # End time of event

                        time_distance_data.append(
                            {
                                "fly": fly,
                                "time": event_time,
                                "distance": event_dist,
                            }
                        )

            if not time_distance_data:
                continue

            # Convert to dataframe
            time_df = pd.DataFrame(time_distance_data)

            if time_df.empty:
                continue

            # Get max time for this pretraining group
            max_time_group = time_df["time"].max()
            time_bins = np.arange(0, max_time_group + 60, 60)

            # For each fly, compute cumulative distance at each time bin
            fly_cumulative_distances = {}

            for fly in time_df["fly"].unique():
                df_fly_time = time_df[time_df["fly"] == fly].copy()
                fly_events = list(zip(df_fly_time["time"], df_fly_time["distance"]))
                fly_events.sort(key=lambda x: x[0])

                cumulative_dist = 0
                event_idx = 0
                fly_distances_at_bins = []

                for time_bin in time_bins:
                    # Add all events up to this time bin
                    while event_idx < len(fly_events) and fly_events[event_idx][0] <= time_bin:
                        cumulative_dist += fly_events[event_idx][1]
                        event_idx += 1
                    fly_distances_at_bins.append(cumulative_dist)

                fly_cumulative_distances[fly] = fly_distances_at_bins

            if not fly_cumulative_distances:
                continue

            # Average cumulative distances across flies at each time bin
            distances_array = np.array(list(fly_cumulative_distances.values()))
            mean_distances = np.nanmean(distances_array, axis=0)
            std_distances = np.nanstd(distances_array, axis=0)

            # Track min/max for global range
            min_val = np.nanmin(mean_distances - std_distances)
            max_val = np.nanmax(mean_distances + std_distances)
            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)

            # Store data for second pass
            time_minutes = time_bins / 60
            genotype_data[pretrain_label] = {
                "time_minutes": time_minutes,
                "mean_distances": mean_distances,
                "std_distances": std_distances,
            }

        if genotype_data:
            all_genotype_data[genotype] = genotype_data

    # Set reasonable global limits
    if global_min == float("inf") or global_max == float("-inf"):
        y_min, y_max = 0, 100
    else:
        y_min = max(0, global_min * 0.95)
        y_max = global_max * 1.05

    # ===== SECOND PASS: Create plots with shared y-limits =====
    for genotype in selected_genotypes:
        if genotype not in all_genotype_data:
            continue

        genotype_data = all_genotype_data[genotype]
        fig, ax = plt.subplots(figsize=(12, 7))

        for pretrain_label in ["Naive", "Pretrained", "All"]:
            if pretrain_label not in genotype_data:
                continue

            data = genotype_data[pretrain_label]
            time_minutes = data["time_minutes"]
            mean_distances = data["mean_distances"]
            std_distances = data["std_distances"]

            # Plot
            color = pretrain_colors.get(pretrain_label, "#808080")
            linestyle = pretrain_linestyles.get(pretrain_label, "-")

            ax.plot(
                time_minutes,
                mean_distances,
                marker="o",
                label=pretrain_label,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                markersize=5,
            )

            # Add confidence interval
            ax.fill_between(
                time_minutes,
                mean_distances - std_distances,
                mean_distances + std_distances,
                alpha=0.15,
                color=color,
            )

        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel("Cumulative Distance Moved (pixels)", fontsize=12)
        ax.set_title(f"Distance Moved Over Time - {genotype}", fontsize=13, fontweight="bold")
        ax.set_ylim(y_min, y_max)  # Apply shared y-limits
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"distance_over_time_{genotype}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plot_data[genotype] = output_path
        plt.close()

        print(f"  Saved plot: {output_path.name}")

    # Create grid layout with all genotypes
    if plot_data:
        create_distance_over_time_grid(plot_data, output_dir, selected_genotypes)


def create_distance_over_time_grid(plot_data, output_dir, selected_genotypes):
    """
    Create a grid layout showing all distance over time plots.
    """
    # Filter to only genotypes that have plots
    genotypes_with_plots = [g for g in selected_genotypes if g in plot_data]

    if len(genotypes_with_plots) == 0:
        return

    # Create grid layout
    n_cols = 3
    n_rows = (len(genotypes_with_plots) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 5 * n_rows))

    for idx, genotype in enumerate(genotypes_with_plots, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)

        # Load and display the image
        img = plt.imread(plot_data[genotype])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(genotype, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = output_dir / "distance_over_time_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid layout: {grid_path.name}")


def plot_validation_comparison(comparison_df, output_dir):
    """
    Create validation plots comparing computed vs summary distances, with pretraining splits.

    Parameters
    ----------
    comparison_df : DataFrame
        Comparison results
    output_dir : Path
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if comparison_df.empty:
        print("  ⚠️  No data to plot - skipping validation comparison plot")
        return

    # Scatter plot: Computed vs Summary with pretraining coloring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Remove NaN values for plotting
    valid_data = comparison_df.dropna(subset=["computed_distance", "summary_distance"])

    if not valid_data.empty:
        # Scatter plot colored by pretraining
        if "pretraining" in valid_data.columns:
            pretrainings = valid_data["pretraining"].unique()
            colors = {"Naive": "#FF6B6B", "Pretrained": "#4ECDC4", "All": "#95A5A6"}

            for pretrain in sorted(pretrainings):
                data_subset = valid_data[valid_data["pretraining"] == pretrain]
                color = colors.get(str(pretrain), "#808080")
                ax1.scatter(
                    data_subset["summary_distance"],
                    data_subset["computed_distance"],
                    alpha=0.6,
                    label=pretrain,
                    color=color,
                    s=50,
                )
        else:
            ax1.scatter(valid_data["summary_distance"], valid_data["computed_distance"], alpha=0.6, s=50)

        # Add diagonal line (perfect agreement)
        lim = [
            min(valid_data["summary_distance"].min(), valid_data["computed_distance"].min()),
            max(valid_data["summary_distance"].max(), valid_data["computed_distance"].max()),
        ]
        ax1.plot(lim, lim, "r--", linewidth=2, label="Perfect agreement")

        ax1.set_xlabel("Summary Dataset Distance (pixels)", fontsize=12)
        ax1.set_ylabel("Computed Distance (pixels)", fontsize=12)
        ax1.set_title("Distance Moved: Summary vs Computed", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax1.transAxes)

    # Distribution of differences by pretraining
    if not comparison_df["pct_difference"].isna().all():
        if "pretraining" in comparison_df.columns:
            pretrainings = sorted(comparison_df["pretraining"].unique())
            for pretrain in pretrainings:
                data_subset = comparison_df[comparison_df["pretraining"] == pretrain]
                ax2.hist(data_subset["pct_difference"].dropna(), bins=15, alpha=0.6, label=pretrain, edgecolor="black")
        else:
            comparison_df["pct_difference"].hist(bins=20, ax=ax2, edgecolor="black", color="steelblue")

        ax2.set_xlabel("Percent Difference (%)", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Distribution of Percent Differences", fontsize=13, fontweight="bold")
        ax2.axvline(0, color="r", linestyle="--", linewidth=2, label="No difference")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No valid percent differences", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    output_path = output_dir / "validation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved validation plot: {output_path.name}")

    # Create summary statistics plot by genotype and pretraining
    if not valid_data.empty and "pretraining" in valid_data.columns and "genotype" in valid_data.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Prepare data for grouped bar plot
        pivot_data = (
            valid_data.groupby(["genotype", "pretraining"])["pct_difference"]
            .apply(lambda x: [x.mean(), x.std(), len(x)])
            .reset_index()
        )
        pivot_data[["mean", "std", "count"]] = pd.DataFrame(
            pivot_data["pct_difference"].tolist(), index=pivot_data.index
        )

        genotypes = sorted(valid_data["genotype"].unique())
        pretrainings = sorted(valid_data["pretraining"].unique())
        x = np.arange(len(genotypes))
        width = 0.35

        for i, pretrain in enumerate(pretrainings):
            subset = pivot_data[pivot_data["pretraining"] == pretrain]
            means = []
            stds = []
            for genotype in genotypes:
                gen_data = subset[subset["genotype"] == genotype]
                if not gen_data.empty:
                    means.append(gen_data["mean"].values[0])
                    stds.append(gen_data["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, label=pretrain, yerr=stds, capsize=5, alpha=0.8)

        ax.set_xlabel("Genotype", fontsize=12)
        ax.set_ylabel("Mean Percent Difference (%)", fontsize=12)
        ax.set_title("Distance Computation Accuracy by Genotype and Pretraining", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(genotypes, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="r", linestyle="--", linewidth=1)

        plt.tight_layout()
        output_path = output_dir / "accuracy_by_genotype_pretraining.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved summary plot: {output_path.name}")


def print_validation_summary(comparison_df):
    """Print summary statistics of the validation comparison."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if comparison_df.empty:
        print("\n⚠️  No valid comparison data found.")
        print("This may indicate:")
        print("  - Fly names don't match between datasets")
        print("  - Summary dataset doesn't have distance_moved column")
        print("  - No successful fly-ball combinations processed")
        return

    # Check for required columns
    if "computed_distance" not in comparison_df.columns or "summary_distance" not in comparison_df.columns:
        print("\n⚠️  Missing expected columns in comparison dataframe")
        print(f"Available columns: {comparison_df.columns.tolist()}")
        return

    valid_data = comparison_df.dropna(subset=["computed_distance", "summary_distance"])

    print(f"\nTotal comparisons: {len(comparison_df)}")
    print(f"Valid comparisons (non-NaN): {len(valid_data)}")

    if valid_data.empty:
        print("⚠️  No valid data with non-NaN computed and summary distances")
        return

    print(f"\nOverall Percent Difference Statistics:")
    print(f"  Mean: {valid_data['pct_difference'].mean():.2f}%")
    print(f"  Median: {valid_data['pct_difference'].median():.2f}%")
    print(f"  Std Dev: {valid_data['pct_difference'].std():.2f}%")
    print(f"  Min: {valid_data['pct_difference'].min():.2f}%")
    print(f"  Max: {valid_data['pct_difference'].max():.2f}%")

    abs_diff = np.abs(valid_data["pct_difference"])
    threshold = 5  # 5% difference threshold
    matches = (abs_diff < threshold).sum()
    print(f"\n  Flies with < {threshold}% difference: {matches} ({100*matches/len(valid_data):.1f}%)")

    # Genotype-specific statistics
    print(f"\nPer-Genotype Statistics:")
    for genotype in sorted(valid_data["genotype"].unique()):
        gen_data = valid_data[valid_data["genotype"] == genotype]
        print(f"  {genotype}:")
        print(f"    Count: {len(gen_data)}")
        print(f"    Mean % diff: {gen_data['pct_difference'].mean():.2f}%")

    # Pretraining-specific statistics
    if "pretraining" in valid_data.columns:
        print(f"\nPer-Pretraining Statistics:")
        for pretrain in sorted(valid_data["pretraining"].unique()):
            pretrain_data = valid_data[valid_data["pretraining"] == pretrain]
            print(f"  {pretrain}:")
            print(f"    Count: {len(pretrain_data)}")
            print(f"    Mean % diff: {pretrain_data['pct_difference'].mean():.2f}%")

            # Per-genotype within pretraining
            for genotype in sorted(pretrain_data["genotype"].unique()):
                gen_pretrain = pretrain_data[pretrain_data["genotype"] == genotype]
                print(f"      {genotype}: {len(gen_pretrain)} flies, {gen_pretrain['pct_difference'].mean():.2f}% diff")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Compute distance moved over time for F1 TNT genotypes")
    parser.add_argument(
        "--genotypes",
        type=str,
        help="Comma-separated list of genotypes to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots instead of just saving",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine genotypes to process
    genotypes_to_process = SELECTED_GENOTYPES
    if args.genotypes:
        genotypes_to_process = [g.strip() for g in args.genotypes.split(",")]

    print("=" * 80)
    print("F1 TNT DISTANCE MOVED OVER TIME ANALYSIS")
    print("=" * 80)

    # Load datasets
    df_coords, df_summary = load_datasets()

    # Detect genotype column
    genotype_col = detect_genotype_column(df_coords)
    print(f"\nGenotype column: {genotype_col}\n")

    # Process each genotype
    print("Processing genotypes...")
    all_results = {}

    for genotype in genotypes_to_process:
        results = process_genotype_group(df_coords, genotype_col, genotype, genotypes_to_process)
        if results:
            all_results[genotype] = results

    # Compare with summary dataset
    print("\n" + "=" * 80)
    print("Comparing with summary dataset...")
    print("=" * 80)

    comparison_df = compare_with_summary(all_results, df_summary, genotype_col)

    # Save comparison results
    comparison_output = output_dir / "distance_validation_comparison.csv"
    comparison_df.to_csv(comparison_output, index=False)
    print(f"Saved comparison results: {comparison_output}")

    # Print validation summary
    print_validation_summary(comparison_df)

    # Create plots
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)

    plot_distance_over_time(df_coords, genotypes_to_process, output_dir)
    plot_validation_comparison(comparison_df, output_dir)

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
