#!/usr/bin/env python3
"""
Compute distance moved over time for F1 TNT genotypes (v2 - optimized for euclidean distances).

This script processes the F1_coordinates dataset with event annotations to compute
cumulative distance moved by the ball during interaction events over time.

The dataset now has:
- interaction_event: event ID for each frame
- training_ball_euclidean_distance: distance from reference point
- test_ball_euclidean_distance: distance from reference point

Usage:
    python compute_distance_moved_over_time_v2.py
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths - UPDATED to new dataset
COORDINATES_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
)
SUMMARY_PATH = Path(
    "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather"
)

OUTPUT_DIR = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_TNT/Distance_Over_Time")

# TNT Groups from f1_tnt_genotype_comparison.py
SELECTED_GENOTYPES = [
    "TNTxEmptySplit",  # Control
    "TNTxDDC",
    "TNTxTH",
    "TNTxTRH",
    "TNTxMB247",
    "TNTxLC10-2",
    "TNTxLC16-1",
]

# Color scheme matching f1_tnt_genotype_comparison.py
GENOTYPE_COLORS = {
    # Controls
    "TNTxEmptyGal4": "#7f7f7f",  # Gray
    "TNTxEmptySplit": "#7f7f7f",  # Gray
    "TNTxPR": "#7f7f7f",  # Gray
    # Experimental genotypes
    "TNTxDDC": "#8B4513",  # Brown (MB extrinsic)
    "TNTxTH": "#8B4513",  # Brown (MB extrinsic)
    "PRxTH": "#8B4513",  # Brown (MB extrinsic)
    "TNTxTRH": "#8B4513",  # Brown (MB extrinsic)
    "TNTxMB247": "#1f77b4",  # Blue (MB)
    "PRxMB247": "#1f77b4",  # Blue (MB)
    "TNTxLC10-2": "#ff7f0e",  # Orange (Vision)
    "TNTxLC16-1": "#ff7f0e",  # Orange (Vision)
    "PRxLC16-1": "#ff7f0e",  # Orange (Vision)
}

# Pretraining styles
PRETRAINING_STYLES = {
    "Naive": {"alpha": 0.6, "linewidth": 1.5},
    "Pretrained": {"alpha": 1.0, "linewidth": 2.5},
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def load_datasets():
    """Load F1 coordinates and summary datasets."""
    print("Loading datasets...")
    df_coords = pd.read_feather(COORDINATES_PATH)
    df_summary = pd.read_feather(SUMMARY_PATH)

    print(f"  F1 coordinates: {df_coords.shape}")
    print(f"  Summary: {df_summary.shape}")

    # Note: Fly objects will be loaded on-demand with caching

    return df_coords, df_summary


def get_ball_tracking_data(fly_name):
    """
    Load ball tracking data directly from the HDF5 file without loading full Fly object.
    Much faster than loading entire Fly object.

    Fly names look like: 260127_TNT_F1_Videos_Checked_arena7_Right
    """
    import h5py
    from pathlib import Path

    # Extract video session directory from fly name
    parts = fly_name.split("_arena")
    if len(parts) < 2:
        return None

    video_session_prefix = parts[0]  # e.g., "260127_TNT_F1_Videos_Checked"
    arena_parts = fly_name.split("_arena")[1]  # e.g., "7_Right"
    arena_num = arena_parts.split("_")[0]  # e.g., "7"
    side = "Right" if "Right" in arena_parts else "Left"  # e.g., "Right"

    # Find matching directory in YAML
    yaml_path = Path("/home/matthias/ballpushing_utils/experiments_yaml/F1_TNT_Full.yaml")
    if not yaml_path.exists():
        return None

    import yaml

    try:
        with open(yaml_path, "r") as f:
            exp_data = yaml.safe_load(f)
    except:
        return None

    # Find matching directory
    fly_folder = None
    if "directories" in exp_data:
        for directory in exp_data["directories"]:
            if video_session_prefix in str(directory):
                fly_folder = Path(directory) / f"arena{arena_num}" / side
                break

    if not fly_folder or not fly_folder.exists():
        return None

    # Find and load HDF5 tracking file for ball
    h5_files = list(fly_folder.glob("*_tracked_ball_processed*.h5"))
    if not h5_files:
        return None

    try:
        h5_file = h5_files[0]  # Use first matching file
        with h5py.File(h5_file, "r") as f:
            # Try to get ball positions from the tracked file
            if "track_occupancy" in f:
                x_data = f["track_occupancy"]["x"][:]
                y_data = f["track_occupancy"]["y"][:]
                return pd.DataFrame({"x_centre": x_data, "y_centre": y_data})
            elif "data" in f:
                if "x_centres" in f["data"]:
                    x_data = f["data"]["x_centres"][:]
                    y_data = f["data"]["y_centres"][:]
                    return pd.DataFrame({"x_centre": x_data, "y_centre": y_data})
    except Exception as e:
        pass

    return None


def get_fly_object(fly_name, cache={}):
    """
    Load a Fly object by name from the experimental YAML files.
    Results are cached to avoid reloading.

    Fly names look like: 260127_TNT_F1_Videos_Checked_arena7_Right
    YAML directories look like: /mnt/upramdya_data/MD/F1_Tracks/Videos/260127_TNT_F1_Videos_Checked
    """
    if fly_name in cache:
        return cache[fly_name]

    sys.path.insert(0, "/home/matthias/ballpushing_utils/src")
    from Ballpushing_utils import Config, Fly
    import yaml
    from pathlib import Path

    config = Config()

    # Extract video session directory from fly name
    # Pattern: [prefix]_arena[N]_[Side]
    # We need to find the prefix
    parts = fly_name.split("_arena")
    if len(parts) < 2:
        return None

    video_session_prefix = parts[0]  # e.g., "260127_TNT_F1_Videos_Checked"

    # Search for this prefix in the YAML
    yaml_path = Path("/home/matthias/ballpushing_utils/experiments_yaml/F1_TNT_Full.yaml")

    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path, "r") as f:
            exp_data = yaml.safe_load(f)

        # Check if 'directories' key exists (older format)
        if "directories" in exp_data:
            for fly_folder in exp_data["directories"]:
                # Check if the folder path ends with our video session prefix
                if video_session_prefix in str(fly_folder):
                    try:
                        # Build arena and side from fly name
                        arena_parts = fly_name.split("_arena")
                        if len(arena_parts) >= 2:
                            arena_side = arena_parts[1]  # e.g., "7_Right"
                            arena_num = arena_side.split("_")[0]  # e.g., "7"
                            side = "Right" if "Right" in arena_side else "Left"  # e.g., "Right"

                            # Build the full fly folder path
                            fly_obj_path = Path(fly_folder) / f"arena{arena_num}" / side

                            if fly_obj_path.exists():
                                fly_obj = Fly(str(fly_obj_path), config=config)
                                cache[fly_name] = fly_obj
                                return fly_obj
                    except Exception:
                        continue

        # Try newer format with experiments
        for exp_name, exp_info in exp_data.items():
            if isinstance(exp_info, dict) and "fly_folders" in exp_info:
                for fly_folder in exp_info["fly_folders"]:
                    if video_session_prefix in str(fly_folder):
                        try:
                            fly_obj = Fly(fly_folder, config=config)
                            cache[fly_name] = fly_obj
                            return fly_obj
                        except Exception:
                            continue

        return None

    except Exception as e:
        return None


def compute_distance_over_time_for_fly(df_fly, fly_name, ball_type):
    """
    Compute distance moved over time for a single fly using actual tracking data.

    Distance moved for each event = sum of frame-to-frame ball movements during the event.

    Parameters
    ----------
    df_fly : DataFrame
        Tracking data for a single fly (from F1_coordinates)
    fly_name : str
        Fly identifier
    ball_type : str
        Ball type label ('training' or 'test')

    Returns
    -------
    dict or None
        Distance metrics organized by event, or None if insufficient data
    """
    if "interaction_event" not in df_fly.columns:
        return None

    # Sort by frame to ensure temporal order
    df_fly = df_fly.sort_values("frame").reset_index(drop=True)

    events = df_fly["interaction_event"].values
    times = df_fly["time"].values
    frames = df_fly["frame"].values

    # Count events
    event_count = (~np.isnan(events)).sum()
    if event_count == 0:
        return None

    # Get ball tracking data directly from HDF5 (fast, no Fly object needed)
    ball_data = get_ball_tracking_data(fly_name)
    if ball_data is None or ball_data.empty:
        return None

    # First pass: compute distance for each unique event using frame-to-frame ball movement
    unique_events = np.unique(events[~np.isnan(events)]).astype(int)
    event_distances = {}

    total_computed = 0.0

    for event_id in unique_events:
        event_frames = df_fly[df_fly["interaction_event"] == event_id]["frame"].values.astype(int)

        if len(event_frames) < 2:
            event_distances[event_id] = 0.0
            continue

        try:
            start_frame = int(event_frames[0])
            end_frame = int(event_frames[-1])

            # Bounds check
            if start_frame >= len(ball_data) or end_frame >= len(ball_data):
                event_distances[event_id] = 0.0
                continue

            # Get ball positions for this event
            event_x = ball_data.iloc[start_frame : end_frame + 1]["x_centre"].values
            event_y = ball_data.iloc[start_frame : end_frame + 1]["y_centre"].values

            if len(event_x) < 2:
                event_distances[event_id] = 0.0
                continue

            # Compute frame-to-frame distances
            dx = np.diff(event_x)
            dy = np.diff(event_y)
            frame_distances = np.sqrt(dx**2 + dy**2)

            event_distance = np.nansum(frame_distances)
            event_distances[event_id] = float(event_distance)
            total_computed += event_distance

        except (IndexError, KeyError):
            event_distances[event_id] = 0.0

    # Only proceed if we have distances
    if total_computed == 0.0:
        return None

    # Second pass: build cumulative distances over time
    cumulative_times = []
    cumulative_distances = []
    cumulative_total = 0.0

    for i, event_id in enumerate(events):
        if pd.notna(event_id):
            event_id = int(event_id)

            # Check if this is the first frame of this event to add distance once
            if i == 0 or (i > 0 and events[i - 1] != event_id):
                cumulative_total += event_distances.get(event_id, 0.0)

            cumulative_times.append(times[i])
            cumulative_distances.append(cumulative_total)

    if not cumulative_times:
        return None

    return {
        "fly_name": fly_name,
        "ball_type": ball_type,
        "total_distance": sum(event_distances.values()),
        "event_data": {eid: {"event_distance": dist} for eid, dist in event_distances.items()},
        "times": np.array(cumulative_times),
        "full_cumulative": np.array(cumulative_distances),
        "events": events,
    }


def process_genotype(df_coords, genotype):
    """
    Process all flies in a genotype, split by pretraining status.

    Parameters
    ----------
    df_coords : DataFrame
        Full coordinates dataset
    genotype : str
        Genotype name

    Returns
    -------
    dict
        Results organized by pretraining and fly
    """
    df_gen = df_coords[df_coords["Genotype"] == genotype].copy()

    if df_gen.empty:
        print(f"  ⚠️  No data for {genotype}")
        return {}

    unique_flies = df_gen["fly"].nunique()
    print(f"  Processing {genotype}: {unique_flies} flies")

    # Debug: show pretraining values for this genotype
    if "Pretraining" in df_gen.columns:
        pretrain_counts = df_gen["Pretraining"].value_counts()
        print(f"    Pretraining values: {pretrain_counts.to_dict()}")
        # Print sample fly name for debugging
        sample_fly = df_gen["fly"].iloc[0]
        print(f"    Sample fly name: {sample_fly}")

    results = {"genotype": genotype, "pretraining_data": {}}

    # Determine pretraining values
    if "Pretraining" in df_gen.columns:
        pretraining_values = sorted(df_gen["Pretraining"].dropna().unique())
    else:
        pretraining_values = [None]

    # Process each pretraining condition
    for pretrain_val in pretraining_values:
        df_pretrain = df_gen[df_gen["Pretraining"] == pretrain_val].copy()

        # Normalize pretraining label
        value_str = str(pretrain_val).lower()
        if value_str in ["y", "yes", "true", "1"]:
            pretrain_label = "Pretrained"
        elif value_str in ["n", "no", "false", "0"]:
            pretrain_label = "Naive"
        else:
            pretrain_label = str(pretrain_val)

        if df_pretrain.empty:
            continue

        pretrain_results = {"pretraining": pretrain_label, "flies": {}}

        # Process each fly
        flies_loaded = 0
        flies_failed = 0

        for fly_name in df_pretrain["fly"].unique():
            df_fly = df_pretrain[df_pretrain["fly"] == fly_name].copy()

            # Check if fly has event data
            if "interaction_event" not in df_fly.columns:
                continue

            if df_fly["interaction_event"].isna().all():
                continue

            # Process test ball - direct call without Fly object
            result = compute_distance_over_time_for_fly(df_fly, fly_name, "test")
            if result:
                flies_loaded += 1
                pretrain_results["flies"][fly_name] = result
            else:
                flies_failed += 1

        if flies_failed > 0:
            print(f"      {pretrain_label}: Loaded {flies_loaded} flies, failed to load {flies_failed} flies")

        if pretrain_results["flies"]:
            results["pretraining_data"][pretrain_label] = pretrain_results

    return results


def diagnose_naive_flies(df_coords):
    """
    Diagnostic function - kept for reference but not used.
    Shows that naive flies have 0 interaction_event annotations because they don't have
    training-ball interactions. Only pretrained flies have event annotations.
    """
    pass


def plot_distance_over_time(all_results, output_dir, global_ylim=None):
    """
    Create plots of averaged distance moved over time for each genotype, split by pretraining.
    Shows mean ± SEM across flies by time bins.

    Parameters
    ----------
    all_results : dict
        Results for all genotypes
    output_dir : Path
        Output directory
    global_ylim : tuple or None
        If provided, use this y-axis range for all plots. Otherwise, compute per-genotype.

    Returns
    -------
    tuple
        (min_y, max_y) of all plotted data for setting global y limits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_y_values = []

    for genotype, gen_results in all_results.items():
        if "pretraining_data" not in gen_results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        pretraining_data = gen_results["pretraining_data"]
        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")

        for pretrain_label in sorted(pretraining_data.keys()):  # Sort for consistent order
            pretrain_results = pretraining_data[pretrain_label]

            if not pretrain_results.get("flies"):
                continue

            # Collect all (time, distance) pairs from all flies
            all_time_distance_pairs = []

            for fly_name, ball_results in pretrain_results["flies"].items():
                times = ball_results["times"]
                cumulative_distances = ball_results["full_cumulative"]

                for i, t in enumerate(times):
                    all_time_distance_pairs.append((t, cumulative_distances[i]))
                    all_y_values.append(cumulative_distances[i])

            if not all_time_distance_pairs:
                continue

            # Sort by time
            all_time_distance_pairs.sort(key=lambda x: x[0])
            times_array = np.array([pair[0] for pair in all_time_distance_pairs])
            distances_array = np.array([pair[1] for pair in all_time_distance_pairs])

            # Create time bins (1-second bins)
            time_bins = np.arange(0, times_array.max() + 1, 1.0)
            bin_indices = np.digitize(times_array, time_bins)

            means = []
            sems = []
            bin_centers = []

            for bin_idx in np.unique(bin_indices):
                if bin_idx == 0 or bin_idx > len(time_bins) - 1:
                    continue
                mask = bin_indices == bin_idx
                bin_distances = distances_array[mask]

                if len(bin_distances) > 0:
                    means.append(np.mean(bin_distances))
                    sems.append(np.std(bin_distances) / np.sqrt(len(bin_distances)))
                    bin_centers.append(time_bins[bin_idx - 1])

            if not means:
                continue

            # Plot with error bars using pretraining-specific styling
            pretrain_style = PRETRAINING_STYLES.get(pretrain_label, {})
            ax.errorbar(
                bin_centers,
                means,
                yerr=sems,
                marker="o",
                label=f"{pretrain_label} (n={len(pretrain_results['flies'])})",
                linewidth=pretrain_style.get("linewidth", 2.5),
                markersize=6,
                capsize=5,
                capthick=2,
                color=genotype_color,
                alpha=pretrain_style.get("alpha", 0.8),
            )

        ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Distance Moved (pixels)", fontsize=13, fontweight="bold")
        ax.set_title(f"{genotype} - Distance Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

        # Apply global y-lim if provided
        if global_ylim:
            ax.set_ylim(global_ylim)

        plt.tight_layout()
        output_path = output_dir / f"distance_over_time_{genotype}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"    ✓ Saved {output_path.name}")
        plt.close()

    # Return the data range for global y-lim
    if all_y_values:
        return (0, np.max(all_y_values) * 1.1)  # Add 10% padding
    return None


def plot_all_genotypes_combined(all_results, output_dir, global_ylim):
    """
    Create a combined plot with all genotypes in a grid layout.

    Parameters
    ----------
    all_results : dict
        Results for all genotypes
    output_dir : Path
        Output directory
    global_ylim : tuple
        (min_y, max_y) for y-axis
    """
    output_dir = Path(output_dir)

    genotypes = sorted([g for g in all_results.keys() if "pretraining_data" in all_results[g]])

    if not genotypes:
        return

    # Create grid layout (2 rows, 4 columns for 7 genotypes, or adjust as needed)
    ncols = 4
    nrows = (len(genotypes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes = axes.flatten()  # Flatten to 1D for easier indexing

    for idx, genotype in enumerate(genotypes):
        ax = axes[idx]
        gen_results = all_results[genotype]
        pretraining_data = gen_results.get("pretraining_data", {})
        genotype_color = GENOTYPE_COLORS.get(genotype, "#808080")

        for pretrain_label in sorted(pretraining_data.keys()):
            pretrain_results = pretraining_data[pretrain_label]

            if not pretrain_results.get("flies"):
                continue

            # Collect all (time, distance) pairs
            all_time_distance_pairs = []
            for fly_name, ball_results in pretrain_results["flies"].items():
                times = ball_results["times"]
                cumulative_distances = ball_results["full_cumulative"]
                for i, t in enumerate(times):
                    all_time_distance_pairs.append((t, cumulative_distances[i]))

            if not all_time_distance_pairs:
                continue

            # Sort by time
            all_time_distance_pairs.sort(key=lambda x: x[0])
            times_array = np.array([pair[0] for pair in all_time_distance_pairs])
            distances_array = np.array([pair[1] for pair in all_time_distance_pairs])

            # Create time bins
            time_bins = np.arange(0, times_array.max() + 1, 1.0)
            bin_indices = np.digitize(times_array, time_bins)

            means = []
            sems = []
            bin_centers = []

            for bin_idx in np.unique(bin_indices):
                if bin_idx == 0 or bin_idx > len(time_bins) - 1:
                    continue
                mask = bin_indices == bin_idx
                bin_distances = distances_array[mask]

                if len(bin_distances) > 0:
                    means.append(np.mean(bin_distances))
                    sems.append(np.std(bin_distances) / np.sqrt(len(bin_distances)))
                    bin_centers.append(time_bins[bin_idx - 1])

            if not means:
                continue

            # Plot
            pretrain_style = PRETRAINING_STYLES.get(pretrain_label, {})
            ax.errorbar(
                bin_centers,
                means,
                yerr=sems,
                marker="o",
                label=pretrain_label,
                linewidth=pretrain_style.get("linewidth", 2.5),
                markersize=5,
                capsize=3,
                capthick=1.5,
                color=genotype_color,
                alpha=pretrain_style.get("alpha", 0.8),
            )

        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Distance (pixels)", fontsize=10, fontweight="bold")
        ax.set_title(genotype, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(global_ylim)

    # Hide unused subplots
    for idx in range(len(genotypes), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / "distance_over_time_all_genotypes.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    ✓ Saved {output_path.name}")
    plt.close()


def generate_summary_statistics(all_results, output_dir):
    """
    Generate summary statistics comparing computed vs summary distances.

    Parameters
    ----------
    all_results : dict
        Results for all genotypes
    output_dir : Path
        Output directory
    """
    summary_data = []

    for genotype, gen_results in all_results.items():
        if "pretraining_data" not in gen_results:
            continue

        for pretrain_label, pretrain_results in gen_results["pretraining_data"].items():
            for fly_name, ball_results in pretrain_results["flies"].items():
                # ball_results is now directly the test ball data
                summary_data.append(
                    {
                        "genotype": genotype,
                        "pretraining": pretrain_label,
                        "fly": fly_name,
                        "total_distance": ball_results["total_distance"],
                        "num_events": len(ball_results["event_data"]),
                    }
                )

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        output_file = Path(output_dir) / "distance_summary.csv"
        df_summary.to_csv(output_file, index=False)
        print(f"\n✓ Saved summary to {output_file.name}")

        # Print statistics by genotype and pretraining
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        for genotype in sorted(df_summary["genotype"].unique()):
            print(f"\n{genotype}:")
            df_gen = df_summary[df_summary["genotype"] == genotype]
            for pretrain in sorted(df_gen["pretraining"].unique()):
                df_pretrain = df_gen[df_gen["pretraining"] == pretrain]
                mean_dist = df_pretrain["total_distance"].mean()
                std_dist = df_pretrain["total_distance"].std()
                num_flies = len(df_pretrain)
                print(f"  {pretrain:12s}: {num_flies} flies | " f"Distance: {mean_dist:8.2f} ± {std_dist:8.2f} pixels")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("COMPUTING DISTANCE MOVED OVER TIME (v2 - Euclidean Distances)")
    print("=" * 70 + "\n")

    # Load datasets
    df_coords, df_summary = load_datasets()

    # Check for required columns
    required_cols = ["interaction_event"]
    missing_cols = [col for col in required_cols if col not in df_coords.columns]
    if missing_cols:
        print(f"\n✗ ERROR: Missing columns in dataset: {missing_cols}")
        print(f"Available columns: {df_coords.columns.tolist()}")
        return False

    # Process each genotype
    print("\nProcessing genotypes...")
    all_results = {}

    for genotype in SELECTED_GENOTYPES:
        results = process_genotype(df_coords, genotype)
        if results:
            all_results[genotype] = results

    if not all_results:
        print("\n✗ ERROR: No results generated!")
        return False

    # Generate visualizations - First pass to get global y-range
    print("\nCalculating global y-axis range...")
    global_ylim = plot_distance_over_time(all_results, OUTPUT_DIR, global_ylim=None)

    if global_ylim is None:
        print("  Warning: No data collected, using default y-axis range")
        global_ylim = (0, 10000000)  # Default fallback range
    else:
        print(f"  Global y-range: 0 to {global_ylim[1]:.0f} pixels")
    plot_all_genotypes_combined(all_results, OUTPUT_DIR, global_ylim)

    # Generate summary statistics
    generate_summary_statistics(all_results, OUTPUT_DIR)

    # Print detailed breakdown of what was found
    print("\n" + "=" * 70)
    print("DATA AVAILABILITY BY GENOTYPE AND PRETRAINING")
    print("=" * 70)
    print("\nBoth naive and pretrained flies have interaction_event annotations.")
    print("Naive flies interact with the test ball while pretrained flies interact with both.\n")

    for genotype in sorted(all_results.keys()):
        pretraining_data = all_results[genotype].get("pretraining_data", {})
        if pretraining_data:
            print(f"{genotype}:")
            for pretrain_label in sorted(pretraining_data.keys()):
                num_flies = len(pretraining_data[pretrain_label].get("flies", {}))
                if num_flies > 0:
                    print(f"  {pretrain_label:12s}: {num_flies:3d} flies with event data")
                else:
                    print(f"  {pretrain_label:12s}: No flies with event data")
            print()

    print("\n" + "=" * 70)
    print(f"✓ Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
