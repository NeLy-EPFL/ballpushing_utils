#!/usr/bin/env python3
"""
Script to generate heatmaps of fly thorax positions grouped by F1_condition.

This script loads the fly_positions dataset and creates spatial heatmaps showing
where flies spend their time in the arena, grouped by F1 condition.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from itertools import combinations
import argparse
import cv2
from matplotlib import colors


def load_background_image(video_path, frame_number=0):
    """
    Load a frame from a video to use as background.

    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (default: 0 for first frame)

    Returns:
        numpy array of the frame, or None if failed
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to grayscale for better overlay
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame
        else:
            return None
    except Exception as e:
        print(f"Error loading background image: {e}")
        return None


def find_video_file(df, experiment_col="experiment"):
    """
    Find a representative video file from the dataset.

    Args:
        df: DataFrame with experiment information
        experiment_col: Column name containing experiment path info

    Returns:
        Path to a video file, or None if not found
    """
    # Look for experiment or video path columns
    possible_cols = ["fly", "experiment", "Experiment", "video_path", "directory"]

    for col in possible_cols:
        if col in df.columns:
            # Get first non-null entry
            first_entry = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first_entry:
                print(f"Found path info in column '{col}': {first_entry}")

                # Try to construct video path
                # Typical structure: experiment_dir/arena#/corridor#/video.mp4
                path = Path(first_entry)

                # If it's already a video file
                if path.suffix in [".mp4", ".avi", ".mov"]:
                    if path.exists():
                        return path

                # Otherwise search for video files in the directory
                if path.exists() and path.is_dir():
                    for video_ext in ["*.mp4", "*.avi", "*.mov"]:
                        video_files = list(path.glob(video_ext))
                        if video_files:
                            return video_files[0]

                # Try parent directories
                for parent_level in range(3):
                    search_path = path.parents[parent_level] if parent_level < len(path.parents) else path
                    if search_path.exists():
                        for video_ext in ["*.mp4", "*.avi", "*.mov"]:
                            video_files = list(search_path.glob(f"**/{video_ext}"))
                            if video_files:
                                return video_files[0]

    return None


def center_positions_on_start(df, x_col, y_col, fly_col="fly", n_frames=10):
    """
    Center positions for each fly based on the average of the first n frames.
    This makes position (0, 0) the average starting position for each fly.

    Args:
        df: DataFrame with position data
        x_col: Column name for x positions
        y_col: Column name for y positions
        fly_col: Column name for fly identifier
        n_frames: Number of initial frames to average (default: 10)

    Returns:
        tuple: (centered_df, x_offset, y_offset) where offsets are the average centering applied
    """
    df = df.copy()

    all_x_offsets = []
    all_y_offsets = []

    if fly_col not in df.columns:
        print(f"Warning: '{fly_col}' column not found, cannot center per fly")
        # Center on global average of first n rows
        x_offset = df[x_col].iloc[:n_frames].mean()
        y_offset = df[y_col].iloc[:n_frames].mean()
        df[x_col] = df[x_col] - x_offset
        df[y_col] = df[y_col] - y_offset
        print(f"Centered all positions on global average: x_offset={x_offset:.1f}, y_offset={y_offset:.1f}")
        return df, x_offset, y_offset

    # Center each fly's positions on its own starting position
    for fly_id in df[fly_col].unique():
        fly_mask = df[fly_col] == fly_id
        fly_data = df[fly_mask].copy()

        if len(fly_data) < n_frames:
            n_use = len(fly_data)
        else:
            n_use = n_frames

        # Calculate average starting position for this fly
        x_offset = fly_data[x_col].iloc[:n_use].mean()
        y_offset = fly_data[y_col].iloc[:n_use].mean()

        all_x_offsets.append(x_offset)
        all_y_offsets.append(y_offset)

        # Center positions
        df.loc[fly_mask, x_col] = df.loc[fly_mask, x_col] - x_offset
        df.loc[fly_mask, y_col] = df.loc[fly_mask, y_col] - y_offset

    # Return average offset across all flies
    avg_x_offset = float(np.mean(all_x_offsets))
    avg_y_offset = float(np.mean(all_y_offsets))

    print(f"Centered positions for {df[fly_col].nunique()} flies on average of first {n_frames} frames")
    print(f"Average centering offset: x={avg_x_offset:.1f}, y={avg_y_offset:.1f}")
    return df, avg_x_offset, avg_y_offset


def average_fly_heatmap(data, x_col, y_col, bins=100, x_range=None, y_range=None):
    flies = data["fly"].unique()
    heatmaps = []
    xedges = yedges = None
    for fly in flies:
        fly_data = data[data["fly"] == fly]
        x = fly_data[x_col].dropna().values
        y = fly_data[y_col].dropna().values
        if len(x) == 0:
            continue
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
        n_points = len(x)
        if n_points > 0:
            heatmap = heatmap / n_points
        heatmaps.append(heatmap)
    if not heatmaps or xedges is None or yedges is None:
        return None, None, None
    avg_heatmap = np.mean(heatmaps, axis=0)
    return avg_heatmap, xedges, yedges


def create_2d_heatmap(
    data, x_col, y_col, title, ax, bins=100, sigma=2, blur_sigma=1.0, clip_percent=0.05, unclipped=False
):
    """
    Create a 2D histogram heatmap of fly positions (per-fly average), with blur and optional value clipping.
    If unclipped=True, show all values (no min threshold).
    """
    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Remove NaN values
    clean_data = data[[x_col, y_col, "fly"]].dropna()

    if clean_data.empty:
        ax.text(0.5, 0.5, "No valid position data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Use global range for all flies for comparability
    x_range = (clean_data[x_col].min(), clean_data[x_col].max())
    y_range = (clean_data[y_col].min(), clean_data[y_col].max())
    avg_heatmap, xedges, yedges = average_fly_heatmap(
        clean_data, x_col, y_col, bins=bins, x_range=x_range, y_range=y_range
    )

    if avg_heatmap is None or xedges is None or yedges is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Apply Gaussian blur
    if blur_sigma > 0:
        avg_heatmap = gaussian_filter(avg_heatmap, sigma=blur_sigma)

    vmax = np.nanmax(avg_heatmap)
    vmin = np.nanmin(avg_heatmap)
    if unclipped:
        display_map = avg_heatmap.copy()
        threshold = None
    else:
        threshold = vmax * clip_percent
        display_map = avg_heatmap.copy()
        display_map[display_map < threshold] = np.nan

    # Create the extent for the heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Create the heatmap with inverted Y axis (origin='upper' flips the Y)
    cmap = plt.get_cmap("hot")
    cmap.set_bad("white")
    im = ax.imshow(
        display_map.T,
        extent=extent,
        origin="upper",
        cmap=cmap,
        aspect="auto",
        interpolation="bilinear",
        alpha=1.0,  # fully opaque
        zorder=1,
        vmin=(threshold if not unclipped else vmin),
        vmax=vmax,
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Avg. Normalized Density (per fly)")

    # Set labels and title
    ax.set_xlabel("X Position (pixels)", fontsize=12)
    ax.set_ylabel("Y Position (pixels)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=2)

    # Add sample size annotation
    n_flies = len(clean_data["fly"].unique())
    n_points = len(clean_data)
    ax.text(
        0.02,
        0.02,
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return ax


def create_hexbin_heatmap(data, x_col, y_col, title, ax, gridsize=50, reduce_C_function=np.sum):
    """
    Create a hexagonal binning heatmap of fly positions.

    Args:
        data: DataFrame with position data
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        title: Title for the plot
        ax: Matplotlib axes object
        gridsize: Number of hexagons in the x-direction
        reduce_C_function: Function to aggregate data in each hexbin
        background_img: Optional background image (numpy array)
        x_offset: X offset used for centering positions (for background alignment)
        y_offset: Y offset used for centering positions (for background alignment)

    Returns:
        ax: Modified axes object
    """
    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    # Remove NaN values
    clean_data = data[[x_col, y_col]].dropna()

    if clean_data.empty:
        ax.text(0.5, 0.5, "No valid position data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return ax

    x = clean_data[x_col].values
    y = clean_data[y_col].values

    # Invert Y coordinates to match video orientation
    y = -y

    # No background image

    # Normalize C by sample size for density
    n_points = len(x)
    if n_points > 0:
        hexbin = ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            cmap="viridis",
            reduce_C_function=lambda c: reduce_C_function(c) / n_points,
            alpha=1.0,
            zorder=1,
        )
    else:
        hexbin = ax.hexbin(
            x, y, gridsize=gridsize, cmap="viridis", reduce_C_function=reduce_C_function, alpha=1.0, zorder=1
        )

    # Add colorbar
    plt.colorbar(hexbin, ax=ax, label="Normalized Density")

    # Set labels and title
    ax.set_xlabel("X Position (pixels)", fontsize=12)
    ax.set_ylabel("Y Position (pixels)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=2)

    # Add sample size annotation
    n_flies = len(data["fly"].unique()) if "fly" in data.columns else "unknown"
    n_points = len(clean_data)
    ax.text(
        0.02,
        0.02,  # Changed from 0.98 to 0.02 for bottom-left
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",  # Changed from "top" to "bottom"
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return ax


def main():
    """Main function to generate fly position heatmaps."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate heatmaps of fly positions by F1 condition")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/upramdya_data/MD/F1_Tracks/Datasets/251112_11_fly_positions_F1_New_Data/fly_positions/pooled_fly_positions.feather",
        help="Path to fly_positions dataset",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["histogram", "hexbin"],
        default="histogram",
        help="Type of heatmap to generate (default: histogram)",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histogram (default: 100)")
    parser.add_argument("--gridsize", type=int, default=50, help="Grid size for hexbin plot (default: 50)")
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Gaussian smoothing sigma for histogram (default: 2.0)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Downsample factor to reduce number of points (default: 10, use every 10th point)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to a video file to use for background image (optional)",
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        default=100,
        help="Frame number to extract from video for background (default: 100)",
    )

    args = parser.parse_args()

    # Alternative paths to try
    alternative_paths = [
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/latest_fly_positions/fly_positions/pooled_fly_positions.feather",
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/fly_positions/pooled_fly_positions.feather",
        # Add pattern-based search for any dataset with fly_positions
    ]

    # Try to load the dataset
    df = None
    used_path = None

    for path in [args.dataset_path] + alternative_paths:
        try:
            path_obj = Path(path)
            if path_obj.exists():
                print(f"Loading dataset from: {path}")
                df = pd.read_feather(path)
                used_path = path
                print(f"Dataset loaded successfully. Shape: {df.shape}")
                break
        except Exception as e:
            print(f"Could not load dataset from {path}: {e}")
            continue

    if df is None:
        print("Could not load fly_positions dataset from any of the specified paths.")
        print("Please check the dataset path or generate it first using:")
        print("  python dataset_builder.py --mode experiment --yaml <your_yaml> ")
        print("  (with 'fly_positions' added to the metrics list)")
        return

    # Print available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    # Identify required columns
    # Look for thorax positions from fly tracking
    x_thorax_col = None
    y_thorax_col = None

    # Try different possible column names for thorax positions
    possible_x_cols = ["x_thorax_fly_0", "x_thorax", "x_thorax_skeleton"]
    possible_y_cols = ["y_thorax_fly_0", "y_thorax", "y_thorax_skeleton"]

    for col in possible_x_cols:
        if col in df.columns:
            x_thorax_col = col
            break

    for col in possible_y_cols:
        if col in df.columns:
            y_thorax_col = col
            break

    # Look for F1_condition column
    f1_condition_col = None
    for col in df.columns:
        if "f1" in col.lower() and "condition" in col.lower():
            f1_condition_col = col
            break

    # Check if we have the required columns
    if x_thorax_col is None or y_thorax_col is None:
        print(f"\nError: Could not find thorax position columns")
        print(f"Searched for: {possible_x_cols} and {possible_y_cols}")
        print(f"Available columns with 'thorax': {[col for col in df.columns if 'thorax' in col.lower()]}")
        return

    if f1_condition_col is None:
        print(f"\nError: Could not find F1_condition column")
        print(
            f"Available columns with 'f1' or 'condition': {[col for col in df.columns if 'f1' in col.lower() or 'condition' in col.lower()]}"
        )
        return

    print(f"\nUsing columns:")
    print(f"  X thorax: {x_thorax_col}")
    print(f"  Y thorax: {y_thorax_col}")
    print(f"  F1 condition: {f1_condition_col}")

    # Filter out rows with missing essential data
    essential_cols = [x_thorax_col, y_thorax_col, f1_condition_col]
    df_clean = df.dropna(subset=essential_cols)

    print(f"\nDataset shape after removing NaN: {df.shape} -> {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with missing data")

    # Downsample the data to make it more manageable
    if args.downsample > 1:
        print(f"\nDownsampling data by factor of {args.downsample}...")
        df_clean = df_clean.iloc[:: args.downsample].copy()
        print(f"Dataset shape after downsampling: {df_clean.shape}")

    # Center positions on the average of first 10 frames for each fly
    print("\nCentering positions on average of first 10 frames per fly...")
    df_clean, x_offset, y_offset = center_positions_on_start(
        df_clean, x_thorax_col, y_thorax_col, fly_col="fly", n_frames=10
    )

    print(f"Position ranges after centering:")
    print(f"  X: {df_clean[x_thorax_col].min():.1f} to {df_clean[x_thorax_col].max():.1f} pixels")
    print(f"  Y: {df_clean[y_thorax_col].min():.1f} to {df_clean[y_thorax_col].max():.1f} pixels")

    # --- Normalize x coordinates: set x=0 to mean x in first hour for each fly ---
    print("\nNormalizing x coordinates: setting x=0 to mean x in first hour for each fly...")
    time_col = None
    for col in ["time", "Time"]:
        if col in df_clean.columns:
            time_col = col
            break

    if time_col is not None:
        df_norm = df_clean.copy()
        for fly_id in df_norm["fly"].unique():
            fly_mask = df_norm["fly"] == fly_id
            fly_data = df_norm[fly_mask]
            first_hour_mask = (fly_data[time_col] >= 0) & (fly_data[time_col] < 3600)
            if first_hour_mask.sum() == 0:
                continue
            x_corridor_center = fly_data.loc[first_hour_mask, x_thorax_col].mean()
            df_norm.loc[fly_mask, x_thorax_col] = df_norm.loc[fly_mask, x_thorax_col] - x_corridor_center
        print("X coordinates normalized to corridor center for each fly.")
    else:
        print("Warning: No 'time' column found, skipping x normalization.")
        df_norm = df_clean
    # Use df_norm for all further analysis

    # No background image logic

    # Get unique F1 conditions
    f1_conditions = sorted(df_norm[f1_condition_col].unique())
    print(f"\nF1 conditions found: {f1_conditions}")

    # Print sample sizes
    print("\nSample sizes by F1 condition:")
    for condition in f1_conditions:
        condition_data = df_norm[df_norm[f1_condition_col] == condition]
        n_flies = len(condition_data["fly"].unique()) if "fly" in condition_data.columns else "unknown"
        n_points = len(condition_data)
        print(f"  {condition}: {n_flies} flies, {n_points:,} data points")

    # Check if we have a time column for splitting into first/second hour
    time_col = None
    for col in ["time", "Time"]:
        if col in df_norm.columns:
            time_col = col
            break

    if time_col is None:
        print("\nWarning: No 'time' column found, cannot split by hour")
        print(f"Available columns: {list(df_norm.columns)}")
        return

    print(f"\nUsing time column: {time_col}")
    print(f"Time range: {df_norm[time_col].min():.2f}s to {df_norm[time_col].max():.2f}s")

    # Split data into full video and second hour (3600-7200 seconds)
    # Assuming 2-hour recording: 0-3600s (first hour), 3600-7200s (second hour)
    second_hour_start = 3600  # 1 hour in seconds
    second_hour_end = 7200  # 2 hours in seconds

    df_second_hour = df_norm[(df_norm[time_col] >= second_hour_start) & (df_norm[time_col] < second_hour_end)].copy()

    print(f"\nFull video data points: {len(df_norm):,}")
    print(f"Second hour data points: {len(df_second_hour):,}")

    # Create output directory for heatmaps
    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving heatmaps to: {output_dir}")

    # Only plot second hour, split by X < 100 and X >= 100 for each condition
    n_conditions = len(f1_conditions)

    # Create CLIPPED version (all conditions in one figure)
    fig_clipped, axes_clipped = plt.subplots(1, n_conditions * 2, figsize=(6 * n_conditions * 2, 6))
    if n_conditions == 1:
        axes_clipped = np.array([axes_clipped]).flatten()
    else:
        axes_clipped = axes_clipped.flatten()

    for i, condition in enumerate(f1_conditions):
        condition_data = df_second_hour[df_second_hour[f1_condition_col] == condition]
        below_100 = condition_data[condition_data[x_thorax_col] < 100]
        above_100 = condition_data[condition_data[x_thorax_col] >= 100]

        # Clipped - Corridor
        create_2d_heatmap(
            data=below_100,
            x_col=x_thorax_col,
            y_col=y_thorax_col,
            title=f"{condition.replace('_', ' ').title()}\nCorridor",
            ax=axes_clipped[i * 2],
            bins=args.bins,
            sigma=args.sigma,
            blur_sigma=1.0,
            clip_percent=0.05,
            unclipped=False,
        )

        # Clipped - F1 tracks
        create_2d_heatmap(
            data=above_100,
            x_col=x_thorax_col,
            y_col=y_thorax_col,
            title=f"{condition.replace('_', ' ').title()}\nF1 tracks",
            ax=axes_clipped[i * 2 + 1],
            bins=args.bins,
            sigma=args.sigma,
            blur_sigma=1.0,
            clip_percent=0.05,
            unclipped=False,
        )

    fig_clipped.tight_layout()
    output_filename_clipped = f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_clipped"
    if args.downsample > 1:
        output_filename_clipped += f"_ds{args.downsample}"
    output_filename_clipped += ".png"
    output_path_clipped = output_dir / output_filename_clipped
    fig_clipped.savefig(output_path_clipped, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap (clipped) saved to: {output_path_clipped}")
    plt.close(fig_clipped)

    # Create UNCLIPPED version (all conditions in one figure)
    fig_unclipped, axes_unclipped = plt.subplots(1, n_conditions * 2, figsize=(6 * n_conditions * 2, 6))
    if n_conditions == 1:
        axes_unclipped = np.array([axes_unclipped]).flatten()
    else:
        axes_unclipped = axes_unclipped.flatten()

    for i, condition in enumerate(f1_conditions):
        condition_data = df_second_hour[df_second_hour[f1_condition_col] == condition]
        below_100 = condition_data[condition_data[x_thorax_col] < 100]
        above_100 = condition_data[condition_data[x_thorax_col] >= 100]

        # Unclipped - Corridor
        create_2d_heatmap(
            data=below_100,
            x_col=x_thorax_col,
            y_col=y_thorax_col,
            title=f"{condition.replace('_', ' ').title()}\nCorridor",
            ax=axes_unclipped[i * 2],
            bins=args.bins,
            sigma=args.sigma,
            blur_sigma=1.0,
            clip_percent=0.05,
            unclipped=True,
        )

        # Unclipped - F1 tracks
        create_2d_heatmap(
            data=above_100,
            x_col=x_thorax_col,
            y_col=y_thorax_col,
            title=f"{condition.replace('_', ' ').title()}\nF1 tracks",
            ax=axes_unclipped[i * 2 + 1],
            bins=args.bins,
            sigma=args.sigma,
            blur_sigma=1.0,
            clip_percent=0.05,
            unclipped=True,
        )

    fig_unclipped.tight_layout()
    output_filename_unclipped = f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_unclipped"
    if args.downsample > 1:
        output_filename_unclipped += f"_ds{args.downsample}"
    output_filename_unclipped += ".png"
    output_path_unclipped = output_dir / output_filename_unclipped
    fig_unclipped.savefig(output_path_unclipped, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap (unclipped) saved to: {output_path_unclipped}")
    plt.close(fig_unclipped)

    # --- Second plot: group by Pretraining if present ---
    if "Pretraining" in df_clean.columns:
        pretraining_col = "Pretraining"
        pretraining_conditions = sorted(df_clean[pretraining_col].dropna().unique())
        print(f"\nPretraining conditions found: {pretraining_conditions}")

        # Create CLIPPED version (all conditions in one figure)
        fig2_clipped, axes2_clipped = plt.subplots(
            1, len(pretraining_conditions) * 2, figsize=(6 * len(pretraining_conditions) * 2, 6)
        )
        if len(pretraining_conditions) == 1:
            axes2_clipped = np.array([axes2_clipped]).flatten()
        else:
            axes2_clipped = axes2_clipped.flatten()

        for i, condition in enumerate(pretraining_conditions):
            condition_data = df_second_hour[df_second_hour[pretraining_col] == condition]
            below_100 = condition_data[condition_data[x_thorax_col] < 100]
            above_100 = condition_data[condition_data[x_thorax_col] >= 100]

            # Clipped - Corridor
            create_2d_heatmap(
                data=below_100,
                x_col=x_thorax_col,
                y_col=y_thorax_col,
                title=f"{str(condition)}\nCorridor",
                ax=axes2_clipped[i * 2],
                bins=args.bins,
                sigma=args.sigma,
                blur_sigma=1.0,
                clip_percent=0.05,
                unclipped=False,
            )

            # Clipped - F1 tracks
            create_2d_heatmap(
                data=above_100,
                x_col=x_thorax_col,
                y_col=y_thorax_col,
                title=f"{str(condition)}\nF1 tracks",
                ax=axes2_clipped[i * 2 + 1],
                bins=args.bins,
                sigma=args.sigma,
                blur_sigma=1.0,
                clip_percent=0.05,
                unclipped=False,
            )

        fig2_clipped.tight_layout()
        output_filename2_clipped = f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_byPretraining_clipped"
        if args.downsample > 1:
            output_filename2_clipped += f"_ds{args.downsample}"
        output_filename2_clipped += ".png"
        output_path2_clipped = output_dir / output_filename2_clipped
        fig2_clipped.savefig(output_path2_clipped, dpi=300, bbox_inches="tight")
        print(f"\nHeatmap by Pretraining (clipped) saved to: {output_path2_clipped}")
        plt.close(fig2_clipped)

        # Create UNCLIPPED version (all conditions in one figure)
        fig2_unclipped, axes2_unclipped = plt.subplots(
            1, len(pretraining_conditions) * 2, figsize=(6 * len(pretraining_conditions) * 2, 6)
        )
        if len(pretraining_conditions) == 1:
            axes2_unclipped = np.array([axes2_unclipped]).flatten()
        else:
            axes2_unclipped = axes2_unclipped.flatten()

        for i, condition in enumerate(pretraining_conditions):
            condition_data = df_second_hour[df_second_hour[pretraining_col] == condition]
            below_100 = condition_data[condition_data[x_thorax_col] < 100]
            above_100 = condition_data[condition_data[x_thorax_col] >= 100]

            # Unclipped - Corridor
            create_2d_heatmap(
                data=below_100,
                x_col=x_thorax_col,
                y_col=y_thorax_col,
                title=f"{str(condition)}\nCorridor",
                ax=axes2_unclipped[i * 2],
                bins=args.bins,
                sigma=args.sigma,
                blur_sigma=1.0,
                clip_percent=0.05,
                unclipped=True,
            )

            # Unclipped - F1 tracks
            create_2d_heatmap(
                data=above_100,
                x_col=x_thorax_col,
                y_col=y_thorax_col,
                title=f"{str(condition)}\nF1 tracks",
                ax=axes2_unclipped[i * 2 + 1],
                bins=args.bins,
                sigma=args.sigma,
                blur_sigma=1.0,
                clip_percent=0.05,
                unclipped=True,
            )

        fig2_unclipped.tight_layout()
        output_filename2_unclipped = f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_byPretraining_unclipped"
        if args.downsample > 1:
            output_filename2_unclipped += f"_ds{args.downsample}"
        output_filename2_unclipped += ".png"
        output_path2_unclipped = output_dir / output_filename2_unclipped
        fig2_unclipped.savefig(output_path2_unclipped, dpi=300, bbox_inches="tight")
        print(f"\nHeatmap by Pretraining (unclipped) saved to: {output_path2_unclipped}")
        plt.close(fig2_unclipped)

        # --- Difference row if exactly two Pretraining conditions ---
        if len(pretraining_conditions) == 2:
            control, test = pretraining_conditions[0], pretraining_conditions[1]
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
            for j, (region, region_label) in enumerate(
                [
                    (lambda df: df[df[x_thorax_col] < 100], "Corridor"),
                    (lambda df: df[df[x_thorax_col] >= 100], "F1 tracks"),
                ]
            ):

                def get_avg_heatmap(df):
                    clean_data = df[[x_thorax_col, y_thorax_col, "fly"]].dropna()
                    if clean_data.empty:
                        return None, None, None
                    x_range = (clean_data[x_thorax_col].min(), clean_data[x_thorax_col].max())
                    y_range = (clean_data[y_thorax_col].min(), clean_data[y_thorax_col].max())
                    return average_fly_heatmap(
                        clean_data, x_thorax_col, y_thorax_col, bins=args.bins, x_range=x_range, y_range=y_range
                    )

                df_control = df_second_hour[df_second_hour[pretraining_col] == control]
                df_test = df_second_hour[df_second_hour[pretraining_col] == test]
                heatmap_control, xedges, yedges = get_avg_heatmap(region(df_control))
                heatmap_test, _, _ = get_avg_heatmap(region(df_test))
                if heatmap_control is None or heatmap_test is None or xedges is None or yedges is None:
                    axes3[j].text(0.5, 0.5, "No data available", ha="center", va="center", transform=axes3[j].transAxes)
                    axes3[j].set_title(f"Difference (test - control)\n{region_label}", fontsize=14, fontweight="bold")
                    continue
                diff = heatmap_test - heatmap_control
                # Blur the difference
                diff_blur = gaussian_filter(diff, sigma=1.0)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                # Use diverging colormap centered at zero
                vmax = np.nanmax(np.abs(diff_blur))
                im = axes3[j].imshow(
                    diff_blur.T,
                    extent=extent,
                    origin="upper",
                    cmap="seismic",
                    aspect="auto",
                    interpolation="bilinear",
                    alpha=1.0,
                    zorder=1,
                    vmin=-vmax,
                    vmax=vmax,
                )
                axes3[j].set_title(f"Difference (test - control)\n{region_label}", fontsize=14, fontweight="bold")
                axes3[j].set_xlabel("X Position (pixels)")
                axes3[j].set_ylabel("Y Position (pixels)")
                plt.colorbar(im, ax=axes3[j], label="Δ Avg. Normalized Density (per fly)")
            plt.tight_layout()
            output_filename3 = f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_byPretraining_DIFF"
            if args.downsample > 1:
                output_filename3 += f"_ds{args.downsample}"
            output_filename3 += ".png"
            output_path3 = output_dir / output_filename3
            plt.savefig(output_path3, dpi=300, bbox_inches="tight")
            print(f"\nDifference heatmap by Pretraining saved to: {output_path3}")
            plt.show()

    # --- F1_condition difference plots: all pairwise comparisons ---
    if len(f1_conditions) >= 2:
        from itertools import combinations

        for cond1, cond2 in combinations(f1_conditions, 2):
            fig_diff, axes_diff = plt.subplots(1, 2, figsize=(12, 6))
            for j, (region, region_label) in enumerate(
                [
                    (lambda df: df[df[x_thorax_col] < 100], "Corridor"),
                    (lambda df: df[df[x_thorax_col] >= 100], "F1 tracks"),
                ]
            ):

                def get_avg_heatmap(df):
                    clean_data = df[[x_thorax_col, y_thorax_col, "fly"]].dropna()
                    if clean_data.empty:
                        return None, None, None
                    x_range = (clean_data[x_thorax_col].min(), clean_data[x_thorax_col].max())
                    y_range = (clean_data[y_thorax_col].min(), clean_data[y_thorax_col].max())
                    return average_fly_heatmap(
                        clean_data, x_thorax_col, y_thorax_col, bins=args.bins, x_range=x_range, y_range=y_range
                    )

                df_cond1 = df_second_hour[df_second_hour[f1_condition_col] == cond1]
                df_cond2 = df_second_hour[df_second_hour[f1_condition_col] == cond2]
                heatmap_cond1, xedges, yedges = get_avg_heatmap(region(df_cond1))
                heatmap_cond2, _, _ = get_avg_heatmap(region(df_cond2))
                if heatmap_cond1 is None or heatmap_cond2 is None or xedges is None or yedges is None:
                    axes_diff[j].text(
                        0.5, 0.5, "No data available", ha="center", va="center", transform=axes_diff[j].transAxes
                    )
                    axes_diff[j].set_title(
                        f"Difference ({cond2} - {cond1})\n{region_label}", fontsize=14, fontweight="bold"
                    )
                    continue
                diff = heatmap_cond2 - heatmap_cond1
                diff_blur = gaussian_filter(diff, sigma=1.0)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                vmax = np.nanmax(np.abs(diff_blur))
                im = axes_diff[j].imshow(
                    diff_blur.T,
                    extent=extent,
                    origin="upper",
                    cmap="seismic",
                    aspect="auto",
                    interpolation="bilinear",
                    alpha=1.0,
                    zorder=1,
                    vmin=-vmax,
                    vmax=vmax,
                )
                axes_diff[j].set_title(
                    f"Difference ({cond2} - {cond1})\n{region_label}", fontsize=14, fontweight="bold"
                )
                axes_diff[j].set_xlabel("X Position (pixels)")
                axes_diff[j].set_ylabel("Y Position (pixels)")
                plt.colorbar(im, ax=axes_diff[j], label="Δ Avg. Normalized Density (per fly)")
            plt.tight_layout()
            output_filename_diff = (
                f"fly_position_heatmap_{args.plot_type}_second_hour_splitX_byF1Condition_DIFF_{cond2}_vs_{cond1}"
            )
            if args.downsample > 1:
                output_filename_diff += f"_ds{args.downsample}"
            output_filename_diff += ".png"
            output_path_diff = output_dir / output_filename_diff
            plt.savefig(output_path_diff, dpi=300, bbox_inches="tight")
            print(f"\nDifference heatmap by F1_condition saved to: {output_path_diff}")
            plt.show()


if __name__ == "__main__":
    data = main()
