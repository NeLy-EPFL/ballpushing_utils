#!/usr/bin/env python3
"""
Debug script to check ball center coordinates and visualize skeleton data with ball positions.
This script will help identify issues with NaN values and coordinate preprocessing.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Ballpushing_utils import Fly, Experiment
import warnings

warnings.filterwarnings("ignore")


def check_ball_coordinates(fly):
    """Check ball coordinate data for NaN values and print summary."""
    print(f"\n=== Ball Coordinates Analysis for {fly.metadata.name} ===")

    # Check raw ball data
    if hasattr(fly.tracking_data, "raw_balltrack") and fly.tracking_data.raw_balltrack:
        ball_data = fly.tracking_data.raw_balltrack.objects[0].dataset
        print(f"\nRaw ball data columns: {list(ball_data.columns)}")

        # Check for different coordinate column names
        coord_columns = [col for col in ball_data.columns if any(coord in col.lower() for coord in ["centre"])]
        print(f"Center-related columns: {coord_columns}")

        for col in coord_columns:
            nan_count = ball_data[col].isna().sum()
            total_count = len(ball_data[col])
            print(f"{col}: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")
            if nan_count < total_count:
                valid_values = ball_data[col].dropna()
                if len(valid_values) > 0:
                    print(f"  Range: {valid_values.min():.2f} to {valid_values.max():.2f}")

        return ball_data
    else:
        print("No raw ball tracking data found")
        return None


def check_skeleton_data(fly):
    """Check skeleton data availability."""
    print(f"\n=== Skeleton Data Analysis ===")

    if hasattr(fly.tracking_data, "skeletontrack") and fly.tracking_data.skeletontrack:
        skeleton_data = fly.tracking_data.skeletontrack.objects[0].dataset
        print(f"Skeleton data shape: {skeleton_data.shape}")
        print(f"Skeleton columns: {list(skeleton_data.columns)}")

        # Check for NaN values in skeleton data
        nan_columns = []
        for col in skeleton_data.columns:
            nan_count = skeleton_data[col].isna().sum()
            if nan_count > 0:
                nan_columns.append(f"{col}: {nan_count}")

        if nan_columns:
            print(f"Columns with NaN values: {nan_columns}")
        else:
            print("No NaN values in skeleton data")

        return skeleton_data
    else:
        print("No skeleton tracking data found")
        return None


def find_sharp_ball_movements(fly, threshold=10):
    """
    Find frames where the ball has sharp movements (displacement > threshold).

    Args:
        fly: Fly object with tracking data
        threshold: Minimum displacement in pixels to consider a sharp movement

    Returns:
        list: List of frame indices where sharp movements occur
    """
    print(f"\n=== Finding Sharp Ball Movements (threshold: {threshold}px) ===")

    # Get raw ball data
    if not hasattr(fly.tracking_data, "raw_balltrack") or not fly.tracking_data.raw_balltrack:
        print("No raw ball tracking data available")
        return []

    ball_data = fly.tracking_data.raw_balltrack.objects[0].dataset

    # Check for coordinate columns
    if "x_centre_raw" in ball_data.columns and "y_centre_raw" in ball_data.columns:
        x_col, y_col = "x_centre_raw", "y_centre_raw"
        print("Using x_centre_raw and y_centre_raw for movement detection")
    elif "x_centre" in ball_data.columns and "y_centre" in ball_data.columns:
        x_col, y_col = "x_centre", "y_centre"
        print("Using x_centre and y_centre for movement detection")
    else:
        print("No valid ball coordinates found for movement detection")
        return []

    # Calculate frame-to-frame displacement
    x_coords = ball_data[x_col].dropna()
    y_coords = ball_data[y_col].dropna()

    if len(x_coords) < 2:
        print("Insufficient valid coordinates for movement calculation")
        return []

    # Calculate displacement between consecutive frames
    x_diff = np.abs(np.diff(x_coords))
    y_diff = np.abs(np.diff(y_coords))
    displacement = np.sqrt(x_diff**2 + y_diff**2)

    # Find frames with displacement > threshold
    sharp_movement_indices = np.where(displacement > threshold)[0]

    # Convert back to original frame indices (accounting for dropna)
    valid_indices = x_coords.index[:-1]  # Remove last index since diff is one element shorter
    sharp_movement_frames = valid_indices[sharp_movement_indices].tolist()

    print(f"Found {len(sharp_movement_frames)} frames with sharp movements (>{threshold}px)")
    if len(sharp_movement_frames) > 0:
        print(f"Sharp movement frames: {sharp_movement_frames[:10]}...")  # Show first 10
        print(f"Max displacement: {displacement.max():.2f}px")
        print(f"Mean displacement during sharp movements: {displacement[sharp_movement_indices].mean():.2f}px")

    return sharp_movement_frames


def select_frames_around_movements(sharp_movement_frames, context_frames=2, max_groups=3):
    """
    Select representative frames around sharp movements for visualization.

    Args:
        sharp_movement_frames: List of frame indices with sharp movements
        context_frames: Number of frames before/after movement to include
        max_groups: Maximum number of movement groups to visualize

    Returns:
        list: Selected frame indices for visualization
    """
    if not sharp_movement_frames:
        return [100, 200, 300]  # Default frames if no movements found

    # Group consecutive movements
    movement_groups = []
    current_group = [sharp_movement_frames[0]]

    for i in range(1, len(sharp_movement_frames)):
        if sharp_movement_frames[i] - sharp_movement_frames[i - 1] <= context_frames * 2:
            current_group.append(sharp_movement_frames[i])
        else:
            movement_groups.append(current_group)
            current_group = [sharp_movement_frames[i]]
    movement_groups.append(current_group)

    # Select frames from the first few movement groups
    selected_frames = []
    for group in movement_groups[:max_groups]:
        # Take the middle frame of the group and add context
        middle_frame = group[len(group) // 2]
        frames_to_add = [middle_frame - context_frames, middle_frame, middle_frame + context_frames]
        selected_frames.extend([f for f in frames_to_add if f > 0])

    # Remove duplicates and sort
    selected_frames = sorted(list(set(selected_frames)))

    print(f"Selected {len(selected_frames)} frames around sharp movements: {selected_frames}")
    return selected_frames


def plot_frames_with_annotations(fly, ball_data, skeleton_data, frames_to_plot=[100, 200, 300]):
    """Plot specific frames with skeleton keypoints and ball positions using SleapUtils."""
    print(f"\n=== Plotting frames {frames_to_plot} with video annotations ===")

    try:
        from utils_behavior import Sleap_utils
        import cv2

        # Get the sleap tracks objects
        sleap_tracks_list = []

        # Add skeleton tracks if available
        if hasattr(fly.tracking_data, "skeletontrack") and fly.tracking_data.skeletontrack:
            sleap_tracks_list.append(fly.tracking_data.skeletontrack)
            print("Added skeleton tracking data")

        # Add ball tracks (preprocessed) if available
        if hasattr(fly, "skeleton_metrics") and fly.skeleton_metrics and fly.skeleton_metrics.ball:
            sleap_tracks_list.append(fly.skeleton_metrics.ball)
            print("Added preprocessed ball tracking data")
        elif hasattr(fly.tracking_data, "balltrack") and fly.tracking_data.balltrack:
            sleap_tracks_list.append(fly.tracking_data.balltrack)
            print("Added regular ball tracking data")

        if not sleap_tracks_list:
            print("No tracking data available for visualization")
            return

        # Get video file
        video_file = None
        if hasattr(fly.tracking_data, "skeletontrack") and fly.tracking_data.skeletontrack:
            video_file = fly.tracking_data.skeletontrack.video
        elif hasattr(fly.tracking_data, "balltrack") and fly.tracking_data.balltrack:
            video_file = fly.tracking_data.balltrack.video

        if not video_file or not Path(video_file).exists():
            print(f"Video file not found: {video_file}")
            return

        print(f"Using video file: {video_file}")

        # Create figure for subplots
        fig, axes = plt.subplots(1, len(frames_to_plot), figsize=(20, 6))
        if len(frames_to_plot) == 1:
            axes = [axes]

        for i, frame_idx in enumerate(frames_to_plot):
            print(f"Processing frame {frame_idx}...")

            try:
                # Generate annotated frame using SleapUtils
                # Define nodes to show - include all skeleton nodes and preprocessed ball center
                nodes_to_show = []

                # Add skeleton nodes
                if hasattr(fly.tracking_data, "skeletontrack") and fly.tracking_data.skeletontrack:
                    skeleton_nodes = fly.tracking_data.skeletontrack.node_names
                    nodes_to_show.extend(skeleton_nodes)

                # Add preprocessed ball center
                if hasattr(fly, "skeleton_metrics") and fly.skeleton_metrics and fly.skeleton_metrics.ball:
                    ball_nodes = [node for node in fly.skeleton_metrics.ball.node_names if "preprocessed" in node]
                    nodes_to_show.extend(ball_nodes)
                    print(f"Ball nodes to show: {ball_nodes}")

                print(f"Nodes to show: {nodes_to_show}")

                # Generate the annotated frame
                annotated_frame = Sleap_utils.generate_annotated_frame(
                    video=video_file,
                    sleap_tracks_list=sleap_tracks_list,
                    frame=frame_idx,
                    nodes=nodes_to_show,
                    labels=False,
                    edges=True,
                    colorby="Nodes",
                )

                # Convert BGR to RGB for matplotlib
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display in subplot
                axes[i].imshow(annotated_frame_rgb)
                axes[i].set_title(f"Frame {frame_idx}")
                axes[i].axis("off")

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Create empty subplot on error
                axes[i].text(
                    0.5, 0.5, f"Error loading\nframe {frame_idx}", ha="center", va="center", transform=axes[i].transAxes
                )
                axes[i].set_title(f"Frame {frame_idx} (Error)")
                axes[i].axis("off")

        plt.tight_layout()

        # Save the plot
        output_dir = Path(__file__).parent / "debug_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"annotated_frames_{fly.metadata.name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Annotated frames saved to: {output_path}")
        plt.show()

        # Also save individual frames for detailed inspection
        for i, frame_idx in enumerate(frames_to_plot):
            try:
                # Generate single frame
                nodes_to_show = []
                if hasattr(fly.tracking_data, "skeletontrack") and fly.tracking_data.skeletontrack:
                    nodes_to_show.extend(fly.tracking_data.skeletontrack.node_names)
                if hasattr(fly, "skeleton_metrics") and fly.skeleton_metrics and fly.skeleton_metrics.ball:
                    ball_nodes = [node for node in fly.skeleton_metrics.ball.node_names if "preprocessed" in node]
                    nodes_to_show.extend(ball_nodes)

                annotated_frame = Sleap_utils.generate_annotated_frame(
                    video=video_file,
                    sleap_tracks_list=sleap_tracks_list,
                    frame=frame_idx,
                    nodes=nodes_to_show,
                    labels=False,
                    edges=True,
                    colorby="Nodes",
                )

                # Save individual frame
                individual_path = output_dir / f"frame_{frame_idx}_{fly.metadata.name}.png"
                cv2.imwrite(str(individual_path), annotated_frame)
                print(f"Individual frame {frame_idx} saved to: {individual_path}")

            except Exception as e:
                print(f"Error saving individual frame {frame_idx}: {e}")

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("Falling back to coordinate plotting...")
        # Fallback to simple coordinate plotting if SleapUtils not available
        plot_frames_simple(fly, ball_data, skeleton_data, frames_to_plot)
    except Exception as e:
        print(f"Error in annotated plotting: {e}")
        import traceback

        traceback.print_exc()


def plot_frames_simple(fly, ball_data, skeleton_data, frames_to_plot):
    """Fallback simple coordinate plotting without video background."""
    print("Using simple coordinate plotting (no video background)")

    fig, axes = plt.subplots(1, len(frames_to_plot), figsize=(15, 5))
    if len(frames_to_plot) == 1:
        axes = [axes]

    # Get video dimensions from fly config
    width = fly.config.template_width if hasattr(fly.config, "template_width") else 640
    height = fly.config.template_height if hasattr(fly.config, "template_height") else 480

    for i, frame_idx in enumerate(frames_to_plot):
        ax = axes[i]

        # Create a blank background
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Inverted y-axis for image coordinates
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Frame {frame_idx}")

        # Plot skeleton keypoints if available
        if skeleton_data is not None and frame_idx < len(skeleton_data):
            coord_cols = [col for col in skeleton_data.columns if col.endswith(("_x", "_y"))]
            keypoint_names = list(set([col.rsplit("_", 1)[0] for col in coord_cols]))

            import matplotlib.cm as cm

            colors = cm.get_cmap("tab10")(range(len(keypoint_names)))
            for j, keypoint in enumerate(keypoint_names):
                x_col = f"{keypoint}_x"
                y_col = f"{keypoint}_y"

                if x_col in skeleton_data.columns and y_col in skeleton_data.columns:
                    x = skeleton_data.iloc[frame_idx][x_col]
                    y = skeleton_data.iloc[frame_idx][y_col]

                    if not (pd.isna(x) or pd.isna(y)):
                        ax.plot(x, y, "o", markersize=4, color=colors[j], label=keypoint if i == 0 else "", alpha=0.7)

        # Plot ball positions if available
        if ball_data is not None and frame_idx < len(ball_data):
            # Check for preprocessed coordinates
            if "x_centre_preprocessed" in ball_data.columns and "y_centre_preprocessed" in ball_data.columns:
                x = ball_data.iloc[frame_idx]["x_centre_preprocessed"]
                y = ball_data.iloc[frame_idx]["y_centre_preprocessed"]
                if not (pd.isna(x) or pd.isna(y)):
                    ax.plot(x, y, "s", markersize=10, color="red", label="Ball (preprocessed)" if i == 0 else "")
                    from matplotlib.patches import Circle

                    ax.add_patch(Circle((x, y), 12, fill=False, color="red", linewidth=2))

        if i == 0:  # Only show legend for first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    # Save the plot
    output_dir = Path(__file__).parent / "debug_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"simple_coordinates_{fly.metadata.name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Simple coordinate plot saved to: {output_path}")
    plt.show()


def test_preprocessing(fly):
    """Test the ball preprocessing function specifically for SkeletonMetrics."""
    print(f"\n=== Testing SkeletonMetrics Ball Preprocessing ===")

    if not hasattr(fly, "skeleton_metrics") or fly.skeleton_metrics is None:
        print("No skeleton metrics found. Creating SkeletonMetrics object...")
        try:
            from Ballpushing_utils.skeleton_metrics import SkeletonMetrics

            # First check the raw ball data columns before preprocessing
            raw_ball_data = fly.tracking_data.raw_balltrack.objects[0].dataset
            print(f"Raw ball data columns before preprocessing: {list(raw_ball_data.columns)}")

            # Check for NaN values in different coordinate columns
            coord_columns = [col for col in raw_ball_data.columns if "centre" in col.lower()]
            for col in coord_columns:
                nan_count = raw_ball_data[col].isna().sum()
                total_count = len(raw_ball_data[col])
                print(f"  {col}: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")

            # Now create SkeletonMetrics which will run preprocessing
            print("\nCreating SkeletonMetrics object (this will run preprocessing)...")
            skeleton_metrics = SkeletonMetrics(fly)
            print("SkeletonMetrics created successfully")

            # Check the preprocessed data
            ball_data = skeleton_metrics.ball.objects[0].dataset
            print(f"\nColumns after preprocessing: {list(ball_data.columns)}")

            # Check for preprocessed coordinates
            if "x_centre_preprocessed" in ball_data.columns:
                nan_count = ball_data["x_centre_preprocessed"].isna().sum()
                total_count = len(ball_data["x_centre_preprocessed"])
                print(f"x_centre_preprocessed: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")
                if nan_count < total_count:
                    valid_values = ball_data["x_centre_preprocessed"].dropna()
                    print(f"  Range: {valid_values.min():.2f} to {valid_values.max():.2f}")

            if "y_centre_preprocessed" in ball_data.columns:
                nan_count = ball_data["y_centre_preprocessed"].isna().sum()
                total_count = len(ball_data["y_centre_preprocessed"])
                print(f"y_centre_preprocessed: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")
                if nan_count < total_count:
                    valid_values = ball_data["y_centre_preprocessed"].dropna()
                    print(f"  Range: {valid_values.min():.2f} to {valid_values.max():.2f}")

            # Test fly-centered tracks
            print(f"\n=== Testing Fly-Centered Tracks ===")
            fly_centered = skeleton_metrics.compute_fly_centered_tracks()
            print(f"Fly-centered tracks shape: {fly_centered.shape}")

            # Check for raw coordinate columns in fly-centered data
            raw_cols = [col for col in fly_centered.columns if "centre_raw" in col]
            print(f"Raw coordinate columns in fly-centered data: {raw_cols}")

            for col in raw_cols:
                nan_count = fly_centered[col].isna().sum()
                total_count = len(fly_centered[col])
                print(f"  {col}: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")

            # Test contact detection
            print(f"\n=== Testing Contact Detection ===")
            print(f"Number of contact events found: {len(skeleton_metrics.all_contacts)}")
            if len(skeleton_metrics.all_contacts) > 0:
                print(f"First few contact events: {skeleton_metrics.all_contacts[:3]}")

            return ball_data

        except Exception as e:
            print(f"Error creating SkeletonMetrics: {e}")
            import traceback

            traceback.print_exc()
            return None
    else:
        print("SkeletonMetrics already exists")
        return fly.skeleton_metrics.ball.objects[0].dataset


def main():
    parser = argparse.ArgumentParser(description="Debug ball center coordinates with focus on sharp movements")
    parser.add_argument("--path", required=True, help="Path to fly or experiment data")
    parser.add_argument(
        "--mode", choices=["fly", "experiment"], required=True, help="Whether to process a single fly or experiment"
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=None,
        help="Specific frame numbers to plot (default: auto-detect based on ball movements)",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=10.0,
        help="Threshold in pixels for detecting sharp ball movements (default: 10.0)",
    )
    parser.add_argument(
        "--use-default-frames",
        action="store_true",
        help="Use default frames [100, 200, 300] instead of movement-based detection",
    )

    args = parser.parse_args()

    # Load the data
    if args.mode == "fly":
        fly = Fly(Path(args.path), as_individual=True)
        flies = [fly]
    else:
        experiment = Experiment(Path(args.path))
        flies = experiment.flies[:1]  # Just use first fly for testing

    if not flies:
        print("No flies found!")
        return

    fly = flies[0]
    print(f"Testing with fly: {fly.metadata.name}")

    # Check original ball coordinates
    ball_data = check_ball_coordinates(fly)

    # Check skeleton data
    skeleton_data = check_skeleton_data(fly)

    # Test preprocessing
    preprocessed_ball_data = test_preprocessing(fly)

    # Determine frames to plot
    if args.use_default_frames or args.frames:
        frames_to_plot = args.frames if args.frames else [100, 200, 300]
        print(f"Using specified/default frames: {frames_to_plot}")
    else:
        # Find frames with sharp ball movements
        print(f"Auto-detecting frames with sharp movements (threshold: {args.movement_threshold}px)")
        sharp_movement_frames = find_sharp_ball_movements(fly, threshold=args.movement_threshold)

        if sharp_movement_frames:
            frames_to_plot = select_frames_around_movements(sharp_movement_frames, context_frames=2, max_groups=3)
            print(f"Selected frames based on ball movements: {frames_to_plot}")
        else:
            print("No sharp movements detected, using default frames")
            frames_to_plot = [100, 200, 300]

    # Plot frames with annotations
    if ball_data is not None or preprocessed_ball_data is not None:
        data_to_plot = preprocessed_ball_data if preprocessed_ball_data is not None else ball_data
        plot_frames_with_annotations(fly, data_to_plot, skeleton_data, frames_to_plot)

    # Additional analysis of ball movements
    if preprocessed_ball_data is not None:
        analyze_movement_quality(fly, preprocessed_ball_data, args.movement_threshold)

    print(f"\n=== Summary ===")
    print(f"✓ Analysis complete for {fly.metadata.name}")
    print(f"✓ Focused on frames with sharp ball movements (>{args.movement_threshold}px)")


def analyze_movement_quality(fly, ball_data, threshold):
    """Analyze the quality of ball tracking during movements."""
    print(f"\n=== Movement Quality Analysis ===")

    # Compare raw vs preprocessed coordinates if both are available
    if "x_centre_raw" in ball_data.columns and "x_centre_preprocessed" in ball_data.columns:
        raw_x = ball_data["x_centre_raw"].dropna()
        raw_y = ball_data["y_centre_raw"].dropna()
        prep_x = ball_data["x_centre_preprocessed"].dropna()
        prep_y = ball_data["y_centre_preprocessed"].dropna()

        if len(raw_x) > 1 and len(prep_x) > 1:
            # Calculate movements for both
            raw_movement = np.sqrt(np.diff(raw_x) ** 2 + np.diff(raw_y) ** 2)
            prep_movement = np.sqrt(np.diff(prep_x) ** 2 + np.diff(prep_y) ** 2)

            # Find frames with large movements
            large_movements = raw_movement > threshold

            if np.any(large_movements):
                print(f"Frames with large movements (>{threshold}px): {np.sum(large_movements)}")
                print(f"Max raw movement: {raw_movement.max():.2f}px")
                print(f"Max preprocessed movement: {prep_movement.max():.2f}px")
                print(f"Mean movement during large movements:")
                print(f"  Raw: {raw_movement[large_movements].mean():.2f}px")
                print(f"  Preprocessed: {prep_movement[large_movements].mean():.2f}px")
            else:
                print(f"No movements larger than {threshold}px detected")

    # Check for NaN values during movements
    total_frames = len(ball_data)
    nan_frames_raw = ball_data["x_centre_raw"].isna().sum() if "x_centre_raw" in ball_data.columns else 0
    nan_frames_prep = (
        ball_data["x_centre_preprocessed"].isna().sum() if "x_centre_preprocessed" in ball_data.columns else 0
    )

    print(f"Data completeness:")
    print(f"  Total frames: {total_frames}")
    if "x_centre_raw" in ball_data.columns:
        print(f"  Raw coordinates NaN: {nan_frames_raw} ({nan_frames_raw/total_frames*100:.1f}%)")
    if "x_centre_preprocessed" in ball_data.columns:
        print(f"  Preprocessed coordinates NaN: {nan_frames_prep} ({nan_frames_prep/total_frames*100:.1f}%)")


if __name__ == "__main__":
    main()
