#!/usr/bin/env python3
"""
Test script to align a single fly's trajectory to the template using template matching.
Shows fly positions colored by time overlaid on the template.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle


def binarize_template(template, invert_background=True):
    """Binarize the template image."""
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if invert_background:
        return binary
    else:
        return cv2.bitwise_not(binary)


def binarize_video_frame(frame, method="otsu"):
    """Binarize video frame to match template."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return binary


def add_padding(image, padding_percent=0.4):
    """Add padding around image."""
    h, w = image.shape[:2]
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)

    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0])
    else:
        mean_color = np.mean(image, axis=(0, 1))
        padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=mean_color.tolist())

    return padded, (pad_h, pad_h, pad_w, pad_w)


def match_binary_multiscale(frame_binary, template_binary, scale_range=(0.3, 0.8), scale_steps=60):
    """Match binary template to binary frame at multiple scales."""
    frame_h, frame_w = frame_binary.shape
    template_h, template_w = template_binary.shape

    best_score = -np.inf
    best_loc = None
    best_scale = None
    best_template = None

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    for scale in scales:
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        if new_w > frame_w or new_h > frame_h:
            continue

        resized_template = cv2.resize(template_binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(frame_binary, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale
            best_template = resized_template

    if best_loc is None:
        return None

    return {
        "location": best_loc,
        "score": best_score,
        "scale": best_scale,
        "template": best_template,
        "size": (best_template.shape[1], best_template.shape[0]),
    }


def detect_arena_in_video(video_path, template_binary, padding_percent=0.4):
    """
    Detect arena position and scale in a video using the last frame.

    Returns:
        Dictionary with detection results including transformation parameters
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read frame from {video_path}")
        return None

    # Add padding
    frame_padded, padding = add_padding(frame, padding_percent=padding_percent)

    # Binarize
    frame_binary = binarize_video_frame(frame_padded, method="otsu")

    # Match
    match_result = match_binary_multiscale(frame_binary, template_binary, scale_range=(0.3, 0.8), scale_steps=60)

    if match_result is None or match_result["score"] < 0.7:
        print(f"Warning: Low match score ({match_result['score'] if match_result else 'None'}) for {video_path}")
        return None

    # Location in padded frame
    padded_loc = match_result["location"]

    # Location in original frame (accounting for padding)
    original_loc = (padded_loc[0] - padding[2], padded_loc[1] - padding[0])

    # Template size in video coordinates
    template_w, template_h = match_result["size"]

    return {
        "video_path": str(video_path),
        "score": match_result["score"],
        "scale": match_result["scale"],
        "arena_x": original_loc[0],  # Top-left corner in original frame
        "arena_y": original_loc[1],
        "arena_width": template_w,
        "arena_height": template_h,
        "video_width": frame.shape[1],
        "video_height": frame.shape[0],
    }


def transform_fly_positions_to_template(fly_data, arena_params, template_shape, x_col="x_thorax_fly_0", y_col="y_thorax_fly_0"):
    """
    Transform fly positions from video coordinates to template coordinates.

    Args:
        fly_data: DataFrame with fly positions in video coordinates
        arena_params: Arena detection parameters from detect_arena_in_video
        template_shape: Shape of the original template (h, w, c)
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates

    Returns:
        DataFrame with transformed coordinates
    """
    df = fly_data.copy()

    # Extract coordinates
    x_video = df[x_col].values
    y_video = df[y_col].values

    # Transform to arena-relative coordinates (top-left of detected arena is 0,0)
    x_arena = x_video - arena_params["arena_x"]
    y_arena = y_video - arena_params["arena_y"]

    # Scale to template size
    scale = arena_params["scale"]
    x_template = x_arena / scale
    y_template = y_arena / scale

    # Add to dataframe
    df["x_template"] = x_template
    df["y_template"] = y_template

    # Filter out points outside template bounds
    template_h, template_w = template_shape[:2]
    mask = (x_template >= 0) & (x_template < template_w) & (y_template >= 0) & (y_template < template_h)
    df_filtered = df[mask].copy()

    print(f"Filtered {len(df) - len(df_filtered)} points outside template bounds ({len(df_filtered)}/{len(df)} kept)")

    return df_filtered


def plot_fly_trajectory_on_template(fly_data, template, output_path, time_col="time"):
    """
    Plot fly trajectory colored by time on the template image.

    Args:
        fly_data: DataFrame with x_template, y_template, and time columns
        template: Template image (BGR)
        output_path: Path to save the output image
        time_col: Column name for time
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))

    # Show template as background
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    ax.imshow(template_rgb, aspect="auto")

    # Get time values for colormap
    if time_col in fly_data.columns:
        times = fly_data[time_col].values
        time_min = times.min()
        time_max = times.max()

        # Normalize times to [0, 1] for colormap
        times_norm = (times - time_min) / (time_max - time_min) if time_max > time_min else np.zeros_like(times)

        # Create colormap
        cmap = plt.cm.viridis

        # Plot points colored by time
        scatter = ax.scatter(
            fly_data["x_template"],
            fly_data["y_template"],
            c=times,
            cmap=cmap,
            s=1,
            alpha=0.5,
            edgecolors="none",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=f"Time (s)", fraction=0.046, pad=0.04)

        title = f"Fly Trajectory on Template\n{len(fly_data)} points, {time_min:.1f}s to {time_max:.1f}s"
    else:
        # Plot without color coding
        ax.scatter(
            fly_data["x_template"],
            fly_data["y_template"],
            c="red",
            s=1,
            alpha=0.5,
            edgecolors="none",
        )
        title = f"Fly Trajectory on Template\n{len(fly_data)} points"

    ax.set_xlim(0, template.shape[1])
    ax.set_ylim(template.shape[0], 0)
    ax.set_xlabel("X (template pixels)", fontsize=12)
    ax.set_ylabel("Y (template pixels)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved trajectory plot to: {output_path}")
    plt.close()


def main():
    """Test with a single fly."""

    # Paths
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    dataset_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Datasets/251112_11_fly_positions_F1_New_Data/fly_positions/pooled_fly_positions.feather")
    output_dir = Path(__file__).parent  # Save in same directory as script

    print("="*70)
    print("SINGLE FLY TEMPLATE ALIGNMENT TEST")
    print("="*70)

    try:
        # Load template
        print("\n1. Loading template...")
        template = cv2.imread(str(template_path))
        if template is None:
            print(f"   ERROR: Could not load template from {template_path}")
            return
        template_binary = binarize_template(template, invert_background=True)
        print(f"   Template shape: {template.shape}")

        # Load dataset
        print("\n2. Loading dataset...")
        df = pd.read_feather(dataset_path)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Number of unique flies: {df['fly'].nunique()}")

        # Get first fly with video path
        print("\n3. Selecting first fly with video path...")

        fly_col = "fly"

        # Get first fly
        first_fly_id = df[fly_col].iloc[0]
        fly_data = df[df[fly_col] == first_fly_id].copy()

    # Get video path - construct from fly ID
    # Fly ID format: 251008_F1_New_Videos_Checked_arena2_Right
    fly_id_str = str(first_fly_id)

    print(f"   Fly ID: {first_fly_id}")
    print(f"   Fly data points: {len(fly_data)}")

    # Parse fly ID to extract arena and side
    parts = fly_id_str.split('_')

    # Find arena and side in the fly ID
    arena = None
    side = None
    for i, part in enumerate(parts):
        if 'arena' in part.lower():
            arena = part
            if i + 1 < len(parts):
                side = parts[i + 1]
            break

    if arena is None or side is None:
        print(f"   ERROR: Could not parse arena and side from fly ID: {fly_id_str}")
        return

    print(f"   Parsed: arena={arena}, side={side}")

    # Construct video path
    videos_base = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked")
    video_path = videos_base / arena / side / f"{side}.mp4"

    print(f"   Video path: {video_path}")

    if not video_path.exists():
        print(f"   ERROR: Could not find video at {video_path}")
        return

    print(f"   Using video: {video_path}")

    # Detect arena in video
    print("\n4. Detecting arena in video using template matching...")
    arena_params = detect_arena_in_video(video_path, template_binary)

    if arena_params is None:
        print("   ERROR: Arena detection failed!")
        return

    print(f"   Match score: {arena_params['score']:.4f}")
    print(f"   Scale factor: {arena_params['scale']:.4f}")
    print(f"   Arena position: ({arena_params['arena_x']}, {arena_params['arena_y']})")
    print(f"   Arena size: {arena_params['arena_width']}x{arena_params['arena_height']}")

    # Find position columns
    print("\n5. Finding position columns...")
    x_col = None
    y_col = None

    possible_x_cols = ["x_thorax_fly_0", "x_thorax", "x_thorax_skeleton"]
    possible_y_cols = ["y_thorax_fly_0", "y_thorax", "y_thorax_skeleton"]

    for col in possible_x_cols:
        if col in fly_data.columns:
            x_col = col
            break

    for col in possible_y_cols:
        if col in fly_data.columns:
            y_col = col
            break

    if x_col is None or y_col is None:
        print(f"   ERROR: Could not find position columns!")
        return

    print(f"   Using x_col: {x_col}, y_col: {y_col}")

    # Transform fly positions to template coordinates
    print("\n6. Transforming fly positions to template coordinates...")
    fly_data_transformed = transform_fly_positions_to_template(
        fly_data,
        arena_params,
        template.shape,
        x_col=x_col,
        y_col=y_col
    )

    if len(fly_data_transformed) == 0:
        print("   ERROR: No valid positions after transformation!")
        return

    print(f"   Transformed {len(fly_data_transformed)} positions")
    print(f"   X range: {fly_data_transformed['x_template'].min():.1f} to {fly_data_transformed['x_template'].max():.1f}")
    print(f"   Y range: {fly_data_transformed['y_template'].min():.1f} to {fly_data_transformed['y_template'].max():.1f}")

    # Find time column
    time_col = None
    for col in ["time", "Time"]:
        if col in fly_data_transformed.columns:
            time_col = col
            break

    # Plot trajectory
    print("\n7. Plotting trajectory on template...")
    output_path = output_dir / "test_single_fly_trajectory_on_template.png"
    if time_col:
        plot_fly_trajectory_on_template(fly_data_transformed, template, output_path, time_col=time_col)
    else:
        plot_fly_trajectory_on_template(fly_data_transformed, template, output_path, time_col="time")

    # Also create a visualization showing the detection
    print("\n8. Creating detection visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Load video frame for visualization
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Show video frame with detected arena
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[0].imshow(frame_rgb)

        # Draw rectangle around detected arena
        rect = Rectangle(
            (arena_params['arena_x'], arena_params['arena_y']),
            arena_params['arena_width'],
            arena_params['arena_height'],
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].set_title(f"Video Frame with Detected Arena\nScore: {arena_params['score']:.4f}, Scale: {arena_params['scale']:.4f}",
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')

    # Show template
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    axes[1].imshow(template_rgb)
    axes[1].set_title("Template", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    detection_path = output_dir / "test_arena_detection.png"
    plt.savefig(detection_path, dpi=150, bbox_inches='tight')
    print(f"Saved detection visualization to: {detection_path}")
    plt.close()

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. Trajectory plot: {output_path}")
    print(f"  2. Detection visualization: {detection_path}")
    print("\nCheck these files to verify the alignment!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
