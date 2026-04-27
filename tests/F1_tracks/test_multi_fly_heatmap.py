#!/usr/bin/env python3
"""
Test script to align multiple flies' trajectories to the template using template matching.
Shows aggregated heatmap of fly positions overlaid on the template.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib import colors as mcolors


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
    }


def transform_fly_positions_to_template(
    fly_data, arena_params, template_shape, x_col="x_thorax_fly_0", y_col="y_thorax_fly_0"
):
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

    return df_filtered


def create_per_fly_heatmap(fly_data, template_shape, bins=100):
    """
    Create a normalized heatmap for a single fly.

    Args:
        fly_data: DataFrame with x_template and y_template columns
        template_shape: Shape of template (h, w, c)
        bins: Number of bins for histogram

    Returns:
        2D numpy array with normalized heatmap
    """
    template_h, template_w = template_shape[:2]

    x = fly_data["x_template"].values
    y = fly_data["y_template"].values

    if len(x) == 0:
        return None

    # Create 2D histogram
    x_edges = np.linspace(0, template_w, bins + 1)
    y_edges = np.linspace(0, template_h, bins + 1)

    heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Normalize by number of points for this fly
    heatmap = heatmap / len(x)

    return heatmap


def plot_heatmap_on_template(
    all_heatmaps, template, template_binary, output_path, n_flies, n_points, blur_sigma=2.0, alpha=0.7
):
    """
    Plot averaged heatmap on template with masking.

    Args:
        all_heatmaps: List of per-fly heatmaps
        template: Template image (BGR)
        template_binary: Binary template for masking
        output_path: Path to save output
        n_flies: Number of flies
        n_points: Total number of points
        blur_sigma: Gaussian blur sigma
        alpha: Alpha for heatmap overlay
    """
    # Average heatmaps across flies
    avg_heatmap = np.mean(all_heatmaps, axis=0)

    # Apply Gaussian blur
    if blur_sigma > 0:
        avg_heatmap = gaussian_filter(avg_heatmap, sigma=blur_sigma)

    # Create mask from template
    template_h, template_w = template.shape[:2]
    bins = avg_heatmap.shape[0]

    # Resize template binary mask to match heatmap bins
    arena_mask_resized = cv2.resize(
        template_binary.astype(np.uint8), (bins, bins), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Apply mask - set non-arena areas to NaN
    heatmap_masked = avg_heatmap.copy()
    heatmap_masked[~arena_mask_resized.T] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))

    # Show template as background
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    ax.imshow(template_rgb, extent=[0, template_w, template_h, 0], aspect="auto")

    # Overlay heatmap
    cmap = plt.cm.hot
    cmap.set_bad(alpha=0)  # Make NaN transparent

    vmax = np.nanmax(heatmap_masked)
    im = ax.imshow(
        heatmap_masked.T,
        extent=[0, template_w, template_h, 0],
        origin="upper",
        cmap=cmap,
        alpha=alpha,
        interpolation="bilinear",
        vmin=0,
        vmax=vmax,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Avg. Normalized Density (per fly)", fraction=0.046, pad=0.04)

    # Add sample size annotation
    ax.text(
        0.02,
        0.98,
        f"N flies = {n_flies}\nN points = {n_points:,}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0, template_w)
    ax.set_ylim(template_h, 0)
    ax.set_xlabel("X (template pixels)", fontsize=12)
    ax.set_ylabel("Y (template pixels)", fontsize=12)
    ax.set_title("Fly Position Heatmap on Template\n(Template-Matched Alignment)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved heatmap to: {output_path}")
    plt.close()


def main():
    """Process multiple flies and create aggregated heatmap."""

    # Paths
    template_path = Path("/mnt/upramdya_data/MD/F1_Tracks/F1_New_Template.png")
    dataset_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251112_11_fly_positions_F1_New_Data/fly_positions/pooled_fly_positions.feather"
    )
    videos_base = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/251008_F1_New_Videos_Checked")
    output_dir = Path(__file__).parent  # Save in same directory as script

    print("=" * 70)
    print("MULTI-FLY HEATMAP WITH TEMPLATE ALIGNMENT")
    print("=" * 70)

    # Load template
    print("\n1. Loading template...")
    template = cv2.imread(str(template_path))
    template_binary = binarize_template(template, invert_background=True)
    print(f"   Template shape: {template.shape}")

    # Load dataset
    print("\n2. Loading dataset...")
    df = pd.read_feather(dataset_path)
    print(f"   Dataset shape: {df.shape}")

    # Get unique flies
    fly_col = "fly"
    flies = df[fly_col].unique()
    print(f"\n3. Found {len(flies)} unique flies")

    # Process settings
    max_flies = 20  # Process first N flies for testing
    downsample = 10  # Downsample factor for faster processing

    print(f"   Processing first {max_flies} flies with downsample={downsample}")

    # Position columns
    x_col = "x_thorax_fly_0"
    y_col = "y_thorax_fly_0"

    # Storage for transformed data and heatmaps
    all_heatmaps = []
    all_transformed_data = []
    total_points = 0
    successful_flies = 0

    # Process each fly
    print(f"\n4. Processing flies...")
    print("=" * 70)

    for i, fly_id in enumerate(flies[:max_flies]):
        print(f"\n[{i+1}/{min(max_flies, len(flies))}] Fly: {fly_id}")

        # Get fly data
        fly_data = df[df[fly_col] == fly_id].copy()

        # Downsample
        if downsample > 1:
            fly_data = fly_data.iloc[::downsample].reset_index(drop=True)

        print(f"   Points: {len(fly_data)}")

        # Construct video path from experiment info
        try:
            # Get experiment identifier
            experiment = fly_data["experiment"].iloc[0]

            # Parse arena and side from fly_id
            # Format: "251008_F1_New_Videos_Checked_arena2_Right"
            parts = fly_id.split("_")

            # Find arena and side
            arena = None
            side = None
            for j, part in enumerate(parts):
                if part.startswith("arena"):
                    arena = part
                    if j + 1 < len(parts):
                        side = parts[j + 1]
                    break

            if arena and side:
                video_path = videos_base / arena / side / f"{side}.mp4"
            else:
                print(f"   ERROR: Could not parse arena/side from fly_id")
                continue

            if not video_path.exists():
                print(f"   ERROR: Video not found at {video_path}")
                continue

            print(f"   Video: {video_path.relative_to(videos_base)}")

        except Exception as e:
            print(f"   ERROR: Could not construct video path: {e}")
            continue

        # Detect arena
        try:
            arena_params = detect_arena_in_video(video_path, template_binary)

            if arena_params is None:
                print(f"   ERROR: Arena detection failed")
                continue

            print(f"   Arena: score={arena_params['score']:.3f}, scale={arena_params['scale']:.3f}")

        except Exception as e:
            print(f"   ERROR: Arena detection error: {e}")
            continue

        # Transform positions
        try:
            fly_data_transformed = transform_fly_positions_to_template(
                fly_data, arena_params, template.shape, x_col=x_col, y_col=y_col
            )

            if len(fly_data_transformed) == 0:
                print(f"   WARNING: No valid positions after transformation")
                continue

            print(f"   Transformed: {len(fly_data_transformed)} points kept")

        except Exception as e:
            print(f"   ERROR: Transformation error: {e}")
            continue

        # Create per-fly heatmap
        try:
            fly_heatmap = create_per_fly_heatmap(fly_data_transformed, template.shape, bins=100)

            if fly_heatmap is not None:
                all_heatmaps.append(fly_heatmap)
                all_transformed_data.append(fly_data_transformed)
                total_points += len(fly_data_transformed)
                successful_flies += 1
                print(f"   ✓ Success")

        except Exception as e:
            print(f"   ERROR: Heatmap creation error: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"\n5. Successfully processed {successful_flies}/{min(max_flies, len(flies))} flies")
    print(f"   Total points: {total_points:,}")

    if len(all_heatmaps) == 0:
        print("\nERROR: No heatmaps were created!")
        return

    # Create aggregated heatmap plot
    print(f"\n6. Creating aggregated heatmap...")
    output_path = output_dir / "test_multi_fly_heatmap_on_template.png"

    plot_heatmap_on_template(
        all_heatmaps,
        template,
        template_binary,
        output_path,
        n_flies=successful_flies,
        n_points=total_points,
        blur_sigma=2.0,
        alpha=0.7,
    )

    # Also create a version showing individual fly trajectories overlaid
    print(f"\n7. Creating individual trajectories overlay...")
    fig, ax = plt.subplots(figsize=(10, 14))

    # Show template
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    ax.imshow(template_rgb, extent=[0, template.shape[1], template.shape[0], 0], aspect="auto")

    # Plot each fly's trajectory with different color
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_transformed_data)))

    for idx, fly_data_transformed in enumerate(all_transformed_data):
        ax.scatter(
            fly_data_transformed["x_template"],
            fly_data_transformed["y_template"],
            c=[colors[idx]],
            s=0.5,
            alpha=0.3,
            edgecolors="none",
        )

    ax.set_xlim(0, template.shape[1])
    ax.set_ylim(template.shape[0], 0)
    ax.set_xlabel("X (template pixels)", fontsize=12)
    ax.set_ylabel("Y (template pixels)", fontsize=12)
    ax.set_title(
        f"Individual Fly Trajectories on Template\n{successful_flies} flies, {total_points:,} points",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_aspect("equal")

    plt.tight_layout()
    trajectories_path = output_dir / "test_multi_fly_trajectories_on_template.png"
    plt.savefig(trajectories_path, dpi=300, bbox_inches="tight")
    print(f"Saved trajectories to: {trajectories_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. Heatmap: {output_path}")
    print(f"  2. Trajectories: {trajectories_path}")


if __name__ == "__main__":
    main()
